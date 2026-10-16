#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Iterable


TORCH_VARIANT_ENV = "GENESIS_TORCH_VARIANT"
SKIP_TRITON_ENV = "GENESIS_SKIP_TRITON_WINDOWS"
DEFAULT_TORCH_VARIANT = "auto"
TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]
PYTORCH_INDEX_BY_VARIANT = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu128",
    "xpu": "https://download.pytorch.org/whl/xpu",
}


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return 1, ""
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
    return completed.returncode, output


def detect_windows_video_devices() -> list[str]:
    ps_command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"`n\"",
    ]
    code, output = run_command(ps_command)
    if code != 0 or not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def detect_linux_pci_devices() -> list[str]:
    if shutil.which("lspci") is None:
        return []
    code, output = run_command(["lspci"])
    if code != 0 or not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def has_nvidia_gpu() -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found"
    code, output = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if code != 0:
        return False, "nvidia-smi probe failed"
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    return bool(first_line), first_line or "NVIDIA GPU detected"


def has_supported_intel_xpu() -> tuple[bool, str]:
    system_name = platform.system().lower()
    device_lines: list[str] = []
    if system_name == "windows":
        device_lines = detect_windows_video_devices()
    elif system_name == "linux":
        device_lines = detect_linux_pci_devices()

    for line in device_lines:
        lowered = line.lower()
        if "intel" not in lowered:
            continue
        if (
            "arc" in lowered
            or "gpu max" in lowered
            or "data center gpu" in lowered
            or "iris" in lowered
            or "xe" in lowered
            or "uhd" in lowered
            or "graphics" in lowered
        ):
            return True, line

    return False, "No supported Intel XPU-class GPU detected"


def normalize_variant(value: str | None) -> str:
    normalized = str(value or DEFAULT_TORCH_VARIANT).strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "xpu"}:
        raise ValueError(
            f"Unsupported torch variant '{normalized}'. Use one of: auto, cpu, cuda, xpu."
        )
    return normalized


def resolve_torch_variant(requested_variant: str) -> tuple[str, str]:
    if requested_variant == "cpu":
        return "cpu", "forced CPU install"
    if requested_variant == "cuda":
        return "cuda", "forced CUDA install"
    if requested_variant == "xpu":
        return "xpu", "forced Intel XPU install"

    has_cuda, cuda_detail = has_nvidia_gpu()
    if has_cuda:
        return "cuda", f"auto-detected NVIDIA GPU: {cuda_detail}"

    has_xpu, xpu_detail = has_supported_intel_xpu()
    if has_xpu:
        return "xpu", f"auto-detected Intel XPU-capable GPU: {xpu_detail}"

    return "cpu", "auto fallback to CPU"


def build_install_command(python_executable: str, variant: str) -> list[str]:
    index_url = PYTORCH_INDEX_BY_VARIANT[variant]
    return [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        *TORCH_PACKAGES,
        "--index-url",
        index_url,
    ]


# Import name of each distribution: a dist-info whose module is gone (an interrupted uninstall) counts as missing.
MODULE_BY_PACKAGE = {"torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio", "triton-windows": "triton"}

_PROBE_SCRIPT = """
import importlib.metadata as metadata, importlib.util as util, json, sys
found = {}
for package, module in json.loads(sys.argv[1]).items():
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = None
    found[package] = version if version and util.find_spec(module) is not None else None
print(json.dumps(found))
"""


def installed_packages(python_executable: str, packages: Iterable[str]) -> dict[str, str | None] | None:
    """Installed version per package in the TARGET interpreter (None = missing); None if it cannot be asked.

    Checked in python_executable, not in the interpreter running this script: start.bat passes the venv python
    explicitly, and a check against the wrong environment reported a torchvision-less venv as complete.
    """
    wanted = {package: MODULE_BY_PACKAGE.get(package, package) for package in packages}
    try:
        completed = subprocess.run(
            [python_executable, "-I", "-c", _PROBE_SCRIPT, json.dumps(wanted)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    lines = completed.stdout.strip().splitlines() if completed.returncode == 0 else []
    try:
        found = json.loads(lines[-1]) if lines else None
    except ValueError:
        return None
    return {package: found.get(package) for package in wanted} if isinstance(found, dict) else None


def build_missing_packages_command(
    python_executable: str,
    variant: str,
    installed: dict[str, str | None] | None = None,
) -> list[str] | None:
    """Install only missing torch packages, pinned to the installed torch so it is not replaced."""
    if installed is None:
        installed = installed_packages(python_executable, TORCH_PACKAGES)
    if installed is None:
        print(
            "[Torch Install] Warning: could not query the target Python; letting pip install every torch package.",
            flush=True,
        )
        installed = {}
    missing = [package for package in TORCH_PACKAGES if not installed.get(package)]
    if not missing:
        return None
    torch_version = installed.get("torch")
    index_url = PYTORCH_INDEX_BY_VARIANT[variant]
    pinned: list[str] = []
    if torch_version and "torch" not in missing:
        pinned = [f"torch=={torch_version}"]
        # Use the index of the installed build (e.g. +cu128), which may differ from the variant detected now.
        local_tag = torch_version.split("+", 1)[1] if "+" in torch_version else ""
        if local_tag.startswith("cu"):
            index_url = f"https://download.pytorch.org/whl/{local_tag}"
        elif local_tag in PYTORCH_INDEX_BY_VARIANT:
            index_url = PYTORCH_INDEX_BY_VARIANT[local_tag]
    return [python_executable, "-m", "pip", "install", *pinned, *missing, "--index-url", index_url]


def _installed_version(python_executable: str, package: str) -> str | None:
    return (installed_packages(python_executable, [package]) or {}).get(package)


def triton_windows_spec(torch_version: str) -> str | None:
    """Each torch minor pairs with one triton minor (2.10 -> 3.6, 2.11 -> 3.7, ...)."""
    try:
        major, minor = (int(part) for part in torch_version.split("+", 1)[0].split(".")[:2])
    except ValueError:
        return None
    if major != 2 or minor < 6:
        return None
    triton_minor = minor - 4
    return f"triton-windows>=3.{triton_minor},<3.{triton_minor + 1}"


def ensure_windows_triton(python_executable: str, dry_run: bool) -> None:
    """Install triton-windows for torch.compile on Windows CUDA builds (Linux CUDA wheels ship Triton)."""
    if platform.system() != "Windows" or os.getenv(SKIP_TRITON_ENV, "0").lower() in {"1", "true", "yes"}:
        return
    torch_version = _installed_version(python_executable, "torch")
    if not torch_version or "+cu" not in torch_version:
        return
    spec = triton_windows_spec(torch_version)
    if spec is None:
        return
    triton_version = _installed_version(python_executable, "triton-windows")
    wanted_prefix = spec.split(">=", 1)[1].split(",", 1)[0] + "."
    if triton_version and triton_version.startswith(wanted_prefix):
        print(f"[Torch Install] triton-windows {triton_version} matches torch {torch_version}.", flush=True)
        return
    command = [python_executable, "-m", "pip", "install", spec]
    print(f"[Torch Install] torch.compile support: {print_command(command)}", flush=True)
    if dry_run:
        return
    if subprocess.run(command, check=False).returncode != 0:
        print("[Torch Install] Warning: triton-windows install failed; torch.compile stays disabled.", flush=True)


def print_command(parts: Iterable[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install a suitable PyTorch backend into the current G3_EMBED venv."
    )
    parser.add_argument(
        "--variant",
        default=os.getenv(TORCH_VARIANT_ENV, DEFAULT_TORCH_VARIANT),
        help="auto, cpu, cuda, or xpu (default: env GENESIS_TORCH_VARIANT or auto)",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable inside the target venv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the chosen backend and pip command without installing anything",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only install what is missing of torch, torchvision, torchaudio and (Windows CUDA) triton-windows, "
        "keeping the installed torch version",
    )
    args = parser.parse_args()

    try:
        requested_variant = normalize_variant(args.variant)
    except ValueError as exc:
        print(f"[Torch Install] {exc}", file=sys.stderr)
        return 2

    resolved_variant, reason = resolve_torch_variant(requested_variant)
    if args.only_missing:
        missing_command = build_missing_packages_command(args.python_executable, resolved_variant)
        if missing_command is None:
            print("[Torch Install] All torch packages present.", flush=True)
            ensure_windows_triton(args.python_executable, args.dry_run)
            return 0
        command = missing_command
    else:
        command = build_install_command(args.python_executable, resolved_variant)

    print(
        f"[Torch Install] Requested variant: {requested_variant} | resolved: {resolved_variant} | reason: {reason}",
        flush=True,
    )
    print(f"[Torch Install] Command: {print_command(command)}", flush=True)

    if args.dry_run:
        print("[Torch Install] triton-windows (Windows CUDA) is matched to the installed torch after installing.", flush=True)
        return 0

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            f"[Torch Install] Failed to install variant '{resolved_variant}'.",
            file=sys.stderr,
            flush=True,
        )
        return completed.returncode

    print(f"[Torch Install] Installed variant '{resolved_variant}' successfully.", flush=True)
    ensure_windows_triton(args.python_executable, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
