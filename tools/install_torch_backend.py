#!/usr/bin/env python3
import argparse
import os
import platform
import shutil
import subprocess
import sys
from typing import Iterable


TORCH_VARIANT_ENV = "GENESIS_TORCH_VARIANT"
DEFAULT_TORCH_VARIANT = "auto"
TORCH_PACKAGES = ["torch", "torchaudio"]
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
    args = parser.parse_args()

    try:
        requested_variant = normalize_variant(args.variant)
    except ValueError as exc:
        print(f"[Torch Install] {exc}", file=sys.stderr)
        return 2

    resolved_variant, reason = resolve_torch_variant(requested_variant)
    command = build_install_command(args.python_executable, resolved_variant)

    print(
        f"[Torch Install] Requested variant: {requested_variant} | resolved: {resolved_variant} | reason: {reason}",
        flush=True,
    )
    print(f"[Torch Install] Command: {print_command(command)}", flush=True)

    if args.dry_run:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
