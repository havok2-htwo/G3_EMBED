from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_SYSTEM_SNAPSHOT_CACHE: tuple[float, dict[str, Any]] | None = None
_SYSTEM_SNAPSHOT_TTL_SECONDS = 10.0


@dataclass(frozen=True)
class RuntimeChoice:
    backend: str
    device: str
    device_label: str
    source: str
    detail: str


def run_probe(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part).strip()
    return completed.returncode, output


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def detect_nvidia_gpus() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    code, output = run_probe(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if code != 0 or not output:
        return []
    return _unique([line.strip() for line in output.splitlines() if line.strip()])


def detect_windows_video_devices() -> list[str]:
    if platform.system().lower() != "windows":
        return []
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join \"`n\"",
    ]
    code, output = run_probe(command)
    if code != 0 or not output:
        return []
    return _unique(
        [
            line.strip()
            for line in output.splitlines()
            if line.strip() and "microsoft basic" not in line.lower()
        ]
    )


def detect_windows_npu_devices() -> list[str]:
    if platform.system().lower() != "windows":
        return []
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match '\\bNPU\\b|AI Boost|Neural|Intel\\\\(R\\\\) AI' } | Select-Object -ExpandProperty Name) -join \"`n\"",
    ]
    code, output = run_probe(command)
    if code != 0 or not output:
        return []
    return _unique(
        [
            line.strip()
            for line in output.splitlines()
            if line.strip() and "input configuration device" not in line.lower()
        ]
    )


def detect_openvino_devices() -> list[str]:
    try:
        from openvino import Core

        return list(Core().available_devices)
    except Exception:
        return []


@functools.lru_cache(maxsize=1)
def openvino_backend_available() -> bool:
    """sentence-transformers needs optimum-intel for backend="openvino"; it does not support transformers 5 yet.

    Import it for real: a venv upgraded from the transformers-4 stack can still contain an
    optimum-intel that is present on disk but fails to import against transformers 5.
    """
    try:
        from optimum.intel.openvino import OVModelForFeatureExtraction  # noqa: F401
    except Exception:
        return False
    return True


def detect_torch_devices() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_devices": [],
        "cuda_memory_mib": [],
        "xpu_available": False,
        "xpu_devices": [],
    }
    try:
        import torch

        info["torch_available"] = True
        info["torch_version"] = getattr(torch, "__version__", None)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_devices"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            info["cuda_memory_mib"] = [
                round(torch.cuda.get_device_properties(index).total_memory / (1024 * 1024))
                for index in range(torch.cuda.device_count())
            ]
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            info["xpu_available"] = True
            count = getattr(torch.xpu, "device_count", lambda: 1)()
            names: list[str] = []
            for index in range(count):
                try:
                    names.append(torch.xpu.get_device_name(index))
                except Exception:
                    names.append(f"Intel XPU {index}")
            info["xpu_devices"] = names
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def get_system_snapshot() -> dict[str, Any]:
    global _SYSTEM_SNAPSHOT_CACHE
    now = time.monotonic()
    if _SYSTEM_SNAPSHOT_CACHE is not None:
        cached_at, cached_payload = _SYSTEM_SNAPSHOT_CACHE
        if now - cached_at <= _SYSTEM_SNAPSHOT_TTL_SECONDS:
            return cached_payload

    torch_info = detect_torch_devices()
    openvino_devices = detect_openvino_devices()
    windows_video = detect_windows_video_devices()
    nvidia_probe = detect_nvidia_gpus()
    npu_probe = detect_windows_npu_devices()
    accelerator_hardware = _unique(
        [
            *nvidia_probe,
            *windows_video,
            *npu_probe,
            *[f"OpenVINO {device}" for device in openvino_devices],
            *[f"CUDA {name}" for name in torch_info.get("cuda_devices", [])],
            *[f"XPU {name}" for name in torch_info.get("xpu_devices", [])],
        ]
    )
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "nvidia_gpus": nvidia_probe,
        "windows_video_devices": windows_video,
        "windows_npu_devices": npu_probe,
        "openvino_devices": openvino_devices,
        "openvino_backend_available": openvino_backend_available(),
        "torch": torch_info,
        "detected_accelerator_hardware": accelerator_hardware,
    }
    _SYSTEM_SNAPSHOT_CACHE = (now, payload)
    return payload


def resolve_gpu_index(setting: Any, device_count: int) -> tuple[int, str | None]:
    """gpu_device ("auto" | "cuda:N") -> CUDA index, plus a warning when the configured card is missing (then GPU 0)."""
    text = str(setting or "auto").strip().lower()
    if text.startswith("cuda:") and text[5:].isdigit():
        index = int(text[5:])
        if index < device_count:
            return index, None
        return 0, f"gpu_device '{text}' is not present ({device_count} CUDA device(s)); using cuda:0"
    return 0, None


# torch cuda index -> nvidia-smi index. Card identity never changes while the process runs, so hits are cached.
_NVML_INDEX_CACHE: dict[int, int] = {}


def nvml_index_for_cuda(cuda_index: int) -> int | None:
    """nvidia-smi number of the card behind torch's cuda:<n>, matched by UUID (g3_gpu). With
    CUDA_VISIBLE_DEVICES the two numberings differ (on a 2070 SUPER + 3090 host the 3090 is cuda:0 but GPU 1)."""
    if cuda_index in _NVML_INDEX_CACHE:
        return _NVML_INDEX_CACHE[cuda_index]
    try:
        from .g3_gpu import gpu_for_cuda_index

        gpu = gpu_for_cuda_index(cuda_index)
    except Exception:
        gpu = None
    index = gpu.get("nvml_index") if gpu else None
    if isinstance(index, int):
        _NVML_INDEX_CACHE[cuda_index] = index
        return index
    return None


def cuda_gpu_label(cuda_index: int, name: str) -> str:
    """"GPU <nvidia-smi index> · <name without NVIDIA>", the label every G3 admin UI uses for a card."""
    short = name.removeprefix("NVIDIA ").strip() or name
    nvml_index = nvml_index_for_cuda(cuda_index)
    return f"GPU {nvml_index} · {short}" if nvml_index is not None else f"cuda:{cuda_index} · {short}"


def gpu_device_options(current: Any = None) -> list[dict[str, str]]:
    """Choices for the GPU dropdown: Auto plus every CUDA device this process sees (values stay cuda:N, the
    labels use nvidia-smi numbering); a saved card that is missing stays listed."""
    torch_info = get_system_snapshot()["torch"]
    names = list(torch_info.get("cuda_devices") or [])
    memory = list(torch_info.get("cuda_memory_mib") or [])
    cards: list[dict[str, str]] = []
    for index, name in enumerate(names):
        size = f" ({memory[index] / 1024:.0f} GB)" if index < len(memory) and memory[index] else ""
        cards.append({"label": f"{cuda_gpu_label(index, name)}{size}", "value": f"cuda:{index}"})
    auto_label = f"Auto ({cards[0]['label']})" if cards else "Auto"
    options = [{"label": auto_label, "value": "auto"}, *cards]
    current_value = str(current or "").strip().lower()
    if current_value.startswith("cuda:") and current_value not in {option["value"] for option in options}:
        options.append({"label": f"{current_value} (not detected)", "value": current_value})
    return options


def resolve_runtime_choice(settings: dict[str, Any]) -> RuntimeChoice:
    target = str(settings.get("execution_target") or "auto").strip()
    backend_override = str(settings.get("backend_override") or "auto").strip()
    snapshot = get_system_snapshot()
    torch_info = snapshot["torch"]
    openvino_available = bool(snapshot.get("openvino_backend_available"))
    # Without optimum-intel, OpenVINO devices cannot run models; route Intel targets to PyTorch instead.
    openvino_devices = set(snapshot["openvino_devices"]) if openvino_available else set()

    cuda_names = list(torch_info.get("cuda_devices") or [])
    cuda_index, cuda_warning = resolve_gpu_index(settings.get("gpu_device"), len(cuda_names))
    cuda_label = cuda_gpu_label(cuda_index, cuda_names[cuda_index]) if cuda_index < len(cuda_names) else "CUDA"

    def choice(backend: str, device: str, label: str, source: str, detail: str) -> RuntimeChoice:
        # Every CUDA choice runs on the card gpu_device selects (cuda:N = torch numbering in this process).
        if backend == "pytorch" and device == "cuda":
            device, label = f"cuda:{cuda_index}", cuda_label
            if cuda_warning:
                detail = f"{detail}; {cuda_warning}"
        return RuntimeChoice(backend=backend, device=device, device_label=label, source=source, detail=detail)

    if backend_override == "pytorch":
        if target in {"auto", "nvidia"} and torch_info.get("cuda_available"):
            return choice("pytorch", "cuda", (torch_info.get("cuda_devices") or ["CUDA"])[0], "settings", "PyTorch CUDA")
        if target in {"auto", "intel_igpu"} and torch_info.get("xpu_available"):
            return choice("pytorch", "xpu", (torch_info.get("xpu_devices") or ["Intel XPU"])[0], "settings", "PyTorch XPU")
        return choice("pytorch", "cpu", "CPU", "settings", "PyTorch CPU")

    if backend_override == "openvino" and not openvino_available:
        backend_override = "auto"
    if backend_override == "openvino":
        if target in {"auto", "intel_npu"} and "NPU" in openvino_devices:
            return choice("openvino", "NPU", "Intel NPU", "settings", "OpenVINO NPU")
        if target in {"auto", "intel_igpu"} and "GPU" in openvino_devices:
            return choice("openvino", "GPU", "Intel GPU", "settings", "OpenVINO GPU")
        return choice("openvino", "CPU", "CPU", "settings", "OpenVINO CPU")

    if target == "nvidia" and torch_info.get("cuda_available"):
        return choice("pytorch", "cuda", (torch_info.get("cuda_devices") or ["CUDA"])[0], "auto", "requested NVIDIA")
    if target == "intel_npu" and "NPU" in openvino_devices:
        return choice("openvino", "NPU", "Intel NPU", "auto", "requested Intel NPU")
    if target == "intel_igpu":
        if "GPU" in openvino_devices:
            return choice("openvino", "GPU", "Intel GPU", "auto", "requested Intel iGPU")
        if torch_info.get("xpu_available"):
            return choice("pytorch", "xpu", (torch_info.get("xpu_devices") or ["Intel XPU"])[0], "auto", "requested Intel iGPU fallback")
    if target == "cpu":
        return choice("pytorch", "cpu", "CPU", "settings", "forced CPU")

    if torch_info.get("cuda_available"):
        return choice("pytorch", "cuda", (torch_info.get("cuda_devices") or ["CUDA"])[0], "auto", "auto CUDA")
    if "NPU" in openvino_devices:
        return choice("openvino", "NPU", "Intel NPU", "auto", "auto OpenVINO NPU")
    if "GPU" in openvino_devices:
        return choice("openvino", "GPU", "Intel GPU", "auto", "auto OpenVINO GPU")
    if torch_info.get("xpu_available"):
        return choice("pytorch", "xpu", (torch_info.get("xpu_devices") or ["Intel XPU"])[0], "auto", "auto PyTorch XPU")
    return choice("pytorch", "cpu", "CPU", "auto", "CPU fallback")
