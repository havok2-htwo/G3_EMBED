from __future__ import annotations

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
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
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


def detect_torch_devices() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_devices": [],
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
        "torch": torch_info,
        "detected_accelerator_hardware": accelerator_hardware,
    }
    _SYSTEM_SNAPSHOT_CACHE = (now, payload)
    return payload


def resolve_runtime_choice(settings: dict[str, Any]) -> RuntimeChoice:
    target = str(settings.get("execution_target") or "auto").strip()
    backend_override = str(settings.get("backend_override") or "auto").strip()
    snapshot = get_system_snapshot()
    torch_info = snapshot["torch"]
    openvino_devices = set(snapshot["openvino_devices"])

    def choice(backend: str, device: str, label: str, source: str, detail: str) -> RuntimeChoice:
        return RuntimeChoice(backend=backend, device=device, device_label=label, source=source, detail=detail)

    if backend_override == "pytorch":
        if target in {"auto", "nvidia"} and torch_info.get("cuda_available"):
            return choice("pytorch", "cuda", (torch_info.get("cuda_devices") or ["CUDA"])[0], "settings", "PyTorch CUDA")
        if target in {"auto", "intel_igpu"} and torch_info.get("xpu_available"):
            return choice("pytorch", "xpu", (torch_info.get("xpu_devices") or ["Intel XPU"])[0], "settings", "PyTorch XPU")
        return choice("pytorch", "cpu", "CPU", "settings", "PyTorch CPU")

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
