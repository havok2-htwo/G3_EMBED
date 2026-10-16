"""tools/install_torch_backend.py --only-missing: what it installs for an existing venv (no pip runs here)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("install_torch_backend", PROJECT_ROOT / "tools" / "install_torch_backend.py")
installer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(installer)

TORCH = "2.11.0+cu128"


def test_missing_torchvision_is_installed_next_to_the_pinned_torch():
    command = installer.build_missing_packages_command(
        "venv/python", "cpu", {"torch": TORCH, "torchvision": None, "torchaudio": TORCH}
    )
    assert command == [
        "venv/python", "-m", "pip", "install", f"torch=={TORCH}", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ]


def test_complete_venv_needs_no_torch_install():
    installed = {"torch": TORCH, "torchvision": "0.26.0+cu128", "torchaudio": TORCH}
    assert installer.build_missing_packages_command("venv/python", "cuda", installed) is None


def test_missing_torch_installs_from_the_detected_variant():
    command = installer.build_missing_packages_command("venv/python", "cuda", {})
    assert command[4:] == ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"]


def test_packages_are_checked_in_the_target_interpreter(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        found = {"torch": TORCH, "torchvision": None, "torchaudio": TORCH}
        return subprocess.CompletedProcess(command, 0, stdout=f"noise\n{json.dumps(found)}\n", stderr="")

    monkeypatch.setattr(installer.subprocess, "run", run)
    command = installer.build_missing_packages_command("X:/target/venv/python.exe", "cuda")
    assert calls[0][:3] == ["X:/target/venv/python.exe", "-I", "-c"]
    assert json.loads(calls[0][4]) == {"torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio"}
    assert "torchvision" in command and f"torch=={TORCH}" in command


def test_probe_reports_a_package_without_its_module_as_missing(tmp_path):
    """The probe script itself, run in this interpreter: a dist-info without an importable module is missing."""
    site = tmp_path / "site"
    (site / "ghostpkg-1.0.dist-info").mkdir(parents=True)
    (site / "ghostpkg-1.0.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: ghostpkg\nVersion: 1.0\n")
    (site / "realpkg-2.0.dist-info").mkdir()
    (site / "realpkg-2.0.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: realpkg\nVersion: 2.0\n")
    (site / "realpkg").mkdir()
    (site / "realpkg" / "__init__.py").write_text("")
    script = f"import sys; sys.path.insert(0, {str(site)!r})\n" + installer._PROBE_SCRIPT
    wanted = {"ghostpkg": "ghostpkg", "realpkg": "realpkg", "nopkg": "nopkg"}
    completed = subprocess.run([sys.executable, "-I", "-c", script, json.dumps(wanted)], capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {"ghostpkg": None, "realpkg": "2.0", "nopkg": None}


def test_unreachable_target_interpreter_returns_none(monkeypatch):
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "boom"))
    assert installer.installed_packages("venv/python", ["torch"]) is None


@pytest.mark.parametrize(
    ("triton", "expected"),
    [(None, ["venv/python", "-m", "pip", "install", "triton-windows>=3.7,<3.8"]), ("3.6.0", ["venv/python", "-m", "pip", "install", "triton-windows>=3.7,<3.8"]), ("3.7.1.post27", None)],
)
def test_triton_windows_is_installed_when_missing_or_mismatched(monkeypatch, triton, expected):
    monkeypatch.setattr(installer.platform, "system", lambda: "Windows")
    monkeypatch.delenv(installer.SKIP_TRITON_ENV, raising=False)
    versions = {"torch": TORCH, "triton-windows": triton}
    monkeypatch.setattr(installer, "_installed_version", lambda python, package: versions[package])
    ran = []
    monkeypatch.setattr(installer.subprocess, "run", lambda command, **kwargs: ran.append(command) or subprocess.CompletedProcess(command, 0))
    installer.ensure_windows_triton("venv/python", dry_run=False)
    assert ran == ([expected] if expected else [])


def test_triton_module_is_probed_under_its_import_name():
    assert installer.MODULE_BY_PACKAGE["triton-windows"] == "triton"
