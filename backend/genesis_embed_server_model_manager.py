from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from .genesis_embed_server_globals import MODEL_SPECS, current_settings, resolve_model_cache_path, settings_lock

BYTES_PER_GB = 1024 ** 3
_model_jobs_lock = threading.RLock()
_model_jobs: dict[tuple[str, str], dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def _resolve_storage_root(storage_path: str) -> Path:
    resolved = resolve_model_cache_path(storage_path)
    if resolved:
        return Path(resolved)
    env_cache = str(os.getenv("HF_HUB_CACHE") or "").strip()
    if env_cache:
        return Path(env_cache).expanduser().resolve(strict=False)
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve(strict=False)


def _resolve_huggingface_token(explicit_token: str | None = None) -> str | None:
    candidate = str(explicit_token or "").strip()
    if candidate:
        return candidate
    with settings_lock:
        settings_token = str(current_settings.get("huggingface_token", "")).strip()
    if settings_token:
        return settings_token
    env_token = str(os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    return env_token or None


def _format_download_error(model_id: str, exc: Exception, token_present: bool) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    is_auth = any(fragment in lowered for fragment in [
        "cannot access gated repo",
        "gated repo",
        "401 client error",
        "401 unauthorized",
        "please log in",
    ])
    if not is_auth:
        return message
    if token_present:
        return f"{message} The configured Hugging Face token may not have access to '{model_id}'."
    return f"{message} Enter a Hugging Face token with access to '{model_id}' in the admin settings and retry."


def _job_key(model_id: str, storage_root: Path) -> tuple[str, str]:
    return model_id, os.path.normcase(str(storage_root.resolve(strict=False)))


def _repo_root_for_model(model_id: str, storage_root: Path) -> Path:
    return storage_root / _repo_dir_name(model_id)


def _resolve_snapshot_path(repo_root: Path) -> Path | None:
    snapshots_dir = repo_root / "snapshots"
    candidates: list[Path] = []
    refs_main = repo_root / "refs" / "main"
    if refs_main.is_file():
        try:
            revision = refs_main.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        if revision:
            candidates.append(snapshots_dir / revision)
    if snapshots_dir.is_dir():
        try:
            candidates.extend(path for path in snapshots_dir.iterdir() if path.is_dir())
        except OSError:
            pass

    for snapshot_path in candidates:
        if (snapshot_path / "config.json").is_file() and (
            (snapshot_path / "model.safetensors").is_file()
            or (snapshot_path / "pytorch_model.bin").is_file()
            or (snapshot_path / "model.safetensors.index.json").is_file()
            or (snapshot_path / "pytorch_model.bin.index.json").is_file()
        ):
            return snapshot_path
    return None


def _directory_size_gb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    total_bytes = 0
    for current_root, _, files in os.walk(path):
        for file_name in files:
            file_path = Path(current_root) / file_name
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
    return round(total_bytes / BYTES_PER_GB, 2)


def list_model_statuses(storage_path: str) -> list[dict[str, Any]]:
    storage_root = _resolve_storage_root(storage_path)
    with _model_jobs_lock:
        jobs = dict(_model_jobs)

    statuses: list[dict[str, Any]] = []
    for model_id, metadata in MODEL_SPECS.items():
        repo_root = _repo_root_for_model(model_id, storage_root)
        repo_exists = repo_root.exists()
        snapshot_path = _resolve_snapshot_path(repo_root) if repo_exists else None
        job = jobs.get(_job_key(model_id, storage_root))

        if job and job.get("status") == "downloading":
            status = "downloading"
        elif snapshot_path is not None:
            status = "ready"
        elif job and job.get("status") == "error":
            status = "error"
        elif repo_exists:
            status = "partial"
        else:
            status = "missing"

        size_source = snapshot_path or (repo_root if repo_exists else None)
        statuses.append({
            "id": model_id,
            "label": metadata["label"],
            "family": metadata["family"],
            "tier": metadata["tier"],
            "status": status,
            "local_path": str(snapshot_path.resolve(strict=False)) if snapshot_path is not None else None,
            "cache_path": str(repo_root.resolve(strict=False)) if repo_exists else None,
            "storage_root": str(storage_root),
            "approx_size_gb": metadata["approx_size_gb"],
            "size_on_disk_gb": _directory_size_gb(size_source),
            "dimensions": metadata.get("dimensions"),
            "recommended_backend": metadata.get("recommended_backend", "auto"),
            "notes": metadata.get("notes", ""),
            "error": job.get("error") if job and job.get("status") == "error" else None,
            "updated_at": job.get("updated_at") if job else None,
        })

    return statuses


def queue_model_download(model_id: str, storage_path: str, huggingface_token: str | None = None) -> dict[str, Any]:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unsupported model id '{model_id}'.")

    storage_root = _resolve_storage_root(storage_path)
    storage_root.mkdir(parents=True, exist_ok=True)
    current_job_key = _job_key(model_id, storage_root)
    download_token = _resolve_huggingface_token(huggingface_token)

    with _model_jobs_lock:
        existing_job = _model_jobs.get(current_job_key)
        if existing_job and existing_job.get("status") == "downloading":
            return dict(existing_job)
        _model_jobs[current_job_key] = {
            "model_id": model_id,
            "storage_root": str(storage_root),
            "status": "downloading",
            "error": None,
            "updated_at": _now_iso(),
        }

    def worker() -> None:
        try:
            snapshot_download(model_id, cache_dir=str(storage_root), resume_download=True, token=download_token)
        except Exception as exc:
            with _model_jobs_lock:
                _model_jobs[current_job_key] = {
                    "model_id": model_id,
                    "storage_root": str(storage_root),
                    "status": "error",
                    "error": _format_download_error(model_id, exc, token_present=bool(download_token)),
                    "updated_at": _now_iso(),
                }
            return
        with _model_jobs_lock:
            _model_jobs[current_job_key] = {
                "model_id": model_id,
                "storage_root": str(storage_root),
                "status": "ready",
                "error": None,
                "updated_at": _now_iso(),
            }

    threading.Thread(target=worker, daemon=True).start()
    with _model_jobs_lock:
        return dict(_model_jobs[current_job_key])


def delete_model_cache(model_id: str, storage_path: str) -> dict[str, Any]:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"Unsupported model id '{model_id}'.")
    storage_root = _resolve_storage_root(storage_path)
    current_job_key = _job_key(model_id, storage_root)
    repo_root = _repo_root_for_model(model_id, storage_root)

    with _model_jobs_lock:
        existing_job = _model_jobs.get(current_job_key)
        if existing_job and existing_job.get("status") == "downloading":
            raise ValueError(f"Model '{model_id}' is still downloading.")

    removed = False
    removed_path: str | None = None
    if repo_root.exists():
        shutil.rmtree(repo_root)
        removed = True
        removed_path = str(repo_root.resolve(strict=False))

    with _model_jobs_lock:
        _model_jobs.pop(current_job_key, None)

    return {"ok": True, "removed": removed, "removed_path": removed_path, "storage_root": str(storage_root)}
