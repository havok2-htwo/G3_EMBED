from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Union

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .genesis_embed_server_admin import create_admin_api
from .genesis_embed_server_auth import get_auth_store
from .genesis_embed_server_api import RequestBodyLimitMiddleware, create_api, request_validation_exception_handler
from .genesis_embed_server_batching import EmbedBatchManager
from .genesis_embed_server_globals import current_settings
from .genesis_embed_server_model_manager import list_model_statuses
from .genesis_embed_server_runtime import encode_texts
from .genesis_embed_server_presets import FIRST_START_PRESET, load_settings_or_first_start_preset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8777


async def _encode_async(model_id: str, texts: list[str], settings: dict | None = None):
    return await asyncio.to_thread(encode_texts, model_id, texts, settings)


def _preload_default_model(model_id: str, cache_path: str) -> None:
    """Load (and compile/warm up) the default model in the background so the first request does not pay for it.

    Only runs when the model is already in the local cache, so startup never triggers a download. Runs on a
    daemon thread so a long compile warmup never blocks server shutdown.
    """
    try:
        statuses = list_model_statuses(cache_path)
        if not any(status["id"] == model_id and status["status"] == "ready" for status in statuses):
            print(f"[G3_EMBED] Preload skipped: '{model_id}' is not in the local model cache.", file=sys.stderr)
            return
        encode_texts(model_id, ["warmup"])
        print(f"[G3_EMBED] Preloaded default model '{model_id}'.", file=sys.stderr)
    except Exception as exc:
        print(f"[G3_EMBED] Preload of '{model_id}' failed: {exc}", file=sys.stderr)


async def startup_server(app: FastAPI) -> None:
    print("--- G3_EMBED Server starting (FastAPI + React + local embeddings) ---", file=sys.stderr)
    loaded, first_start = load_settings_or_first_start_preset()
    current_settings.clear()
    current_settings.update(loaded)
    if first_start:
        print(f"[G3_EMBED] First start: no saved settings, applied the '{FIRST_START_PRESET}' preset.", file=sys.stderr)
    print(
        f"Loaded settings: model='{current_settings['default_model']}', "
        f"target='{current_settings['execution_target']}', "
        f"backend_override='{current_settings['backend_override']}', cache='{current_settings['model_cache_path']}'",
        file=sys.stderr,
    )
    batch_manager = EmbedBatchManager(_encode_async)
    app.state.embed_batch_manager = batch_manager
    await batch_manager.start()
    threading.Thread(
        target=_preload_default_model,
        args=(str(current_settings.get("default_model") or ""), str(current_settings.get("model_cache_path", ""))),
        name="g3-embed-preload",
        daemon=True,
    ).start()


async def shutdown_server(app: FastAPI) -> None:
    batch_manager = getattr(app.state, "embed_batch_manager", None)
    if batch_manager is not None:
        await batch_manager.stop()
    get_auth_store().flush()  # pending per-key usage counters


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_server(app)
    try:
        yield
    finally:
        await shutdown_server(app)


app = FastAPI(title="G3_EMBED API", version="1.0.0", lifespan=lifespan)
# 413 for oversized /embed and /v1/embeddings bodies before FastAPI buffers and parses them.
app.add_middleware(RequestBodyLimitMiddleware)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app = create_api(app)
app = create_admin_api(app)


def _frontend_index_response() -> Union[HTMLResponse, FileResponse]:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
  <head><meta charset="utf-8" /><title>G3 Embed</title></head>
  <body style="font-family: system-ui, sans-serif; padding: 32px;">
    <h1>Admin UI build missing</h1>
    <p>Run <code>npm ci</code> and <code>npm run build</code> inside <code>frontend</code>, then reload.</p>
  </body>
</html>
""",
        status_code=503,
    )


_LANDING_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="dark light" />
    <title>G3 Embed</title>
    <style>
      :root {
        color-scheme: dark;
        --bg: #10161c; --text: #ebf0df; --text-strong: #faf7ef; --text-dim: #a9b2b8;
        --panel: rgba(15, 23, 29, .82); --panel-border: rgba(255, 248, 224, .12);
        --accent: #ffc16c; --accent-strong: #ff8b47; --line-strong: rgba(255, 255, 255, .16);
        --surface-hover: rgba(255, 255, 255, .08); --text-on-accent: #1b1510;
        --bg-image: radial-gradient(circle at top left, rgba(255, 194, 117, .20), transparent 28%),
          radial-gradient(circle at top right, rgba(139, 230, 192, .16), transparent 24%),
          linear-gradient(145deg, #182028, #10161c 45%, #17131c);
      }
      @media (prefers-color-scheme: light) {
        :root {
          color-scheme: light;
          --bg: #f5f1e8; --text: #1f272d; --text-strong: #11171b; --text-dim: #56626b;
          --panel: rgba(255, 255, 255, .86); --panel-border: rgba(31, 39, 45, .12);
          --accent: #ffb24d; --accent-strong: #e8661f; --line-strong: rgba(31, 39, 45, .20);
          --surface-hover: rgba(31, 39, 45, .07);
          --bg-image: radial-gradient(circle at top left, rgba(255, 170, 90, .22), transparent 30%),
            radial-gradient(circle at top right, rgba(90, 200, 160, .18), transparent 26%),
            linear-gradient(145deg, #faf6ee, #f2eee5 45%, #f4eff3);
        }
      }
      * { box-sizing: border-box; }
      html { background: var(--bg); }
      body {
        margin: 0; min-height: 100vh; color: var(--text); background: var(--bg-image); background-attachment: fixed;
        font-family: Bahnschrift, "Segoe UI Variable", "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
        font-size: 1rem; line-height: 1.5;
      }
      main { width: min(880px, calc(100vw - 32px)); margin: 0 auto; padding: 56px 0; }
      section {
        display: grid; gap: 16px; padding: 24px; border-radius: 24px;
        background: var(--panel); border: 1px solid var(--panel-border);
        box-shadow: 0 28px 60px rgba(0, 0, 0, .28); backdrop-filter: blur(18px);
      }
      .eyebrow { margin: 0; color: var(--accent); font-size: .75rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
      h1 {
        margin: 0; color: var(--text-strong); line-height: 1.2;
        font-family: Rockwell, "Roboto Slab", Bahnschrift, "Segoe UI Variable", system-ui, sans-serif;
        font-size: clamp(1.75rem, 1.25rem + 1.5vw, 2.25rem);
      }
      p { margin: 0; color: var(--text-dim); }
      code { font-family: "Cascadia Code", Consolas, "DejaVu Sans Mono", monospace; font-size: .9em; color: var(--text-strong); }
      .button-row { display: flex; gap: 12px; flex-wrap: wrap; }
      a {
        display: inline-flex; align-items: center; justify-content: center; min-height: 40px; padding: 0 20px;
        border-radius: 999px; text-decoration: none; font-size: .875rem; font-weight: 700; letter-spacing: .02em;
      }
      .primary { background: linear-gradient(135deg, var(--accent), var(--accent-strong)); color: var(--text-on-accent); }
      .secondary { color: var(--text-strong); border: 1px solid var(--line-strong); background: var(--surface-hover); }
      a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    </style>
  </head>
  <body>
    <main>
      <section>
        <p class="eyebrow">G3 GPU service</p>
        <h1>G3 Embed</h1>
        <p>
          Local text and media embeddings. <code>POST /embed</code>, <code>POST /v1/embeddings</code> and
          <code>GET /v1/models</code> are open while no API key exists. Once an API key is created in the admin
          dashboard, these routes require <code>Authorization: Bearer &lt;key&gt;</code> or
          <code>X-API-Key: &lt;key&gt;</code>. The admin dashboard is protected by a username and password.
        </p>
        <div class="button-row">
          <a class="primary" href="/admin">Open admin dashboard</a>
          <a class="secondary" href="/docs">API docs</a>
        </div>
      </section>
    </main>
  </body>
</html>
"""


def _landing_response() -> HTMLResponse:
    return HTMLResponse(_LANDING_HTML, status_code=200)


def _resolve_frontend_asset(full_path: str) -> Path | None:
    """File under frontend/dist for full_path, or None (never a file outside dist).

    The path is checked lexically before the filesystem is touched: on Windows, dist / "//host/share/x" is a UNC
    path, and resolving it would open an SMB connection to that host. Backslashes, drive letters / colons,
    absolute paths and '..' never reach the filesystem; symlinks that point outside dist are never served.
    """
    if not full_path or "\x00" in full_path or "\\" in full_path or ":" in full_path:
        return None
    relative = PurePosixPath(full_path)
    if relative.is_absolute() or full_path.startswith("/") or ".." in relative.parts:
        return None
    parts = [part for part in relative.parts if part not in {"", "."}]
    if not parts:
        return None
    try:
        dist_root = FRONTEND_DIST_DIR.resolve(strict=True)
        candidate = dist_root.joinpath(*parts).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not candidate.is_relative_to(dist_root) or not candidate.is_file():
        return None
    return candidate


@app.get("/")
async def serve_landing():
    return _landing_response()


@app.get("/admin")
@app.get("/admin/")
async def serve_frontend_index():
    return _frontend_index_response()


@app.get("/{full_path:path}")
async def serve_frontend_assets(full_path: str):
    if full_path.startswith(("api/", "v1/")) or full_path in {"api", "v1"}:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if full_path.startswith("admin/"):
        asset_path = _resolve_frontend_asset(full_path.removeprefix("admin/"))
        return FileResponse(asset_path) if asset_path is not None else _frontend_index_response()
    asset_path = _resolve_frontend_asset(full_path)
    if asset_path is not None:
        return FileResponse(asset_path)
    return _landing_response()


if __name__ == "__main__":
    print("\n--- G3_EMBED is ready ---", file=sys.stderr)
    print("Open: http://127.0.0.1:8777/", file=sys.stderr)
    print("Admin: http://127.0.0.1:8777/admin", file=sys.stderr)
    print("API: POST http://127.0.0.1:8777/embed", file=sys.stderr)
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
