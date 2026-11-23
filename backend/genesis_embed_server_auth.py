from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from fastapi import HTTPException, Request, Response

from .genesis_embed_password_hash import build_pbkdf2_hash, verify_pbkdf2_hash

# --- configuration ---
SESSION_COOKIE_NAME = "g3_embed_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"
API_KEY_PREFIX = "embed"
SECRETS_VERSION = 2
SECRETS_FILE_NAME = "genesis_embed_secrets.json"
ALIAS_MAX_LENGTH = 64
# Per-key usage is counted in memory and written at most this often (a full rewrite per request would cost more
# than a small embedding request, and a reader could see a half-written file).
USAGE_FLUSH_SECONDS = 5.0
MIN_PASSWORD_LENGTH = 8
# Login attempts per client IP: after this many failures within the window, 429 with Retry-After (no PBKDF2 run).
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 60.0
LOGIN_TRACKED_CLIENTS_MAX = 10_000
# Verified instead of a real hash when the username is unknown, so the response time does not reveal which
# usernames exist (the digest never matches; the cost is the same PBKDF2 run).
_DUMMY_PASSWORD_HASH = "pbkdf2_sha256$390000$" + "0" * 32 + "$" + "0" * 64
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
API_KEY_REQUIRED_DETAIL = "A valid API key is required (Authorization: Bearer <key> or X-API-Key: <key>)."
LOG_PREFIX = "[G3_EMBED]"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class SecretsFileError(RuntimeError):
    """The secrets file exists but cannot be read or parsed. The store then refuses every check (fail closed)."""


def _json_load(path: Path) -> dict[str, Any]:
    """{} only when the file does not exist yet. An unreadable or broken file raises: treating it as "no keys"
    would open the public API, and rewriting it would reset the admin password."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise SecretsFileError(f"{path.name} cannot be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise SecretsFileError(f"{path.name} does not hold a JSON object")
    return payload


def write_private_file(path: Path, text: str) -> None:
    """Write UTF-8 text readable only by its owner on POSIX (0600 from creation on)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(text.encode("utf-8"))
    if os.name == "posix":
        os.chmod(path, 0o600)  # a temp file left behind by a crash keeps its old mode through O_TRUNC


def _json_dump_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Temp file + os.replace: a crash or a concurrent reader never sees a truncated secrets file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_private_file(tmp_path, json.dumps(payload, indent=2, ensure_ascii=True))
    os.replace(tmp_path, path)


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _new_api_key_id() -> str:
    return f"key_{secrets.token_urlsafe(9)}"


def _generate_api_key() -> str:
    return f"{API_KEY_PREFIX}_{secrets.token_urlsafe(24)}"


def key_hint(raw_key: str) -> str:
    """Masked preview stored at creation: prefix + first 4 and last 4 characters of the random part."""
    prefix, _, body = raw_key.partition("_")
    if not body:
        prefix, body = "", raw_key
    lead = f"{prefix}_" if prefix else ""
    return f"{lead}{body[:4]}…{body[-4:]}"


class AuthStore:
    """Admin users (username/password login), browser sessions (cookie) and client API keys, persisted to one
    JSON file (logs/genesis_embed_secrets.json).

    The payload is held in memory (this process is the only writer): API-key checks never touch the disk,
    mutations are written atomically under one lock, and per-key usage (texts, requests) is flushed in the
    background at most every USAGE_FLUSH_SECONDS. A secrets file that exists but cannot be parsed puts the store
    into a failed state: every check answers 503 and the file is never overwritten."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / "logs"
        self.secrets_path = self.logs_dir / SECRETS_FILE_NAME
        self._lock = threading.RLock()
        self._payload: dict[str, Any] = {}
        self._key_index: dict[str, str] = {}  # sha256(raw key) -> key id, rebuilt on every key mutation
        self._usage_dirty = False
        self._flush_timer: threading.Timer | None = None
        self.load_error: str | None = None
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap()

    # ---- persistence ----
    def _save(self) -> None:
        if self.load_error:
            return
        try:
            _json_dump_atomic(self.secrets_path, self._payload)
        except OSError as exc:
            print(f"{LOG_PREFIX} Could not save {SECRETS_FILE_NAME}: {exc}", file=sys.stderr, flush=True)
            return
        self._usage_dirty = False

    def ensure_usable(self) -> None:
        if self.load_error:
            raise HTTPException(
                status_code=503,
                detail=f"Access control unavailable: {self.load_error}. Fix or restore the file and restart the service.",
            )

    def _mutate(self, mutator: Callable[[dict[str, Any]], Any]) -> Any:
        self.ensure_usable()
        with self._lock:
            result = mutator(self._payload)
            self._rebuild_key_index()
            self._save()
            return result

    def _rebuild_key_index(self) -> None:
        self._key_index = {
            str(rec.get("key_hash")): str(key_id)
            for key_id, rec in self._payload.get("api_keys", {}).items()
            if isinstance(rec, dict) and rec.get("key_hash")
        }

    @staticmethod
    def _normalize_shape(payload: dict[str, Any]) -> None:
        payload.setdefault("version", SECRETS_VERSION)
        for section in ("users", "sessions", "api_keys"):
            if not isinstance(payload.get(section), dict):
                payload[section] = {}

    def _bootstrap(self) -> None:
        with self._lock:
            try:
                payload = _json_load(self.secrets_path)
            except SecretsFileError as exc:
                self.load_error = str(exc)
                self._payload = {"version": SECRETS_VERSION, "users": {}, "sessions": {}, "api_keys": {}}
                self._key_index = {}
                print(f"{LOG_PREFIX} ERROR: {exc}. Admin login and the protected API answer 503 until it is fixed.",
                      file=sys.stderr, flush=True)
                return
            self.load_error = None
            # Legacy file only had {"admin_key": {...}}; that token is obsolete.
            if "version" not in payload:
                payload.pop("admin_key", None)
            self._normalize_shape(payload)
            payload["version"] = SECRETS_VERSION
            if not payload["users"]:
                payload["users"][DEFAULT_ADMIN_USERNAME] = self._make_user(
                    DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD, must_change=True
                )
            self._prune_sessions_in(payload)
            self._payload = payload
            self._rebuild_key_index()
            self._save()

    def reload(self) -> None:
        """Re-read the secrets file (tests, or after editing it by hand while the server runs)."""
        self.flush()
        self._bootstrap()

    @staticmethod
    def _make_user(username: str, password: str, *, must_change: bool) -> dict[str, Any]:
        stamp = now_iso()
        return {
            "username": username,
            "password_hash": build_pbkdf2_hash(password),
            "must_change_password": must_change,
            "created_at": stamp,
            "updated_at": stamp,
            "last_login_at": None,
        }

    # ---- users ----
    def verify_user(self, username: str, password: str) -> Optional[dict[str, Any]]:
        self.ensure_usable()
        username = (username or "").strip().lower()
        if not username or not password:
            return None
        with self._lock:
            user = copy.deepcopy(self._payload["users"].get(username))
        if not user:
            verify_pbkdf2_hash(password, _DUMMY_PASSWORD_HASH)
            return None
        if not verify_pbkdf2_hash(password, user.get("password_hash", "")):
            return None
        return user

    def setup_required(self) -> bool:
        """True while every admin still has the default password (first-login flow)."""
        with self._lock:
            users = list(self._payload["users"].values())
        return bool(users) and all(bool(user.get("must_change_password")) for user in users)

    def touch_login(self, username: str) -> None:
        def apply(payload: dict[str, Any]) -> None:
            user = payload["users"].get(username)
            if user:
                user["last_login_at"] = now_iso()

        self._mutate(apply)

    def set_password(self, username: str, new_password: str) -> bool:
        username = (username or "").strip().lower()
        if len(new_password or "") < MIN_PASSWORD_LENGTH:
            raise HTTPException(status_code=400, detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        if new_password == DEFAULT_ADMIN_PASSWORD:
            raise HTTPException(status_code=400, detail="Choose a password other than the default one.")

        def apply(payload: dict[str, Any]) -> bool:
            user = payload["users"].get(username)
            if not user:
                return False
            user["password_hash"] = build_pbkdf2_hash(new_password)
            user["must_change_password"] = False
            user["updated_at"] = now_iso()
            stale = [th for th, s in payload["sessions"].items() if s.get("username") == username]
            for th in stale:
                payload["sessions"].pop(th, None)
            return True

        return self._mutate(apply)

    # ---- sessions ----
    def create_session(self, username: str) -> str:
        raw = secrets.token_urlsafe(32)
        token_hash = hash_token(raw)
        stamp = now_iso()
        expires = (datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)).isoformat()

        def apply(payload: dict[str, Any]) -> None:
            self._prune_sessions_in(payload)
            payload["sessions"][token_hash] = {
                "username": username,
                "created_at": stamp,
                "expires_at": expires,
                "last_seen_at": stamp,
            }

        self._mutate(apply)
        return raw

    def get_session(self, raw_token: Optional[str]) -> Optional[dict[str, Any]]:
        if not raw_token or self.load_error:
            return None
        token_hash = hash_token(raw_token)
        with self._lock:
            session = copy.deepcopy(self._payload["sessions"].get(token_hash))
            user = copy.deepcopy(self._payload["users"].get((session or {}).get("username")))
        if not session:
            return None
        expires = _parse_iso(session.get("expires_at"))
        if expires is not None and datetime.now(timezone.utc) > expires:
            self.delete_session(raw_token)
            return None
        if not user:
            return None
        return {
            "username": user["username"],
            "must_change_password": bool(user.get("must_change_password")),
            "expires_at": session.get("expires_at"),
        }

    def delete_session(self, raw_token: Optional[str]) -> None:
        if not raw_token:
            return
        token_hash = hash_token(raw_token)

        def apply(payload: dict[str, Any]) -> None:
            payload["sessions"].pop(token_hash, None)

        self._mutate(apply)

    @staticmethod
    def _prune_sessions_in(payload: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc)
        expired = []
        for th, session in payload.get("sessions", {}).items():
            expires = _parse_iso(session.get("expires_at"))
            if expires is not None and now > expires:
                expired.append(th)
        for th in expired:
            payload["sessions"].pop(th, None)

    # ---- client API keys ----
    def list_api_keys(self) -> list[dict[str, Any]]:
        self.ensure_usable()
        with self._lock:
            records = copy.deepcopy(list(self._payload["api_keys"].values()))
        out: list[dict[str, Any]] = []
        for rec in records:
            usage = rec.get("usage") or {}
            out.append(
                {
                    "id": rec.get("id"),
                    "alias": rec.get("alias", ""),
                    "hint": rec.get("hint"),
                    "created_at": rec.get("created_at"),
                    "usage": {
                        "total_items_processed": int(usage.get("total_items_processed") or 0),
                        "request_count": int(usage.get("request_count") or 0),
                        "last_used_at": usage.get("last_used_at"),
                    },
                }
            )
        out.sort(key=lambda r: r.get("created_at") or "")
        return out

    def has_api_keys(self) -> bool:
        with self._lock:
            return bool(self._key_index)

    def create_api_key(self, alias: str) -> dict[str, Any]:
        alias = (alias or "").strip()[:ALIAS_MAX_LENGTH] or "Unnamed key"
        raw = _generate_api_key()
        key_id = _new_api_key_id()
        stamp = now_iso()
        hint = key_hint(raw)

        def apply(payload: dict[str, Any]) -> None:
            payload["api_keys"][key_id] = {
                "id": key_id,
                "key_hash": hash_token(raw),
                "hint": hint,
                "alias": alias,
                "created_at": stamp,
                "usage": {"total_items_processed": 0, "request_count": 0, "last_used_at": None},
            }

        self._mutate(apply)
        return {"id": key_id, "alias": alias, "hint": hint, "created_at": stamp, "token": raw}

    def delete_api_key(self, key_id: str) -> bool:
        def apply(payload: dict[str, Any]) -> bool:
            return payload["api_keys"].pop(key_id, None) is not None

        return self._mutate(apply)

    def match_api_key(self, raw_key: Optional[str]) -> Optional[str]:
        if not raw_key or not raw_key.strip():
            return None
        provided = hash_token(raw_key.strip())
        with self._lock:
            index = dict(self._key_index)
        # Constant-time compare against every stored hash (no early exit on a hit).
        matched: Optional[str] = None
        for key_hash, key_id in index.items():
            if secrets.compare_digest(key_hash, provided):
                matched = key_id
        return matched

    def record_api_key_usage(self, key_id: Optional[str], items: int) -> None:
        if not key_id:
            return
        with self._lock:
            rec = self._payload["api_keys"].get(key_id)
            if not rec:
                return
            usage = rec.setdefault("usage", {"total_items_processed": 0, "request_count": 0, "last_used_at": None})
            usage["total_items_processed"] = int(usage.get("total_items_processed") or 0) + int(items or 0)
            usage["request_count"] = int(usage.get("request_count") or 0) + 1
            usage["last_used_at"] = now_iso()
            self._usage_dirty = True
            if self._flush_timer is None:
                self._flush_timer = threading.Timer(USAGE_FLUSH_SECONDS, self.flush)
                self._flush_timer.daemon = True
                self._flush_timer.start()

    def flush(self) -> None:
        """Write pending usage counters now (timer callback and server shutdown)."""
        with self._lock:
            timer, self._flush_timer = self._flush_timer, None
            if timer is not None:
                timer.cancel()
            if self._usage_dirty:
                self._save()


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_auth_store: Optional[AuthStore] = None
_auth_store_lock = threading.Lock()


def get_auth_store() -> AuthStore:
    """The process-wide store, created on first use: importing this module never writes logs/ (tests, tools)."""
    global _auth_store
    if _auth_store is None:
        with _auth_store_lock:
            if _auth_store is None:
                _auth_store = AuthStore(PROJECT_ROOT)
    return _auth_store


# ---- cookie helpers ----
def _cookie_secure(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def set_session_cookie(response: Response, request: Request, raw_token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=raw_token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        # Strict: the dashboard is same-origin, and Lax would still send the cookie with form POSTs from other
        # services on the same host (ports are not part of a cookie's site).
        samesite="strict",
        secure=_cookie_secure(request),
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


# ---- login rate limit ----
class LoginRateLimiter:
    """Login attempts per client key (IP) in a sliding window. An attempt counts as a failure until it succeeds,
    so parallel attempts cannot all slip past the check before the first failure is recorded."""

    def __init__(self, max_failures: int = LOGIN_MAX_FAILURES, window_seconds: float = LOGIN_WINDOW_SECONDS):
        self.max_failures = max(1, int(max_failures))
        self.window_seconds = float(window_seconds)
        self._attempts: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def _recent_locked(self, key: str, now: float) -> deque[float]:
        attempts = self._attempts.get(key)
        if attempts is None:
            attempts = deque()
            self._attempts[key] = attempts
            while len(self._attempts) > LOGIN_TRACKED_CLIENTS_MAX:
                self._attempts.popitem(last=False)
        while attempts and attempts[0] <= now - self.window_seconds:
            attempts.popleft()
        return attempts

    def begin(self, key: str) -> int:
        """0 when the attempt may run (it is counted), else the seconds to wait (refused, not counted)."""
        now = time.monotonic()
        with self._lock:
            attempts = self._recent_locked(key, now)
            if len(attempts) >= self.max_failures:
                return max(1, math.ceil(attempts[0] + self.window_seconds - now))
            attempts.append(now)
            return 0

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._attempts.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._attempts.clear()


login_rate_limiter = LoginRateLimiter()


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# ---- CSRF: state-changing admin requests must come from this origin ----
def _host_key(netloc: str, scheme: str = "") -> str:
    """host[:port] in lower case without the scheme's default port."""
    netloc = netloc.strip().lower()
    for suffix, schemes in ((":80", ("http", "")), (":443", ("https", ""))):
        if netloc.endswith(suffix) and scheme in schemes:
            return netloc[: -len(suffix)]
    return netloc


def require_same_origin(request: Request) -> None:
    """403 for a state-changing request sent by another site's page (CSRF): cookies are not port-scoped, so a page
    of any other service on this host is "same-site". Browsers send Origin (or Referer / Sec-Fetch-Site) with every
    cross-origin POST; clients that send none of them (curl, scripts) are not browsers and pass."""
    if request.method.upper() in SAFE_METHODS:
        return
    fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()
    if fetch_site in {"cross-site", "same-site"}:
        raise HTTPException(status_code=403, detail="Cross-origin request refused.")
    source = (request.headers.get("origin") or request.headers.get("referer") or "").strip()
    if not source:
        return
    parts = urlsplit(source)
    if source.lower() == "null" or not parts.netloc:
        raise HTTPException(status_code=403, detail="Cross-origin request refused.")
    allowed = {_host_key(request.headers.get("host") or "")}
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        allowed.add(_host_key(forwarded_host.split(",")[0]))
    if _host_key(parts.netloc, parts.scheme.lower()) not in allowed and parts.netloc.lower() not in allowed:
        raise HTTPException(status_code=403, detail="Cross-origin request refused.")


# ---- FastAPI dependencies ----
def require_session(request: Request) -> dict[str, Any]:
    require_same_origin(request)
    store = get_auth_store()
    store.ensure_usable()
    ctx = store.get_session(request.cookies.get(SESSION_COOKIE_NAME))
    if not ctx:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return ctx


def require_admin(request: Request) -> dict[str, Any]:
    ctx = require_session(request)
    if ctx.get("must_change_password"):
        raise HTTPException(status_code=403, detail="password_change_required")
    return ctx


def extract_api_key(request: Request) -> Optional[str]:
    """`Authorization: Bearer <key>` (OpenAI SDK and compatible clients) or `X-API-Key: <key>`."""
    authorization = request.headers.get("authorization") or ""
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() == "bearer" and credentials.strip():
        return credentials.strip()
    api_key = request.headers.get("x-api-key")
    return api_key.strip() if api_key and api_key.strip() else None


def authorize_api_request(request: Request) -> Optional[str]:
    """Public-endpoint gate. No client keys configured -> open (returns None). Otherwise a valid Bearer or
    X-API-Key key, or a signed-in admin session (the dashboard's own tools), is required. A key that is sent but
    invalid is refused even with a session. Returns the matched key id (usage accounting) or None; raises 401,
    or 503 when the secrets file cannot be read (fail closed)."""
    store = get_auth_store()
    store.ensure_usable()
    if not store.has_api_keys():
        return None
    provided = extract_api_key(request)
    if provided:
        key_id = store.match_api_key(provided)
        if key_id:
            return key_id
    else:
        session = store.get_session(request.cookies.get(SESSION_COOKIE_NAME))
        if session and not session.get("must_change_password"):
            return None
    raise HTTPException(status_code=401, detail=API_KEY_REQUIRED_DETAIL, headers={"WWW-Authenticate": "Bearer"})


# Backwards-compatible name (older imports and tests).
authorize_api_key = authorize_api_request
