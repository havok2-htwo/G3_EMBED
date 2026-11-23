"""API keys, admin login hardening, the Hugging Face token and the performance endpoint.

Regression tests for the G3 admin UI standard: fail-closed key storage,
Bearer + X-API-Key + admin session on every public route, and no secret in a GET response.
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import genesis_embed_server_admin as admin
from backend import genesis_embed_server_api as api
from backend import genesis_embed_server_auth as auth
from backend import genesis_embed_server_runtime as runtime
from backend.g3_gpu import RateWindow

STRONG_PASSWORD = "correct horse battery"


# ---------- API keys ----------
@pytest.fixture
def store(tmp_path):
    return auth.get_auth_store()


@pytest.fixture
def public_client(fake_model, monkeypatch):
    """Public routes with the REAL key gate (conftest replaces it with an open stub)."""
    monkeypatch.setattr(api, "authorize_api_request", auth.authorize_api_request)
    app = FastAPI()
    api.create_api(app)
    return TestClient(app)


def signed_in_cookie(store) -> dict[str, str]:
    username = auth.DEFAULT_ADMIN_USERNAME
    store._payload["users"][username]["must_change_password"] = False
    return {auth.SESSION_COOKIE_NAME: store.create_session(username)}


def test_routes_are_open_while_no_key_exists(public_client, store):
    assert public_client.post("/embed", json={"inputs": ["a"]}).status_code == 200
    assert public_client.get("/v1/models").status_code == 200


@pytest.mark.parametrize("route", ["/embed", "/v1/embeddings", "/v1/models"])
def test_one_key_protects_every_public_route(public_client, store, route):
    store.create_api_key("client")
    call = public_client.get if route == "/v1/models" else public_client.post
    body = {"inputs": ["a"]} if route == "/embed" else {"input": ["a"], "model": "google/embeddinggemma-2"}
    response = call(route) if route == "/v1/models" else call(route, json=body)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "Authorization: Bearer" in response.json()["detail"]


def test_bearer_and_x_api_key_both_work_and_count_usage(public_client, store):
    created = store.create_api_key("client")
    token = created["token"]
    assert public_client.post("/embed", json={"inputs": ["a", "b"]}, headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert public_client.post("/v1/embeddings", json={"input": "a"}, headers={"X-API-Key": token}).status_code == 200
    assert public_client.get("/v1/models", headers={"authorization": f"bearer {token}"}).status_code == 200
    usage = store.list_api_keys()[0]["usage"]
    assert usage["request_count"] == 2 and usage["total_items_processed"] == 3


def test_admin_session_passes_but_an_invalid_explicit_key_does_not(public_client, store):
    store.create_api_key("client")
    cookies = signed_in_cookie(store)
    public_client.cookies.update(cookies)
    assert public_client.post("/embed", json={"inputs": ["a"]}).status_code == 200
    assert public_client.post("/embed", json={"inputs": ["a"]}, headers={"X-API-Key": "embed_wrong"}).status_code == 401
    assert store.list_api_keys()[0]["usage"]["request_count"] == 0, "session use is not counted on a key"


def test_session_that_must_change_password_does_not_unlock_the_api(public_client, store):
    store.create_api_key("client")
    public_client.cookies.update({auth.SESSION_COOKIE_NAME: store.create_session(auth.DEFAULT_ADMIN_USERNAME)})
    assert public_client.post("/embed", json={"inputs": ["a"]}).status_code == 401


def test_revoking_the_last_key_opens_the_api_again(public_client, store):
    created = store.create_api_key("client")
    assert public_client.post("/embed", json={"inputs": ["a"]}).status_code == 401
    assert store.delete_api_key(created["id"])
    assert public_client.post("/embed", json={"inputs": ["a"]}).status_code == 200


def test_key_records_store_a_hint_and_never_the_token(store):
    created = store.create_api_key("  a name that is far too long " * 5)
    token = created["token"]
    body = token.split("_", 1)[1]
    assert created["hint"] == f"embed_{body[:4]}…{body[-4:]}"
    listed = store.list_api_keys()[0]
    assert listed["hint"] == created["hint"] and len(listed["alias"]) <= auth.ALIAS_MAX_LENGTH
    assert "key_hash" not in listed
    text = store.secrets_path.read_text(encoding="utf-8")
    assert token not in text and created["hint"] in json.loads(text)["api_keys"][created["id"]]["hint"]


def test_writes_are_atomic_and_usage_is_flushed_later(store, monkeypatch):
    created = store.create_api_key("client")
    before = store.secrets_path.read_text(encoding="utf-8")
    store.record_api_key_usage(created["id"], 5)
    assert store.secrets_path.read_text(encoding="utf-8") == before, "usage is not written per request"
    store.flush()
    saved = json.loads(store.secrets_path.read_text(encoding="utf-8"))
    assert saved["api_keys"][created["id"]]["usage"]["total_items_processed"] == 5
    assert not list(store.logs_dir.glob("*.tmp"))


def test_broken_secrets_file_fails_closed_and_is_not_overwritten(tmp_path, public_client):
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    secrets_path = logs / auth.SECRETS_FILE_NAME
    secrets_path.write_text('{"version": 2, "api_keys": {"k": {"key_h', encoding="utf-8")  # truncated
    store = auth.get_auth_store()
    assert store.load_error
    response = public_client.post("/embed", json={"inputs": ["a"]})
    assert response.status_code == 503, "a broken file must not open the API"
    assert public_client.get("/v1/models").status_code == 503
    assert secrets_path.read_text(encoding="utf-8").endswith('"key_h'), "the broken file is kept for repair"


def test_store_reads_never_touch_the_disk_per_request(store, monkeypatch):
    created = store.create_api_key("client")
    monkeypatch.setattr(auth, "_json_load", lambda path: (_ for _ in ()).throw(AssertionError("disk read")))
    assert store.match_api_key(created["token"]) == created["id"]
    assert store.has_api_keys()


# ---------- admin login hardening ----------
@pytest.fixture
def admin_app(monkeypatch):
    monkeypatch.setattr(admin, "list_model_statuses", lambda path: [])
    monkeypatch.setattr(admin, "get_runtime_status", lambda: {})
    monkeypatch.setattr(admin, "gpu_device_options", lambda current=None: [{"label": "Auto", "value": "auto"}])
    app = FastAPI()
    admin.create_admin_api(app)
    return TestClient(app)


def test_setup_status_and_default_hint_only_while_setup_is_pending(admin_app, store):
    assert admin_app.get("/api/admin/auth/setup").json() == {"setup_required": True, "default_username": "admin"}
    store._payload["users"]["admin"]["must_change_password"] = False
    assert admin_app.get("/api/admin/auth/setup").json() == {"setup_required": False, "default_username": None}


def test_login_sets_a_strict_cookie_and_rate_limits_failures(admin_app, store):
    for _ in range(auth.LOGIN_MAX_FAILURES):
        assert admin_app.post("/api/admin/auth/login", json={"username": "admin", "password": "wrong"}).status_code == 401
    blocked = admin_app.post("/api/admin/auth/login", json={"username": "admin", "password": "admin"})
    assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) >= 1
    auth.login_rate_limiter.reset()
    ok = admin_app.post("/api/admin/auth/login", json={"username": "admin", "password": "admin"})
    assert ok.status_code == 200
    assert "samesite=strict" in ok.headers["set-cookie"].lower() and "httponly" in ok.headers["set-cookie"].lower()


def test_new_password_needs_eight_characters(admin_app, store):
    admin_app.post("/api/admin/auth/login", json={"username": "admin", "password": "admin"})
    short = admin_app.post("/api/admin/auth/change-password", json={"current_password": "admin", "new_password": "1234567"})
    assert short.status_code == 400 and "8" in short.json()["detail"]
    ok = admin_app.post("/api/admin/auth/change-password", json={"current_password": "admin", "new_password": STRONG_PASSWORD})
    assert ok.status_code == 200


def test_cross_origin_admin_writes_are_refused(admin_app, store):
    admin_app.cookies.update(signed_in_cookie(store))
    evil = admin_app.post("/api/admin/api-keys", json={"alias": "x"}, headers={"Origin": "http://192.168.20.119:8091"})
    assert evil.status_code == 403
    same = admin_app.post("/api/admin/api-keys", json={"alias": "x"}, headers={"Origin": "http://testserver"})
    assert same.status_code == 200 and same.json()["token"].startswith("embed_")
    listed = admin_app.get("/api/admin/api-keys").json()["keys"]
    assert listed[0]["hint"] and "token" not in listed[0]
    assert admin_app.delete(f"/api/admin/api-keys/{listed[0]['id']}").status_code == 204


# ---------- Hugging Face token ----------
REQUIRED = {"default_model": "google/embeddinggemma-2"}


def test_huggingface_token_is_masked_kept_and_cleared(admin_client, settings):
    settings(huggingface_token="hf_abcdefghijklmnopWXYZ")
    body = admin_client.get("/api/admin/settings").json()
    assert body["settings"]["huggingface_token"] == "hf_…WXYZ"
    assert body["huggingface_token_set"] is True
    assert "hf_abcdefghijklmnopWXYZ" not in json.dumps(body)

    from backend.genesis_embed_server_globals import current_settings

    for sent in ({}, {"huggingface_token": ""}, {"huggingface_token": "hf_…WXYZ"}, {"huggingface_token": None}):
        response = admin_client.put("/api/admin/settings", json={**REQUIRED, **sent})
        assert response.status_code == 200, response.text
        assert current_settings["huggingface_token"] == "hf_abcdefghijklmnopWXYZ", sent
        assert "hf_abcdefghijklmnopWXYZ" not in response.text

    admin_client.put("/api/admin/settings", json={**REQUIRED, "huggingface_token": "hf_newtoken12345678"})
    assert current_settings["huggingface_token"] == "hf_newtoken12345678"
    cleared = admin_client.put("/api/admin/settings", json={**REQUIRED, "clear_huggingface_token": True}).json()
    assert current_settings["huggingface_token"] == "" and cleared["huggingface_token_set"] is False


def test_omitted_or_null_numbers_keep_their_value(admin_client, settings):
    from backend.genesis_embed_server_globals import current_settings

    settings(batch_max_texts=48)
    assert admin_client.put("/api/admin/settings", json={**REQUIRED, "batch_max_texts": None}).status_code == 200
    assert current_settings["batch_max_texts"] == 48


# ---------- performance endpoint ----------
def test_rate_window_falls_to_zero_when_idle(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    window = RateWindow(30.0)
    window.add(60)
    assert window.rate() == pytest.approx(2.0)  # texts per second over 30 s, not the last batch's speed
    clock[0] += 31
    assert window.rate() == 0.0


def test_every_encode_feeds_the_texts_rate(fake_model, monkeypatch):
    window = RateWindow(30.0)
    monkeypatch.setattr(runtime, "texts_rate", window)
    runtime.encode_texts("google/embeddinggemma-2", ["a", "b", "c"])
    assert window.rate() == pytest.approx(3 / 30)


def test_perf_endpoint_contract(admin_client, monkeypatch):
    gpus = [
        {"nvml_index": 0, "cuda_index": None, "uuid": "GPU-a", "pci_bus_id": "0", "name": "NVIDIA GeForce RTX 2070 SUPER",
         "used_by_service": False, "util_pct": 1, "mem_total_mib": 8192, "mem_used_mib": 344, "process_mib": 0,
         "temp_c": 40, "power_w": 10.0, "power_limit_w": 215.0},
        {"nvml_index": 1, "cuda_index": 0, "uuid": "GPU-b", "pci_bus_id": "1", "name": "NVIDIA GeForce RTX 3090",
         "used_by_service": True, "util_pct": 37, "mem_total_mib": 24576, "mem_used_mib": 17980, "process_mib": 4488,
         "temp_c": 54, "power_w": 112.4, "power_limit_w": 350.0},
    ]
    import backend.g3_gpu as g3_gpu

    monkeypatch.setattr(g3_gpu, "gpu_snapshot", lambda: gpus)
    window = RateWindow(30.0)
    window.add(90)
    monkeypatch.setattr(admin, "texts_rate", window)
    body = admin_client.get("/api/admin/perf").json()
    assert body["schema"] == 1 and body["service"] == "G3 Embed" and body["error"] is None
    assert [g["nvml_index"] for g in body["gpus"] if g["used_by_service"]] == [1]
    metrics = {m["key"]: m for m in body["metrics"]}
    assert metrics["throughput"] == {"key": "throughput", "label": "Texts/s", "unit": "texts/s", "value": 3.0, "window_s": 30}
    assert metrics["queue"]["integer"] is True and metrics["queue"]["value"] == 0


def test_stats_count_requests_and_errors_since_start(admin_client, monkeypatch):
    from backend import genesis_embed_server_globals as globals_module

    monkeypatch.setitem(globals_module.request_counters, "requests", 0)
    monkeypatch.setitem(globals_module.request_counters, "errors", 0)
    api._record_history({"success": True})
    api._record_history({"success": False, "error": "x"})
    summary = admin_client.get("/api/admin/stats").json()["summary"]
    assert summary["requests_since_start"] == 2 and summary["errors_since_start"] == 1
    assert summary["started_at"]
