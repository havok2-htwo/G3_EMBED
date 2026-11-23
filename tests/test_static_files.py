"""Static file serving: nothing outside frontend/dist is ever served (path traversal regression), the admin
index and assets still load, unknown API paths answer JSON 404, and the landing page is English."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import genesis_embed_server as server


@pytest.fixture
def site(monkeypatch, tmp_path):
    """The real app with frontend/dist in tmp_path/frontend/dist and a private file two levels above it."""
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>admin</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    (tmp_path / "README.md").write_text("PRIVATE-README", encoding="utf-8")
    (tmp_path / ".env").write_text("HF_TOKEN=hf_private", encoding="utf-8")
    monkeypatch.setattr(server, "FRONTEND_DIST_DIR", dist)
    return TestClient(server.app)  # no lifespan: no model preload, no batch worker


@pytest.mark.parametrize(
    "path",
    [
        "/..%2f..%2fREADME.md",
        "/../../README.md",
        "/assets/..%2f..%2f..%2fREADME.md",
        "/assets/../../../README.md",
        "/admin/..%2f..%2f..%2fREADME.md",
        "/..%2f..%2f.env",
        "/%2e%2e/%2e%2e/README.md",
        "/assets/..%5c..%5c..%5cREADME.md",
    ],
)
def test_static_route_never_serves_files_outside_dist(site, path):
    response = site.get(path)
    assert "PRIVATE" not in response.text and "hf_private" not in response.text
    assert response.status_code in (200, 404)  # 200 = landing page / admin index, never the file


def test_path_as_is_traversal_is_refused(site):
    # httpx normalises "/../" in URLs; send the raw target like `curl --path-as-is` does.
    import httpx

    transport = httpx.ASGITransport(app=server.app)

    async def fetch(raw: str):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = client.build_request("GET", "http://test/")
            request.url = request.url.copy_with(raw_path=raw.encode())
            return await client.send(request)

    import asyncio

    for raw in ("/../../README.md", "/assets/../../../README.md", "/../../.env"):
        response = asyncio.run(fetch(raw))
        assert "PRIVATE" not in response.text and "hf_private" not in response.text, raw


def test_dist_assets_and_admin_index_still_load(site):
    assert site.get("/assets/app.js").text == "console.log('ok')"
    assert site.get("/admin").status_code == 200
    assert "<title>admin</title>" in site.get("/admin/settings").text


def test_unknown_api_paths_answer_json_404(site):
    for path in ("/api/nope", "/v1/nope"):
        response = site.get(path)
        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}


def test_landing_page_is_english(site):
    text = site.get("/").text
    assert '<html lang="en">' in text and "G3 Embed" in text
    assert "oeffentliche" not in text and "fuer" not in text


