"""Pages and health checks answer GET and HEAD (the uptime pinger uses HEAD)."""
import pytest


@pytest.mark.parametrize("path", ["/", "/api/health"])
@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_page_answers(client, method, path):
    assert client.request(method, path).status_code == 200


def test_health_body(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_admin_page_served(client):
    r = client.get("/admin")
    assert r.status_code == 200 and "KeifPicks Admin" in r.text


def test_admin_api_needs_key(client):
    assert client.get("/api/admin/status").status_code == 403
    assert client.get("/api/admin/status", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert client.post("/api/props/refresh-odds").status_code == 403


def test_admin_status_with_key(client):
    r = client.get("/api/admin/status", headers={"X-Admin-Key": "ci-test-key"})
    assert r.status_code == 200
    assert "runs" in r.json()
