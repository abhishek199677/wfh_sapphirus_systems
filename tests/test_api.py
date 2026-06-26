import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "AI Copilot API is running"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)
    assert "checks" in resp.json()
    assert "status" in resp.json()


def test_ask_no_query(client):
    resp = client.post("/api/ask", json={})
    assert resp.status_code == 422


def test_documents_invalid(client):
    resp = client.post("/api/documents", json={})
    assert resp.status_code == 422
