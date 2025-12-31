import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_menu():
    r = client.get("/")
    assert r.status_code == 200

def test_chat_empty():
    r = client.post("/chat", data={"query": ""})
    assert r.status_code == 400
