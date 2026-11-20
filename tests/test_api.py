import pytest
from fastapi.testclient import TestClient
from app.api import app
from app.config import EXACT_ABSTENTION_MESSAGE

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["total_pages"] == 50

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["pdf_loaded"] is True
    assert data["chroma_collection_count"] == 50

def test_api_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_pages"] == 50
    assert "page_10" in data["special_pages"]

def test_api_get_page():
    response = client.get("/page/10")
    assert response.status_code == 200
    data = response.json()
    assert data["page_number"] == 10
    assert "RECIPE CARD 2.4" in data["text"]

def test_api_query_in_scope():
    response = client.post("/query", json={
        "query": "What are the ingredients and quantities for the Signature Truffle Ganache?",
        "top_k": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_abstained"] is False
    assert 10 in data["citations"]
    assert "850" in data["answer"]
    assert len(data["retrieved_chunks"]) > 0

def test_api_query_out_of_scope():
    response = client.post("/query", json={
        "query": "What is the capital of France?",
        "top_k": 3
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_abstained"] is True
    assert data["answer"] == EXACT_ABSTENTION_MESSAGE
    assert data["citations"] == []
