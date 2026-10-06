from fastapi.testclient import TestClient

from api.main import app


def test_api_lists_questions_and_models():
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"] is True
        questions = client.get("/questions").json()
        assert len(questions) == 40
        assert {q["subset"] for q in questions} == {"P1", "P2", "P3"}
        assert len(client.get("/questions", params={"subset": "P2", "lang": "en"}).json()) == 10
        keys = {m["key"] for m in client.get("/models").json()}
        assert {"qwen3-8b", "gemini-flash-lite"} <= keys


def test_api_rejects_bad_requests():
    with TestClient(app) as client:
        assert client.post("/solve", json={"model_key": "yok", "text": "1+1"}).status_code == 404
        assert client.post("/solve", json={"model_key": "qwen3-8b", "strategy": "xyz",
                                           "text": "1+1"}).status_code == 422
        assert client.get("/runs/999/summary").status_code == 404
