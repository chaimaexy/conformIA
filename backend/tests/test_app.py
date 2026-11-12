import pytest
from fastapi.testclient import TestClient

from app.core.config import BACKEND_DIR, load_app_config
from app.llm import client as llm_client
from app.main import app


def test_health_ok():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "mistral-embed"


def test_config_yaml_is_valid():
    cfg = load_app_config(BACKEND_DIR / "config.yaml")
    assert cfg.embeddings.dimension == 1024
    assert cfg.retrieval.final_k <= cfg.retrieval.rerank_candidates <= cfg.retrieval.candidates
    assert cfg.reranker.model in cfg.reranker.candidates
    assert 0 < cfg.reranker.answerability_threshold < cfg.reranker.verification_threshold < 1
    assert {s.id for s in cfg.corpus.sources} >= {"ai_act", "rgpd"}


def test_missing_config_file_falls_back_to_defaults(tmp_path):
    cfg = load_app_config(tmp_path / "absent.yaml")
    assert cfg.llm.provider == "mistral"


def test_chat_model_requires_api_key(monkeypatch):
    monkeypatch.setattr(
        llm_client, "get_settings", lambda: type("S", (), {"mistral_api_key": None})()
    )
    with pytest.raises(llm_client.MissingAPIKeyError):
        llm_client._api_key()
