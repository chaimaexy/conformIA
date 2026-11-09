"""Tests du cache d'embeddings, des lots Mistral et du format pgvector (sans réseau)."""

import numpy as np

from app.embeddings.base import CachedEmbedder, EmbeddingCache
from app.embeddings.mistral import MAX_BATCH, MAX_BATCH_TOKENS, batches, estimate_tokens
from app.vectordb.store import vector_literal


class CountingEmbedder:
    name = "fake"
    dimension = 3

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([[len(t), 1.0, 0.0] for t in texts])


def test_cache_only_computes_missing_texts(tmp_path):
    inner = CountingEmbedder()
    embedder = CachedEmbedder(inner, EmbeddingCache(tmp_path / "e.sqlite"))
    first = embedder.embed(["a", "bb", "a"])
    second = embedder.embed(["bb", "ccc"])
    assert inner.calls == [["a", "bb"], ["ccc"]]  # doublons et textes déjà vus non recalculés
    assert np.allclose(np.linalg.norm(first, axis=1), 1)  # vecteurs normalisés
    assert np.allclose(first[1], second[0])


def test_batches_respect_count_and_token_limits():
    texts = ["x" * 30] * 200
    groups = batches(texts)
    assert sum(len(g) for g in groups) == 200
    assert all(len(g) <= MAX_BATCH for g in groups)
    big = ["y" * 9000] * 10
    assert all(sum(estimate_tokens(t) for t in g) <= MAX_BATCH_TOKENS for g in batches(big))


def test_vector_literal_for_pgvector():
    assert vector_literal(np.array([0.5, -0.25, 1 / 3])) == "[0.500000,-0.250000,0.333333]"
