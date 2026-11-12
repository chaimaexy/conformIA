"""Tests du retrieval avancé : BM25, RRF, multi-query, garde-fou (sans réseau)."""

import pytest

from app.evaluation.retrieval import gate_analysis
from app.retrieval.bm25 import BM25, tokenize
from app.retrieval.fusion import rrf
from app.retrieval.multiquery import _clean


def test_tokenize_french():
    tokens = tokenize("L'annexe III s'applique aux systèmes d'IA à haut risque")
    assert "annex" in tokens and "iii" in tokens and "system" in tokens
    assert not {"l", "s", "d", "aux", "à"} & set(tokens)  # élisions et mots-outils retirés


def test_bm25_prefers_rare_terms():
    docs = [
        "Les systèmes d'IA à haut risque respectent les exigences.",
        "Les systèmes d'IA sont soumis au présent règlement.",
        "Le présent règlement s'applique à partir du 2 août 2026.",
    ]
    index = BM25(docs)
    assert index.search("à partir de quand le règlement s'applique-t-il ?")[0][0] == 2
    assert index.search("haut risque")[0][0] == 0
    assert index.search("cybersécurité") == []  # terme absent : aucun résultat


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b", "c"], ["b", "c", "a"], ["b", "d"]], k=60)
    assert [doc for doc, _ in fused][:2] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("**Texte**", "Texte"), (["a", "b"], "a"), (None, ""), ([], "")],
)
def test_multiquery_clean(value, expected):
    assert _clean(value) == expected


def test_gate_analysis():
    per_question = [
        {"category": "directe", "top_score": 0.9},
        {"category": "directe", "top_score": 0.6},
        {"category": "reformulee", "top_score": 0.3},
        {"category": "hors_corpus", "top_score": 0.2},
        {"category": "hors_corpus", "top_score": 0.4},
    ]
    gate = gate_analysis(per_question)
    # le seuil le plus haut qui ne refuse aucune question du corpus est 0.3 : 1 hors corpus sur 2
    assert gate["sans_refus_a_tort"] == {"seuil": 0.3, "hors_corpus_bloquees": 0.5}
    assert gate["separation_parfaite"] is False


def _section(kind, number, text="", chapter=None, section=None, title=None):
    from app.ingestion.models import LegalSection

    return LegalSection(
        source_id="ai_act",
        section_type=kind,
        number=number,
        text=text,
        url="u",
        chapter=chapter,
        section=section,
        title=title,
    )


def test_reference_resolver():
    from app.chunking.references import ReferenceResolver

    resolver = ReferenceResolver(
        [
            _section("article", "1", chapter="CHAPITRE I - DISPOSITIONS GÉNÉRALES"),
            _section("article", "5", chapter="CHAPITRE II - PRATIQUES INTERDITES EN MATIÈRE D’IA"),
            _section(
                "article",
                "16",
                chapter="CHAPITRE III - SYSTÈMES D’IA À HAUT RISQUE",
                section="SECTION 3 - Obligations des fournisseurs",
            ),
            _section("annexe", "II", title="Infractions pénales"),
            _section("annexe", "III", title="Systèmes d'IA à haut risque"),
        ]
    )
    refs = resolver.resolve("les chapitres I et II s'appliquent ; le chapitre III, section 3")
    assert refs == [
        "chapitre I (Dispositions générales)",
        "chapitre II (Pratiques interdites en matière d’IA)",
        "chapitre III (Systèmes d’IA à haut risque)",
        "chapitre III, section 3 (Obligations des fournisseurs)",
    ]
    assert resolver.resolve("annexes II à III") == [
        "annexe II (Infractions pénales)",
        "annexe III (Systèmes d'IA à haut risque)",
    ]
    # renvoi vers un autre texte : non résolu
    assert resolver.resolve("l'annexe II du règlement (UE) 2019/1020") == []


def test_article_references():
    from app.chunking.references import ReferenceResolver

    sections = [
        _section("article", str(n), title=f"Titre {n}") for n in (5, 12, 15, 22, 50, 99)
    ] + [_section("article", "4 bis", title="Biais")]
    off = ReferenceResolver(sections)
    on = ReferenceResolver(sections, articles=True)
    text = "Le non-respect de l'article 5 et des articles 12 à 22, ou de l'article 4 bis"
    assert off.resolve(text) == []  # désactivé par défaut
    assert on.resolve(text) == [
        "article 5 (Titre 5)",
        "article 12 (Titre 12)",
        "article 15 (Titre 15)",  # dans la plage 12 à 22
        "article 22 (Titre 22)",
        "article 4 bis (Biais)",
    ]
    assert on.resolve("conformément à l'article 50", own_article="50") == []  # pas soi-même
    assert on.resolve("l'article 5 du règlement (UE) 2016/679") == []  # autre texte
    assert on.resolve("l'article 99 du présent règlement") == ["article 99 (Titre 99)"]


def test_context_expansion_follows_citations(monkeypatch):
    from app.retrieval import pipeline

    def chunk(id_, label, header, cites=()):
        meta = {"label": label, "header": header, "cites": list(cites), "sections": []}
        return {"id": id_, "document_id": "rgpd", "content": id_, "metadata": meta}

    chunks = {
        "a83": chunk("a83", "Article 83", "Article 83 (partie 3/4)", ["Article 15", "Article 22"]),
        "a15": chunk("a15", "Article 15", "Article 15"),
        "a22b": chunk("a22b", "Article 22", "Article 22 (partie 2/2)"),
        "a22a": chunk("a22a", "Article 22", "Article 22 (partie 1/2)"),
        "a6": chunk("a6", "Article 6", "Article 6"),
    }
    monkeypatch.setitem(pipeline._CHUNKS, "test", chunks)
    retriever = pipeline.Retriever(pipeline.RetrievalSettings(strategy="test"), db=object())
    rows = [{**chunks["a83"], "score": 0.9}, {**chunks["a6"], "score": 0.5}]
    context = retriever.with_citations(rows, final_k=2)
    assert [c["id"] for c in context] == ["a83", "a6", "a15", "a22a"]  # 1re partie de l'art. 22
    assert context[2]["cited_by"] == "Article 83" and context[2]["score"] is None
    # place limitée : la section citée que le reranker a notée (art. 22, rang 3) passe devant
    rows.append({**chunks["a22b"], "score": 0.7})
    context = retriever.with_citations(rows, final_k=2, max_extra=1)
    assert [c["id"] for c in context] == ["a83", "a6", "a22a"] and context[2]["score"] == 0.7


def test_cohere_reranker_maps_scores_by_index(monkeypatch, tmp_path):
    import httpx

    from app.retrieval import rerank

    monkeypatch.setattr(rerank, "CACHE_PATH", tmp_path / "rr.sqlite")
    monkeypatch.setattr(rerank.RerankCache.__init__, "__defaults__", (tmp_path / "rr.sqlite",))
    monkeypatch.setattr(rerank.CohereReranker, "MIN_INTERVAL", 0)
    monkeypatch.setattr(rerank, "get_settings", lambda: type("S", (), {"cohere_api_key": "k"}))
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        results = [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]
        return httpx.Response(200, json={"results": results}, request=httpx.Request("POST", url))

    monkeypatch.setattr(rerank.httpx, "post", fake_post)
    reranker = rerank.get_reranker("cohere:rerank-test")
    assert reranker.scores("q", ["a", "b"]) == [0.2, 0.9]
    assert reranker.scores("q", ["a", "b"]) == [0.2, 0.9]  # 2e fois : cache, pas d'appel
    assert len(calls) == 1 and calls[0]["model"] == "rerank-test"


@pytest.mark.parametrize(
    ("spec", "url", "results_field", "extra"),
    [
        ("voyage:rerank-2.5", "https://api.voyageai.com/v1/rerank", "data", {}),
        (
            "jina:jina-reranker-v2-base-multilingual",
            "https://api.jina.ai/v1/rerank",
            "results",
            {"top_n": 2},
        ),
    ],
)
def test_other_api_rerankers(monkeypatch, tmp_path, spec, url, results_field, extra):
    import httpx

    from app.retrieval import rerank

    monkeypatch.setattr(rerank.RerankCache.__init__, "__defaults__", (tmp_path / "rr.sqlite",))
    monkeypatch.setattr(rerank.ApiReranker, "MIN_INTERVAL", 0)
    reranker = rerank.get_reranker(spec)
    monkeypatch.setattr(reranker, "MIN_INTERVAL", 0)
    keys = {"voyage_api_key": "k", "jina_api_key": "k"}
    monkeypatch.setattr(rerank, "get_settings", lambda: type("S", (), keys))
    calls = []

    def fake_post(called_url, headers, json, timeout):
        calls.append((called_url, json))
        results = [{"index": 0, "relevance_score": 0.3}, {"index": 1, "relevance_score": 0.8}]
        return httpx.Response(
            200, json={results_field: results}, request=httpx.Request("POST", called_url)
        )

    monkeypatch.setattr(rerank.httpx, "post", fake_post)
    assert reranker.scores("q", ["a", "b"]) == [0.3, 0.8]
    called_url, body = calls[0]
    assert called_url == url and body["model"] == spec.split(":", 1)[1]
    assert body.items() >= {"query": "q", "documents": ["a", "b"], **extra}.items()


def test_missing_api_key_is_explicit(monkeypatch, tmp_path):
    from app.retrieval import rerank

    monkeypatch.setattr(rerank.RerankCache.__init__, "__defaults__", (tmp_path / "rr.sqlite",))
    monkeypatch.setattr(rerank, "get_settings", lambda: type("S", (), {"voyage_api_key": None}))
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        rerank.get_reranker("voyage:rerank-2.5").scores("q", ["a"])


def test_production_settings_match_phase4_choice():
    from app.evaluation.retrieval import PRODUCTION
    from app.retrieval.pipeline import RetrievalSettings

    # config.yaml décrit bien la configuration mesurée sur le golden set
    assert RetrievalSettings.from_config() == PRODUCTION


def test_qwen3_prompt_format():
    from app.retrieval.rerank import Qwen3Reranker, get_reranker

    reranker = get_reranker("tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
    assert isinstance(reranker, Qwen3Reranker)
    ((query, doc),) = reranker._pairs("Quand ?", ["Texte"])
    assert query.endswith("<Query>: Quand ?\n") and "<Instruct>:" in query
    assert doc.startswith("<Document>: Texte<|im_end|>")
