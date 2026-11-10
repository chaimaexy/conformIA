"""Tests du golden set, de la relecture CSV et des métriques de retrieval (sans réseau)."""

import pytest
from pydantic import ValidationError

from app.evaluation.golden import ExpectedRef, GoldenItem, check_against_corpus
from app.evaluation.retrieval import first_rank, found_refs, summarize
from app.evaluation.review import ref_to_text, text_to_refs
from app.ingestion.models import LegalSection

FICHE = "IA : Annoter les données"


def item(expected, category="directe", id_="q1"):
    return GoldenItem(
        id=id_, question="?", category=category, expected=expected, answer="r", origin="manuel"
    )


def row(doc, *labels, score=0.8):
    return {
        "document_id": doc,
        "score": score,
        "metadata": {"sections": [{"label": label} for label in labels]},
    }


SOURCES = {"d1": "ai_act", "d2": "rgpd", "d3": "cnil_ia"}


def test_expected_ref_matching():
    art = ExpectedRef(source_id="ai_act", label="Article 6")
    assert art.matches("ai_act", "Article 6")
    assert not art.matches("rgpd", "Article 6")  # même numéro, autre texte
    assert not art.matches("ai_act", "Article 60")
    fiche = ExpectedRef(source_id="cnil_ia", fiche=FICHE)
    assert fiche.matches("cnil_ia", f"CNIL, « {FICHE} », Le principe de minimisation")
    with pytest.raises(ValidationError):
        ExpectedRef(source_id="ai_act")  # ni label ni fiche


def test_out_of_corpus_questions_have_no_expected_sections():
    with pytest.raises(ValidationError):
        item([], category="directe")
    with pytest.raises(ValidationError):
        item([ExpectedRef(source_id="ai_act", label="Article 6")], category="hors_corpus")
    assert item([], category="hors_corpus").expected == []


def test_check_against_corpus():
    sections = [
        LegalSection(source_id="ai_act", section_type="article", number="6", text="t", url="u")
    ]
    ok = item([ExpectedRef(source_id="ai_act", label="Article 6")])
    bad = item([ExpectedRef(source_id="ai_act", label="Article 999")], id_="q2")
    assert check_against_corpus([ok], sections) == []
    assert "Article 999" in check_against_corpus([ok, bad], sections)[0]


def test_first_rank_and_found_refs():
    q = item(
        [
            ExpectedRef(source_id="rgpd", label="Article 22"),
            ExpectedRef(source_id="ai_act", label="Article 86"),
        ]
    )
    rows = [
        row("d1", "Article 22"),
        row("d2", "Article 21"),
        row("d2", "Article 22"),
        row("d1", "Article 86"),
    ]
    assert first_rank(q, rows, SOURCES) == 3  # « Article 22 » de l'AI Act ne compte pas
    assert found_refs(q, rows, SOURCES, k=3) == 1
    assert found_refs(q, rows, SOURCES, k=4) == 2


def test_summarize():
    per_question = [
        {
            "id": "a",
            "category": "directe",
            "rank": 1,
            "found@8": 1,
            "n_expected": 1,
            "top_score": 0.9,
        },
        {"id": "b", "category": "date", "rank": 5, "found@8": 1, "n_expected": 1, "top_score": 0.8},
        {
            "id": "c",
            "category": "date",
            "rank": None,
            "found@8": 0,
            "n_expected": 2,
            "top_score": 0.7,
        },
        {
            "id": "d",
            "category": "hors_corpus",
            "rank": None,
            "found@8": 0,
            "n_expected": 0,
            "top_score": 0.6,
        },
    ]
    s = summarize(per_question)
    assert s["questions"] == 3
    assert (s["hit@1"], s["hit@4"], s["hit@8"]) == (0.333, 0.333, 0.667)
    assert s["recall@8"] == 0.5
    assert s["mrr@20"] == round((1 + 1 / 5 + 0) / 3, 3)
    assert s["top1_score_hors_corpus"] == 0.6
    assert s["hit@8_par_categorie"] == {"date": 0.5, "directe": 1.0}


def test_csv_refs_roundtrip():
    refs = [
        ExpectedRef(source_id="ai_act", label="Article 113"),
        ExpectedRef(source_id="cnil_ia", fiche=FICHE),
    ]
    text = " | ".join(ref_to_text(r) for r in refs)
    assert text == f"ai_act / Article 113 | cnil_ia / fiche « {FICHE} »"
    assert text_to_refs(text) == refs
    assert text_to_refs("") == []


@pytest.mark.parametrize(
    ("payload", "kind", "expected"),
    [
        ({"directe": {"question": "a"}}, "directe", "a"),
        ({"Directe": {"question": "a"}}, "directe", "a"),
        ({"reformulée": {"question": "b"}}, "reformulee", "b"),
        (
            {"questions": [{"directe": {"question": "a"}, "reformulee": {"question": "b"}}]},
            "reformulee",
            "b",
        ),
        ({"questions": [{"question": "a"}, {"question": "b"}]}, "reformulee", "b"),
        ([{"question": "a"}, {"question": "b"}], "directe", "a"),
        (
            {"questions": [{"directe": {"question": "a"}}, {"reformulee": {"question": "b"}}]},
            "reformulee",
            "b",
        ),
    ],
)
def test_generation_accepts_observed_json_variants(payload, kind, expected):
    from app.evaluation.generate import _find

    assert _find(payload, kind)["question"] == expected
