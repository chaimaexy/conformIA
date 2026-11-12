"""Tests des trois stratégies de chunking (sans réseau ni modèle)."""

import numpy as np
import pytest

from app.chunking import recursive, semantic, structural
from app.chunking.common import header, pack, sentences, split_to_fit
from app.ingestion.models import LegalSection


def section(kind, number, text, **kw):
    return LegalSection(
        source_id="ai_act", section_type=kind, number=number, text=text, url=f"u#{number}", **kw
    )


ART5 = section(
    "article",
    "5",
    "\n".join(
        [
            "1. Les pratiques suivantes sont interdites:",
            *[f"{c}) pratique interdite numéro {c}, " + "détail " * 40 for c in "abcdefgh"],
            "2. Le deuxième paragraphe précise les exceptions.",
        ]
    ),
    title="Pratiques interdites",
    chapter="CHAPITRE II - PRATIQUES INTERDITES",
)
RECITAL = section("considerant", "12", "Premier alinéa du considérant.\nSecond alinéa.")


# --------------------------------------------------------------------------- outils communs


def test_sentences_keep_abbreviations_and_paragraph_numbers():
    text = "1. Voir l'art. 5 du RGPD. Ensuite, la CNIL (cf. p. 12) précise. Fin."
    assert sentences(text) == [
        "1. Voir l'art. 5 du RGPD.",
        "Ensuite, la CNIL (cf. p. 12) précise.",
        "Fin.",
    ]


def test_split_to_fit_and_pack_respect_max():
    long_line = " ".join(["mot"] * 400)  # ~1600 caractères, sans ponctuation
    pieces = split_to_fit(long_line, 500)
    assert all(len(p) <= 500 for p in pieces)
    assert " ".join(pieces) == long_line  # rien de perdu
    packed = pack(["a" * 300, "b" * 300, "c" * 300], 650)
    assert packed == ["a" * 300 + "\n" + "b" * 300, "c" * 300]


def test_long_table_is_split_between_rows_with_its_header():
    from app.chunking.common import split_tables

    table = ["| A | B |", "|---|---|", *(f"| ligne {i} | {'x' * 30} |" for i in range(6))]
    pieces = split_tables(["Intro :\n" + "\n".join(table)], max_chars=120)
    assert pieces[0] == "Intro :"
    assert len(pieces) > 2
    for piece in pieces[1:]:
        assert piece.startswith("| A | B |\n|---|---|\n| ligne") and len(piece) <= 120
    rows = [line for p in pieces[1:] for line in p.splitlines()[2:]]
    assert rows == table[2:]  # aucune ligne perdue ni dupliquée


def test_recital_header_warns_it_is_not_updated():
    from app.chunking.common import act_label, header
    from app.ingestion.models import LegalSection

    recital = LegalSection(
        source_id="ai_act",
        section_type="considerant",
        number="179",
        text="...",
        url="u",
        not_updated_by=["32026R1744"],
    )
    assert header(recital).endswith(
        "Considérant 179 (texte d'origine, non mis à jour par le règlement (UE) 2026/1744)"
    )
    assert act_label("32016L0680") == "directive (UE) 2016/680"


def test_header_paths():
    assert header(ART5) == (
        "AI Act (règlement (UE) 2024/1689) > CHAPITRE II - PRATIQUES INTERDITES"
        " > Article 5 - Pratiques interdites"
    )
    assert header(RECITAL) == "AI Act (règlement (UE) 2024/1689) > Considérant 12"


# --------------------------------------------------------------------------- structurel


def test_structural_small_section_is_one_chunk():
    chunks = structural.chunk_document([RECITAL], max_chars=2000)
    assert len(chunks) == 1
    assert chunks[0].body == RECITAL.text
    assert chunks[0].content.startswith("AI Act (règlement (UE) 2024/1689) > Considérant 12\n\n")


def test_structural_long_article_keeps_paragraph_context():
    chunks = structural.chunk_document([ART5, RECITAL], max_chars=900)
    art = [c for c in chunks if c.sections[0].number == "5"]
    assert len(art) > 2
    assert all(len(c.body) <= 900 for c in chunks)
    assert "\n".join(c.body for c in art) == ART5.text  # découpage sans perte
    assert art[1].paragraphs[0] == "1"  # commence par « c) ... » : toujours le paragraphe 1
    assert "(partie 2/" in art[1].header and "paragraphe 1)" in art[1].header
    assert art[-1].paragraphs[-1] == "2"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(len(c.sections) == 1 for c in chunks)  # jamais deux sections mélangées


# --------------------------------------------------------------------------- récursif


def test_recursive_maps_chunks_back_to_sections():
    chunks = recursive.chunk_document([RECITAL, ART5], chunk_size=300, chunk_overlap=50)
    assert all(len(c.body) <= 300 for c in chunks)
    assert chunks[0].sections[0].label == "Considérant 12"
    assert chunks[-1].sections[-1].label == "Article 5"
    assert all(c.header is None for c in chunks)  # référence naïve : pas d'en-tête


# --------------------------------------------------------------------------- sémantique


def fake_embed(texts):
    """Deux « sujets » : phrases sur les biais vs phrases sur la sécurité."""
    vectors = [[1.0, 0.1] if "biais" in t else [0.1, 1.0] for t in texts]
    v = np.array(vectors)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_semantic_cuts_on_topic_change_within_a_section():
    text = (
        "Les biais doivent être détectés avant l'entraînement. "
        "Les biais sont corrigés par des données équilibrées. "
        "La sécurité des données impose le chiffrement. "
        "La sécurité suppose aussi une journalisation."
    )
    sec = section("considerant", "1", text)
    chunks, threshold = semantic.chunk_document(
        [sec, RECITAL], fake_embed, breakpoint_percentile=50, max_chars=2000, min_chars=0
    )
    first = [c for c in chunks if c.sections[0].number == "1"]
    assert [c.body.count("biais") for c in first] == [2, 0]
    assert all(c.sections[0].number in {"1", "12"} for c in chunks)
    assert 0 <= round(threshold, 6) < 1  # distance cosinus médiane (ici ~0)


@pytest.mark.parametrize("max_chars", [120, 400])
def test_semantic_respects_max_chars(max_chars):
    chunks, _ = semantic.chunk_document([ART5], fake_embed, max_chars=max_chars, min_chars=0)
    assert all(len(c.body) <= max_chars for c in chunks)
