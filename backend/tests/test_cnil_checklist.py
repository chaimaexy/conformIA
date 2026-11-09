"""Tests de la liste de vérification CNIL (logique indépendante du PDF)."""

from datetime import date

from app.ingestion import cnil_checklist as cl

VOCAB = cl.vocabulary(["les sous-traitants", "les développeurs", "une granularité"])


def test_line_break_hyphen_is_removed():
    d = cl.Dehyphenator(VOCAB)
    assert d.join("les déve-", "loppeurs et") == "les développeurs et"
    assert d.undecided == []


def test_real_hyphen_is_kept():
    d = cl.Dehyphenator(VOCAB)
    assert d.join("vos sous-", "traitants.") == "vos sous-traitants."


def test_unknown_word_is_merged_and_reported():
    d = cl.Dehyphenator(set())
    assert d.join("la pseudo-", "nymisation") == "la pseudonymisation"
    assert d.undecided == ["pseudo-nymisation"]


def test_plain_line_join():
    assert cl.Dehyphenator(VOCAB).join("une ligne", "suivante") == "une ligne suivante"


def test_cell_lines_keep_bullets_as_sub_items():
    cell = "Analyser si le RGPD s’applique au modèle.\n• Pour cela, établir la né-\ncessité d’attaques."
    lines = cl._cell_lines(cell, cl.Dehyphenator({"nécessité"}))
    assert lines == [
        "Analyser si le RGPD s’applique au modèle.",
        "- Pour cela, établir la nécessité d’attaques.",
    ]


def test_sections_numbering_and_subthemes():
    checklist = cl.Checklist(
        title="Développement des systèmes d'IA : Que faut-il vérifier ?",
        published=date(2025, 7, 22),
        intro="La CNIL propose une liste des points à vérifier.",
        themes=[
            {
                "number": "1",
                "title": "Déterminer le régime juridique applicable",
                "items": [
                    ("Identifier si le RGPD s’applique", ["Mesure A.", "- détail"]),
                    ("Identifier si le RGPD s’applique", ["Mesure B."]),
                    ("Définir les responsabilités", ["Mesure C."]),
                ],
            },
            {"number": "7", "title": "Assurer la transparence", "items": [(None, ["Informer."])]},
        ],
        undecided=[],
    )
    sections = cl.to_sections(checklist)
    assert [s.number for s in sections] == ["0", "1", "7"]
    theme = sections[1]
    assert theme.text.splitlines() == [
        "Identifier si le RGPD s’applique",
        "- Mesure A.",
        "  - détail",
        "- Mesure B.",
        "Définir les responsabilités",
        "- Mesure C.",
    ]
    assert [p.number for p in theme.paragraphs] == ["1.1", "1.2", "1.3"]
    assert theme.paragraphs[2].text == "[Définir les responsabilités] Mesure C."
    assert sections[2].paragraphs[0].text == "Informer."  # pas de sous-thème
    assert theme.label == (
        "CNIL, liste de vérification IA, point 1 « Déterminer le régime juridique applicable »"
    )
    assert {s.source_date for s in sections} == {date(2025, 7, 22)}
