"""Tests de l'application des rectificatifs (format du JO L 127 du 23.5.2018)."""

import pytest

from app.ingestion import corrigenda
from app.ingestion.models import LegalSection

CORRIGENDUM_HTML = """<html><body>
<p>Rectificatif au règlement (UE) 2016/679</p>
<p>Page 14, au considérant 71, cinquième phrase:</p>
<p>au lieu de:</p>
<p>«(71)</p>
<p>[…] que le risques d’erreur soit réduit au minimum.»</p>
<p>lire:</p>
<p>«(71)</p>
<p>[…] que le risque d’erreur soit réduit au minimum.»</p>
<p>Page 74, à l’article 65, paragraphe 1, point a):</p>
<p>au lieu de:</p>
<p>«a) lorsque l’autorité n’a pas donné suite à l’objection. […];»</p>
<p>lire:</p>
<p>«a) lorsque l’autorité de contrôle chef de file n’a pas donné suite à l’objection. […];».</p>
<p>ELI: http://data.europa.eu/eli/reg/2016/679/corrigendum/2018-05-23/oj</p>
</body></html>"""


def section(kind, number, text):
    return LegalSection(
        source_id="rgpd", section_type=kind, number=number, text=text, url="u", source_celex="x"
    )


def test_parse_entries():
    entries = corrigenda.parse(CORRIGENDUM_HTML, "32016R0679R(02)")
    assert [(e.section_type, e.number) for e in entries] == [
        ("considerant", "71"),
        ("article", "65"),
    ]
    old, new = entries[0].pairs[0]
    assert "[…]" in old and "le risque d’erreur" in new


def test_recital_is_corrected_and_article_is_verified():
    rct = section("considerant", "71", "Il faut que le risques d'erreur soit réduit au minimum.")
    art = section(
        "article",
        "65",
        "a) lorsque l’autorité de contrôle chef de file n’a pas donné suite à l’objection. "
        "La décision contraignante concerne toutes les questions;",
    )
    report = corrigenda.apply([rct, art], corrigenda.parse(CORRIGENDUM_HTML, "32016R0679R(02)"))

    assert rct.text == "Il faut que le risque d’erreur soit réduit au minimum."  # ’ vs ' toléré
    assert rct.corrected_by == ["32016R0679R(02)"]
    assert [line.split()[0] for line in report] == ["appliqué", "vérifié"]


def test_apply_is_idempotent():
    rct = section("considerant", "71", "Il faut que le risques d'erreur soit réduit au minimum.")
    art = section(
        "article",
        "65",
        "a) lorsque l’autorité de contrôle chef de file n’a pas donné suite à l’objection.",
    )
    entries = corrigenda.parse(CORRIGENDUM_HTML, "32016R0679R(02)")
    corrigenda.apply([rct, art], entries)
    once = rct.text
    corrigenda.apply([rct, art], entries)
    assert rct.text == once


def test_missing_correction_raises():
    rct = section("considerant", "71", "Texte sans rapport.")
    art = section("article", "65", "Version consolidée non corrigée.")
    with pytest.raises(ValueError, match="introuvable|absente"):
        corrigenda.apply([rct, art], corrigenda.parse(CORRIGENDUM_HTML, "32016R0679R(02)"))
