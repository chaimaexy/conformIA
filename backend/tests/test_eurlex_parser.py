"""Tests du parser EUR-Lex sur des extraits reproduisant les deux formats HTML.

- OJ_HTML : Journal officiel d'origine (source des considérants)
- CONSOLIDATED_HTML : version consolidée (source des articles et annexes)
"""

import pytest

from app.ingestion import eurlex

ROW = (
    '<table><col width="4%"/><col width="96%"/><tbody><tr>'
    '<td valign="top"><p class="oj-normal">{label}</p></td>'
    '<td valign="top"><p class="oj-normal">{text}</p></td>'
    "</tr></tbody></table>"
)
GRID = (
    '<div class="grid-container grid-list">'
    '<div class="list grid-list-column-1"><span>{label} </span></div>'
    '<div class="grid-list-column-2"><p class="norm">{text}</p></div>'
    "</div>"
)
M1 = '<p class="modref"><a href="#" title="32026R1744: REPLACED">▼M1</a></p>'
B = '<p class="modref"><a href="#" title="32024R1689">▼B</a></p>'

OJ_HTML = f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p class="oj-hd-date">12.7.2024</p>
<div id="pbl_1">
  <div class="eli-subdivision" id="rct_1">
    {ROW.format(label="(1)", text="L’objectif du présent règlement est d’améliorer le marché.")}
  </div>
  <div class="eli-subdivision" id="rct_2">
    {ROW.format(label="(2)", text='Voir la directive du Conseil <a id="ntc1-X" href="#ntr1-X">(<span class="oj-super oj-note-tag">1</span>)</a>. Fin.')}
  </div>
</div>
<div id="enc_1">
  <div id="cpt_XIII">
    <p class="oj-ti-section-1">CHAPITRE XIII</p>
    <div class="eli-title"><p class="oj-ti-section-2">DISPOSITIONS FINALES</p></div>
    <div class="eli-subdivision" id="art_113">
      <p class="oj-ti-art">Article 113</p>
      <div class="eli-title"><p class="oj-sti-art">Entrée en vigueur</p></div>
      <p class="oj-normal">Publié au <span class="oj-italic">Journal officiel</span>.</p>
      {ROW.format(label="a)", text="les chapitres I et II s’appliquent le 2 février 2025;")}
      <p class="oj-note">(1) Note de bas de page à ignorer.</p>
    </div>
  </div>
</div>
</body></html>"""

CONSOLIDATED_HTML = f"""<html><body>
<div id="enc_1">
  <div id="cpt_I">
    <p class="title-division-1">CHAPITRE I</p>
    <p class="title-division-2">DISPOSITIONS GÉNÉRALES</p>
    <div class="eli-subdivision" id="art_1">
      <p class="title-article-norm">Article premier</p>
      <div class="eli-title"><p class="stitle-article-norm">Objet</p></div>
      <p class="norm">Le présent règlement établit des règles.</p>
    </div>
    <div class="eli-subdivision" id="art_4a">
      <p class="title-article-norm">Article 4 bis</p>
      <div class="eli-title"><p class="stitle-article-norm">Détection des biais</p></div>
      {M1}
      <div class="norm"><span class="no-parag">1.  </span>
        <div class="norm inline-element"><p class="norm inline-element">Premier paragraphe:</p>
          {GRID.format(label="a)", text="point a;")}
        </div>
      </div>
      <div class="norm"><span class="no-parag">1 <span class="italics">bis</span>.  </span>
        <div class="norm inline-element"><span><a href="#" title="32016R0679R(02): REPLACED"><span class="boldface">►C1</span></a></span> Texte rectifié<span class="boldface"> ◄ </span> et suite.</div>
      </div>
    </div>
  </div>
  <div id="cpt_XIII">
    <p class="title-division-1">CHAPITRE XIII</p>
    <p class="title-division-2">DISPOSITIONS FINALES</p>
    <div class="eli-subdivision" id="art_113">
      <p class="title-article-norm">Article 113</p>
      <div class="eli-title"><p class="stitle-article-norm">Entrée en vigueur et application</p></div>
      <p class="norm">Toutefois:</p>
      {M1}
      {GRID.format(label="c)", text="les sections 1 à 3 s'appliquent à partir du 2 décembre 2027;")}
      {B}
    </div>
  </div>
</div>
<div id="anx_III">
  <p><br/></p><hr class="separator-annex"/>
  <p class="title-annex-1">ANNEXE III</p>
  <p class="title-annex-2">Systèmes d'IA à haut risque visés à l'article 6, paragraphe 2</p>
  <p class="norm">Domaines:</p>
  {GRID.format(label="1.", text="Biométrie")}
</div>
<div id="anx_XIV">
  <p class="title-annex-1">ANNEXE XIV</p>
  <p class="norm">Liste des codes des systèmes d'IA</p>
</div>
</body></html>"""


@pytest.fixture(scope="module")
def recitals():
    return {s.number: s for s in eurlex.parse_recitals(OJ_HTML, "ai_act", "32024R1689")}


@pytest.fixture(scope="module")
def body():
    sections = eurlex.parse_body(CONSOLIDATED_HTML, "ai_act", "02024R1689-20260727", "32024R1689")
    return {(s.section_type, s.number): s for s in sections}


# --------------------------------------------------------------------------- JO d'origine


def test_recitals(recitals):
    assert list(recitals) == ["1", "2"]
    assert recitals["1"].text.startswith("L’objectif")  # « (1) » retiré
    assert recitals["1"].url.endswith("32024R1689#rct_1")


def test_footnote_call_removed_without_space_before_dot(recitals):
    assert recitals["2"].text == "Voir la directive du Conseil. Fin."


def test_oj_body_still_supported():
    sections = eurlex.parse_body(OJ_HTML, "ai_act", "32024R1689", "32024R1689")
    art = sections[0]
    assert art.chapter == "CHAPITRE XIII - DISPOSITIONS FINALES"
    assert "Publié au Journal officiel." in art.text
    assert "a) les chapitres I et II" in art.text
    assert "Note de bas de page" not in art.text


def test_publication_date():
    assert eurlex.publication_date(OJ_HTML).isoformat() == "2024-07-12"


# --------------------------------------------------------------------------- version consolidée


def test_official_article_numbers(body):
    assert ("article", "1") in body  # « Article premier »
    art = body[("article", "4 bis")]
    assert art.label == "Article 4 bis"
    assert art.url.endswith("#art_4a")  # l'ancre EUR-Lex garde l'identifiant technique


def test_consolidated_paragraphs_and_enumerations(body):
    art = body[("article", "4 bis")]
    assert [p.number for p in art.paragraphs] == ["1", "1 bis"]
    assert art.text.splitlines()[:2] == ["1. Premier paragraphe:", "a) point a;"]
    assert art.chapter == "CHAPITRE I - DISPOSITIONS GÉNÉRALES"


def test_markers_removed_and_inline_text_kept_together(body):
    art = body[("article", "4 bis")]
    assert "1 bis. Texte rectifié et suite." in art.text.splitlines()
    for marker in ("▼", "►", "◄"):
        assert marker not in art.text


def test_amendments_and_corrections_are_tracked(body):
    art = body[("article", "4 bis")]
    assert art.amended_by == ["32026R1744"]
    assert art.corrected_by == ["32016R0679R(02)"]
    assert body[("article", "1")].amended_by == []


def test_amended_dates(body):
    art = body[("article", "113")]
    assert "c) les sections 1 à 3 s'appliquent à partir du 2 décembre 2027;" in art.text
    assert "Article 113" not in art.text  # le titre n'est pas répété dans le corps


def test_annexes(body):
    assert (
        body[("annexe", "III")].title
        == "Systèmes d'IA à haut risque visés à l'article 6, paragraphe 2"
    )
    assert body[("annexe", "III")].text.splitlines() == ["Domaines:", "1. Biométrie"]
    assert (
        body[("annexe", "XIV")].title == "Liste des codes des systèmes d'IA"
    )  # pas de title-annex-2


# --------------------------------------------------------------------------- fidélité


def test_coverage_is_complete(recitals, body):
    assert eurlex.coverage(OJ_HTML, list(recitals.values())) == []
    assert eurlex.coverage(CONSOLIDATED_HTML, list(body.values())) == []


def test_coverage_detects_lost_text(body):
    damaged = [s.model_copy(update={"text": "texte tronqué"}) for s in body.values()]
    assert eurlex.coverage(CONSOLIDATED_HTML, damaged)
