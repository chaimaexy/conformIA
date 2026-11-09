"""Tests du parser des fiches CNIL sur un extrait reproduisant la structure de cnil.fr."""

from datetime import date

import pytest

from app.ingestion import cnil

URL = "https://www.cnil.fr/fr/ia-annoter-les-donnees"

FICHE_HTML = """<html><body><main>
<div class="ctn-gen">
  <div class="ctn-gen-titre"><h1> <span>IA : Annoter les données</span> </h1></div>
  <p class="ctn-gen-auteur">22 juillet 2025</p>
  <hr class="separator"/>
  <div class="page-introduction"><p>La phase d’annotation est cruciale.</p></div>
  <div class="page--tools"><button><span class="sr-only">Imprimer l’article</span></button></div>
  <div class="ctn-gen-body">
    <p>Elle sert en apprentissage <a href="/fr/definition/x">supervisé</a> (<em>ground truth</em>).</p>
    <p>Exemples :</p>
    <ul>
      <li>un modèle de reconnaissance ;</li>
      <li>un modèle de détection :
        <ul><li>de chutes ;</li></ul>
      </li>
    </ul>
    <h2>En pratique :</h2>
  </div>
  <div class="ctn-gen-after">
    <div class="ctn-gen-ascenseur"><div>
      <h2><span id="ascenseur-titre-1" class="ctn-gen-ascenseur-titre">Le principe de minimisation</span>
        <button class="ctn-gen-ascenseur-collapse"><span>Déplier</span></button></h2>
      <div class="ctn-gen-ascenseur-toCollapse"><div class="ctn-gen-ascenseur-texte">
        <p>Ne traiter que le <button class="glossary-tooltips" data-definition="x"><span class="glossary-tooltips-title">nécessaire</span></button>.</p>
        <div class="apparte">Exemple : la 5<sup>ème</sup> étape.</div>
      </div></div>
    </div></div>
    <div class="ctn-gen-push-3"><p><a href="#">&lt; Fiche précédente : Informer</a></p></div>
    <div class="general-tags"><ul><li><a href="#">#Intelligence artificielle (IA)</a></li></ul></div>
  </div>
  <script>const btn = document.getElementById('x');</script>
</div>
</main>
<div class="interest"><p class="date">27 août 2026</p></div>
</body></html>"""

INDEX_HTML = """<html><body><main>
<a href="/fr">Accueil</a>
<h3>Annoter les données</h3><p><a href="/fr/ia-annoter-les-donnees">&gt; En savoir plus</a></p>
<h3>Définir une finalité</h3><p><a href="/fr/definir-une-finalite-0">&gt; En savoir plus</a></p>
<h3>Liste de vérification</h3><p><a href="/sites/default/files/liste.pdf">&gt; En savoir plus</a></p>
</main></body></html>"""


@pytest.fixture(scope="module")
def sections():
    return cnil.parse(FICHE_HTML, URL)


def test_discover_keeps_html_fiches_only():
    assert cnil.discover(INDEX_HTML) == [
        "https://www.cnil.fr/fr/ia-annoter-les-donnees",
        "https://www.cnil.fr/fr/definir-une-finalite-0",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("22 juillet 2025", date(2025, 7, 22)), ("1er août 2024", date(2024, 8, 1)), ("hier", None)],
)
def test_parse_french_dates(text, expected):
    assert cnil.parse_date(text) == expected


def test_metadata(sections):
    assert [s.title for s in sections] == ["Introduction", "Le principe de minimisation"]
    assert {s.chapter for s in sections} == {"IA : Annoter les données"}
    assert {s.source_date for s in sections} == {date(2025, 7, 22)}  # pas la date d'un encart
    assert sections[1].url == f"{URL}#ascenseur-titre-1"
    assert sections[1].section == "En pratique :"  # intertitre sans contenu propre
    assert sections[1].label == "CNIL, « IA : Annoter les données », Le principe de minimisation"


def test_intro_body_and_nested_lists(sections):
    assert sections[0].text.splitlines() == [
        "La phase d’annotation est cruciale.",
        "Elle sert en apprentissage supervisé (ground truth).",
        "Exemples :",
        "- un modèle de reconnaissance ;",
        "- un modèle de détection :",
        "  - de chutes ;",
    ]


def test_glossary_terms_kept_and_noise_removed(sections):
    text = "\n".join(s.text for s in sections)
    assert "Ne traiter que le nécessaire." in text
    assert "Exemple : la 5ème étape." in text
    for noise in ("Déplier", "Imprimer", "Fiche précédente", "#Intelligence", "getElementById"):
        assert noise not in text


def test_fidelity_ok(sections):
    ok, n_chars, detail = cnil.fidelity(FICHE_HTML, sections)
    assert ok, detail
    assert n_chars > 100


def test_fidelity_detects_a_single_changed_character(sections):
    tampered = [s.model_copy() for s in sections]
    tampered[1] = tampered[1].model_copy(update={"text": tampered[1].text.replace("5ème", "6ème")})
    ok, _, detail = cnil.fidelity(FICHE_HTML, tampered)
    assert not ok
    assert "écart" in detail
