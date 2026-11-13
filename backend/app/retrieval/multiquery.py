"""Multi-query : le LLM reformule la question, chaque version est cherchée, tout est fusionné (RRF).

Une question d'utilisateur (« Depuis quand les pratiques interdites sont-elles applicables ? »)
n'emploie pas toujours les mots du texte qui y répond (« s'appliquent à partir du »). Trois
reformulations complémentaires élargissent la recherche :
- juridique   : le vocabulaire qu'emploierait le règlement ;
- mots_cles   : les termes essentiels, pour BM25 ;
- hypothese   : une phrase telle qu'elle pourrait figurer dans le texte (idée de HyDE).

Usage (depuis backend/) : précalcule les reformulations du golden set (cache, 2 requêtes/min)
    python -m app.retrieval.multiquery
"""

import logging

from app.llm.json_chat import chat_json

logger = logging.getLogger(__name__)
KINDS = ("juridique", "mots_cles", "hypothese")

PROMPT = """Tu aides un moteur de recherche sur des textes juridiques européens et français
(règlement sur l'IA dit « AI Act », RGPD, recommandations de la CNIL).

Question de l'utilisateur :
<question>
{question}
</question>

Propose trois reformulations en français pour retrouver les passages qui y répondent :
- "juridique" : la même question avec le vocabulaire juridique de ces textes (20 mots maximum) ;
- "mots_cles" : 4 à 8 mots clés de la question, séparés par des virgules ;
- "hypothese" : une AFFIRMATION (pas une question) de 25 mots maximum, au style d'un texte
  de loi, qui pourrait contenir la réponse.
Reste fidèle à la question : n'ajoute ni acteur, ni notion, ni texte qu'elle ne mentionne pas.
Ne cite aucun texte par son numéro (ni règlement, ni article, ni annexe). Texte brut.

Réponds uniquement en JSON : {{"juridique": "...", "mots_cles": "...", "hypothese": "..."}}"""


MODEL = "ministral-8b-latest"


def expand(question: str) -> list[str]:
    """Reformulations de la question (sans la question elle-même). Liste vide en cas d'échec."""
    try:
        # modèle de la mesure de phase 4 (réponses en cache) ; le multi-query a été écarté
        result = chat_json(PROMPT.format(question=question), model=MODEL)
    except Exception as exc:  # le multi-query ne doit jamais bloquer la recherche
        logger.warning("Multi-query indisponible (%s) : question seule", exc)
        return []
    if not isinstance(result, dict):
        return []
    variants = [_clean(result.get(kind)) for kind in KINDS]
    return [v for v in variants if v and v.lower() != question.lower()]


def _clean(value) -> str:
    """Le modèle renvoie parfois une liste ou du Markdown : on garde un texte brut."""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").replace("**", "").strip()[:500]


def main() -> None:
    from app.core.logging import setup_logging
    from app.evaluation.golden import load

    setup_logging()
    items = load()
    for i, item in enumerate(items, start=1):
        variants = expand(item.question)
        logger.info("%d/%d %s : %s", i, len(items), item.id, " || ".join(variants))


if __name__ == "__main__":
    main()
