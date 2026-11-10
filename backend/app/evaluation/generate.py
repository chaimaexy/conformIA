"""Génération assistée de questions pour le golden set (à relire par un humain).

Pour chaque section tirée au sort (échantillon stratifié par source et type), Mistral
propose DEUX questions dont la réponse est dans la section :
- « directe »    : reprend le vocabulaire du texte ;
- « reformulee » : cas pratique, avec les mots d'un chef de projet ou d'un DPO.
Offre gratuite : 2 requêtes/min -> une requête toutes les 31 s, reprise possible.

Usage (depuis backend/) :
    python -m app.evaluation.generate            # ~30 sections -> ~60 questions (~16 min)
"""

import json
import logging
import random
import time

import httpx

from app.core.config import get_app_config
from app.core.logging import setup_logging
from app.evaluation.golden import EVAL_DIR, ExpectedRef, GoldenItem, corpus_sections
from app.ingestion.models import LegalSection
from app.llm.client import _api_key

logger = logging.getLogger("golden.generate")
OUT = EVAL_DIR / "golden_generated.jsonl"
URL = "https://api.mistral.ai/v1/chat/completions"
INTERVAL = 31  # secondes (2 requêtes/min)

# (source, type de section) -> nombre de sections tirées
PLAN = {
    ("ai_act", "article"): 10,
    ("ai_act", "annexe"): 2,
    ("ai_act", "considerant"): 3,
    ("rgpd", "article"): 6,
    ("rgpd", "considerant"): 2,
    ("cnil_ia", "fiche"): 6,
    ("cnil_ia", "checklist"): 1,
}
# Articles de l'AI Act qui modifient d'autres règlements : peu utiles pour un utilisateur
EXCLUDED = {("ai_act", "article", str(n)) for n in range(102, 111)}

PROMPT = """Tu aides à construire un jeu de test pour un assistant juridique sur l'IA.

Voici un extrait d'un texte de référence :
<extrait>
{header}
{text}
</extrait>

Rédige DEUX questions en français auxquelles cet extrait répond, et pour chacune une
réponse de référence courte (1 à 2 phrases), fidèle au texte, sans rien inventer.
- "directe" : une question claire qui peut reprendre le vocabulaire de l'extrait.
- "reformulee" : une situation concrète posée par un chef de projet IA, un développeur
  ou un DPO, avec des mots du quotidien (par exemple « mon entreprise... »).
Ne mentionne aucun numéro d'article, d'annexe ou de considérant dans les questions.
Chaque question doit avoir une réponse précise DANS l'extrait.

Réponds uniquement en JSON :
{{"directe": {{"question": "...", "reponse": "..."}},
  "reformulee": {{"question": "...", "reponse": "..."}}}}"""


def sample_sections(sections: list[LegalSection], seed: int = 42) -> list[LegalSection]:
    rng = random.Random(seed)
    chosen = []
    for (source_id, kind), n in PLAN.items():
        pool = [
            s
            for s in sections
            if s.source_id == source_id
            and s.section_type == kind
            and len(s.text) >= 400
            and (source_id, kind, s.number) not in EXCLUDED
            and not (kind == "checklist" and s.number == "0")
        ]
        if kind == "fiche":  # au plus une partie par fiche, pour couvrir plus de fiches
            by_fiche: dict[str, list[LegalSection]] = {}
            for s in pool:
                by_fiche.setdefault(s.chapter or "", []).append(s)
            pool = [rng.choice(v) for v in by_fiche.values()]
        chosen += rng.sample(pool, min(n, len(pool)))
    return chosen


def ask(client: httpx.Client, section: LegalSection) -> dict:
    from app.chunking.common import header

    body = {
        "model": get_app_config().llm.chat_model,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": PROMPT.format(header=header(section), text=section.text[:3500]),
            }
        ],
    }
    for _ in range(5):
        r = client.post(URL, headers={"Authorization": f"Bearer {_api_key()}"}, json=body)
        if r.status_code == 200:
            return json.loads(r.json()["choices"][0]["message"]["content"])
        if r.status_code == 429 or r.status_code >= 500:
            logger.warning("Mistral %s, nouvel essai dans 60 s", r.status_code)
            time.sleep(60)
            continue
        r.raise_for_status()
    raise RuntimeError("Mistral : échec après 5 essais")


def _unwrap(result) -> dict:
    """Ramène les variantes de format renvoyées par le modèle à {"directe": .., "reformulee": ..}.

    Vu en pratique : {"questions": [{"directe": .., "reformulee": ..}]},
    {"questions": [q1, q2]}, [q1, q2], ou des clés avec majuscules / accents.
    """
    if isinstance(result, dict) and len(result) == 1:
        ((key, only),) = result.items()
        is_kind = key.lower().replace("é", "e").startswith(("direc", "refor"))
        if not is_kind and isinstance(only, list | dict):  # enveloppe type {"questions": ...}
            result = only
    if isinstance(result, list):
        dicts = [r for r in result if isinstance(r, dict)]
        keyed = any(
            k.lower().replace("é", "e").startswith(("direc", "refor")) for d in dicts for k in d
        )
        if keyed:  # [{"directe": ..}, {"reformulee": ..}] ou [{"directe": .., "reformulee": ..}]
            result = {k: v for d in dicts for k, v in d.items()}
        else:  # [question directe, question reformulée]
            result = dict(zip(("directe", "reformulee"), result, strict=False))
    return result if isinstance(result, dict) else {}


def _find(result, kind: str) -> dict:
    """Clé tolérante : « directe », « Directe », « reformulée »... (voir `_unwrap`)."""
    for key, value in _unwrap(result).items():
        normalized = key.lower().replace("é", "e").strip()
        if normalized.startswith(kind[:6]) and isinstance(value, dict):
            return value
    return {}


def main() -> None:
    setup_logging()
    sections = sample_sections(corpus_sections())
    # Reprise : on garde TOUTES les questions déjà écrites (2 par section, même `note`)
    items: list[GoldenItem] = []
    if OUT.exists():
        lines = OUT.read_text(encoding="utf-8").splitlines()
        items = [GoldenItem.model_validate_json(line) for line in lines if line.strip()]
    done = {item.note for item in items}  # note = clé de la section source
    client = httpx.Client(timeout=120)
    for i, section in enumerate(sections, start=1):
        key = f"{section.source_id}|{section.label}"
        if key in done:
            continue
        try:
            result = ask(client, section)
        except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("%s ignorée : %s", key, exc)
            continue
        found = 0
        for kind in ("directe", "reformulee"):
            q = _find(result, kind)
            if not q.get("question"):
                continue
            found += 1
            items.append(
                GoldenItem(
                    id=f"g{len(items) + 1:03d}",
                    question=q["question"].strip(),
                    category=kind,
                    expected=[ExpectedRef(source_id=section.source_id, label=section.label)],
                    answer=(q.get("reponse") or q.get("réponse") or "").strip(),
                    origin="genere",
                    note=key,
                )
            )
        if found < 2:
            logger.warning(
                "%s : %d question(s) exploitable(s), JSON reçu : %s",
                key,
                found,
                json.dumps(result, ensure_ascii=False)[:300],
            )
        OUT.write_text("".join(it.model_dump_json() + "\n" for it in items), encoding="utf-8")
        logger.info("%d/%d : %s -> %d questions au total", i, len(sections), key, len(items))
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
