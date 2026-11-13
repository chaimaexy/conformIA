"""Juge de fidélité et d'exactitude : ce que les contrôles par le code ne voient pas.

Le code vérifie les renvois, les citations mot pour mot et les chiffres. Il ne voit pas un
raisonnement faux sans chiffre (« la date dépend de l'entrée en vigueur dans votre État
membre », faux pour un règlement). Le juge, à la manière de RAGAS :
1. découpe la réponse en affirmations juridiques ;
2. note chacune contre les SOURCES fournies à l'assistant : soutenue / non soutenue / contredite
   (fidélité : l'assistant n'a-t-il rien ajouté de sa mémoire ?) ;
3. compare la réponse à la RÉPONSE DE RÉFÉRENCE validée : correcte / partielle / incorrecte.

Le juge (`llm.judge_model`, mistral-large) raisonne avec l'effort par défaut, plus long que
celui des réponses (`low`). Comme il est du même modèle que celui qui répond, ses verdicts ont
été recoupés à la main (12 réponses de ministral-8b, 8 de mistral-large ; voir le README).
"""

from dataclasses import dataclass, field

from app.core.config import get_app_config
from app.llm.json_chat import chat_json

VERDICTS = ("soutenue", "non_soutenue", "contredite")
ACCURACY = ("correcte", "partielle", "incorrecte")

PROMPT = """Tu es un juriste chargé de contrôler la réponse d'un assistant juridique.

QUESTION :
{question}

RÉPONSE DE RÉFÉRENCE (validée par un humain) :
{reference}

SOURCES fournies à l'assistant (il devait répondre uniquement à partir d'elles) :
{sources}

RÉPONSE DE L'ASSISTANT :
{answer}

Travail demandé :
1. Découpe la réponse de l'assistant en affirmations juridiques (une idée par affirmation ; \
ignore les formules d'introduction, les avertissements et la mention de ce qui n'est pas couvert).
2. Pour chaque affirmation, donne un verdict par rapport aux SOURCES seulement :
   - "soutenue" : les sources le disent, explicitement ou par une déduction immédiate ;
   - "non_soutenue" : les sources ne le disent pas (même si c'est vrai par ailleurs) ;
   - "contredite" : les sources disent autre chose.
3. Exactitude de la réponse par rapport à la RÉPONSE DE RÉFÉRENCE :
   - "correcte" : elle contient l'essentiel de la référence, sans erreur ;
   - "partielle" : elle est juste mais omet un point important de la référence ;
   - "incorrecte" : elle contient une erreur juridique ou contredit la référence.

Réponds en JSON :
{{"affirmations": [{{"texte": "...", "verdict": "soutenue"}}], \
"exactitude": "correcte", "erreurs": ["erreur juridique précise, s'il y en a"]}}"""


@dataclass
class Judgement:
    claims: list[dict] = field(default_factory=list)  # {"texte", "verdict"}
    accuracy: str = "incorrecte"
    errors: list[str] = field(default_factory=list)

    @property
    def unsupported(self) -> list[dict]:
        return [c for c in self.claims if c["verdict"] != "soutenue"]

    @property
    def faithfulness(self) -> float | None:
        if not self.claims:
            return None
        return sum(c["verdict"] == "soutenue" for c in self.claims) / len(self.claims)


def parse(data: dict) -> Judgement:
    """Réponse du juge -> Judgement ; valeurs inconnues ramenées au cas le plus sévère."""
    claims = []
    for claim in data.get("affirmations") or []:
        if isinstance(claim, dict) and str(claim.get("texte", "")).strip():
            verdict = claim.get("verdict")
            claims.append(
                {
                    "texte": str(claim["texte"]).strip(),
                    "verdict": verdict if verdict in VERDICTS else "non_soutenue",
                }
            )
    accuracy = data.get("exactitude")
    errors = [str(e) for e in data.get("erreurs") or [] if str(e).strip()]
    return Judgement(claims, accuracy if accuracy in ACCURACY else "incorrecte", errors)


def judge(question: str, reference: str, sources: list[str], answer: str) -> Judgement:
    blocks = "\n\n".join(f"[{i}]\n{text}" for i, text in enumerate(sources, start=1))
    prompt = PROMPT.format(question=question, reference=reference, sources=blocks, answer=answer)
    data = chat_json(prompt, model=get_app_config().llm.judge_model)
    return parse(data if isinstance(data, dict) else {})
