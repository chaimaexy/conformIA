"""Relecture humaine du golden set dans Excel (ou LibreOffice).

    python -m app.evaluation.review export   # -> eval/relecture.csv
    python -m app.evaluation.review import   # relecture.csv -> eval/golden_set.jsonl
    python -m app.evaluation.review import eval/autre.csv   # autre fichier relu
    python -m app.evaluation.review export-nouvelles        # questions « a_relire » seulement

Dans le CSV, pour chaque question :
- statut     : a_relire -> « valide » ou « rejete »
- question, reponse_reference : corrigeables
- sections_attendues : corrigeables, format « ai_act / Article 113 » séparés par « | »,
                       ou « cnil_ia / fiche « Titre de la fiche » » pour une fiche entière
- extrait_section : aide à la relecture (non réimporté)
Le fichier utilise « ; » et l'encodage UTF-8 avec BOM : Excel l'ouvre correctement.
"""

import csv
import re
import sys
from pathlib import Path

from app.evaluation.golden import (
    EVAL_DIR,
    GOLDEN_PATH,
    ExpectedRef,
    GoldenItem,
    check_against_corpus,
    corpus_sections,
    load,
    save,
)

CSV_PATH = EVAL_DIR / "relecture.csv"
SOURCES = [EVAL_DIR / "golden_manual.jsonl", EVAL_DIR / "golden_generated.jsonl"]
FIELDS = [
    "id",
    "statut",
    "categorie",
    "question",
    "sections_attendues",
    "reponse_reference",
    "extrait_section",
    "origine",
    "commentaire",
]
_FICHE = re.compile(r"^fiche « (.+) »$")


def ref_to_text(ref: ExpectedRef) -> str:
    return f"{ref.source_id} / " + (ref.label if ref.label else f"fiche « {ref.fiche} »")


def text_to_refs(text: str) -> list[ExpectedRef]:
    refs = []
    for part in filter(None, (p.strip() for p in text.split("|"))):
        source_id, _, target = (x.strip() for x in part.partition("/"))
        m = _FICHE.match(target)
        refs.append(
            ExpectedRef(source_id=source_id, fiche=m.group(1))
            if m
            else ExpectedRef(source_id=source_id, label=target)
        )
    return refs


def merged_items() -> list[GoldenItem]:
    """Golden set existant, complété par les questions des fichiers sources pas encore reprises."""
    items = load()
    known = {i.id for i in items}
    for path in SOURCES:
        items += [i for i in load(path) if i.id not in known]
    return items


_KEY = re.compile(r"\d[\d  ]*\d|\d")  # dates, montants, numéros cités dans la réponse


def evidence(item: GoldenItem, sections: list) -> str:
    """Passages des sections attendues qui contiennent les chiffres de la réponse de référence
    (dates, montants...) ; à défaut, le début de la 1re section. Aide la relecture."""
    keys = {k for k in _KEY.findall(item.answer) if len(k) >= 2}
    lines = []
    for ref in item.expected:
        match = next((s for s in sections if ref.matches(s.source_id, s.label)), None)
        if match is None:
            continue
        found = [line for line in match.text.splitlines() if any(k in line for k in keys)]
        lines += [f"[{match.label}] {line}" for line in found[:4]]
    if not lines and item.expected:
        ref = item.expected[0]
        match = next((s for s in sections if ref.matches(s.source_id, s.label)), None)
        lines = [match.text[:600]] if match else []
    return "\n".join(lines)


def export(path: Path = CSV_PATH, pending_only: bool = False) -> None:
    items = merged_items()
    save(items)
    sections = corpus_sections()
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        writer.writeheader()
        for item in items:
            if pending_only and item.status != "a_relire":
                continue
            extract = evidence(item, sections)
            writer.writerow(
                {
                    "id": item.id,
                    "statut": item.status,
                    "categorie": item.category,
                    "question": item.question,
                    "sections_attendues": " | ".join(ref_to_text(r) for r in item.expected),
                    "reponse_reference": item.answer,
                    "extrait_section": extract,
                    "origine": item.origin,
                    "commentaire": "",
                }
            )
    print(f"Questions exportées -> {path}")


def import_(path: Path = CSV_PATH) -> None:
    by_id = {i.id: i for i in load()}
    errors = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            item = by_id.get(row["id"])
            if item is None:
                errors.append(f"{row['id']} : identifiant inconnu")
                continue
            try:
                by_id[item.id] = GoldenItem(
                    id=item.id,
                    question=row["question"].strip(),
                    category=row["categorie"].strip(),
                    expected=text_to_refs(row["sections_attendues"]),
                    answer=row["reponse_reference"].strip(),
                    origin=item.origin,
                    status=row["statut"].strip(),
                    note=(row.get("commentaire") or "").strip() or item.note,
                )
            except ValueError as exc:
                errors.append(f"{row['id']} : {exc}")
    items = list(by_id.values())
    errors += check_against_corpus([i for i in items if i.status != "rejete"], corpus_sections())
    if errors:
        raise SystemExit("Corrections nécessaires :\n" + "\n".join(errors))
    save(items)
    counts = {s: sum(i.status == s for i in items) for s in ("valide", "rejete", "a_relire")}
    print(f"Golden set mis à jour ({GOLDEN_PATH.name}) : {counts}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "export"
    if command == "import":
        import_(Path(sys.argv[2]) if len(sys.argv) > 2 else CSV_PATH)
    elif command == "export-nouvelles":  # seulement les questions à relire
        export(EVAL_DIR / "relecture_nouvelles.csv", pending_only=True)
    else:
        export()
