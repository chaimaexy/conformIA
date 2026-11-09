"""Affiche une section du corpus, pour la comparer à la main avec EUR-Lex.

Usage (depuis backend/) :
    python -m app.ingestion.show ai_act article 113
    python -m app.ingestion.show rgpd considerant 71
    python -m app.ingestion.show ai_act annexe III
"""

import argparse
import sys

from app.core.config import PROCESSED_DIR
from app.ingestion.models import LegalDocument


def main() -> None:
    parser = argparse.ArgumentParser(description="Afficher une section du corpus")
    parser.add_argument("source", help="ai_act, rgpd...")
    parser.add_argument("type", choices=["considerant", "article", "annexe"])
    parser.add_argument("number", nargs="+", help='ex. 6, "4 bis", III')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    path = PROCESSED_DIR / f"{args.source}.json"
    if not path.exists():
        raise SystemExit(f"{path} absent : lancer d'abord python -m app.ingestion.cli")
    document = LegalDocument.model_validate_json(path.read_text(encoding="utf-8"))
    number = " ".join(args.number)
    for s in document.sections:
        if s.section_type == args.type and s.number == number:
            print(f"{s.label} - {s.title or ''}".rstrip(" -"))
            for meta in (s.chapter, s.section):
                if meta:
                    print(f"  {meta}")
            print(f"  Source : {s.url}")
            if s.amended_by:
                print(f"  Modifié par : {', '.join(s.amended_by)}")
            if s.corrected_by:
                print(f"  Rectifié par : {', '.join(s.corrected_by)}")
            print("\n" + s.text)
            return
    raise SystemExit(f"{args.type} {number} introuvable dans {args.source}")


if __name__ == "__main__":
    main()
