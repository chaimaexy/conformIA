# ConformIA

**Copilote RAG de conformité pour les projets d'IA** : AI Act (Règlement UE 2024/1689), RGPD et recommandations CNIL.

- **Q&A juridique sourcé** : chaque réponse cite l'article, le considérant ou l'annexe exact.
- **Analyse de conformité** (agent LangGraph) : à partir de la description d'un projet, l'agent détermine le niveau de risque, le rôle (fournisseur, déployeur) et la checklist des obligations.

> ⚠️ Projet pédagogique : ConformIA ne remplace pas un avis juridique.

## Stack

| Couche | Technologies |
|---|---|
| Frontend | Next.js, TypeScript, Supabase Auth |
| Backend | FastAPI, Python 3.13 |
| Données | Supabase (Postgres + pgvector + plein texte `french`) |
| IA | Mistral AI (`mistral-small` / `mistral-large`, `mistral-embed`), LangChain, LangGraph |
| Évaluation | RAGAS, pytest |
| DevOps | Docker, GitHub Actions |

## Architecture (cible)

```
EUR-Lex / CNIL ──► ingestion ──► chunking (structurel | récursif | sémantique)
                                      │
                                      ▼
                              embeddings Mistral ──► Supabase (pgvector + tsvector)
                                                              │
question ──► réécriture (historique) ──► multi-query ──► dense + plein texte ──► RRF ──► rerank
                                                                                         │
                                                                            Mistral ◄────┘
                                                                               │
                                                                   réponse + citations
```

## Structure

```
backend/
  app/
    ingestion/  chunking/  embeddings/  vectordb/  retrieval/
    prompts/    llm/       agents/      api/       workers/   core/
  eval/         # golden set + RAGAS
  tests/
  config.yaml   # paramètres non sensibles (modèles, chunking, retrieval)
frontend/       # Next.js (phase 7)
supabase/migrations/
data/           # raw/ et processed/ (non versionnés)
```

## Démarrage

```bash
cp .env.example .env          # puis renseigner MISTRAL_API_KEY et les clés Supabase
cd backend
py -3.13 -m venv .venv
.venv/Scripts/activate        # Linux/macOS : source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload # http://localhost:8000/docs
pytest
```

## Feuille de route

- [x] **Phase 0** : setup (backend, config, schéma Supabase, CI)
- [ ] **Phase 1** : ingestion AI Act + RGPD (EUR-Lex, FR) et fiches CNIL
- [ ] **Phase 2** : chunking (3 stratégies) + embeddings + indexation
- [ ] **Phase 3** : golden set (~80 questions) + évaluation de référence
- [ ] **Phase 4** : retrieval avancé (hybride, multi-query, RRF, reranking), mesuré à chaque étape
- [ ] **Phase 5** : génération avec citations, mémoire conversationnelle, streaming
- [ ] **Phase 6** : agent LangGraph d'analyse de conformité (human-in-the-loop)
- [ ] **Phase 7** : frontend Next.js, upload PDF (Celery + Redis), auth
- [ ] **Phase 8** : déploiement, démo vidéo

## Sources

- [Règlement (UE) 2024/1689 (AI Act)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32024R1689) - © Union européenne, réutilisation autorisée (décision 2011/833/UE)
- [Règlement (UE) 2016/679 (RGPD)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679)
- [CNIL - Intelligence artificielle](https://www.cnil.fr/fr/intelligence-artificielle)
