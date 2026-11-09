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
| IA | Mistral AI (`ministral-8b`, `mistral-embed`), encodeurs BERT locaux (`bge-m3`, cross-encodeur `bge-reranker-v2-m3` ou CamemBERT), LangChain, LangGraph |
| Évaluation | métriques de retrieval (recall@k, MRR), RAGAS sur échantillon, pytest |
| DevOps | Docker, GitHub Actions |

## Choix techniques

Contraintes assumées : offre Mistral gratuite (2 requêtes/min sur `ministral-8b`), et un
PC portable sans GPU (i5, 8 Go de RAM). Chaque brique doit tourner sur CPU.

| Brique | Choix | Pourquoi |
|---|---|---|
| Génération | **Mistral** (`ministral-8b`, configurable) | Seul un modèle génératif rédige une réponse ; un BERT est un encodeur, il ne génère pas de texte. 1 appel par question, cache et retry |
| Embeddings | **`mistral-embed`**, comparé à **`bge-m3`** (XLM-RoBERTa, famille BERT, local) | Mesurer si un encodeur ouvert multilingue fait mieux en français ; même dimension (1024) |
| Reranking | **Cross-encodeur BERT local** : `bge-reranker-v2-m3` vs CamemBERT français (`crossencoder-camembert-base-mmarcoFR`) | C'est là que les encodeurs BERT excellent ; choix final par qualité et latence CPU |
| Garde-fou | Seuil sur le score du reranker : **refuser de répondre** si le corpus ne couvre pas la question | Idée des « modèles de décision » (Jev, GLiDE), réalisée avec des modèles ouverts |
| Vérification | Chaque phrase de la réponse doit s'appuyer sur un extrait cité | Pas d'affirmation juridique sans source |

**Technologies étudiées et écartées** (novembre 2026) :

| Technologie | Ce que c'est | Pourquoi écartée |
|---|---|---|
| [GLiDE](https://huggingface.co/collections/fastino/glide) (Fastino) | Modèles de décision (oui/non, choix, score) de 0,4B à 27B | Langues non documentées (français non garanti), licence propriétaire « Fastino Community », dépendances figées sur CUDA, testé uniquement sur GPU 96 Go |
| [open-jev](https://github.com/kyegomez/open-jev) | Réimplémentation non officielle de Jev (TypeSafe AI) | Aucun poids entraîné publié (poids aléatoires) ; il faudrait l'entraîner soi-même |
| Jev (TypeSafe AI) | Modèle de décision hébergé | Propriétaire, en accès anticipé, aucune information sur le français |

## Architecture (cible)

```
EUR-Lex / CNIL ──► ingestion ──► chunking (structurel | récursif | sémantique)
                                      │
                                      ▼
                 embeddings (mistral-embed | bge-m3) ──► Supabase (pgvector + tsvector)
                                                              │
question ──► réécriture (historique) ──► multi-query ──► dense + plein texte ──► RRF
                                                                                  │
                                              cross-encodeur BERT (rerank + garde-fou)
                                                                                  │
                                                 Mistral ◄── contexte suffisant ? ┘
                                                    │              non -> « je ne sais pas »
                                       réponse + citations vérifiées
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

Encodeurs locaux (bge-m3, cross-encodeurs), sur CPU :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[local]"
```

> **Windows 11 et Smart App Control** : ce contrôle bloque les DLL de paquets publiés
> très récemment (pas encore de « réputation »), avec l'erreur *« Une stratégie de contrôle
> d'application a bloqué ce fichier »*. Solution sans désactiver la protection : installer
> une version plus ancienne et répandue du paquet concerné (ici `regex==2025.11.3` et
> `scikit-learn==1.7.2`).

## Ingestion du corpus

```bash
cd backend
python -m app.ingestion.cli                          # AI Act + RGPD + CNIL -> data/processed/*.json
python -m app.ingestion.cli cnil_ia                  # une seule source
python -m app.ingestion.cli --strict                 # échoue si une source est périmée
python -m app.ingestion.cli --refresh                # re-télécharge les textes
python -m app.ingestion.show ai_act article 113      # affiche une section pour vérification
```

**Exactitude des données** : un assistant juridique qui cite un texte périmé est dangereux.
L'ingestion garantit que le corpus correspond au droit en vigueur :

| Contrôle | Comment |
|---|---|
| Source officielle | Dépôt **Cellar** de l'Office des publications de l'UE (accès machine officiel ; eur-lex.europa.eu bloque les robots) |
| Droit en vigueur | Articles et annexes tirés de la **dernière version consolidée** (l'AI Act intègre l'omnibus numérique, règlement (UE) 2026/1744) |
| Considérants | Absents des versions consolidées : tirés du JO d'origine, puis **rectificatifs français appliqués** automatiquement |
| Rectificatifs | Chaque correction est appliquée (considérants) ou vérifiée dans la version consolidée (articles) ; échec = erreur |
| Fraîcheur | UE : requête SPARQL à chaque ingestion (consolidation ou rectificatif FR plus récent ?). CNIL : liste des fiches comparée à la précédente, cache de plus de 30 jours signalé |
| Fidélité | UE : chaque paragraphe du HTML source se retrouve dans le texte extrait. CNIL : comparaison **caractère par caractère** (ponctuation comprise) entre la page et l'extrait. 100 % exigé |
| Provenance | [data/manifest.json](data/manifest.json) : CELEX ou URL, SHA-256 et date d'extraction de chaque fichier source |

| Source | Contenu | Version |
|---|---|---|
| AI Act (2024/1689) | 180 considérants, 119 articles, 14 annexes | consolidée au 27/07/2026 |
| RGPD (2016/679) | 173 considérants, 99 articles | consolidée au 04/05/2016 + rectificatif 2018 |
| CNIL, fiches pratiques IA | 17 fiches, 92 sections | dates de publication de chaque fiche (2024-2025) |
| CNIL, liste de vérification (PDF) | 11 thèmes, 79 mesures numérotées (1.1 à 11.2) | 22/07/2025 |

Les fiches CNIL sont découpées par partie (titre `h2`, blocs dépliables compris), avec le lien
direct vers chaque partie. Boutons, navigation, encarts « Ceci peut vous intéresser » et
scripts sont exclus ; les termes du glossaire restent dans la phrase.

La liste de vérification est un tableau PDF (thème / sous-thème / mesure) lu avec `pdfplumber` ;
les cellules fusionnées sont propagées, les coupures de mots (« déve-loppeurs ») sont tranchées
grâce au vocabulaire des fiches (« sous-traitants » garde son trait d'union). Deux contrôles :
même multiensemble de caractères que le PDF entier, et colonne des mesures identique **dans
l'ordre**. Elle servira de grille à l'agent de conformité (phase 6).

> Les textes consolidés sont un outil de documentation : seules les versions publiées au
> Journal officiel font foi.

## Chunking, embeddings et indexation

```bash
cd backend
python -m app.indexing                 # 3 stratégies, 2 modèles, envoi dans Supabase
python -m app.indexing --no-upload     # calcul local seulement (cache data/cache/)
```

| Stratégie | Principe | Chunks | Taille médiane | < 100 car. | À cheval sur 2 sections |
|---|---|---|---|---|---|
| **Structurelle** | 1 unité juridique = 1 chunk ; découpe aux paragraphes / intertitres si > 2 000 car. ; en-tête de contexte (« AI Act > CHAPITRE III > Article 6 - ... (partie 2/3 - paragraphe 1) ») | 1 190 | 1 402 | 1 | 0 |
| **Récursive** (référence des tutoriels) | `RecursiveCharacterTextSplitter`, 1 000 car., chevauchement 150, sans structure ni en-tête | 2 354 | 764 | 189 | 24 |
| **Sémantique** | Coupe là où deux phrases consécutives s'éloignent (10 % des distances cosinus les plus fortes), sans franchir les sections | 1 334 | 1 084 | 4 | 0 |

Plan d'expérience, pour comparer à coût raisonnable sans GPU : les 3 stratégies avec
`mistral-embed` (comparaison du chunking), et la stratégie structurelle aussi avec `bge-m3`
(comparaison des modèles). Les vecteurs sont mis en cache (SQLite) : ré-indexer ne coûte rien.

| Modèle | Où | Temps pour 1 190 chunks (i5-10210U, sans GPU) |
|---|---|---|
| `mistral-embed` | API Mistral (lots de 64, 60 requêtes/min) | ~1 min |
| `bge-m3` (XLM-RoBERTa, 568M paramètres) | CPU local | **74 min** |

Supabase contient 4 878 chunks (~35 Mo, colonnes `embedding` et `embedding_bge_m3`, index HNSW
avec parcours itératif pgvector 0.8 pour filtrer par stratégie sans perdre de résultats).

**Premier test, premier échec instructif.** À la question « À partir de quand les règles sur les
systèmes d'IA à haut risque de l'annexe III s'appliquent-elles ? », la bonne réponse est
l'article 113 (2 décembre 2027, version omnibus). La recherche dense le classe au rang 5
(récursif), 11 (structurel), 19 (sémantique) et au-delà de 50 (bge-m3) : elle ramène les articles
qui *parlent* de haut risque (6, 7) plutôt que celui qui donne les *dates*. C'est ce que le golden
set (phase 3) mesurera, et ce que la recherche hybride, le multi-query et le reranker (phase 4)
doivent corriger.

## Feuille de route

- [x] **Phase 0** : setup (backend, config, schéma Supabase, CI)
- [x] **Phase 1a** : ingestion AI Act + RGPD (versions consolidées, rectificatifs, contrôles qualité)
- [x] **Phase 1b** : fiches pratiques IA de la CNIL (17 fiches + liste de vérification PDF, fidélité caractère par caractère)
- [x] **Phase 2** : chunking (3 stratégies) + embeddings (`mistral-embed` vs `bge-m3`) + indexation Supabase (4 878 chunks)
- [ ] **Phase 3** : golden set (~80 questions, dont des questions hors corpus) + évaluation de référence
- [ ] **Phase 4** : retrieval avancé (hybride, multi-query, RRF, cross-encodeur BERT), garde-fou « je ne sais pas » calibré, mesuré à chaque étape
- [ ] **Phase 5** : génération Mistral avec citations vérifiées, mémoire conversationnelle, streaming
- [ ] **Phase 6** : agent LangGraph d'analyse de conformité, guidé par la liste de vérification CNIL (human-in-the-loop)
- [ ] **Phase 7** : frontend Next.js, upload PDF (Celery + Redis), auth
- [ ] **Phase 8** : déploiement, démo vidéo

## Sources

- [Règlement (UE) 2024/1689 (AI Act)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32024R1689), [version consolidée](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:02024R1689-20260727), modifié par le [règlement (UE) 2026/1744](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32026R1744) - © Union européenne, réutilisation autorisée (décision 2011/833/UE)
- [Règlement (UE) 2016/679 (RGPD)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679), [version consolidée](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:02016R0679-20160504)
- [CNIL - Fiches pratiques IA](https://www.cnil.fr/fr/les-fiches-pratiques-ia) - Source : CNIL – https://www.cnil.fr, licence [CC-BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/deed.fr) : reproduction à l'identique, avec titre, source, date d'extraction (voir [data/manifest.json](data/manifest.json)) et licence
