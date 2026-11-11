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
| Données | Supabase (Postgres + pgvector) |
| IA | Mistral AI (`ministral-8b`, `mistral-embed`), `bge-m3` local (BERT), BM25, reranker Cohere `rerank-v5.0-pro`, LangChain, LangGraph |
| Évaluation | métriques de retrieval (recall@k, MRR), RAGAS sur échantillon, pytest |
| DevOps | Docker, GitHub Actions |

## Choix techniques

Contraintes assumées : offre Mistral gratuite (2 requêtes/min sur `ministral-8b`), et un
PC portable sans GPU (i5, 8 Go de RAM). Chaque brique doit tourner sur CPU.

| Brique | Choix | Pourquoi |
|---|---|---|
| Génération | **Mistral** (`ministral-8b`, configurable) | Seul un modèle génératif rédige une réponse ; un BERT est un encodeur, il ne génère pas de texte. 1 appel par question, cache et retry |
| Embeddings | **`mistral-embed`**, comparé à **`bge-m3`** (XLM-RoBERTa, famille BERT, local) | Mesurer si un encodeur ouvert multilingue fait mieux en français ; même dimension (1024) |
| Recherche | **Hybride** : 2 recherches denses (`mistral-embed`, `bge-m3`) + BM25, fusion RRF | Le dense comprend les reformulations, BM25 les mots rares ; +15 points de hit@8 (phase 4) |
| Reranking | **Cohere `rerank-v5.0-pro`** (API), comparé à `bge-reranker-v2-m3`, Qwen3-Reranker et CamemBERT en local | Meilleure qualité, < 1 s par question ; les modèles locaux prennent de 20 s à plusieurs minutes sur CPU |
| Garde-fou | Seuil sur le score du reranker : **refuser de répondre** si le corpus ne couvre pas la question | Idée des « modèles de décision » (Jev, GLiDE) ; seul le score Cohere sépare parfaitement les questions hors corpus |
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
                 embeddings (mistral-embed | bge-m3) ──► Supabase (pgvector)
                                                              │
question ──► réécriture (historique) ──► 2 denses + BM25 ──► RRF
                                                              │
                                          reranker Cohere (rerank + garde-fou)
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
| **Récursive** (référence des tutoriels) | `RecursiveCharacterTextSplitter` avec ses réglages par défaut, 1 000 car., chevauchement 150, sans structure ni en-tête | 2 329 | 804 | 187 | 24 |
| **Sémantique** | Coupe là où deux phrases consécutives s'éloignent (10 % des distances cosinus les plus fortes), sans franchir les sections | 1 334 | 1 084 | 4 | 0 |

Plan d'expérience, pour comparer à coût raisonnable sans GPU : les 3 stratégies avec
`mistral-embed` (comparaison du chunking), et la stratégie structurelle aussi avec `bge-m3`
(comparaison des modèles). Les vecteurs sont mis en cache (SQLite) : ré-indexer ne coûte rien.

| Modèle | Où | Temps pour 1 190 chunks (i5-10210U, sans GPU) |
|---|---|---|
| `mistral-embed` | API Mistral (lots de 64, 60 requêtes/min) | ~1 min |
| `bge-m3` (XLM-RoBERTa, 568M paramètres) | CPU local | **74 min** |

Supabase contient 4 853 chunks (~35 Mo, colonnes `embedding` et `embedding_bge_m3`, index HNSW
avec parcours itératif pgvector 0.8 pour filtrer par stratégie sans perdre de résultats).

**Premier test, premier échec instructif.** À la question « À partir de quand les règles sur les
systèmes d'IA à haut risque de l'annexe III s'appliquent-elles ? », la bonne réponse est
l'article 113 (2 décembre 2027, version omnibus). La recherche dense le classe au rang 5
(récursif), 11 (structurel), 19 (sémantique) et au-delà de 50 (bge-m3) : elle ramène les articles
qui *parlent* de haut risque (6, 7) plutôt que celui qui donne les *dates*. C'est ce que le golden
set (phase 3) mesurera, et ce que la recherche hybride, le multi-query et le reranker (phase 4)
doivent corriger.

## Évaluation : le golden set

```bash
cd backend
python -m app.evaluation.generate          # questions proposées par Mistral (2 par section)
python -m app.evaluation.review export     # -> eval/relecture.csv, à relire dans Excel
python -m app.evaluation.review import [fichier.csv]   # relecture -> eval/golden_set.jsonl (validé contre le corpus)
python -m app.evaluation.retrieval --validated   # hit@k, recall@k, MRR par configuration
```

Le golden set ([eval/golden_set.jsonl](backend/eval/golden_set.jsonl)) associe chaque question
aux **sections** du corpus qui contiennent la réponse (« ai_act / Article 113 », ou une fiche
CNIL entière). Noter au niveau de la section, et non du chunk, rend la comparaison équitable
entre stratégies de découpage.

| Catégorie | Ce qu'elle teste | Exemple |
|---|---|---|
| directe | vocabulaire du texte | « Quelles obligations pour les fournisseurs de modèles d'IA à usage général ? » |
| reformulee | cas pratique, mots du quotidien | « Notre logiciel trie des CV : est-ce à haut risque ? » |
| reference | numéro cité | « Que dit l'article 22 du RGPD ? » |
| date | calendrier modifié par l'omnibus | « Depuis quand les pratiques interdites s'appliquent-elles ? » |
| multi_sources | plusieurs textes | « Biais et données sensibles : AI Act art. 4 bis + RGPD art. 9 » |
| cnil_pratique | recommandations CNIL | « Peut-on se fonder sur l'intérêt légitime pour entraîner un modèle ? » |
| hors_corpus | savoir dire « je ne sais pas » | « Quel taux de TVA pour un logiciel SaaS ? » |

Construction : 36 questions écrites à la main (cas difficiles, hors corpus) et 60 questions
proposées par Mistral à partir de 30 sections tirées au sort (échantillon stratifié). Toute
section attendue est vérifiée contre le corpus à l'import.

**Relecture, en trois passes :**
1. Relecture par un autre LLM (ChatGPT), **sans accès aux textes** : 8 rejets, dont 7 étaient
   faux (il jugeait de mémoire, par exemple « l'article 78 ne parle pas de suppression des
   données » alors que son paragraphe 2 le prévoit). Seules 2 corrections ont été retenues.
   Leçon : un LLM qui juge du droit sans la source se trompe avec assurance, et c'est
   précisément le problème que ce RAG doit résoudre.
2. Vérification **contre le texte exact** de chaque section attendue (96 questions) :
   1 erreur de sens corrigée (secret professionnel « sauf à l'égard des » autorités
   notifiantes, inversé), 1 mécanisme inventé, des exemples ou affirmations absents du texte,
   des réponses incomplètes ; et 11 sections qui répondent **aussi** à une question ajoutées
   aux sections attendues (ex. annexe XI en plus de l'annexe XII, considérant 30 pour les
   adresses IP), pour ne pas compter à tort un bon résultat comme un échec.
3. Arbitrage humain des 5 cas où la question ou la vérité terrain elle-même avait changé
   (tous validés). Résultat : 96 questions validées, 0 rejetée.

Le détail (avant/après et justification de chaque correction) est dans
[eval/relecture_verifiee.csv](backend/eval/relecture_verifiee.csv), colonne `commentaire`.

Métriques, sans appel au LLM : **hit@k** (une section attendue dans les k premiers chunks),
**recall@8** (part des sections attendues retrouvées, utile pour les questions multi-sources),
**MRR@20** (1/rang du premier bon chunk). Pour les questions hors corpus, on relève le score du
premier résultat : il servira à calibrer le garde-fou « je ne sais pas ».

### Résultats de référence (recherche dense seule)

96 questions validées (88 dans le corpus, 8 hors corpus).
Détail par question : [eval/results/baseline.json](backend/eval/results/baseline.json).

| Configuration | hit@1 | hit@4 | hit@8 | hit@20 | recall@8 | MRR@20 |
|---|---|---|---|---|---|---|
| récursif + mistral-embed | 0.39 | 0.61 | 0.70 | 0.83 | 0.64 | 0.498 |
| sémantique + mistral-embed | 0.44 | 0.67 | 0.75 | **0.89** | 0.69 | 0.557 |
| structurel + mistral-embed | **0.52** | 0.67 | 0.74 | 0.86 | 0.69 | **0.609** |
| structurel + bge-m3 | 0.44 | **0.69** | **0.77** | 0.86 | **0.73** | 0.563 |

hit@8 par catégorie :

| Catégorie (n) | récursif | sémantique | structurel | structurel + bge-m3 |
|---|---|---|---|---|
| directe (35) | 0.77 | 0.89 | 0.91 | 0.91 |
| reformulee (37) | 0.65 | 0.65 | 0.62 | 0.68 |
| date (4) | 0.25 | 0.00 | 0.00 | 0.00 |
| reference (4) | 0.75 | 1.00 | 0.75 | 0.75 |
| multi_sources (3) | 0.67 | 1.00 | 1.00 | 1.00 |
| cnil_pratique (5) | 1.00 | 0.80 | 0.80 | 1.00 |

Ce qu'on en retient :
- **Le chunking structurel gagne** : +13 points de hit@1 et +0,11 de MRR sur la référence
  récursive. Respecter les unités juridiques et ajouter un en-tête de contexte paie.
- **bge-m3 (BERT, local) retrouve plus** (hit@8, recall@8, cas pratiques) mais classe moins
  bien en tête que mistral-embed : les deux modèles sont complémentaires.
- **Trois faiblesses nettes** que la phase 4 doit corriger : les **cas pratiques** (~65 %),
  les **dates** (0 sur 4 : la recherche dense ramène les articles *sur* le haut risque, pas
  l'article 113 qui donne les *dates*), et le **« je ne sais pas »** : le score cosinus du
  1er résultat ne sépare pas les questions hors corpus (max 0,85 avec mistral-embed) de
  questions légitimes (dès 0,81). Il faudra un autre signal, celui du reranker.

## Retrieval avancé (phase 4)

```bash
cd backend
python -m app.retrieval.multiquery                                # reformulations du golden set (cache)
python -m app.evaluation.retrieval --validated --suite phase4     # chaque brique, une par une
python -m app.evaluation.retrieval --validated --suite final      # comparaison finale des rerankers
```

Pipeline retenu (`config.yaml`, `RetrievalSettings.from_config()`) :

```
question ─┬─ dense mistral-embed (pgvector HNSW) ─┐
          ├─ dense bge-m3        (pgvector HNSW) ─┼─ RRF (k = 60) ─> 20 candidats ─> reranker Cohere ─> 4 chunks
          └─ BM25 (racines françaises)          ──┘   50 par retriever                │
                                                                        score du 1er < 0,25 ─> « je ne sais pas »
```

| Brique | Ce qu'elle fait | Paramètres |
|---|---|---|
| **BM25** | Recherche par mots-clés, pondérée par la rareté des mots (IDF) et la longueur du chunk. Écrite en Python (index inversé en mémoire, 1 190 chunks) | racinisation Snowball française, mots-outils et élisions (« l' », « d' ») retirés, k1 = 1,2, b = 0,75 |
| **RRF** (Reciprocal Rank Fusion) | Fusionne des classements de natures différentes (cosinus, BM25) par leurs **rangs** : score = Σ 1/(k + rang). Un chunk bien classé par plusieurs retrievers passe devant | k = 60, 50 candidats par retriever |
| **Multi-query** | `ministral-8b` réécrit la question en 3 variantes (juridique, mots-clés, réponse hypothétique « HyDE ») ; chaque variante est cherchée puis tout est fusionné | JSON, température 0, cache SQLite. **Écarté** (voir ci-dessous) |
| **Reranker** (cross-encodeur) | Lit la question et le chunk **ensemble** (attention croisée) et note leur pertinence de 0 à 1. Bien plus précis qu'un cosinus, mais une passe du modèle par paire : appliqué aux 20 meilleurs candidats seulement | 20 candidats, score sigmoïde (local) ou `relevance_score` (Cohere), cache SQLite |
| **Renvois** | L'en-tête de chaque chunk liste le titre des chapitres, sections et annexes du même texte qu'il cite : « chapitre II » devient « chapitre II (Pratiques interdites en matière d'IA) ». Le texte de loi n'est pas modifié | 102 chunks sur 1 190 ; renvois vers d'autres actes ignorés |
| **Garde-fou** | « Je ne sais pas » si le score du meilleur chunk après reranking est sous un seuil | seuil 0,25 (Cohere v5 pro) |

**Pourquoi BM25 en Python plutôt que le plein texte de Postgres ?** `ts_rank` n'a pas d'IDF :
un mot présent dans tout le corpus (« règlement ») pèse autant qu'un mot rare (« biométrique »).
Sur une question type, le mode ET ne trouve aucun chunk ; le mode OU en remonte 1 165 sur 1 190,
presque à égalité. BM25 règle ce problème, et l'index de 1 190 chunks tient en mémoire.

### Les briques une par une

Mesures sur le golden set, **avant** l'ajout des Renvois (88 questions du corpus).
Détail : [hybride.json](backend/eval/results/hybride.json),
[multiquery.json](backend/eval/results/multiquery.json),
[rerank.json](backend/eval/results/rerank.json), [cohere.json](backend/eval/results/cohere.json).

| Configuration | hit@1 | hit@4 | hit@8 | MRR@20 | cas pratiques (hit@8) |
|---|---|---|---|---|---|
| dense mistral-embed (référence) | 0.52 | 0.67 | 0.73 | 0.608 | 0.59 |
| BM25 seul | 0.51 | 0.72 | 0.78 | 0.607 | 0.73 |
| mistral-embed + BM25 | 0.59 | 0.76 | 0.84 | 0.679 | 0.78 |
| mistral-embed + bge-m3 + BM25 (**hybride**) | 0.59 | 0.82 | 0.88 | 0.697 | 0.86 |
| hybride + multi-query | 0.49 | 0.78 | 0.90 | 0.638 | 0.95 |
| hybride + CamemBERT (reranker français, 110M) | 0.51 | 0.81 | 0.88 | 0.647 | 0.84 |
| hybride + bge-reranker-v2-m3 | 0.63 | 0.89 | 0.92 | 0.744 | 0.92 |
| hybride + Cohere v5 pro | 0.73 | 0.91 | 0.92 | 0.814 | 0.95 |
| hybride + multi-query + Cohere v5 pro | 0.72 | 0.90 | 0.92 | 0.807 | 0.97 |

- **BM25 seul fait déjà mieux que le dense** sur les cas pratiques : les questions juridiques
  contiennent des mots rares et décisifs (« biométrique », « notifiés ») qu'un cosinus dilue.
- **L'hybride à 3 retrievers** gagne +15 points de hit@8 et +0,09 de MRR sur la référence.
- **Le multi-query est écarté** : il retrouve un peu plus (hit@8), mais ses reformulations
  amènent des chunks voisins qui passent devant le bon (hit@1 de 0,59 à 0,49). Avec un reranker,
  il n'apporte plus rien et brouille le garde-fou. En prime, il coûte un appel LLM par question.
- **CamemBERT n'améliore rien** : il a été entraîné sur des questions web traduites (mMARCO),
  loin du vocabulaire juridique.

### Comparaison finale des rerankers

Index avec Renvois, 96 questions (88 du corpus, 8 hors corpus).
Détail : [final.json](backend/eval/results/final.json).

| Configuration | hit@1 | hit@4 | hit@8 | recall@8 | MRR@20 | dates (hit@8) |
|---|---|---|---|---|---|---|
| dense mistral-embed | 0.50 | 0.65 | 0.72 | 0.67 | 0.590 | 0 / 4 |
| hybride | 0.58 | 0.81 | 0.88 | 0.86 | 0.690 | 1 / 4 |
| hybride + bge-reranker-v2-m3 | 0.64 | 0.89 | 0.94 | 0.91 | 0.757 | 3 / 4 |
| hybride + Cohere v5 fast | 0.73 | **0.94** | **0.95** | **0.92** | 0.828 | 3 / 4 |
| **hybride + Cohere v5 pro** | **0.75** | 0.93 | 0.94 | 0.91 | **0.837** | 3 / 4 |

Latence mesurée pour une question et 20 chunks réels (médiane de 3 appels pour Cohere), sur le
portable i5-10210U sans GPU :

| Reranker | Qualité (MRR@20) | Latence par question | Garde-fou | Licence, coût |
|---|---|---|---|---|
| **Cohere rerank v5.0 pro** | **0.837** | **0,8 s** (API) | **séparation parfaite** | API payante à l'usage ; clé d'essai gratuite, non commerciale |
| Cohere rerank v5.0 fast | 0.828 | 0,8 s (API) | séparation parfaite | idem |
| bge-reranker-v2-m3 (568M) | 0.757 | 133 s (6,6 s / paire, CPU) | non | Apache 2.0, local |
| Qwen3-Reranker-0.6B | 0.64 \* | ~200 s (~10 s / paire, CPU) | non | Apache 2.0, local |
| CamemBERT mmarcoFR (110M) | 0.647 \*\* | 20 s (1 s / paire, CPU) | non | MIT, local |

\* Qwen3 : évaluation arrêtée après 3 h de calcul ; mesuré sur les 64 questions entièrement
notées (56 du corpus, [qwen3_partiel.json](backend/eval/results/qwen3_partiel.json)).
Sur ces mêmes questions : hybride seul 0,672, bge-v2-m3 0,764, Cohere fast 0,818, Cohere pro
0,823. Qwen3 **dégrade** le classement de l'hybride (cause non étudiée : conversion
« seq-cls » du modèle, ou instruction en anglais sur un corpus français).
\*\* mesuré avant l'ajout des Renvois.

**Choix : Cohere v5 pro.** Les deux versions Cohere sont proches (1 à 2 questions d'écart sur
88) et répondent aussi vite à ce volume. *Pro* classe mieux en tête (hit@1, MRR) et place
l'article 113 au 1er rang sur les 3 questions de date retrouvées, là où *fast* met une fois
devant un considérant aux dates périmées (voir ci-dessous). Seuls les modèles Cohere ont un
score qui permet un garde-fou fiable.

Les modèles locaux restent intéressants sur un serveur avec GPU ; sur un portable sans GPU, leur
latence (20 s à plus de 2 min par question) les rend inutilisables en conversation.

### Les questions de date et les Renvois

Les 4 questions de date (« depuis quand les pratiques interdites s'appliquent-elles ? ») ont
toutes pour réponse l'article 113, qui ne cite que des **numéros** : « les chapitres I et II
s'appliquent à partir du 2 février 2025 ». Les mots de la question (« pratiques interdites »)
n'y figurent pas ; ils sont dans le titre du chapitre II, ailleurs dans le texte. Aucun retriever
ne pouvait faire le lien.

Le piège : le **considérant 179** parle lui aussi des dates d'application, mais dans leur
version de 2024, jamais mise à jour par l'omnibus 2026/1744 (« le présent règlement devrait
s'appliquer à partir du 2 août 2026 »). Un RAG qui le cite donne une date périmée, avec une
source à l'appui.

Avec les Renvois, l'en-tête de l'article 113 contient « chapitre II (Pratiques interdites en
matière d'IA) ; chapitre V (Modèles d'IA à usage général) ; … ». Résultat avec un reranker :
**3 questions de date sur 4** retrouvées (contre 1 sur 4 avant), au 1er rang avec Cohere pro et
bge-v2-m3. Cohere fast classe l'article 113 au 2e rang pour les pratiques interdites, derrière
le considérant 179. La 4e (modèles d'IA à usage
général) échoue encore : l'article 113 n'entre pas dans les 20 candidats de l'hybride, le
reranker ne peut donc pas le remonter. Pour la génération (phase 5), le considérant 179 sera
signalé comme non mis à jour.

### Garde-fou « je ne sais pas »

Score du meilleur chunk après reranking :

| Reranker | Questions hors corpus (max) | Questions du corpus (min) | Séparation |
|---|---|---|---|
| dense (cosinus mistral-embed) | 0.851 | 0.812 | non : 3 hors corpus sur 8 passent |
| bge-reranker-v2-m3 | 0.412 | 0.002 | non : 1 question du corpus notée 0,002 |
| Cohere v5 fast | 0.129 | 0.440 | oui |
| **Cohere v5 pro** | **0.131** | **0.435** | **oui** |

Avec Cohere v5 pro, un seuil à **0,25** (milieu de l'écart) bloque les 8 questions hors
corpus et ne refuse aucune des 88 questions légitimes. Avec seulement 8 questions hors corpus,
ce seuil reste à surveiller : la phase 5 le vérifiera sur les réponses générées.

## Feuille de route

- [x] **Phase 0** : setup (backend, config, schéma Supabase, CI)
- [x] **Phase 1a** : ingestion AI Act + RGPD (versions consolidées, rectificatifs, contrôles qualité)
- [x] **Phase 1b** : fiches pratiques IA de la CNIL (17 fiches + liste de vérification PDF, fidélité caractère par caractère)
- [x] **Phase 2** : chunking (3 stratégies) + embeddings (`mistral-embed` vs `bge-m3`) + indexation Supabase (4 853 chunks)
- [x] **Phase 3** : golden set (96 questions vérifiées contre les sources, dont 8 hors corpus) + évaluation de référence
- [x] **Phase 4** : retrieval hybride (2 denses + BM25, RRF), reranker Cohere comparé à 3 rerankers ouverts, Renvois dans les en-têtes, garde-fou « je ne sais pas » calibré (multi-query mesuré puis écarté)
- [ ] **Phase 5** : génération Mistral avec citations vérifiées, mémoire conversationnelle, streaming
- [ ] **Phase 6** : agent LangGraph d'analyse de conformité, guidé par la liste de vérification CNIL (human-in-the-loop)
- [ ] **Phase 7** : frontend Next.js, upload PDF (Celery + Redis), auth
- [ ] **Phase 8** : déploiement, démo vidéo

## Sources

- [Règlement (UE) 2024/1689 (AI Act)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32024R1689), [version consolidée](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:02024R1689-20260727), modifié par le [règlement (UE) 2026/1744](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32026R1744) - © Union européenne, réutilisation autorisée (décision 2011/833/UE)
- [Règlement (UE) 2016/679 (RGPD)](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32016R0679), [version consolidée](https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:02016R0679-20160504)
- [CNIL - Fiches pratiques IA](https://www.cnil.fr/fr/les-fiches-pratiques-ia) - Source : CNIL – https://www.cnil.fr, licence [CC-BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/deed.fr) : reproduction à l'identique, avec titre, source, date d'extraction (voir [data/manifest.json](data/manifest.json)) et licence
