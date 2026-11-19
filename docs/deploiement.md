# Mise en ligne de ConformIA (phase 8)

Tout tient dans des offres gratuites. Render ne demande un moyen de paiement que pour dépasser
ses quotas gratuits ; sans moyen de paiement, il suspend le service jusqu'au mois suivant.

| Pièce | Où | Offre | Pourquoi |
|---|---|---|---|
| Interface Next.js | **Vercel** | Hobby (usage personnel, non commercial) | L'hébergeur de Next.js : rien à configurer, chaque `git push` peut redéployer |
| API FastAPI | **Render** | Free (512 Mo, 0,1 CPU, Francfort) | Accepte un Dockerfile ; l'API occupe 124 Mo une fois chargée (mesuré dans le conteneur) |
| Base, comptes, vecteurs | **Supabase** | le projet existant | Déjà en place depuis la phase 2 |
| Modèles | Mistral, Cohere | clés gratuites | Inchangés |

**Ce qui change par rapport à docker compose : ni Redis ni worker.** Sans `REDIS_URL`, l'API
exécute les analyses elle-même, une à la fois, dans son propre processus (la même règle que le
worker : 2 requêtes par minute sur l'offre gratuite de Mistral). Un seul service à héberger, et
l'état des analyses reste dans Postgres.

**Écartés :** Hugging Face Spaces (les Spaces Docker demandent maintenant un abonnement PRO),
Koyeb (racheté par Mistral AI, inscription gratuite fermée d'après plusieurs sources), Fly.io et
Railway (payants), Google Cloud Run (carte bancaire). Conditions relevées le 19/11/2026.

## Les limites de l'offre gratuite de Render, et ce que le code en fait

| Limite | Conséquence | Réponse |
|---|---|---|
| Mise en veille après 15 min sans visite, réveil en une minute environ (selon Render), plus les 35 s que met l'API à démarrer sur 0,1 CPU | Le premier visiteur attend une à deux minutes | L'interface appelle `/health` dès l'arrivée sur le site et affiche « Le serveur de démonstration se réveille » si la réponse tarde (`frontend/src/components/server-wake.tsx`) |
| Chargement des 1 193 passages et de l'index BM25 au démarrage : 29 s sur un portable, environ 3 min 20 s avec 0,1 CPU (conteneur limité comme chez Render) | La première question attendrait ce chargement | Préchargement en arrière-plan dès le démarrage ; `/health` répond tout de suite (`backend/app/main.py`). Une question arrivée pendant ce temps attend la fin du chargement au lieu de le refaire en parallèle |
| « Render peut redémarrer un service gratuit à tout moment » | Une analyse en cours serait perdue, et sa page attendrait un flux disparu | Au démarrage, les analyses en attente ou en cours passent en erreur : la page dit « L'analyse a échoué », on peut la relancer |
| Disque éphémère | Les caches SQLite (embeddings, réponses, reranking) repartent de zéro à chaque redémarrage | Ce ne sont que des caches : tout ce qui compte est dans Supabase |
| 750 heures par mois | Un service allumé tout le mois : 744 h au plus | Suffisant pour un seul service |

Les flux (chat au fil de l'eau, étapes de l'analyse) sont envoyés avec `X-Accel-Buffering: no`
et `Cache-Control: no-cache`, pour qu'aucun proxy ne les retienne avant de les transmettre.

## Avant de commencer

- Le code est poussé sur GitHub, sur la branche à déployer (`main` de préférence).
- Sous la main, **sans les coller dans un chat** :
  - la clé Mistral et la clé Cohere ;
  - dans Supabase (Project Settings > API Keys) : l'URL du projet, la clé **publishable**
    (`sb_publishable_…`, pour l'interface) et la clé **secrète** (`sb_secret_…`, pour l'API
    seulement) ;
  - la chaîne de connexion Postgres : bouton **Connect** du projet > **Session pooler** (port
    5432). Pas la connexion directe : elle est en IPv6 seulement.

## 1. L'API sur Render

1. Créer un compte sur [render.com](https://render.com) (connexion avec GitHub), puis autoriser
   l'accès au dépôt `conformIA`.
2. **New > Blueprint**, choisir le dépôt et la branche. Render lit `render.yaml` (à la racine) :
   un service web `conformia-api`, Docker, offre gratuite, Francfort, contrôle de santé sur
   `/health`.
3. Render demande les valeurs marquées `sync: false` :

   | Variable | Valeur |
   |---|---|
   | `CORS_ORIGINS` | pour l'instant `["http://localhost:3000"]` (on mettra l'adresse Vercel à l'étape 3) |
   | `MISTRAL_API_KEY` | ta clé Mistral |
   | `COHERE_API_KEY` | ta clé Cohere |
   | `SUPABASE_URL` | `https://<ref>.supabase.co` |
   | `SUPABASE_SECRET_KEY` | la clé secrète |
   | `DATABASE_URL` | la chaîne du Session pooler, avec le mot de passe de la base |

4. **Apply**. La première construction prend quelques minutes. Quand le service est « Live »,
   noter son adresse (`https://conformia-api-xxxx.onrender.com`) et ouvrir
   `<adresse>/health` : il doit répondre `"status": "ok"`, `"environment": "production"`,
   `"mistral_configured": true` et `"supabase_configured": true`.
5. Dans **Logs**, environ 3 minutes plus tard : `Préchargement terminé : 1193 passages`. Une
   question posée avant attend la fin de ce chargement.

`autoDeploy` est désactivé : on redéploie à la main (**Manual Deploy**) quand on le décide, pas à
chaque push.

## 2. L'interface sur Vercel

1. Créer un compte sur [vercel.com](https://vercel.com) (offre Hobby, connexion avec GitHub).
2. **Add New > Project**, importer `conformIA`.
3. **Root Directory : `frontend`**. Vercel reconnaît Next.js tout seul.
4. Variables d'environnement (toutes publiques : elles partent dans le navigateur) :

   | Variable | Valeur |
   |---|---|
   | `NEXT_PUBLIC_SUPABASE_URL` | `https://<ref>.supabase.co` |
   | `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` | la clé publishable (**jamais** la secrète) |
   | `NEXT_PUBLIC_API_URL` | l'adresse Render de l'étape 1, sans `/` final |

5. **Deploy**, puis noter l'adresse de production (`https://<projet>.vercel.app`).

Les variables `NEXT_PUBLIC_…` sont écrites dans le code au moment de la compilation : après en
avoir changé une, il faut redéployer.

## 3. Relier les trois

1. **Render** > le service > **Environment** : `CORS_ORIGINS` =
   `["https://<projet>.vercel.app","http://localhost:3000"]` (format JSON, sans `/` final), puis
   enregistrer : le service redémarre. Sans cela, le navigateur refuse les réponses de l'API.
   Les adresses de prévisualisation de Vercel (une par branche) ne sont pas autorisées : on
   essaie sur l'adresse de production.
2. **Supabase** > Authentication > **URL Configuration** :
   - **Site URL** : `https://<projet>.vercel.app` ;
   - **Redirect URLs** : ajouter `https://<projet>.vercel.app/**` et `http://localhost:3000/**`.

   Le lien de confirmation d'un nouveau compte ramène sur le site d'où l'on s'est inscrit
   (`emailRedirectTo`), à condition qu'il soit dans cette liste.
3. Les connexions anonymes (« Essayer sans compte ») sont déjà activées depuis la phase 7b.

## 4. Vérifier en ligne

Sur l'adresse Vercel, comme pour l'essai local :

- [ ] Après 15 minutes sans visite, l'arrivée sur le site affiche le bandeau de réveil, qui
      disparaît tout seul.
- [ ] « Essayer sans compte », puis une question : la réponse arrive au fil de l'eau, les renvois
      [n] ouvrent leur source.
- [ ] Une analyse du PDF `docs/exemples/cahier_des_charges_aide_logement.pdf` : étapes en
      direct, questions, rapport, téléchargement en Markdown.
- [ ] Recharger la page pendant l'analyse : le suivi reprend.
- [ ] « Mes analyses » et l'historique du chat ; effacer une analyse et une conversation.
- [ ] Sur téléphone (ou en vue mobile).
- [ ] Dans Supabase, Advisors > Security : pas de nouvelle alerte (celle des connexions anonymes
      est attendue).

## Ce que coûte une visite

Rien en argent, mais des quotas partagés par tous les visiteurs :

- **Mistral** (offre gratuite) : 2 requêtes par minute pour tout le compte. Une question en
  utilise 1 à 3 (génération, plus le routeur ou le contrôle de couverture quand ils servent),
  une analyse 4 à 9 (environ 8). Avec plusieurs visiteurs à la fois, les réponses
  ralentissent (les appels attendent leur tour) ; c'est la raison de la file d'analyses.
- **Cohere** (clé d'essai) : quota mensuel d'appels de reranking (1 à 2 « recherches »
  par question).
- **Par utilisateur** (code de l'API) : 3 analyses par 24 h et 20 questions par heure.
- **Supabase** : 30 sessions anonymes par heure et par adresse IP.
