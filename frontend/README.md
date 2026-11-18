# Frontend ConformIA (Next.js 16 + TypeScript + Tailwind)

```bash
cp .env.local.example .env.local   # URL Supabase + clé publishable (jamais la clé secrète)
npm install
npm run dev                         # http://localhost:3000 (l'API doit tourner sur le port 8000)
npm run lint && npm run build
```

| Page | Rôle |
|---|---|
| `/` | présentation, « Essayer sans compte » (session anonyme Supabase), créer un compte |
| `/connexion` | e-mail et mot de passe, ou session d'essai |
| `/chat` | questions sourcées : réponse au fil de l'eau, sources à droite (renvois [n] cliquables), conversations à gauche |
| `/analyses/nouvelle` | PDF du projet ou description libre |
| `/analyses/{id}` | file d'attente, étapes en direct, questions, rapport (téléchargeable en Markdown) |
| `/analyses` | mes analyses : statut, niveau de risque, effacement |

Choix :
- **Tout l'espace connecté est rendu dans le navigateur.** La session Supabase y vit
  (`localStorage`), et c'est l'API FastAPI qui vérifie le jeton à chaque requête : le serveur
  Next.js n'a rien à protéger. Avec Cache Components (Next.js 16), ces pages sont préparées à
  l'avance avec leur seul indicateur de chargement (`<Suspense>`), et marquées `instant = false`.
- **Les flux (SSE) sont lus avec `fetch`** (`src/lib/api.ts`) : `EventSource` ne sait pas
  envoyer l'en-tête `Authorization`.
- **Pas de couche Next.js entre l'interface et l'API** : une seule vérification des droits,
  celle de FastAPI.
- `@supabase/supabase-js` seul : `@supabase/ssr` ne sert que si le serveur Next.js lit la session.

Structure :

```
src/
  app/                 pages (le groupe (espace) demande une session)
  components/          auth-provider, espace-shell, analysis-view, ui
  lib/                 api.ts (jeton, erreurs, flux SSE), supabase.ts, types.ts (formats de l'API)
```
