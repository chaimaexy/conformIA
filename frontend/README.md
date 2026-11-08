# Frontend ConformIA (Next.js + TypeScript)

Le frontend sera généré en phase 7, une fois Node.js installé :

```bash
npx create-next-app@latest . --ts --tailwind --eslint --app --src-dir --import-alias "@/*"
npm install @supabase/supabase-js @supabase/ssr
```

Prévu :
- page de chat avec panneau latéral des sources (article, considérant, annexe)
- assistant « analyse de conformité » (étapes de l'agent LangGraph)
- authentification Supabase Auth
