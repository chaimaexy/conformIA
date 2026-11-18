-- ConformIA, phase 7b : analyses et conversations rattachées à leur utilisateur (Supabase Auth).
--
-- Le backend écrit par une connexion Postgres directe (DATABASE_URL), qui n'est pas soumise à
-- RLS : il vérifie lui-même le jeton de l'utilisateur et le propriétaire de chaque ligne. Les
-- politiques RLS ci-dessous protègent l'accès par l'API publique de Supabase : chacun ne lit
-- que ses lignes, personne n'y écrit.

-- État de LangGraph (points de reprise des analyses) : un schéma à part, que l'API publique de
-- Supabase n'expose pas. Les tables y sont créées par le backend (PostgresSaver.setup()).
create schema if not exists langgraph;
revoke all on schema langgraph from public, anon, authenticated;
grant usage, create on schema langgraph to postgres;

create table public.analyses (
    id          text primary key,                -- identifiant de l'analyse (fil LangGraph)
    owner_id    uuid not null references auth.users (id) on delete cascade,
    titre       text not null,                   -- nom du PDF ou début de la description
    source      text not null check (source in ('description', 'document')),
    statut      text not null default 'en_attente'
                check (statut in ('en_attente', 'en_cours', 'questions', 'termine', 'erreur')),
    niveau      text,                            -- niveau de risque, une fois le rapport fait
    rapport     jsonb,                           -- copie du rapport : lisible sans LangGraph
    markdown    text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index analyses_owner_created_idx on public.analyses (owner_id, created_at desc);

create table public.conversations (
    id          text primary key,
    owner_id    uuid not null references auth.users (id) on delete cascade,
    titre       text not null,                   -- début de la première question
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index conversations_owner_updated_idx on public.conversations (owner_id, updated_at desc);

create table public.messages (
    id              bigint generated always as identity primary key,
    conversation_id text not null references public.conversations (id) on delete cascade,
    role            text not null check (role in ('user', 'assistant')),
    content         text not null,
    sources         jsonb,                       -- sources de la réponse (sans leur texte)
    created_at      timestamptz not null default now()
);
create index messages_conversation_idx on public.messages (conversation_id, id);

alter table public.analyses enable row level security;
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- (select auth.uid()) est évalué une seule fois par requête, pas par ligne
create policy "analyses de leur auteur" on public.analyses
    for select to authenticated
    using (owner_id = (select auth.uid()));

create policy "conversations de leur auteur" on public.conversations
    for select to authenticated
    using (owner_id = (select auth.uid()));

create policy "messages de leur auteur" on public.messages
    for select to authenticated
    using (
        exists (
            select 1 from public.conversations c
            where c.id = messages.conversation_id
              and c.owner_id = (select auth.uid())
        )
    );

-- Aucune écriture par l'API publique (sans politique, RLS les refuse déjà : double protection)
revoke insert, update, delete, truncate on public.analyses, public.conversations, public.messages
    from anon, authenticated;
revoke all on public.analyses, public.conversations, public.messages from anon;
