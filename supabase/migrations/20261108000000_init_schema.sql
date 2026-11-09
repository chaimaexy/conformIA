-- ConformIA : schéma initial
-- documents  = une source (AI Act, RGPD, fiche CNIL, PDF utilisateur)
-- chunks     = morceaux indexés (embedding Mistral 1024 dims + plein texte français)

create extension if not exists vector with schema extensions;

create table public.documents (
    id           uuid primary key default gen_random_uuid(),
    source_id    text not null,                 -- ex. 'ai_act', 'rgpd', 'cnil_fiche_3'
    title        text not null,
    url          text,
    celex        text,
    language     text not null default 'fr',
    version_date date,                          -- date de la version consolidée ingérée
    owner_id     uuid references auth.users (id) on delete cascade, -- null = corpus public
    created_at   timestamptz not null default now(),
    unique (source_id, version_date)
);

create table public.chunks (
    id           uuid primary key default gen_random_uuid(),
    document_id  uuid not null references public.documents (id) on delete cascade,
    chunk_index  int not null,
    content      text not null,
    -- métadonnées juridiques : section_type (article|considerant|annexe), number, chapter, title...
    metadata     jsonb not null default '{}'::jsonb,
    strategy     text not null default 'structural',
    embedding    extensions.vector(1024),
    fts          tsvector generated always as (to_tsvector('french', content)) stored,
    created_at   timestamptz not null default now(),
    unique (document_id, strategy, chunk_index)
);

create index documents_owner_id_idx on public.documents (owner_id);

create index chunks_embedding_hnsw on public.chunks
    using hnsw (embedding extensions.vector_cosine_ops);
create index chunks_fts_gin on public.chunks using gin (fts);
create index chunks_metadata_gin on public.chunks using gin (metadata);

-- Recherche dense (similarité cosinus)
create or replace function public.match_chunks_dense(
    query_embedding extensions.vector(1024),
    match_count int default 8,
    filter_strategy text default 'structural'
)
returns table (id uuid, document_id uuid, content text, metadata jsonb, score float)
language sql stable
set search_path = public, extensions
as $$
    select c.id, c.document_id, c.content, c.metadata,
           1 - (c.embedding <=> query_embedding) as score
    from public.chunks c
    where c.strategy = filter_strategy
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- Recherche plein texte (sparse) en français
create or replace function public.match_chunks_fts(
    query_text text,
    match_count int default 8,
    filter_strategy text default 'structural'
)
returns table (id uuid, document_id uuid, content text, metadata jsonb, score float)
language sql stable
set search_path = public
as $$
    select c.id, c.document_id, c.content, c.metadata,
           ts_rank_cd(c.fts, websearch_to_tsquery('french', query_text))::float as score
    from public.chunks c
    where c.strategy = filter_strategy
      and c.fts @@ websearch_to_tsquery('french', query_text)
    order by score desc
    limit match_count;
$$;

-- Row Level Security : corpus public lisible par tous, documents privés réservés à leur auteur
alter table public.documents enable row level security;
alter table public.chunks enable row level security;

create policy "documents lisibles" on public.documents
    for select to anon, authenticated
    using (owner_id is null or owner_id = (select auth.uid()));

-- (select auth.uid()) est évalué une seule fois par requête, pas par ligne
create policy "chunks lisibles" on public.chunks
    for select to anon, authenticated
    using (
        exists (
            select 1 from public.documents d
            where d.id = chunks.document_id
              and (d.owner_id is null or d.owner_id = (select auth.uid()))
        )
    );
-- Les écritures passent par le backend avec la clé secrète (qui contourne RLS).
