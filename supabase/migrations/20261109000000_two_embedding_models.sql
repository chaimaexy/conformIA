-- Phase 2 : deux modèles d'embedding comparés sur les mêmes chunks
--   embedding         : mistral-embed (API Mistral)
--   embedding_bge_m3  : BAAI/bge-m3 (encodeur BERT local), calculé pour la stratégie structurelle

alter table public.chunks add column embedding_bge_m3 extensions.vector(1024);

create index chunks_embedding_bge_m3_hnsw on public.chunks
    using hnsw (embedding_bge_m3 extensions.vector_cosine_ops);
create index chunks_strategy_idx on public.chunks (strategy);

-- Recherche dense paramétrée par le modèle.
-- hnsw.iterative_scan = strict_order (pgvector >= 0.8) : avec un filtre (stratégie), l'index
-- continue à parcourir le graphe tant qu'il n'a pas `match_count` résultats, dans l'ordre exact.
drop function if exists public.match_chunks_dense(extensions.vector, integer, text);

create or replace function public.match_chunks_dense(
    query_embedding extensions.vector(1024),
    match_count int default 8,
    filter_strategy text default 'structural',
    embedding_model text default 'mistral-embed'
)
returns table (id uuid, document_id uuid, content text, metadata jsonb, score float)
language plpgsql stable
set search_path = public, extensions
set hnsw.iterative_scan = strict_order
as $$
begin
    if embedding_model = 'mistral-embed' then
        return query
        select c.id, c.document_id, c.content, c.metadata,
               1 - (c.embedding <=> query_embedding) as score
        from public.chunks c
        where c.strategy = filter_strategy and c.embedding is not null
        order by c.embedding <=> query_embedding
        limit match_count;
    elsif embedding_model = 'local:BAAI/bge-m3' then
        return query
        select c.id, c.document_id, c.content, c.metadata,
               1 - (c.embedding_bge_m3 <=> query_embedding) as score
        from public.chunks c
        where c.strategy = filter_strategy and c.embedding_bge_m3 is not null
        order by c.embedding_bge_m3 <=> query_embedding
        limit match_count;
    else
        raise exception 'Modèle d''embedding inconnu : %', embedding_model;
    end if;
end;
$$;
