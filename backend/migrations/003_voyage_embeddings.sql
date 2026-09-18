-- Run after 002. Stop the old backend before migrating, then start the new code.
-- Additive migration: preserves papers, PDFs and all legacy BGE chunks/vectors.
begin;
alter table public.papers add column if not exists embedding_model text;
alter table public.papers add column if not exists ingestion_detail text;
update public.papers p set embedding_model = 'BAAI/bge-base-en-v1.5'
where embedding_model is null and exists(select 1 from public.document_chunks c where c.paper_id = p.id);

create table if not exists public.document_chunks_voyage (
    id uuid primary key default gen_random_uuid(),
    project_id uuid not null references public.projects(id) on delete cascade,
    paper_id uuid not null,
    chunk_index integer not null check(chunk_index >= 0),
    page_start integer not null check(page_start > 0),
    page_end integer not null check(page_end >= page_start),
    content text not null check(length(content) > 0),
    embedding extensions.vector(1024) not null,
    token_count integer not null check(token_count > 0),
    created_at timestamptz not null default now(),
    foreign key(project_id, paper_id) references public.papers(project_id, id) on delete cascade,
    unique(paper_id, chunk_index)
);
create index if not exists idx_voyage_chunks_project on public.document_chunks_voyage(project_id);
create index if not exists idx_voyage_chunks_paper on public.document_chunks_voyage(paper_id);
alter table public.document_chunks_voyage enable row level security;
revoke all on public.document_chunks_voyage from public, anon, authenticated;
grant all on public.document_chunks_voyage to service_role;

create or replace function public.complete_voyage_ingestion(
    target_project_id uuid, target_paper_id uuid, claim_token uuid, chunks jsonb)
returns void language plpgsql set search_path = public, extensions as $$
begin
    perform 1 from public.papers where id = target_paper_id and project_id = target_project_id
        and ingestion_token = claim_token and status = 'storing' for update;
    if not found then raise exception 'Ingestion claim expired or paper was removed'; end if;
    if jsonb_typeof(chunks) is distinct from 'array' or jsonb_array_length(chunks) not between 1 and 3000 then
        raise exception 'Invalid chunk count';
    end if;
    delete from public.document_chunks_voyage where paper_id = target_paper_id and project_id = target_project_id;
    insert into public.document_chunks_voyage(project_id, paper_id, chunk_index, page_start, page_end, content, embedding, token_count)
    select target_project_id, target_paper_id, (c->>'chunk_index')::integer,
        (c->>'page_start')::integer, (c->>'page_end')::integer, c->>'content',
        (c->>'embedding')::extensions.vector(1024), (c->>'token_count')::integer
    from jsonb_array_elements(chunks) c;
    update public.papers set status = 'ready', ingestion_error = null, ingestion_token = null,
        embedding_model = 'voyage-4-lite', ingestion_detail = 'Indexed with Voyage'
    where id = target_paper_id and project_id = target_project_id;
end;
$$;

create or replace function public.match_voyage_chunks(
    query_embedding extensions.vector(1024), match_project_id uuid,
    match_count integer default 6, filter_paper_ids uuid[] default null,
    min_similarity double precision default 0.35
) returns table(id uuid, paper_id uuid, paper_title text, content text,
    page_start integer, page_end integer, similarity double precision)
language sql stable set search_path = public, extensions as $$
    select dc.id, dc.paper_id, p.title, dc.content, dc.page_start, dc.page_end,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks_voyage dc
    join public.papers p on p.id = dc.paper_id and p.project_id = dc.project_id
    where dc.project_id = match_project_id and p.status = 'ready' and p.embedding_model = 'voyage-4-lite'
        and (filter_paper_ids is null or dc.paper_id = any(filter_paper_ids))
        and 1 - (dc.embedding <=> query_embedding) >= min_similarity
    order by dc.embedding <=> query_embedding
    limit greatest(1, least(match_count, 20));
$$;
revoke all on function public.complete_voyage_ingestion(uuid, uuid, uuid, jsonb) from public, anon, authenticated;
revoke all on function public.match_voyage_chunks(extensions.vector, uuid, integer, uuid[], double precision) from public, anon, authenticated;
grant execute on function public.complete_voyage_ingestion(uuid, uuid, uuid, jsonb) to service_role;
grant execute on function public.match_voyage_chunks(extensions.vector, uuid, integer, uuid[], double precision) to service_role;
notify pgrst, 'reload schema';
commit;
