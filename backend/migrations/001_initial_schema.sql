-- Run once in the Supabase SQL editor. Credentials never belong in this file.
begin;
create extension if not exists vector with schema extensions;

create table public.projects (
    id uuid primary key default gen_random_uuid(),
    name text not null check (length(trim(name)) between 1 and 160),
    description text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table public.papers (
    id uuid primary key default gen_random_uuid(),
    project_id uuid not null references public.projects(id) on delete cascade,
    semantic_scholar_paper_id text,
    title text not null check (length(trim(title)) > 0),
    normalized_title text generated always as (lower(regexp_replace(title, '[^[:alnum:]]', '', 'g'))) stored,
    authors jsonb not null default '[]'::jsonb,
    abstract text,
    publication_year integer,
    publication_date date,
    venue text,
    citation_count integer,
    doi text,
    arxiv_id text,
    source_url text,
    pdf_url text,
    storage_path text,
    status text not null default 'saved' check (status in ('saved', 'no_pdf', 'downloading', 'extracting', 'embedding', 'storing', 'ready', 'failed')),
    ingestion_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (project_id, id)
);
create index idx_papers_project on public.papers(project_id);
create unique index idx_papers_title on public.papers(project_id, normalized_title);
create unique index idx_papers_scholar on public.papers(project_id, semantic_scholar_paper_id) where semantic_scholar_paper_id is not null;
create unique index idx_papers_doi on public.papers(project_id, lower(doi)) where doi is not null;
create unique index idx_papers_arxiv on public.papers(project_id, lower(arxiv_id)) where arxiv_id is not null;

create table public.document_chunks (
    id uuid primary key default gen_random_uuid(),
    project_id uuid not null references public.projects(id) on delete cascade,
    paper_id uuid not null,
    chunk_index integer not null check (chunk_index >= 0),
    page_start integer not null check (page_start > 0),
    page_end integer not null check (page_end >= page_start),
    content text not null check (length(content) > 0),
    embedding extensions.vector(768) not null,
    token_count integer not null check (token_count > 0),
    created_at timestamptz not null default now(),
    foreign key (project_id, paper_id) references public.papers(project_id, id) on delete cascade,
    unique (paper_id, chunk_index)
);
create index idx_chunks_project on public.document_chunks(project_id);
create index idx_chunks_paper on public.document_chunks(paper_id);

create function public.touch_updated_at() returns trigger language plpgsql set search_path = '' as $$
begin
    new.updated_at := now();
    return new;
end;
$$;
create trigger projects_updated before update on public.projects for each row execute function public.touch_updated_at();
create trigger papers_updated before update on public.papers for each row execute function public.touch_updated_at();

-- Serializes concurrent saves within a project and deduplicates alternate identifiers.
create function public.save_paper(target_project_id uuid, metadata jsonb)
returns jsonb language plpgsql set search_path = public, extensions as $$
declare
    existing public.papers;
    saved public.papers;
    normalized text := lower(regexp_replace(metadata->>'title', '[^[:alnum:]]', '', 'g'));
    paper_doi text := nullif(lower(regexp_replace(trim(metadata->>'doi'), '^https?://(dx\.)?doi\.org/', '', 'i')), '');
    paper_arxiv text := nullif(lower(regexp_replace(trim(metadata->>'arxiv_id'), 'v[0-9]+$', '')), '');
begin
    perform pg_advisory_xact_lock(hashtextextended(target_project_id::text, 0));
    perform 1 from public.projects where id = target_project_id for update;
    if not found then raise exception 'Project not found'; end if;
    select * into existing from public.papers where project_id = target_project_id and (
        semantic_scholar_paper_id = metadata->>'semantic_scholar_paper_id' or
        lower(doi) = paper_doi or lower(arxiv_id) = paper_arxiv or normalized_title = normalized
    ) limit 1;
    if found then return jsonb_build_object('status', 'already_exists', 'paper', to_jsonb(existing)); end if;
    insert into public.papers (
        project_id, semantic_scholar_paper_id, title, authors, abstract, publication_year,
        publication_date, venue, citation_count, doi, arxiv_id, source_url, pdf_url, status
    ) values (
        target_project_id, nullif(metadata->>'semantic_scholar_paper_id', ''), metadata->>'title',
        coalesce(metadata->'authors', '[]'::jsonb), metadata->>'abstract', (metadata->>'publication_year')::integer,
        nullif(metadata->>'publication_date', '')::date, metadata->>'venue', (metadata->>'citation_count')::integer,
        paper_doi, paper_arxiv, metadata->>'source_url', metadata->>'pdf_url',
        case when nullif(metadata->>'pdf_url', '') is null then 'no_pdf' else 'saved' end
    ) returning * into saved;
    return jsonb_build_object('status', 'saved', 'paper', to_jsonb(saved));
end;
$$;

-- A failed request rolls back every chunk; ready can never mean partially indexed.
create function public.complete_ingestion(target_project_id uuid, target_paper_id uuid, chunks jsonb)
returns void language plpgsql set search_path = public, extensions as $$
begin
    perform 1 from public.papers where id = target_paper_id and project_id = target_project_id for update;
    if not found then raise exception 'Paper not found in project'; end if;
    if jsonb_array_length(chunks) < 1 or jsonb_array_length(chunks) > 3000 then
        raise exception 'Invalid chunk count';
    end if;
    delete from public.document_chunks where paper_id = target_paper_id and project_id = target_project_id;
    insert into public.document_chunks(project_id, paper_id, chunk_index, page_start, page_end, content, embedding, token_count)
    select target_project_id, target_paper_id, (c->>'chunk_index')::integer,
        (c->>'page_start')::integer, (c->>'page_end')::integer, c->>'content',
        (c->>'embedding')::extensions.vector(768), (c->>'token_count')::integer
    from jsonb_array_elements(chunks) c;
    update public.papers set status = 'ready', ingestion_error = null where id = target_paper_id and project_id = target_project_id;
end;
$$;

create function public.match_document_chunks(
    query_embedding extensions.vector(768), match_project_id uuid,
    match_count integer default 6, filter_paper_ids uuid[] default null,
    min_similarity double precision default 0.35
) returns table (
    id uuid, paper_id uuid, paper_title text, content text,
    page_start integer, page_end integer, similarity double precision
) language sql stable set search_path = public, extensions as $$
    select dc.id, dc.paper_id, p.title, dc.content, dc.page_start, dc.page_end,
           1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    join public.papers p on p.id = dc.paper_id and p.project_id = dc.project_id
    where dc.project_id = match_project_id and p.status = 'ready'
      and (filter_paper_ids is null or dc.paper_id = any(filter_paper_ids))
      and 1 - (dc.embedding <=> query_embedding) >= min_similarity
    order by dc.embedding <=> query_embedding
    limit greatest(1, least(match_count, 20));
$$;

-- V1 is a private single-user application. All data access goes through FastAPI.
alter table public.projects enable row level security;
alter table public.papers enable row level security;
alter table public.document_chunks enable row level security;
revoke all on public.projects, public.papers, public.document_chunks from anon, authenticated;
grant all on public.projects, public.papers, public.document_chunks to service_role;
revoke all on function public.save_paper(uuid, jsonb) from public, anon, authenticated;
revoke all on function public.complete_ingestion(uuid, uuid, jsonb) from public, anon, authenticated;
revoke all on function public.match_document_chunks(extensions.vector, uuid, integer, uuid[], double precision) from public, anon, authenticated;
grant execute on function public.save_paper(uuid, jsonb) to service_role;
grant execute on function public.complete_ingestion(uuid, uuid, jsonb) to service_role;
grant execute on function public.match_document_chunks(extensions.vector, uuid, integer, uuid[], double precision) to service_role;

insert into storage.buckets(id, name, public, file_size_limit, allowed_mime_types)
values ('research-papers', 'research-papers', false, 31457280, array['application/pdf'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;
commit;
