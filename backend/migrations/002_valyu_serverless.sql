-- Run after 001_initial_schema.sql. Preserves existing papers, PDFs and vectors.
begin;
alter table public.papers add column if not exists search_provider text;
alter table public.papers add column if not exists provider_paper_id text;
alter table public.papers add column if not exists ingestion_token uuid;
create unique index if not exists idx_papers_provider
    on public.papers(project_id, search_provider, provider_paper_id)
    where search_provider is not null and provider_paper_id is not null;

-- Reuse the original bibliographic deduplication without changing legacy IDs.
create or replace function public.save_research_paper(target_project_id uuid, metadata jsonb)
returns jsonb language plpgsql set search_path = public, extensions as $$
declare
    existing public.papers;
    result jsonb;
    provider text := nullif(metadata->>'search_provider', '');
    provider_id text := nullif(metadata->>'provider_paper_id', '');
begin
    perform pg_advisory_xact_lock(hashtextextended(target_project_id::text, 0));
    select * into existing from public.papers where project_id = target_project_id
        and search_provider = provider and provider_paper_id = provider_id limit 1;
    if found then
        return jsonb_build_object('status', 'already_exists', 'paper', to_jsonb(existing));
    end if;
    result := public.save_paper(target_project_id, metadata);
    if result->>'status' = 'saved' then
        update public.papers set search_provider = provider, provider_paper_id = provider_id
        where id = (result->'paper'->>'id')::uuid and project_id = target_project_id
        returning * into existing;
        result := jsonb_build_object('status', 'saved', 'paper', to_jsonb(existing));
    end if;
    return result;
end;
$$;

-- Atomic claims coordinate independent serverless instances.
create or replace function public.claim_ingestion(target_project_id uuid, target_paper_id uuid)
returns uuid language plpgsql set search_path = public, extensions as $$
declare token uuid := gen_random_uuid();
begin
    update public.papers set status = 'downloading', ingestion_token = token, ingestion_error = null
    where id = target_paper_id and project_id = target_project_id and status = 'saved';
    if not found then return null; end if;
    return token;
end;
$$;

create or replace function public.complete_claimed_ingestion(
    target_project_id uuid, target_paper_id uuid, claim_token uuid, chunks jsonb)
returns void language plpgsql set search_path = public, extensions as $$
begin
    perform 1 from public.papers where id = target_paper_id and project_id = target_project_id
        and ingestion_token = claim_token and status = 'storing' for update;
    if not found then raise exception 'Ingestion claim expired or paper was removed'; end if;
    perform public.complete_ingestion(target_project_id, target_paper_id, chunks);
    update public.papers set ingestion_token = null
        where id = target_paper_id and project_id = target_project_id;
end;
$$;

revoke all on function public.save_research_paper(uuid, jsonb) from public, anon, authenticated;
revoke all on function public.claim_ingestion(uuid, uuid) from public, anon, authenticated;
revoke all on function public.complete_claimed_ingestion(uuid, uuid, uuid, jsonb) from public, anon, authenticated;
grant execute on function public.save_research_paper(uuid, jsonb) to service_role;
grant execute on function public.claim_ingestion(uuid, uuid) to service_role;
grant execute on function public.complete_claimed_ingestion(uuid, uuid, uuid, jsonb) to service_role;
notify pgrst, 'reload schema';
commit;
