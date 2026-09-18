export interface Project {
  id: string; name: string; description: string; created_at: string; updated_at: string;
}

export interface Paper {
  id?: string; semantic_scholar_paper_id: string | null; title: string; authors: string[];
  search_provider?: string | null; provider_paper_id?: string | null; search_excerpt?: string | null;
  abstract: string | null; publication_year: number | null; publication_date?: string | null;
  venue: string | null; citation_count: number | null; doi: string | null; arxiv_id: string | null;
  source_url: string | null; pdf_url: string | null; ai_summary?: string | null;
  status?: string; storage_path?: string | null; ingestion_error?: string | null;
  embedding_model?: string | null; ingestion_detail?: string | null;
}

export interface Source {
  source_number: number; paper_id: string; paper_title: string; page_start: number; page_end: number; similarity: number | null;
}

export interface SearchResponse { papers: Paper[]; warnings: string[]; message?: string; search_performed?: boolean }
export interface AnswerResponse { answer: string; sources: Source[] }

export function paperKey(paper: Paper): string {
  return paper.provider_paper_id ? `${paper.search_provider}:${paper.provider_paper_id}` : paper.semantic_scholar_paper_id || paper.doi || paper.arxiv_id || paper.title;
}

export function samePaper(a: Paper, b: Paper): boolean {
  const doi = (s: string) => s.trim().toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, '');
  const arxiv = (s: string) => s.trim().toLowerCase().replace(/v\d+$/, '');
  const title = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  return Boolean((a.provider_paper_id && a.search_provider && a.search_provider === b.search_provider && a.provider_paper_id === b.provider_paper_id)
    || (a.semantic_scholar_paper_id && a.semantic_scholar_paper_id === b.semantic_scholar_paper_id)
    || (a.doi && b.doi && doi(a.doi) === doi(b.doi))
    || (a.arxiv_id && b.arxiv_id && arxiv(a.arxiv_id) === arxiv(b.arxiv_id))
    || title(a.title) === title(b.title));
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  let response: Response;
  try { response = await fetch(`/api${path}`, { ...options, headers, signal: options.signal ?? AbortSignal.timeout(250_000) }); }
  catch (error) {
    if (error instanceof DOMException && ['TimeoutError', 'AbortError'].includes(error.name)) throw new Error('Request timed out. Check the paper status before retrying.');
    throw new Error('Cannot reach the backend. Check that the server is running.');
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
  return data as T;
}

export function safeUrl(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  try { const parsed = new URL(url); return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : undefined; }
  catch { return undefined; }
}
