export interface Project {
  id: string; name: string; description: string; created_at: string; updated_at: string;
}

export interface Paper {
  id?: string; semantic_scholar_paper_id: string | null; title: string; authors: string[];
  abstract: string | null; publication_year: number | null; publication_date?: string | null;
  venue: string | null; citation_count: number | null; doi: string | null; arxiv_id: string | null;
  source_url: string | null; pdf_url: string | null; ai_summary?: string | null;
  status?: string; storage_path?: string | null; ingestion_error?: string | null;
}

export interface Source {
  source_number: number; paper_id: string; paper_title: string; page_start: number; page_end: number; similarity: number | null;
}

export interface SearchResponse { papers: Paper[]; warnings: string[] }
export interface AnswerResponse { answer: string; sources: Source[] }

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  let response: Response;
  try { response = await fetch(`/api${path}`, { ...options, headers }); }
  catch { throw new Error('Cannot reach the backend. Check that the server is running.'); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status}).`);
  return data as T;
}

export function safeUrl(url: string | null | undefined): string | undefined {
  if (!url) return undefined;
  try { const parsed = new URL(url); return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : undefined; }
  catch { return undefined; }
}
