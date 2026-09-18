import { useEffect, useRef, useState } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ArrowLeft, ArrowUp, BookOpen, Check, ChevronRight, FileText, FolderOpen, LoaderCircle, Plus, Search, Settings2, Sparkles, Trash2, Upload, X } from 'lucide-react';
import { api, safeUrl, paperKey, samePaper, type AnswerResponse, type Paper, type Project, type SearchResponse, type Source } from './api';

type Message = { id: number; role: 'user' | 'assistant'; text: string; papers?: Paper[]; sources?: Source[] };
const activeStatuses = new Set(['saved', 'downloading', 'extracting', 'embedding', 'storing']);
const statusNames: Record<string, string> = {
  saved: 'Queued', downloading: 'Downloading PDF', extracting: 'Extracting text', embedding: 'Generating embeddings',
  storing: 'Saving vectors', ready: 'Ready', failed: 'Needs attention', no_pdf: 'PDF needed',
};

function errorText(error: unknown) { return error instanceof Error ? error.message : 'Something went wrong. Please retry.'; }

export default function App() {
  const [loading, setLoading] = useState(true);
  const [projects, setProjects] = useState<Project[]>([]);
  const [current, setCurrent] = useState<Project | null>(null);
  const [error, setError] = useState('');
  const [editor, setEditor] = useState<'new' | 'edit' | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);

  async function refreshProjects() { setProjects(await api<Project[]>('/projects')); }
  useEffect(() => {
    let active = true;
    api<Project[]>('/projects').then(data => { if (active) setProjects(data); })
      .catch(error => { if (active) setError(errorText(error)); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);
  function edit(mode: 'new' | 'edit') {
    setName(mode === 'edit' ? current!.name : '');
    setDescription(mode === 'edit' ? current!.description : ''); setEditor(mode); setError('');
  }
  async function saveProject(event: React.FormEvent) {
    event.preventDefault(); setSaving(true); setError('');
    try {
      const project = await api<Project>(editor === 'edit' ? `/projects/${current!.id}` : '/projects', {
        method: editor === 'edit' ? 'PATCH' : 'POST', body: JSON.stringify({ name: name.trim(), description }),
      });
      await refreshProjects(); setCurrent(project); setEditor(null);
    } catch (error) { setError(errorText(error)); }
    finally { setSaving(false); }
  }
  async function deleteProject() {
    if (!current || !window.confirm(`Delete “${current.name}” and all its papers and PDFs? This cannot be undone.`)) return;
    setSaving(true); setError('');
    try { await api(`/projects/${current.id}`, { method: 'DELETE' }); setCurrent(null); setEditor(null); await refreshProjects(); }
    catch (error) { setError(errorText(error)); }
    finally { setSaving(false); }
  }

  return <div className="app-shell">
    <header className="topbar"><button className="brand" onClick={() => { setCurrent(null); void refreshProjects().catch(e => setError(errorText(e))); }}><BookOpen size={22} />research<span>ai</span></button>
      <div className="breadcrumb">Workspace {current && <><ChevronRight size={14} /><span>{current.name}</span></>}</div>
      <span className="local-label">Your research workspace</span>
    </header>
    {error && !editor && <div className="global-error error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}
    {current ? <Workspace key={current.id} project={current} onBack={() => { setCurrent(null); void refreshProjects().catch(e => setError(errorText(e))); }} onEdit={() => edit('edit')} /> :
      <main className="dashboard"><div className="dashboard-heading"><div><div className="eyebrow">YOUR RESEARCH, CONNECTED</div><h1>A home for your next idea.</h1><p>Each project is a focused library. Your questions stay grounded in the papers you choose.</p></div>
        <button className="primary" onClick={() => edit('new')}><Plus size={17} /> New project</button></div>
        <div className="section-label">RESEARCH PROJECTS <span>{projects.length.toString().padStart(2, '0')}</span>{loading && <LoaderCircle size={14} className="spin" />}</div>
        <div className="project-grid">{projects.map(project => <button className="project-card" key={project.id} onClick={() => setCurrent(project)}><div className="project-icon"><FolderOpen size={24} /></div><h2>{project.name}</h2><p>{project.description || 'A space to explore, collect papers, and ask better questions.'}</p><footer><span>Updated {new Date(project.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span><ChevronRight size={19} /></footer></button>)}
          <button className="project-card new-project" onClick={() => edit('new')}><Plus size={28} /><h2>Start a new project</h2><p>Follow a question worth exploring.</p></button></div>
        <div className="dashboard-note"><Sparkles size={18} /><p><strong>You decide what becomes knowledge.</strong> Search results only join your library when you save them.</p></div>
      </main>}
    {editor && <div className="modal-backdrop"><section className="modal" role="dialog" aria-modal="true" aria-labelledby="project-editor-title"><button className="modal-close quiet" aria-label="Close" onClick={() => setEditor(null)} disabled={saving}><X size={20} /></button>
      <h2 id="project-editor-title">{editor === 'new' ? 'Start a research project' : 'Project settings'}</h2><p>Give your research a clear focus.</p>
      <form onSubmit={saveProject}><label htmlFor="project-name">Project name</label><input autoFocus id="project-name" value={name} onChange={e => setName(e.target.value)} maxLength={160} required />
        <label htmlFor="project-description">Description <span className="muted">(optional)</span></label><textarea id="project-description" value={description} onChange={e => setDescription(e.target.value)} maxLength={5000} rows={4} />
        {error && <div className="error" role="alert">{error}</div>}
        <div className="modal-actions">{editor === 'edit' && <button type="button" className="danger quiet" disabled={saving} onClick={() => void deleteProject()}><Trash2 size={15} />Delete project</button>}<button className="primary" disabled={saving || !name.trim()}>{saving ? 'Saving…' : 'Save project'}</button></div>
      </form></section></div>}
  </div>;
}

function Workspace({ project, onBack, onEdit }: { project: Project; onBack: () => void; onEdit: () => void }) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'chat' | 'search'>('chat');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [pending, setPending] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);
  const [uploadLimit, setUploadLimit] = useState<number | null>(null);
  const [indexModel, setIndexModel] = useState('');
  const uploadRef = useRef<HTMLInputElement>(null);
  const attachTarget = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const alive = useRef(true);
  const base = `/projects/${project.id}`;
  const isReady = (p: Paper) => p.status === 'ready' && p.embedding_model === indexModel;
  const readyCount = papers.filter(isReady).length;

  async function refresh() { const data = await api<Paper[]>(`${base}/papers`); if (alive.current) setPapers(data); }
  useEffect(() => {
    alive.current = true;
    api<{ max_upload_bytes: number; embedding_model: string }>('/status').then(s => { if (alive.current) { setUploadLimit(s.max_upload_bytes); setIndexModel(s.embedding_model); } }).catch(e => { if (alive.current) setError(errorText(e)); });
    void refresh().catch(e => setError(errorText(e)));
    const interval = window.setInterval(() => { void refresh().catch(e => { if (alive.current) setError(errorText(e)); }); }, 3000);
    return () => { alive.current = false; window.clearInterval(interval); };
    // A workspace is remounted when its project changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, busy]);

  async function submit(event: React.FormEvent) {
    event.preventDefault(); if (!input.trim() || busy) return;
    const question = input.trim(); setInput(''); setBusy(true); setError('');
    setMessages(prev => [...prev, { id: Date.now(), role: 'user', text: question }]);
    try {
      if (mode === 'search') {
        const data = await api<SearchResponse>(`${base}/papers/search`, { method: 'POST', body: JSON.stringify({ query: question, limit: 8 }) });
        if (alive.current) setMessages(prev => [...prev, { id: Date.now(), role: 'assistant', text: data.papers.length ? `Found ${data.papers.length} papers to explore. Save the ones you want in your library.${data.warnings.length ? '\n' + data.warnings.join(' ') : ''}` : 'No papers found. Try a broader topic or different keywords.', papers: data.papers }]);
      } else {
        const data = await api<AnswerResponse>(`${base}/chat`, { method: 'POST', body: JSON.stringify({ message: question, paper_ids: selected.size ? [...selected] : null }) });
        if (alive.current) setMessages(prev => [...prev, { id: Date.now(), role: 'assistant', text: data.answer, sources: data.sources }]);
      }
    } catch (error) { if (alive.current) { setError(errorText(error)); setInput(question); } }
    finally { if (alive.current) setBusy(false); }
  }
  async function savePaper(paper: Paper) {
    const key = paperKey(paper);
    setPending(prev => new Set(prev).add(key)); setError(''); setNotice('');
    try {
      const data = await api<{ status: string; paper: Paper; processing_required?: boolean }>(`${base}/papers`, { method: 'POST', body: JSON.stringify(paper) });
      if (data.processing_required && data.paper.id) {
        // Keep the processing request alive without blocking save buttons or chat.
        void api(`${base}/papers/${data.paper.id}/process`, { method: 'POST' })
          .then(() => refresh()).catch(e => { if (alive.current) setError(errorText(e)); });
      }
      await refresh();
      setNotice(data.status === 'already_exists' ? 'This paper is already in your library.' : data.paper.status === 'no_pdf' ? 'Paper saved. Upload its PDF from the library to make it searchable.' : data.paper.status === 'ready' ? 'Paper saved and ready for questions.' : data.paper.status === 'failed' ? 'Paper saved, but processing needs attention. Check its library status.' : 'Paper saved. PDF ingestion is starting.');
    } catch (error) { setError(errorText(error)); }
    finally { setPending(prev => { const next = new Set(prev); next.delete(key); return next; }); }
  }
  async function uploadFile(file: File | undefined) {
    if (!file) return;
    if (!uploadLimit) { setError('Upload settings are not available. Refresh and check the backend.'); return; }
    if (file.size > uploadLimit || !file.name.toLowerCase().endsWith('.pdf')) { setError(`Choose a PDF no larger than ${(uploadLimit / 1_000_000).toFixed(1)} MB for this deployment.`); return; }
    setUploading(true); setError(''); setNotice('');
    const data = new FormData(); data.append('file', file);
    const target = attachTarget.current;
    try {
      const response = await api<{ status: string }>(target ? `${base}/papers/${target}/upload` : `${base}/papers/upload`, { method: 'POST', body: data });
      await refresh(); setNotice(response.status === 'already_exists' ? 'This paper already exists. Use its Attach PDF action if it needs a PDF.' : 'Upload request finished. Check the paper’s status in the library.');
    } catch (error) { setError(errorText(error)); }
    finally { setUploading(false); attachTarget.current = null; if (uploadRef.current) uploadRef.current.value = ''; }
  }
  async function removePaper(paper: Paper) {
    if (!window.confirm(`Delete “${paper.title}”, its PDF and searchable text?`)) return;
    try { await api(`${base}/papers/${paper.id}`, { method: 'DELETE' }); setSelected(prev => { const n = new Set(prev); n.delete(paper.id!); return n; }); await refresh(); }
    catch (error) { setError(errorText(error)); }
  }
  async function openPdf(paperId: string, page?: number) {
    const tab = window.open('about:blank', '_blank');
    if (tab) tab.opener = null;
    try { const data = await api<{ url: string }>(`${base}/papers/${paperId}/pdf`); const url = safeUrl(data.url); if (url && tab) tab.location.href = url + (page ? `#page=${page}` : ''); else { tab?.close(); setError('Allow pop-ups to view this PDF.'); } }
    catch (error) { tab?.close(); setError(errorText(error)); }
  }
  async function retry(paper: Paper, action = 'retry') {
    try { await api(`${base}/papers/${paper.id}/${action}`, { method: 'POST' }); await refresh(); }
    catch (error) { setError(errorText(error)); }
  }

  return <div className="workspace">
    <aside className="library"><button className="back quiet" onClick={onBack}><ArrowLeft size={15} /> All projects</button>
      <div className="project-title"><h1>{project.name}</h1><button className="quiet" aria-label="Project settings" onClick={onEdit}><Settings2 size={17} /></button></div>
      {project.description && <p className="project-description">{project.description}</p>}
      <div className="library-heading"><span>YOUR LIBRARY</span><span className="count">{papers.length}</span></div>
      <button className="upload-button" disabled={uploading || !uploadLimit} onClick={() => { attachTarget.current = null; uploadRef.current?.click(); }}><Upload size={16} /> {uploading ? 'Uploading / processing…' : 'Upload a PDF'}<span>+</span></button>
      <input ref={uploadRef} type="file" accept=".pdf,application/pdf" className="hidden" aria-label="Upload research paper PDF" onChange={e => void uploadFile(e.target.files?.[0])} />
      <div className="paper-list">{papers.length === 0 ? <div className="library-empty"><FileText size={28} /><p>Your next discovery<br />belongs here.</p><small>Save a search result or upload a PDF.</small></div> : papers.map(paper => <article className="library-paper" key={paper.id}>
        <div className="paper-title-row"><FileText size={16} /><h3>{paper.title}</h3>{isReady(paper) && <input type="checkbox" aria-label={`Focus chat on ${paper.title}`} checked={selected.has(paper.id!)} onChange={e => setSelected(prev => { const next = new Set(prev); if (e.target.checked) next.add(paper.id!); else next.delete(paper.id!); return next; })} />}</div>
        <div className={`status status-${paper.status}`}>{activeStatuses.has(paper.status || '') ? <LoaderCircle className="spin" size={11} /> : paper.status === 'ready' ? <Check size={11} /> : null}{statusNames[paper.status || ''] || paper.status}</div>
        {paper.ingestion_error && <p className="paper-error">{paper.ingestion_error}</p>}
        {paper.ingestion_detail && <p className="muted">{paper.ingestion_detail}</p>}
        {paper.status === 'ready' && indexModel && !isReady(paper) && <button onClick={() => void retry(paper, 'reindex')}>Reindex for Zilliz</button>}
        {paper.status === 'saved' && <button onClick={() => void retry(paper, 'process')}>Start queued processing</button>}
        {paper.status === 'failed' && paper.storage_path && !paper.pdf_url && <button onClick={() => void retry(paper)}>Retry stored PDF</button>}
        <div className="paper-actions">{paper.storage_path && <button onClick={() => void openPdf(paper.id!)}>View PDF</button>}{['failed', 'no_pdf'].includes(paper.status || '') && <button disabled={uploading} onClick={() => { attachTarget.current = paper.id!; uploadRef.current?.click(); }}>Attach PDF</button>}{paper.status === 'failed' && paper.pdf_url && <button onClick={() => void retry(paper)}>Retry</button>}<button className="delete-paper" disabled={activeStatuses.has(paper.status || '')} onClick={() => void removePaper(paper)} aria-label={`Delete ${paper.title}`}><Trash2 size={12} /></button></div>
      </article>)}</div>
      <div className="library-footer"><span className="green-dot" />{readyCount} searchable {readyCount === 1 ? 'paper' : 'papers'}<small>{selected.size ? `${selected.size} selected for focused questions` : 'Knowledge stays inside this project'}</small></div>
    </aside>
    <main className="conversation"><header className="conversation-header"><div><Sparkles size={17} /><span>Research assistant</span></div><span className="scope-badge">PROJECT KNOWLEDGE</span></header>
      <div className="message-scroll">{messages.length === 0 ? <section className="welcome"><div className="welcome-mark"><BookOpen size={28} /></div><div className="eyebrow">FOLLOW YOUR CURIOSITY</div><h2>What are you exploring?</h2><p>Find the papers that matter.<br />Then ask questions of the research you’ve saved.</p><div className="suggestions"><button onClick={() => { setMode('search'); setInput('Retrieval augmented generation hallucination reduction'); }}><Search size={18} /><span><strong>Discover research</strong><small>Find papers on a topic</small></span><ChevronRight size={16} /></button><button onClick={() => { setMode('chat'); setInput('What are the main contributions of my saved papers?'); }}><BookOpen size={18} /><span><strong>Connect the ideas</strong><small>Explore your saved library</small></span><ChevronRight size={16} /></button></div></section> : <div className="messages">{messages.map(message => <article key={message.id} className={`message ${message.role}`}><div className="message-label">{message.role === 'user' ? 'YOU' : <><Sparkles size={13} /> RESEARCH AI</>}</div><div className="message-text">{message.role === 'assistant' ? <Markdown remarkPlugins={[remarkGfm]} skipHtml components={{ a: ({ children }) => <span>{children}</span>, img: () => null }}>{message.text}</Markdown> : message.text}</div>
        {message.papers && <div className="result-list">{message.papers.map(paper => { const key = paperKey(paper); const saved = papers.some(p => samePaper(p, paper)); return <article className="result-card" key={key}><div className="result-meta"><span>{paper.publication_year || 'Year unavailable'}</span><span>{paper.venue || 'Research paper'}</span></div><h3>{paper.title}</h3><div className="authors">{paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}</div><p>{paper.ai_summary || paper.abstract || paper.search_excerpt || 'No abstract is available for this paper.'}</p><footer><span className="pdf-badge">{paper.pdf_url ? 'Open-access PDF listed' : 'Manual PDF upload needed'}</span><div>{safeUrl(paper.source_url) && <a href={safeUrl(paper.source_url)} target="_blank" rel="noreferrer">Open ↗</a>}<button className="save-paper" disabled={saved || pending.has(key)} onClick={() => void savePaper(paper)}>{saved ? <Check size={14} /> : pending.has(key) ? <LoaderCircle size={14} className="spin" /> : <Plus size={14} />}{saved ? 'Saved' : pending.has(key) ? 'Saving…' : 'Save paper'}</button></div></footer></article>; })}</div>}
        {!!message.sources?.length && <div className="sources"><span>SOURCES</span>{message.sources.map(source => <button key={source.source_number} onClick={() => void openPdf(source.paper_id, source.page_start)}><span className="source-number">{source.source_number}</span><span>{source.paper_title}<small>Page {source.page_start}{source.page_end !== source.page_start ? `–${source.page_end}` : ''}</small></span></button>)}</div>}
      </article>)}</div>}{busy && <div className="thinking"><LoaderCircle size={15} className="spin" />{mode === 'search' ? 'Finding papers with Valyu…' : 'Reading your project’s research…'}</div>}<div ref={bottomRef} /></div>
      <div className="composer-area">{error && <div className="error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError('')}><X size={15} /></button></div>}{notice && <div className="notice" role="status">{notice}<button aria-label="Dismiss notification" onClick={() => setNotice('')}><X size={15} /></button></div>}
        <form className="composer" onSubmit={submit}><div className="mode-switch"><button type="button" className={mode === 'chat' ? 'selected' : ''} disabled={busy} onClick={() => setMode('chat')}><Sparkles size={14} /> Ask library</button><button type="button" className={mode === 'search' ? 'selected' : ''} disabled={busy} onClick={() => setMode('search')}><Search size={14} /> Find papers</button>{selected.size > 0 && <button className="clear-selection" type="button" onClick={() => setSelected(new Set())}>{selected.size} selected <X size={12} /></button>}</div>
          <div className="input-row"><textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }} maxLength={mode === 'search' ? 500 : 6000} rows={2} aria-label={mode === 'search' ? 'Paper search query' : 'Question for your library'} placeholder={mode === 'search' ? 'Search for a research topic…' : 'Ask a question about your saved papers…'} /><button className="send-button" aria-label="Send message" disabled={busy || !input.trim()}>{busy ? <LoaderCircle size={19} className="spin" /> : <ArrowUp size={20} />}</button></div>
        </form><div className="composer-note">{mode === 'search' ? 'Discovery is temporary. Save a paper to add it to your knowledge base.' : 'Answers are grounded in your saved papers. Always check the cited sources.'}</div>
      </div>
    </main>
  </div>;
}
