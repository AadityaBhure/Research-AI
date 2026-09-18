import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import Settings
from app.models import ChatRequest, PaperInput, ProjectCreate, ProjectUpdate, SearchRequest
from app.services.database import SupabaseService
from app.services.discovery import ValyuService
from app.services.vectors import ZillizService
from app.services.errors import ServiceError
from app.services.ingestion import ACTIVE, IngestionService
from app.services.llm import LLMService
from app.services.pdf import validate_pdf
from app.services.rag import RAGService

logger = logging.getLogger('research_ai')


class BodyLimitMiddleware:
    def __init__(self, app, limit=31 * 1024 * 1024):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = dict(scope.get('headers', []))
        try:
            too_large = int(headers.get(b'content-length', b'0')) > self.limit
        except ValueError:
            too_large = True
        if too_large:
            return await JSONResponse({'detail': 'Request exceeds the upload limit.'}, 413)(scope, receive, send)
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get('body', b''))
            if size > self.limit:
                from starlette.exceptions import HTTPException
                raise HTTPException(413, 'Request exceeds the upload limit.')
            return message
        await self.app(scope, limited_receive, send)


class ConfiguredCORSMiddleware:
    def __init__(self, app):
        self.app = app
        self.cors = None

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if self.cors is None:
            self.cors = CORSMiddleware(self.app,
                allow_origins=scope['app'].state.settings.cors_origins,
                allow_methods=['GET', 'POST', 'PATCH', 'DELETE'], allow_headers=['Content-Type'])
        await self.cors(scope, receive, send)


def create_app(settings=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings or Settings()
        cfg = app.state.settings
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            app.state.db = SupabaseService(client, cfg)
            app.state.llm = LLMService(client, cfg)
            app.state.vectors = ZillizService(client, cfg)
            app.state.discovery = ValyuService(client, cfg, app.state.llm)
            app.state.ingestion = IngestionService(app.state.db, app.state.vectors, cfg)
            app.state.rag = RAGService(app.state.db, app.state.vectors, app.state.llm, cfg)
            app.state.startup_error = None
            try:
                await app.state.ingestion.recover()
            except ServiceError:
                app.state.startup_error = 'Supabase is unavailable or the initial migration has not been applied.'
                logger.warning('database_initialization_pending')
            yield
            await app.state.ingestion.close()

    app = FastAPI(title='Research AI', version='1.0.0', lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(ConfiguredCORSMiddleware)
    app.add_middleware(BodyLimitMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse({'detail': exc.message}, status_code=exc.status_code)

    @app.exception_handler(TimeoutError)
    async def request_timeout(request, exc):
        return JSONResponse({'detail': 'The request exceeded its processing time limit. Please retry.'}, 504)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic's default input field can echo user content; return locations/messages only.
        return JSONResponse({'detail': '; '.join(f'{".".join(str(x) for x in e["loc"])}: {e["msg"]}' for e in exc.errors())}, 422)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error('request_failed category=%s', type(exc).__name__)
        return JSONResponse({'detail': 'The request failed unexpectedly. Please retry.'}, 500)

    async def local_access(request: Request):
        # Local desktop app: no login. Reject requests from other hosts/websites.
        if request.client is None or request.client.host not in ('127.0.0.1', '::1'):
            raise ServiceError('This application is available on this computer only.', 403)
        origin = request.headers.get('origin')
        allowed = request.app.state.settings.cors_origins + [str(request.base_url).rstrip('/')]
        if (origin and origin not in allowed) or request.headers.get('sec-fetch-site') == 'cross-site':
            raise ServiceError('Cross-site requests are not allowed.', 403)

    from fastapi import APIRouter
    api = APIRouter(prefix='/api')

    @app.get('/health')
    async def health():
        return {'status': 'running'}

    @api.get('/status')
    async def status(request: Request):
        await request.app.state.db.rows('projects', limit='1')
        await request.app.state.db.rows('papers', select='id,search_provider,provider_paper_id,ingestion_token,embedding_model,ingestion_detail', limit='1')
        await request.app.state.vectors.check()
        cfg = request.app.state.settings
        return {'status': 'ready', 'llm_provider': 'groq', 'search_provider': 'valyu',
                'embedding_model': cfg.embedding_model, 'vector_provider': 'zilliz',
                'retrieval': 'hybrid', 'max_upload_bytes': cfg.max_upload_bytes}

    @api.get('/projects')
    async def projects(request: Request):
        return await request.app.state.db.rows('projects', order='updated_at.desc')

    @api.post('/projects', status_code=201)
    async def create_project(body: ProjectCreate, request: Request):
        result = await request.app.state.db.insert('projects', body.model_dump())
        logger.info('project_created project_id=%s', result[0]['id'])
        return result[0]

    @api.get('/projects/{project_id}')
    async def get_project(project_id: UUID, request: Request):
        return await request.app.state.db.project(str(project_id))

    @api.patch('/projects/{project_id}')
    async def update_project(project_id: UUID, body: ProjectUpdate, request: Request):
        async with request.app.state.ingestion.lock(project_id):
            await request.app.state.db.project(str(project_id))
            changes = body.model_dump(exclude_unset=True)
            if not changes:
                raise ServiceError('Provide a project name or description to update.', 400)
            return (await request.app.state.db.update('projects', changes, id=f'eq.{project_id}'))[0]

    async def all_papers(db, project_id):
        result = []
        while True:
            rows = await db.rows('papers', project_id=f'eq.{project_id}', order='created_at.desc,id',
                                  limit='500', offset=str(len(result)))
            result.extend(rows)
            if len(rows) < 500:
                return result

    @api.delete('/projects/{project_id}')
    async def delete_project(project_id: UUID, request: Request):
        db = request.app.state.db
        async with request.app.state.ingestion.lock(project_id):
            await db.project(str(project_id))
            papers = await all_papers(db, project_id)
            if any(p['status'] in ACTIVE for p in papers):
                raise ServiceError('Wait for active paper processing to finish before deleting the project.', 409)
            await request.app.state.vectors.delete(project_id)
            await db.remove_pdfs([p['storage_path'] for p in papers if p['storage_path']])
            await db.delete('projects', id=f'eq.{project_id}')
        return {'status': 'deleted'}

    @api.post('/projects/{project_id}/papers/search')
    async def search(project_id: UUID, body: SearchRequest, request: Request):
        await request.app.state.db.project(str(project_id))
        async with asyncio.timeout(240):
            return await request.app.state.discovery.search(body.query, body.limit)

    @api.get('/projects/{project_id}/papers')
    async def list_papers(project_id: UUID, request: Request):
        await request.app.state.db.project(str(project_id))
        await request.app.state.ingestion.recover()
        return await all_papers(request.app.state.db, project_id)

    @api.post('/projects/{project_id}/papers', status_code=202)
    async def save_paper(project_id: UUID, body: PaperInput, request: Request):
        state = request.app.state
        async with state.ingestion.lock(project_id):
            await state.db.project(str(project_id))
            result = await state.db.rpc('save_research_paper', {'target_project_id': str(project_id), 'metadata': body.model_dump(mode='json')})
        if result['status'] != 'already_exists' and result['paper']['pdf_url']:
            if state.settings.vercel:
                # Browser starts a separate request; metadata save returns immediately.
                result['processing_required'] = True
            else:
                state.ingestion.schedule(project_id, result['paper']['id'])
        return result

    async def uploaded_bytes(file, maximum):
        try:
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                raise ServiceError('Choose a .pdf file.', 400)
            if file.content_type not in ('application/pdf', 'application/octet-stream'):
                raise ServiceError('The upload must have a PDF content type.', 400)
            data = await file.read(maximum + 1)
            validate_pdf(data, maximum)
            return data
        finally:
            await file.close()

    @api.post('/projects/{project_id}/papers/upload', status_code=202)
    async def upload(project_id: UUID, request: Request, file: UploadFile = File(...),
                     title: str | None = Form(None), authors: str = Form('[]')):
        state = request.app.state
        data = await uploaded_bytes(file, state.settings.max_upload_bytes)
        try:
            body = PaperInput(title=title or Path(file.filename).stem, authors=json.loads(authors))
        except (ValueError, TypeError):
            raise ServiceError('Provide a title and a JSON list of author names.', 400) from None
        async with state.ingestion.lock(project_id):
            await state.db.project(str(project_id))
            result = await state.db.rpc('save_research_paper', {'target_project_id': str(project_id), 'metadata': body.model_dump(mode='json')})
            if result['status'] == 'already_exists':
                return result
            await state.ingestion.state(str(project_id), result['paper']['id'], 'saved')
            result['paper']['status'] = 'saved'
        await state.ingestion.dispatch(project_id, result['paper']['id'], data)
        result['paper'] = await state.db.paper(str(project_id), result['paper']['id'])
        return result

    @api.get('/projects/{project_id}/papers/{paper_id}')
    async def paper(project_id: UUID, paper_id: UUID, request: Request):
        return await request.app.state.db.paper(str(project_id), str(paper_id))

    @api.post('/projects/{project_id}/papers/{paper_id}/process', status_code=202)
    async def process_paper(project_id: UUID, paper_id: UUID, request: Request):
        state = request.app.state
        paper = await state.db.paper(str(project_id), str(paper_id))
        if paper['status'] != 'saved':
            raise ServiceError('This paper is not queued for processing.', 409)
        await state.ingestion.dispatch(project_id, paper_id)
        return {'paper': await state.db.paper(str(project_id), str(paper_id))}

    @api.post('/projects/{project_id}/papers/{paper_id}/reindex', status_code=202)
    async def reindex_paper(project_id: UUID, paper_id: UUID, request: Request):
        state = request.app.state
        async with state.ingestion.lock(project_id):
            paper = await state.db.paper(str(project_id), str(paper_id))
            if paper['status'] in ACTIVE:
                raise ServiceError('This paper is already processing.', 409)
            if paper.get('embedding_model') == state.settings.embedding_model and paper['status'] == 'ready':
                return {'status': 'already_ready', 'paper': paper}
            if not (paper.get('storage_path') or paper.get('pdf_url')):
                raise ServiceError('Attach a PDF before reindexing this paper.', 400)
            changed = await state.db.update('papers', {'status': 'saved', 'ingestion_error': None,
                'ingestion_token': None, 'ingestion_detail': 'Queued for Zilliz reindex'},
                id=f'eq.{paper_id}', project_id=f'eq.{project_id}', status=f'eq.{paper["status"]}',
                updated_at=f'eq.{paper["updated_at"]}')
            if not changed:
                raise ServiceError('This paper changed while reindexing was requested. Refresh and retry.', 409)
        await state.ingestion.dispatch(project_id, paper_id)
        return {'status': 'saved', 'paper': await state.db.paper(str(project_id), str(paper_id))}

    @api.post('/projects/{project_id}/papers/{paper_id}/upload', status_code=202)
    async def attach_pdf(project_id: UUID, paper_id: UUID, request: Request, file: UploadFile = File(...)):
        state = request.app.state
        data = await uploaded_bytes(file, state.settings.max_upload_bytes)
        async with state.ingestion.lock(project_id):
            paper = await state.db.paper(str(project_id), str(paper_id))
            if paper['status'] in ACTIVE or paper['status'] == 'ready':
                raise ServiceError('This paper is already processing or ready.', 409)
            await state.ingestion.prepare_retry(str(project_id), str(paper_id))
        await state.ingestion.dispatch(project_id, paper_id, data)
        return {'status': 'saved', 'paper': await state.db.paper(str(project_id), str(paper_id))}

    @api.post('/projects/{project_id}/papers/{paper_id}/retry', status_code=202)
    async def retry(project_id: UUID, paper_id: UUID, request: Request):
        state = request.app.state
        async with state.ingestion.lock(project_id):
            paper = await state.db.paper(str(project_id), str(paper_id))
            if paper['status'] not in ('failed', 'no_pdf'):
                raise ServiceError('Only failed or missing-PDF papers can be retried.', 409)
            if not (paper['pdf_url'] or paper.get('storage_path')):
                raise ServiceError('Upload a PDF for this paper to retry ingestion.', 400)
            await state.ingestion.prepare_retry(str(project_id), str(paper_id))
        await state.ingestion.dispatch(project_id, paper_id)
        return {'status': 'saved', 'paper': await state.db.paper(str(project_id), str(paper_id))}

    @api.get('/projects/{project_id}/papers/{paper_id}/pdf')
    async def open_pdf(project_id: UUID, paper_id: UUID, request: Request):
        db = request.app.state.db
        paper = await db.paper(str(project_id), str(paper_id))
        if not paper['storage_path']:
            raise ServiceError('This paper does not have a stored PDF yet.', 404)
        return {'url': await db.signed_pdf(paper['storage_path'])}

    @api.delete('/projects/{project_id}/papers/{paper_id}')
    async def delete_paper(project_id: UUID, paper_id: UUID, request: Request):
        state = request.app.state
        async with state.ingestion.lock(project_id):
            paper = await state.db.paper(str(project_id), str(paper_id))
            if paper['status'] in ACTIVE:
                raise ServiceError('Wait for this paper to finish processing before deleting it.', 409)
            await state.vectors.delete(project_id, paper_id)
            if paper['storage_path']:
                await state.db.remove_pdfs([paper['storage_path']])
            await state.db.delete('papers', id=f'eq.{paper_id}', project_id=f'eq.{project_id}')
        return {'status': 'deleted'}

    @api.post('/projects/{project_id}/chat')
    async def chat(project_id: UUID, body: ChatRequest, request: Request):
        await request.app.state.db.project(str(project_id))
        async with asyncio.timeout(120):
            return await request.app.state.rag.answer(project_id, body)

    app.include_router(api)
    frontend = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
    if frontend.is_dir():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')
    return app


app = create_app()
