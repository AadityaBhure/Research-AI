import asyncio
import logging
from datetime import datetime, timedelta, timezone
from time import perf_counter

from app.services.errors import ServiceError
from app.services.embeddings import chunk_pages
from app.services.pdf import download_pdf, extract_pages, validate_pdf

logger = logging.getLogger(__name__)
ACTIVE = ['saved', 'downloading', 'extracting', 'embedding', 'storing']


class IngestionService:
    def __init__(self, db, vectors, settings):
        self.db, self.vectors, self.settings = db, vectors, settings
        self.tasks = set()
        self.locks = {}
        self.capacity = asyncio.Semaphore(2)

    def lock(self, project_id):
        return self.locks.setdefault(str(project_id), asyncio.Lock())

    async def recover(self):
        # Never reset other instances' live jobs on a serverless cold start.
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        await self.db.update('papers', {'status': 'failed', 'ingestion_token': None, 'ingestion_error':
                            'Ingestion was interrupted or expired. Retry or upload the PDF again.'},
                             status='in.(' + ','.join(ACTIVE) + ')', updated_at=f'lt.{cutoff}')

    async def dispatch(self, project_id, paper_id, data=None):
        if self.settings.vercel:
            # Work must finish before the function response, not in a detached task.
            await self.ingest(str(project_id), str(paper_id), data)
        else:
            self.schedule(project_id, paper_id, data)

    def schedule(self, project_id, paper_id, data=None):
        task = asyncio.create_task(self.ingest(str(project_id), str(paper_id), data))
        self.tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task):
        self.tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error('ingestion_task_failed category=%s', type(task.exception()).__name__)

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def state(self, project_id, paper_id, status, token=None, **extra):
        filters = {'ingestion_token': f'eq.{token}'} if token else {}
        rows = await self.db.update('papers', {'status': status, **extra},
                                    id=f'eq.{paper_id}', project_id=f'eq.{project_id}', **filters)
        if token and not rows:
            raise ServiceError('Ingestion claim expired or paper was removed.', 409)
        return rows

    async def prepare_retry(self, project_id, paper_id):
        rows = await self.db.update('papers', {'status': 'saved', 'ingestion_error': None, 'ingestion_token': None},
                                    id=f'eq.{paper_id}', project_id=f'eq.{project_id}', status='in.(failed,no_pdf)')
        if not rows:
            raise ServiceError('This paper is already processing or ready.', 409)

    async def ingest(self, project_id, paper_id, data):
        try:
            async with asyncio.timeout(self.settings.ingestion_timeout_seconds):
                async with self.capacity:
                    await self._process(project_id, paper_id, data)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            logger.warning('pdf_ingestion_timeout paper_id=%s', paper_id)
            # An unclaimed job may have timed out while waiting for local capacity.
            await self.db.update('papers', {'status': 'failed', 'ingestion_error': 'Processing timed out. Retry with a smaller PDF.'},
                                 id=f'eq.{paper_id}', project_id=f'eq.{project_id}', status='eq.saved', ingestion_token='is.null')

    async def _process(self, project_id, paper_id, data):
        # Only claim under the short project mutation lock, never the whole job.
        async with self.lock(project_id):
            token = await self.db.rpc('claim_ingestion', {'target_project_id': project_id, 'target_paper_id': paper_id})
        if not token:
            return
        started = perf_counter()
        try:
            paper = await self.db.paper(project_id, paper_id)
            if data is None:
                if paper.get('storage_path'):
                    await self.state(project_id, paper_id, 'downloading', token=token, ingestion_detail='Reading stored PDF')
                    data = await self.db.read_pdf(paper['storage_path'])
                elif not paper['pdf_url']:
                    await self.state(project_id, paper_id, 'no_pdf', token=token, ingestion_error='Upload a PDF to make this paper searchable.')
                    return
                else:
                    await self.state(project_id, paper_id, 'downloading', token=token, ingestion_error=None, ingestion_detail='Downloading public PDF')
                    logger.info('pdf_download_started paper_id=%s', paper_id)
                    data = await asyncio.wait_for(download_pdf(paper['pdf_url'], self.settings.max_pdf_bytes), timeout=120)
            validate_pdf(data, self.settings.max_pdf_bytes)
            path = f'projects/{project_id}/papers/{paper_id}/paper.pdf'
            # Persist path before uploading so deletion can clean up even after partial failures.
            await self.state(project_id, paper_id, 'extracting', token=token, storage_path=path, ingestion_error=None,
                             ingestion_detail='Extracting PDF text')
            await self.db.store_pdf(path, data)
            logger.info('pdf_ingestion_started paper_id=%s', paper_id)
            pages = await asyncio.to_thread(extract_pages, data)
            stage_started = perf_counter()
            await self.state(project_id, paper_id, 'embedding', token=token, ingestion_detail='Splitting text into page-aware chunks')
            chunks = await asyncio.to_thread(chunk_pages, pages)
            logger.info('pdf_chunking_completed paper_id=%s chunks=%s seconds=%.3f', paper_id, len(chunks), perf_counter() - stage_started)
            async def progress(done, total):
                await self.state(project_id, paper_id, 'embedding', token=token,
                                 ingestion_detail=f'Zilliz hybrid indexing: {done}/{total} chunks')
            await progress(0, len(chunks))
            await self.vectors.index(project_id, paper_id, token, chunks, progress)
            await self.state(project_id, paper_id, 'storing', token=token, ingestion_detail='Saving searchable vectors')
            # Publish only after all vector writes succeed. Retain the generation
            # token on the ready row; retrieval requires this exact token.
            await self.state(project_id, paper_id, 'ready', token=token,
                             embedding_model=self.settings.embedding_model,
                             ingestion_error=None, ingestion_detail='Indexed with Zilliz hybrid search')
            logger.info('pdf_ingestion_completed paper_id=%s chunks=%s seconds=%.3f', paper_id, len(chunks), perf_counter() - started)
        except asyncio.CancelledError:
            await self.state(project_id, paper_id, 'failed', token=token, ingestion_token=None,
                             ingestion_error='Processing was interrupted or exceeded its time limit. Retry with a smaller PDF.')
            raise
        except Exception as exc:
            message = exc.message if isinstance(exc, ServiceError) else 'Ingestion failed. Please retry or upload another PDF.'
            logger.warning('pdf_ingestion_failed paper_id=%s category=%s', paper_id, type(exc).__name__)
            try:
                await self.state(project_id, paper_id, 'failed', token=token, ingestion_token=None, ingestion_error=message)
            except Exception:
                logger.error('ingestion_status_update_failed paper_id=%s', paper_id)
