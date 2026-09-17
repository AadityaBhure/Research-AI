import asyncio
import logging

from app.services.errors import ServiceError
from app.services.pdf import download_pdf, extract_pages, validate_pdf

logger = logging.getLogger(__name__)
ACTIVE = ['saved', 'downloading', 'extracting', 'embedding', 'storing']


class IngestionService:
    def __init__(self, db, embeddings, settings):
        self.db, self.embeddings, self.settings = db, embeddings, settings
        self.tasks = set()
        self.locks = {}
        self.capacity = asyncio.Semaphore(2)

    def lock(self, project_id):
        return self.locks.setdefault(str(project_id), asyncio.Lock())

    async def recover(self):
        # V1 runs one API worker. In-flight jobs cannot survive a process restart.
        await self.db.update('papers', {'status': 'failed', 'ingestion_error':
                            'Ingestion was interrupted by a server restart. Retry or upload the PDF again.'},
                             status='in.(' + ','.join(ACTIVE) + ')')

    def schedule(self, project_id, paper_id, data=None):
        task = asyncio.create_task(self.ingest(str(project_id), str(paper_id), data))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def state(self, project_id, paper_id, status, **extra):
        await self.db.update('papers', {'status': status, **extra},
                             id=f'eq.{paper_id}', project_id=f'eq.{project_id}')

    async def ingest(self, project_id, paper_id, data):
        async with self.capacity, self.lock(project_id):
            try:
                paper = await self.db.paper(project_id, paper_id)
                if data is None:
                    if not paper['pdf_url']:
                        await self.state(project_id, paper_id, 'no_pdf', ingestion_error='Upload a PDF to make this paper searchable.')
                        return
                    await self.state(project_id, paper_id, 'downloading', ingestion_error=None)
                    logger.info('pdf_download_started paper_id=%s', paper_id)
                    data = await asyncio.wait_for(download_pdf(paper['pdf_url'], self.settings.max_pdf_bytes), timeout=120)
                validate_pdf(data, self.settings.max_pdf_bytes)
                path = f'projects/{project_id}/papers/{paper_id}/paper.pdf'
                # Persist path before uploading so deletion can clean up even after partial failures.
                await self.state(project_id, paper_id, 'extracting', storage_path=path, ingestion_error=None)
                await self.db.store_pdf(path, data)
                logger.info('pdf_ingestion_started paper_id=%s', paper_id)
                pages = await asyncio.to_thread(extract_pages, data)
                await self.state(project_id, paper_id, 'embedding')
                chunks = await asyncio.to_thread(self.embeddings.chunk_pages, pages)
                vectors = await asyncio.to_thread(self.embeddings.documents, [c.content for c in chunks])
                if len(chunks) != len(vectors) or not chunks:
                    raise ServiceError('Embedding count does not match document chunks.')
                await self.state(project_id, paper_id, 'storing')
                # Transactional RPC replaces old chunks and marks ready in one commit.
                await self.db.rpc('complete_ingestion', {
                    'target_project_id': project_id, 'target_paper_id': paper_id,
                    'chunks': [{**chunk.model_dump(), 'embedding': vector} for chunk, vector in zip(chunks, vectors)],
                })
                logger.info('pdf_ingestion_completed paper_id=%s chunks=%s', paper_id, len(chunks))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message = exc.message if isinstance(exc, ServiceError) else 'Ingestion failed. Please retry or upload another PDF.'
                logger.warning('pdf_ingestion_failed paper_id=%s category=%s', paper_id, type(exc).__name__)
                try:
                    await self.state(project_id, paper_id, 'failed', ingestion_error=message)
                except Exception:
                    logger.error('ingestion_status_update_failed paper_id=%s', paper_id)
