"""Async Zilliz REST integration; Cohere dense vectors plus built-in BM25.

Each ingestion has a generation UUID. Only generations committed as ready in
Supabase are searchable, so partial/expired indexing never leaks into answers.
"""
import asyncio
from uuid import UUID

import httpx

from app.services.errors import ServiceError

OUTPUT = ['id', 'project_id', 'paper_id', 'generation', 'content', 'page_start', 'page_end', 'chunk_index']


class ZillizService:
    def __init__(self, client, settings):
        self.client, self.settings = client, settings
        self.documents_capacity = asyncio.Semaphore(1)
        self.queries_capacity = asyncio.Semaphore(4)

    async def call(self, operation, payload=None, timeout=45):
        cfg = self.settings
        if not cfg.zilliz_public_endpoint or not cfg.zilliz_api_key.get_secret_value():
            raise ServiceError('Configure the backend Zilliz endpoint and API key.', 503)
        try:
            response = await self.client.post(
                f'{cfg.zilliz_public_endpoint}/v2/vectordb/{operation}',
                headers={'Authorization': f'Bearer {cfg.zilliz_api_key.get_secret_value()}'},
                json={'collectionName': cfg.zilliz_collection, **(payload or {})},
                timeout=timeout,
            )
            if response.status_code == 429:
                raise ServiceError('Zilliz or its embedding provider is rate limited. Retry later.', 503)
            if response.status_code in (401, 403):
                raise ServiceError('Zilliz denied this operation. Check key and database/collection permissions.', 503)
            response.raise_for_status()
            result = response.json()
            if result.get('code') != 0:
                if result.get('code') == 80020:
                    raise ServiceError('Zilliz cluster not found or API key lacks cluster access.', 503)
                # Never expose provider bodies, credentials, or source text.
                raise ServiceError('Zilliz rejected the request. Check collection, model integration and provider quota.', 502)
            return result.get('data')
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            raise ServiceError('Zilliz request failed. Check connectivity and retry.', 502) from None

    def schema(self):
        def text_field(name, maximum):
            return {'fieldName': name, 'dataType': 'VarChar', 'elementTypeParams': {'max_length': str(maximum)}}
        fields = [dict(text_field('id', 80), isPrimary=True), text_field('project_id', 36),
                  text_field('paper_id', 36), text_field('generation', 36),
                  text_field('content', 4096)]
        fields[-1]['elementTypeParams'].update({'enable_analyzer': True})
        fields += [{'fieldName': n, 'dataType': 'Int64'} for n in ('page_start', 'page_end', 'chunk_index')]
        fields += [{'fieldName': 'dense', 'dataType': 'FloatVector', 'elementTypeParams': {'dim': '1024'}},
                   {'fieldName': 'sparse', 'dataType': 'SparseFloatVector'}]
        return {'autoId': False, 'enableDynamicField': False, 'fields': fields, 'functions': [
            {'name': 'cohere_embedding', 'type': 'TextEmbedding', 'inputFieldNames': ['content'],
             'outputFieldNames': ['dense'], 'params': {'provider': 'cohere',
             'model_name': self.settings.zilliz_embedding_model,
             'integration_id': self.settings.zilliz_embedding_integration_id.get_secret_value(), 'truncate': 'NONE'}},
            {'name': 'lexical', 'type': 'BM25', 'inputFieldNames': ['content'], 'outputFieldNames': ['sparse'], 'params': {}},
        ]}

    async def setup(self):
        """Explicit provisioning only; never create/drop collections at startup."""
        if not self.settings.zilliz_embedding_integration_id.get_secret_value():
            raise ServiceError('Set ZILLIZ_EMBEDDING_INTEGRATION_ID for a Cohere integration.', 503)
        exists = await self.call('collections/has')
        if not exists.get('has'):
            await self.call('collections/create', {'schema': self.schema(),
                'params': {'consistencyLevel': 'Strong'}, 'indexParams': [
                    {'fieldName': 'dense', 'indexName': 'dense_index', 'metricType': 'COSINE', 'params': {'index_type': 'AUTOINDEX'}},
                    {'fieldName': 'sparse', 'indexName': 'sparse_index', 'metricType': 'BM25', 'params': {'index_type': 'AUTOINDEX'}},
                ]})
        await self.call('collections/load')
        description = await self.call('collections/describe')
        self.validate_schema(description)
        return description

    def validate_schema(self, description):
        fields = {f.get('name'): f for f in description.get('fields', [])}
        functions = {f.get('name'): f for f in description.get('functions', [])}
        def params(value):
            return value if isinstance(value, dict) else {p['key']: p['value'] for p in value or []}
        dense = fields.get('dense', {})
        cohere = params(functions.get('cohere_embedding', {}).get('params'))
        if (not set(OUTPUT + ['dense', 'sparse']).issubset(fields)
                or str(params(dense.get('params')).get('dim')) != '1024'
                or 'lexical' not in functions
                or cohere.get('model_name') != self.settings.zilliz_embedding_model
                or cohere.get('provider') != 'cohere'
                or description.get('consistencyLevel') != 'Strong'):
            raise ServiceError('Zilliz collection schema does not match this application. Use a new compatible collection; do not overwrite existing data.', 503)

    async def check(self):
        exists = await self.call('collections/has')
        if not exists.get('has'):
            raise ServiceError('Zilliz collection is not provisioned. Run python -m app.setup_zilliz.', 503)
        description = await self.call('collections/describe')
        self.validate_schema(description)
        if description.get('load') != 'LoadStateLoaded':
            raise ServiceError('Zilliz collection is not loaded. Run python -m app.setup_zilliz.', 503)

    @staticmethod
    def scope(project_id, papers):
        project = str(UUID(str(project_id)))
        versions = [f'(paper_id == "{UUID(str(p["id"]))}" and generation == "{UUID(str(p["ingestion_token"]))}")'
                    for p in papers if p.get('ingestion_token')]
        if not versions:
            raise ServiceError('No committed Zilliz paper generations are available.', 409)
        return f'project_id == "{project}" and (' + ' or '.join(versions) + ')'

    async def index(self, project_id, paper_id, token, chunks, progress):
        for start in range(0, len(chunks), 32):
            rows = [{**{k: v for k, v in c.model_dump().items() if k != 'token_count'},
                     'id': f'{token}:{c.chunk_index}', 'project_id': str(project_id),
                     'paper_id': str(paper_id), 'generation': str(token)} for c in chunks[start:start + 32]]
            async with self.documents_capacity:
                await self.call('entities/upsert', {'data': rows})
            await progress(min(start + 32, len(chunks)), len(chunks))

    async def search(self, project_id, papers, question, limit):
        if len(question.encode('utf-8')) > 480:
            raise ServiceError('For this embedding model, use a concise question of at most 480 UTF-8 bytes. Split longer questions into parts.', 400)
        scope = self.scope(project_id, papers)
        async with self.queries_capacity:
            rows = await self.call('entities/hybrid_search', {
                'search': [{'data': [question], 'annsField': field, 'filter': scope, 'limit': max(20, limit)}
                           for field in ('dense', 'sparse')],
                'rerank': {'strategy': 'rrf', 'params': {'k': 60}}, 'limit': limit,
                'outputFields': OUTPUT, 'consistencyLevel': 'Strong'}, timeout=30)
        return self.results(rows, papers)

    @staticmethod
    def results(rows, papers):
        allowed = {(str(p['id']), str(p['ingestion_token'])): p['title'] for p in papers}
        result = []
        for row in rows or []:
            title = allowed.get((row.get('paper_id'), row.get('generation')))
            if title is not None:
                # RRF is a ranking score, NOT cosine similarity/confidence.
                result.append({**row, 'paper_title': title, 'similarity': None})
        return result

    async def opening(self, project_id, paper):
        rows = await self.call('entities/query', {'filter': self.scope(project_id, [paper]) + ' and chunk_index == 0',
            'outputFields': OUTPUT, 'limit': 1, 'consistencyLevel': 'Strong'})
        return self.results(rows, [paper])

    async def delete(self, project_id, paper_id=None):
        scope = f'project_id == "{UUID(str(project_id))}"'
        if paper_id:
            scope += f' and paper_id == "{UUID(str(paper_id))}"'
        await self.call('entities/delete', {'filter': scope})
