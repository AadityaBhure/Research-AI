from urllib.parse import quote

import httpx

from app.services.errors import ServiceError


class SupabaseService:
    """Use Supabase's REST, RPC and Storage APIs with one async connection pool."""

    def __init__(self, client, settings):
        self.client, self.settings = client, settings
        self.base = settings.supabase_url
        key = settings.supabase_secret_key.get_secret_value()
        self.headers = {'apikey': key, 'Authorization': f'Bearer {key}'}

    async def request(self, method, path, *, params=None, json=None, content=None, headers=None):
        try:
            response = await self.client.request(
                method, self.base + path, params=params, json=json, content=content,
                headers={**self.headers, **(headers or {})}, timeout=90,
            )
            if response.status_code == 409:
                raise ServiceError('This record already exists or is currently being processed.', 409)
            response.raise_for_status()
            return response.json() if response.content else None
        except (httpx.HTTPError, ValueError):
            raise ServiceError('Supabase request failed. Check its configuration, migrations and availability.') from None

    async def rows(self, table, **filters):
        return await self.request('GET', f'/rest/v1/{table}', params={'select': '*', **filters})

    async def insert(self, table, data):
        return await self.request('POST', f'/rest/v1/{table}', json=data,
                                  headers={'Prefer': 'return=representation'})

    async def update(self, table, data, **filters):
        return await self.request('PATCH', f'/rest/v1/{table}', params=filters, json=data,
                                  headers={'Prefer': 'return=representation'})

    async def delete(self, table, **filters):
        return await self.request('DELETE', f'/rest/v1/{table}', params=filters)

    async def rpc(self, name, data):
        return await self.request('POST', f'/rest/v1/rpc/{name}', json=data)

    async def project(self, project_id):
        rows = await self.rows('projects', id=f'eq.{project_id}')
        if not rows:
            raise ServiceError('Project not found.', 404)
        return rows[0]

    async def paper(self, project_id, paper_id):
        rows = await self.rows('papers', id=f'eq.{paper_id}', project_id=f'eq.{project_id}')
        if not rows:
            raise ServiceError('Paper not found in this project.', 404)
        return rows[0]

    async def store_pdf(self, path, content):
        await self.request('POST', f'/storage/v1/object/{self.settings.storage_bucket}/{quote(path, safe="/")}',
                           content=content, headers={'Content-Type': 'application/pdf', 'x-upsert': 'true'})

    async def read_pdf(self, path):
        # Only the stored, server-owned bucket path is accepted, never a client URL.
        url = self.base + f'/storage/v1/object/authenticated/{self.settings.storage_bucket}/{quote(path, safe="/")}'
        try:
            async with self.client.stream('GET', url, headers=self.headers, timeout=45) as response:
                response.raise_for_status()
                content = bytearray()
                async for block in response.aiter_bytes():
                    content.extend(block)
                    if len(content) > self.settings.max_pdf_bytes:
                        raise ServiceError('Stored PDF exceeds the download limit.', 400)
                return bytes(content)
        except httpx.HTTPError:
            raise ServiceError('Could not read the stored PDF. Check storage or attach the PDF again.', 502) from None

    async def remove_pdfs(self, paths):
        for start in range(0, len(paths), 100):
            await self.request('DELETE', f'/storage/v1/object/{self.settings.storage_bucket}',
                               json={'prefixes': paths[start:start + 100]})

    async def signed_pdf(self, path):
        data = await self.request('POST', f'/storage/v1/object/sign/{self.settings.storage_bucket}/{quote(path, safe="/")}',
                                  json={'expiresIn': 300})
        return self.base + '/storage/v1' + data['signedURL']
