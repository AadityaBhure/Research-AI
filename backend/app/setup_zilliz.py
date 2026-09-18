"""Run from backend: python -m app.setup_zilliz. Never prints credentials."""
import asyncio
import httpx
from app.config import Settings
from app.services.vectors import ZillizService
from app.services.errors import ServiceError


async def main():
    async with httpx.AsyncClient(follow_redirects=False) as client:
        try:
            await ZillizService(client, Settings()).setup()
            print('Zilliz collection provisioned. Cohere embedding access still needs an ingestion smoke test.')
        except ServiceError as exc:
            print(exc.message)
            raise SystemExit(1) from None


if __name__ == '__main__':
    asyncio.run(main())
