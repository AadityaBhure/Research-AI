import asyncio

import httpx

from app.services.errors import ServiceError


class LLMService:
    def __init__(self, client: httpx.AsyncClient, settings):
        self.client = client
        self.settings = settings
        self.semaphore = asyncio.Semaphore(2)

    async def generate(self, system: str, prompt: str, max_tokens: int = 1400) -> str:
        async with self.semaphore:
            for attempt in range(3):
                try:
                    response = await self.client.post(
                        'https://api.groq.com/openai/v1/chat/completions',
                        headers={'Authorization': f'Bearer {self.settings.groq_api_key.get_secret_value()}'},
                        json={
                            'model': self.settings.groq_model,
                            'messages': [{'role': 'system', 'content': system},
                                         {'role': 'user', 'content': prompt}],
                            'temperature': 0.2,
                            'max_completion_tokens': max_tokens,
                            **({'reasoning_effort': 'low'} if self.settings.groq_model.startswith('openai/gpt-oss') else {}),
                        },
                        timeout=90,
                    )
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        retry_after = response.headers.get('retry-after')
                        try:
                            delay = float(retry_after) if retry_after else float(min(12, 3 * (attempt + 1)))
                        except ValueError:
                            delay = float(min(12, 3 * (attempt + 1)))
                        await asyncio.sleep(max(1.0, delay))
                        continue
                    response.raise_for_status()
                    text = response.json()['choices'][0]['message']['content']
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('No response text')
                    return text.strip()
                except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                    if attempt < 2:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    raise ServiceError('Groq is unavailable or its model/key configuration needs attention. Please retry.') from None
        raise ServiceError('Groq did not return an answer.')
