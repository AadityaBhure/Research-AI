"""Request-scoped MCP transport; only the Valyu search tool is callable."""
import asyncio
import json
import logging
import re
from datetime import timedelta
from urllib.parse import urlencode

import httpx
from jsonschema import validate
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.services.errors import ServiceError


class _ProtocolLogFilter(logging.Filter):
    def filter(self, record):
        # The hosted endpoint authenticates through its URL. SDK exceptions and
        # HTTP request logs can include that URL, so never emit protocol logs.
        return False


class _HttpLogFilter(logging.Filter):
    def filter(self, record):
        return 'mcp.valyu.ai' not in record.getMessage() and not record.exc_info


logging.getLogger('mcp.client.streamable_http').addFilter(_ProtocolLogFilter())
logging.getLogger('httpx').addFilter(_HttpLogFilter())


def decode_search_result(result):
    """Accept structured JSON or the hosted server's numbered Markdown records.

    Metadata is parsed deterministically, never reconstructed by the LLM.
    A changed server format fails closed instead of producing invented cards.
    """
    payload = result.structuredContent
    if isinstance(payload, dict) and isinstance(payload.get('results'), list):
        return payload
    texts = [c.text for c in result.content if c.type == 'text']
    for text in texts:
        try:
            candidate = json.loads(text)
            if isinstance(candidate, dict) and isinstance(candidate.get('results'), list):
                return candidate
        except ValueError:
            pass
    text = '\n'.join(t for t in texts if not t.lstrip().startswith('{"conversation_id"'))
    count = re.match(r'Found (\d+) results? for ', text)
    if not count:
        raise ValueError('Unknown MCP search format')
    pattern = re.compile(r'^### (\d+)\. (.+)\r?\n(https?://\S+) ([^\r\n]*relevance:[^\r\n]*)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != int(count[1]):
        raise ValueError('Incomplete MCP search records')
    records = []
    for i, match in enumerate(matches):
        metadata = match[4]
        published = re.search(r'published: (\d{4}-\d{2}-\d{2})', metadata)
        doi = re.search(r'doi: (\S+)', metadata)
        records.append({
            'title': match[2], 'url': match[3],
            'publication_date': published[1] if published else None,
            'doi': doi[1].removeprefix('https://doi.org/') if doi else None,
            # Author strings can be truncated with "et al."; do not pretend
            # these are complete structured author lists or abstracts.
            'content': text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()[:8000],
        })
    return {'success': True, 'results': records}


class ValyuMCP:
    def __init__(self, settings):
        self.settings = settings

    async def search(self, query, limit):
        key = self.settings.valyu_api_key.get_secret_value().strip()
        if not key:
            raise ServiceError('Research search is not configured. Set VALYU_API_KEY on the backend.', 503)
        url = 'https://mcp.valyu.ai/mcp?' + urlencode({'valyuApiKey': key})
        arguments = {
            'query': query, 'max_num_results': limit,
            'included_sources': ['valyu/valyu-arxiv', 'valyu/valyu-pubmed'],
            'response_length': 8000,
            'context': 'A user is discovering academic research papers to review relevant literature and select sources for a research project.',
            'llm_model': self.settings.groq_model,
        }
        stage = 'connect'
        try:
            async with asyncio.timeout(65):
                async with httpx.AsyncClient(timeout=60, follow_redirects=False) as http:
                    async with streamable_http_client(url, http_client=http) as (read, write, _):
                        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)) as session:
                            await session.initialize()
                            stage = 'list_tools'
                            listing = await session.list_tools()
                            tool = next(t for t in listing.tools if t.name == 'valyu_search')
                            validate(arguments, tool.inputSchema)
                            stage = 'call_tool'
                            # Never retry: an uncertain response may already be billed.
                            result = await session.call_tool('valyu_search', arguments)
                            if result.isError:
                                raise ValueError('Tool failure')
                            stage = 'decode_result'
                            payload = decode_search_result(result)
                            if not isinstance(payload, dict) or not isinstance(payload.get('results'), list) or payload.get('success') is False:
                                raise ValueError('Invalid research response')
                            return payload
        except Exception:
            logging.getLogger(__name__).warning('valyu_mcp_failed stage=%s', stage)
            # Do not leak SDK exception groups, URLs, tool content or credentials.
            raise ServiceError('Valyu MCP search failed. Check the backend key, credits and service availability.', 502) from None
