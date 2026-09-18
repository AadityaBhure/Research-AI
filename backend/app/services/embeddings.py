"""Page-aware splitting only. Zilliz performs hosted embedding inference."""
import math
from app.models import Chunk
from app.services.errors import ServiceError


def chunk_pages(pages, target=480, overlap=60):
    """Conservative UTF-8 byte budget for Cohere v3's 512-token context.

    No model download or silent truncation. token_count is an estimate only.
    """
    if target <= overlap or overlap < 0 or target > 480:
        raise ValueError('Chunk byte target must exceed overlap and be at most 480')
    chunks = []
    for page in pages:
        text = page.text.strip()
        start = 0
        while start < len(text):
            end, size = start, 0
            while end < len(text) and size + len(text[end].encode('utf-8')) <= target:
                size += len(text[end].encode('utf-8'))
                end += 1
            if end == start:
                raise ValueError('Chunk target is smaller than a Unicode character')
            if end < len(text):
                boundary = max(text.rfind(' ', start + (end-start)//2, end),
                               text.rfind('\n', start + (end-start)//2, end))
                if boundary > start:
                    end = boundary
            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(chunk_index=len(chunks), page_start=page.page_number,
                    page_end=page.page_number, content=content,
                    token_count=max(1, math.ceil(len(content.encode('utf-8'))/4))))
            if len(chunks) > 3000:
                raise ServiceError('PDF exceeds 3,000 chunks. Split it into smaller PDFs.', 400)
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    if not chunks:
        raise ServiceError('PDF contains no searchable text.', 400)
    return chunks
