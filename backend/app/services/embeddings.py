import threading

import numpy as np

from app.models import Chunk
from app.services.errors import ServiceError


class EmbeddingService:
    """Lazy CPU model; no embedding API key or external document transfer."""

    def __init__(self, settings):
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_pretrained(self.settings.embedding_model)
            self._tokenizer.no_truncation()
            self._model = TextEmbedding(model_name=self.settings.embedding_model, threads=2)

    def chunk_pages(self, pages) -> list[Chunk]:
        with self._lock:
            self._load()
            return chunk_pages(pages, self._tokenizer)

    def documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            self._load()
            return self._validate(list(self._model.passage_embed(texts, batch_size=16)))

    def query(self, text: str) -> list[float]:
        with self._lock:
            self._load()
            if len(self._tokenizer.encode(text).ids) > 480:
                raise ServiceError('Please shorten your question to fewer than 480 model tokens.', 400)
            return self._validate(list(self._model.query_embed(text)))[0]

    def _validate(self, vectors) -> list[list[float]]:
        result = []
        for vector in vectors:
            v = np.asarray(vector, dtype=np.float32)
            if v.shape != (self.settings.embedding_dimension,) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
                raise ServiceError('The embedding model returned an invalid vector.')
            result.append((v / np.linalg.norm(v)).tolist())
        return result


def chunk_pages(pages, tokenizer, target: int = 450, overlap: int = 64) -> list[Chunk]:
    """Page-local chunks, exact model token counts, whole words and bounded overlap.

    BGE's 512-token window is smaller than the original Gemini specification.
    Reserving token headroom avoids truncation inside the embedding model.
    """
    chunks = []
    for page in pages:
        words = page.text.split()
        start = 0
        while start < len(words):
            low, high = start + 1, len(words)
            end = start
            while low <= high:
                mid = (low + high) // 2
                count = len(tokenizer.encode(' '.join(words[start:mid])).ids)
                if count <= target:
                    end, low = mid, mid + 1
                else:
                    high = mid - 1
            if end == start:
                raise ServiceError('PDF contains an oversized text token that cannot be embedded safely.', 400)
            content = ' '.join(words[start:end])
            chunks.append(Chunk(chunk_index=len(chunks), page_start=page.page_number,
                                page_end=page.page_number, content=content,
                                token_count=len(tokenizer.encode(content).ids)))
            if len(chunks) > 3000:
                raise ServiceError('PDF exceeds the V1 limit of 3,000 chunks.', 400)
            if end == len(words):
                break
            next_start = end
            while next_start > start + 1 and len(tokenizer.encode(' '.join(words[next_start - 1:end])).ids) <= overlap:
                next_start -= 1
            start = next_start
    return chunks
