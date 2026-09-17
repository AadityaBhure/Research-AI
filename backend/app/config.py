from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / '.env', extra='ignore'
    )
    supabase_url: str
    supabase_secret_key: SecretStr
    groq_api_key: SecretStr
    groq_model: str = 'openai/gpt-oss-20b'
    semantic_scholar_api_key: SecretStr = SecretStr('')
    embedding_model: str = 'BAAI/bge-base-en-v1.5'
    embedding_dimension: int = 768
    rag_top_k: int = Field(default=6, ge=1, le=20)
    rag_min_similarity: float = Field(default=0.35, ge=-1, le=1)
    cors_origins: list[str] = ['http://localhost:5173', 'http://127.0.0.1:5173']
    max_pdf_bytes: int = 30 * 1024 * 1024
    storage_bucket: str = 'research-papers'

    @field_validator('supabase_url')
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit
        url = urlsplit(value)
        if url.scheme != 'https' or not url.hostname or url.username or url.query or url.fragment:
            raise ValueError('SUPABASE_URL must be an HTTPS project URL')
        return value.rstrip('/')

    @field_validator('supabase_secret_key', 'groq_api_key')
    @classmethod
    def validate_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError('A required API key is missing')
        return value

    @field_validator('embedding_model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value != 'BAAI/bge-base-en-v1.5':
            raise ValueError('Changing embeddings requires a matching tokenizer and reindex migration')
        return value

    @field_validator('embedding_dimension')
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value != 768:
            raise ValueError('The V1 schema and BGE model require 768 dimensions')
        return value
