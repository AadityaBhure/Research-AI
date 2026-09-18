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
    valyu_api_key: SecretStr = SecretStr('')
    zilliz_api_key: SecretStr = SecretStr('')
    zilliz_public_endpoint: str = ''
    zilliz_embedding_integration_id: SecretStr = SecretStr('')
    zilliz_collection: str = Field(default='research_chunks_cohere_v1', pattern=r'^[A-Za-z_][A-Za-z0-9_]{0,254}$')
    zilliz_embedding_model: str = 'embed-english-v3.0'
    vercel: bool = False
    ingestion_timeout_seconds: int = Field(default=240, ge=30, le=240)
    rag_top_k: int = Field(default=6, ge=1, le=20)
    cors_origins: list[str] = ['http://localhost:5173', 'http://127.0.0.1:5173']
    max_pdf_bytes: int = 30 * 1024 * 1024
    storage_bucket: str = 'research-papers'

    @property
    def max_upload_bytes(self) -> int:
        # Leave room for multipart framing under Vercel's 4.5 MB request limit.
        return min(self.max_pdf_bytes, 4_000_000) if self.vercel else self.max_pdf_bytes

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

    @property
    def embedding_model(self) -> str:
        return f'zilliz:cohere:{self.zilliz_embedding_model}:hybrid-v1'

    @field_validator('zilliz_embedding_model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value != 'embed-english-v3.0':
            raise ValueError('The Zilliz collection requires embed-english-v3.0')
        return value

    @field_validator('zilliz_public_endpoint')
    @classmethod
    def validate_zilliz_endpoint(cls, value: str) -> str:
        from urllib.parse import urlsplit
        if not value:
            return value
        url = urlsplit(value)
        if (url.scheme != 'https' or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path not in ('', '/')
                or not url.hostname.endswith(('.zilliz.com', '.zillizcloud.com'))):
            raise ValueError('ZILLIZ_Public_Endpoint must be an HTTPS Zilliz cluster endpoint')
        return value.rstrip('/')
