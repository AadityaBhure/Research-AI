from uuid import UUID
from datetime import date

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default='', max_length=5000)

    @field_validator('name')
    @classmethod
    def strip_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Name cannot be blank')
        return value.strip()


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=5000)

    @field_validator('name', 'description')
    @classmethod
    def reject_null(cls, value):
        if value is None:
            raise ValueError('Provided fields cannot be null')
        return value

    @field_validator('name')
    @classmethod
    def strip_name(cls, value):
        if not value.strip():
            raise ValueError('Name cannot be blank')
        return value.strip()


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    limit: int = Field(default=10, ge=1, le=10)


class PaperInput(BaseModel):
    model_config = ConfigDict(extra='ignore')
    semantic_scholar_paper_id: str | None = Field(default=None, max_length=100)
    title: str = Field(min_length=1, max_length=1000)
    authors: list[str] = Field(default_factory=list, max_length=5000)
    abstract: str | None = Field(default=None, max_length=30000)
    publication_year: int | None = Field(default=None, ge=1000, le=3000)
    publication_date: date | None = None
    venue: str | None = Field(default=None, max_length=1000)
    citation_count: int | None = Field(default=None, ge=0)
    doi: str | None = Field(default=None, max_length=500)
    arxiv_id: str | None = Field(default=None, max_length=200)
    source_url: HttpUrl | None = None
    pdf_url: HttpUrl | None = None

    @field_validator('source_url', 'pdf_url', 'publication_date', 'doi', 'arxiv_id', 'semantic_scholar_paper_id', mode='before')
    @classmethod
    def normalize_missing_optional_values(cls, value):
        # Scholarly APIs sometimes use empty strings rather than null for missing links.
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator('title')
    @classmethod
    def strip_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('Title cannot be blank')
        return value.strip()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    paper_ids: list[UUID] | None = Field(default=None, max_length=20)


class ExtractedPage(BaseModel):
    page_number: int
    text: str


class Chunk(BaseModel):
    chunk_index: int
    page_start: int
    page_end: int
    content: str
    token_count: int
