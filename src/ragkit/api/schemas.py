"""API DTOs — the contract between HTTP and the domain.

We never expose domain objects directly over HTTP. DTOs let the API shape
evolve independently of internal models and give FastAPI clean OpenAPI docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What triggers a refund?"])


class ContextDTO(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citation_status: str
    contexts: list[ContextDTO]
    trace: list[str]
