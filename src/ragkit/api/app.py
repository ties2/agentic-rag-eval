"""API layer (Controllers).

Thin: parse the request, call a service, map the result to a DTO, return it.
No business logic lives here — that is the whole point of the layering. The
container is built once at startup and shared.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from ragkit.api.schemas import AnswerResponse, AskRequest, ContextDTO
from ragkit.container import Container, build_container
_container: Container | None = None

from pathlib import Path
from fastapi.responses import FileResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _container
    _container = build_container()
    yield

app = FastAPI(title="Agentic RAG + Eval", version="0.1.0", lifespan=lifespan)

_STATIC = Path(__file__).parent / "static"

@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")
@app.get("/health")
def health() -> dict:
    assert _container is not None
    return {"status": "ok", "indexed_chunks": _container.store.count(),
            "orchestrator": _container.settings.agent_provider,
            }


@app.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest) -> AnswerResponse:
    assert _container is not None
    ans = _container.agent.answer(req.question)
    return AnswerResponse(
        question=ans.question,
        answer=ans.text,
        citation_status=ans.citation_status.value,
        contexts=[
            ContextDTO(
                chunk_id=c.chunk.chunk_id,
                doc_id=c.chunk.doc_id,
                score=round(c.score, 4),
                text=c.chunk.text,
            )
            for c in ans.contexts
        ],
        trace=ans.trace,
    )
