# Project Document Agentic RAG with a Self-Measuring Evaluation Harness
---

## 1. One-line summary

A retrieval-augmented question-answering system over a domain corpus that
**measures its own answer quality** on every change, built with a clean layered
architecture and a real evaluation + MLOps pipeline.

## 2. Problem statement

Organisations hold large bodies of text (policies, regulations, manuals) that
people need answers from. Plain keyword search returns documents, not answers;
naive LLM chatbots invent facts ("hallucinate") and can't cite sources.

This project answers natural-language questions **grounded in a specific
corpus**, returns the supporting passages, and critically **proves how good
it is with metrics** rather than vibes. The evaluation layer is the
differentiator: most RAG projects have no way to know if a change helped or hurt.

## 3. Goals and non-goals

**Goals**
- Answer questions using only the provided corpus, with citations.
- Detect and reduce hallucination via a faithfulness self-check.
- Provide a two-stage retriever (vector recall → reranker precision).
- Make the agent's reasoning visible through a step trace.
- Ship a measurable evaluation harness (retrieval + generation metrics).
- Demonstrate MLOps fundamentals: config, logging, tests, tracking, CI, Docker.
- Run **fully offline by default** so it is reproducible and easy to learn.

**Non-goals (explicitly out of scope, for now)**
- Multi-user auth, accounts, or a production-grade UI.
- Real-time document ingestion / streaming updates.
- Fine-tuning a model (this project uses retrieval, not training).
- Handling images, audio, or tables inside documents (text only for v1).

## 4. What the system does (functional description)

### 4.1 Ingestion (offline / batch)
1. Load `.txt` documents from `data/corpus/`.
2. Split each into overlapping chunks (recursive: paragraph → sentence → hard cut).
3. Embed each chunk into a vector.
4. Store vectors + payloads in the vector database.

### 4.2 Answering a question (online / agentic)
1. **Rewrite** the question into a denser search query.
2. **Vector search** for the top-K candidate chunks (high recall).
3. **Rerank** candidates with a more precise model; keep the best few.
4. **Grade** whether the retrieved context is sufficient. If weak, refine the
   query and retry, up to a bounded budget.
5. **Generate** an answer using only the context, citing chunk IDs.
6. **Self-check faithfulness** — is every claim supported by the context?
   If not, regenerate once; otherwise abstain.
7. Return the answer, the supporting passages, a citation status
   (`grounded` / `partial` / `unsupported`), and the full decision **trace**.

### 4.3 Evaluation (on demand / in CI)
1. Load a labelled golden set of questions (`data/eval/golden_set.jsonl`).
2. Run each question through the agent.
3. Score **retrieval** and **generation** separately.
4. Aggregate, log the run (MLflow or JSON), and write a markdown scorecard.

### 4.4 API
- `GET /health` — liveness + number of indexed chunks.
- `POST /ask` — `{ "question": "..." }` → answer, contexts, status, trace.

## 5. Architecture

Layered design (lightweight DDD). **Dependencies point inward**: business logic
depends on interfaces, never on vendor SDKs.

| Layer (`src/ragkit/`) | Role | Key files |
| --- | --- | --- |
| `domain/` | Pure entities (no deps) | `models.py` |
| `repositories/` | Persistence (vector DB) | `vector_store.py` |
| `ml/` | Swappable model parts | `chunking.py`, `embeddings.py`, `reranker.py`, `llm.py` |
| `services/` | Business workflows | `ingestion.py`, `retrieval.py`, `agent.py` |
| `api/` | HTTP controllers (thin) | `app.py`, `schemas.py` |
| `eval/` | Evaluation harness | `metrics.py`, `runner.py`, `report.py` |
| `config/`, `observability/` | Cross-cutting | `settings.py`, `logging.py` |
| `container.py` | Composition root (DI) | wires everything from settings |

Request flow: `api/app.py → services/agent.py → services/retrieval.py →
repositories/vector_store.py` (+ `ml/*` components throughout).

## 6. Data

- **Corpus** (`data/corpus/*.txt`): the source documents. v1 ships a small
  fintech-policy demo (refunds, disputes, KYC, fees). **Swap this for your real
  target domain.**
- **Golden set** (`data/eval/golden_set.jsonl`): one JSON object per line:
  ```json
  {"question": "...", "ground_truth": "...", "relevant_doc_ids": ["doc_id"]}
  ```
  `relevant_doc_ids` lets retrieval be scored independently of generation.

## 7. Evaluation methodology

RAG fails in two distinct places; measure them separately.

**Retrieval metrics** (no LLM; uses `relevant_doc_ids`)
- `hit_rate@k` — was at least one relevant doc retrieved?
- `precision@k` — fraction of retrieved docs that are relevant.
- `recall@k` — fraction of relevant docs that were retrieved.
- `mrr` — 1 / rank of the first relevant doc (rewards good ranking).

**Generation metrics** (LLM-as-judge)
- `faithfulness` — is every claim supported by the context? (anti-hallucination)
- `answer_relevance` — does the answer actually address the question?

**Acceptance targets** (tune to your corpus):
`hit_rate ≥ 0.9`, `faithfulness ≥ 0.9`, `precision@k` trending up as you improve
embeddings/reranking. CI fails if `hit_rate` drops below 0.5.

## 8. Tech stack

- **Language**: Python 3.11
- **API**: FastAPI + Uvicorn
- **Vector DB**: Qdrant (in-memory by default; container in compose)
- **Config**: pydantic-settings
- **Embeddings / LLM**: OpenAI (real) or deterministic mock (offline)
- **Reranker**: sentence-transformers cross-encoder (real) or lexical mock
- **Tracking**: MLflow (optional)
- **Quality**: pytest, ruff, mypy, pre-commit
- **Infra**: Docker, docker-compose, GitHub Actions

## 9. Configuration

All behaviour is driven by env vars (`RAG_` prefix) via `config/settings.py`.
Copy `.env.example` → `.env`. Safe mock defaults mean it runs with an empty
`.env`. Key switches: `RAG_LLM_PROVIDER`, `RAG_EMBEDDING_PROVIDER`,
`RAG_RERANKER_PROVIDER`, `RAG_QDRANT_URL`, `RAG_MLFLOW_TRACKING_URI`,
`RAG_TOP_K_VECTOR`, `RAG_TOP_K_RERANK`, `RAG_CHUNK_SIZE`.

## 10. How to run

```bash
make install     # pip install -e ".[dev]"
make ingest      # build the index from data/corpus
make eval        # run the harness → reports/scorecard.md
make serve       # API docs at http://localhost:8000/docs
make test        # offline test suite
make up          # docker: qdrant + mlflow + api
```

## 11. Testing strategy

- **Unit**: pure logic — chunking, retrieval metrics.
- **Integration**: full agent loop end-to-end in offline mock mode (no keys).
- **Eval gate**: the evaluation harness runs in CI and fails on quality
  regression — so every PR reports model quality, not just whether code imports.

## 12. MLOps practices demonstrated

Config-as-data · dependency injection via composition root · structured logging ·
experiment tracking (MLflow) · unit + integration tests · CI with an eval gate ·
containerised infra · deterministic/reproducible runs.

## 13. Milestones / roadmap

- [x] M1 — Layered skeleton + domain models + config.
- [x] M2 — Ingestion pipeline (load → chunk → embed → store).
- [x] M3 — Two-stage retrieval (rewrite → search → rerank).
- [x] M4 — Agent loop (grade → generate → self-check).
- [x] M5 — Evaluation harness + scorecard + CI gate.
- [ ] M6 — Swap in your real domain corpus + write its golden set.
- [ ] M7 — Real embedder + cross-encoder; record precision@k before/after.
- [ ] M8 — Port agent to LangGraph; add a trace-visualising UI.
- [ ] M9 — Scheduled eval + drift alerting.

## 14. Success criteria

1. Runs end-to-end offline with one command (`make eval`).
2. Produces a committed scorecard with both retrieval and generation metrics.
3. Demonstrates a measurable quality improvement (M7 before/after).
4. README + this document let a reviewer understand the system in minutes.

## 15. Glossary

- **RAG** — Retrieval-Augmented Generation: fetch relevant text, then have an
  LLM answer using it.
- **Chunk** — a small, retrievable slice of a document.
- **Embedding** — a vector representation of text for similarity search.
- **Reranker** — a precise model that re-scores retrieved candidates.
- **Faithfulness** — whether an answer's claims are supported by its context.
- **Golden set** — labelled test questions used to measure quality.
- **LLM-as-judge** — using an LLM to score outputs against criteria.
