# Agentic RAG with a Self-Measuring Evaluation Harness

A retrieval-augmented question-answering system over a domain corpus that
**grades its own quality** on every change. The point of this project is not
"chat with a PDF" — it is to demonstrate the things hiring teams actually look
for in 2026: clean layered architecture, an *agentic* retrieval loop with
self-correction, and a real evaluation + MLOps pipeline.

It runs **fully offline by default** (deterministic mock providers), so you can
read, run, and understand every layer with zero API keys, then flip a single
environment variable to use real models.

---

## 1. Why this project stands out

Most portfolio RAG projects stop at `retrieve -> generate` in a notebook and
have **no way to tell if they are any good**. This one:

- separa\tes the system into **layers** (domain / repositories / ML / services /
  API) so each part is swappable and testable — the same discipline you'd see
  in a production codebase;
- is **agentic**: it rewrites weak queries, re-retrieves, and self-checks the
  answer for faithfulness before returning it;
- ships a **two-stage retriever** (vector recall → reranker precision), the
  single biggest quality win most projects omit;
- has a **measurable evaluation harness** with retrieval *and* generation
  metrics, logged to MLflow, and an **eval gate in CI** that fails the build if
  quality regresses.

## 2. Architecture (the layered design)

The folder structure mirrors a classic layered / lightweight-DDD app. If you
know MVC, the mapping is direct:

| This project (`src/ragkit/`) | Classic MVC / layered term | Responsibility |
| --- | --- | --- |
| `domain/` | Models / Entities | Pure data objects (`Document`, `Chunk`, `Answer`). No dependencies. |
| `repositories/` | Database / DAO | Persistence — the Qdrant vector store wrapper. |
| `ml/` | (the ML detail layer) | Swappable model components: chunking, embeddings, reranker, LLM. |
| `services/` | Services / Business logic | The workflows: ingestion, retrieval, the agent. |
| `api/` | Controllers | Thin FastAPI layer: HTTP in, DTO out, no logic. |
| `eval/` | (the showpiece) | The self-measuring evaluation harness. |
| `config/` + `observability/` | Cross-cutting | Typed settings and structured logging. |
| `container.py` | Composition root | Wires the object graph from settings (dependency injection). |

The golden rule enforced throughout: **dependencies point inward**. Services
depend on *interfaces* (`Embedder`, `LLM`, `Reranker`), never on OpenAI or
Qdrant directly. That is why the whole thing runs offline and why swapping a
vendor touches exactly one file.

```
agentic-rag-eval/
├── config/settings.py          # typed, env-driven configuration
├── data/
│   ├── corpus/                 # the domain documents (.txt) — swap these
│   └── eval/golden_set.jsonl   # labelled test questions
├── src/ragkit/
│   ├── domain/models.py        # entities
│   ├── repositories/vector_store.py
│   ├── ml/{chunking,embeddings,reranker,llm}.py
│   ├── services/{ingestion,retrieval,agent}.py
│   ├── api/{app,schemas}.py
│   ├── eval/{metrics,runner,report}.py
│   ├── observability/logging.py
│   └── container.py            # composition root
├── scripts/{ingest,evaluate}.py
├── tests/
├── Dockerfile · docker-compose.yml · Makefile
└── .github/workflows/ci.yml
```

## 3. The agent loop

Plain RAG is one straight shot. This agent adds decisions and self-correction
(`services/agent.py`):

```
rewrite query
   → vector search (top_k_vector candidates)
   → rerank (keep top_k_rerank)
   → grade context: sufficient?
        └─ no  → refine query and retry (bounded budget)
   → generate answer with citations
   → self-check faithfulness
        └─ unsupported → regenerate once
   → return Answer + full decision trace
```

Every step is recorded in `Answer.trace`, so you can *see* the reasoning. It is
hand-rolled as an explicit state machine so the control flow is fully visible;
the node functions map 1:1 onto LangGraph nodes when you're ready to lift it
into a framework (see §8).

## 4. The evaluation harness (the part that gets you hired)

RAG fails in two different places, so we measure them separately (`eval/`):

**Retrieval** (did we fetch the right chunks? — scored against labelled
`relevant_doc_ids`, no LLM needed):
`hit_rate@k`, `precision@k`, `recall@k`, `mrr`.

**Generation** (given that context, is the answer good? — LLM-as-judge):
`faithfulness` (anti-hallucination) and `answer_relevance`.

Running `make eval` produces a committed scorecard like:

| Metric | Score |
| --- | --- |
| Hit rate@k | 1.000 |
| Precision@k | 0.438 |
| Recall@k | 1.000 |
| MRR | 1.000 |
| Faithfulness | 1.000 |
| Answer relevance | 1.000 |

> Note: in offline mock mode the embedder is a non-semantic hash, so
> `precision@k` is intentionally mediocre — that's the metric a real embedder +
> cross-encoder reranker visibly improves. Demonstrating that improvement *with
> numbers* is the whole story your README should tell.

## 5. Quickstart (offline, no keys)

```bash
conda create -n agent-rag python=3.14 -y
conda activate agent-rag
pip install -e ".[dev]"     # or: make install
make ingest                 # build the in-memory index from data/corpus
make eval                   # run the harness → reports/scorecard.md
make serve                  # API docs at http://localhost:8000/docs
make test                   # 6 offline tests

#for monitoring
mlflow ui
```



Ask a question once the server is up:

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How long do I have to request a refund?"}'
```

## 6. Going real

Copy `.env.example` to `.env` and set:

```
RAG_LLM_PROVIDER=openai
RAG_EMBEDDING_PROVIDER=openai
RAG_RERANKER_PROVIDER=cross_encoder
RAG_OPENAI_API_KEY=sk-...
RAG_QDRANT_URL=http://localhost:6333     # via docker compose up
RAG_MLFLOW_TRACKING_URI=http://localhost:5000
```

Install the extras: `pip install -e ".[openai,rerank,mlops]"

```bash
docker compose up -d
```


## 7. MLOps practices demonstrated

- **Config as data** — one typed `Settings`, zero hard-coded values.
- **Dependency injection** via a composition root; layers depend on interfaces.
- **Structured logging** through a single configured logger.
- **Experiment tracking** — every eval run logs params + metrics to MLflow.
- **Testing** — unit tests for pure logic + an offline end-to-end integration test.
- **CI with an eval gate** — PRs fail if retrieval quality drops below threshold.
- **Containerised infra** — `docker-compose` brings up Qdrant + MLflow + API.
- **Reproducibility** — mock providers make the pipeline deterministic.

## 8. Extension ideas (turn it into a portfolio centerpiece)

1. Swap the corpus for a domain you care about (legal, medical, EU regulation).
2. Replace the hash embedder with a real one and **show the precision lift** in
   the scorecard — that before/after is your headline.
3. Port `services/agent.py` to **LangGraph**; the nodes already map cleanly.
4. Add a **drift check**: re-run eval on a schedule, alert on metric regressions.
5. Build a tiny front end that renders the answer *and the trace*, so reviewers
   see the agent's reasoning.

---

Built as a learning project. Everything is intentionally readable — start at
`container.py` to see how the layers connect, then follow a request through
`api/app.py → services/agent.py → services/retrieval.py`.
