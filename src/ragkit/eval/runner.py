"""Eval: runner — executes the golden set and aggregates metrics.

This turns "I think retrieval got better" into "precision_at_4 went 0.61 -> 0.78,
faithfulness 0.80 -> 0.92, logged as MLflow run abc123". That measurability is
the MLOps mindset and the thing that separates this project from the pack.

MLflow is optional: if no tracking URI is set we fall back to a JSON dump, so
the harness always runs.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from config.settings import Settings
from ragkit.container import Container
from ragkit.domain.models import EvalCase
from ragkit.eval import metrics as M
from ragkit.ml.llm import build_llm
from ragkit.observability.logging import get_logger

log = get_logger("eval.runner")


def load_golden_set(path: str) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cases.append(
            EvalCase(
                question=row["question"],
                ground_truth=row.get("ground_truth", ""),
                relevant_doc_ids=row.get("relevant_doc_ids", []),
            )
        )
    return cases


def run_eval(container: Container, settings: Settings) -> dict:
    cases = load_golden_set(settings.golden_set_path)
    judge = build_llm(settings)  # judge can differ from generator if you want
    per_case = []

    t0 = time.time()
    for case in cases:
        ans = container.agent.answer(case.question)
        row = {
            "question": case.question,
            "status": ans.citation_status.value,
            "hit_rate": M.hit_rate(ans.contexts, case.relevant_doc_ids),
            "precision_at_k": M.precision_at_k(ans.contexts, case.relevant_doc_ids),
            "recall_at_k": M.recall_at_k(ans.contexts, case.relevant_doc_ids),
            "mrr": M.mrr(ans.contexts, case.relevant_doc_ids),
            "faithfulness": M.faithfulness(ans, judge),
            "answer_relevance": M.answer_relevance(ans, judge),
        }
        per_case.append(row)
    elapsed = time.time() - t0

    metric_keys = [
        "hit_rate", "precision_at_k", "recall_at_k", "mrr",
        "faithfulness", "answer_relevance",
    ]
    aggregate = {
        k: round(statistics.mean(r[k] for r in per_case), 4) for k in metric_keys
    }
    aggregate["n_cases"] = len(per_case)
    aggregate["seconds"] = round(elapsed, 2)

    _log_run(settings, aggregate)
    return {"aggregate": aggregate, "per_case": per_case}


def _log_run(settings: Settings, aggregate: dict) -> None:
    if settings.mlflow_tracking_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment("agentic-rag-eval")
            with mlflow.start_run():
                mlflow.log_params(
                    {
                        "llm_provider": settings.llm_provider,
                        "embedding_provider": settings.embedding_provider,
                        "reranker_provider": settings.reranker_provider,
                        "top_k_vector": settings.top_k_vector,
                        "top_k_rerank": settings.top_k_rerank,
                        "chunk_size": settings.chunk_size,
                    }
                )
                mlflow.log_metrics(
                    {k: v for k, v in aggregate.items() if isinstance(v, (int, float))}
                )
            log.info("Logged eval run to MLflow at %s", settings.mlflow_tracking_uri)
            return
        except Exception as exc:  # noqa: BLE001 - never let logging crash eval
            log.warning("MLflow logging failed (%s); falling back to JSON", exc)

    Path(settings.eval_report_dir).mkdir(parents=True, exist_ok=True)
    out = Path(settings.eval_report_dir) / "last_run.json"
    out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    log.info("Wrote aggregate metrics to %s", out)
