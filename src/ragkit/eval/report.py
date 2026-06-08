"""Eval: report — turn raw numbers into a markdown scorecard.

A reviewer should grasp your system's quality in 10 seconds. This writes a
report you can commit to the repo and screenshot in your README, which is what
makes the evaluation work *visible* to a recruiter.
"""

from __future__ import annotations

from pathlib import Path


def render_markdown(results: dict) -> str:
    agg = results["aggregate"]
    lines = [
        "# RAG Evaluation Scorecard",
        "",
        f"- Cases: **{agg['n_cases']}**  |  Wall time: **{agg['seconds']}s**",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Hit rate_at_k | {agg['hit_rate']:.3f} |",
        f"| Precision_at_k | {agg['precision_at_k']:.3f} |",
        f"| Recall_at_k | {agg['recall_at_k']:.3f} |",
        f"| MRR | {agg['mrr']:.3f} |",
        f"| Faithfulness | {agg['faithfulness']:.3f} |",
        f"| Answer relevance | {agg['answer_relevance']:.3f} |",
        "",
        "## Per-question",
        "",
        "| Question | Status | P_at_k | Recall | MRR | Faith | Rel |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results["per_case"]:
        q = (r["question"][:50] + "…") if len(r["question"]) > 50 else r["question"]
        lines.append(
            f"| {q} | {r['status']} | {r['precision_at_k']:.2f} | "
            f"{r['recall_at_k']:.2f} | {r['mrr']:.2f} | "
            f"{r['faithfulness']:.0f} | {r['answer_relevance']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(results: dict, report_dir: str) -> Path:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / "scorecard.md"
    path.write_text(render_markdown(results), encoding="utf-8")
    return path
