"""CLI: run the evaluation harness and write a scorecard.

    python scripts/evaluate.py

In ":memory:" mode the index does not persist between processes, so we ingest
first, then evaluate, in one run.
"""

from __future__ import annotations

from ragkit.container import build_container
from ragkit.eval.report import write_report
from ragkit.eval.runner import run_eval


def main() -> None:
    c = build_container()
    if c.store.count() == 0:
        print("Index empty — ingesting corpus first...")
        c.ingestion.run()

    results = run_eval(c, c.settings)
    path = write_report(results, c.settings.eval_report_dir)

    agg = results["aggregate"]
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k:18s}: {v}")
    print(f"\nScorecard written to {path}")


if __name__ == "__main__":
    main()
