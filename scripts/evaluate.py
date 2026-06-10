"""CLI: run the evaluation harness and write a scorecard.

    python scripts/evaluate.py        # evaluate ALL questions
    python scripts/evaluate.py 15     # evaluate a random sample of 15

In ":memory:" mode the index does not persist between processes, so we ingest
first, then evaluate, in one run.
"""

from __future__ import annotations
import pathlib
import sys
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from ragkit.container import build_container
from ragkit.eval.report import write_report
from ragkit.eval.runner import run_eval


def main() -> None:
    # Optional first CLI arg = how many questions to sample (for fast iteration).
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None

    c = build_container()
    if c.store.count() == 0:
        print("Index empty — ingesting corpus first...")
        c.ingestion.run()

    if sample:
        print(f"Evaluating a random sample of {sample} question(s)...")
    results = run_eval(c, c.settings, sample=sample)
    path = write_report(results, c.settings.eval_report_dir)

    agg = results["aggregate"]
    print("\n=== Aggregate ===")
    for k, v in agg.items():
        print(f"  {k:18s}: {v}")
    print(f"\nScorecard written to {path}")


if __name__ == "__main__":
    main()