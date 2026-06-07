"""CLI: build the vector index from the corpus.

    python scripts/ingest.py
"""

from __future__ import annotations

from ragkit.container import build_container


def main() -> None:
    c = build_container()
    n = c.ingestion.run()
    print(f"Ingested {n} chunks. Collection now holds {c.store.count()} points.")


if __name__ == "__main__":
    main()
