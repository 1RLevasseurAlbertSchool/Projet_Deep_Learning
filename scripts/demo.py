"""Command-line demo of the clause-review assistant (bonus task T4). No training.

Review a clause (top-3 clause types, automatic / send to a lawyer, 5 closest training clauses):
    python scripts/demo.py "This Agreement shall be governed by the laws of the State of New York."

Search the training clause bank with a lawyer's free-text question (retrieval only):
    python scripts/demo.py --question "termination clauses with a 30-day notice period"

Needs the final checkpoint in models/legal-bert_final/ (the error message says which files to download from Drive).
The first run builds and caches the retrieval index over the 60,000 training clauses (data/indexes/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

from src.pipeline import DEFAULT_CHECKPOINT_DIR, ClauseReviewer, format_hits, format_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("clause", nargs="?", help="clause text to review")
    group.add_argument("--question", help="lawyer's free-text question (retrieval only)")
    parser.add_argument("-k", type=int, default=5, help="number of similar training clauses (default 5)")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    try:
        reviewer = ClauseReviewer.from_local(args.checkpoint_dir)
    except FileNotFoundError as error:
        sys.exit(str(error))
    print()
    if args.question:
        print(format_hits(reviewer.search(args.question, k=args.k)))
    else:
        print(format_review(reviewer.review(args.clause, k=args.k)))


if __name__ == "__main__":
    main()
