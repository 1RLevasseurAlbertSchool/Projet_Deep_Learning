"""Token-length distribution of the LEDGAR training clauses for a given tokenizer (task A3: choice of max_length).

Example:
    python scripts/token_lengths.py --model distilbert-base-uncased
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

from src.config import TEXT_COL  # noqa: E402
from src.data import load_ledgar  # noqa: E402
from src.models.transformer import token_length_stats  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model name, e.g. distilbert-base-uncased")
    args = p.parse_args(argv)

    ds = load_ledgar()
    stats = token_length_stats(args.model, ds["train"][TEXT_COL])
    print(f"Token lengths (special tokens included) — {args.model}, train split")
    for key, value in stats.items():
        if key.startswith("share_gt_"):
            print(f"  {'> ' + key.removeprefix('share_gt_') + ' tokens':<14}{value:>8.2%}")
        else:
            print(f"  {key:<14}{value:>8,.0f}" if key in ("n", "max") else f"  {key:<14}{value:>8.1f}")


if __name__ == "__main__":
    main()
