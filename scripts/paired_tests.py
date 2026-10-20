"""Paired tests between final models on the test set (stream A, task A8).

Protocol pre-registered in README.md: paired bootstrap on macro-F1 (1,000 resamples, seed 42, 95 % CI) and McNemar on
accuracy (exact binomial if b + c < 25). For each `--pair A B`, reads results/predictions/<run_id>_<split>.parquet of
both runs, checks that `idx` and `label` are identical and in the same order, and reports ΔF1 = F1(A) − F1(B).
With `--exclude-idx`, each pair gets an extra "deduplicated" row without those idx (train/test near-duplicates, B4).

Writes or updates results/paired_tests.csv (one row per pair, split and variant; earlier rows of the same key are
replaced) and prints a markdown table.

Examples:
    python scripts/paired_tests.py --pair legal-bert_final_20261019-1523 legal-bert_lora-r8_20261020-0940
    python scripts/paired_tests.py --pair <bert_run> <tfidf_run> --pair <bert_run> <lstm_run> --exclude-idx <file>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

import pandas as pd  # noqa: E402

from src.comparison import compare_runs, prediction_path, update_results_csv  # noqa: E402
from src.config import RESULTS_DIR, ROOT  # noqa: E402
from src.utils import markdown_table  # noqa: E402

OUT_PATH = RESULTS_DIR / "paired_tests.csv"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RUN_A", "RUN_B"),
                   help="two run_ids; ΔF1 = F1(A) - F1(B). Repeat for several pairs")
    p.add_argument("--split", default="test", choices=["validation", "test"])
    p.add_argument("--exclude-idx", help="CSV with an `idx` column (or one idx per line): extra deduplicated row")
    return p.parse_args(argv)


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "A": f"`{row['model_a']}`",
        "B": f"`{row['model_b']}`",
        "variant": row["variant"],
        "n": f"{row['n']:,d}",
        "macro-F1 A": f"{row['macro_f1_a']:.4f}",
        "macro-F1 B": f"{row['macro_f1_b']:.4f}",
        "ΔF1 [95 % CI]": f"{row['delta_macro_f1']:+.4f} [{row['ci95_low']:+.4f}; {row['ci95_high']:+.4f}]",
        "significant": "yes" if row["significant"] else "no",
        "accuracy A": f"{row['accuracy_a']:.4f}",
        "accuracy B": f"{row['accuracy_b']:.4f}",
        "b": f"{row['b']:,d}",
        "c": f"{row['c']:,d}",
        "McNemar p": f"{row['mcnemar_p_value']:.2e}",
        "method": row["mcnemar_method"],
    } for row in rows])


def main(argv=None) -> None:
    args = parse_args(argv)
    missing = [str(prediction_path(run_id, args.split).relative_to(ROOT))
               for pair in args.pair for run_id in pair if not prediction_path(run_id, args.split).exists()]
    if missing:
        sys.exit(f"Missing prediction file(s), nothing computed: {', '.join(dict.fromkeys(missing))}")

    rows = [row for run_a, run_b in args.pair for row in compare_runs(run_a, run_b, args.split, args.exclude_idx)]
    update_results_csv(rows, OUT_PATH)
    print(markdown_table(results_table(rows)))
    print(f"\nΔF1 = F1(A) − F1(B); paired bootstrap ({rows[0]['n_boot']:,d} resamples, seed {rows[0]['seed']}); "
          f"McNemar b = A right / B wrong, c = A wrong / B right. Written to {OUT_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
