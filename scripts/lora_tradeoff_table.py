"""Full fine-tuning vs LoRA trade-off table (stream A, task A7).

Reads results/runs/: the full fine-tuning run given with --full-run and every LoRA run of the same encoder on the
full data. Writes results/lora_tradeoff.csv and prints a markdown table (trainable parameters, % of full, training
time, peak GPU memory, validation and test macro-F1 and accuracy). Differences are not tested here: paired tests
are task A8 (scripts/paired_tests.py).

Example:
    python scripts/lora_tradeoff_table.py --full-run legal-bert_final_20261019-1523
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

from src.config import RESULTS_DIR  # noqa: E402
from src.models.transformer import lora_tradeoff_table  # noqa: E402
from src.utils import markdown_table  # noqa: E402

OUT_PATH = RESULTS_DIR / "lora_tradeoff.csv"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full-run", required=True, help="run_id of the full fine-tuning final model (with test metrics)")
    return p.parse_args(argv)


def _fmt(value, spec: str) -> str:
    return "not run" if value is None or value != value else format(value, spec)  # NaN != NaN


def main(argv=None) -> None:
    args = parse_args(argv)
    df = lora_tradeoff_table(args.full_run)
    df.to_csv(OUT_PATH, index=False)

    shown = df.copy()
    shown["run_id"] = "`" + shown["run_id"] + "`"
    for col, spec in [("lr", "g"), ("trainable_params", ",d"), ("pct_of_full", ".2f"), ("train_time_s", ",.1f"),
                      ("peak_gpu_memory_mb", ",.1f"), ("val_macro_f1", ".4f"), ("val_accuracy", ".4f"),
                      ("test_macro_f1", ".4f"), ("test_accuracy", ".4f")]:
        shown[col] = [_fmt(v, spec) for v in shown[col]]
    print(markdown_table(shown.drop(columns="r")))
    print(f"\n{len(df) - 1} LoRA run(s). Written to {OUT_PATH.relative_to(RESULTS_DIR.parent)}. "
          "Differences are not tested here: see the paired tests (A8).")


if __name__ == "__main__":
    main()
