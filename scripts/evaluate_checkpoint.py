"""Single test evaluation of the run chosen on validation, from its best checkpoint (stream A, task A6).

No retraining: the best checkpoint saved by scripts/train_transformer.py is reloaded and predicts validation and
test with the same tokenisation. Writes a new run `<model>_final_<timestamp>` whose config.json records
`source_run` and `best_checkpoint`. Refuses to run if the test set was already evaluated for this run.

Example:
    python scripts/evaluate_checkpoint.py --source-run legal-bert_full_20261020-1430
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

from src.models.transformer import (  # noqa: E402
    TEST_SET_WARNING,
    RepeatedTestEvaluationError,
    evaluate_checkpoint,
    print_metrics,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-run", required=True, help="run_id chosen on validation (results/runs/<run_id>/)")
    p.add_argument("--force", action="store_true",
                   help="evaluate the test set again although it was already evaluated for this run (logged)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    print("\n" + TEST_SET_WARNING + "\n")
    try:
        out = evaluate_checkpoint(args.source_run, force=args.force)
    except RepeatedTestEvaluationError as e:
        sys.exit(str(e))
    print_metrics(out)
    print(f"source run {args.source_run}: validation macro-F1 "
          f"{out['source_metrics']['validation']['macro_f1']:.4f} (same checkpoint: should match the line above)")
    print(f"\n!! TEST SET EVALUATED for run_id = {out['run_id']} — log it in PROGRESS.md, do not re-run.")


if __name__ == "__main__":
    main()
