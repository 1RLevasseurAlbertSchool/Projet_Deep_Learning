"""Fine-tune a Hugging Face encoder on LEDGAR (stream A, tasks A1-A5).

Examples:
    python scripts/train_transformer.py --model distilbert-base-uncased --variant smoke \
        --subset 2000 --epochs 1 --max-length 128
    python scripts/train_transformer.py --model distilbert-base-uncased --variant full

The single test evaluation of the run chosen on validation (A6) reuses its best checkpoint, without retraining:
    python scripts/evaluate_checkpoint.py --source-run <run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

from src.config import MODELS_DIR, SEED  # noqa: E402
from src.models.transformer import DEFAULT_CONFIG, TEST_SET_WARNING, print_metrics, train_and_evaluate  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model name, e.g. distilbert-base-uncased")
    p.add_argument("--variant", default="base", help="run-id label, e.g. smoke / full / final")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup", type=float, default=0.1, help="warmup as a ratio of the total steps")
    p.add_argument("--subset", type=int, default=None, help="first N examples per split (smoke tests only)")
    p.add_argument("--eval-test", action="store_true", help="also predict the test set (final model only, once)")
    p.add_argument("--output-dir", default=str(MODELS_DIR), help="checkpoints folder (gitignored / Drive)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.eval_test:
        print("\n" + TEST_SET_WARNING + "\n")

    config = {
        **DEFAULT_CONFIG,
        "model_name": args.model,
        "variant": args.variant,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "subset": args.subset,
        "eval_test": args.eval_test,
        "output_dir": args.output_dir,
        "seed": args.seed,
    }
    out = train_and_evaluate(config)
    print_metrics(out)
    if args.eval_test:
        print(f"\n!! TEST SET EVALUATED for run_id = {out['run_id']} — log it in PROGRESS.md, do not re-run.")


if __name__ == "__main__":
    main()
