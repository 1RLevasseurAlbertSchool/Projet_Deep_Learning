"""Calibration and human-review threshold of a final model (stream A, task A9).

Reads results/predictions/<run_id>_{validation,test}.parquet (`label`, `pred`, `proba_pred`) and measures top-label
calibration (ECE, reliability diagram) on both splits. For each target precision, the threshold is CHOSEN ON
VALIDATION (smallest threshold whose accuracy on the clauses handled automatically reaches the target), then coverage
and precision are REPORTED ON TEST with that fixed threshold.

The prediction files hold only `proba_pred` and `top5`, not the full probability matrix: no temperature scaling here.

Writes results/calibration_<run_id>.json, results/figures/reliability_<run_id>.png and
results/figures/coverage_<run_id>.png.

Example:
    python scripts/calibration_report.py --run-id legal-bert_final_20261019-1523 --target-precision 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

import pandas as pd  # noqa: E402

from src.calibration import calibration_report, coverage_precision_figure, reliability_diagram  # noqa: E402
from src.config import FIGURES_DIR, PREDICTIONS_DIR, RESULTS_DIR, ROOT  # noqa: E402
from src.utils import markdown_table  # noqa: E402

SPLITS = ("validation", "test")
METHOD = ("top-label calibration on proba_pred (probability of the predicted class); no temperature scaling: the "
          "prediction files hold only proba_pred and top5, not the full probability matrix")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="final run with validation and test prediction files")
    p.add_argument("--target-precision", type=float, default=0.95, help="recommended operating point")
    p.add_argument("--also-report", type=float, nargs="*", default=[0.90, 0.98],
                   help="other target precisions written to the JSON, to show the trade-off")
    p.add_argument("--n-bins", type=int, default=15, help="equal-width confidence bins for the ECE")
    return p.parse_args(argv)


def _fmt(value, spec: str) -> str:
    return "not reached" if value is None else format(value, spec)


def operating_points_table(points: list[dict]) -> pd.DataFrame:
    rows = []
    for point in points:
        val, test = point["validation"] or {}, point["test"] or {}
        ci = test.get("precision_ci95")
        rows.append({
            "target precision": f"{point['target_precision']:.2f}",
            "threshold (val.)": _fmt(point["threshold"], ".3f"),
            "val. coverage": _fmt(val.get("coverage"), ".1%"),
            "val. precision": _fmt(val.get("precision"), ".4f"),
            "test coverage": _fmt(test.get("coverage"), ".1%"),
            "test precision [95 % CI]": "not reached" if ci is None
            else f"{test['precision']:.4f} [{ci[0]:.4f}; {ci[1]:.4f}]",
            "test clauses reviewed": _fmt(test.get("n_reviewed"), ",d"),
            "test errors sent to review": _fmt(test.get("share_of_errors_reviewed"), ".1%"),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = {split: PREDICTIONS_DIR / f"{args.run_id}_{split}.parquet" for split in SPLITS}
    missing = [str(path.relative_to(ROOT)) for path in paths.values() if not path.exists()]
    if missing:
        sys.exit(f"Missing prediction file(s): {', '.join(missing)}")
    preds = {split: pd.read_parquet(path) for split, path in paths.items()}

    targets = sorted({args.target_precision, *args.also_report})
    report = calibration_report(preds["validation"], preds["test"], targets, n_bins=args.n_bins)
    report = {"run_id": args.run_id, "method": METHOD, "recommended_target_precision": args.target_precision,
              **report}
    out_path = RESULTS_DIR / f"calibration_{args.run_id}.json"
    out_path.write_text(json.dumps(report, indent=2))

    arrays = {split: (df["proba_pred"].to_numpy(), (df["pred"] == df["label"]).to_numpy())
              for split, df in preds.items()}
    reliability_path = FIGURES_DIR / f"reliability_{args.run_id}.png"
    coverage_path = FIGURES_DIR / f"coverage_{args.run_id}.png"
    reliability_diagram(arrays, reliability_path, n_bins=args.n_bins, title=f"Reliability diagram — {args.run_id}")
    coverage_precision_figure(arrays, report["operating_points"], coverage_path,
                              title=f"Coverage vs precision — {args.run_id}")

    summary = pd.DataFrame([{"split": split, "clauses": f"{report[split]['n']:,d}",
                             "accuracy": f"{report[split]['accuracy']:.4f}",
                             "mean confidence": f"{report[split]['mean_confidence']:.4f}",
                             "confidence − accuracy": f"{report[split]['confidence_minus_accuracy']:+.4f}",
                             f"ECE ({args.n_bins} bins)": f"{report[split]['ece']:.4f}"} for split in SPLITS])
    print(markdown_table(summary))
    print()
    print(markdown_table(operating_points_table(report["operating_points"])))
    print(f"\nThresholds chosen on validation, applied as is to test. Recommended: target precision "
          f"{args.target_precision:.2f}.")
    for path in (out_path, reliability_path, coverage_path):
        print(f"Written: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
