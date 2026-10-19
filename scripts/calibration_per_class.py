"""Per-class view of the human-review threshold of a final model (stream A, task A9).

Reads the recommended threshold from results/calibration_<run_id>.json (CHOSEN ON VALIDATION by
scripts/calibration_report.py) and applies it as is to the TEST predictions. For each of the 100 classes: support,
share of its clauses labelled automatically, and precision of the automatic labels of that class (per predicted class,
95 % Wilson interval); classes with fewer than --min-auto automatic clauses are flagged as unreliable. Classes are
grouped head / middle / tail by terciles of their training-set frequency (src.data.frequency_groups). Nothing is
selected on test.

Writes results/calibration_per_class_<run_id>.csv and results/figures/precision_per_class_at_threshold.png.

Example:
    python scripts/calibration_per_class.py --run-id legal-bert_final_20261019-1523
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.calibration import per_class_operating_point, per_class_precision_figure  # noqa: E402
from src.config import FIGURES_DIR, NUM_LABELS, PREDICTIONS_DIR, RESULTS_DIR, ROOT  # noqa: E402
from src.data import frequency_groups, get_label_names, load_ledgar  # noqa: E402
from src.utils import markdown_table  # noqa: E402

SPLIT = "test"
COLUMNS = ["label", "label_name", "frequency_group", "train_count", "support", "n_auto", "share_auto", "n_auto_pred",
           "n_auto_correct", "precision_auto", "precision_ci95_low", "precision_ci95_high", "reliable"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="final run already processed by scripts/calibration_report.py")
    p.add_argument("--min-auto", type=int, default=20,
                   help="fewer automatic clauses of a class than this -> its precision is flagged unreliable")
    p.add_argument("--precision-floor", type=float, default=0.90, help="classes below it are listed and named")
    return p.parse_args(argv)


def recommended_point(report: dict) -> dict:
    """Operating point of the recommended target precision in a calibration_report JSON."""
    target = report["recommended_target_precision"]
    point = next(p for p in report["operating_points"] if p["target_precision"] == target)
    if point["threshold"] is None:
        sys.exit(f"Target precision {target} was not reached on validation: no threshold to apply")
    return point


def group_summary(table: pd.DataFrame, precision_floor: float) -> pd.DataFrame:
    rows = []
    for group, part in table.groupby("frequency_group", observed=True):
        below = part["precision_auto"] < precision_floor
        rows.append({
            "group": group,
            "classes": len(part),
            "train clauses per class": f"{part['train_count'].min():,d}–{part['train_count'].max():,d}",
            "test support": f"{part['support'].sum():,d}",
            "share automatic": f"{part['n_auto'].sum() / part['support'].sum():.1%}",
            "precision of automatic labels": f"{part['n_auto_correct'].sum() / part['n_auto_pred'].sum():.4f}",
            "classes without automatic label": int(part["precision_auto"].isna().sum()),
            "unreliable (< min auto.)": int((~part["reliable"]).sum()),
            f"below {precision_floor:.2f} (of which reliable)": f"{below.sum()} ({(below & part['reliable']).sum()})",
        })
    return pd.DataFrame(rows)


def below_floor_table(table: pd.DataFrame, precision_floor: float) -> pd.DataFrame:
    below = table[table["precision_auto"] < precision_floor].sort_values("precision_auto")
    return pd.DataFrame({
        "class": below["label_name"],
        "group": below["frequency_group"],
        "train clauses": below["train_count"].map("{:,d}".format),
        "test support": below["support"],
        "share automatic": below["share_auto"].map("{:.1%}".format),
        "automatic labels": below["n_auto_pred"],
        "precision [95 % CI]": [f"{p:.3f} [{lo:.3f}; {hi:.3f}]" for p, lo, hi in
                                below[["precision_auto", "precision_ci95_low", "precision_ci95_high"]].to_numpy()],
        "reliable": below["reliable"].map({True: "yes", False: "no"}),
    })


def main(argv=None) -> None:
    args = parse_args(argv)
    report_path = RESULTS_DIR / f"calibration_{args.run_id}.json"
    preds_path = PREDICTIONS_DIR / f"{args.run_id}_{SPLIT}.parquet"
    missing = [str(path.relative_to(ROOT)) for path in (report_path, preds_path) if not path.exists()]
    if missing:
        sys.exit(f"Missing file(s): {', '.join(missing)} (run scripts/calibration_report.py first)")
    point = recommended_point(json.loads(report_path.read_text()))
    threshold = point["threshold"]
    preds = pd.read_parquet(preds_path)

    ds = load_ledgar()
    if not (preds["label"].to_numpy() == np.asarray(ds[SPLIT]["label"])[preds["idx"].to_numpy()]).all():
        sys.exit(f"Labels of {preds_path.name} differ from the LEDGAR {SPLIT} split")
    table = per_class_operating_point(preds["label"], preds["pred"], preds["proba_pred"], threshold, NUM_LABELS,
                                      min_auto=args.min_auto)
    table = table.merge(frequency_groups(ds["train"]["label"]), on="label")
    table["label_name"] = get_label_names(ds)
    table = table[COLUMNS]
    if table["n_auto_pred"].sum() != point[SPLIT]["n_auto"]:  # same clauses as the global operating point
        sys.exit("Per-class counts do not add up to the global operating point of the calibration report")

    csv_path = RESULTS_DIR / f"calibration_per_class_{args.run_id}.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")
    fig_path = FIGURES_DIR / "precision_per_class_at_threshold.png"
    per_class_precision_figure(
        table, fig_path, min_auto=args.min_auto, precision_floor=args.precision_floor,
        overall_precision=point[SPLIT]["precision"],
        title=f"Per-class precision at threshold {threshold:.3f} (chosen on validation) — {SPLIT}, {args.run_id}")

    print(f"Threshold {threshold:.3f} (target precision {point['target_precision']:.2f} on validation), {SPLIT} split, "
          f"{len(preds):,d} clauses.\n")
    print(markdown_table(group_summary(table, args.precision_floor)))
    print()
    print(markdown_table(below_floor_table(table, args.precision_floor)))
    for path in (csv_path, fig_path):
        print(f"Written: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
