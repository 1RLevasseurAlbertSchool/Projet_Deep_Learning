"""Regenerate the test predictions of the TF-IDF baseline run (stream B, input of notebook 03).

`results/predictions/` is gitignored, so a fresh clone has the run `tfidf-logreg_baseline_20261023` (config.json,
metrics.json) but not its prediction file. This script refits exactly the configuration of notebook 01 (train only)
and writes results/predictions/<run_id>_test.parquet ONLY if the recomputed test macro-F1 and accuracy are identical
to the run's metrics.json. It never writes the run's config.json or metrics.json.

Example:
    python scripts/regenerate_tfidf_predictions.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `python scripts/...` from the repo root

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.config import NUM_LABELS, SEED  # noqa: E402
from src.data import load_ledgar  # noqa: E402
from src.metrics import classification_metrics  # noqa: E402
from src.utils import load_run, save_predictions, set_seed  # noqa: E402

RUN_ID = "tfidf-logreg_baseline_20261023"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=RUN_ID)
    args = parser.parse_args()

    set_seed(SEED)
    config, metrics = load_run(args.run_id)
    ds = load_ledgar()

    # Same settings as notebook 01 (cell "TF-IDF + LogReg"), read back from the run's config.json.
    vectorizer = TfidfVectorizer(ngram_range=tuple(config["ngram_range"]), min_df=config["min_df"],
                                 sublinear_tf=config["sublinear_tf"], lowercase=True)
    X_train = vectorizer.fit_transform(ds["train"]["text"])
    if X_train.shape[1] != config["n_features"]:
        sys.exit(f"STOP: {X_train.shape[1]} TF-IDF features, the run recorded {config['n_features']}; nothing saved")
    classifier = LogisticRegression(C=config["C"], max_iter=config["max_iter"], random_state=SEED, solver="lbfgs")
    classifier.fit(X_train, ds["train"]["label"])
    if not np.array_equal(classifier.classes_, np.arange(NUM_LABELS)):
        sys.exit("STOP: probability columns are not in label-id order; nothing saved")

    X_test = vectorizer.transform(ds["test"]["text"])
    y_test = np.asarray(ds["test"]["label"])
    probas = classifier.predict_proba(X_test)
    preds = classifier.predict(X_test)
    recomputed = classification_metrics(y_test, preds, num_labels=NUM_LABELS)

    recorded = metrics["test"]
    for key in ("macro_f1", "accuracy"):
        print(f"{key}: recomputed {recomputed[key]!r}, recorded {recorded[key]!r}")
    if recomputed["macro_f1"] != recorded["macro_f1"] or recomputed["accuracy"] != recorded["accuracy"]:
        sys.exit("STOP: recomputed test metrics differ from metrics.json; nothing saved")

    save_predictions(args.run_id, "test", y_test, preds, probas)
    print(f"Saved results/predictions/{args.run_id}_test.parquet")


if __name__ == "__main__":
    main()
