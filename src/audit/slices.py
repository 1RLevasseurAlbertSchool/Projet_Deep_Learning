"""Per-slice classification evaluation for the LEDGAR test predictions."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

_PREDICTION_COLUMNS = {"idx", "label", "pred", "proba_pred"}
_PARTY_PAIRS = {
    "borrower_lender": (r"\bborrower\b", r"\blender\b"),
    "employer_employee": (r"\bemployer\b", r"\bemployee\b"),
    "landlord_tenant": (r"\blandlord\b", r"\btenant\b"),
    "buyer_seller": (r"\bbuyer\b", r"\bseller\b"),
}


def _validate_unique_indices(frame: pd.DataFrame, name: str) -> None:
    if "idx" not in frame:
        raise ValueError(f"{name} must contain an idx column")
    if frame["idx"].isna().any() or frame["idx"].duplicated().any():
        raise ValueError(f"{name} idx values must be non-null and unique")


def _frequency_group(count: int, lower: int, upper: int) -> str:
    if count <= lower:
        return "tail"
    if count <= upper:
        return "middle"
    return "head"


def build_audit_frame(
    evaluation: pd.DataFrame,
    predictions: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Join ground truth, text, and common-format predictions; derive audit attributes.

    The class-frequency cut points and clause-length quartiles are estimated from the
    training split only. A year column is preserved if the caller has verified year
    metadata; this project’s shared LEDGAR loader does not expose per-row years.
    """
    _validate_unique_indices(evaluation, "evaluation data")
    _validate_unique_indices(predictions, "predictions")
    _validate_unique_indices(train, "training data")
    missing_predictions = _PREDICTION_COLUMNS.difference(predictions.columns)
    if missing_predictions:
        raise ValueError(f"Predictions are missing columns: {sorted(missing_predictions)}")
    if not {"text", "label"}.issubset(evaluation.columns):
        raise ValueError("evaluation data must contain text and label columns")
    if not {"text", "label"}.issubset(train.columns):
        raise ValueError("training data must contain text and label columns")
    if not predictions["idx"].isin(evaluation["idx"]).all():
        raise ValueError("predictions contain idx values not present in evaluation data")
    if set(predictions["idx"]) != set(evaluation["idx"]):
        raise ValueError("predictions and evaluation data must contain the same idx values")

    frame = evaluation.merge(
        predictions.loc[:, ["idx", "pred", "proba_pred"]],
        on="idx",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_prediction"),
    )
    if "label_prediction" in frame:
        if not frame["label"].equals(frame["label_prediction"]):
            raise ValueError("prediction labels do not match evaluation ground truth")
        frame = frame.drop(columns="label_prediction")

    class_counts = train["label"].value_counts()
    count_values = class_counts.to_numpy()
    lower, upper = np.quantile(count_values, [1 / 3, 2 / 3]).tolist()
    frame["train_class_count"] = frame["label"].map(class_counts).fillna(0).astype(int)
    frame["frequency_group"] = frame["train_class_count"].map(
        lambda count: _frequency_group(count, int(lower), int(upper))
    )

    train_lengths = train["text"].astype(str).str.split().str.len()
    length_edges = np.quantile(train_lengths, [0.25, 0.75]).tolist()
    frame["word_count"] = frame["text"].astype(str).str.split().str.len()
    frame["length_group"] = np.select(
        [
            frame["word_count"] <= length_edges[0],
            frame["word_count"] <= length_edges[1],
        ],
        ["short", "medium"],
        default="long",
    )

    lowered = frame["text"].astype(str).str.lower()
    for group_name, (left_pattern, right_pattern) in _PARTY_PAIRS.items():
        frame[group_name] = lowered.map(
            lambda text: bool(re.search(left_pattern, text) and re.search(right_pattern, text))
        )
    party_columns = list(_PARTY_PAIRS)
    party_count = frame[party_columns].sum(axis=1)
    frame["party_type"] = "not_detected"
    for group_name in party_columns:
        frame.loc[frame[group_name] & (party_count == 1), "party_type"] = group_name
    frame.loc[party_count > 1, "party_type"] = "multiple"

    return frame


def _classification_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y_true = frame["label"].to_numpy(dtype=int)
    y_pred = frame["pred"].to_numpy(dtype=int)
    observed_labels = np.unique(y_true)
    return {
        "support": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true,
            y_pred,
            labels=observed_labels,
            average="macro",
            zero_division=0,
        )),
        "mean_predicted_confidence": float(frame["proba_pred"].mean()),
    }


def slice_performance(
    audit_frame: pd.DataFrame,
    *,
    slice_columns: Iterable[str] = (
        "frequency_group",
        "length_group",
        "party_type",
        *_PARTY_PAIRS.keys(),
    ),
) -> pd.DataFrame:
    """Return support, accuracy, macro-F1, and confidence for each non-empty slice."""
    rows = []
    for column in slice_columns:
        if column not in audit_frame:
            continue
        if pd.api.types.is_bool_dtype(audit_frame[column]):
            group = audit_frame.loc[audit_frame[column]]
            if not group.empty:
                rows.append({
                    "slice": column,
                    "value": "detected",
                    **_classification_metrics(group),
                })
            continue
        for value, group in audit_frame.groupby(column, observed=True, dropna=False):
            rows.append({
                "slice": column,
                "value": str(value),
                **_classification_metrics(group),
            })
    if "year" in audit_frame:
        for year, group in audit_frame.groupby("year", observed=True, dropna=False):
            rows.append({
                "slice": "year",
                "value": str(year),
                **_classification_metrics(group),
            })
    return pd.DataFrame(
        rows,
        columns=[
            "slice",
            "value",
            "support",
            "accuracy",
            "macro_f1",
            "mean_predicted_confidence",
        ],
    )
