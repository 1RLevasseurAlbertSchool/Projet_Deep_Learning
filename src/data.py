"""LEDGAR (LexGLUE) loading — interface contract shared by the three work streams.

Usage:
    from src.data import load_ledgar, get_label_names, to_pandas
    ds = load_ledgar()                 # DatasetDict train / validation / test
    names = get_label_names(ds)        # list of 100 class names, index = label id
    df_train = to_pandas(ds, "train")  # DataFrame with idx, text, label, label_name
    groups = frequency_groups(ds["train"]["label"])  # head / middle / tail of every class

Never re-split: the chronological split (2016-17 / 2018 / 2019) is part of the evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATASET_CONFIG,
    DATASET_NAME,
    EXPECTED_SIZES,
    LABEL_COL,
    NUM_LABELS,
    SPLITS,
    TEXT_COL,
)


def load_ledgar(subset: int | None = None, check: bool = True):
    """Load LEDGAR from the Hugging Face Hub.

    Args:
        subset: if given, keep only the first `subset` examples of each split
            (smoke tests only — never for a reported result).
        check: check sizes and number of labels on the full dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    if check and subset is None:
        for split in SPLITS:
            n = len(ds[split])
            if n != EXPECTED_SIZES[split]:
                raise ValueError(f"Unexpected size for {split}: {n} (expected {EXPECTED_SIZES[split]})")
        n_labels = len(get_label_names(ds))
        if n_labels != NUM_LABELS:
            raise ValueError(f"{n_labels} labels found, {NUM_LABELS} expected")

    if subset is not None:
        for split in SPLITS:
            ds[split] = ds[split].select(range(min(subset, len(ds[split]))))
    return ds


def get_label_names(ds) -> list[str]:
    """Class names, in id order (0 to 99)."""
    return list(ds["train"].features[LABEL_COL].names)


def to_pandas(ds, split: str):
    """DataFrame of a split with `idx` (position in the split, key of the prediction files)."""
    names = get_label_names(ds)
    df = ds[split].to_pandas()
    df.insert(0, "idx", range(len(df)))
    df["label_name"] = df[LABEL_COL].map(lambda i: names[i])
    return df[["idx", TEXT_COL, LABEL_COL, "label_name"]]


FREQUENCY_GROUPS = ("tail", "middle", "head")  # rarest third -> most frequent third


def frequency_groups(train_labels, num_labels: int = NUM_LABELS) -> pd.DataFrame:
    """Training-set count and frequency group of every class, in label-id order.

    Groups = terciles of the per-class training counts (`pd.qcut`): classes with the same count always share a group,
    so the three groups may differ by one class in size. Shared definition of the head / middle / tail slices.
    """
    counts = np.bincount(np.asarray(train_labels), minlength=num_labels)
    if len(counts) != num_labels:
        raise ValueError(f"labels above {num_labels - 1} in train_labels")
    return pd.DataFrame({
        LABEL_COL: np.arange(num_labels),
        "train_count": counts,
        "frequency_group": pd.qcut(counts, len(FREQUENCY_GROUPS), labels=FREQUENCY_GROUPS),
    })


if __name__ == "__main__":
    ds = load_ledgar()
    print(ds)
    names = get_label_names(ds)
    print(f"{len(names)} labels, e.g.: {names[:5]}")
