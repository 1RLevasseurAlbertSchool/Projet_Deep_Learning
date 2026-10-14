"""Chargement de LEDGAR (LexGLUE) — contrat d'interface commun aux trois volets.

Usage :
    from src.data import load_ledgar, get_label_names, to_pandas
    ds = load_ledgar()                 # DatasetDict train / validation / test
    names = get_label_names(ds)        # liste de 100 noms de classes, index = id du label
    df_train = to_pandas(ds, "train")  # DataFrame avec idx, text, label, label_name

Ne jamais re-splitter : le split chronologique (2016-17 / 2018 / 2019) fait partie de l'évaluation.
"""

from __future__ import annotations

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
    """Charge LEDGAR depuis le Hub Hugging Face.

    Args:
        subset: si fourni, ne garde que les `subset` premiers exemples de chaque split
            (smoke tests uniquement — jamais pour un résultat rapporté).
        check: vérifie tailles et nombre de labels sur le dataset complet.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    if check and subset is None:
        for split in SPLITS:
            n = len(ds[split])
            if n != EXPECTED_SIZES[split]:
                raise ValueError(f"Taille inattendue pour {split}: {n} (attendu {EXPECTED_SIZES[split]})")
        n_labels = len(get_label_names(ds))
        if n_labels != NUM_LABELS:
            raise ValueError(f"{n_labels} labels trouvés, {NUM_LABELS} attendus")

    if subset is not None:
        for split in SPLITS:
            ds[split] = ds[split].select(range(min(subset, len(ds[split]))))
    return ds


def get_label_names(ds) -> list[str]:
    """Noms des classes, dans l'ordre des ids (0 à 99)."""
    return list(ds["train"].features[LABEL_COL].names)


def to_pandas(ds, split: str):
    """DataFrame d'un split avec `idx` (position dans le split, clé des fichiers de prédictions)."""
    names = get_label_names(ds)
    df = ds[split].to_pandas()
    df.insert(0, "idx", range(len(df)))
    df["label_name"] = df[LABEL_COL].map(lambda i: names[i])
    return df[["idx", TEXT_COL, LABEL_COL, "label_name"]]


if __name__ == "__main__":
    ds = load_ledgar()
    print(ds)
    names = get_label_names(ds)
    print(f"{len(names)} labels, ex. : {names[:5]}")
