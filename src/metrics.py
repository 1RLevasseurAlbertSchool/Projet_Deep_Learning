"""Métriques et tests appariés partagés (S01, S04, S06).

- `classification_metrics` : macro-F1 (métrique principale), micro-F1 (= accuracy en mono-label), F1 par classe.
- `paired_bootstrap_f1`   : IC à 95 % de la différence de macro-F1 entre deux modèles sur le même test set.
- `mcnemar_test`          : test de McNemar sur les désaccords (accuracy), exact si b + c < 25.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred, num_labels: int | None = None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.arange(num_labels) if num_labels is not None else None
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_f1": [float(x) for x in per_class],
    }


def paired_bootstrap_f1(y_true, pred_a, pred_b, n_boot: int = 1000, seed: int = 42,
                        num_labels: int | None = None) -> dict:
    """Bootstrap apparié : on rééchantillonne les MÊMES exemples pour A et B (S06).

    Retourne la différence observée F1(A) - F1(B), l'IC à 95 % (percentiles 2.5 / 97.5)
    et la part des rééchantillonnages où A ne bat pas B.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    labels = np.arange(num_labels) if num_labels is not None else np.unique(y_true)

    def f1(y, p):
        return f1_score(y, p, labels=labels, average="macro", zero_division=0)

    observed = f1(y_true, pred_a) - f1(y_true, pred_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = f1(y_true[s], pred_a[s]) - f1(y_true[s], pred_b[s])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta_macro_f1": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "significant": bool(low > 0 or high < 0),
        "p_a_not_better": float((diffs <= 0).mean()),
        "n_boot": n_boot,
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar avec correction de continuité ; binomial exact si b + c < 25 (S06).

    b = A juste et B faux, c = A faux et B juste.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    a = int(np.sum(ok_a & ok_b))
    b = int(np.sum(ok_a & ~ok_b))
    c = int(np.sum(~ok_a & ok_b))
    d = int(np.sum(~ok_a & ~ok_b))
    if b + c == 0:
        return {"a": a, "b": b, "c": c, "d": d, "chi2": 0.0, "p_value": 1.0, "method": "no_disagreement"}
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
        return {"a": a, "b": b, "c": c, "d": d, "chi2": None, "p_value": float(p), "method": "exact_binomial"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = stats.chi2.sf(chi2, df=1)
    return {"a": a, "b": b, "c": c, "d": d, "chi2": float(chi2), "p_value": float(p), "method": "chi2_cc"}
