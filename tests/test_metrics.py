"""Tests des métriques partagées, vérifiés sur les exercices du cours (S01 Ex. 3, S06 Ex. 5)."""

import numpy as np
import pytest

from src.metrics import classification_metrics, mcnemar_test, paired_bootstrap_f1


def _from_confusion(cm):
    y_true, y_pred = [], []
    for i, row in enumerate(cm):
        for j, n in enumerate(row):
            y_true += [i] * n
            y_pred += [j] * n
    return np.array(y_true), np.array(y_pred)


def test_macro_f1_course_exercise():
    # S01, Exercice 3 : termination / payment / confidentiality, 70 clauses
    y_true, y_pred = _from_confusion([[44, 4, 2], [2, 6, 2], [1, 1, 8]])
    m = classification_metrics(y_true, y_pred, num_labels=3)
    assert m["accuracy"] == pytest.approx(58 / 70)
    assert m["micro_f1"] == pytest.approx(m["accuracy"])
    f1 = [2 * 44 / (50 + 47), 2 * 6 / (10 + 11), 2 * 8 / (10 + 12)]
    assert m["macro_f1"] == pytest.approx(np.mean(f1))


def _paired(b, c, a=0, d=0):
    y = np.zeros(a + b + c + d, dtype=int)
    pa = np.array([0] * a + [0] * b + [1] * c + [1] * d)
    pb = np.array([0] * a + [1] * b + [0] * c + [1] * d)
    return y, pa, pb


def test_mcnemar_course_exercise():
    # S06, Exercice 5 : b = 60, c = 40 -> chi2 = 19^2 / 100 = 3.61, non significatif à 5 %
    r = mcnemar_test(*_paired(b=60, c=40, a=830, d=70))
    assert r["chi2"] == pytest.approx(3.61)
    assert r["p_value"] > 0.05
    # b = 25, c = 5 -> chi2 = 19^2 / 30 = 12.03, significatif
    r = mcnemar_test(*_paired(b=25, c=5, a=865, d=105))
    assert r["chi2"] == pytest.approx(361 / 30)
    assert r["p_value"] < 0.05


def test_mcnemar_exact_when_few_disagreements():
    r = mcnemar_test(*_paired(b=10, c=2, a=100))
    assert r["method"] == "exact_binomial"


def test_paired_bootstrap_detects_clear_gain():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    good = np.where(rng.random(2000) < 0.9, y, (y + 1) % 5)
    bad = np.where(rng.random(2000) < 0.6, y, (y + 1) % 5)
    r = paired_bootstrap_f1(y, good, bad, n_boot=200, num_labels=5)
    assert r["significant"] and r["ci95_low"] > 0
