"""Tests of the data helpers that do not need the dataset."""

import pytest

from src.data import frequency_groups


def test_frequency_groups_are_terciles_of_training_counts():
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # class i appears counts[i] times
    groups = frequency_groups([label for label, n in enumerate(counts) for _ in range(n)], num_labels=9)
    assert groups["label"].tolist() == list(range(9))
    assert groups["train_count"].tolist() == counts
    assert groups["frequency_group"].tolist() == ["tail"] * 3 + ["middle"] * 3 + ["head"] * 3


def test_frequency_groups_keep_equal_counts_together_and_check_labels():
    groups = frequency_groups([0, 1, 2, 2, 3, 3, 4, 4, 4], num_labels=6)  # class 5 never appears
    assert groups["train_count"].tolist() == [1, 1, 2, 2, 3, 0]
    assert groups["frequency_group"].iloc[0] == groups["frequency_group"].iloc[1]
    assert groups["frequency_group"].iloc[2] == groups["frequency_group"].iloc[3]
    with pytest.raises(ValueError):
        frequency_groups([0, 7], num_labels=5)
