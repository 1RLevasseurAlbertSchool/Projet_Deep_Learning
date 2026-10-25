"""Tests of the end-to-end clause reviewer (bonus task T4) — mocked model, tokenizer and index, no download."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from src.pipeline import (
    AUTOMATIC,
    CALIBRATION_PATH,
    CHECKPOINT_FILES,
    DEFAULT_ENCODER,
    RETRIEVAL_SUMMARY_PATH,
    SEND_TO_LAWYER,
    ClauseReviewer,
    best_retrieval_encoder,
    check_checkpoint_dir,
    format_review,
    load_review_threshold,
)

LABELS = ["Governing Laws", "Notices", "Terminations", "Waivers"]


class FakeModel(torch.nn.Module):
    """Returns the same logits for every input; records the inputs it received."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)
        self.config = SimpleNamespace(id2label=dict(enumerate(LABELS)), num_labels=len(LABELS))
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append(input_ids)
        return SimpleNamespace(logits=self.logits.repeat(input_ids.shape[0], 1))


class FakeTokenizer:
    def __init__(self):
        self.kwargs = None

    def __call__(self, text, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": torch.ones(1, 4, dtype=torch.long), "attention_mask": torch.ones(1, 4, dtype=torch.long)}


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=5):
        self.queries.append((query, k))
        return [
            {"idx": i, "text": f"clause {i} " + "x" * 300, "label": 2, "label_name": "Terminations",
             "score": 0.9 - 0.1 * i}
            for i in range(k)
        ]


def make_reviewer(logits, threshold=0.892):
    return ClauseReviewer(FakeModel(logits), FakeTokenizer(), FakeRetriever(), threshold)


def test_review_confident_clause_is_automatic_with_top3_in_order():
    reviewer = make_reviewer([0.0, 0.0, 6.0, 1.0])
    result = reviewer.review("This Agreement may be terminated by either party.", k=3)

    assert [t["label_name"] for t in result["top3"]] == ["Terminations", "Waivers", "Governing Laws"]
    probabilities = [t["probability"] for t in result["top3"]]
    assert probabilities == sorted(probabilities, reverse=True) and sum(probabilities) <= 1.0
    assert result["confidence"] == probabilities[0] > 0.892
    assert result["decision"] == AUTOMATIC
    assert len(result["similar"]) == 3


def test_review_below_threshold_is_sent_to_a_lawyer():
    result = make_reviewer([1.0, 1.0, 1.2, 1.0]).review("Ambiguous clause.")
    assert result["confidence"] < 0.892
    assert result["decision"] == SEND_TO_LAWYER


def test_confidence_equal_to_threshold_is_automatic():
    # Same rule as the calibration report: automatic when confidence >= threshold.
    reviewer = make_reviewer([0.0, 0.0, 0.0, 0.0], threshold=0.25)
    assert reviewer.review("Any clause.")["decision"] == AUTOMATIC


def test_classifier_truncates_at_training_max_length():
    reviewer = make_reviewer([0.0, 0.0, 6.0, 1.0])
    reviewer.classify("A clause.")
    assert reviewer.tokenizer.kwargs["truncation"] is True
    assert reviewer.tokenizer.kwargs["max_length"] == 256


def test_search_returns_contract_fields_and_200_char_snippets():
    reviewer = make_reviewer([0.0, 0.0, 6.0, 1.0])
    hits = reviewer.search("termination clauses with a 30-day notice period", k=4)

    assert reviewer.retriever.queries == [("termination clauses with a 30-day notice period", 4)]
    assert len(hits) == 4
    assert set(hits[0]) == {"idx", "label", "label_name", "score", "snippet"}
    assert all(len(hit["snippet"]) == 200 for hit in hits)
    assert [hit["score"] for hit in hits] == sorted((hit["score"] for hit in hits), reverse=True)


def test_review_rejects_empty_text():
    with pytest.raises(ValueError):
        make_reviewer([0.0, 0.0, 6.0, 1.0]).review("   ")


def test_format_review_mentions_decision_and_threshold():
    text = format_review(make_reviewer([0.0, 0.0, 6.0, 1.0]).review("A clause."))
    assert "AUTOMATIC" in text and "threshold 0.892" in text and "Terminations" in text


def test_missing_checkpoint_explains_which_files_to_download(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as error:
        check_checkpoint_dir(tmp_path)
    message = str(error.value)
    assert "model.safetensors" in message and "Drive" in message
    assert all(name in message for name in CHECKPOINT_FILES)


def test_complete_checkpoint_dir_is_accepted(tmp_path):
    for name in CHECKPOINT_FILES:
        (tmp_path / name).write_text("")
    assert check_checkpoint_dir(tmp_path) == tmp_path


def test_review_threshold_of_the_final_model():
    assert load_review_threshold(CALIBRATION_PATH) == 0.892


def test_review_threshold_uses_the_recommended_target(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "recommended_target_precision": 0.98,
        "operating_points": [{"target_precision": 0.95, "threshold": 0.8},
                             {"target_precision": 0.98, "threshold": 0.97}],
    }))
    assert load_review_threshold(path) == 0.97


def test_best_retrieval_encoder_ignores_non_dense_methods(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame({
        "method": ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BM25",
                   "MiniLM + BM25 (RRF, c=60)"],
        "precision@5": [0.70, 0.75, 0.90, 0.95],
    }).to_csv(path, index=False)
    assert best_retrieval_encoder(path) == "BAAI/bge-small-en-v1.5"


def test_best_retrieval_encoder_falls_back_to_minilm(tmp_path):
    assert best_retrieval_encoder(tmp_path / "missing.csv") == DEFAULT_ENCODER


def test_best_retrieval_encoder_from_the_benchmark():
    assert best_retrieval_encoder(RETRIEVAL_SUMMARY_PATH) == "BAAI/bge-small-en-v1.5"
