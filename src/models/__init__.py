"""Clause classification models.

Stream A (Robin): Hugging Face fine-tuning (full) and LoRA (`transformer.py`).
The TF-IDF + logistic regression baseline lives in notebook 01 and `scripts/regenerate_tfidf_predictions.py`;
the LSTM was dropped (team decision, 25/10).
Every model writes its runs and predictions through src.utils.save_run / save_predictions.
"""
