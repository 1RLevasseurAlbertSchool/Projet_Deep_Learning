"""Clause classification models.

Stream B (Amandine): TF-IDF + logistic regression baseline, LSTM / BiLSTM (packing, GloVe).
Stream A (Robin)   : Hugging Face fine-tuning (full) and LoRA.
Every model writes its runs and predictions through src.utils.save_run / save_predictions.
"""
