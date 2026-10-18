"""Configuration shared by all work streams: paths, seed, dataset."""

from pathlib import Path

# Repository root (works locally and in Colab after `git clone`)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = RESULTS_DIR / "figures"

MODELS_DIR = ROOT / "models"  # local checkpoints (gitignored)

SEED = 42

# LEDGAR, LexGLUE version (Chalkidis et al., 2022)
DATASET_NAME = "coastalcph/lex_glue"
DATASET_CONFIG = "ledgar"
SPLITS = ("train", "validation", "test")
NUM_LABELS = 100
TEXT_COL = "text"
LABEL_COL = "label"

# Expected sizes: checked by src.data.load_ledgar
EXPECTED_SIZES = {"train": 60_000, "validation": 10_000, "test": 10_000}
