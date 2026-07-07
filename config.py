"""
config.py — All paths and hyperparameters.
Save this to: /content/drive/MyDrive/AllData/config.py
Never run this directly. Every other script imports from here.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path('/content/drive/MyDrive/AllData')

TRAIN_TSV  = BASE_DIR / 'SubtaskA_Train/train/subtask_a_train.tsv'
TRAIN_ROOT = BASE_DIR / 'SubtaskA_Train/train'

DEV_TSV    = BASE_DIR / 'SubtaskA_Dev/dev/subtask_a_dev.tsv'
DEV_ROOT   = BASE_DIR / 'SubtaskA_Dev/dev'

TEST_TSV   = BASE_DIR / 'test(labelled)/subtask_a_test.tsv'
TEST_ROOT  = BASE_DIR / 'test(labelled)'

MODEL_PATH   = BASE_DIR / 'best_model.pth'
HISTORY_PATH = BASE_DIR / 'history.json'
PLOTS_DIR    = BASE_DIR / 'plots'

# ── Fixed hyperparameters (professor requirements — never change) ──────────────
LR_BERT              = 2e-5
DROPOUT_PROB         = 0.2
BERT_UNFREEZE_EP     = 10
BERT_UNFREEZE_LAYERS = 3

# ── Training hyperparameters ──────────────────────────────────────────────────
LR_BASE             = 0.001   # FIXED — 0.01 causes gradient explosion
BATCH_SIZE          = 4
EPOCHS              = 50
WEIGHT_DECAY        = 5e-2
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTH_ALPHA  = 0.1
MAX_LENGTH          = 64

# ── Model constants ───────────────────────────────────────────────────────────
EMBED_DIM  = 512
NUM_HEADS  = 2
BERT_MODEL = 'bert-base-uncased'

# ── Ground truth score mapping ────────────────────────────────────────────────
RANK_SCORE = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
