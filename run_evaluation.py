"""
================================================================================
 run_evaluation.py  ← RUN THIS SECOND (after run_training.py)
 AdMIRe Three-Stream Model | Test Evaluation
 Researcher: Farhana Sultana Ananna

 What this does:
   Loads best_model.pth (saved by run_training.py)
   Evaluates on the labeled test set (15 rows)
   Prints final Top-1 Accuracy and NDCG
   Tests both WITH and WITHOUT sentence type

 These are your final paper numbers.
================================================================================
"""
import sys
import importlib

sys.path.insert(0, '/content/drive/MyDrive/AllData')
importlib.invalidate_caches()
for mod in ['config', 'utils', 'dataset_loader', 'model_architecture']:
    if mod in sys.modules:
        del sys.modules[mod]

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import torch
import pandas as pd
from torch.utils.data import DataLoader

# Sourced cleanly from config
from config import TEST_TSV, TEST_ROOT, BATCH_SIZE, BASE_DIR
from utils import evaluate_loader
from dataset_loader import IdiomDataset
from model_architecture import IdiomFusionModel

# Resolve the name mismatch safely
MODEL_CHECKPOINT_PATH = BASE_DIR / 'best_model.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Device: {device}')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading checkpoint from: {MODEL_CHECKPOINT_PATH}')
model = IdiomFusionModel().to(device)
state = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print('✅ Model loaded successfully.')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD TEST DATA
# ─────────────────────────────────────────────────────────────────────────────
test_df = pd.read_csv(TEST_TSV, sep='\t')
print(f'📦 Test rows: {len(test_df)}')

results = {}

# Testing both configurations back-to-back
for use_st, label in [(True,  'With Sentence Type'),
                       (False, 'Without Sentence Type')]:
    ds     = IdiomDataset(test_df, TEST_ROOT,
                          augment=False, use_sentence_type=use_st)
    loader = DataLoader(ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2, pin_memory=False)

    ndcg, acc = evaluate_loader(model, loader, device)
    results[label] = (ndcg, acc)
    print(f'  {label:25s} → NDCG: {ndcg:.4f} | Top-1 Acc: {acc:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"─"*55}')
print('  FINAL TEST RESULTS')
print(f'{"─"*55}')
print(f'  {"Configuration":<25} | {"NDCG":>8} | {"Top-1 Acc":>10}')
print(f'{"─"*55}')
for label, (ndcg, acc) in results.items():
    print(f'  {label:<25} | {ndcg:>8.4f} | {acc:>10.4f}')
print(f'{"─"*55}')
print('\n✅ DONE — next run run_plots.py')
"""
================================================================================
Mounted at /content/drive
✅ Device: cuda

Loading checkpoint from: /content/drive/MyDrive/AllData/best_model.pth
Loading weights: 100%
 199/199 [00:00<00:00, 5372.74it/s]
[transformers] BertModel LOAD REPORT from: bert-base-uncased
Key                                        | Status     |  | 
-------------------------------------------+------------+--+-
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 

Notes:
- UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
✅ Model loaded successfully.
📦 Test rows: 15
  With Sentence Type        → NDCG: 0.8800 | Top-1 Acc: 0.3333
  Without Sentence Type     → NDCG: 0.8768 | Top-1 Acc: 0.2667

───────────────────────────────────────────────────────
  FINAL TEST RESULTS
───────────────────────────────────────────────────────
  Configuration             |     NDCG |  Top-1 Acc
───────────────────────────────────────────────────────
  With Sentence Type        |   0.8800 |     0.3333
  Without Sentence Type     |   0.8768 |     0.2667
───────────────────────────────────────────────────────

✅ DONE — next run run_plots.py
================================================================================
"""
