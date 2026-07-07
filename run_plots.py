"""
================================================================================
 run_plots.py  ← RUN THIS THIRD (after run_evaluation.py)
 AdMIRe Three-Stream Model | Training Curves
 Researcher: Farhana Sultana Ananna

 What this does:
   Reads history.json saved by run_training.py
   Generates publication-quality training curve plots
   Saves PNG files to AllData/plots/ folder
================================================================================
"""
import sys
import json
import importlib

sys.path.insert(0, '/content/drive/MyDrive/AllData')
importlib.invalidate_caches()
for mod in ['config']:
    if mod in sys.modules:
        del sys.modules[mod]

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import matplotlib.pyplot as plt
from config import HISTORY_PATH, PLOTS_DIR, BERT_UNFREEZE_EP

# ─────────────────────────────────────────────────────────────────────────────
# LOAD HISTORY
# ─────────────────────────────────────────────────────────────────────────────
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

with open(HISTORY_PATH, 'r') as f:
    history = json.load(f)

epochs = range(1, len(history['train_loss']) + 1)
print(f'📊 Loaded history: {len(history["train_loss"])} epochs')

# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'AdMIRe Three-Stream Model — Training Curves\nFarhana Sultana Ananna',
    fontsize=13, fontweight='bold'
)

# Loss curve
axes[0].plot(epochs, history['train_loss'],
             color='crimson', lw=2, label='Train Loss (ListNet)')
axes[0].axvline(x=BERT_UNFREEZE_EP, color='navy',
                linestyle='--', lw=1.5, label=f'BERT unfrozen (ep {BERT_UNFREEZE_EP})')
axes[0].set_title('Training Loss', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('ListNet Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation metrics
axes[1].plot(epochs, history['val_ndcg'],
             color='steelblue', lw=2, label='Val NDCG')
axes[1].plot(epochs, history['val_top1'],
             color='darkorange', lw=2, label='Val Top-1 Accuracy')
axes[1].axvline(x=BERT_UNFREEZE_EP, color='navy',
                linestyle='--', lw=1.5, label=f'BERT unfrozen (ep {BERT_UNFREEZE_EP})')
axes[1].set_title('Validation Metrics (Dev Set)', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Score')
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = PLOTS_DIR / 'training_curves.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()

print(f'\n✅ Plot saved to: {out_path}')
print('\nFinal epoch metrics:')
print(f'  Train Loss : {history["train_loss"][-1]:.4f}')
print(f'  Val NDCG   : {history["val_ndcg"][-1]:.4f}  '
      f'(best: {max(history["val_ndcg"]):.4f})')
print(f'  Val Top-1  : {history["val_top1"][-1]:.4f}  '
      f'(best: {max(history["val_top1"]):.4f})')
"""
================================================================================
 Mounted at /content/drive
📊 Loaded history: 25 epochs

✅ Plot saved to: /content/drive/MyDrive/AllData/plots/training_curves.png

Final epoch metrics:
  Train Loss : 1.5830
  Val NDCG   : 0.8918  (best: 0.9080)
  Val Top-1  : 0.4667  (best: 0.4667)

================================================================================
"""
