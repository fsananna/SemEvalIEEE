"""
================================================================================
fixing the error
 run_training.py  ← RUN THIS
 AdMIRe Three-Stream Model | Final Training
 Researcher: Farhana Sultana Ananna
================================================================================
"""
import sys, json, shutil, importlib
sys.path.insert(0, '/content/drive/MyDrive/AllData')
importlib.invalidate_caches()
for mod in ['config','utils','dataset_loader','model_architecture']:
    if mod in sys.modules: del sys.modules[mod]

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from config import (
    TRAIN_TSV, TRAIN_ROOT, DEV_TSV, DEV_ROOT,
    MODEL_PATH, HISTORY_PATH,
    LR_BASE, LR_BERT, BATCH_SIZE, EPOCHS,
    WEIGHT_DECAY, LABEL_SMOOTH_ALPHA,
    BERT_UNFREEZE_EP, BERT_UNFREEZE_LAYERS, BASE_DIR
)
from utils import listnet_loss, evaluate_loader
from dataset_loader import IdiomDataset
from model_architecture import IdiomFusionModel

# ── KEY FIX: patience must be long enough for BERT to take effect ─────────────
# BERT unfreezes at epoch 10. We need at least 15 epochs after that.
# So minimum useful run = 10 + 15 = 25 epochs before stopping.
EARLY_STOP_PATIENCE = 15   # was 10 — too aggressive, stopped before BERT could help

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ Device: {device}')

# ─────────────────────────────────────────────────────────────────────────────
def train(use_sentence_type, save_path):
    label = 'WITH sentence type' if use_sentence_type else 'WITHOUT sentence type'
    print(f'\n{"="*60}\n  Training: {label}\n{"="*60}')

    train_df = pd.read_csv(TRAIN_TSV, sep='\t')
    dev_df   = pd.read_csv(DEV_TSV,   sep='\t')
    print(f'  Train: {len(train_df)} rows | Dev: {len(dev_df)} rows')

    train_ds = IdiomDataset(train_df, TRAIN_ROOT,
                            augment=True, use_sentence_type=use_sentence_type)
    dev_ds   = IdiomDataset(dev_df,   DEV_ROOT,
                            augment=False, use_sentence_type=use_sentence_type)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=False)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=False)

    model = IdiomFusionModel().to(device)
    model.freeze_bert()

    optimizer = torch.optim.AdamW([
        {'params': model.sent_proj.parameters(),
         'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': model.cap_proj.parameters(),
         'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': model.vis_proj.parameters(),
         'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': model.fusion_proj.parameters(),
         'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': model.cross_attn.parameters(),
         'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
        {'params': model.ranking_head.parameters(),
         'lr': LR_BASE, 'weight_decay': WEIGHT_DECAY},
    ])

    scheduler      = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=EPOCHS)
    bert_added     = False
    best_ndcg      = -1.0      # ← watch NDCG only, not combined
    patience_count = 0
    local_ckpt     = Path('/content/best_model_tmp.pth')

    history = {'train_loss': [], 'val_ndcg': [], 'val_top1': []}

    for epoch in range(1, EPOCHS + 1):

        # Partial BERT unfreeze at epoch 10
        if epoch == BERT_UNFREEZE_EP and not bert_added:
            model.unfreeze_bert_last_n(BERT_UNFREEZE_LAYERS)
            bert_params = model.trainable_bert_params()
            if bert_params:
                optimizer.add_param_group({
                    'params':       bert_params,
                    'lr':           LR_BERT,
                    'weight_decay': WEIGHT_DECAY
                })
            bert_added = True
            print(f'  🔓 Epoch {epoch}: BERT last {BERT_UNFREEZE_LAYERS} '
                  f'layers unfrozen (lr={LR_BERT})')

        # Train
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            s_ids, s_msk, imgs, c_ids, c_msk, targets = batch
            s_ids    = s_ids.to(device)
            s_msk    = s_msk.to(device)
            imgs     = imgs.to(device)
            c_ids    = c_ids.to(device)
            c_msk    = c_msk.to(device)
            targets  = targets.to(device)

            optimizer.zero_grad()
            out  = model(s_ids, s_msk, imgs, c_ids, c_msk)
            loss = listnet_loss(out, targets, alpha=LABEL_SMOOTH_ALPHA)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # Validate
        val_ndcg, val_acc = evaluate_loader(model, dev_loader, device)

        history['train_loss'].append(round(avg_loss,  4))
        history['val_ndcg'].append(  round(val_ndcg,  4))
        history['val_top1'].append(  round(val_acc,   4))

        print(f'  Epoch {epoch:02d}/{EPOCHS} | '
              f'Loss: {avg_loss:.4f} | '
              f'Val NDCG: {val_ndcg:.4f} | '
              f'Val Top-1: {val_acc:.4f}')

        # ── Save checkpoint on best NDCG only ────────────────────────────
        if val_ndcg > best_ndcg:
            best_ndcg      = val_ndcg
            patience_count = 0
            torch.save(model.state_dict(), local_ckpt)
            shutil.copy(local_ckpt, save_path)
            print(f'  💾 Saved  (Val NDCG={val_ndcg:.4f})')
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f'\n  ⏹ Early stopping at epoch {epoch} '
                      f'(no NDCG improvement for {EARLY_STOP_PATIENCE} epochs).')
                break

    print(f'\n  Best Val NDCG : {best_ndcg:.4f}')
    print(f'  Best Val Top-1: {max(history["val_top1"]):.4f}')
    print(f'  Model saved to: {save_path}')
    return history

# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — WITH sentence type  (main model)
history_with = train(
    use_sentence_type=True,
    save_path=MODEL_PATH
)

# Save history for plots
with open(HISTORY_PATH, 'w') as f:
    json.dump(history_with, f)
print(f'\nHistory saved to: {HISTORY_PATH}')

# Pass 2 — WITHOUT sentence type  (ablation)
history_without = train(
    use_sentence_type=False,
    save_path=BASE_DIR / 'best_model_no_st.pth'
)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print('  TRAINING SUMMARY')
print(f'{"="*60}')
print(f'  With ST    — Best Val NDCG: {max(history_with["val_ndcg"]):.4f} | '
      f'Best Val Top-1: {max(history_with["val_top1"]):.4f}')
print(f'  Without ST — Best Val NDCG: {max(history_without["val_ndcg"]):.4f} | '
      f'Best Val Top-1: {max(history_without["val_top1"]):.4f}')
print(f'{"="*60}')
print('\n✅ DONE — next run run_evaluation.py')
"""
================================================================================
Output:Mounted at /content/drive
✅ Device: cuda

============================================================
  Training: WITH sentence type
============================================================
  Train: 70 rows | Dev: 15 rows
Loading weights: 100%
 199/199 [00:00<00:00, 2578.61it/s]
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
  Epoch 01/50 | Loss: 1.6195 | Val NDCG: 0.8752 | Val Top-1: 0.2000
  💾 Saved  (Val NDCG=0.8752)
  Epoch 02/50 | Loss: 1.6111 | Val NDCG: 0.8996 | Val Top-1: 0.4667
  💾 Saved  (Val NDCG=0.8996)
  Epoch 03/50 | Loss: 1.6069 | Val NDCG: 0.8766 | Val Top-1: 0.2000
  Epoch 04/50 | Loss: 1.5996 | Val NDCG: 0.8679 | Val Top-1: 0.2000
  Epoch 05/50 | Loss: 1.6015 | Val NDCG: 0.8624 | Val Top-1: 0.2667
  Epoch 06/50 | Loss: 1.5925 | Val NDCG: 0.8974 | Val Top-1: 0.3333
  Epoch 07/50 | Loss: 1.5932 | Val NDCG: 0.8939 | Val Top-1: 0.4667
  Epoch 08/50 | Loss: 1.5914 | Val NDCG: 0.8846 | Val Top-1: 0.2667
  Epoch 09/50 | Loss: 1.5901 | Val NDCG: 0.8666 | Val Top-1: 0.2667
  🔓 Epoch 10: BERT last 3 layers unfrozen (lr=2e-05)
  Epoch 10/50 | Loss: 1.5924 | Val NDCG: 0.9080 | Val Top-1: 0.2667
  💾 Saved  (Val NDCG=0.9080)
  Epoch 11/50 | Loss: 1.5909 | Val NDCG: 0.8631 | Val Top-1: 0.1333
  Epoch 12/50 | Loss: 1.5899 | Val NDCG: 0.8798 | Val Top-1: 0.3333
  Epoch 13/50 | Loss: 1.5888 | Val NDCG: 0.8641 | Val Top-1: 0.2667
  Epoch 14/50 | Loss: 1.5862 | Val NDCG: 0.8570 | Val Top-1: 0.0667
  Epoch 15/50 | Loss: 1.5869 | Val NDCG: 0.8706 | Val Top-1: 0.2000
  Epoch 16/50 | Loss: 1.5854 | Val NDCG: 0.8684 | Val Top-1: 0.2000
  Epoch 17/50 | Loss: 1.5854 | Val NDCG: 0.8640 | Val Top-1: 0.1333
  Epoch 18/50 | Loss: 1.5842 | Val NDCG: 0.8693 | Val Top-1: 0.2667
  Epoch 19/50 | Loss: 1.5851 | Val NDCG: 0.8785 | Val Top-1: 0.3333
  Epoch 20/50 | Loss: 1.5849 | Val NDCG: 0.8626 | Val Top-1: 0.2000
  Epoch 21/50 | Loss: 1.5856 | Val NDCG: 0.8982 | Val Top-1: 0.4000
  Epoch 22/50 | Loss: 1.5849 | Val NDCG: 0.8735 | Val Top-1: 0.2667
  Epoch 23/50 | Loss: 1.5837 | Val NDCG: 0.8882 | Val Top-1: 0.4000
  Epoch 24/50 | Loss: 1.5842 | Val NDCG: 0.8745 | Val Top-1: 0.3333
  Epoch 25/50 | Loss: 1.5830 | Val NDCG: 0.8918 | Val Top-1: 0.4667

  ⏹ Early stopping at epoch 25 (no NDCG improvement for 15 epochs).

  Best Val NDCG : 0.9080
  Best Val Top-1: 0.4667
  Model saved to: /content/drive/MyDrive/AllData/best_model.pth

History saved to: /content/drive/MyDrive/AllData/history.json

============================================================
  Training: WITHOUT sentence type
============================================================
  Train: 70 rows | Dev: 15 rows
Loading weights: 100%
 199/199 [00:00<00:00, 4146.01it/s]
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
  Epoch 01/50 | Loss: 1.6212 | Val NDCG: 0.8444 | Val Top-1: 0.0667
  💾 Saved  (Val NDCG=0.8444)
  Epoch 02/50 | Loss: 1.6081 | Val NDCG: 0.8732 | Val Top-1: 0.2667
  💾 Saved  (Val NDCG=0.8732)
  Epoch 03/50 | Loss: 1.6081 | Val NDCG: 0.8525 | Val Top-1: 0.1333
  Epoch 04/50 | Loss: 1.6023 | Val NDCG: 0.8711 | Val Top-1: 0.2667
  Epoch 05/50 | Loss: 1.5977 | Val NDCG: 0.8690 | Val Top-1: 0.3333
  Epoch 06/50 | Loss: 1.5929 | Val NDCG: 0.9026 | Val Top-1: 0.4667
  💾 Saved  (Val NDCG=0.9026)
  Epoch 07/50 | Loss: 1.5920 | Val NDCG: 0.8749 | Val Top-1: 0.2667
  Epoch 08/50 | Loss: 1.5906 | Val NDCG: 0.8822 | Val Top-1: 0.4000
  Epoch 09/50 | Loss: 1.5876 | Val NDCG: 0.8842 | Val Top-1: 0.4667
  🔓 Epoch 10: BERT last 3 layers unfrozen (lr=2e-05)
  Epoch 10/50 | Loss: 1.5885 | Val NDCG: 0.8765 | Val Top-1: 0.4000
  Epoch 11/50 | Loss: 1.5874 | Val NDCG: 0.8553 | Val Top-1: 0.2000
  Epoch 12/50 | Loss: 1.5878 | Val NDCG: 0.8663 | Val Top-1: 0.3333
  Epoch 13/50 | Loss: 1.5863 | Val NDCG: 0.8659 | Val Top-1: 0.4000
  Epoch 14/50 | Loss: 1.5862 | Val NDCG: 0.8896 | Val Top-1: 0.4000
  Epoch 15/50 | Loss: 1.5866 | Val NDCG: 0.8767 | Val Top-1: 0.4667
  Epoch 16/50 | Loss: 1.5868 | Val NDCG: 0.8786 | Val Top-1: 0.4000
  Epoch 17/50 | Loss: 1.5855 | Val NDCG: 0.8804 | Val Top-1: 0.4000
  Epoch 18/50 | Loss: 1.5853 | Val NDCG: 0.8802 | Val Top-1: 0.4667
  Epoch 19/50 | Loss: 1.5848 | Val NDCG: 0.8795 | Val Top-1: 0.4000
  Epoch 20/50 | Loss: 1.5843 | Val NDCG: 0.8758 | Val Top-1: 0.3333
  Epoch 21/50 | Loss: 1.5839 | Val NDCG: 0.8849 | Val Top-1: 0.4667

  ⏹ Early stopping at epoch 21 (no NDCG improvement for 15 epochs).

  Best Val NDCG : 0.9026
  Best Val Top-1: 0.4667
  Model saved to: /content/drive/MyDrive/AllData/best_model_no_st.pth

============================================================
  TRAINING SUMMARY
============================================================
  With ST    — Best Val NDCG: 0.9080 | Best Val Top-1: 0.4667
  Without ST — Best Val NDCG: 0.9026 | Best Val Top-1: 0.4667
============================================================

✅ DONE — next run run_evaluation.py
================================================================================
"""
