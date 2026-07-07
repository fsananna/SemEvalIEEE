"""
utils.py — Shared loss and evaluation functions.
Save this to: /content/drive/MyDrive/AllData/utils.py
Never run this directly.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import ndcg_score


def listnet_loss(pred, target, alpha=0.1):
    """
    ListNet ranking loss with label smoothing.
    pred   : [B, 5] model output logits
    target : [B, 5] ground truth scores
    Returns: scalar loss
    """
    if alpha > 0:
        uniform = torch.full_like(target, 1.0 / target.size(1))
        target  = (1.0 - alpha) * target + alpha * uniform
    pred_p   = F.softmax(pred,   dim=1)
    target_p = F.softmax(target, dim=1)
    return (-torch.sum(target_p * torch.log(pred_p + 1e-10), dim=1)).mean()


def evaluate_loader(model, loader, device):
    """
    Run inference over a DataLoader and return (ndcg, top1_accuracy).
    Returns two values: ndcg (float), acc (float)
    """
    model.eval()
    all_accs  = []
    all_ndcgs = []

    with torch.no_grad():
        for batch in loader:
            sent_ids, sent_mask, imgs, cap_ids, cap_mask, targets = batch
            sent_ids  = sent_ids.to(device)
            sent_mask = sent_mask.to(device)
            imgs      = imgs.to(device)
            cap_ids   = cap_ids.to(device)
            cap_mask  = cap_mask.to(device)

            out = model(sent_ids, sent_mask, imgs, cap_ids, cap_mask)

            logits_np  = out.cpu().numpy()
            targets_np = targets.numpy()

            all_accs.append(
                (np.argmax(logits_np, axis=1) == np.argmax(targets_np, axis=1))
                .astype(float).mean()
            )
            all_ndcgs.append(ndcg_score(targets_np, logits_np))

    return float(np.mean(all_ndcgs)), float(np.mean(all_accs))
