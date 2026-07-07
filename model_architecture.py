"""
model_architecture.py — IdiomFusionModel (three-stream).
Save this to: /content/drive/MyDrive/AllData/model_architecture.py
Never run this directly.

Three streams:
  1. Sentence text  → shared BERT → [B, 1, 512]
  2. Raw images     → frozen ResNet50 → [B, 5, 512]
  3. Image captions → shared BERT → [B, 5, 512]
  Fusion → Cross-attention → Ranking head → scores [B, 5]
"""
import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel


BERT_MODEL           = 'bert-base-uncased'
EMBED_DIM            = 512
NUM_HEADS            = 2
DROPOUT_PROB         = 0.2
BERT_UNFREEZE_LAYERS = 3
MAX_LENGTH           = 64


class IdiomFusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ── Shared BERT (sentence + captions) ─────────────────────────
        self.bert      = BertModel.from_pretrained(BERT_MODEL)
        self.sent_proj = nn.Linear(768, EMBED_DIM)
        self.cap_proj  = nn.Linear(768, EMBED_DIM)

        # ── Frozen ResNet-50 ───────────────────────────────────────────
        resnet = models.resnet50(weights='DEFAULT')
        self.vision_enc = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_enc.parameters():
            p.requires_grad = False          # always frozen
        self.vis_proj = nn.Linear(2048, EMBED_DIM)

        # ── Fusion: concat(visual, caption) → 512 ─────────────────────
        self.fusion_proj = nn.Linear(EMBED_DIM * 2, EMBED_DIM)

        # ── Cross-attention: Q=fused, K=sentence, V=sentence ──────────
        self.cross_attn = nn.MultiheadAttention(
            EMBED_DIM, num_heads=NUM_HEADS, batch_first=True)

        # ── Ranking head (Dropout=0.2 FIXED) ──────────────────────────
        self.ranking_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(64, 1)
        )

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert_last_n(self, n=BERT_UNFREEZE_LAYERS):
        total = len(self.bert.encoder.layer)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i >= total - n:
                for p in layer.parameters():
                    p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True

    def trainable_bert_params(self):
        return [p for p in self.bert.parameters() if p.requires_grad]

    def forward(self, sent_ids, sent_mask, imgs, cap_ids, cap_mask):
        B = sent_ids.size(0)

        # Stream 1 — sentence
        sent_cls  = self.bert(sent_ids, sent_mask).last_hidden_state[:, 0, :]
        sent_feat = self.sent_proj(sent_cls).unsqueeze(1)      # [B, 1, 512]

        # Stream 2 — images (ResNet, always frozen)
        with torch.no_grad():
            vis_raw = self.vision_enc(
                imgs.view(B * 5, 3, 224, 224)
            ).squeeze(-1).squeeze(-1)                          # [B*5, 2048]
        vis_feat = self.vis_proj(vis_raw).view(B, 5, EMBED_DIM) # [B, 5, 512]

        # Stream 3 — captions (shared BERT)
        cap_cls  = self.bert(
            cap_ids.view(B * 5, MAX_LENGTH),
            cap_mask.view(B * 5, MAX_LENGTH)
        ).last_hidden_state[:, 0, :]                           # [B*5, 768]
        cap_feat = self.cap_proj(cap_cls).view(B, 5, EMBED_DIM) # [B, 5, 512]

        # Fusion — concat visual + caption then project
        fused_vc = self.fusion_proj(
            torch.cat([vis_feat, cap_feat], dim=-1)
        )                                                       # [B, 5, 512]

        # Cross-attention — Q=fused, K=sentence, V=sentence
        attn_out, _ = self.cross_attn(
            query=fused_vc,
            key=sent_feat,
            value=sent_feat
        )                                                       # [B, 5, 512]

        # Residual + ranking
        out = fused_vc + attn_out                              # [B, 5, 512]
        return self.ranking_head(out).squeeze(-1)              # [B, 5]
