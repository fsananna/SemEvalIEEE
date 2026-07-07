"""
dataset_loader.py — IdiomDataset (three-stream).
Save this to: /content/drive/MyDrive/AllData/dataset_loader.py
Never run this directly.

Returns a 6-tuple per item:
  sent_ids   [64]           sentence token ids
  sent_mask  [64]           sentence attention mask
  imgs       [5, 3,224,224] five image tensors
  cap_ids    [5, 64]        five caption token id tensors
  cap_mask   [5, 64]        five caption attention masks
  scores     [5]            ground truth relevance scores
"""
import ast
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import BertTokenizer
from PIL import Image
from pathlib import Path


RANK_SCORE = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
MAX_LENGTH = 64
BERT_MODEL = 'bert-base-uncased'


class IdiomDataset(Dataset):

    def __init__(self, dataframe, image_root,
                 augment=False, use_sentence_type=True):
        self.df                = dataframe.reset_index(drop=True)
        self.image_root        = Path(image_root)
        self.use_sentence_type = use_sentence_type
        self.tokenizer         = BertTokenizer.from_pretrained(BERT_MODEL)

        if augment:
            self.img_transform = T.Compose([
                T.Resize((256, 256)),
                T.RandomCrop((224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def _tok(self, text):
        """Tokenize one string → (input_ids [64], attention_mask [64])"""
        enc = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def _load_img(self, compound, name):
        """Load one image. Return zeros tensor on any failure."""
        if not name or name == 'nan':
            return torch.zeros(3, 224, 224)
        path = self.image_root / compound / name
        try:
            return self.img_transform(Image.open(path).convert('RGB'))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        sentence = str(row['sentence']).strip()
        compound = str(row['compound']).strip()

        # ── Sentence text ──────────────────────────────────────────────
        if (self.use_sentence_type
                and 'sentence_type' in row.index
                and pd.notna(row['sentence_type'])):
            sent_type = str(row['sentence_type']).strip().capitalize()
            text_str  = f"{sentence} {sent_type}."
        else:
            text_str = sentence

        sent_ids, sent_mask = self._tok(text_str)

        # ── Images and captions ────────────────────────────────────────
        imgs, cap_ids_list, cap_mask_list = [], [], []

        for i in range(1, 6):
            # Image
            name = str(row.get(f'image{i}_name', '')).strip()
            imgs.append(self._load_img(compound, name))

            # Caption
            raw = row.get(f'image{i}_caption', '')
            cap = str(raw).strip() if pd.notna(raw) else ''
            c_ids, c_mask = self._tok(cap)
            cap_ids_list.append(c_ids)
            cap_mask_list.append(c_mask)

        # ── Scores ────────────────────────────────────────────────────
        scores    = torch.zeros(5)
        img_names = [str(row.get(f'image{i}_name', '')).strip()
                     for i in range(1, 6)]
        try:
            gt = ast.literal_eval(str(row['expected_order']))
            for rank, fname in enumerate(gt[:5]):
                if fname in img_names:
                    scores[img_names.index(fname)] = RANK_SCORE.get(rank, 0.0)
        except Exception:
            pass

        return (
            sent_ids,
            sent_mask,
            torch.stack(imgs),
            torch.stack(cap_ids_list),
            torch.stack(cap_mask_list),
            scores
        )
