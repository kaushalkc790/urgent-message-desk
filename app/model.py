"""Desk scores from the best saved model: optimized DistilBERT.

File: models/distilbert_optimized.joblib
Epoch 1, cutoff 0.25. Val recall 1.00, miss 0, extra 0.

One cutoff for every row. A miss costs more than an extra alarm,
so the line sits at 0.25 rather than 0.50.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
JOBLIB = ROOT / "models" / "distilbert_optimized.joblib"
ARCH = ROOT / "models" / "distilbert_p1"
MAX_LEN = 128

_MODEL = None
_TOK = None
_CUTOFF = 0.25


def model_text(row: pd.Series) -> str:
    att = pd.to_numeric(row.get("attachments"), errors="coerce")
    att = 0 if pd.isna(att) else int(att)
    return (
        f"[channel={row.get('channel', '')} attachments={att} "
        f"after_hours={int(row.get('is_after_hours') or 0)} "
        f"tonight={int(row.get('mentions_tonight') or 0)} "
        f"need_person={int(row.get('mentions_human') or 0)}] "
        f"{row.get('subject') or ''} {row.get('message') or ''}"
    )


def load() -> tuple:
    global _MODEL, _TOK, _CUTOFF
    if _MODEL is not None:
        return _MODEL, _TOK, _CUTOFF
    if not JOBLIB.exists():
        raise FileNotFoundError(f"Missing {JOBLIB}")
    torch.set_num_threads(1)
    bundle = joblib.load(JOBLIB)
    _CUTOFF = float(bundle["cutoff"])
    _TOK = DistilBertTokenizerFast.from_pretrained(str(ARCH), local_files_only=True)
    config = DistilBertConfig.from_pretrained(str(ARCH), local_files_only=True)
    _MODEL = DistilBertForSequenceClassification(config)
    _MODEL.load_state_dict(bundle["state_dict"])
    _MODEL.eval()
    return _MODEL, _TOK, _CUTOFF


@torch.no_grad()
def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    model, tok, cutoff = load()
    texts = [model_text(r) for _, r in df.iterrows()]
    enc = tok(
        texts,
        truncation=True,
        padding=True,
        max_length=int(MAX_LEN),
        return_tensors="pt",
    )
    proba = torch.softmax(model(**enc).logits, dim=1)[:, 1].cpu().numpy()
    return pd.DataFrame(
        {
            "case_id": df["case_id"].astype(str).to_numpy(),
            "desk_score": np.round(proba, 6),
            "desk_flagged": (proba >= cutoff).astype(int),
        }
    )
