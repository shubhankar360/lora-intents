"""The shared backbone: all-MiniLM-L6-v2 (22M parameters, 6 layers)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 64  # Banking77 messages are short; 64 tokens covers all but a handful


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    model = AutoModel.from_pretrained(BACKBONE).eval()
    return tok, model


def tokenizer():
    return _load()[0]


def embed(texts: list[str], batch: int = 64) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence embeddings, the way the model was trained to be used."""
    import torch
    tok, model = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(pooled, dim=-1).numpy())
    return np.concatenate(out)
