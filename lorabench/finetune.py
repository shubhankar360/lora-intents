"""LoRA and full fine-tuning of the backbone as a 77-way classifier, in plain PyTorch.

No Trainer: the loop is short enough to read, and every number in the report
(steps, seconds, trainable parameters, bytes on disk) comes from here.
"""
from __future__ import annotations

import math
import os
import random
import tempfile
import time
from types import SimpleNamespace

import torch

from .encoder import BACKBONE, MAX_LEN

STEPS_TARGET = 400  # small sets get more epochs, so every run sees a similar number of updates


def epochs_for(n_examples: int, batch: int) -> int:
    per_epoch = math.ceil(n_examples / batch)
    return max(3, min(30, math.ceil(STEPS_TARGET / per_epoch)))


class MeanPoolClassifier(torch.nn.Module):
    """The backbone, mean-pooled the way the frozen baseline uses it, then one linear layer.

    BERT's own classification head reads the [CLS] token through a pooler that
    this sentence model was never trained to use. Using it here would make the
    fine-tuned arms look worse than the frozen one for a reason that has nothing
    to do with fine-tuning.
    """

    def __init__(self, encoder, n_labels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        logits = self.classifier((hidden * mask).sum(1) / mask.sum(1))
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def build(method: str, n_labels: int, r: int = 8, backbone: str = BACKBONE, config=None):
    from transformers import AutoModel
    if config is not None:  # tests pass a tiny config instead of downloading weights
        encoder = AutoModel.from_config(config, add_pooling_layer=False)
    else:
        encoder = AutoModel.from_pretrained(backbone, add_pooling_layer=False)
    model = MeanPoolClassifier(encoder, n_labels)
    if method == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.1,
                         target_modules=["query", "key", "value", "dense"], modules_to_save=["classifier"])
        model = get_peft_model(model, cfg)
    elif method != "full":
        raise ValueError(method)
    return model


def trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def saved_bytes(model) -> int:
    """Size on disk of what you would ship: the adapter for LoRA, the whole model otherwise."""
    with tempfile.TemporaryDirectory() as d:
        if hasattr(model, "save_pretrained"):  # PEFT saves only the adapter and the head
            model.save_pretrained(d)
        else:
            torch.save(model.state_dict(), os.path.join(d, "model.pt"))
        return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(d) for f in files)


def train(model, tok, texts, label_ids, *, lr: float, head_lr: float = 1e-3, batch: int = 32,
          epochs: int | None = None, seed: int = 0) -> dict:
    """`lr` for the backbone (or its adapters), `head_lr` for the new classifier.

    The head starts from random weights in every method, so it gets the same
    rate everywhere; only how the backbone is updated differs.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    epochs = epochs or epochs_for(len(texts), batch)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": [p for n, p in named if "classifier" not in n], "lr": lr},
                             {"params": [p for n, p in named if "classifier" in n], "lr": head_lr}], weight_decay=0.01)
    total = epochs * math.ceil(len(texts) / batch)
    warm = max(1, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm))))
    model.train()
    idx = list(range(len(texts)))
    t0, steps = time.perf_counter(), 0
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            loss = model(**enc, labels=torch.tensor([label_ids[j] for j in b])).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            steps += 1
    model.eval()
    return {"train_s": time.perf_counter() - t0, "steps": steps, "epochs": epochs, "final_loss": float(loss)}


def predict(model, tok, texts, batch: int = 128) -> list[int]:
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            out.extend(model(**enc).logits.argmax(-1).tolist())
    return out
