"""Banking77: 13,083 customer messages, 77 intents (PolyAI, CC BY 4.0).

The CSVs are committed under data/ with attribution, so runs don't depend on a
dataset hub. `subsample` draws k examples per intent with a fixed seed; the same
seed gives every method exactly the same training examples.
"""
from __future__ import annotations

import csv
import pathlib
import random
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"


def download() -> None:
    DATA.mkdir(exist_ok=True)
    for split in ("train", "test"):
        path = DATA / f"{split}.csv"
        if not path.exists():
            urllib.request.urlretrieve(SOURCE.format(split=split), path)


def load(split: str) -> tuple[list[str], list[str]]:
    with open(DATA / f"{split}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def labels() -> list[str]:
    return sorted(set(load("train")[1]))


def subsample(texts: list[str], cats: list[str], k: int | None, seed: int = 0) -> tuple[list[str], list[str]]:
    """k examples per intent (all of them when k is None), shuffled."""
    if k is None:
        pairs = list(zip(texts, cats))
    else:
        by_cat: dict[str, list[str]] = {}
        for t, c in zip(texts, cats):
            by_cat.setdefault(c, []).append(t)
        rng = random.Random(seed)
        pairs = [(t, c) for c in sorted(by_cat) for t in rng.sample(by_cat[c], min(k, len(by_cat[c])))]
    random.Random(seed + 1).shuffle(pairs)
    return [p[0] for p in pairs], [p[1] for p in pairs]
