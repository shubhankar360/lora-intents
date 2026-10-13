"""Learning curves: accuracy against labelled examples per intent, for four methods.

    python -m lorabench.run            full grid, writes results/ (about 45 min on a 4-core laptop CPU)
    python -m lorabench.run --quick    k=5 and k=10, one seed, no full fine-tune (CI-sized)
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

from . import baselines, data, finetune
from .encoder import BACKBONE, embed, tokenizer
from .metrics import accuracy, macro_f1

GRID = [5, 10, 20, 50, None]          # examples per intent; None = the full training set
MULTI_SEED = {5: (0, 1, 2), 10: (0, 1, 2)}
FULL_FT_AT = {10, None}               # full fine-tuning is the slow arm; two points are enough to place it
LR = {"lora": 1e-3, "full": 5e-5}


def one_at_a_time_ms(fn, texts) -> float:
    """Median latency classifying one message at a time, as a live router would."""
    times = []
    for t in texts:
        t0 = time.perf_counter()
        fn([t])
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def run(grid=GRID, multi_seed=MULTI_SEED, full_ft_at=FULL_FT_AT, latency_sample: int = 100) -> list[dict]:
    data.download()
    train_x, train_c = data.load("train")
    test_x, test_c = data.load("test")
    names = data.labels()
    to_id = {c: i for i, c in enumerate(names)}
    tok = tokenizer()
    test_emb = embed(test_x)
    sample = test_x[:latency_sample]
    rows = []

    def record(method, k, seed, n, pred, stats, latency, extra=None):
        rows.append({"method": method, "k": k or "all", "seed": seed, "n_train": n,
                     "accuracy": round(accuracy(test_c, pred), 4), "macro_f1": round(macro_f1(test_c, pred), 4),
                     "train_s": round(stats["train_s"], 1), "trainable_params": stats["trainable_params"],
                     "ms_per_message": round(latency, 2), **(extra or {})})
        print(f"{method:>7} k={k or 'all':<4} seed={seed} acc={rows[-1]['accuracy']:.3f} train={stats['train_s']:.0f}s", flush=True)

    for k in grid:
        for seed in multi_seed.get(k, (0,)):
            sx, sc = data.subsample(train_x, train_c, k, seed)
            ids = [to_id[c] for c in sc]

            predict, stats = baselines.tfidf_lr(sx, sc)
            record("tfidf", k, seed, len(sx), predict(test_x), stats, one_at_a_time_ms(predict, sample))

            predict, stats = baselines.frozen_lr(sx, sc)
            record("frozen", k, seed, len(sx), predict(test_x, test_emb), stats, one_at_a_time_ms(predict, sample))

            for method in ("lora", "full"):
                if method == "full" and (k not in full_ft_at or seed != 0):
                    continue
                model = finetune.build(method, len(names))
                stats = finetune.train(model, tok, sx, ids, lr=LR[method], seed=seed)
                stats["trainable_params"] = finetune.trainable(model)
                pred = [names[i] for i in finetune.predict(model, tok, test_x)]
                latency = one_at_a_time_ms(lambda t: finetune.predict(model, tok, t), sample)
                record(method, k, seed, len(sx), pred, stats, latency,
                       {"saved_bytes": finetune.saved_bytes(model), "steps": stats["steps"], "epochs": stats["epochs"]})
    return rows


def _cell(rows, metric="accuracy"):
    v = [r[metric] for r in rows]
    if not v:
        return "–"
    return f"{statistics.mean(v):.1%} ± {statistics.stdev(v):.1%}" if len(v) > 1 else f"{v[0]:.1%}"


def report(rows: list[dict]) -> str:
    ks = list(dict.fromkeys(r["k"] for r in rows))
    methods = ["tfidf", "frozen", "lora", "full"]
    label = {"tfidf": "TF-IDF + logistic regression", "frozen": "frozen MiniLM + logistic regression",
             "lora": "LoRA on MiniLM (r=8)", "full": "full fine-tune of MiniLM"}
    n_of = {r["k"]: r["n_train"] for r in rows}
    out = ["# lora-intents results", "",
           f"Banking77 test set (3,080 messages, 77 intents). Backbone: `{BACKBONE}`. CPU only: "
           f"{platform.machine()}, Python {platform.python_version()}. Where three seeds ran, cells show mean ± standard deviation; "
           "each seed draws a different training subset and every method in that row trains on the same one.", "",
           "## Accuracy by labelled examples per intent", "",
           "| method | " + " | ".join(f"{k}/intent ({n_of[k]:,})" if k != "all" else f"all ({n_of[k]:,})" for k in ks) + " |",
           "| --- |" + " --- |" * len(ks)]
    for m in methods:
        out.append(f"| {label[m]} | " + " | ".join(_cell([r for r in rows if r["method"] == m and r["k"] == k]) for k in ks) + " |")
    out += ["", "## Macro-F1", "", "| method | " + " | ".join(str(k) for k in ks) + " |", "| --- |" + " --- |" * len(ks)]
    for m in methods:
        out.append(f"| {label[m]} | " + " | ".join(_cell([r for r in rows if r["method"] == m and r["k"] == k], "macro_f1") for k in ks) + " |")

    k_last = ks[-1]
    out += ["", f"## What each one costs (trained on {k_last if k_last == 'all' else str(k_last) + '/intent'})", "",
            "| method | trainable parameters | what you ship | training time | one message, CPU |", "| --- | --- | --- | --- | --- |"]
    for m in methods:
        rs = [r for r in rows if r["method"] == m and r["k"] == k_last and r["seed"] == 0]
        if not rs:
            continue
        r = rs[0]
        ship = f"{r['saved_bytes'] / 1e6:.1f} MB" if "saved_bytes" in r else "a few MB (vectoriser + weights)" if m == "tfidf" else "the 90 MB backbone + classifier"
        if m == "lora":
            ship += " adapter (+ the shared 90 MB backbone)"
        out.append(f"| {label[m]} | {r['trainable_params']:,} | {ship} | {r['train_s']:.0f} s | {r['ms_per_message']:.1f} ms |")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args(argv)
    if a.quick:
        rows = run(grid=[5, 10], multi_seed={}, full_ft_at=set(), latency_sample=20)
        print(report(rows))
        return
    rows = run()
    (data.ROOT / "results" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (data.ROOT / "results" / "report.md").write_text(report(rows))
    print(report(rows))


if __name__ == "__main__":
    main()
