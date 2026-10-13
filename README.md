# lora-intents

[![ci](https://github.com/shubhankar360/lora-intents/actions/workflows/ci.yml/badge.svg)](https://github.com/shubhankar360/lora-intents/actions/workflows/ci.yml)

How many labelled examples does a LoRA adapter need before it beats the cheap options? Measured on Banking77 (77 card and banking intents) on a laptop CPU, against TF-IDF, a frozen sentence encoder with logistic regression, and a full fine-tune.

I built it because "fine-tune a small model" is the usual answer to intent routing, and I wanted to know where that answer starts being right. A support agent's router (like the keyword one in [handoff-graph](https://github.com/shubhankar360/handoff-graph)) needs one label per message, fast and cheap, and the labelled data is the expensive part.

## Setup

- **Data:** Banking77, 10,003 training and 3,080 test messages (PolyAI, CC BY 4.0, committed under `data/`). Every method trains on the same subset: k examples per intent, drawn with a fixed seed, for k = 5, 10, 20, 50 and all.
- **Backbone:** `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 6 layers).
- **Methods:**
  - TF-IDF (word 1–2 grams) and logistic regression.
  - Frozen MiniLM: mean-pooled sentence embeddings, logistic regression on top.
  - LoRA (rank 8 on the attention and feed-forward projections) with a new linear head.
  - Full fine-tune of every weight, at 10 per intent and on all the data.
- **Same head everywhere.** All three MiniLM methods read the same mean-pooled embedding, and the new classifier gets the same learning rate in both fine-tunes. My first version used BERT's [CLS] pooler head for the fine-tunes, which this sentence model was never trained to use; that would have made fine-tuning look worse for a reason unrelated to fine-tuning.
- **Training:** plain PyTorch, AdamW with warm-up and linear decay, batch 32, 390 to 484 updates for each subset (more epochs when there's less data) and 3 epochs (939 updates) on the full set. LoRA learning rate 1e-3, full fine-tune 5e-5.
- **Seeds:** three seeds at 5 and 10 per intent, where the variance is largest. Each seed draws a different training subset.

## Results

Accuracy on the 3,080 test messages ([results/report.md](results/report.md) has macro-F1 too, and every run is in [results/runs.jsonl](results/runs.jsonl)). Cells with ± are the mean and standard deviation over three seeds.

| examples per intent | 5 | 10 | 20 | 50 | all (10,003) |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 54.5% ± 1.9 | 68.0% ± 0.7 | 77.0% | 85.5% | 89.4% |
| frozen MiniLM + logistic regression | **80.3% ± 0.5** | **85.1% ± 0.4** | **88.3%** | **91.4%** | 93.0% |
| LoRA on MiniLM (rank 8) | 78.0% ± 2.9 | 84.3% ± 0.3 | 87.7% | 90.4% | 92.5% |
| full fine-tune of MiniLM | | 85.3% | | | **93.2%** |

What it costs, trained on all the data, on this laptop:

| | trainable parameters | per-task artefact | training time |
| --- | --- | --- | --- |
| TF-IDF + logistic regression | 1.8M | a few MB | 19 s |
| frozen MiniLM + logistic regression | 29,645 | about 120 KB of weights, on the shared backbone | 176 s, almost all of it embedding the data once |
| LoRA on MiniLM | 361,421 | 1.5 MB adapter, on the shared backbone | 36 min |
| full fine-tune | 22.6M | a 90 MB copy of the model | 21 min |

What I take from it:

- **The frozen encoder was the one to beat, and LoRA didn't beat it.** At every data size, logistic regression on the untouched MiniLM embeddings matched or beat the LoRA adapter, while training in a couple of minutes instead of half an hour. At 5 examples per intent LoRA was also the least stable: its accuracy moved 2.9 points between seeds, against 0.5 for the frozen encoder.
- **Updating the backbone only paid off with all the data, and barely.** The full fine-tune edged past the frozen encoder at 10 per intent (85.3% vs 85.1%) and on the full set (93.2% vs 93.0%), each on a single seed, so both gaps are within the noise.
- **LoRA's usual selling point, a small file per task, doesn't separate it here.** If many tasks share one backbone, a frozen-encoder classifier per task (about 120 KB) is less than a tenth of the size of an adapter (1.5 MB). I'd expect adapters to earn their keep where the sentence embedding can't separate the classes, which isn't the case on Banking77.
- **On a CPU, LoRA didn't train faster than a full fine-tune.** Gradients still flow back through the whole network; LoRA only shrinks what the optimiser updates. At 10 per intent both took 455 s. On the full set LoRA took longer, partly because the laptop was busier during that run, so treat the times as rough.
- **TF-IDF needed between 4 and 10 times the data** to match what the frozen encoder got from 5 examples per intent (77.0% at 20 per intent and 85.5% at 50, against 80.3%).

So for a router like the one in handoff-graph: start with a frozen sentence encoder and a linear classifier, label a few hundred messages, and only fine-tune when the confusion matrix shows the embedding itself is the problem.

## Limits

- One dataset, one backbone, one rank. Banking77 has short, clean messages and fine-grained intents; a messier dataset or a bigger backbone could move the crossover.
- Hyperparameters were set once, not tuned per method or per data size. A higher rank, more epochs or a learning-rate search might close the gap for LoRA; a tuned logistic regression would also improve. One seed above 10 per intent.
- CPU only, on a 2020 MacBook Air, which is why the full grid takes hours. Times are for comparison between methods, not absolute.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt     # torch 2.2.2 is the last build for Intel Macs; any CPU build works elsewhere
python -m lorabench.run --quick      # 5 and 10 per intent, one seed, no full fine-tune
python -m lorabench.run              # the full grid; rewrites results/ (hours on a laptop CPU)
python -m pytest -q                  # 7 tests; the LoRA ones use a tiny in-memory BERT, no download
```

CI runs the tests on Python 3.10 and 3.12 and the `--quick` run with the real backbone on CPU.

## Layout

```
lorabench/data.py       Banking77 loading and per-intent subsampling
lorabench/encoder.py    MiniLM tokenizer and mean-pooled embeddings
lorabench/baselines.py  TF-IDF and frozen-encoder logistic regression
lorabench/finetune.py   mean-pooled classifier, LoRA and full fine-tune, training loop
lorabench/run.py        the grid and the report
data/                   Banking77 CSVs and attribution
results/                report.md and every run in runs.jsonl
```

MIT licensed. Banking77 is CC BY 4.0 (Casanueva et al., 2020).
