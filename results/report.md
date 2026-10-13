# lora-intents results

Banking77 test set (3,080 messages, 77 intents). Backbone: `sentence-transformers/all-MiniLM-L6-v2`. CPU only: x86_64, Python 3.12.7. Where three seeds ran, cells show mean ± standard deviation; each seed draws a different training subset and every method in that row trains on the same one.

## Accuracy by labelled examples per intent

| method | 5/intent (385) | 10/intent (770) | 20/intent (1,540) | 50/intent (3,826) | all (10,003) |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 54.5% ± 1.9% | 68.0% ± 0.7% | 77.0% | 85.5% | 89.4% |
| frozen MiniLM + logistic regression | 80.3% ± 0.5% | 85.1% ± 0.4% | 88.3% | 91.4% | 93.0% |
| LoRA on MiniLM (r=8) | 78.0% ± 2.9% | 84.3% ± 0.3% | 87.7% | 90.4% | 92.5% |
| full fine-tune of MiniLM | – | 85.3% | – | – | 93.2% |

## Macro-F1

| method | 5 | 10 | 20 | 50 | all |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 54.0% ± 2.4% | 67.8% ± 1.0% | 76.7% | 85.5% | 89.4% |
| frozen MiniLM + logistic regression | 80.2% ± 0.6% | 85.0% ± 0.4% | 88.2% | 91.3% | 93.0% |
| LoRA on MiniLM (r=8) | 77.9% ± 2.9% | 84.3% ± 0.3% | 87.6% | 90.4% | 92.5% |
| full fine-tune of MiniLM | – | 85.2% | – | – | 93.2% |

## What each one costs (trained on all)

| method | trainable parameters | what you ship | training time | one message, CPU |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 1,817,893 | a few MB (vectoriser + weights) | 19 s | 4.5 ms |
| frozen MiniLM + logistic regression | 29,645 | the 90 MB backbone + classifier | 176 s | 23.5 ms |
| LoRA on MiniLM (r=8) | 361,421 | 1.5 MB adapter (+ the shared 90 MB backbone) | 2181 s | 16.9 ms |
| full fine-tune of MiniLM | 22,595,021 | 90.4 MB | 1264 s | 12.5 ms |
