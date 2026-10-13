import pytest

from lorabench import data
from lorabench.metrics import accuracy, macro_f1


def test_committed_data_is_banking77():
    tx, tc = data.load("train")
    sx, sc = data.load("test")
    assert (len(tx), len(sx), len(set(tc))) == (10003, 3080, 77)


def test_subsample_is_k_per_intent_and_seeded():
    tx, tc = data.load("train")
    a = data.subsample(tx, tc, 5, seed=0)
    b = data.subsample(tx, tc, 5, seed=0)
    c = data.subsample(tx, tc, 5, seed=1)
    assert a == b and a != c
    assert len(a[0]) == 5 * 77 and all(a[1].count(cat) == 5 for cat in set(tc))


def test_metrics():
    assert accuracy(["a", "b", "b"], ["a", "b", "a"]) == pytest.approx(2 / 3)
    assert macro_f1(["a", "a", "b"], ["a", "a", "b"]) == 1.0
    assert macro_f1(["a", "b"], ["b", "a"]) == 0.0


torch = pytest.importorskip("torch")
pytest.importorskip("peft")


def tiny():
    """A two-layer BERT and a word-level tokenizer built in memory, so the
    LoRA wiring can be tested without downloading the real backbone."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, PreTrainedTokenizerFast

    words = "freeze my card lost stolen refund charge dispute where is transfer money".split()
    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]")
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=3)
    return tok, cfg


def test_lora_freezes_the_backbone_and_trains_the_head():
    from lorabench import finetune
    tok, cfg = tiny()
    full = finetune.build("full", 3, config=cfg)
    lora = finetune.build("lora", 3, r=2, config=cfg)  # rank 2: the toy model is only 32 wide
    assert finetune.trainable(lora) < finetune.trainable(full) / 5
    names = [n for n, p in lora.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in names) and any("classifier" in n for n in names)
    assert not any("word_embeddings" in n for n in names)
    assert finetune.saved_bytes(lora) < finetune.saved_bytes(full)


def test_lora_learns_a_separable_toy_task():
    from lorabench import finetune
    tok, cfg = tiny()
    texts = ["freeze my card", "my card lost", "stolen card"] * 8 + ["refund charge", "dispute charge"] * 8 + ["transfer money", "where is money"] * 8
    labels = [0] * 24 + [1] * 16 + [2] * 16
    model = finetune.build("lora", 3, config=cfg)
    stats = finetune.train(model, tok, texts, labels, lr=5e-3, batch=8, epochs=15)
    assert stats["steps"] == 15 * 7
    assert accuracy(labels, finetune.predict(model, tok, texts)) >= 0.9


def test_epochs_scale_with_data_size():
    from lorabench.finetune import epochs_for
    assert epochs_for(5 * 77, 32) == 30
    assert epochs_for(10003, 32) == 3


def test_report_renders():
    from lorabench.run import report
    rows = []
    for m in ("tfidf", "frozen", "lora", "full"):
        for k, n in ((10, 770), ("all", 10003)):
            rows.append({"method": m, "k": k, "seed": 0, "n_train": n, "accuracy": 0.8, "macro_f1": 0.79,
                         "train_s": 1.0, "trainable_params": 10, "ms_per_message": 2.0, "saved_bytes": 1_000_000})
    text = report(rows)
    assert "LoRA on MiniLM" in text and "10/intent (770)" in text
