"""Two baselines a fine-tune has to beat to be worth it."""
from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .encoder import embed


def tfidf_lr(train_x, train_y):
    """Word 1-2 grams into logistic regression. Returns a predict(texts) function and stats."""
    t0 = time.perf_counter()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(C=10, max_iter=2000).fit(vec.fit_transform(train_x), train_y)
    stats = {"train_s": time.perf_counter() - t0, "trainable_params": int(clf.coef_.size + clf.intercept_.size)}
    return (lambda texts: list(clf.predict(vec.transform(texts)))), stats


def frozen_lr(train_x, train_y, train_emb=None):
    """The backbone untouched, a linear classifier on its sentence embeddings."""
    t0 = time.perf_counter()
    clf = LogisticRegression(C=10, max_iter=3000).fit(train_emb if train_emb is not None else embed(train_x), train_y)
    stats = {"train_s": time.perf_counter() - t0, "trainable_params": int(clf.coef_.size + clf.intercept_.size)}
    return (lambda texts, emb=None: list(clf.predict(emb if emb is not None else embed(texts)))), stats
