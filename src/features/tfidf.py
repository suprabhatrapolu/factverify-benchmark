"""Shared TF-IDF feature factory (brief §6.1).

One vectorizer configuration used by every classical and ensemble model:
word 1–2 grams, 100k max features, min_df=2, sublinear tf, lowercase. It is
fitted on FEVER train `input_text` ONLY and applied unchanged to every other
split and evaluation set, so no test-set vocabulary ever leaks into features.

The fitted vectorizer and the transformed FEVER train matrix are cached under
models_store/ so Phase 3 (ensembles) reuses them without refitting.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

VECTORIZER_PATH = config.MODELS_STORE_DIR / "tfidf_vectorizer.joblib"
TRAIN_MATRIX_PATH = config.MODELS_STORE_DIR / "X_fever_train.npz"


def build_vectorizer() -> TfidfVectorizer:
    """Construct the (unfitted) shared vectorizer from config.TFIDF_PARAMS."""
    params = dict(config.TFIDF_PARAMS)
    params["dtype"] = np.float32  # stored as a string in config for JSON-ability
    return TfidfVectorizer(**params)


def get_vectorizer(logger=None) -> TfidfVectorizer:
    """Return the fitted shared vectorizer, fitting and caching on first call."""
    if VECTORIZER_PATH.exists():
        return joblib.load(VECTORIZER_PATH)
    train = pd.read_parquet(config.FEVER_PARQUET["train"])
    vectorizer = build_vectorizer()
    start = time.perf_counter()
    vectorizer.fit(train["input_text"])
    if logger:
        logger.info("fitted TF-IDF on %d FEVER train docs (%.1f s, vocab=%d)",
                    len(train), time.perf_counter() - start,
                    len(vectorizer.vocabulary_))
    config.MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return vectorizer


def get_train_features(logger=None):
    """Return (X_train, y_train) for FEVER train, using the cached matrix."""
    vectorizer = get_vectorizer(logger)
    train = pd.read_parquet(config.FEVER_PARQUET["train"])
    if TRAIN_MATRIX_PATH.exists():
        X = sparse.load_npz(TRAIN_MATRIX_PATH)
    else:
        start = time.perf_counter()
        X = vectorizer.transform(train["input_text"])
        if logger:
            logger.info("transformed FEVER train: %s, nnz=%d (%.1f s)",
                        X.shape, X.nnz, time.perf_counter() - start)
        sparse.save_npz(TRAIN_MATRIX_PATH, X)
    return X, train["label"].to_numpy()
