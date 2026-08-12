"""Ablation studies (brief §9.1): training size and vocabulary.

Training-size ablation: the winning LogReg (C=4) and LinearSVC (C=1)
configurations retrained on seeded stratified subsets of FEVER train
(1k / 5k / 20k / 60k / full), scored by macro-F1 on FEVER test → fig_13
(log x), with DistilBERT's single 40k-subset point added for context. The
shared full-train TF-IDF vectorizer is held FIXED across sizes so the curve
isolates the effect of supervision, not vocabulary coverage.

Vocabulary ablation: LinearSVC (C=1) with the vectorizer refit at
max_features ∈ {5k, 20k, 50k, 100k}, unigram-only vs uni+bigram → fig_14.

Both write output/tables/ablation_results.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.features import tfidf

ABLATION_MODELS = {
    "logreg": lambda: LogisticRegression(random_state=config.SEED, C=4,
                                         **config.LOGREG_FIXED),
    "linear_svm": lambda: LinearSVC(random_state=config.SEED, C=1,
                                    **config.SVM_FIXED),
}


def fever_test_labels_and_features(vectorizer):
    test = pd.read_parquet(config.FEVER_PARQUET["test"])
    return vectorizer.transform(test["input_text"]), test["label"]


def train_size_ablation(logger) -> list:
    """Macro-F1 on FEVER test vs number of training rows."""
    X_train, y_train = tfidf.get_train_features(logger)
    vectorizer = tfidf.get_vectorizer(logger)
    X_test, y_test = fever_test_labels_and_features(vectorizer)

    rows = []
    for size in config.ABLATION_TRAIN_SIZES:
        if size is None:
            X_sub, y_sub, label = X_train, y_train, X_train.shape[0]
        else:
            X_sub, _, y_sub, _ = train_test_split(
                X_train, y_train, train_size=size, stratify=y_train,
                random_state=config.SEED)
            label = size
        for name, factory in ABLATION_MODELS.items():
            with utils.timed(logger, f"train-size ablation: {name} @ {label}"):
                model = factory().fit(X_sub, y_sub)
            score = f1_score(y_test, model.predict(X_test),
                             labels=config.LABELS, average="macro",
                             zero_division=0)
            logger.info("train_size=%s %s: fever_test macro_f1=%.4f",
                        label, name, score)
            rows.append({"ablation": "train_size", "model": name,
                         "train_size": label, "vocab_size": "", "ngrams": "",
                         "macro_f1_fever_test": round(score, 4)})
    return rows


def vocab_ablation(logger) -> list:
    """LinearSVC macro-F1 on FEVER test vs vocabulary size and n-gram range."""
    train = pd.read_parquet(config.FEVER_PARQUET["train"])
    test = pd.read_parquet(config.FEVER_PARQUET["test"])
    rows = []
    for ngram_range, ngram_label in [((1, 1), "unigram"), ((1, 2), "uni+bigram")]:
        for vocab_size in config.ABLATION_VOCAB_SIZES:
            vec = tfidf.build_vectorizer()
            vec.set_params(max_features=vocab_size, ngram_range=ngram_range)
            with utils.timed(logger,
                             f"vocab ablation: {ngram_label} @ {vocab_size}"):
                X_train = vec.fit_transform(train["input_text"])
                model = LinearSVC(random_state=config.SEED, C=1,
                                  **config.SVM_FIXED).fit(X_train,
                                                          train["label"])
            score = f1_score(test["label"],
                             model.predict(vec.transform(test["input_text"])),
                             labels=config.LABELS, average="macro",
                             zero_division=0)
            logger.info("vocab=%d %s: fever_test macro_f1=%.4f", vocab_size,
                        ngram_label, score)
            rows.append({"ablation": "vocab", "model": "linear_svm",
                         "train_size": len(train), "vocab_size": vocab_size,
                         "ngrams": ngram_label,
                         "macro_f1_fever_test": round(score, 4)})
    return rows


def plot_train_size(rows: list, logger) -> None:
    """fig_13: macro-F1 vs training-set size, log x, DistilBERT for context."""
    utils.apply_style()
    frame = pd.DataFrame([r for r in rows if r["ablation"] == "train_size"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, label in [("logreg", "Logistic Regression"),
                        ("linear_svm", "Linear SVM")]:
        sub = frame[frame["model"] == name].sort_values("train_size")
        ax.plot(sub["train_size"], sub["macro_f1_fever_test"], marker="o",
                label=label)
    distilbert = utils.read_json(config.RESULTS_JSON)["distilbert"]
    ax.scatter([distilbert["train_size_used"]],
               [distilbert["eval"]["fever_test"]["macro_f1"]], marker="*",
               s=220, color=utils.PALETTE[2], zorder=5,
               label="DistilBERT (40k subset)")
    ax.set_xscale("log")
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("FEVER-test macro-F1")
    ax.set_title("Macro-F1 vs training-set size")
    ax.legend()
    utils.save_figure(fig, "fig_13_ablation_train_size")
    logger.info("saved fig_13_ablation_train_size")


def plot_vocab(rows: list, logger) -> None:
    """fig_14: macro-F1 vs vocabulary size, unigram vs uni+bigram."""
    utils.apply_style()
    frame = pd.DataFrame([r for r in rows if r["ablation"] == "vocab"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ngram_label in ["unigram", "uni+bigram"]:
        sub = frame[frame["ngrams"] == ngram_label].sort_values("vocab_size")
        ax.plot(sub["vocab_size"], sub["macro_f1_fever_test"], marker="o",
                label=ngram_label)
    ax.set_xlabel("TF-IDF max_features")
    ax.set_ylabel("FEVER-test macro-F1")
    ax.set_title("Linear SVM: macro-F1 vs vocabulary size")
    ax.legend()
    utils.save_figure(fig, "fig_14_ablation_vocab")
    logger.info("saved fig_14_ablation_vocab")


def main() -> None:
    logger = utils.setup_logging(5)
    utils.set_seed()
    rows = train_size_ablation(logger) + vocab_ablation(logger)
    table = pd.DataFrame(rows)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / "ablation_results.csv", index=False)
    logger.info("wrote ablation_results.csv (%d rows)", len(table))
    plot_train_size(rows, logger)
    plot_vocab(rows, logger)


if __name__ == "__main__":
    main()
