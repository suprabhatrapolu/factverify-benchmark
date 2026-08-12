"""Checkpoint-2 resolution: settle LogReg C ∈ {4, 16} on FEVER VALIDATION.

Project rule (report session, Checkpoint-2 review): CV macro-F1 gains below
0.01 are treated as noise, and such ties are broken by macro-F1 on the FEVER
validation split — never test. This script refits both candidate
configurations on the full train set, scores them on validation, selects the
winner, and records it through the unified harness (results.json, prediction
CSVs, hyperparams.csv).
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness
from src.features import tfidf

CANDIDATE_CS = [4, 16]


def main() -> None:
    logger = utils.setup_logging(3)
    utils.set_seed()

    X_train, y_train = tfidf.get_train_features(logger)
    vectorizer = tfidf.get_vectorizer(logger)
    val = pd.read_parquet(config.FEVER_PARQUET["val"])
    X_val = vectorizer.transform(val["input_text"])

    fitted = {}
    for c in CANDIDATE_CS:
        model = LogisticRegression(random_state=config.SEED, C=c,
                                   **config.LOGREG_FIXED)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        val_f1 = f1_score(val["label"], model.predict(X_val),
                          labels=config.LABELS, average="macro",
                          zero_division=0)
        fitted[c] = {"model": model, "train_time": train_time, "val_f1": val_f1}
        logger.info("logreg C=%s: full-train refit %.1f s, FEVER val "
                    "macro_f1=%.4f", c, train_time, val_f1)

    best_c = max(CANDIDATE_CS, key=lambda c: fitted[c]["val_f1"])
    logger.info("val tiebreaker selects C=%s (val macro-F1 %.4f vs %.4f)",
                best_c, fitted[best_c]["val_f1"],
                fitted[min(CANDIDATE_CS, key=lambda c: fitted[c]["val_f1"])]["val_f1"])

    winner = fitted[best_c]["model"]
    model_path = config.MODELS_STORE_DIR / "logreg.joblib"
    joblib.dump(winner, model_path)

    def predict_fn(df: pd.DataFrame):
        return winner.predict(vectorizer.transform(df["input_text"]))

    harness.evaluate_and_record(
        "logreg", "Logistic Regression (TF-IDF)", "classical", predict_fn,
        best_hyperparams={"C": best_c},
        train_time_s=fitted[best_c]["train_time"],
        model_size_mb=utils.file_size_mb(model_path, tfidf.VECTORIZER_PATH),
        train_size_used=X_train.shape[0],
        logger=logger)
    utils.append_hyperparams(
        "logreg", "C",
        "[0.25, 1, 4] + extension [4, 8, 16]; CV gap < 0.01 -> FEVER-val tiebreaker",
        best_c)


if __name__ == "__main__":
    main()
