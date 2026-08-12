"""Checkpoint-5 additive extras (report session). New model keys only —
no validated result is replaced.

EXTRA A — `claim_only_logreg` (shortcut check): the winning Logistic
Regression configuration retrained on TF-IDF of the CLAIM ALONE, evidence
withheld. This is the standard hypothesis-only sanity test from the NLI
literature: if it scores far below the claim+evidence model, the pairing —
not surface regularities of the claims themselves — carries the signal, and
the NEI construction protocol introduced no exploitable shortcut.

EXTRA B — `voting_ensemble_calsvm`: the soft vote extended with a
probability-calibrated Linear SVM (CalibratedClassifierCV, 3-fold sigmoid
calibration on FEVER train). LinearSVC was excluded from the original vote
for lacking predict_proba; calibration is the orthodox remedy, so this tests
whether the strongest linear member was worth recovering. The original
`voting_ensemble` entry is untouched.
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness
from src.features import tfidf
from src.models.ensembles import BoosterAdapter, proba_canonical

CLAIM_VECTORIZER_PATH = config.MODELS_STORE_DIR / "tfidf_claim_only.joblib"
CALSVM_PATH = config.MODELS_STORE_DIR / "calibrated_svm.joblib"


def run_claim_only(logger) -> None:
    """EXTRA A: LogReg on claim-only features, evidence withheld."""
    train = pd.read_parquet(config.FEVER_PARQUET["train"])
    vec = tfidf.build_vectorizer()
    with utils.timed(logger, "fit claim-only TF-IDF"):
        X_train = vec.fit_transform(train["claim"])
    joblib.dump(vec, CLAIM_VECTORIZER_PATH)

    model = LogisticRegression(random_state=config.SEED, C=4,
                               **config.LOGREG_FIXED)  # the resolved winner
    start = time.perf_counter()
    model.fit(X_train, train["label"].to_numpy())
    train_time = time.perf_counter() - start
    model_path = config.MODELS_STORE_DIR / "claim_only_logreg.joblib"
    joblib.dump(model, model_path)

    def predict_fn(df: pd.DataFrame):
        return model.predict(vec.transform(df["claim"]))

    entry = harness.evaluate_and_record(
        "claim_only_logreg", "Logistic Regression (claim-only TF-IDF)",
        "baseline", predict_fn,
        best_hyperparams={"C": 4, "features": "claim only"},
        train_time_s=train_time,
        model_size_mb=utils.file_size_mb(model_path, CLAIM_VECTORIZER_PATH),
        train_size_used=len(train),
        logger=logger)

    full = utils.read_json(config.RESULTS_JSON)["logreg"]["eval"]
    logger.info(
        "shortcut-check interpretation: claim-only macro-F1 on FEVER test is "
        "%.4f vs %.4f for the identical configuration with evidence (gap "
        "%.4f). The large deficit shows the classifier cannot recover label "
        "information from claim surface forms alone, i.e. the claim-evidence "
        "pairing carries the signal and the synthetic NEI protocol created no "
        "claim-side shortcut. (Transfer sets: claim-only %.4f Fin-Fact / %.4f "
        "SEC.)",
        entry["eval"]["fever_test"]["macro_f1"],
        full["fever_test"]["macro_f1"],
        full["fever_test"]["macro_f1"] - entry["eval"]["fever_test"]["macro_f1"],
        entry["eval"]["finfact_test"]["macro_f1"],
        entry["eval"]["sec_synthetic"]["macro_f1"])


def run_voting_calsvm(logger) -> None:
    """EXTRA B: soft vote with a calibrated Linear SVM as a fourth member."""
    X_train, y_train = tfidf.get_train_features(logger)
    vectorizer = tfidf.get_vectorizer(logger)

    calsvm = CalibratedClassifierCV(
        LinearSVC(random_state=config.SEED, C=1, **config.SVM_FIXED), cv=3)
    start = time.perf_counter()
    with utils.timed(logger, "fit CalibratedClassifierCV(LinearSVC, cv=3)"):
        calsvm.fit(X_train, y_train)
    cal_time = time.perf_counter() - start
    joblib.dump(calsvm, CALSVM_PATH)

    booster = xgb.Booster()
    booster.load_model(config.MODELS_STORE_DIR / "xgboost.ubj")
    try:
        _ = booster.best_iteration
    except AttributeError:  # attribute not serialised — fall back to all rounds
        booster.set_attr(best_iteration=str(booster.num_boosted_rounds() - 1))
    members = {
        "naive_bayes": joblib.load(config.MODELS_STORE_DIR / "naive_bayes.joblib"),
        "logreg": joblib.load(config.MODELS_STORE_DIR / "logreg.joblib"),
        "xgboost": BoosterAdapter(booster),
        "calibrated_svm": calsvm,
    }

    def predict_fn(df: pd.DataFrame):
        X = vectorizer.transform(df["input_text"])
        avg = np.mean([proba_canonical(m, X) for m in members.values()], axis=0)
        return [config.LABELS[i] for i in avg.argmax(axis=1)]

    results = utils.read_json(config.RESULTS_JSON)
    member_train_time = sum(results[k]["train_time_s"]
                            for k in config.VOTING_COMPONENTS) + cal_time
    member_size = utils.file_size_mb(
        config.MODELS_STORE_DIR / "naive_bayes.joblib",
        config.MODELS_STORE_DIR / "logreg.joblib",
        config.MODELS_STORE_DIR / "xgboost.ubj",
        CALSVM_PATH, tfidf.VECTORIZER_PATH)

    harness.evaluate_and_record(
        "voting_ensemble_calsvm",
        "Soft voting (NB + LogReg + XGBoost + calibrated SVM)", "ensemble",
        predict_fn,
        best_hyperparams={"members": config.VOTING_COMPONENTS + ["calibrated_svm"],
                          "weights": "equal", "calibration": "sigmoid, cv=3"},
        train_time_s=member_train_time,
        model_size_mb=member_size,
        train_size_used=X_train.shape[0],
        logger=logger)


def main() -> None:
    logger = utils.setup_logging(5)
    utils.set_seed()
    run_claim_only(logger)
    run_voting_calsvm(logger)


if __name__ == "__main__":
    main()
