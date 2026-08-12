"""Ensemble tier: Random Forest, XGBoost, soft voting (brief §7).

Design choices (documented for the report):
  * Random Forest runs on a TruncatedSVD-300 projection of the TF-IDF matrix —
    axis-aligned trees on 100k sparse dimensions are impractical (each split
    considers sqrt(100k) mostly-zero features), while 300 dense SVD components
    retain the dominant variance at tractable cost. The OOB sweep grows one
    forest incrementally (warm_start) so the 50→400 tree curve costs a single
    400-tree fit.
  * XGBoost consumes the raw sparse TF-IDF directly (hist tree method handles
    sparsity natively) with early stopping on FEVER validation multi-class
    log-loss.
  * The soft-voting ensemble averages predict_proba from Multinomial NB,
    Logistic Regression, and XGBoost with equal weights. LinearSVC is
    deliberately excluded: it is margin-based and offers no calibrated
    predict_proba, so it cannot participate in soft voting.

Also selects the models for figs 07/08 by FEVER *validation* macro-F1
(project rule from the Checkpoint-2 resolution: selection never touches test)
and renders their FEVER-test confusion matrices from results.json.
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness
from src.features import tfidf

SVD_PATH = config.MODELS_STORE_DIR / "svd300.joblib"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def proba_canonical(model, X) -> np.ndarray:
    """predict_proba with columns re-ordered to canonical label order.

    Handles both string-labelled models (NB, LogReg: classes_ sorted
    alphabetically) and integer-labelled ones (XGBoost: canonical class ids).
    """
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    order = [classes.index(label) if label in classes
             else classes.index(config.LABEL_TO_ID[label])
             for label in config.LABELS]
    return proba[:, order]


def get_svd(X_train, logger) -> TruncatedSVD:
    """Fit (or load) the shared TruncatedSVD-300 projection of TF-IDF."""
    if SVD_PATH.exists():
        return joblib.load(SVD_PATH)
    svd = TruncatedSVD(n_components=config.SVD_COMPONENTS,
                       random_state=config.SEED)
    with utils.timed(logger, f"fit TruncatedSVD-{config.SVD_COMPONENTS}"):
        svd.fit(X_train)
    logger.info("SVD explained variance ratio: %.3f",
                svd.explained_variance_ratio_.sum())
    joblib.dump(svd, SVD_PATH)
    return svd


# ---------------------------------------------------------------------------
# Random Forest (on SVD components) with OOB sweep
# ---------------------------------------------------------------------------

def run_random_forest(X_train, y_train, vectorizer, logger):
    """Warm-start OOB sweep to 400 trees, fig_05, harness evaluation."""
    svd = get_svd(X_train, logger)
    model_path = config.MODELS_STORE_DIR / "random_forest.joblib"
    if model_path.exists() and "random_forest" in utils.read_json(config.RESULTS_JSON):
        # Resume path: RF already trained, evaluated, and recorded (results are
        # written incrementally); reload instead of repeating a ~30-min fit.
        logger.info("random_forest already recorded — loading from store")
        return joblib.load(model_path), svd
    X_svd = svd.transform(X_train).astype(np.float32)

    rf = RandomForestClassifier(bootstrap=True, oob_score=True, n_jobs=-1,
                                warm_start=True, random_state=config.SEED)
    oob_errors, start = [], time.perf_counter()
    for n in config.RF_N_ESTIMATORS_SWEEP:
        rf.set_params(n_estimators=n)
        with utils.timed(logger, f"grow random forest to {n} trees"):
            rf.fit(X_svd, y_train)
        oob_errors.append(1 - rf.oob_score_)
        logger.info("random_forest OOB error at %d trees: %.4f", n, oob_errors[-1])
    train_time = time.perf_counter() - start

    utils.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(config.RF_N_ESTIMATORS_SWEEP, oob_errors, marker="o")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB error (1 − OOB accuracy)")
    ax.set_title("Random Forest out-of-bag error vs forest size")
    utils.save_figure(fig, "fig_05_rf_oob_error")
    logger.info("saved fig_05_rf_oob_error")

    joblib.dump(rf, model_path)

    def predict_fn(df: pd.DataFrame):
        return rf.predict(svd.transform(vectorizer.transform(df["input_text"])))

    harness.evaluate_and_record(
        "random_forest", "Random Forest (TF-IDF + SVD-300)", "ensemble",
        predict_fn,
        best_hyperparams={"n_estimators": config.RF_FINAL_N_ESTIMATORS,
                          "svd_components": config.SVD_COMPONENTS,
                          "oob_error_at_400": round(oob_errors[-1], 4)},
        train_time_s=train_time,  # SVD fit logged separately
        model_size_mb=utils.file_size_mb(model_path, SVD_PATH,
                                         tfidf.VECTORIZER_PATH),
        train_size_used=X_train.shape[0],
        logger=logger)
    utils.append_hyperparams("random_forest", "n_estimators",
                             config.RF_N_ESTIMATORS_SWEEP,
                             config.RF_FINAL_N_ESTIMATORS)
    utils.append_hyperparams("random_forest", "svd_components",
                             [config.SVD_COMPONENTS], config.SVD_COMPONENTS)
    return rf, svd


# ---------------------------------------------------------------------------
# XGBoost (on raw sparse TF-IDF) with early stopping
# ---------------------------------------------------------------------------

class BoosterAdapter:
    """Give the native-API Booster the minimal sklearn face (predict_proba,
    classes_) that the soft-voting ensemble and the harness rely on."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster
        self.classes_ = list(range(len(config.LABELS)))  # canonical ids

    def predict_proba(self, X) -> np.ndarray:
        return self.booster.predict(
            xgb.DMatrix(X),
            iteration_range=(0, self.booster.best_iteration + 1))

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def run_xgboost(X_train, y_train, vectorizer, logger):
    """Early-stopped XGBoost on sparse TF-IDF, fig_06, harness evaluation.

    Uses the native API with QuantileDMatrix and bounded histogram memory
    (config.XGB_MAX_BIN / XGB_NTHREAD): the sklearn wrapper's defaults
    over-allocated on this 15.3 GB machine (see phase_3.log).
    """
    val = pd.read_parquet(config.FEVER_PARQUET["val"])
    X_val = vectorizer.transform(val["input_text"])
    y_train_ids = np.array([config.LABEL_TO_ID[l] for l in y_train])
    y_val_ids = np.array([config.LABEL_TO_ID[l] for l in val["label"]])

    dtrain = xgb.QuantileDMatrix(X_train, label=y_train_ids,
                                 max_bin=config.XGB_MAX_BIN)
    dval = xgb.QuantileDMatrix(X_val, label=y_val_ids, ref=dtrain,
                               max_bin=config.XGB_MAX_BIN)
    params = {
        "objective": config.XGB_PARAMS["objective"],
        "num_class": len(config.LABELS),
        "tree_method": config.XGB_PARAMS["tree_method"],
        "max_depth": config.XGB_PARAMS["max_depth"],
        "eta": config.XGB_PARAMS["learning_rate"],
        "max_bin": config.XGB_MAX_BIN,
        "nthread": config.XGB_NTHREAD,
        "max_cached_hist_node": config.XGB_MAX_CACHED_HIST_NODE,
        "eval_metric": "mlogloss",
        "seed": config.SEED,
    }
    start = time.perf_counter()
    with utils.timed(logger, "fit XGBoost (early stopping on FEVER val)"):
        booster = xgb.train(params, dtrain,
                            num_boost_round=config.XGB_PARAMS["n_estimators"],
                            evals=[(dval, "val")],
                            early_stopping_rounds=config.XGB_EARLY_STOPPING_ROUNDS,
                            verbose_eval=False)
    train_time = time.perf_counter() - start
    del dtrain, dval
    model = BoosterAdapter(booster)
    logger.info("xgboost best_iteration=%d (of max %d), val mlogloss=%.4f",
                booster.best_iteration, config.XGB_PARAMS["n_estimators"],
                float(booster.best_score))

    # fig_06: top-30 gain importances with real token names.
    gains = booster.get_score(importance_type="gain")
    feature_names = vectorizer.get_feature_names_out()
    top = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)[:30]
    names = [feature_names[int(k[1:])] for k, _ in top][::-1]
    values = [v for _, v in top][::-1]
    utils.apply_style()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(range(len(names)), values, color=utils.PALETTE[0])
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("Average split gain")
    ax.set_title("XGBoost: top-30 features by gain")
    utils.save_figure(fig, "fig_06_xgb_feature_importance")
    logger.info("saved fig_06_xgb_feature_importance")

    model_path = config.MODELS_STORE_DIR / "xgboost.ubj"
    booster.save_model(model_path)

    def predict_fn(df: pd.DataFrame):
        ids = model.predict(vectorizer.transform(df["input_text"]))
        return [config.ID_TO_LABEL[i] for i in ids]

    harness.evaluate_and_record(
        "xgboost", "XGBoost (TF-IDF)", "ensemble", predict_fn,
        best_hyperparams={**{k: v for k, v in config.XGB_PARAMS.items()
                             if k != "objective"},
                          "max_bin": config.XGB_MAX_BIN,
                          "best_iteration": int(booster.best_iteration)},
        train_time_s=train_time,
        model_size_mb=utils.file_size_mb(model_path, tfidf.VECTORIZER_PATH),
        train_size_used=X_train.shape[0],
        logger=logger)
    for param in ("max_depth", "learning_rate", "n_estimators"):
        utils.append_hyperparams("xgboost", param, [config.XGB_PARAMS[param]],
                                 config.XGB_PARAMS[param])
    utils.append_hyperparams("xgboost", "best_iteration",
                             f"early stopping, patience "
                             f"{config.XGB_EARLY_STOPPING_ROUNDS} on FEVER val",
                             int(booster.best_iteration))
    return model


# ---------------------------------------------------------------------------
# Soft-voting ensemble
# ---------------------------------------------------------------------------

def run_voting(xgb_model, vectorizer, logger):
    """Equal-weight soft vote of NB + LogReg + XGBoost via the harness."""
    members = {
        "naive_bayes": joblib.load(config.MODELS_STORE_DIR / "naive_bayes.joblib"),
        "logreg": joblib.load(config.MODELS_STORE_DIR / "logreg.joblib"),
        "xgboost": xgb_model,
    }
    assert set(members) == set(config.VOTING_COMPONENTS)

    def predict_fn(df: pd.DataFrame):
        X = vectorizer.transform(df["input_text"])
        avg = np.mean([proba_canonical(m, X) for m in members.values()], axis=0)
        return [config.LABELS[i] for i in avg.argmax(axis=1)]

    # No fitting of its own: training cost = sum of the members' train times.
    results = utils.read_json(config.RESULTS_JSON)
    member_train_time = sum(results[k]["train_time_s"]
                            for k in config.VOTING_COMPONENTS)
    member_size = utils.file_size_mb(
        config.MODELS_STORE_DIR / "naive_bayes.joblib",
        config.MODELS_STORE_DIR / "logreg.joblib",
        config.MODELS_STORE_DIR / "xgboost.ubj",
        tfidf.VECTORIZER_PATH)

    harness.evaluate_and_record(
        "voting_ensemble", "Soft voting (NB + LogReg + XGBoost)", "ensemble",
        predict_fn,
        best_hyperparams={"members": config.VOTING_COMPONENTS,
                          "weights": "equal"},
        train_time_s=member_train_time,
        model_size_mb=member_size,
        train_size_used=results["xgboost"]["train_size_used"],
        logger=logger)
    return members


# ---------------------------------------------------------------------------
# figs 07/08: best classical / best ensemble by FEVER val macro-F1
# ---------------------------------------------------------------------------

def select_and_plot_cms(rf, svd, xgb_model, voting_members, vectorizer, logger):
    """Score every candidate on FEVER val, pick winners, plot test CMs."""
    val = pd.read_parquet(config.FEVER_PARQUET["val"])
    X_val = vectorizer.transform(val["input_text"])
    X_val_svd = svd.transform(X_val)

    def val_f1(y_pred) -> float:
        return f1_score(val["label"], y_pred, labels=config.LABELS,
                        average="macro", zero_division=0)

    scores = {}
    for key in ("naive_bayes", "logreg", "linear_svm"):
        model = joblib.load(config.MODELS_STORE_DIR / f"{key}.joblib")
        scores[key] = val_f1(model.predict(X_val))
    scores["random_forest"] = val_f1(rf.predict(X_val_svd))
    scores["xgboost"] = val_f1([config.ID_TO_LABEL[i]
                                for i in xgb_model.predict(X_val)])
    avg = np.mean([proba_canonical(m, X_val) for m in voting_members.values()],
                  axis=0)
    scores["voting_ensemble"] = val_f1([config.LABELS[i]
                                        for i in avg.argmax(axis=1)])
    logger.info("FEVER val macro-F1 (selection scores): %s",
                {k: round(v, 4) for k, v in scores.items()})

    results = utils.read_json(config.RESULTS_JSON)
    best_classical = max(("naive_bayes", "logreg", "linear_svm"),
                         key=scores.get)
    best_ensemble = max(("random_forest", "xgboost", "voting_ensemble"),
                        key=scores.get)
    logger.info("best classical by val: %s; best ensemble by val: %s",
                best_classical, best_ensemble)
    for key, fig_name, group in [(best_classical, "fig_07_cm_best_classical",
                                  "best classical"),
                                 (best_ensemble, "fig_08_cm_best_ensemble",
                                  "best ensemble")]:
        entry = results[key]
        utils.plot_confusion_matrix(
            entry["eval"]["fever_test"]["confusion_matrix"],
            f"{entry['display_name']} — FEVER test\n"
            f"({group} by FEVER val macro-F1)", fig_name)
        logger.info("saved %s (%s)", fig_name, key)


def main() -> None:
    logger = utils.setup_logging(3)
    utils.set_seed()

    with utils.timed(logger, "load shared TF-IDF features"):
        X_train, y_train = tfidf.get_train_features(logger)
        vectorizer = tfidf.get_vectorizer(logger)

    rf, svd = run_random_forest(X_train, y_train, vectorizer, logger)
    xgb_model = run_xgboost(X_train, y_train, vectorizer, logger)
    voting_members = run_voting(xgb_model, vectorizer, logger)
    select_and_plot_cms(rf, svd, xgb_model, voting_members, vectorizer, logger)


if __name__ == "__main__":
    main()
