"""RF fairness check (report session, mid-Phase-3 note).

LogReg and LinearSVC were trained with class_weight="balanced" on the
50%-SUPPORTS-skewed FEVER train set, but the first Random Forest used the
default (no class weighting) — an inconsistent imbalance treatment that could
contaminate the tier comparison. This script retrains the RF with
class_weight="balanced_subsample" under the identical warm-start OOB sweep
protocol, scores BOTH variants on FEVER validation (the project's selection
split — never test), records the val winner as `random_forest` through the
unified harness, and redraws fig_05 with both OOB curves.
"""

import re
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness
from src.features import tfidf
from src.models.ensembles import SVD_PATH, get_svd

UNWEIGHTED_PATH = config.MODELS_STORE_DIR / "random_forest_unweighted.joblib"
BALANCED_PATH = config.MODELS_STORE_DIR / "random_forest_balanced.joblib"
OOB_LINE_RE = re.compile(r"random_forest(?:\[balanced_subsample\])? OOB error at (\d+) trees: ([0-9.]+)")


def baseline_oob_from_log() -> dict:
    """Recover the unweighted RF's OOB curve from phase_3.log (avoid a refit)."""
    text = (config.LOGS_DIR / "phase_3.log").read_text(encoding="utf-8")
    curve = {}
    for line in text.splitlines():
        if "[balanced_subsample]" in line:
            continue  # only the original run's lines
        match = OOB_LINE_RE.search(line)
        if match:
            curve[int(match.group(1))] = float(match.group(2))
    return curve


def train_balanced(X_svd, y_train, logger):
    """Warm-start OOB sweep with class_weight='balanced_subsample'."""
    rf = RandomForestClassifier(bootstrap=True, oob_score=True, n_jobs=-1,
                                warm_start=True, random_state=config.SEED,
                                class_weight="balanced_subsample")
    curve, start = {}, time.perf_counter()
    for n in config.RF_N_ESTIMATORS_SWEEP:
        rf.set_params(n_estimators=n)
        with utils.timed(logger, f"grow balanced RF to {n} trees"):
            rf.fit(X_svd, y_train)
        curve[n] = 1 - rf.oob_score_
        logger.info("random_forest[balanced_subsample] OOB error at %d trees: "
                    "%.4f", n, curve[n])
    return rf, curve, time.perf_counter() - start


def plot_both_curves(baseline_curve: dict, balanced_curve: dict, logger):
    """fig_05 with both variants' OOB curves."""
    utils.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sorted(baseline_curve), [baseline_curve[n] for n in sorted(baseline_curve)],
            marker="o", label="class_weight=None")
    ax.plot(sorted(balanced_curve), [balanced_curve[n] for n in sorted(balanced_curve)],
            marker="s", label="class_weight=balanced_subsample")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("OOB error (1 − OOB accuracy)")
    ax.set_title("Random Forest out-of-bag error vs forest size")
    ax.legend()
    utils.save_figure(fig, "fig_05_rf_oob_error")
    logger.info("saved fig_05_rf_oob_error (both variants)")


def main() -> None:
    logger = utils.setup_logging(3)
    utils.set_seed()

    X_train, y_train = tfidf.get_train_features(logger)
    vectorizer = tfidf.get_vectorizer(logger)
    svd = joblib.load(SVD_PATH)
    X_svd = svd.transform(X_train).astype(np.float32)

    # Preserve the original (unweighted) forest under a distinct name.
    original = joblib.load(config.MODELS_STORE_DIR / "random_forest.joblib")
    joblib.dump(original, UNWEIGHTED_PATH)
    baseline_curve = baseline_oob_from_log()
    logger.info("baseline (unweighted) OOB curve from log: %s", baseline_curve)

    balanced, balanced_curve, balanced_time = train_balanced(X_svd, y_train, logger)
    joblib.dump(balanced, BALANCED_PATH)
    plot_both_curves(baseline_curve, balanced_curve, logger)

    # Val tiebreaker (project rule): winner becomes the recorded RF.
    val = pd.read_parquet(config.FEVER_PARQUET["val"])
    X_val_svd = svd.transform(vectorizer.transform(val["input_text"]))
    scores = {}
    for name, model in [("none", original), ("balanced_subsample", balanced)]:
        scores[name] = f1_score(val["label"], model.predict(X_val_svd),
                                labels=config.LABELS, average="macro",
                                zero_division=0)
        logger.info("RF class_weight=%s: FEVER val macro_f1=%.4f", name,
                    scores[name])
    winner_name = max(scores, key=scores.get)
    winner = original if winner_name == "none" else balanced
    logger.info("val tiebreaker selects RF class_weight=%s", winner_name)

    model_path = config.MODELS_STORE_DIR / "random_forest.joblib"
    joblib.dump(winner, model_path)

    def predict_fn(df: pd.DataFrame):
        return winner.predict(svd.transform(vectorizer.transform(df["input_text"])))

    prior = utils.read_json(config.RESULTS_JSON)["random_forest"]
    harness.evaluate_and_record(
        "random_forest", "Random Forest (TF-IDF + SVD-300)", "ensemble",
        predict_fn,
        best_hyperparams={"n_estimators": config.RF_FINAL_N_ESTIMATORS,
                          "svd_components": config.SVD_COMPONENTS,
                          "class_weight": winner_name,
                          "oob_error_at_400": round(
                              (baseline_curve if winner_name == "none"
                               else balanced_curve)[400], 4)},
        train_time_s=(prior["train_time_s"] if winner_name == "none"
                      else balanced_time),
        model_size_mb=utils.file_size_mb(model_path, SVD_PATH,
                                         tfidf.VECTORIZER_PATH),
        train_size_used=X_train.shape[0],
        logger=logger)
    utils.append_hyperparams("random_forest", "class_weight",
                             ["None", "balanced_subsample"], winner_name)


if __name__ == "__main__":
    main()
