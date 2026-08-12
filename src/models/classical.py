"""Classical tier: Multinomial NB, Logistic Regression, Linear SVM (brief §6.3).

Protocol (documented for the report): hyperparameters are selected with 3-fold
GridSearchCV (scoring = macro-F1) on a 60k stratified subsample of FEVER train
— grid-searching saga-solver logistic regression on the full 228k rows would
multiply wall time ~4x for no expected ranking change — then the best
configuration is refit on the FULL train set and that refit is what gets
evaluated, timed, and shipped. All randomness is pinned to config.SEED.

Also produces fig_04: top-15 TF-IDF features per class from the fitted
LinearSVC's coefficients.
"""

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness
from src.features import tfidf

MODEL_SPECS = [
    {
        "key": "naive_bayes",
        "display_name": "Multinomial Naive Bayes (TF-IDF)",
        "estimator": lambda: MultinomialNB(),
        "grid": config.NB_GRID,
    },
    {
        "key": "logreg",
        "display_name": "Logistic Regression (TF-IDF)",
        "estimator": lambda: LogisticRegression(random_state=config.SEED,
                                                **config.LOGREG_FIXED),
        "grid": config.LOGREG_GRID,
    },
    {
        "key": "linear_svm",
        "display_name": "Linear SVM (TF-IDF)",
        "estimator": lambda: LinearSVC(random_state=config.SEED,
                                       **config.SVM_FIXED),
        "grid": config.SVM_GRID,
    },
]


def subsample_for_grid(X, y, logger):
    """60k stratified subsample of FEVER train used only for grid search."""
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=config.GRID_SUBSAMPLE_SIZE, stratify=y,
        random_state=config.SEED)
    logger.info("grid-search subsample: %d rows, label mix %s",
                X_sub.shape[0],
                dict(zip(*np.unique(y_sub, return_counts=True))))
    return X_sub, y_sub


def run_model(spec, X_train, y_train, X_grid, y_grid, vectorizer, logger):
    """Grid-search one model, refit on full train, evaluate via the harness."""
    key = spec["key"]

    with utils.timed(logger, f"{key} grid search"):
        search = GridSearchCV(spec["estimator"](), spec["grid"],
                              cv=config.GRID_CV_FOLDS,
                              scoring=config.GRID_SCORING, n_jobs=-1)
        search.fit(X_grid, y_grid)
    for params, score in zip(search.cv_results_["params"],
                             search.cv_results_["mean_test_score"]):
        logger.info("%s cv: %s -> macro_f1=%.4f", key, params, score)
    logger.info("%s best: %s (cv macro_f1=%.4f)", key, search.best_params_,
                search.best_score_)
    for param, values in spec["grid"].items():
        utils.append_hyperparams(key, param, values, search.best_params_[param])

    model = spec["estimator"]().set_params(**search.best_params_)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    logger.info("%s refit on full train (%d rows) in %.1f s", key,
                X_train.shape[0], train_time)

    model_path = config.MODELS_STORE_DIR / f"{key}.joblib"
    joblib.dump(model, model_path)

    def predict_fn(df: pd.DataFrame):
        return model.predict(vectorizer.transform(df["input_text"]))

    harness.evaluate_and_record(
        key, spec["display_name"], "classical", predict_fn,
        best_hyperparams=search.best_params_,
        train_time_s=train_time,
        # Deployment size = classifier + the shared TF-IDF vectorizer it needs.
        model_size_mb=utils.file_size_mb(model_path, tfidf.VECTORIZER_PATH),
        train_size_used=X_train.shape[0],
        logger=logger)
    return model


def plot_svm_top_features(model, vectorizer, logger) -> None:
    """fig_04: top-15 highest-weight features per class from LinearSVC."""
    utils.apply_style()
    feature_names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, label in zip(axes, config.LABELS):
        class_idx = list(model.classes_).index(label)
        coefs = model.coef_[class_idx]
        top = np.argsort(coefs)[-15:]
        ax.barh(range(15), coefs[top], color=utils.CLASS_COLORS[label])
        ax.set_yticks(range(15), feature_names[top], fontsize=9)
        ax.set_title(label)
        ax.set_xlabel("SVM coefficient")
    fig.suptitle("Linear SVM: top-15 TF-IDF features per class", y=1.02)
    utils.save_figure(fig, "fig_04_tfidf_top_features")
    logger.info("saved fig_04_tfidf_top_features")


def main() -> None:
    logger = utils.setup_logging(2)
    utils.set_seed()

    with utils.timed(logger, "load shared TF-IDF features"):
        X_train, y_train = tfidf.get_train_features(logger)
        vectorizer = tfidf.get_vectorizer(logger)
    X_grid, y_grid = subsample_for_grid(X_train, y_train, logger)

    svm_model = None
    for spec in MODEL_SPECS:
        model = run_model(spec, X_train, y_train, X_grid, y_grid, vectorizer,
                          logger)
        if spec["key"] == "linear_svm":
            svm_model = model

    plot_svm_top_features(svm_model, vectorizer, logger)


if __name__ == "__main__":
    main()
