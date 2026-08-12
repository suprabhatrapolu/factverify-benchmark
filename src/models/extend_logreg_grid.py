"""Checkpoint-2 addendum: extend the Logistic Regression C search upward.

The original grid {0.25, 1, 4} selected C=4 at the grid edge. Per the report
session's correction, the bracket is extended to {4, 8, 16} under the identical
protocol (3-fold GridSearchCV, macro-F1, same seeded 60k stratified subsample).
Decision rule: a larger C must beat C=4 by more than
config.LOGREG_EXTENSION_MIN_GAIN CV macro-F1 to replace it; otherwise C=4
stands and the confirmation is recorded in hyperparams.csv.
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness
from src.features import tfidf
from src.models.classical import subsample_for_grid


def main() -> None:
    logger = utils.setup_logging(2)
    utils.set_seed()

    X_train, y_train = tfidf.get_train_features(logger)
    vectorizer = tfidf.get_vectorizer(logger)
    X_grid, y_grid = subsample_for_grid(X_train, y_train, logger)

    with utils.timed(logger, "logreg extended grid search C in {4, 8, 16}"):
        search = GridSearchCV(
            LogisticRegression(random_state=config.SEED, **config.LOGREG_FIXED),
            config.LOGREG_EXTENDED_GRID, cv=config.GRID_CV_FOLDS,
            scoring=config.GRID_SCORING, n_jobs=-1)
        search.fit(X_grid, y_grid)

    scores = dict(zip((p["C"] for p in search.cv_results_["params"]),
                      search.cv_results_["mean_test_score"]))
    for c, score in scores.items():
        logger.info("logreg extended cv: C=%s -> macro_f1=%.4f", c, score)
    best_c = max(scores, key=scores.get)
    gain = scores[best_c] - scores[4]
    logger.info("extended-bracket best C=%s, gain over C=4 = %+.4f "
                "(threshold %.3f)", best_c, gain, config.LOGREG_EXTENSION_MIN_GAIN)

    if best_c != 4 and gain > config.LOGREG_EXTENSION_MIN_GAIN:
        logger.info("switching to C=%s: refitting on full train", best_c)
        model = LogisticRegression(random_state=config.SEED, C=best_c,
                                   **config.LOGREG_FIXED)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        model_path = config.MODELS_STORE_DIR / "logreg.joblib"
        joblib.dump(model, model_path)

        def predict_fn(df: pd.DataFrame):
            return model.predict(vectorizer.transform(df["input_text"]))

        harness.evaluate_and_record(
            "logreg", "Logistic Regression (TF-IDF)", "classical", predict_fn,
            best_hyperparams={"C": best_c},
            train_time_s=train_time,
            model_size_mb=utils.file_size_mb(model_path, tfidf.VECTORIZER_PATH),
            train_size_used=X_train.shape[0],
            logger=logger)
        utils.append_hyperparams("logreg", "C",
                                 "[0.25, 1, 4] + extension [4, 8, 16]", best_c)
    else:
        logger.info("keeping C=4: extension did not clear the gain threshold")
        utils.append_hyperparams(
            "logreg", "C",
            "[0.25, 1, 4] + extension [4, 8, 16]",
            f"4 (extended bracket confirmed; best extension gain {gain:+.4f})")


if __name__ == "__main__":
    main()
