"""Unified evaluation harness (brief §10.1) — used by EVERY model.

`evaluate_and_record()` takes a model key, its metadata, and a single
`predict_fn(df) -> list[str]` callable (end-to-end: raw claim/evidence text in,
canonical label strings out, featurisation included). It then:

  1. evaluates on every eval set in config.EVAL_SETS (FEVER test, Fin-Fact,
     SEC-synthetic) — accuracy, macro/weighted F1, per-class P/R/F1/support,
     and the 3x3 confusion matrix in canonical label order (rows = true);
  2. writes one predictions CSV per eval set
     (output/predictions/<model_key>__<eval_set>.csv);
  3. times batch inference on the same fixed 2,000-example FEVER-test slice
     for every model (samples/s therefore includes featurisation cost);
  4. merges the model's entry into output/results/results.json immediately,
     so a later crash can never lose a completed result.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils


def evaluate_predictions(y_true, y_pred) -> dict:
    """All contract metrics for one eval set, canonical label order."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=config.LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=config.LABELS,
                                         average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=config.LABELS,
                                            average="weighted", zero_division=0)), 4),
        "per_class": {
            label: {"precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1[i]), 4),
                    "support": int(support[i])}
            for i, label in enumerate(config.LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=config.LABELS).tolist(),
    }


def measure_inference_speed(predict_fn) -> float:
    """Batch samples/s on the fixed FEVER-test slice (same for every model)."""
    df = pd.read_parquet(config.EVAL_SETS["fever_test"])
    df_slice = df.head(config.EFFICIENCY_SLICE_SIZE)
    start = time.perf_counter()
    predict_fn(df_slice)
    elapsed = time.perf_counter() - start
    return round(len(df_slice) / elapsed, 1)


def evaluate_and_record(model_key: str, display_name: str, tier: str,
                        predict_fn, *, best_hyperparams: dict,
                        train_time_s: float, model_size_mb: float,
                        train_size_used: int, logger) -> dict:
    """Run the full contract evaluation for one model and persist everything."""
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "display_name": display_name,
        "tier": tier,
        "best_hyperparams": best_hyperparams,
        "train_time_s": round(float(train_time_s), 1),
        "model_size_mb": model_size_mb,
        "inference_samples_per_s": None,  # filled below
        "train_size_used": int(train_size_used),
        "eval": {},
    }
    for eval_key, path in config.EVAL_SETS.items():
        df = pd.read_parquet(path)
        y_pred = np.asarray(predict_fn(df))
        entry["eval"][eval_key] = evaluate_predictions(df["label"], y_pred)
        logger.info("%s on %s: acc=%.4f macro_f1=%.4f", model_key, eval_key,
                    entry["eval"][eval_key]["accuracy"],
                    entry["eval"][eval_key]["macro_f1"])
        pred_df = pd.DataFrame({"id": df["id"], "y_true": df["label"],
                                "y_pred": y_pred, "claim": df["claim"]})
        if "perturbation_type" in df.columns:
            pred_df["perturbation_type"] = df["perturbation_type"]
        pred_df.to_csv(config.PREDICTIONS_DIR / f"{model_key}__{eval_key}.csv",
                       index=False)

    entry["inference_samples_per_s"] = measure_inference_speed(predict_fn)
    logger.info("%s inference: %.1f samples/s on %d-example FEVER-test slice",
                model_key, entry["inference_samples_per_s"],
                config.EFFICIENCY_SLICE_SIZE)

    utils.update_results(model_key, entry)
    logger.info("%s written to results.json", model_key)
    return entry
