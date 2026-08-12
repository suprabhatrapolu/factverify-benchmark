"""Lexical overlap baseline (brief §6.2) — the motivating weak system.

Predicts from a single statistic: the fraction of (stopword-free) claim tokens
that also appear in the evidence. Two thresholds t_low <= t_high are tuned on
FEVER validation over a 0.05–0.95 grid (step 0.05):

    overlap >= t_high  ->  SUPPORTS
    overlap <  t_low   ->  NOT ENOUGH INFO
    otherwise          ->  REFUTES

The mid-band-as-REFUTES assignment reflects that a refuting evidence passage
usually shares topic words with the claim without matching it fully. The model
is intentionally blind to word order, negation, and numbers — Chapter 3's
motivating failure case.
"""

import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness

TOKEN_RE = re.compile(r"[a-z0-9']+")
MODEL_PATH = config.MODELS_STORE_DIR / "lexical_baseline.joblib"


def overlap_ratio(claim: str, evidence: str) -> float:
    """|claim ∩ evidence| / |claim| over lowercased, stopword-free token sets."""
    claim_tokens = set(TOKEN_RE.findall(claim.lower())) - ENGLISH_STOP_WORDS
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(TOKEN_RE.findall(evidence.lower()))
    return len(claim_tokens & evidence_tokens) / len(claim_tokens)


def compute_overlaps(df: pd.DataFrame) -> np.ndarray:
    """Vector of overlap ratios for a claim–evidence DataFrame."""
    return np.array([overlap_ratio(c, e) for c, e
                     in zip(df["claim"], df["evidence_text"])])


def classify(overlaps: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """Apply the two-threshold decision rule."""
    predictions = np.full(len(overlaps), "REFUTES", dtype=object)
    predictions[overlaps >= t_high] = "SUPPORTS"
    predictions[overlaps < t_low] = "NOT ENOUGH INFO"
    return predictions


def tune_thresholds(logger) -> tuple[float, float, float]:
    """Grid-search (t_low, t_high) on FEVER validation, maximising macro-F1."""
    val = pd.read_parquet(config.FEVER_PARQUET["val"])
    overlaps = compute_overlaps(val)
    grid = np.round(np.arange(config.LEXICAL_THRESHOLD_GRID_START,
                              config.LEXICAL_THRESHOLD_GRID_STOP + 1e-9,
                              config.LEXICAL_THRESHOLD_GRID_STEP), 2)
    best = (0.0, grid[0], grid[0])
    for t_low in grid:
        for t_high in grid[grid >= t_low]:
            score = f1_score(val["label"], classify(overlaps, t_low, t_high),
                             labels=config.LABELS, average="macro",
                             zero_division=0)
            if score > best[0]:
                best = (score, t_low, t_high)
    logger.info("lexical thresholds tuned on FEVER val: t_low=%.2f t_high=%.2f "
                "(val macro-F1=%.4f)", best[1], best[2], best[0])
    return best[1], best[2], best[0]


def main() -> None:
    logger = utils.setup_logging(2)
    utils.set_seed()

    start = time.perf_counter()
    t_low, t_high, val_f1 = tune_thresholds(logger)
    tune_time = time.perf_counter() - start

    config.MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"t_low": t_low, "t_high": t_high}, MODEL_PATH)
    grid_desc = (f"{config.LEXICAL_THRESHOLD_GRID_START}-"
                 f"{config.LEXICAL_THRESHOLD_GRID_STOP} step "
                 f"{config.LEXICAL_THRESHOLD_GRID_STEP}")
    utils.append_hyperparams("lexical_baseline", "t_low", grid_desc, t_low)
    utils.append_hyperparams("lexical_baseline", "t_high", grid_desc, t_high)

    def predict_fn(df: pd.DataFrame):
        return classify(compute_overlaps(df), t_low, t_high)

    harness.evaluate_and_record(
        "lexical_baseline", "Lexical overlap baseline", "baseline", predict_fn,
        best_hyperparams={"t_low": t_low, "t_high": t_high},
        train_time_s=tune_time,  # threshold tuning is this model's "training"
        model_size_mb=utils.file_size_mb(MODEL_PATH),
        train_size_used=0,  # rule-based; tuned on validation, no training set
        logger=logger)


if __name__ == "__main__":
    main()
