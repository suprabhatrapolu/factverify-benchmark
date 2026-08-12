"""Transfer and efficiency summaries (brief §9.2–§9.3).

Both are pure aggregations of results.json — every number was produced by the
unified harness during the model phases; nothing is re-measured here.

  transfer_results.csv + fig_15   macro-F1 for every recorded model on the
                                  three evaluation sets (grouped bars)
  efficiency.csv + fig_16         size / train time / inference throughput vs
                                  FEVER-test macro-F1 (labelled scatter,
                                  marker area ∝ model size)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils

# Canonical display order: baselines, classical, ensembles, deep, extras.
MODEL_ORDER = ["lexical_baseline", "claim_only_logreg", "naive_bayes",
               "logreg", "linear_svm", "random_forest", "xgboost",
               "voting_ensemble", "voting_ensemble_calsvm", "bilstm",
               "distilbert", "distilbert_full"]
SHORT_NAMES = {
    "lexical_baseline": "Lexical", "claim_only_logreg": "Claim-only LR",
    "naive_bayes": "NB", "logreg": "LogReg", "linear_svm": "SVM",
    "random_forest": "RF", "xgboost": "XGBoost",
    "voting_ensemble": "Voting", "voting_ensemble_calsvm": "Voting+calSVM",
    "bilstm": "BiLSTM", "distilbert": "DistilBERT",
    "distilbert_full": "DistilBERT-full",
}
EVAL_LABELS = {"fever_test": "FEVER test", "finfact_test": "Fin-Fact",
               "sec_synthetic": "SEC-synthetic"}


def ordered_models(results: dict) -> list:
    """All recorded model keys in canonical display order."""
    return [k for k in MODEL_ORDER if k in results] + \
           [k for k in results if k not in MODEL_ORDER]


def build_transfer(results: dict, logger) -> pd.DataFrame:
    rows = [{"model": key,
             "display_name": results[key]["display_name"],
             **{eval_key: results[key]["eval"][eval_key]["macro_f1"]
                for eval_key in config.EVAL_SETS}}
            for key in ordered_models(results)]
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES_DIR / "transfer_results.csv", index=False)
    logger.info("wrote transfer_results.csv:\n%s", table.to_string(index=False))
    return table


def plot_transfer(table: pd.DataFrame, logger) -> None:
    """fig_15: grouped bars, models x eval sets, macro-F1."""
    utils.apply_style()
    n = len(table)
    x = np.arange(n)
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(9, 1.05 * n), 4.8))
    for i, eval_key in enumerate(config.EVAL_SETS):
        ax.bar(x + (i - 1) * width, table[eval_key], width,
               label=EVAL_LABELS[eval_key], color=utils.PALETTE[i])
    ax.set_xticks(x, [SHORT_NAMES.get(k, k) for k in table["model"]],
                  rotation=20, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("In-domain vs financial transfer, all models")
    ax.legend()
    utils.save_figure(fig, "fig_15_transfer_grouped_bars")
    logger.info("saved fig_15_transfer_grouped_bars")


def build_efficiency(results: dict, logger) -> pd.DataFrame:
    rows = [{"model": key,
             "tier": results[key]["tier"],
             "model_size_mb": results[key]["model_size_mb"],
             "train_time_s": results[key]["train_time_s"],
             "inference_samples_per_s": results[key]["inference_samples_per_s"],
             "fever_test_macro_f1": results[key]["eval"]["fever_test"]["macro_f1"]}
            for key in ordered_models(results)]
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES_DIR / "efficiency.csv", index=False)
    logger.info("wrote efficiency.csv:\n%s", table.to_string(index=False))
    return table


def plot_efficiency(table: pd.DataFrame, logger) -> None:
    """fig_16: throughput (log x) vs macro-F1, marker area ∝ size on disk."""
    utils.apply_style()
    tier_colors = {"baseline": utils.PALETTE[3], "classical": utils.PALETTE[0],
                   "ensemble": utils.PALETTE[1], "deep": utils.PALETTE[2]}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sizes = 30 + 12 * np.sqrt(table["model_size_mb"].clip(lower=0.01))
    for tier, color in tier_colors.items():
        sub = table[table["tier"] == tier]
        ax.scatter(sub["inference_samples_per_s"], sub["fever_test_macro_f1"],
                   s=sizes[sub.index], color=color, alpha=0.85, label=tier,
                   edgecolors="white", linewidth=0.8)
    # Per-point label placement: the mid-throughput cluster (LogReg / SVM /
    # both voting ensembles / XGBoost) needs fanned-out offsets to stay legible.
    offsets = {"logreg": (9, -3, "left"), "linear_svm": (9, -15, "left"),
               "voting_ensemble": (-9, 12, "right"),
               "voting_ensemble_calsvm": (9, 9, "left"),
               "xgboost": (-9, -18, "right"),
               "claim_only_logreg": (-9, 10, "right")}
    for _, row in table.iterrows():
        dx, dy, ha = offsets.get(row["model"], (7, 5, "left"))
        ax.annotate(SHORT_NAMES.get(row["model"], row["model"]),
                    (row["inference_samples_per_s"],
                     row["fever_test_macro_f1"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Inference throughput (samples/s, log scale)")
    ax.set_ylabel("FEVER-test macro-F1")
    ax.set_title("Cost vs accuracy (marker area ∝ model size on disk)")
    ax.legend(title=None, loc="lower left")
    utils.save_figure(fig, "fig_16_cost_vs_accuracy")
    logger.info("saved fig_16_cost_vs_accuracy")


def main() -> None:
    logger = utils.setup_logging(5)
    results = utils.read_json(config.RESULTS_JSON)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    transfer = build_transfer(results, logger)
    plot_transfer(transfer, logger)
    efficiency = build_efficiency(results, logger)
    plot_efficiency(efficiency, logger)


if __name__ == "__main__":
    main()
