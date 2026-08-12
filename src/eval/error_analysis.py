"""Error analysis (brief §9.4) on the champion and the best classical model.

Champion: distilbert (no Colab full-train result was integrated before this
phase started). Best classical: logreg — selected by FEVER validation
macro-F1 (0.5705), per the project's val-selection rule.

Outputs:
  sec_perturbation_accuracy.csv   accuracy by perturbation_type per model
                                  (the controlled part: which perturbations
                                  fool which model) — additive table
  error_taxonomy_counts.csv       model x heuristic category counts over
                                  FEVER-test + Fin-Fact misclassifications
  fig_17_error_taxonomy           grouped bars of the above
  (misclassified candidates are printed for curation into
  output/predictions/misclassified_examples.md)

Taxonomy (first matching category wins, in this order — documented choice):
  negation                 claim contains a negation cue (not, no, never,
                           n't, without, fails)
  numeric                  claim contains a digit
  low_overlap_paraphrase   claim-evidence token overlap < 0.2
  nei_confusion            true or predicted label is NOT ENOUGH INFO
  other                    none of the above
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils

MODELS = {"distilbert": "DistilBERT (champion)",
          "logreg": "LogReg (best classical by val)"}
CATEGORIES = ["negation", "numeric", "low_overlap_paraphrase",
              "nei_confusion", "other"]
NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(c for c in config.NEGATION_CUES if c != "n't") + r")\b"
    r"|n't", re.IGNORECASE)


def categorize(claim: str, evidence: str, y_true: str, y_pred: str) -> str:
    """First-match taxonomy assignment (priority documented in module docstring)."""
    if NEGATION_RE.search(claim):
        return "negation"
    if any(ch.isdigit() for ch in claim):
        return "numeric"
    if utils.token_overlap(claim, evidence) < config.LOW_OVERLAP_THRESHOLD:
        return "low_overlap_paraphrase"
    if "NOT ENOUGH INFO" in (y_true, y_pred):
        return "nei_confusion"
    return "other"


def load_errors(model_key: str, eval_key: str, parquet_path) -> pd.DataFrame:
    """Misclassified rows of one predictions file, joined with evidence text."""
    preds = pd.read_csv(config.PREDICTIONS_DIR / f"{model_key}__{eval_key}.csv",
                        dtype={"id": str})
    data = pd.read_parquet(parquet_path)[["id", "evidence_text"]]
    data["id"] = data["id"].astype(str)
    merged = preds.merge(data, on="id", how="left")
    errors = merged[merged["y_true"] != merged["y_pred"]].copy()
    errors["eval_set"] = eval_key
    return errors


def sec_perturbation_accuracy(logger) -> None:
    """Accuracy by perturbation_type on SEC-synthetic, per model."""
    rows = []
    for model_key in MODELS:
        preds = pd.read_csv(
            config.PREDICTIONS_DIR / f"{model_key}__sec_synthetic.csv")
        for ptype, group in preds.groupby("perturbation_type"):
            rows.append({"model": model_key, "perturbation_type": ptype,
                         "n": len(group),
                         "accuracy": round(float(
                             (group["y_true"] == group["y_pred"]).mean()), 4)})
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES_DIR / "sec_perturbation_accuracy.csv",
                 index=False)
    logger.info("wrote sec_perturbation_accuracy.csv:\n%s",
                table.to_string(index=False))


def taxonomy_counts(logger) -> pd.DataFrame:
    """Categorise FEVER-test + Fin-Fact misclassifications for both models."""
    rows = []
    for model_key in MODELS:
        errors = pd.concat([
            load_errors(model_key, "fever_test", config.FEVER_PARQUET["test"]),
            load_errors(model_key, "finfact_test", config.FINFACT_PARQUET)])
        errors["category"] = [
            categorize(c, e if isinstance(e, str) else "", t, p)
            for c, e, t, p in zip(errors["claim"], errors["evidence_text"],
                                  errors["y_true"], errors["y_pred"])]
        for category in CATEGORIES:
            rows.append({"model": model_key, "category": category,
                         "count": int((errors["category"] == category).sum())})
        logger.info("%s: %d misclassifications (FEVER test + Fin-Fact), "
                    "taxonomy %s", model_key, len(errors),
                    errors["category"].value_counts().to_dict())
    table = pd.DataFrame(rows)
    table.to_csv(config.TABLES_DIR / "error_taxonomy_counts.csv", index=False)
    logger.info("wrote error_taxonomy_counts.csv")
    return table


def plot_taxonomy(table: pd.DataFrame, logger) -> None:
    """fig_17: error-category counts, champion vs best classical."""
    utils.apply_style()
    x = np.arange(len(CATEGORIES))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (model_key, label) in enumerate(MODELS.items()):
        sub = table[table["model"] == model_key].set_index("category")
        values = [sub.loc[c, "count"] for c in CATEGORIES]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=label,
                      color=utils.PALETTE[i])
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(x, ["negation", "numeric", "low-overlap\nparaphrase",
                      "NEI\nconfusion", "other"])
    ax.set_ylabel("Misclassified examples")
    ax.set_title("Error taxonomy on FEVER test + Fin-Fact")
    ax.legend()
    utils.save_figure(fig, "fig_17_error_taxonomy")
    logger.info("saved fig_17_error_taxonomy")


def print_candidates(logger) -> None:
    """Print curation candidates: interesting misclassifications per model,
    spread over categories, datasets, and SEC perturbation types."""
    rng = np.random.default_rng(config.SEED)
    for model_key in MODELS:
        errors = pd.concat([
            load_errors(model_key, "fever_test", config.FEVER_PARQUET["test"]),
            load_errors(model_key, "finfact_test", config.FINFACT_PARQUET),
            load_errors(model_key, "sec_synthetic", config.SEC_PARQUET)])
        errors["category"] = [
            categorize(c, e if isinstance(e, str) else "", t, p)
            for c, e, t, p in zip(errors["claim"], errors["evidence_text"],
                                  errors["y_true"], errors["y_pred"])]
        print(f"\n================ candidates for {model_key} ================")
        for category in CATEGORIES:
            sub = errors[errors["category"] == category]
            if not len(sub):
                continue
            take = sub.iloc[rng.choice(len(sub), size=min(3, len(sub)),
                                       replace=False)]
            for _, row in take.iterrows():
                evidence = (row["evidence_text"] or "")[:220]
                ptype = row.get("perturbation_type", "")
                ptype = f" | ptype={ptype}" if isinstance(ptype, str) and ptype else ""
                print(f"--- [{category}] {row['eval_set']} id={row['id']}"
                      f" true={row['y_true']} pred={row['y_pred']}{ptype}\n"
                      f"CLAIM: {row['claim'][:260]}\nEVID : {evidence}")


def main() -> None:
    logger = utils.setup_logging(5)
    utils.set_seed()
    sec_perturbation_accuracy(logger)
    table = taxonomy_counts(logger)
    plot_taxonomy(table, logger)
    print_candidates(logger)


if __name__ == "__main__":
    main()
