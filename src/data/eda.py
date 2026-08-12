"""Dataset statistics and EDA figures (brief §5.4).

Reads the processed parquet files and produces:
  output/tables/dataset_stats.csv       per dataset x split statistics
  fig_01_fever_class_distribution      FEVER label counts, grouped bars
  fig_02_finance_class_distribution    Fin-Fact + SEC-synthetic, two panels
  fig_03_length_histograms             claim/evidence token lengths, two panels
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils


def load_all() -> dict:
    """Load every processed dataset as {(dataset, split): DataFrame}."""
    frames = {("FEVER", split): pd.read_parquet(path)
              for split, path in config.FEVER_PARQUET.items()}
    frames[("Fin-Fact", "test")] = pd.read_parquet(config.FINFACT_PARQUET)
    frames[("SEC-synthetic", "test")] = pd.read_parquet(config.SEC_PARQUET)
    return frames


def build_stats(frames: dict) -> pd.DataFrame:
    """One row per dataset x split: size, label mix, token-length stats.

    For FEVER splits the number of rows dropped for empty evidence_text
    (recorded by load_fever.py in run_metadata.json) is included as a column.
    """
    dropped = utils.read_json(config.RUN_METADATA_JSON).get(
        "fever_dropped_empty_evidence", {})
    rows = []
    for (dataset, split), df in frames.items():
        claim_tokens = df["claim"].str.split().str.len()
        evidence_tokens = df["evidence_text"].str.split().str.len()
        counts = df["label"].value_counts()
        row = {"dataset": dataset, "split": split, "rows": len(df)}
        for label in config.LABELS:
            short = label.lower().replace(" ", "_")
            row[f"n_{short}"] = int(counts.get(label, 0))
            row[f"pct_{short}"] = round(100 * counts.get(label, 0) / len(df), 1)
        row.update({
            "claim_tokens_mean": round(claim_tokens.mean(), 1),
            "claim_tokens_median": int(claim_tokens.median()),
            "evidence_tokens_mean": round(evidence_tokens.mean(), 1),
            "evidence_tokens_median": int(evidence_tokens.median()),
            "dropped_empty_evidence": dropped.get(split, 0) if dataset == "FEVER" else 0,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def fig_fever_distribution(frames: dict) -> None:
    """fig_01: FEVER label counts per split, grouped bars."""
    fig, ax = plt.subplots(figsize=(7, 4))
    splits = ["train", "val", "test"]
    x = np.arange(len(config.LABELS))
    width = 0.26
    for i, split in enumerate(splits):
        counts = frames[("FEVER", split)]["label"].value_counts()
        values = [counts.get(label, 0) for label in config.LABELS]
        bars = ax.bar(x + (i - 1) * width, values, width,
                      label=split, color=utils.PALETTE[i])
        ax.bar_label(bars, fmt="{:,.0f}", fontsize=8, padding=2)
    ax.set_xticks(x, config.LABELS)
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of claims")
    ax.set_title("FEVER class distribution by split")
    ax.legend(title=None)
    utils.save_figure(fig, "fig_01_fever_class_distribution")


def fig_finance_distribution(frames: dict) -> None:
    """fig_02: Fin-Fact and SEC-synthetic label counts, two panels."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, dataset in zip(axes, ["Fin-Fact", "SEC-synthetic"]):
        counts = frames[(dataset, "test")]["label"].value_counts()
        values = [counts.get(label, 0) for label in config.LABELS]
        colors = [utils.CLASS_COLORS[label] for label in config.LABELS]
        bars = ax.bar(range(len(config.LABELS)), values, color=colors)
        ax.bar_label(bars, fmt="{:,.0f}", fontsize=9, padding=2)
        ax.set_xticks(range(len(config.LABELS)),
                      ["SUPPORTS", "REFUTES", "NEI"])
        ax.set_xlabel("Label")
        ax.set_title(dataset)
    axes[0].set_ylabel("Number of claims")
    fig.suptitle("Financial evaluation sets: class distribution", y=1.02)
    utils.save_figure(fig, "fig_02_finance_class_distribution")


def fig_length_histograms(frames: dict) -> None:
    """fig_03: claim and evidence token-length histograms, three datasets."""
    datasets = {
        "FEVER": pd.concat([frames[("FEVER", s)] for s in ("train", "val", "test")]),
        "Fin-Fact": frames[("Fin-Fact", "test")],
        "SEC-synthetic": frames[("SEC-synthetic", "test")],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    specs = [("claim", axes[0], 60, "Claim length (tokens, clipped at 60)"),
             ("evidence_text", axes[1], 300, "Evidence length (tokens, clipped at 300)")]
    for column, ax, clip, xlabel in specs:
        for i, (name, df) in enumerate(datasets.items()):
            lengths = df[column].str.split().str.len().clip(upper=clip)
            ax.hist(lengths, bins=40, range=(0, clip), density=True,
                    histtype="stepfilled", alpha=0.35,
                    color=utils.PALETTE[i], edgecolor=utils.PALETTE[i],
                    linewidth=1.2, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
    axes[1].legend()
    fig.suptitle("Token-length distributions across datasets", y=1.02)
    utils.save_figure(fig, "fig_03_length_histograms")


def main() -> None:
    logger = utils.setup_logging(1)
    utils.apply_style()

    frames = load_all()
    stats = build_stats(frames)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = config.TABLES_DIR / "dataset_stats.csv"
    stats.to_csv(stats_path, index=False)
    logger.info("wrote %s:\n%s", stats_path, stats.to_string(index=False))

    fig_fever_distribution(frames)
    fig_finance_distribution(frames)
    fig_length_histograms(frames)
    logger.info("saved figures 01-03")


if __name__ == "__main__":
    main()
