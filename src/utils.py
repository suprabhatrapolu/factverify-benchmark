"""Shared utilities: seeding, timing, IO helpers, and the figure style guide.

Every module in src/ imports from here so that seeding, JSON/CSV writing and
plot styling are identical across the whole project.
"""

import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering; figures are only saved to disk
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ---------------------------------------------------------------------------
# Colour constants (figure style guide, brief §11)
# ---------------------------------------------------------------------------
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
CLASS_COLORS = {
    "SUPPORTS": "#009E73",
    "REFUTES": "#D55E00",
    "NOT ENOUGH INFO": "#0072B2",
}


def set_seed(seed: int = config.SEED) -> None:
    """Seed python's `random` and numpy. Framework-specific seeds
    (TensorFlow/PyTorch) are set inside the modules that import them."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(phase: int) -> logging.Logger:
    """Return a logger writing to both stdout and logs/phase_<N>.log."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"phase_{phase}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:  # avoid duplicate handlers on repeated calls
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(config.LOGS_DIR / f"phase_{phase}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str):
    """Context manager logging the wall time of the enclosed block."""
    start = time.perf_counter()
    logger.info("START %s", label)
    yield
    logger.info("DONE  %s (%.1f s)", label, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# JSON helpers (incremental contract writes, brief §1.3)
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    """Read a JSON file, returning {} if it does not exist yet."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_json(path: Path, obj: dict) -> None:
    """Write JSON atomically (tmp file + replace) so a crash mid-write can
    never corrupt previously saved results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def update_results(model_key: str, entry: dict) -> None:
    """Merge one model's entry into output/results/results.json immediately."""
    results = read_json(config.RESULTS_JSON)
    results[model_key] = entry
    write_json(config.RESULTS_JSON, results)


def append_hyperparams(model: str, param: str, values_searched, best_value) -> None:
    """Upsert one row of output/tables/hyperparams.csv (keyed on model+param).

    Re-running a phase overwrites that model's rows instead of duplicating them.
    """
    import pandas as pd

    path = config.TABLES_DIR / "hyperparams.csv"
    row = {"model": model, "param": param,
           "values_searched": str(values_searched), "best_value": str(best_value)}
    if path.exists():
        table = pd.read_csv(path)
        table = table[~((table["model"] == model) & (table["param"] == param))]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def file_size_mb(*paths: Path) -> float:
    """Total size of the given files in MB (missing files count as 0)."""
    total = sum(p.stat().st_size for p in paths if p.exists())
    return round(total / 1024**2, 2)


# ---------------------------------------------------------------------------
# Figure style guide (brief §11)
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply the project-wide matplotlib style. Call once per script."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
    })


def save_figure(fig, name: str) -> None:
    """Save a figure to output/figures/ as both 300-dpi PNG and PDF twin."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(config.FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, title: str, fig_name: str) -> None:
    """Confusion-matrix figure per the style guide: Blues colormap, each cell
    annotated with the raw count and the row-normalised percentage."""
    apply_style()
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, 100 * cm / row_sums, 0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    short = ["SUPPORTS", "REFUTES", "NEI"]
    ax.set_xticks(range(3), short)
    ax.set_yticks(range(3), short)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.grid(False)
    threshold = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > threshold else "black")
    save_figure(fig, fig_name)


# ---------------------------------------------------------------------------
# Text helpers shared across data modules
# ---------------------------------------------------------------------------

def token_count(text: str) -> int:
    """Whitespace token count (used for dataset statistics)."""
    return len(text.split())


def token_overlap(text_a: str, text_b: str) -> float:
    """|A ∩ B| / |A| over lowercased whitespace token sets (0 if A empty)."""
    a = set(text_a.lower().split())
    b = set(text_b.lower().split())
    return len(a & b) / len(a) if a else 0.0
