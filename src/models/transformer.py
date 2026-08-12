"""DistilBERT fine-tuning, CPU path (brief §8.2).

This machine has no GPU (run_metadata.json), so the documented CPU fallback
applies: a 40k stratified subset of FEVER train, 2 epochs, batch 16, claims
and evidence encoded as a sentence PAIR truncated at 256 tokens.

Safeguards and extras required by the report session:
  * Truncation audit — before training, the percentage of pairs whose
    untruncated encoding exceeds 256 tokens is logged for the train subset and
    every eval set (Fin-Fact's long evidence makes this a report metric).
  * ETA rule — after 200 optimiser steps the total wall time is projected;
    if it exceeds 12 hours the run aborts cleanly (exit marker "ETA_ABORT")
    and the Colab handoff (train_distilbert_colab.py) becomes the fallback.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils
from src.eval import harness

MODEL_DIR = config.MODELS_STORE_DIR / "distilbert"
CKPT_DIR = config.MODELS_STORE_DIR / "distilbert_ckpt"


class PairDataset(torch.utils.data.Dataset):
    """Tokenised (claim, evidence) pairs with canonical integer labels."""

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


class EtaCallback(TrainerCallback):
    """Project total wall time at step N; abort if it exceeds the cap."""

    def __init__(self, logger):
        self.logger = logger
        self.start = None
        self.aborted = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == config.DISTILBERT_ETA_CHECK_STEP:
            per_step = (time.perf_counter() - self.start) / state.global_step
            projected_h = per_step * state.max_steps / 3600
            self.logger.info("ETA check at step %d: %.2f s/step, %d total "
                             "steps -> projected %.1f h (cap %d h)",
                             state.global_step, per_step, state.max_steps,
                             projected_h, config.DISTILBERT_MAX_PROJECTED_HOURS)
            if projected_h > config.DISTILBERT_MAX_PROJECTED_HOURS:
                self.logger.error("ETA_ABORT: projected wall time exceeds cap; "
                                  "stopping — use the Colab handoff instead")
                self.aborted = True
                control.should_training_stop = True


def tokenize_pairs(tokenizer, df: pd.DataFrame) -> dict:
    """Pair-encode claims and evidence, truncated at the configured length."""
    return dict(tokenizer(df["claim"].tolist(), df["evidence_text"].tolist(),
                          truncation=True,
                          max_length=config.DISTILBERT_MAX_LEN))


def log_truncation_rates(tokenizer, named_frames: dict, logger) -> dict:
    """Report-session forward note: % of pairs longer than max_length."""
    rates = {}
    for name, df in named_frames.items():
        lengths = [len(ids) for ids in tokenizer(
            df["claim"].tolist(), df["evidence_text"].tolist())["input_ids"]]
        rates[name] = round(100 * np.mean(
            [l > config.DISTILBERT_MAX_LEN for l in lengths]), 2)
        logger.info("truncation@%d for %s: %.2f%% of %d pairs "
                    "(untruncated median %d tokens)",
                    config.DISTILBERT_MAX_LEN, name, rates[name], len(df),
                    int(np.median(lengths)))
    return rates


def plot_training_curves(log_history: list, logger) -> None:
    """fig_12: training loss per logging step + eval macro-F1 per epoch."""
    utils.apply_style()
    loss_steps = [(h["step"], h["loss"]) for h in log_history if "loss" in h]
    eval_points = [(h["step"], h["eval_macro_f1"]) for h in log_history
                   if "eval_macro_f1" in h]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(*zip(*loss_steps))
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Training loss")
    axes[1].plot(*zip(*eval_points), marker="o", color=utils.PALETTE[1])
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation macro-F1")
    fig.suptitle("DistilBERT fine-tuning curves (CPU path)", y=1.03)
    utils.save_figure(fig, "fig_12_distilbert_training_curves")
    logger.info("saved fig_12_distilbert_training_curves")


def main() -> None:
    logger = utils.setup_logging(4)
    utils.set_seed()
    torch.manual_seed(config.SEED)
    logger.info("torch threads: %d, cuda: %s", torch.get_num_threads(),
                torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(config.DISTILBERT_MODEL_ID)
    train = pd.read_parquet(config.FEVER_PARQUET["train"])
    val = pd.read_parquet(config.FEVER_PARQUET["val"])

    # CPU path: stratified 40k subset (recorded in run_metadata.json).
    subset, _ = train_test_split(
        train, train_size=config.DISTILBERT_CPU_TRAIN_SIZE,
        stratify=train["label"], random_state=config.SEED)
    logger.info("CPU path: %d-row stratified train subset, label mix %s",
                len(subset), subset["label"].value_counts().to_dict())

    eval_frames = {key: pd.read_parquet(path)
                   for key, path in config.EVAL_SETS.items()}
    log_truncation_rates(tokenizer,
                         {"train_subset_40k": subset, **eval_frames}, logger)

    y_train = np.array([config.LABEL_TO_ID[l] for l in subset["label"]])
    y_val = np.array([config.LABEL_TO_ID[l] for l in val["label"]])
    train_ds = PairDataset(tokenize_pairs(tokenizer, subset), y_train)
    val_ds = PairDataset(tokenize_pairs(tokenizer, val), y_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.DISTILBERT_MODEL_ID, num_labels=len(config.LABELS))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"macro_f1": f1_score(labels, preds, average="macro",
                                     zero_division=0)}

    args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=config.DISTILBERT_CPU_BATCH,
        per_device_eval_batch_size=config.DISTILBERT_EVAL_BATCH,
        learning_rate=config.DISTILBERT_LR,
        num_train_epochs=config.DISTILBERT_EPOCHS,
        logging_steps=50,
        seed=config.SEED,
        data_seed=config.SEED,
        report_to="none",
    )
    eta = EtaCallback(logger)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      callbacks=[eta])

    start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start
    if eta.aborted:
        logger.error("DistilBERT CPU run aborted by ETA rule after %.1f s — "
                     "no results recorded; proceed with the Colab handoff",
                     train_time)
        sys.exit(2)
    logger.info("DistilBERT trained in %.1f s", train_time)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    model_size = round(sum(f.stat().st_size for f in MODEL_DIR.rglob("*")
                           if f.is_file()) / 1024**2, 2)

    model.eval()

    def predict_fn(df: pd.DataFrame):
        preds = []
        with torch.no_grad():
            for lo in range(0, len(df), config.DISTILBERT_EVAL_BATCH):
                chunk = df.iloc[lo:lo + config.DISTILBERT_EVAL_BATCH]
                batch = tokenizer(chunk["claim"].tolist(),
                                  chunk["evidence_text"].tolist(),
                                  truncation=True,
                                  max_length=config.DISTILBERT_MAX_LEN,
                                  padding=True, return_tensors="pt")
                logits = model(**batch).logits
                preds.extend(logits.argmax(dim=-1).tolist())
        return [config.ID_TO_LABEL[i] for i in preds]

    entry = harness.evaluate_and_record(
        "distilbert", "DistilBERT (fine-tuned, CPU 40k subset)", "deep",
        predict_fn,
        best_hyperparams={"lr": config.DISTILBERT_LR,
                          "batch_size": config.DISTILBERT_CPU_BATCH,
                          "epochs": config.DISTILBERT_EPOCHS,
                          "max_length": config.DISTILBERT_MAX_LEN,
                          "train_subset": config.DISTILBERT_CPU_TRAIN_SIZE},
        train_time_s=train_time,
        model_size_mb=model_size,
        train_size_used=config.DISTILBERT_CPU_TRAIN_SIZE,
        logger=logger)

    plot_training_curves(trainer.state.log_history, logger)
    utils.plot_confusion_matrix(
        entry["eval"]["fever_test"]["confusion_matrix"],
        "DistilBERT (CPU 40k subset) — FEVER test", "fig_10_cm_distilbert")
    logger.info("saved fig_10_cm_distilbert")

    metadata = utils.read_json(config.RUN_METADATA_JSON)
    metadata["distilbert_train_size"] = config.DISTILBERT_CPU_TRAIN_SIZE
    utils.write_json(config.RUN_METADATA_JSON, metadata)


if __name__ == "__main__":
    main()
