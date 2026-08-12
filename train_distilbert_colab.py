"""Standalone Colab GPU fine-tune of DistilBERT on the FULL FEVER train set.

Self-contained: no project imports. Upload this script together with the five
processed parquet files (fever_train, fever_val, fever_test, finfact_test,
sec_synthetic_test — see logs/distilbert_colab_handoff.md) into the Colab
working directory, select a GPU runtime, and run:

    python train_distilbert_colab.py

Protocol (identical to the project brief §8.2 GPU path): 227,943-row train,
2 epochs, batch 32, lr 2e-5, fp16, (claim, evidence) pair encoding truncated
at 256 tokens, eval on FEVER validation each epoch, best epoch restored by
validation macro-F1, seed 42 everywhere.

Produces results_distilbert_full.zip containing:
    distilbert_results_fragment.json   results.json contract entry
    distilbert__fever_test.csv         predictions (id, y_true, y_pred, claim)
    distilbert__finfact_test.csv       predictions
    distilbert__sec_synthetic.csv      predictions (+ perturbation_type)
    training_history.json              Trainer log history (for fig_12)
    truncation_rates.json              % of pairs truncated at 256, per set
"""

import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

SEED = 42
MODEL_ID = "distilbert/distilbert-base-uncased"
MAX_LEN = 256
BATCH = 32
EVAL_BATCH = 128
EPOCHS = 2
LR = 2e-5
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}
EVAL_SETS = {"fever_test": "fever_test.parquet",
             "finfact_test": "finfact_test.parquet",
             "sec_synthetic": "sec_synthetic_test.parquet"}
SPEED_SLICE = 2000  # FEVER-test slice for inference-throughput measurement

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class PairDataset(torch.utils.data.Dataset):
    """Tokenised (claim, evidence) pairs with integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


def load(name: str) -> pd.DataFrame:
    path = Path(name)
    assert path.exists(), f"missing {name} — upload the processed parquets"
    return pd.read_parquet(path)


def encode(tokenizer, df: pd.DataFrame) -> dict:
    return dict(tokenizer(df["claim"].tolist(), df["evidence_text"].tolist(),
                          truncation=True, max_length=MAX_LEN))


def metrics_block(y_true, y_pred) -> dict:
    """Contract-shaped metrics for one eval set (canonical label order)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS,
                                         average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=LABELS,
                                            average="weighted",
                                            zero_division=0)), 4),
        "per_class": {label: {"precision": round(float(precision[i]), 4),
                              "recall": round(float(recall[i]), 4),
                              "f1": round(float(f1[i]), 4),
                              "support": int(support[i])}
                      for i, label in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=LABELS).tolist(),
    }


def main() -> None:
    assert torch.cuda.is_available(), "select a GPU runtime in Colab"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    train = load("fever_train.parquet")
    val = load("fever_val.parquet")
    eval_frames = {key: load(name) for key, name in EVAL_SETS.items()}

    # Truncation audit at MAX_LEN, per set (report metric).
    truncation = {}
    for name, df in {"train": train, "fever_val": val, **eval_frames}.items():
        lengths = [len(ids) for ids in tokenizer(
            df["claim"].tolist(), df["evidence_text"].tolist())["input_ids"]]
        truncation[name] = round(100 * float(np.mean(
            [l > MAX_LEN for l in lengths])), 2)
    print("truncation rates (%):", truncation)

    y_train = np.array([LABEL_TO_ID[l] for l in train["label"]])
    y_val = np.array([LABEL_TO_ID[l] for l in val["label"]])
    train_ds = PairDataset(encode(tokenizer, train), y_train)
    val_ds = PairDataset(encode(tokenizer, val), y_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"macro_f1": f1_score(labels, logits.argmax(axis=-1),
                                     average="macro", zero_division=0)}

    args = TrainingArguments(
        output_dir="distilbert_ckpt",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        logging_steps=100,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tokenizer))
    start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start

    trainer.save_model("distilbert_full")
    tokenizer.save_pretrained("distilbert_full")
    model_size = round(sum(f.stat().st_size
                           for f in Path("distilbert_full").rglob("*")
                           if f.is_file()) / 1024**2, 2)

    model.eval()
    device = next(model.parameters()).device

    def predict(df: pd.DataFrame) -> list:
        preds = []
        with torch.no_grad():
            for lo in range(0, len(df), EVAL_BATCH):
                chunk = df.iloc[lo:lo + EVAL_BATCH]
                batch = tokenizer(chunk["claim"].tolist(),
                                  chunk["evidence_text"].tolist(),
                                  truncation=True, max_length=MAX_LEN,
                                  padding=True, return_tensors="pt").to(device)
                preds.extend(model(**batch).logits.argmax(dim=-1).tolist())
        return [LABELS[i] for i in preds]

    entry = {
        "display_name": "DistilBERT (fine-tuned, full FEVER train, Colab GPU)",
        "tier": "deep",
        "best_hyperparams": {"lr": LR, "batch_size": BATCH, "epochs": EPOCHS,
                             "max_length": MAX_LEN, "fp16": True},
        "train_time_s": round(train_time, 1),
        "model_size_mb": model_size,
        "inference_samples_per_s": None,
        "train_size_used": len(train),
        "eval": {},
        "gpu": torch.cuda.get_device_name(0),
        "truncation_rates_pct": truncation,
    }
    out_files = []
    for key, df in eval_frames.items():
        y_pred = predict(df)
        entry["eval"][key] = metrics_block(df["label"], y_pred)
        print(key, "->", entry["eval"][key]["macro_f1"])
        pred_df = pd.DataFrame({"id": df["id"], "y_true": df["label"],
                                "y_pred": y_pred, "claim": df["claim"]})
        if "perturbation_type" in df.columns:
            pred_df["perturbation_type"] = df["perturbation_type"]
        name = f"distilbert__{key}.csv"
        pred_df.to_csv(name, index=False)
        out_files.append(name)

    fever_slice = eval_frames["fever_test"].head(SPEED_SLICE)
    tic = time.perf_counter()
    predict(fever_slice)
    entry["inference_samples_per_s"] = round(
        len(fever_slice) / (time.perf_counter() - tic), 1)

    Path("distilbert_results_fragment.json").write_text(
        json.dumps({"distilbert": entry}, indent=2))
    Path("training_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2))
    Path("truncation_rates.json").write_text(json.dumps(truncation, indent=2))
    out_files += ["distilbert_results_fragment.json", "training_history.json",
                  "truncation_rates.json"]

    with zipfile.ZipFile("results_distilbert_full.zip", "w",
                         zipfile.ZIP_DEFLATED) as zf:
        for name in out_files:
            zf.write(name)
    print("DONE -> download results_distilbert_full.zip")


if __name__ == "__main__":
    main()
