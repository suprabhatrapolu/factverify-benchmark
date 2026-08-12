"""BiLSTM with pre-trained GloVe embeddings (brief §8.1).

Architecture: Embedding(100d, GloVe-initialised, trainable, masked) →
BiLSTM(128) → Dropout(0.3) → Dense(64, relu) → Dense(3, softmax), trained
with Adam 1e-3, batch 256, ≤10 epochs, early stopping (patience 2, restore
best) on FEVER-validation macro-F1, and class weights from the train
distribution.

Implementation notes for the report:
  * Keras 3 (bundled with TF 2.21) removed the legacy `Tokenizer`; the 50k
    vocabulary is built with a TextVectorization layer (frequency-ordered,
    fitted on FEVER train input_text only, sequences padded/truncated to 128).
  * Per-epoch train macro-F1 for fig_11 is measured on a fixed stratified
    20k sample of the train set — a full-train predict every epoch would
    roughly double wall time without changing the curve's story.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.eval import harness

MODEL_PATH = config.MODELS_STORE_DIR / "bilstm.keras"
VECTORIZER_PATH = config.MODELS_STORE_DIR / "bilstm_vectorizer.keras"


def build_vectorizer(train_texts) -> layers.TextVectorization:
    """50k-vocab TextVectorization fitted on FEVER train input_text."""
    vectorizer = layers.TextVectorization(
        max_tokens=config.BILSTM_VOCAB_SIZE,
        output_sequence_length=config.BILSTM_MAX_LEN)
    vectorizer.adapt(tf.constant(train_texts))
    return vectorizer


def build_embedding_matrix(vocab, logger) -> np.ndarray:
    """GloVe-100 initialisation for the vocabulary; seeded normal for OOV."""
    import gensim.downloader
    with utils.timed(logger, f"load {config.GLOVE_GENSIM_NAME} via gensim"):
        glove = gensim.downloader.load(config.GLOVE_GENSIM_NAME)
    rng = np.random.default_rng(config.SEED)
    matrix = rng.normal(0, 0.1, (len(vocab), config.BILSTM_EMBED_DIM)).astype("float32")
    matrix[0] = 0.0  # padding index stays zero
    hits = 0
    for i, word in enumerate(vocab):
        if word in glove:
            matrix[i] = glove[word]
            hits += 1
    oov_rate = 1 - hits / len(vocab)
    logger.info("GloVe coverage: %d/%d vocab words (OOV rate %.2f%%)",
                hits, len(vocab), 100 * oov_rate)
    return matrix


def build_model(embedding_matrix: np.ndarray) -> keras.Model:
    """Assemble and compile the BiLSTM classifier."""
    inputs = keras.Input(shape=(config.BILSTM_MAX_LEN,), dtype="int32")
    x = layers.Embedding(
        embedding_matrix.shape[0], config.BILSTM_EMBED_DIM,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        mask_zero=True, trainable=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(config.BILSTM_UNITS))(x)
    x = layers.Dropout(config.BILSTM_DROPOUT)(x)
    x = layers.Dense(config.BILSTM_DENSE_UNITS, activation="relu")(x)
    outputs = layers.Dense(len(config.LABELS), activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(config.BILSTM_LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class MacroF1Callback(keras.callbacks.Callback):
    """Adds val_macro_f1 (full FEVER val) and train_macro_f1 (fixed 20k
    stratified sample) to the epoch logs so EarlyStopping can monitor them."""

    def __init__(self, X_val, y_val, X_train_sample, y_train_sample, logger):
        super().__init__()
        self.X_val, self.y_val = X_val, y_val
        self.X_ts, self.y_ts = X_train_sample, y_train_sample
        self.logger = logger

    def _macro_f1(self, X, y) -> float:
        pred = self.model.predict(X, batch_size=512, verbose=0).argmax(axis=1)
        return f1_score(y, pred, average="macro", zero_division=0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logs["val_macro_f1"] = self._macro_f1(self.X_val, self.y_val)
        logs["train_macro_f1"] = self._macro_f1(self.X_ts, self.y_ts)
        self.logger.info("epoch %d: loss=%.4f val_loss=%.4f "
                         "train_macro_f1(20k)=%.4f val_macro_f1=%.4f",
                         epoch + 1, logs.get("loss", -1), logs.get("val_loss", -1),
                         logs["train_macro_f1"], logs["val_macro_f1"])


def plot_training_curves(history: dict, logger) -> None:
    """fig_11: loss and macro-F1 per epoch, train vs validation."""
    utils.apply_style()
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_macro_f1"], marker="o",
                 label="train (20k sample)")
    axes[1].plot(epochs, history["val_macro_f1"], marker="s", label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend()
    fig.suptitle("BiLSTM training curves", y=1.03)
    utils.save_figure(fig, "fig_11_bilstm_training_curves")
    logger.info("saved fig_11_bilstm_training_curves")


def main() -> None:
    logger = utils.setup_logging(4)
    keras.utils.set_random_seed(config.SEED)
    utils.set_seed()

    train = pd.read_parquet(config.FEVER_PARQUET["train"])
    val = pd.read_parquet(config.FEVER_PARQUET["val"])
    train_size = len(train)  # full train: int32 sequences fit comfortably in RAM
    logger.info("BiLSTM trains on the FULL FEVER train set (%d rows)", train_size)

    vectorizer = build_vectorizer(train["input_text"].tolist())
    vocab = vectorizer.get_vocabulary()
    logger.info("vocabulary size: %d (cap %d)", len(vocab),
                config.BILSTM_VOCAB_SIZE)

    def encode(texts) -> np.ndarray:
        return vectorizer(tf.constant(list(texts))).numpy().astype("int32")

    X_train = encode(train["input_text"])
    y_train = np.array([config.LABEL_TO_ID[l] for l in train["label"]])
    X_val = encode(val["input_text"])
    y_val = np.array([config.LABEL_TO_ID[l] for l in val["label"]])

    # Fixed stratified 20k sample for the per-epoch train macro-F1 curve.
    sample_idx, _ = train_test_split(
        np.arange(train_size), train_size=config.BILSTM_TRAIN_F1_SAMPLE,
        stratify=y_train, random_state=config.SEED)
    class_counts = np.bincount(y_train)
    class_weight = {i: train_size / (len(config.LABELS) * c)
                    for i, c in enumerate(class_counts)}
    logger.info("class weights: %s", {k: round(v, 3)
                                      for k, v in class_weight.items()})

    embedding_matrix = build_embedding_matrix(vocab, logger)
    model = build_model(embedding_matrix)
    f1_callback = MacroF1Callback(X_val, y_val, X_train[sample_idx],
                                  y_train[sample_idx], logger)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_macro_f1", mode="max", patience=config.BILSTM_PATIENCE,
        restore_best_weights=True)

    start = time.perf_counter()
    fit_history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        batch_size=config.BILSTM_BATCH_SIZE, epochs=config.BILSTM_MAX_EPOCHS,
        class_weight=class_weight, callbacks=[f1_callback, early_stop],
        verbose=2)
    train_time = time.perf_counter() - start
    logger.info("BiLSTM trained for %d epochs in %.1f s",
                len(fit_history.history["loss"]), train_time)

    config.MODELS_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    def predict_fn(df: pd.DataFrame):
        probs = model.predict(encode(df["input_text"]), batch_size=512,
                              verbose=0)
        return [config.LABELS[i] for i in probs.argmax(axis=1)]

    entry = harness.evaluate_and_record(
        "bilstm", "BiLSTM + GloVe-100", "deep", predict_fn,
        best_hyperparams={"lstm_units": config.BILSTM_UNITS,
                          "dropout": config.BILSTM_DROPOUT,
                          "dense_units": config.BILSTM_DENSE_UNITS,
                          "lr": config.BILSTM_LR,
                          "batch_size": config.BILSTM_BATCH_SIZE,
                          "vocab_size": len(vocab),
                          "max_len": config.BILSTM_MAX_LEN,
                          "epochs_trained": len(fit_history.history["loss"])},
        train_time_s=train_time,
        model_size_mb=utils.file_size_mb(MODEL_PATH),
        train_size_used=train_size,
        logger=logger)

    plot_training_curves(fit_history.history, logger)
    utils.plot_confusion_matrix(
        entry["eval"]["fever_test"]["confusion_matrix"],
        "BiLSTM + GloVe-100 — FEVER test", "fig_09_cm_bilstm")
    logger.info("saved fig_09_cm_bilstm")

    metadata = utils.read_json(config.RUN_METADATA_JSON)
    metadata["bilstm_train_size"] = train_size
    utils.write_json(config.RUN_METADATA_JSON, metadata)


if __name__ == "__main__":
    main()
