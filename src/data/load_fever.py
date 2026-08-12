"""Download and flatten the FEVER gold-evidence dataset (brief §5.1).

Loads all three splits of `copenlu/fever_gold_evidence`, flattens each
example's evidence triples ([page, sent_id, text]) into a single evidence
string, cleans Wikipedia tokenisation artifacts, and saves one parquet file
per split with columns: id, claim, evidence_text, input_text, label.

Note (for the report's dataset docs): for NOT ENOUGH INFO claims the evidence
in this dataset is *retrieved* (nearest-page) rather than gold-annotated,
because FEVER annotators mark NEI claims without evidence.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils

# Wikipedia tokenisation artifacts present in FEVER evidence text.
WIKI_ARTIFACTS = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
}

# HF split name -> our split name (used in file names and stats).
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}


def clean_text(text: str) -> str:
    """Replace Wikipedia bracket artifacts and collapse whitespace."""
    for artifact, replacement in WIKI_ARTIFACTS.items():
        text = text.replace(artifact, replacement)
    return re.sub(r"\s+", " ", text).strip()


def flatten_evidence(evidence: list) -> str:
    """Join the text element of each [page, sent_id, text] triple with a space.

    The text element is the last item of each triple; malformed or empty
    triples are skipped.
    """
    parts = []
    for triple in evidence:
        if isinstance(triple, (list, tuple)) and triple:
            parts.append(str(triple[-1]))
        elif isinstance(triple, str):
            parts.append(triple)
    return " ".join(parts)


def process_split(dataset_split) -> pd.DataFrame:
    """Convert one HF split into the flat claim–evidence schema."""
    records = []
    for example in dataset_split:
        claim = clean_text(example["claim"])
        evidence_text = clean_text(flatten_evidence(example["evidence"]))
        records.append({
            "id": str(example["id"]),
            "claim": claim,
            "evidence_text": evidence_text,
            "input_text": claim + config.SEP_TOKEN + evidence_text,
            "label": example["label"],
        })
    return pd.DataFrame(records)


def main() -> None:
    logger = utils.setup_logging(1)
    utils.set_seed()

    with utils.timed(logger, "load FEVER from Hugging Face"):
        dataset = load_dataset(config.FEVER_HF_ID)
    logger.info("splits: %s", {k: len(v) for k, v in dataset.items()})

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dropped_counts = {}
    for hf_split, our_split in SPLIT_MAP.items():
        with utils.timed(logger, f"process FEVER {our_split}"):
            df = process_split(dataset[hf_split])
        assert set(df["label"]) <= set(config.LABELS), set(df["label"])
        # Empty evidence is a meaningless input for a claim–evidence
        # classifier: drop such rows and record how many, per split.
        empty_mask = df["evidence_text"].str.len() == 0
        dropped_counts[our_split] = int(empty_mask.sum())
        if dropped_counts[our_split]:
            logger.warning("%s: dropped %d rows with empty evidence_text "
                           "(label mix: %s)", our_split, dropped_counts[our_split],
                           df.loc[empty_mask, "label"].value_counts().to_dict())
            df = df[~empty_mask].reset_index(drop=True)
        df.to_parquet(config.FEVER_PARQUET[our_split], index=False)
        logger.info("saved %s: %d rows, labels=%s",
                    config.FEVER_PARQUET[our_split], len(df),
                    df["label"].value_counts().to_dict())

    # Persist the dropped-row counts so the EDA stats table can cite them.
    metadata = utils.read_json(config.RUN_METADATA_JSON)
    metadata["fever_dropped_empty_evidence"] = dropped_counts
    utils.write_json(config.RUN_METADATA_JSON, metadata)
    logger.info("empty-evidence rows dropped per split: %s", dropped_counts)


if __name__ == "__main__":
    main()
