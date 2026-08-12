"""Download and map the Fin-Fact financial fact-checking dataset (brief §5.2).

Loads `amanrangapur/Fin-Fact` (GitHub raw JSON as fallback), records the full
raw label distribution, maps labels onto the canonical FEVER scheme, builds
evidence text from the best available source per row, and saves
data/processed/finfact_test.parquet. The mapping decisions are written to
output/tables/label_mapping_finfact.csv.

Fin-Fact is used strictly as an out-of-domain evaluation set — no model is
ever trained or tuned on it.
"""

import csv
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils

# Raw Fin-Fact label -> canonical label (None = drop, recorded in the CSV).
# The dataset's three-way scheme maps naturally onto FEVER's:
#   true  -> SUPPORTS, false -> REFUTES, NEI-like -> NOT ENOUGH INFO.
# Any unexpected value found at runtime is dropped conservatively and logged.
LABEL_MAP = {
    "true": "SUPPORTS",
    "false": "REFUTES",
    "nei": "NOT ENOUGH INFO",
    "neutral": "NOT ENOUGH INFO",
    "not enough info": "NOT ENOUGH INFO",
    "not enough information": "NOT ENOUGH INFO",
}


def load_finfact_records(logger) -> list:
    """Load Fin-Fact rows as a list of dicts, HF first, GitHub fallback."""
    try:
        from datasets import load_dataset
        dataset = load_dataset(config.FINFACT_HF_ID)
        split = list(dataset.keys())[0]  # single split expected
        logger.info("loaded %s from HF, split=%s, rows=%d",
                    config.FINFACT_HF_ID, split, len(dataset[split]))
        return list(dataset[split])
    except Exception as exc:  # noqa: BLE001 — any load failure triggers fallback
        logger.warning("HF load failed (%s); falling back to GitHub raw JSON", exc)
        raw_path = config.RAW_DIR / "finfact.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            urllib.request.urlretrieve(config.FINFACT_FALLBACK_URL, raw_path)
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        logger.info("loaded %d rows from GitHub fallback", len(records))
        return records


def extract_evidence(row: dict) -> tuple[str, str]:
    """Return (evidence_text, source_name) using the best available source.

    Priority: sentence text inside the `evidence` field, then `justification`,
    then `sci_digest`.
    """
    parts = []
    for item in row.get("evidence") or []:
        if isinstance(item, dict):
            sentence = item.get("sentence") or item.get("text") or ""
            if sentence and str(sentence).strip():
                parts.append(str(sentence).strip())
        elif isinstance(item, str) and item.strip():
            parts.append(item.strip())
    if parts:
        return " ".join(parts), "evidence"

    for field in ("justification", "sci_digest"):
        value = row.get(field)
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        if value and str(value).strip():
            return str(value).strip(), field
    return "", "none"


def main() -> None:
    logger = utils.setup_logging(1)
    utils.set_seed()

    records = load_finfact_records(logger)

    # Record the actual full raw label distribution before mapping.
    raw_labels = Counter(str(row.get("label")).strip().lower() for row in records)
    logger.info("raw Fin-Fact label distribution: %s", dict(raw_labels))

    rows, evidence_sources = [], Counter()
    for i, row in enumerate(records):
        raw_label = str(row.get("label")).strip().lower()
        mapped = LABEL_MAP.get(raw_label)
        if mapped is None:
            continue  # dropped; recorded via the mapping CSV below
        claim = re.sub(r"\s+", " ", str(row.get("claim") or "")).strip()
        evidence_text, source = extract_evidence(row)
        evidence_text = re.sub(r"\s+", " ", evidence_text).strip()
        evidence_sources[source] += 1
        if not claim or not evidence_text:
            evidence_sources["skipped_empty"] += 1
            continue
        rows.append({
            "id": f"finfact_{i:04d}",
            "claim": claim,
            "evidence_text": evidence_text,
            "input_text": claim + config.SEP_TOKEN + evidence_text,
            "label": mapped,
            "evidence_source": source,
        })

    # Label-mapping audit table (every raw value, kept or not).
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    mapping_path = config.TABLES_DIR / "label_mapping_finfact.csv"
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_label", "count", "mapped_to", "kept"])
        for raw_label, count in sorted(raw_labels.items()):
            mapped = LABEL_MAP.get(raw_label)
            writer.writerow([raw_label, count, mapped or "dropped", mapped is not None])
    logger.info("wrote %s", mapping_path)
    logger.info("evidence sources used: %s", dict(evidence_sources))

    df = pd.DataFrame(rows)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.FINFACT_PARQUET, index=False)
    logger.info("saved %s: %d rows, labels=%s", config.FINFACT_PARQUET,
                len(df), df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
