# Controlled Financial Claim Verification Benchmark

Code and data for "When FEVER Meets Finance: A Controlled Benchmark Reveals Overlap-Driven Financial Claim Verification", Suprabhat Rapolu, FinNLP 2026 (11th Workshop on Financial Technology and Natural Language Processing) @ EMNLP 2026.

## Contents
- `data/processed/sec_synthetic_test.parquet` — 1,489 controlled
  claim–evidence pairs from SEC 10-K filings, each tagged with its
  perturbation rule (verbatim / number_change / negation /
  entity_swap / evidence_mismatch). Class balance 497/497/495.
- `src/data/build_sec_synthetic.py` — benchmark generation
  (deterministic, seed 42).
- `src/eval/harness.py` — unified evaluation harness; every model
  is scored through this single function.
- `src/models/` — all model implementations across three tiers.
- `output/results/results.json` — recorded metrics for all 12
  models on all three evaluation sets.
- `output/predictions/` — per-model predictions on every eval set.
- `output/tables/` — dataset statistics, hyperparameter search
  record, ablations, per-perturbation accuracy, error taxonomy.
- `output/results/truncation_rates.json` — share of pairs
  exceeding the 256-token transformer input, per dataset.

## Not included
FEVER and Fin-Fact are public datasets and are not redistributed
here; `src/data/load_fever.py` and `load_finfact.py` download them.
Model weights are omitted for size.

## Reproducing
```
pip install -r requirements.txt
python src/data/load_fever.py
python src/data/load_finfact.py
python src/data/build_sec_synthetic.py
python src/models/lexical_baseline.py
python src/models/classical.py
python src/models/ensembles.py
python src/models/bilstm.py
python src/models/transformer.py
python src/models/extras.py
python src/eval/ablations.py
python src/eval/transfer_efficiency.py
python src/eval/error_analysis.py
```

The full-train transformer was trained separately on a cloud GPU
via `train_distilbert_colab.py`. Seed is 42 throughout (`config.py`).

## Camera-ready analyses

Three analyses were added for the camera-ready version in response to reviewer questions. Each is reproducible from the released predictions with no retraining:

- `src/analysis/subset_analysis.py` — Fin-Fact macro-F1 split by whether the untruncated DistilBERT pair encoding fits the 256-token input (paper Table 5). Outputs `output/tables/subset_analysis.csv` and `output/tables/subset_bootstrap_ci.csv`.
- `src/analysis/year_edit_split.py` — number-change edits split by whether the edited token is a four-digit year (paper Limitations). Outputs `output/tables/year_edit_split.csv`.
- `src/analysis/scan_perturbations.py` — automated scan of the released pairs for implausible edits and double negatives (paper Limitations). Outputs `output/results/scan_perturbations.json`.

`src/analysis/verify_v2.py` re-derives every number reported in the camera-ready from the released outputs; receipts are in `output/results/verification-v2.json`.

## License
Code: MIT. SEC filing text is US public domain.
