# Controlled Financial Claim Verification Benchmark

Code and data for an anonymous double-blind paper submission.

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

## License
Code: MIT. SEC filing text is US public domain.
