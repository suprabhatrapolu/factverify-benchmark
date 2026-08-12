# Controlled Financial Claim Verification Benchmark

Code and data for an anonymous double-blind paper submission.

## Contents
- `data/processed/sec_synthetic_test.parquet` — 1,489 controlled
  claim–evidence pairs from SEC 10-K filings, each tagged with its
  perturbation rule (verbatim / number_change / negation /
  entity_swap / evidence_mismatch).
- `src/data/build_sec_synthetic.py` — benchmark generation
  (deterministic, seed 42).
- `src/eval/harness.py` — unified evaluation harness.
- `src/models/` — all model implementations.
- `output/results/results.json` — full recorded metrics.
- `output/predictions/` — per-model predictions on every eval set.

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
python src/models/classical.py
```

Seed is fixed at 42 throughout (`config.py`).

## License
Code: MIT. SEC filing text is US public domain.
