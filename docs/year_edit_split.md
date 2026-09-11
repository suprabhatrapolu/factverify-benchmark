# CR-2 — year-edit split of the number-change slice

Source: `sec_synthetic_test.parquet` (sha256 696a69fe01ba9a12…), released predictions; split by the same deterministic diff as `scan_perturbations.py` (edited source token is a four-digit 19xx/20xx year). Counts: {'non_year': 126, 'year': 95, 'unlocalised': 9}. CIs are 95% percentile bootstraps over items (2,000 resamples, seed 0).

## Table row (Table 1b or the per-perturbation table)

| Model | number change: year edits (n=95) | number change: other edits (n=126) | all number changes (n=230) |
|---|---|---|---|
| DistilBERT (full FEVER) | 39.0 [28.4, 49.5] | 28.6 [21.4, 36.5] | 32.6 [26.5, 38.3] |
| DistilBERT (40k subset) | 31.6 [22.1, 41.0] | 23.8 [16.7, 30.9] | 26.5 [20.9, 32.2] |
| XGBoost | 11.6 [6.3, 17.9] | 9.5 [4.8, 15.1] | 10.9 [7.0, 15.2] |
| Lexical baseline | 0.0 [0.0, 0.0] | 0.8 [0.0, 2.4] | 0.4 [0.0, 1.3] |

## The two sentences (Limitations, after the year-edit disclosure)

Splitting the 230 number-change edits by whether the edited token is a four-digit year (95 year edits, 126 other numeric edits; 9 not localisable by an automated diff), the full-FEVER DistilBERT detects year edits at 39.0% (95% CI 28.4 to 49.5) and other numeric edits at 28.6% (21.4 to 36.5); the 40k-subset model gives 31.6% and 23.8%. The intervals overlap, so the number-change result does not depend on the implausible year edits: the headline holds on the plausible half.

## Receipt

`python3 /work/finnlp/year_edit_split.py` regenerates `year_edit_split.csv` and this file; the sha256 of the parquet is checked against `scan_perturbations.json`.
