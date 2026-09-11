# A9 — Fin-Fact within-256-token subset (evaluation only)

Run 2026-09-05T11:27:44.464176+00:00. finfact.json sha256 `5e36d32a0e0149f5d86b1f81ab6fd85144136cb7975374364060f8fcf43629fd` (29972446 bytes).
Rebuilt Fin-Fact eval set: 3367 rows, labels {'REFUTES': 1492, 'SUPPORTS': 1273, 'NOT ENOUGH INFO': 602} (paper dataset_stats.csv: 3367 rows; 1271 / 1485 / 611).
Untruncated DistilBERT pair encodings > 256 tokens: **62.01%** (paper truncation_rates.json: 62.01%). Median untruncated length 324 tokens.
Subset sizes: fits (≤256) n = 1279; exceeds (>256) n = 2088.

Macro-F1 by subset (recomputed from the released per-example predictions; the `full` column must equal results.json):

| model | n full | full | n fits | fits ≤256 | n exceeds | exceeds >256 |
|---|---|---|---|---|---|---|
| lexical_baseline | 3367 | 0.367 | 1279 | 0.283 | 2088 | 0.404 |
| claim_only_logreg | 3367 | 0.258 | 1279 | 0.265 | 2088 | 0.256 |
| naive_bayes | 3367 | 0.108 | 1279 | 0.105 | 2088 | 0.109 |
| logreg | 3367 | 0.205 | 1279 | 0.178 | 2088 | 0.224 |
| linear_svm | 3367 | 0.201 | 1279 | 0.174 | 2088 | 0.220 |
| random_forest | 3367 | 0.244 | 1279 | 0.255 | 2088 | 0.233 |
| xgboost | 3367 | 0.364 | 1279 | 0.376 | 2088 | 0.333 |
| voting_ensemble | 3367 | 0.181 | 1279 | 0.152 | 2088 | 0.201 |
| voting_ensemble_calsvm | 3367 | 0.185 | 1279 | 0.157 | 2088 | 0.205 |
| bilstm | 3367 | 0.151 | 1279 | 0.144 | 2088 | 0.156 |
| distilbert | 3367 | 0.337 | 1279 | 0.290 | 2088 | 0.353 |
| distilbert_full | 3367 | 0.313 | 1279 | 0.274 | 2088 | 0.321 |

Cross-check against the paper's results.json (full set):

| model                  |   paper_results_json |   recomputed_full | match   |
|:-----------------------|---------------------:|------------------:|:--------|
| bilstm                 |               0.151  |            0.151  | True    |
| claim_only_logreg      |               0.2584 |            0.2584 | True    |
| distilbert             |               0.3366 |            0.3366 | True    |
| distilbert_full        |               0.3127 |            0.3127 | True    |
| lexical_baseline       |               0.3675 |            0.3675 | True    |
| linear_svm             |               0.2009 |            0.2009 | True    |
| logreg                 |               0.2049 |            0.2049 | True    |
| naive_bayes            |               0.1076 |            0.1076 | True    |
| random_forest          |               0.2439 |            0.2439 | True    |
| voting_ensemble        |               0.1806 |            0.1806 | True    |
| voting_ensemble_calsvm |               0.185  |            0.185  | True    |
| xgboost                |               0.364  |            0.364  | True    |

Label composition by subset (share of each class):

| fits_256   |   NOT ENOUGH INFO |   REFUTES |   SUPPORTS |
|:-----------|------------------:|----------:|-----------:|
| False      |             0.196 |     0.401 |      0.403 |
| True       |             0.151 |     0.511 |      0.338 |

Evidence-source composition by subset:

| fits_256   |   evidence |   justification |
|:-----------|-----------:|----------------:|
| False      |       1988 |             100 |
| True       |       1277 |               2 |