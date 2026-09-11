"""Central configuration for the fact-verification project.

Every constant used anywhere in src/ lives here: seeds, paths, dataset
identifiers, feature-extraction parameters, hyperparameter grids, and
deep-model settings. No magic numbers inside src/.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths (repo-relative, resolved from this file's location)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_STORE_DIR = ROOT_DIR / "models_store"  # fitted models (git-ignored)

RESULTS_JSON = RESULTS_DIR / "results.json"
RUN_METADATA_JSON = RESULTS_DIR / "run_metadata.json"

# ---------------------------------------------------------------------------
# Labels (canonical order everywhere: indices, confusion matrices, reports)
# ---------------------------------------------------------------------------
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
FEVER_HF_ID = "copenlu/fever_gold_evidence"
FINFACT_HF_ID = "amanrangapur/Fin-Fact"
# Raw Fin-Fact JSON from the archived GitHub repo (fallback if the HF load fails).
FINFACT_FALLBACK_URL = (
    "https://raw.githubusercontent.com/IIT-DM/Fin-Fact/FinFact/finfact.json"
)

FEVER_PARQUET = {
    "train": PROCESSED_DIR / "fever_train.parquet",
    "val": PROCESSED_DIR / "fever_val.parquet",
    "test": PROCESSED_DIR / "fever_test.parquet",
}
FINFACT_PARQUET = PROCESSED_DIR / "finfact_test.parquet"
SEC_PARQUET = PROCESSED_DIR / "sec_synthetic_test.parquet"

# Evaluation sets used by the harness: eval_key -> parquet path.
EVAL_SETS = {
    "fever_test": FEVER_PARQUET["test"],
    "finfact_test": FINFACT_PARQUET,
    "sec_synthetic": SEC_PARQUET,
}

# Separator token joining claim and evidence for bag-of-words style models.
SEP_TOKEN = " [SEP] "

# ---------------------------------------------------------------------------
# SEC EDGAR synthetic set
# ---------------------------------------------------------------------------
SEC_TICKERS = [
    "AAPL", "MSFT", "NVDA", "JPM", "GS", "XOM",
    "PFE", "WMT", "KO", "BA", "CAT", "T",
]
# Company names used for entity-swap perturbations (ticker -> registrant name).
SEC_COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "GS": "The Goldman Sachs Group, Inc.",
    "XOM": "Exxon Mobil Corporation",
    "PFE": "Pfizer Inc.",
    "WMT": "Walmart Inc.",
    "KO": "The Coca-Cola Company",
    "BA": "The Boeing Company",
    "CAT": "Caterpillar Inc.",
    "T": "AT&T Inc.",
}
# EDGAR query overrides. XOM's ticker now resolves to a new holding-company
# CIK (2115436, "ExxonMobil Holdings Corp") that has no 10-K filings; the
# historical 10-Ks sit under the operating company's CIK 0000034088.
SEC_CIK_OVERRIDES = {"XOM": "0000034088"}

# Name variants a filing actually uses in prose (searched for entity_swap).
SEC_NAME_VARIANTS = {
    "AAPL": ["Apple Inc.", "Apple"],
    "MSFT": ["Microsoft Corporation", "Microsoft"],
    "NVDA": ["NVIDIA Corporation", "NVIDIA"],
    "JPM": ["JPMorgan Chase & Co.", "JPMorgan Chase", "JPMorgan"],
    "GS": ["The Goldman Sachs Group, Inc.", "Goldman Sachs"],
    "XOM": ["Exxon Mobil Corporation", "ExxonMobil", "Exxon Mobil"],
    "PFE": ["Pfizer Inc.", "Pfizer"],
    "WMT": ["Walmart Inc.", "Walmart"],
    "KO": ["The Coca-Cola Company", "Coca-Cola"],
    "BA": ["The Boeing Company", "Boeing"],
    "CAT": ["Caterpillar Inc.", "Caterpillar"],
    "T": ["AT&T Inc.", "AT&T"],
}
SEC_EDGAR_COMPANY = "Dirmacs"
SEC_EDGAR_EMAIL = "suprabhat@dirmacs.com"
SEC_DOWNLOAD_DIR = RAW_DIR / "sec_edgar"
SEC_MIN_SENT_TOKENS = 8
SEC_MAX_SENT_TOKENS = 60
SEC_TARGET_PER_CLASS = 500
SEC_NEI_MAX_OVERLAP = 0.2
SEC_NUMBER_SCALE_RANGE = (0.15, 0.60)  # |relative change| drawn from this range

# ---------------------------------------------------------------------------
# TF-IDF (shared factory: src/features/tfidf.py)
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "ngram_range": (1, 2),
    "max_features": 100_000,
    "min_df": 2,
    "sublinear_tf": True,
    "lowercase": True,
    "dtype": "float32",
    # The "[SEP]" joiner in input_text must not become a feature: it occurs
    # exactly once in every document, so after L2 normalisation it acts as an
    # inverse-document-length signal and creates artificial boundary bigrams
    # ("born sep"). Listing it as a stop word removes both artifacts.
    "stop_words": ["sep"],
}

# ---------------------------------------------------------------------------
# Phase 2 — classical models
# ---------------------------------------------------------------------------
GRID_SUBSAMPLE_SIZE = 60_000  # stratified subsample of FEVER train for GridSearchCV
GRID_CV_FOLDS = 3
GRID_SCORING = "f1_macro"
NB_GRID = {"alpha": [0.1, 0.5, 1.0]}
LOGREG_GRID = {"C": [0.25, 1, 4]}
# Checkpoint-2 addendum: C=4 sat at the grid edge, so the bracket was extended.
LOGREG_EXTENDED_GRID = {"C": [4, 8, 16]}
LOGREG_EXTENSION_MIN_GAIN = 0.005  # CV macro-F1 gain required to switch off C=4
LOGREG_FIXED = {"solver": "saga", "class_weight": "balanced", "max_iter": 2000}
SVM_GRID = {"C": [0.1, 0.5, 1]}
SVM_FIXED = {"class_weight": "balanced"}
LEXICAL_THRESHOLD_GRID_START = 0.05
LEXICAL_THRESHOLD_GRID_STOP = 0.95
LEXICAL_THRESHOLD_GRID_STEP = 0.05

# ---------------------------------------------------------------------------
# Phase 3 — ensembles
# ---------------------------------------------------------------------------
SVD_COMPONENTS = 300
RF_N_ESTIMATORS_SWEEP = [50, 100, 200, 400]
RF_FINAL_N_ESTIMATORS = 400
XGB_PARAMS = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.1,
    "n_estimators": 400,
}
XGB_EARLY_STOPPING_ROUNDS = 50
# Memory controls for the 15.3 GB machine. With max_depth=8 the depth-wise
# histogram cache alone needs ~2^(depth-1) nodes x n_features x max_bin x 16 B
# (~13 GB at 128 bins over 100k features), which caused bad_malloc crashes.
# max_cached_hist_node bounds the cached node histograms (rebuild cost, no
# change to the learned trees); QuantileDMatrix + fewer bins/threads shrink
# the rest of the working set.
XGB_MAX_BIN = 128
XGB_NTHREAD = 6
XGB_MAX_CACHED_HIST_NODE = 8
# Soft-voting members: every member must expose predict_proba, which is why
# LinearSVC (margin-based, no calibrated probabilities) is excluded.
VOTING_COMPONENTS = ["naive_bayes", "logreg", "xgboost"]

# ---------------------------------------------------------------------------
# Phase 4 — deep models
# ---------------------------------------------------------------------------
BILSTM_VOCAB_SIZE = 50_000
BILSTM_MAX_LEN = 128
BILSTM_EMBED_DIM = 100
GLOVE_GENSIM_NAME = "glove-wiki-gigaword-100"
GLOVE_FALLBACK_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
BILSTM_UNITS = 128
BILSTM_DROPOUT = 0.3
BILSTM_DENSE_UNITS = 64
BILSTM_LR = 1e-3
BILSTM_BATCH_SIZE = 256
BILSTM_MAX_EPOCHS = 10
BILSTM_PATIENCE = 2
BILSTM_FALLBACK_TRAIN_SIZE = 100_000  # stratified subset if full train exceeds RAM
# Per-epoch train macro-F1 for fig_11 is computed on a fixed stratified sample
# (a full 228k-row predict per epoch would double training time).
BILSTM_TRAIN_F1_SAMPLE = 20_000

DISTILBERT_MODEL_ID = "distilbert/distilbert-base-uncased"
DISTILBERT_MAX_LEN = 256
DISTILBERT_EPOCHS = 2
DISTILBERT_LR = 2e-5
DISTILBERT_GPU_BATCH = 32
DISTILBERT_CPU_BATCH = 16
DISTILBERT_CPU_TRAIN_SIZE = 40_000  # stratified subset on the CPU-only path
DISTILBERT_EVAL_BATCH = 32
# ETA rule (report session): project total wall time after this many steps;
# above the cap, abort the CPU run and rely on the Colab handoff instead.
DISTILBERT_ETA_CHECK_STEP = 200
DISTILBERT_MAX_PROJECTED_HOURS = 12

# ---------------------------------------------------------------------------
# Phase 5 — ablations, efficiency, error analysis
# ---------------------------------------------------------------------------
ABLATION_TRAIN_SIZES = [1_000, 5_000, 20_000, 60_000, None]  # None = full train
ABLATION_VOCAB_SIZES = [5_000, 20_000, 50_000, 100_000]
EFFICIENCY_SLICE_SIZE = 2_000  # FEVER-test slice for inference-speed timing
NEGATION_CUES = ["not", "no", "never", "n't", "without", "fails"]
LOW_OVERLAP_THRESHOLD = 0.2
