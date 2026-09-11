"""A9 — within-256-token Fin-Fact subset analysis (evaluation only).

Reproduces the Fin-Fact evaluation set exactly as src/data/load_finfact.py builds it
(same id scheme, same label map, same evidence extraction), measures each pair's
untruncated DistilBERT pair-encoding length exactly as src/models/transformer.py
log_truncation_rates does, and re-scores every model's saved predictions
(output/predictions/<model>__finfact_test.csv) on the subset whose encoding fits the
256-token budget versus the subset that does not. No model is run; nothing is trained.

Inputs : finfact.json (GitHub raw, hashed), the mirrored predictions, the tokenizer.
Outputs: subset_analysis.md, subset_analysis.csv, finfact_lengths.csv
"""
import hashlib, json, re, sys, datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

RAW = config.RAW_DIR / "finfact.json"
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
MAX_LEN = 256
SEP = " [SEP] "
LABEL_MAP = {"true": "SUPPORTS", "false": "REFUTES", "nei": "NOT ENOUGH INFO",
             "neutral": "NOT ENOUGH INFO", "not enough info": "NOT ENOUGH INFO",
             "not enough information": "NOT ENOUGH INFO"}

def extract_evidence(row):
    parts = []
    for item in row.get("evidence") or []:
        if isinstance(item, dict):
            s = item.get("sentence") or item.get("text") or ""
            if s and str(s).strip(): parts.append(str(s).strip())
        elif isinstance(item, str) and item.strip(): parts.append(item.strip())
    if parts: return " ".join(parts), "evidence"
    for field in ("justification", "sci_digest"):
        v = row.get(field)
        if isinstance(v, list): v = " ".join(str(x) for x in v if x)
        if v and str(v).strip(): return str(v).strip(), field
    return "", "none"

def rebuild_finfact():
    records = json.loads(RAW.read_text(encoding="utf-8"))
    rows = []
    for i, row in enumerate(records):
        mapped = LABEL_MAP.get(str(row.get("label")).strip().lower())
        if mapped is None: continue
        claim = re.sub(r"\s+", " ", str(row.get("claim") or "")).strip()
        ev, src = extract_evidence(row)
        ev = re.sub(r"\s+", " ", ev).strip()
        if not claim or not ev: continue
        rows.append({"id": f"finfact_{i:04d}", "claim": claim, "evidence_text": ev, "label": mapped, "evidence_source": src})
    return pd.DataFrame(rows)

def main():
    raw_bytes = RAW.read_bytes()
    receipts = {"finfact_json_sha256": hashlib.sha256(raw_bytes).hexdigest(), "finfact_json_bytes": len(raw_bytes),
                "finfact_json_url": "https://raw.githubusercontent.com/IIT-DM/Fin-Fact/FinFact/finfact.json",
                "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    df = rebuild_finfact()
    receipts["rows"] = int(len(df)); receipts["label_counts"] = df["label"].value_counts().to_dict()
    tok = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    lengths = [len(ids) for ids in tok(df["claim"].tolist(), df["evidence_text"].tolist())["input_ids"]]
    df["pair_tokens"] = lengths
    df["fits_256"] = df["pair_tokens"] <= MAX_LEN
    receipts["truncation_rate_pct"] = round(100 * float(np.mean(df["pair_tokens"] > MAX_LEN)), 2)
    receipts["median_untruncated_tokens"] = int(np.median(lengths))
    receipts["n_fits"] = int(df["fits_256"].sum()); receipts["n_exceeds"] = int((~df["fits_256"]).sum())
    df[["id", "label", "evidence_source", "pair_tokens", "fits_256"]].to_csv(config.TABLES_DIR / "finfact_lengths.csv", index=False)

    rows = []
    for pred_path in sorted(config.PREDICTIONS_DIR.glob("*__finfact_test.csv")):
        model = pred_path.name.split("__")[0]
        p = pd.read_csv(pred_path)
        m = df.merge(p, on="id", how="inner", suffixes=("", "_pred"))
        assert len(m) == len(df) == len(p), (model, len(m), len(df), len(p))
        # The live GitHub finfact.json carries label edits relative to the copy the paper scored
        # against (the paper loaded HF `amanrangapur/Fin-Fact` first). Scoring uses the paper's
        # y_true from the released predictions; the rebuilt copy is used ONLY for token lengths.
        label_drift = int((m["label"] != m["y_true"]).sum()); receipts.setdefault("label_drift_rows_vs_paper", {})[model] = label_drift
        # claim text identity check (whitespace-normalised)
        same_claim = (m["claim"].str.strip() == m["claim_pred"].astype(str).str.strip()).mean()
        for name, mask in [("full", np.ones(len(m), bool)), ("fits_256", m["fits_256"].values), ("exceeds_256", ~m["fits_256"].values)]:
            sub = m[mask]
            rows.append({"model": model, "subset": name, "n": int(len(sub)), "label_drift_rows": label_drift,
                         "macro_f1": round(float(f1_score(sub["y_true"], sub["y_pred"], labels=LABELS, average="macro", zero_division=0)), 4),
                         "accuracy": round(float(accuracy_score(sub["y_true"], sub["y_pred"])), 4),
                         "claim_text_match_rate": round(float(same_claim), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(config.TABLES_DIR / "subset_analysis.csv", index=False)
    # cross-check the full-set macro-F1 against the paper's results.json
    results = json.loads(config.RESULTS_JSON.read_text())
    checks = []
    for model in res["model"].unique():
        paper = results.get(model, {}).get("eval", {}).get("finfact_test", {}).get("macro_f1")
        ours = float(res[(res.model == model) & (res.subset == "full")]["macro_f1"].iloc[0])
        checks.append({"model": model, "paper_results_json": paper, "recomputed_full": ours, "match": (paper is not None and abs(paper - ours) < 1e-3)})
    checks = pd.DataFrame(checks)
    (config.RESULTS_DIR / "subset_receipts.json").write_text(json.dumps(receipts, indent=1))
    # Markdown
    piv = res.pivot(index="model", columns="subset", values="macro_f1")[["full", "fits_256", "exceeds_256"]]
    ns = res.pivot(index="model", columns="subset", values="n")[["full", "fits_256", "exceeds_256"]]
    order = ["lexical_baseline", "claim_only_logreg", "naive_bayes", "logreg", "linear_svm", "random_forest", "xgboost", "voting_ensemble", "voting_ensemble_calsvm", "bilstm", "distilbert", "distilbert_full"]
    piv = piv.reindex([o for o in order if o in piv.index])
    md = ["# A9 — Fin-Fact within-256-token subset (evaluation only)", "",
          f"Run {receipts['run_utc']}. finfact.json sha256 `{receipts['finfact_json_sha256']}` ({receipts['finfact_json_bytes']} bytes).",
          f"Rebuilt Fin-Fact eval set: {receipts['rows']} rows, labels {receipts['label_counts']} (paper dataset_stats.csv: 3367 rows; 1271 / 1485 / 611).",
          f"Untruncated DistilBERT pair encodings > 256 tokens: **{receipts['truncation_rate_pct']}%** (paper truncation_rates.json: 62.01%). Median untruncated length {receipts['median_untruncated_tokens']} tokens.",
          f"Subset sizes: fits (≤256) n = {receipts['n_fits']}; exceeds (>256) n = {receipts['n_exceeds']}.", "",
          "Macro-F1 by subset (recomputed from the released per-example predictions; the `full` column must equal results.json):", "",
          "| model | n full | full | n fits | fits ≤256 | n exceeds | exceeds >256 |", "|---|---|---|---|---|---|---|"]
    for mdl, r in piv.iterrows():
        md.append(f"| {mdl} | {int(ns.loc[mdl,'full'])} | {r['full']:.3f} | {int(ns.loc[mdl,'fits_256'])} | {r['fits_256']:.3f} | {int(ns.loc[mdl,'exceeds_256'])} | {r['exceeds_256']:.3f} |")
    md += ["", "Cross-check against the paper's results.json (full set):", "", checks.to_markdown(index=False), "",
           "Label composition by subset (share of each class):", ""]
    comp = df.groupby("fits_256")["label"].value_counts(normalize=True).unstack().round(3)
    md.append(comp.to_markdown())
    md += ["", "Evidence-source composition by subset:", "", df.groupby("fits_256")["evidence_source"].value_counts().unstack().fillna(0).astype(int).to_markdown()]
    (config.ROOT_DIR / "docs" / "subset_analysis.md").write_text("\n".join(md))
    print("\n".join(md))

if __name__ == "__main__":
    main()
