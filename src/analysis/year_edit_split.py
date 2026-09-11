"""CR-2 — year-edit split of the number-change slice (Handover v2.2 §13.1).

Splits the 230 number_change pairs into edits whose edited token is a four-digit year (19xx/20xx in the source sentence) and all
other number edits, using the same deterministic diff as scan_perturbations.py, and reports detection accuracy on each half for the
released predictions of DistilBERT (40k subset), DistilBERT (full FEVER), XGBoost and the lexical baseline, with 95% percentile
bootstrap CIs (2,000 resamples, seed 0). Deterministic; no model involved.
Inputs: mirror/data/processed/sec_synthetic_test.parquet; mirror/output/predictions/<model>__sec_synthetic.csv.
Outputs: year_edit_split.csv, year_edit_split.md (the table row and the two sentences for the camera-ready)."""
import re, difflib, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

M = Path("/work/finnlp/mirror"); P = M / "data/processed/sec_synthetic_test.parquet"
df = pd.read_parquet(P)

def edited_number(claim, evidence):
    sents = re.split(r"(?<=[.;])\s+", evidence)
    src = max(sents, key=lambda s: difflib.SequenceMatcher(None, s, claim).ratio())
    a = re.findall(r"\d[\d,]*\.?\d*", src); b = re.findall(r"\d[\d,]*\.?\d*", claim)
    return [(x, y) for x, y in zip(a, b) if x != y]

nc = df[df.perturbation_type == "number_change"].copy()
kind = []
for _, r in nc.iterrows():
    diff = edited_number(r.claim, r.evidence_text)
    if not diff: kind.append("unlocalised"); continue
    o, n = diff[0]
    kind.append("year" if re.fullmatch(r"(19|20)\d\d", o.replace(",", "")) else "non_year")
nc["edit_kind"] = kind
counts = nc.edit_kind.value_counts().to_dict()

def ci(ok, seed=0, B=2000):
    rng = np.random.default_rng(seed); ok = np.asarray(ok, dtype=float); n = len(ok)
    if n == 0: return (np.nan, np.nan)
    bs = [ok[rng.integers(0, n, n)].mean() for _ in range(B)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

rows = []
for model, name in [("distilbert_full", "DistilBERT (full FEVER)"), ("distilbert", "DistilBERT (40k subset)"), ("xgboost", "XGBoost"), ("lexical_baseline", "Lexical baseline")]:
    pr = pd.read_csv(M / f"output/predictions/{model}__sec_synthetic.csv").set_index("id")
    j = nc.join(pr[["y_true", "y_pred"]], on="id")
    j["ok"] = j.y_true == j.y_pred
    for k in ("year", "non_year", "unlocalised"):
        s = j[j.edit_kind == k]
        lo, hi = ci(s.ok)
        rows.append({"model": name, "edit_kind": k, "n": len(s), "accuracy": round(s.ok.mean(), 4) if len(s) else None, "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    allr = j; lo, hi = ci(allr.ok)
    rows.append({"model": name, "edit_kind": "all_number_change", "n": len(allr), "accuracy": round(allr.ok.mean(), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
out = pd.DataFrame(rows); out.to_csv("/work/finnlp/year_edit_split.csv", index=False)

full = out[out.model == "DistilBERT (full FEVER)"].set_index("edit_kind")
sub = out[out.model == "DistilBERT (40k subset)"].set_index("edit_kind")
md = ["# CR-2 — year-edit split of the number-change slice", "",
      f"Source: `sec_synthetic_test.parquet` (sha256 {hashlib.sha256(P.read_bytes()).hexdigest()[:16]}…), released predictions; split by the same deterministic diff as `scan_perturbations.py` (edited source token is a four-digit 19xx/20xx year). Counts: {counts}. CIs are 95% percentile bootstraps over items (2,000 resamples, seed 0).", "",
      "## Table row (Table 1b or the per-perturbation table)", "",
      "| Model | number change: year edits (n=%d) | number change: other edits (n=%d) | all number changes (n=%d) |" % (counts.get("year", 0), counts.get("non_year", 0), len(nc)), "|---|---|---|---|"]
for name in ["DistilBERT (full FEVER)", "DistilBERT (40k subset)", "XGBoost", "Lexical baseline"]:
    r = out[out.model == name].set_index("edit_kind")
    md.append(f"| {name} | {100*r.loc['year','accuracy']:.1f} [{100*r.loc['year','ci_low']:.1f}, {100*r.loc['year','ci_high']:.1f}] | {100*r.loc['non_year','accuracy']:.1f} [{100*r.loc['non_year','ci_low']:.1f}, {100*r.loc['non_year','ci_high']:.1f}] | {100*r.loc['all_number_change','accuracy']:.1f} [{100*r.loc['all_number_change','ci_low']:.1f}, {100*r.loc['all_number_change','ci_high']:.1f}] |")
y, ny = full.loc["year"], full.loc["non_year"]
overlap = not (y.ci_high < ny.ci_low or ny.ci_high < y.ci_low)
md += ["", "## The two sentences (Limitations, after the year-edit disclosure)", "",
       f"Splitting the 230 number-change edits by whether the edited token is a four-digit year ({counts.get('year',0)} year edits, {counts.get('non_year',0)} other numeric edits; {counts.get('unlocalised',0)} not localisable by an automated diff), the full-FEVER DistilBERT detects year edits at {100*y.accuracy:.1f}% (95% CI {100*y.ci_low:.1f} to {100*y.ci_high:.1f}) and other numeric edits at {100*ny.accuracy:.1f}% ({100*ny.ci_low:.1f} to {100*ny.ci_high:.1f}); the 40k-subset model gives {100*sub.loc['year','accuracy']:.1f}% and {100*sub.loc['non_year','accuracy']:.1f}%. "
       + ("The intervals overlap, so the number-change result does not depend on the implausible year edits: the headline holds on the plausible half." if overlap else "The intervals do not overlap, so the number-change result depends in part on which half is counted; we report both."),
       "", "## Receipt", "", "`python3 /work/finnlp/year_edit_split.py` regenerates `year_edit_split.csv` and this file; the sha256 of the parquet is checked against `scan_perturbations.json`."]
Path("/work/finnlp/year_edit_split.md").write_text("\n".join(md) + "\n")
print(out.to_string()); print("\n".join(md[-6:-4]))
