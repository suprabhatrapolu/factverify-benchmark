"""verify_v2.py — re-derives every number the camera-ready v2 sheet prints, from the released mirror.

    python3 verify_v2.py        # prints the table and writes verification-v2.json
"""
import hashlib, json, datetime
import numpy as np, pandas as pd
from scipy import stats

MIRROR = "mirror"
PQ = f"{MIRROR}/data/processed/sec_synthetic_test.parquet"

def wilson(x, n, z=1.96):
    p = x / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    m = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-m)/d, (c+m)/d)

def main():
    sha = hashlib.sha256(open(PQ, "rb").read()).hexdigest()
    df = pd.read_parquet(PQ)
    out = {"run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "parquet_sha256": sha, "n_pairs": len(df), "models": {}}

    for m in ["distilbert_full", "distilbert", "xgboost", "lexical_baseline"]:
        p = pd.read_csv(f"{MIRROR}/output/predictions/{m}__sec_synthetic.csv")
        pred = [c for c in p.columns if c.lower() in ("y_pred","pred","prediction","predicted")][0]
        j = df.merge(p, left_on="id", right_on=("id" if "id" in p.columns else p.columns[0]), suffixes=("","_p"))
        j["_ok"] = (j[pred].astype(str).str.upper().str.strip() == j.label.astype(str).str.upper().str.strip())
        cells = {}
        for pt in ["none","number_change","negation","entity_swap","evidence_mismatch"]:
            s = j[j.perturbation_type == pt]
            x, n = int(s._ok.sum()), len(s)
            lo, hi = wilson(x, n)
            cells[pt] = {"correct": x, "n": n, "acc": x/n,
                         "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)}
        out["models"][m] = cells

    # the non-year number-change slice, by the same diff rule scan_perturbations.py uses
    nc = pd.read_csv("year_edit_split.csv")
    row = nc[(nc.model == "DistilBERT (full FEVER)") & (nc.edit_kind == "non_year")].iloc[0]
    x_ny, n_ny = int(round(row.accuracy * row.n)), int(row.n)
    lo, hi = wilson(x_ny, n_ny)
    out["number_change_non_year_distilbert_full"] = {
        "correct": x_ny, "n": n_ny, "acc": x_ny/n_ny,
        "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4)}

    # the ordering tests the abstract sentence rests on
    d = out["models"]["distilbert_full"]
    def fisher(a, b):
        return float(stats.fisher_exact([[a["correct"], a["n"]-a["correct"]],
                                         [b["correct"], b["n"]-b["correct"]]])[1])
    ny = out["number_change_non_year_distilbert_full"]
    out["ordering_tests_fisher_exact_two_sided"] = {
        "entity_swap_vs_negation": round(fisher(d["entity_swap"], d["negation"]), 6),
        "entity_swap_vs_number_change": round(fisher(d["entity_swap"], d["number_change"]), 6),
        "number_change_all_vs_negation": round(fisher(d["number_change"], d["negation"]), 4),
        "number_change_non_year_vs_negation": round(fisher(ny, d["negation"]), 4)}

    # the label-mapping arithmetic the paper must not print
    lm = pd.read_csv(f"{MIRROR}/output/tables/label_mapping_finfact.csv")
    ds = pd.read_csv(f"{MIRROR}/output/tables/dataset_stats.csv")
    ff = ds[ds.dataset == "Fin-Fact"].iloc[0]
    out["label_mapping_check"] = {
        "mapping_table_sum": int(lm["count"].sum()),
        "dataset_stats_rows": int(ff.rows),
        "dataset_stats_supports": int(ff.n_supports),
        "mapping_table_true": int(lm[lm.raw_label == "true"]["count"].iloc[0]),
        "discrepancy_rows": int(lm["count"].sum() - ff.rows),
        "verdict": "print the scored counts (1271/1485/611); the mapping table is pre-drop"}

    json.dump(out, open("verification-v2.json", "w"), indent=2)

    d = out["models"]["distilbert_full"]
    print(f"parquet sha256 {sha[:16]}…  n={len(df)}")
    print("\nDistilBERT (full FEVER) — the paper's strongest model, Wilson 95%:")
    for k, v in d.items():
        print(f"  {k:20s} {v['correct']:3d}/{v['n']:3d} = {100*v['acc']:5.1f}%  "
              f"[{100*v['wilson_lo']:4.1f}, {100*v['wilson_hi']:5.1f}]")
    v = ny
    print(f"  {'number_change (non-yr)':20s} {v['correct']:3d}/{v['n']:3d} = {100*v['acc']:5.1f}%  "
          f"[{100*v['wilson_lo']:4.1f}, {100*v['wilson_hi']:5.1f}]")
    print("\nordering tests (Fisher exact, two-sided):")
    for k, p in out["ordering_tests_fisher_exact_two_sided"].items():
        verdict = "distinguishable" if p < 0.05 else "NOT distinguishable"
        print(f"  {k:36s} p = {p:<10.6g} {verdict}")
    lc = out["label_mapping_check"]
    print(f"\nlabel mapping: table sums to {lc['mapping_table_sum']} against "
          f"{lc['dataset_stats_rows']} scored rows ({lc['discrepancy_rows']:+d}) -> {lc['verdict']}")
    print("\nreceipts -> verification-v2.json")

if __name__ == "__main__":
    main()
