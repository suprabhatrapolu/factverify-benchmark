"""Automated plausibility scan of the released SEC-synthetic pairs (for the Limitations edit and the response).

Counts, from data/processed/sec_synthetic_test.parquet:
  - number_change edits whose edited token is a four-digit year, and how many produce a value outside 1990–2035;
  - negation edits whose claim contains two negation cues (possible double negatives);
  - pairs where an automated diff cannot localise the edit (manual check recommended).
Deterministic; no model involved.
"""
import re, difflib, json, hashlib, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

P = config.SEC_PARQUET
df = pd.read_parquet(P)

def edited_number(claim, evidence):
    sents = re.split(r"(?<=[.;])\s+", evidence)
    src = max(sents, key=lambda s: difflib.SequenceMatcher(None, s, claim).ratio())
    a = re.findall(r"\d[\d,]*\.?\d*", src); b = re.findall(r"\d[\d,]*\.?\d*", claim)
    return src, [(x, y) for x, y in zip(a, b) if x != y]

nc = df[df.perturbation_type == "number_change"]
yearish, implausible, nodiff = 0, [], 0
for _, r in nc.iterrows():
    _, diff = edited_number(r.claim, r.evidence_text)
    if not diff:
        nodiff += 1; continue
    o, n = diff[0]
    if re.fullmatch(r"(19|20)\d\d", o.replace(",", "")):
        yearish += 1
        ny = n.replace(",", "")
        if not re.fullmatch(r"\d{4}", ny) or not (1990 <= int(ny) <= 2035):
            implausible.append((r.id, o, n))
neg = df[df.perturbation_type == "negation"]
dbl = neg[neg.claim.str.contains(r"\b(?:no|not|never)\b.*\b(?:not|no|never)\b", regex=True)]
out = {"parquet_sha256": hashlib.sha256(P.read_bytes()).hexdigest(),
       "number_change_pairs": int(len(nc)), "edited_token_is_year": yearish,
       "edited_year_outside_1990_2035": len(implausible), "edit_not_localised_by_diff": nodiff,
       "negation_pairs": int(len(neg)), "negation_double_cue_claims": int(len(dbl)),
       "examples_implausible_year": implausible[:10], "examples_double_negation": dbl.claim.head(7).tolist()}
(config.RESULTS_DIR / "scan_perturbations.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
