"""Build the synthetic SEC claim–evidence test set (brief §5.3).

The original-contribution piece: download the latest 10-K filing for 12
cross-sector tickers from SEC EDGAR, extract clean prose sentences containing
financial quantities, and generate a balanced three-class test set of
claim–evidence pairs using FactCC-style controlled perturbations
(Kryscinski et al., 2020):

  SUPPORTS         claim = a filing sentence verbatim; evidence = its context.
  REFUTES          claim = the sentence with exactly one perturbation applied
                   (number_change / entity_swap / negation, ~equal thirds).
  NOT ENOUGH INFO  claim = an unperturbed sentence; evidence = a context
                   window from a different filing with token overlap < 0.2.

All sampling is driven by one seeded random.Random instance, so the set is
fully deterministic. Evaluation only — no model is ever trained on it.
"""

import csv
import random
import re
import sys
from pathlib import Path

import nltk
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src import utils

# Directional antonym pairs for the negation perturbation (both directions).
ANTONYM_PAIRS = [
    ("increased", "decreased"), ("increase", "decrease"),
    ("increases", "decreases"), ("above", "below"),
    ("gain", "loss"), ("gains", "losses"), ("higher", "lower"),
]
# Auxiliaries after which "not" can be inserted.
AUXILIARIES = ["was", "were", "is", "are", "has", "have", "had",
               "will", "would", "does", "do", "did", "may", "can", "could"]

# Numbers with optional thousands grouping; never captures a trailing comma.
NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
# Signals that a sentence states a financial quantity worth turning into a
# claim (bare cross-references like "See Note 4" carry a digit but no fact).
QUANTITY_RE = re.compile(
    r"[%$]|\b(?:million|billion|trillion|thousand|percent|bps)\b"
    r"|\b(?:19|20)\d{2}\b|\d{1,3}(?:,\d{3})+", re.IGNORECASE)
# Running page headers glued into extracted text ("Apple Inc. | 2025 Form 10-K | 21").
PAGE_HEADER_RE = re.compile(r"[A-Za-z0-9.,&'\- ]+\|\s*\d{4} Form 10-K\s*\|\s*\d+")


# ---------------------------------------------------------------------------
# Download and text extraction
# ---------------------------------------------------------------------------

def download_filings(logger) -> dict:
    """Download the latest 10-K per ticker. Returns ticker -> filing dir."""
    from sec_edgar_downloader import Downloader

    downloader = Downloader(config.SEC_EDGAR_COMPANY, config.SEC_EDGAR_EMAIL,
                            config.SEC_DOWNLOAD_DIR)
    filing_dirs = {}
    for ticker in config.SEC_TICKERS:
        # Query by CIK where the ticker no longer resolves to the filer
        # (see config.SEC_CIK_OVERRIDES); files land under the query string.
        query = config.SEC_CIK_OVERRIDES.get(ticker, ticker)
        ticker_dir = (config.SEC_DOWNLOAD_DIR / "sec-edgar-filings"
                      / query / "10-K")
        try:
            if not ticker_dir.exists():  # skip re-download on re-runs
                downloader.get("10-K", query, limit=1, download_details=True)
            accession_dirs = sorted(d for d in ticker_dir.iterdir() if d.is_dir())
            filing_dirs[ticker] = accession_dirs[-1]
            logger.info("10-K ready for %s: %s", ticker, filing_dirs[ticker].name)
        except Exception as exc:  # noqa: BLE001 — per brief §13, skip and log
            logger.warning("EDGAR download failed for %s: %s (skipping)", ticker, exc)
    if len(filing_dirs) < 8:
        raise RuntimeError(f"only {len(filing_dirs)} filings downloaded; need >= 8")
    return filing_dirs


def filing_year(accession_dir: Path) -> int:
    """Filing year from the accession number (XXXXXXXXXX-YY-NNNNNN)."""
    return 2000 + int(accession_dir.name.split("-")[1])


def extract_filing_text(accession_dir: Path) -> str:
    """Extract prose text from a filing's primary document.

    Prefers primary-document.html; falls back to full-submission.txt. Tables,
    scripts, styles and hidden XBRL metadata are stripped before extraction —
    they hold no prose and would pollute sentence splitting.
    """
    primary = list(accession_dir.glob("primary-document.*"))
    source = primary[0] if primary else accession_dir / "full-submission.txt"
    html = source.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["table", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:(header|hidden)$")):
        tag.decompose()
    text = soup.get_text(" ")
    text = text.replace("\xa0", " ")
    # Normalise typographic punctuation and drop stray list markers so that
    # claims read cleanly (these pairs are quoted verbatim in the report).
    for src, dst in [("’", "'"), ("‘", "'"), ("“", '"'),
                     ("”", '"'), ("–", "-"), ("—", "-"),
                     ("•", " "), ("�", "'")]:
        text = text.replace(src, dst)
    text = re.sub(r"\s[#+]\s", " ", text)
    text = PAGE_HEADER_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text)


def is_candidate(sentence: str) -> bool:
    """Sentence filter (brief §5.3 plus minimal junk control).

    Keep sentences of 8–60 tokens containing a digit, '%' or '$'. To exclude
    residual cover-page/TOC fragments, additionally require that at least
    40% of tokens are purely alphabetic, the sentence is not a TOC line, and
    the digit expresses a quantity (money, percentage, magnitude word, year,
    or grouped number) rather than a bare cross-reference like "See Note 4".
    """
    tokens = sentence.split()
    if not (config.SEC_MIN_SENT_TOKENS <= len(tokens) <= config.SEC_MAX_SENT_TOKENS):
        return False
    if not re.search(r"[\d%$]", sentence):
        return False
    if not QUANTITY_RE.search(sentence):
        return False
    alpha = sum(1 for t in tokens if t.isalpha())
    if alpha / len(tokens) < 0.4:
        return False
    if re.match(r"^(Item|ITEM)\s+\d", sentence) or "Table of Contents" in sentence:
        return False
    if not sentence[0].isalnum():  # bullet/exhibit-index fragments
        return False
    if sentence.endswith(("No.", "Nos.")):  # abbreviation split artifacts
        return False
    return True


def collect_candidates(filing_dirs: dict, logger) -> dict:
    """Extract candidate (sentence, context) pairs per ticker.

    context = the sentence plus its immediate neighbours on both sides.
    Duplicate sentences within a filing (boilerplate) are dropped.
    """
    candidates = {}
    for ticker, accession_dir in filing_dirs.items():
        text = extract_filing_text(accession_dir)
        sentences = nltk.sent_tokenize(text)
        seen, ticker_candidates = set(), []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not is_candidate(sentence) or sentence in seen:
                continue
            seen.add(sentence)
            context = " ".join(sentences[max(0, i - 1): i + 2]).strip()
            ticker_candidates.append({"sentence": sentence, "context": context,
                                      "position": i})
        candidates[ticker] = ticker_candidates
        logger.info("%s: %d sentences, %d candidates (year %d)",
                    ticker, len(sentences), len(ticker_candidates),
                    filing_year(accession_dir))
    return candidates


# ---------------------------------------------------------------------------
# Perturbation rules (each returns the perturbed sentence or None if
# the rule is not applicable to this sentence)
# ---------------------------------------------------------------------------

def perturb_number(sentence: str, rng: random.Random) -> str | None:
    """Scale one number by ±15–60%, or swap two adjacent distinct digits."""
    matches = list(NUMBER_RE.finditer(sentence))
    if not matches:
        return None
    match = rng.choice(matches)
    original = match.group()
    value = float(original.replace(",", ""))
    digits = re.sub(r"\D", "", original)

    replaced = None
    if len(set(digits)) >= 2 and rng.random() < 0.3:
        # Try swapping the first adjacent pair of distinct digits; reject
        # results with a leading zero (e.g. 2026 -> 0226 reads as a typo,
        # not a factual perturbation) and fall back to scaling.
        chars = list(original)
        digit_positions = [i for i, c in enumerate(chars) if c.isdigit()]
        for a, b in zip(digit_positions, digit_positions[1:]):
            if chars[a] != chars[b]:
                chars[a], chars[b] = chars[b], chars[a]
                swapped = "".join(chars)
                if swapped != original and not (swapped[0] == "0"
                                                and original[0] != "0"):
                    replaced = swapped
                break
    if replaced is None:
        if value == 0:
            return None
        magnitude = rng.uniform(*config.SEC_NUMBER_SCALE_RANGE)
        factor = 1 + magnitude if rng.random() < 0.5 else 1 - magnitude
        new_value = value * factor
        decimals = len(original.split(".")[1]) if "." in original else 0
        if decimals:
            replaced = f"{new_value:,.{decimals}f}" if "," in original else f"{new_value:.{decimals}f}"
        else:
            replaced = f"{round(new_value):,}" if "," in original else str(round(new_value))
        if replaced == original:
            return None
    return sentence[:match.start()] + replaced + sentence[match.end():]


def perturb_entity(sentence: str, ticker: str, rng: random.Random) -> str | None:
    """Replace this company's name with a different ticker's company name."""
    for variant in config.SEC_NAME_VARIANTS[ticker]:
        if variant in sentence:
            donor = rng.choice([t for t in config.SEC_TICKERS
                                if t != ticker and t in config.SEC_COMPANY_NAMES])
            swapped = sentence.replace(variant, config.SEC_COMPANY_NAMES[donor], 1)
            # "Apple Inc." + sentence-final "." would leave "Inc..".
            return swapped.replace("..", ".")
    return None


def perturb_negation(sentence: str, rng: random.Random) -> str | None:
    """Flip a directional word, or insert/remove 'not'."""
    # 1. Directional antonym swap (either direction), word-boundary safe.
    pairs = ANTONYM_PAIRS + [(b, a) for a, b in ANTONYM_PAIRS]
    rng.shuffle(pairs)
    for source, target in pairs:
        pattern = re.compile(rf"\b{source}\b")
        if pattern.search(sentence):
            return pattern.sub(target, sentence, count=1)
    # 2. Remove an existing "not".
    if re.search(r"\bnot\b", sentence):
        return re.sub(r"\s?\bnot\b", "", sentence, count=1)
    # 3. Insert "not" after the first auxiliary verb.
    for aux in AUXILIARIES:
        pattern = re.compile(rf"\b{aux}\b")
        match = pattern.search(sentence)
        if match:
            return sentence[:match.end()] + " not" + sentence[match.end():]
    return None


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pairs(candidates: dict, filing_dirs: dict, logger) -> pd.DataFrame:
    """Generate the balanced three-class pair set, deterministically."""
    rng = random.Random(config.SEED)
    target = config.SEC_TARGET_PER_CLASS
    tickers = sorted(candidates.keys())
    per_ticker = -(-target // len(tickers))  # ceil division

    # Disjoint per-ticker sentence pools per class (no claim reuse across classes).
    pools = {ticker: {"supports": [], "refutes": [], "nei": []}
             for ticker in tickers}
    for ticker in tickers:
        shuffled = candidates[ticker][:]
        rng.shuffle(shuffled)
        for i, cand in enumerate(shuffled):
            pools[ticker][("supports", "refutes", "nei")[i % 3]].append(cand)

    rows = []
    refute_rule_counts = {"number_change": 0, "entity_swap": 0, "negation": 0}

    def add_row(ticker, claim, evidence, label, perturbation):
        rows.append({
            "id": f"sec_{ticker}_{len(rows):04d}",
            "claim": claim,
            "evidence_text": evidence,
            "input_text": claim + config.SEP_TOKEN + evidence,
            "label": label,
            "perturbation_type": perturbation,
            "ticker": ticker,
            "filing_year": filing_year(filing_dirs[ticker]),
        })

    # SUPPORTS: sentence verbatim against its own context window.
    for ticker in tickers:
        for cand in pools[ticker]["supports"][:per_ticker]:
            if sum(r["label"] == "SUPPORTS" for r in rows) >= target:
                break
            add_row(ticker, cand["sentence"], cand["context"], "SUPPORTS", "none")

    # REFUTES: exactly one perturbation; keep the three rules ~equal by always
    # choosing the currently least-used applicable rule.
    for ticker in tickers:
        made = 0
        for cand in pools[ticker]["refutes"]:
            if made >= per_ticker or sum(refute_rule_counts.values()) >= target:
                break
            applicable = {}
            perturbed = perturb_number(cand["sentence"], rng)
            if perturbed:
                applicable["number_change"] = perturbed
            perturbed = perturb_entity(cand["sentence"], ticker, rng)
            if perturbed:
                applicable["entity_swap"] = perturbed
            perturbed = perturb_negation(cand["sentence"], rng)
            if perturbed:
                applicable["negation"] = perturbed
            if not applicable:
                continue
            rule = min(applicable, key=lambda r: refute_rule_counts[r])
            refute_rule_counts[rule] += 1
            add_row(ticker, applicable[rule], cand["context"], "REFUTES", rule)
            made += 1

    # NOT ENOUGH INFO: unperturbed sentence against a context window from a
    # different ticker's filing, with token overlap < 0.2.
    donor_contexts = {ticker: [c["context"] for c in pools[ticker]["nei"]]
                      for ticker in tickers}
    for ticker in tickers:
        made = 0
        donors = [t for t in tickers if t != ticker]
        for cand in pools[ticker]["nei"]:
            if made >= per_ticker or sum(r["label"] == "NOT ENOUGH INFO"
                                         for r in rows) >= target:
                break
            claim = cand["sentence"]
            for _ in range(10):  # up to 10 donor attempts per claim
                donor = rng.choice(donors)
                if not donor_contexts[donor]:
                    continue
                evidence = rng.choice(donor_contexts[donor])
                if utils.token_overlap(claim, evidence) < config.SEC_NEI_MAX_OVERLAP:
                    add_row(ticker, claim, evidence, "NOT ENOUGH INFO",
                            "evidence_mismatch")
                    made += 1
                    break

    df = pd.DataFrame(rows)
    logger.info("generated pairs: %s", df["label"].value_counts().to_dict())
    logger.info("refute rules: %s", refute_rule_counts)

    # Rule summary table for the report.
    rules = [
        ("supports_verbatim", "Claim is a filing sentence verbatim; evidence is "
         "its own three-sentence context window",
         int((df["label"] == "SUPPORTS").sum())),
        ("number_change", "One number scaled by a factor of +/-15-60% or two "
         "adjacent digits swapped", refute_rule_counts["number_change"]),
        ("entity_swap", "Company name replaced with a different ticker's "
         "registered company name", refute_rule_counts["entity_swap"]),
        ("negation", "Directional word flipped (increased/decreased, "
         "above/below, gain/loss, higher/lower) or 'not' inserted/removed",
         refute_rule_counts["negation"]),
        ("evidence_mismatch", "Unperturbed sentence paired with a context "
         "window from a different filing (token overlap < 0.2)",
         int((df["label"] == "NOT ENOUGH INFO").sum())),
    ]
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rules_path = config.TABLES_DIR / "sec_perturbation_rules.csv"
    with open(rules_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "description", "count"])
        writer.writerows(rules)
    logger.info("wrote %s", rules_path)
    return df


def main() -> None:
    logger = utils.setup_logging(1)
    utils.set_seed()

    with utils.timed(logger, "download 10-K filings"):
        filing_dirs = download_filings(logger)
    with utils.timed(logger, "extract candidate sentences"):
        candidates = collect_candidates(filing_dirs, logger)
    with utils.timed(logger, "generate synthetic pairs"):
        df = generate_pairs(candidates, filing_dirs, logger)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.SEC_PARQUET, index=False)
    logger.info("saved %s: %d rows", config.SEC_PARQUET, len(df))


if __name__ == "__main__":
    main()
