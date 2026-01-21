import sys
import os
from datetime import datetime

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except OSError:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except OSError:
                pass

log_file_path = os.path.join(
    ".",
    "Siko_Sat_Decades.py.log"   # single growing log file
)

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# Siko_Sat_Decades_Final.py
# ------------------------------------------------------------
# FINAL: Saturday Lotto decade quota predictor (6 numbers)
# - Uses cross-lottery daily data (Set for Life + Others) ONLY as a directional "pressure" signal
# - Predicts decades via suppression-first voting across windows Y=2..6
# - Then generates candidate decade patterns (D1..D5 sum=6) with HARD per-decade bounds
# - Applies configurable pattern recency avoidance using Saturday-only history (NO lookahead)
# - Backtest is STRICT: for each target Saturday, training uses only dates < target
#
# NO CLI args: edit variables below.
# ------------------------------------------------------------

import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

# ============================================================
# CONFIG (edit these)
# ============================================================

# Input files
DAILY_CSV_PATH = "lotto_last_3_months.csv"   # can also be a .log containing CSV text
SATURDAY_ONLY_CSV_PATH = "Tattslotto.csv"    # Saturday-only CSV (Draw,Date,Winning Number 1..6,...)

# Mode: "PREDICT" or "BACKTEST"
MODE = "BACKTEST"

# Predict settings
TARGET_DATE = "2026-01-17"  # Saturday date you want decades for (YYYY-MM-DD)

# Backtest settings (last N Saturdays from SATURDAY_ONLY_CSV_PATH)
BACKTEST_LAST_N = 16
BACKTEST_DATES: List[str] = []  # if non-empty, overrides BACKTEST_LAST_N

# Window settings
SKIP_FRIDAY = True
WINDOWS_Y = [2, 3, 4, 5, 6]   # IMPORTANT: suppression votes need at least 1 day back; use 2..6

# Cross-lottery inputs used for pressure signal
USE_SET_FOR_LIFE = True
USE_OTHERS = True

# Pressure scoring
PRESSURE_MODE = "share"   # "share" or "count"
PRESSURE_AGG = "sum"      # "sum" or "mean" across days in a window

# Voting
VOTE_THRESHOLD = 0.06     # how far from baseline share to call "heavy" / "light"
HEAVY_VOTE = -1.0         # if decade heavy in window => vote suppression
LIGHT_VOTE = +1.0         # if decade light in window => vote boost
NEUTRAL_VOTE = 0.0

# Candidate generation (hard bounds)
# You said historically D5=3 never happens; set D5 max=2.
MAX_PER_DECADE = {"D1": 2, "D2": 3, "D3": 3, "D4": 3, "D5": 1}
MIN_PER_DECADE = {"D1": 0, "D2": 0, "D3": 0, "D4": 0, "D5": 0}

# Candidate ranking
TOPK = 5
LAPLACE = 0.50           # smoothing for priors
PREFER_BALANCE = 0.12    # penalty strength for "too peaky" patterns

# Recency avoidance (Saturday-only patterns)
# HARD_AVOID_YEARS: if >0, patterns seen within this many years are completely forbidden.
HARD_AVOID_YEARS = 1.0

# Soft recency penalties (applied if not hard-avoided)
# age <= yrs => multiplier
RECENCY_STEP_TABLE = [
    (1.0, 0.10),
    (2.0, 0.70),
    (3.0, 0.85),
]

# Output verbosity
PRINT_DETAILS = True

# ============================================================
# Decades definition
# ============================================================
DECADES = [
    (1, 10),   # D1
    (11, 20),  # D2
    (21, 30),  # D3
    (31, 40),  # D4
    (41, 45),  # D5
]

D_KEYS = ["D1", "D2", "D3", "D4", "D5"]

# ============================================================
# Parsing helpers
# ============================================================

def parse_date_any(s: str) -> date:
    s = (s or "").strip().replace("\ufeff", "")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%a %d-%b-%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date: {s!r}")

def parse_number_list(cell: str) -> List[int]:
    if cell is None:
        return []
    nums = re.findall(r"\d+", str(cell))
    return [int(n) for n in nums]

def num_to_decade_key(n: int) -> Optional[str]:
    for i, (lo, hi) in enumerate(DECADES, start=1):
        if lo <= n <= hi:
            return f"D{i}"
    return None

def decade_counts(nums: Iterable[int]) -> Dict[str, int]:
    c = {k: 0 for k in D_KEYS}
    for n in nums:
        dk = num_to_decade_key(int(n))
        if dk:
            c[dk] += 1
    return c

def decade_share(nums: Iterable[int]) -> Dict[str, float]:
    c = decade_counts(nums)
    total = sum(c.values()) or 1
    return {k: c[k] / total for k in D_KEYS}

def saturday_pattern(main6: List[int]) -> Tuple[int, int, int, int, int]:
    c = decade_counts(main6)
    return tuple(c[k] for k in D_KEYS)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ============================================================
# Loading data
# ============================================================

def load_daily_csv_or_log(path: str) -> Dict[date, Dict[str, List[int]]]:
    """
    Expects lines like:
      Date,Set for Life (incl supp),Others (incl supp)
      Fri 16-Jan-2026,"[...], [...]","[...], [...]"
    Or a .log containing that CSV block.
    Returns daily[d] = {"set":[...], "others":[...]}
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("date,") and "set for life" in ln.lower():
            header_idx = i
            break
    if header_idx is None:
        # best-effort: assume first non-empty line is header
        header_idx = 0

    csv_lines = lines[header_idx:]
    reader = csv.DictReader(csv_lines)

    daily: Dict[date, Dict[str, List[int]]] = {}
    for row in reader:
        d_raw = (row.get("Date") or "").strip()
        if not d_raw:
            continue
        d = parse_date_any(d_raw)

        set_cell = row.get("Set for Life (incl supp)") or row.get("Set for Life (incl supps)") or ""
        oth_cell = row.get("Others (incl supp)") or row.get("Others (incl supps)") or ""

        set_nums = [n for n in parse_number_list(set_cell) if 1 <= n <= 45]
        oth_nums = [n for n in parse_number_list(oth_cell) if 1 <= n <= 45]

        daily[d] = {"set": set_nums, "others": oth_nums}

    return daily

def load_saturday_csv(path: str) -> Dict[date, List[int]]:
    """
    Tattslotto.csv format:
      Draw,Date,Winning Number 1..6,Supplementary Number 1..2
    Date typically dd/mm/yyyy.
    """
    sat: Dict[date, List[int]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        required = [f"Winning Number {i}" for i in range(1, 7)]
        for c in required:
            if c not in headers:
                raise RuntimeError(f"Missing {c}. Headers={headers}")

        for row in reader:
            d_raw = (row.get("Date") or "").strip()
            if not d_raw:
                continue
            try:
                d = datetime.strptime(d_raw, "%d/%m/%Y").date()
            except ValueError:
                try:
                    d = datetime.strptime(d_raw, "%Y-%m-%d").date()
                except ValueError:
                    continue

            main6: List[int] = []
            ok = True
            for i in range(1, 7):
                v = (row.get(f"Winning Number {i}") or "").strip()
                if not v:
                    ok = False
                    break
                try:
                    main6.append(int(v))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            sat[d] = main6

    return sat

# ============================================================
# Window selection
# ============================================================

def previous_days_for_window(target_sat: date, y: int, skip_friday: bool) -> List[date]:
    """
    y=2 => [Sat-1]
    y=3 => [Sat-2, Sat-1]
    ...
    y=6 => [Sat-5, Sat-4, Sat-3, Sat-2, Sat-1]
    (Then remove Friday if skip_friday=True)
    """
    days = [target_sat - timedelta(days=i) for i in range(1, y)]
    if skip_friday:
        days = [d for d in days if d.weekday() != 4]  # Friday
    days.sort()
    return days

# ============================================================
# Recency penalty (Saturday-only)
# ============================================================

def build_pattern_last_seen(sat_hist: Dict[date, List[int]]) -> Dict[Tuple[int, int, int, int, int], date]:
    last = {}
    for d in sorted(sat_hist.keys()):
        last[saturday_pattern(sat_hist[d])] = d
    return last

def recency_multiplier(pat: Tuple[int, int, int, int, int], target: date,
                       last_seen: Dict[Tuple[int, int, int, int, int], date]) -> float:
    d = last_seen.get(pat)
    if not d:
        return 1.0
    age_years = (target - d).days / 365.25

    if HARD_AVOID_YEARS and age_years <= HARD_AVOID_YEARS:
        return 0.0

    for yrs, mul in RECENCY_STEP_TABLE:
        if age_years <= yrs:
            return mul
    return 1.0

# ============================================================
# Suppression-first decade voting
# ============================================================

def collect_window_numbers(daily_hist: Dict[date, Dict[str, List[int]]], days: List[date]) -> List[int]:
    nums: List[int] = []
    for d in days:
        rec = daily_hist.get(d)
        if not rec:
            continue
        if USE_SET_FOR_LIFE:
            nums.extend(rec["set"])
        if USE_OTHERS:
            nums.extend(rec["others"])
    return nums

def window_pressure(days: List[date], daily_hist: Dict[date, Dict[str, List[int]]]) -> Dict[str, float]:
    nums = collect_window_numbers(daily_hist, days)
    if PRESSURE_MODE == "count":
        c = decade_counts(nums)
        if PRESSURE_AGG == "mean":
            denom_days = max(1, len(days))
            return {k: c[k] / denom_days for k in D_KEYS}
        return {k: float(c[k]) for k in D_KEYS}

    # share mode
    # baseline share for "uniform decades" isn't uniform by range width, but your decades are roughly similar.
    # Still, the data itself will define deviations; we use 0.2 baseline as reference.
    s = decade_share(nums)
    return s

def suppression_votes_for_target(target: date, daily_hist: Dict[date, Dict[str, List[int]]]) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    Returns:
      votes[Di] aggregated across windows
      per_window_delta[y][Di] for debugging (delta from baseline share)
    """
    votes = {k: 0.0 for k in D_KEYS}
    per_window_delta: Dict[int, Dict[str, float]] = {}

    baseline = {k: 0.2 for k in D_KEYS}  # simple reference baseline

    for y in WINDOWS_Y:
        days = previous_days_for_window(target, y, SKIP_FRIDAY)
        p = window_pressure(days, daily_hist)

        # In share-mode: compare to baseline share
        # In count-mode: compare to mean count across decades
        if PRESSURE_MODE == "count":
            mean = sum(p.values()) / len(D_KEYS)
            delta = {k: (p[k] - mean) / (mean + 1e-9) for k in D_KEYS}  # normalized
        else:
            delta = {k: (p[k] - baseline[k]) for k in D_KEYS}

        per_window_delta[y] = delta

        for k in D_KEYS:
            if delta[k] >= VOTE_THRESHOLD:
                votes[k] += HEAVY_VOTE
            elif delta[k] <= -VOTE_THRESHOLD:
                votes[k] += LIGHT_VOTE
            else:
                votes[k] += NEUTRAL_VOTE

    return votes, per_window_delta

# ============================================================
# Candidate pattern generation + scoring
# ============================================================

def enumerate_patterns(max_per: Dict[str, int], min_per: Dict[str, int]) -> List[Tuple[int, int, int, int, int]]:
    out = []
    # brute force with constraints (small space)
    for a in range(min_per["D1"], max_per["D1"] + 1):
        for b in range(min_per["D2"], max_per["D2"] + 1):
            for c in range(min_per["D3"], max_per["D3"] + 1):
                for d in range(min_per["D4"], max_per["D4"] + 1):
                    e = 6 - a - b - c - d
                    if e < min_per["D5"] or e > max_per["D5"]:
                        continue
                    out.append((a, b, c, d, e))
    return out

def pattern_balance_penalty(pat: Tuple[int, int, int, int, int]) -> float:
    # penalize extremely peaky distributions (e.g. 5 in one decade)
    # simple: sum of squares above balanced
    target = 6 / 5.0
    return sum((x - target) ** 2 for x in pat)

def learn_pattern_prior(sat_hist: Dict[date, List[int]], laplace: float) -> Dict[Tuple[int, int, int, int, int], float]:
    """
    Prior from Saturday-only history (pre-target): P(pattern)
    """
    cnt = Counter()
    for d in sat_hist:
        cnt[saturday_pattern(sat_hist[d])] += 1

    total = sum(cnt.values())
    # Laplace smoothing over observed patterns only (we’ll normalize later across candidate list)
    prior = {}
    for pat, n in cnt.items():
        prior[pat] = (n + laplace) / (total + laplace * max(1, len(cnt)))
    return prior

def score_candidates(
    candidates: List[Tuple[int, int, int, int, int]],
    votes: Dict[str, float],
    prior: Dict[Tuple[int, int, int, int, int], float],
    last_seen: Dict[Tuple[int, int, int, int, int], date],
    target: date
) -> List[Tuple[Tuple[int, int, int, int, int], float]]:
    """
    Scoring components:
      - vote alignment: decades that are suppressed (negative votes) should be 0-1, boosted should be 1-3
      - Saturday-only prior
      - balance penalty
      - recency multiplier (hard avoid + soft)
    """
    scored = []
    for pat in candidates:
        rec_mul = recency_multiplier(pat, target, last_seen)
        if rec_mul <= 0.0:
            continue

        # vote alignment
        # If votes[k] is negative => suppress => prefer low count
        # If votes[k] is positive => boost => prefer moderate count
        # Map count -> desirability:
        #   suppress: best at 0, then 1, then 2.. (steep)
        #   boost: best around 1-2, then 0 or 3, then 4..
        align = 0.0
        for k, x in zip(D_KEYS, pat):
            v = votes[k]
            if v < 0:
                # suppressed
                align += -v * (2.5 if x == 0 else 1.5 if x == 1 else 0.3 if x == 2 else 0.05)
            elif v > 0:
                # boosted
                align += v * (2.0 if x in (1, 2) else 0.8 if x == 3 else 0.4 if x == 0 else 0.05)
            else:
                align += 0.0

        # prior from Saturday-only
        pr = prior.get(pat, 1e-6)

        # balance penalty (smaller is better)
        bal = pattern_balance_penalty(pat)

        # combine multiplicatively in log space
        score = math.log(pr + 1e-12) + 0.75 * math.log(align + 1e-9) - PREFER_BALANCE * bal
        score += math.log(rec_mul + 1e-12)

        scored.append((pat, score))

    # softmax normalize
    if not scored:
        return []
    m = max(s for _, s in scored)
    exps = [(pat, math.exp(s - m)) for pat, s in scored]
    Z = sum(p for _, p in exps) or 1.0
    ranked = [(pat, p / Z) for pat, p in exps]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked

# ============================================================
# End-to-end prediction for a target Saturday
# ============================================================

def predict_decade_patterns_for_target(target: date, daily_hist: Dict[date, Dict[str, List[int]]],
                                       sat_hist: Dict[date, List[int]]) -> Dict:
    votes, per_win_delta = suppression_votes_for_target(target, daily_hist)
    last_seen = build_pattern_last_seen(sat_hist)
    prior = learn_pattern_prior(sat_hist, LAPLACE)

    # Hard max: enforce observed max D5 from sat_hist as extra safety (and your config cap)
    observed_max_d5 = 0
    for pat in last_seen.keys():
        observed_max_d5 = max(observed_max_d5, pat[4])
    max_per = dict(MAX_PER_DECADE)
    max_per["D5"] = min(max_per["D5"], observed_max_d5 if observed_max_d5 else max_per["D5"])

    candidates = enumerate_patterns(max_per=max_per, min_per=MIN_PER_DECADE)
    ranked = score_candidates(candidates, votes, prior, last_seen, target)

    return {
        "target": target,
        "votes": votes,
        "per_window_delta": per_win_delta,
        "ranked": ranked,
        "candidate_count": len(candidates),
        "max_d5_used": max_per["D5"],
    }

# ============================================================
# Strict backtest (NO lookahead)
# ============================================================

def backtest_strict(daily_all: Dict[date, Dict[str, List[int]]], sat_all: Dict[date, List[int]],
                    targets: List[date]) -> None:
    hits_top1 = 0
    hits_topk = 0
    tested = 0

    print("\n==============================")
    print("BACKTEST SUMMARY (STRICT)")
    print("==============================")
    print(f"DAILY_CSV_PATH={DAILY_CSV_PATH}")
    print(f"SATURDAY_ONLY_CSV_PATH={SATURDAY_ONLY_CSV_PATH}")
    print(f"WINDOWS_Y={WINDOWS_Y} SKIP_FRIDAY={SKIP_FRIDAY}")
    print(f"PRESSURE_MODE={PRESSURE_MODE} PRESSURE_AGG={PRESSURE_AGG} VOTE_THRESHOLD={VOTE_THRESHOLD}")
    print(f"HARD_AVOID_YEARS={HARD_AVOID_YEARS} RECENCY_STEP_TABLE={RECENCY_STEP_TABLE}")
    print(f"MAX_PER_DECADE={MAX_PER_DECADE} TOPK={TOPK}")
    print("")

    for td in targets:
        # STRICT: history must be < td
        sat_hist = {d: nums for d, nums in sat_all.items() if d < td}
        daily_hist = {d: rec for d, rec in daily_all.items() if d < td}

        if len(sat_hist) < 30:
            if PRINT_DETAILS:
                print(f"{td} | SKIP (not enough Saturday history: {len(sat_hist)})")
            continue

        res = predict_decade_patterns_for_target(td, daily_hist, sat_hist)
        ranked = res["ranked"]

        actual = saturday_pattern(sat_all[td])
        top = [p for p, _ in ranked[:TOPK]]
        top1 = top[0] if top else None

        hit1 = int(top1 == actual)
        hitk = int(actual in top)

        tested += 1
        hits_top1 += hit1
        hits_topk += hitk

        print(f"{td} | actual={actual} | top1={top1} | hit1={hit1} | hit@{TOPK}={hitk}")

        if PRINT_DETAILS:
            v = {k: round(res["votes"][k], 2) for k in D_KEYS}
            print(f"    votes={v}  max_d5_used={res['max_d5_used']}  candidates={res['candidate_count']}")
            # show one window delta as a quick sanity check
            y_show = max(WINDOWS_Y)
            delta = res["per_window_delta"].get(y_show, {})
            delta2 = {k: round(delta.get(k, 0.0), 3) for k in D_KEYS}
            print(f"    delta@Y{y_show}={delta2}")

    print("\n==============================")
    print("RESULTS")
    print("==============================")
    print(f"Total tested Saturdays: {tested}")
    if tested:
        print(f"Top1 exact-pattern:  {hits_top1}/{tested} = {hits_top1/tested:.3f}")
        print(f"Top{TOPK} exact-pattern: {hits_topk}/{tested} = {hits_topk/tested:.3f}")

# ============================================================
# Main
# ============================================================

def main():
    # Load
    daily_all = load_daily_csv_or_log(DAILY_CSV_PATH)
    sat_all = load_saturday_csv(SATURDAY_ONLY_CSV_PATH)

    if not sat_all:
        raise RuntimeError("No Saturday draws loaded from Saturday-only CSV. Check headers and Date format dd/mm/yyyy.")

    sat_dates = sorted(sat_all.keys())

    if MODE.upper() == "BACKTEST":
        if BACKTEST_DATES:
            targets = [parse_date_any(s) for s in BACKTEST_DATES]
            targets = [d for d in targets if d in sat_all]
            targets.sort()
        else:
            targets = sat_dates[-BACKTEST_LAST_N:]

        backtest_strict(daily_all, sat_all, targets)

    else:
        td = parse_date_any(TARGET_DATE)

        # STRICT: use only history < td
        sat_hist = {d: nums for d, nums in sat_all.items() if d < td}
        daily_hist = {d: rec for d, rec in daily_all.items() if d < td}

        if len(sat_hist) < 30:
            raise RuntimeError(f"Not enough Saturday history before {td} (have {len(sat_hist)}).")

        res = predict_decade_patterns_for_target(td, daily_hist, sat_hist)

        print("\n==============================")
        print("DECADE PREDICTOR RESULT (FINAL)")
        print("==============================")
        print(f"Target date: {td}  (SKIP_FRIDAY={SKIP_FRIDAY})")
        print(f"Candidates considered: {res['candidate_count']}  max_d5_used={res['max_d5_used']}\n")

        votes = {k: round(res["votes"][k], 2) for k in D_KEYS}
        print(f"Suppression/Boost votes: {votes}")
        print(f"Rule of thumb: negative vote => suppress (0-1), positive => boost (1-2)\n")

        print(f"--- Top {TOPK} predicted decade patterns (sum=6) ---")
        ranked = res["ranked"][:TOPK]
        for i, (pat, p) in enumerate(ranked, 1):
            print(f"{i:02d}) P={p:.3f}  [D1={pat[0]}, D2={pat[1]}, D3={pat[2]}, D4={pat[3]}, D5={pat[4]}]")

        if PRINT_DETAILS:
            print("\n--- Window deltas (diagnostic) ---")
            for y in WINDOWS_Y:
                delta = res["per_window_delta"].get(y, {})
                delta2 = {k: round(delta.get(k, 0.0), 3) for k in D_KEYS}
                print(f"Y={y} delta={delta2}")

if __name__ == "__main__":
    main()
