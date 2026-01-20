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

# Siko_Sat_CrossAvoid_Local.py
# Cross-lottery (daily) -> Saturday Lotto decade pattern predictor
# WITH optional "avoid recent Saturday patterns" recency penalty.
#
# No CLI args: configure variables below.

import csv
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Iterable

# ==============================
# CONFIG (edit these variables)
# ==============================

# 1) Inputs
DAILY_CSV_PATH = "lotto_last_3_months.csv"  # can also be a .log containing CSV text
SATURDAY_ONLY_CSV_PATH = "Tattslotto.csv"

# 2) Mode: "PREDICT" or "BACKTEST"
MODE = "BACKTEST"

# --- PREDICT mode options ---
TARGET_DATE = "2026-01-17"  # used only when MODE="PREDICT"

# --- BACKTEST options ---
BACKTEST_LAST_N = 12                 # backtest the last N overlapping Saturdays
BACKTEST_DATES: List[str] = []       # optional: ["2026-01-17","2026-01-10"] (if non-empty, overrides BACKTEST_LAST_N)

# 3) Window rule
SKIP_FRIDAY = True                   # your rule B (skip Friday counts)
WINDOWS_Y = [1, 2, 3, 4, 5, 6]        # must be within 1..6

# 4) Predictor knobs
TOPK = 10
BIN_SIZE = 0.10                      # for decade share binning
KNN_K = 12                           # K for kNN on share vectors
SMOOTHING = 1.0                      # +alpha smoothing for bins

PRINT_DETAILS = False                # verbose prints

# 5) Pattern avoidance / penalty
# HARD_AVOID_YEARS:
#   if > 0, patterns seen within this window are fully disallowed (hard filter)
HARD_AVOID_YEARS = 0.0

# RECENCY_STEP_TABLE:
#   multiplies probability by penalty_factor if pattern seen within N years.
#   Example: avoid last 1 year strongly => 0.10 multiplier
RECENCY_STEP_TABLE = [
    (1.0, 0.10),   # seen within last 1 year => strong penalty
    (2.0, 0.70),   # seen within 1-2 years
    (3.0, 0.85),   # seen within 2-3 years
]
# If you want to "totally avoid last 1 year", set:
#   HARD_AVOID_YEARS = 1.0
# and optionally keep RECENCY_STEP_TABLE too.

# ==============================
# Decade definition (your rule)
# ==============================
DECADES = [
    (1, 10),    # D1
    (11, 20),   # D2
    (21, 30),   # D3
    (31, 40),   # D4
    (41, 45),   # D5  (Saturday Lotto max is 45)
]


# ==============================
# Helpers
# ==============================

def parse_date_any(s: str) -> date:
    s = s.strip().replace("\ufeff", "")
    # Handles: "Sat 17-Jan-2026" or "2026-01-17"
    for fmt in ("%a %d-%b-%Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {s!r}")

def safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def parse_number_list(cell: str) -> List[int]:
    """
    Parse a cell like:
      "[1, 2, 3, 10, 18, 23, 44], [4, 20]"
      "[4, 5, 9, 15, 29, 36], [17, 19]"
    Returns ALL ints found (main+supp). We use all for decade shares.
    """
    if cell is None:
        return []
    nums = re.findall(r"\d+", str(cell))
    return [int(n) for n in nums]

def num_to_decade_idx(n: int) -> Optional[int]:
    for i, (lo, hi) in enumerate(DECADES, start=1):
        if lo <= n <= hi:
            return i
    return None

def decade_counts(nums: Iterable[int]) -> Dict[str, int]:
    out = {f"D{i}": 0 for i in range(1, 6)}
    for n in nums:
        di = num_to_decade_idx(n)
        if di is None:
            continue
        out[f"D{di}"] += 1
    return out

def decade_pattern_6(main6: List[int]) -> Tuple[int, int, int, int, int]:
    """
    Given Saturday Lotto main 6 numbers, return decade quota tuple (D1..D5) summing to 6.
    """
    c = decade_counts(main6)
    return (c["D1"], c["D2"], c["D3"], c["D4"], c["D5"])

def counts_to_share_vec(c: Dict[str, int]) -> Tuple[float, float, float, float, float]:
    total = sum(c.values()) or 1
    return tuple(c[f"D{i}"] / total for i in range(1, 6))

def share_to_bin(share_vec: Tuple[float, ...], bin_size: float) -> Tuple[int, ...]:
    # quantize each share into bins
    return tuple(int(math.floor(x / bin_size + 1e-9)) for x in share_vec)

def euclid(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def daterange(d0: date, d1: date) -> List[date]:
    # inclusive
    if d1 < d0:
        return []
    days = (d1 - d0).days
    return [d0 + timedelta(days=i) for i in range(days + 1)]


# ==============================
# Load daily cross-lottery data
# ==============================

def load_daily_csv(path: str) -> Dict[date, Dict[str, List[int]]]:
    """
    Returns:
      daily[date] = {"set": [ints...], "others": [ints...]}
    Accepts actual .csv OR a .log that contains CSV lines (Date,Set...,Others...)
    """
    text = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # If it's a log containing CSV, find the CSV header and parse from there
    # We accept either "Date,Set for Life..." or already clean CSV.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.lower().startswith("date,") and "set for life" in ln.lower():
            header_idx = i
            break
    if header_idx is None:
        # maybe it's already a plain CSV file starting from first line
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

        set_nums = parse_number_list(set_cell)
        oth_nums = parse_number_list(oth_cell)

        daily[d] = {"set": set_nums, "others": oth_nums}

    return daily


# ==============================
# Load Saturday-only ground truth
# ==============================

def load_sat_csv(path: str) -> dict:
    """
    Your Tattslotto.csv format:
    Draw,Date,Winning Number 1..6,Supplementary Number 1..2

    Example:
    4639,10/01/2026,1, 8, 23, 25, 30, 41,32,37
    """
    import csv
    from datetime import datetime

    sat = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)

        # Sanity check headers
        headers = reader.fieldnames or []
        if "Date" not in headers:
            raise RuntimeError(f"Expected 'Date' column, got headers={headers}")

        required = [f"Winning Number {i}" for i in range(1, 7)]
        for c in required:
            if c not in headers:
                raise RuntimeError(f"Missing column {c}. Headers={headers}")

        row_count = 0
        kept = 0

        for row in reader:
            row_count += 1

            d_raw = (row.get("Date") or "").strip()
            if not d_raw:
                continue

            # Your date format is dd/mm/yyyy
            try:
                d = datetime.strptime(d_raw, "%d/%m/%Y").date()
            except ValueError:
                # fallback: try yyyy-mm-dd just in case
                try:
                    d = datetime.strptime(d_raw, "%Y-%m-%d").date()
                except ValueError:
                    continue

            main6 = []
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

            if not ok or len(main6) != 6:
                continue

            sat[d] = main6
            kept += 1

        print(f"[SAT CSV] path={path}")
        print(f"[SAT CSV] rows read={row_count}, draws kept={kept}")
        if kept:
            ds = sorted(sat.keys())
            print(f"[SAT CSV] date range: {ds[0]} .. {ds[-1]}")

    return sat



# ==============================
# Build training rows (history)
# ==============================

@dataclass
class TrainRow:
    sat_date: date
    y: int
    share_vec: Tuple[float, float, float, float, float]
    share_bin: Tuple[int, int, int, int, int]
    pattern: Tuple[int, int, int, int, int]

def previous_days_for_window(target_sat: date, y: int, skip_friday: bool) -> List[date]:
    """
    Window definition:
      y=1 => (special) no days (kept for compatibility; will produce empty)
      y=2 => 1 day: Thursday (if skip_friday)
      y=3 => Wed+Thu
      y=4 => Tue+Wed+Thu
      y=5 => Mon..Thu
      y=6 => Sun..Thu  (if skip_friday)
    If skip_friday=False, include Friday and shift accordingly (not recommended in your rule B).
    """
    # We treat Saturday as target day. Build backward day list.
    # Base days considered are previous 1..6 days (Sun..Fri).
    # If skip_friday=True, exclude Friday always.
    days = []
    for back in range(1, 7):  # 1..6 days
        d = target_sat - timedelta(days=back)
        if skip_friday and d.weekday() == 4:  # Fri=4
            continue
        days.append(d)
    # days currently [Fri,Thu,Wed,Tue,Mon,Sun] filtered; reverse later.

    # Map y to how many of these to keep (but aligned to "most recent first")
    if y <= 1:
        return []
    # We want the closest y-1 eligible days before Saturday
    keep = y - 1
    out = days[:keep]
    out = sorted(out)  # chronological for printing
    return out

def compute_share_for_days(daily: Dict[date, Dict[str, List[int]]], days: List[date]) -> Tuple[float, float, float, float, float]:
    nums = []
    for d in days:
        rec = daily.get(d)
        if not rec:
            continue
        nums.extend(rec.get("set", []))
        nums.extend(rec.get("others", []))
    c = decade_counts(nums)
    return counts_to_share_vec(c)

def build_training_rows(
    daily: Dict[date, Dict[str, List[int]]],
    sat_truth: Dict[date, List[int]],
    windows_y: List[int],
    skip_friday: bool,
    bin_size: float
) -> List[TrainRow]:
    rows: List[TrainRow] = []
    for sd, main6 in sat_truth.items():
        patt = decade_pattern_6(main6)
        for y in windows_y:
            days = previous_days_for_window(sd, y, skip_friday)
            share = compute_share_for_days(daily, days)
            b = share_to_bin(share, bin_size)
            rows.append(TrainRow(sd, y, share, b, patt))
    return rows


# ==============================
# Pattern recency logic (avoid)
# ==============================

def build_pattern_last_seen(sat_truth: Dict[date, List[int]]) -> Dict[Tuple[int,int,int,int,int], List[date]]:
    seen: Dict[Tuple[int,int,int,int,int], List[date]] = defaultdict(list)
    for d, main6 in sat_truth.items():
        patt = decade_pattern_6(main6)
        seen[patt].append(d)
    for patt in seen:
        seen[patt] = sorted(seen[patt])
    return seen

def pattern_penalty_multiplier(
    patt: Tuple[int,int,int,int,int],
    target_date: date,
    pattern_seen_dates: Dict[Tuple[int,int,int,int,int], List[date]],
    hard_avoid_years: float,
    step_table: List[Tuple[float, float]]
) -> float:
    dates = pattern_seen_dates.get(patt, [])
    if not dates:
        return 1.0

    # find most recent date before target_date
    prev = None
    for d in reversed(dates):
        if d < target_date:
            prev = d
            break
    if prev is None:
        return 1.0

    age_days = (target_date - prev).days
    age_years = age_days / 365.25

    if hard_avoid_years > 0 and age_years <= hard_avoid_years:
        return 0.0

    mult = 1.0
    for years, factor in step_table:
        if age_years <= years:
            mult *= factor
            break
    return mult


# ==============================
# Predictor
# ==============================

def score_patterns_for_target(
    target_sat: date,
    daily: Dict[date, Dict[str, List[int]]],
    training: List[TrainRow],
    pattern_seen_dates: Dict[Tuple[int,int,int,int,int], List[date]],
) -> List[Tuple[Tuple[int,int,int,int,int], float]]:
    """
    For each y, we compute share vec and:
      - bin model: P(pattern|bin,y)
      - knn model: P(pattern|nearest shares,y)
    Then ensemble across y with learned weights from history inside training.
    """
    # --- Learn window weights from history ---
    # We evaluate each y by log-likelihood of true patterns (simple proxy)
    y_rows: Dict[int, List[TrainRow]] = defaultdict(list)
    for r in training:
        y_rows[r.y].append(r)

    y_scores: Dict[int, float] = {}
    for y, rows in y_rows.items():
        # build bin conditional
        bin_map: Dict[Tuple[int,...], Counter] = defaultdict(Counter)
        for r in rows:
            bin_map[r.share_bin][r.pattern] += 1

        ll = 0.0
        for r in rows:
            cnt = bin_map[r.share_bin]
            total = sum(cnt.values()) + SMOOTHING * (len(cnt) + 1)
            p = (cnt[r.pattern] + SMOOTHING) / total
            ll += math.log(max(p, 1e-12))
        y_scores[y] = ll

    # softmax to weights
    max_ll = max(y_scores.values()) if y_scores else 0.0
    exps = {y: math.exp(v - max_ll) for y, v in y_scores.items()}
    z = sum(exps.values()) or 1.0
    y_weight = {y: exps[y] / z for y in exps}

    # --- Now compute per-y distribution for target date ---
    per_y_dist: Dict[int, Counter] = {}
    per_y_mode: Dict[int, str] = {}
    per_y_days: Dict[int, List[date]] = {}
    per_y_share: Dict[int, Tuple[float,...]] = {}

    for y in WINDOWS_Y:
        days = previous_days_for_window(target_sat, y, SKIP_FRIDAY)
        per_y_days[y] = days
        share = compute_share_for_days(daily, days)
        per_y_share[y] = share
        b = share_to_bin(share, BIN_SIZE)

        rows = y_rows.get(y, [])
        if not rows:
            per_y_dist[y] = Counter()
            per_y_mode[y] = "none"
            continue

        # build bin map
        bin_map: Dict[Tuple[int,...], Counter] = defaultdict(Counter)
        for r in rows:
            bin_map[r.share_bin][r.pattern] += 1

        if b in bin_map and sum(bin_map[b].values()) > 0:
            # bin mode
            per_y_dist[y] = bin_map[b].copy()
            per_y_mode[y] = "bin"
        else:
            # knn mode on share vecs
            pairs = [(euclid(share, r.share_vec), r.pattern) for r in rows]
            pairs.sort(key=lambda x: x[0])
            nn = pairs[:max(1, min(KNN_K, len(pairs)))]
            c = Counter([p for _, p in nn])
            per_y_dist[y] = c
            per_y_mode[y] = "knn"

    # --- Ensemble: combine distributions with y weights ---
    combined = Counter()
    for y in WINDOWS_Y:
        w = y_weight.get(y, 0.0)
        if w <= 0:
            continue
        dist = per_y_dist.get(y, Counter())
        tot = sum(dist.values())
        if tot <= 0:
            continue
        for patt, cnt in dist.items():
            combined[patt] += w * (cnt / tot)

    # --- Apply recency avoidance penalty using Saturday-only history ---
    final_scores: Dict[Tuple[int,int,int,int,int], float] = {}
    for patt, p in combined.items():
        mult = pattern_penalty_multiplier(
            patt=patt,
            target_date=target_sat,
            pattern_seen_dates=pattern_seen_dates,
            hard_avoid_years=HARD_AVOID_YEARS,
            step_table=RECENCY_STEP_TABLE,
        )
        final_scores[patt] = p * mult

    # Renormalize
    s = sum(final_scores.values())
    if s > 0:
        for k in list(final_scores.keys()):
            final_scores[k] /= s

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # --- Optional prints ---
    print("\n==============================")
    print("DECADE PREDICTOR RESULT")
    print("==============================")
    print(f"Target date: {target_sat}  (SKIP_FRIDAY={SKIP_FRIDAY})")
    print("\n--- Learned window weights (higher = better from history) ---")
    for y in sorted(y_weight):
        print(f"Y={y}: weight={y_weight[y]:.3f}  (score={y_scores.get(y,0.0):.3f})")

    best_y = max(y_weight.items(), key=lambda x: x[1])[0] if y_weight else None
    print(f"\nBest learned window (by weight): Y={best_y}")

    print("\n--- Window scan details (target date) ---")
    for y in WINDOWS_Y:
        days = per_y_days.get(y, [])
        share = per_y_share.get(y, (0,0,0,0,0))
        print(
            f"Y={y} mode={per_y_mode.get(y)} "
            f"days={[d.strftime('%Y-%m-%d') for d in days]} "
            f"shares={{D1:{share[0]:.2f}, D2:{share[1]:.2f}, D3:{share[2]:.2f}, D4:{share[3]:.2f}, D5:{share[4]:.2f}}}"
        )

    if ranked:
        # Expected decade counts (sum approx 6)
        exp = [0.0]*5
        for patt, p in ranked[:200]:
            for i in range(5):
                exp[i] += patt[i] * p
        print("\n--- Ensemble expected decade counts (sum≈6) ---")
        print(f"D1={exp[0]:.2f}, D2={exp[1]:.2f}, D3={exp[2]:.2f}, D4={exp[3]:.2f}, D5={exp[4]:.2f}")

    print(f"\n--- Top predicted decade patterns (counts across D1..D5 summing to 6) ---")
    for i, (patt, p) in enumerate(ranked[:TOPK], 1):
        print(f"{i:02d}) P={p:.3f}  [D1={patt[0]}, D2={patt[1]}, D3={patt[2]}, D4={patt[3]}, D5={patt[4]}]")

    print("\n(Each pattern is a Saturday Lotto decade quota for 6 numbers.)")

    if PRINT_DETAILS and ranked:
        print("\n--- Debug: top 25 patterns with penalties applied ---")
        for patt, p in ranked[:25]:
            mult = pattern_penalty_multiplier(
                patt=patt,
                target_date=target_sat,
                pattern_seen_dates=pattern_seen_dates,
                hard_avoid_years=HARD_AVOID_YEARS,
                step_table=RECENCY_STEP_TABLE,
            )
            print(f"{patt} p={p:.5f} mult={mult:.3f}")

    return ranked


# ==============================
# Run modes
# ==============================

def predict_one(target_date_str: str):
    daily_all = load_daily_csv(DAILY_CSV_PATH)
    sat_all = load_sat_csv(SATURDAY_ONLY_CSV_PATH)

    td = parse_date_any(target_date_str)

    sat_hist = {d: nums for d, nums in sat_all.items() if d < td}
    daily_hist = {d: rec for d, rec in daily_all.items() if d < td}

    training = build_training_rows(daily_hist, sat_hist, WINDOWS_Y, SKIP_FRIDAY, BIN_SIZE)
    seen = build_pattern_last_seen(sat_hist)

    score_patterns_for_target(td, daily_hist, training, seen)



def backtest_dates(dates: List[str]):
    daily = load_daily_csv(DAILY_CSV_PATH)
    sat = load_sat_csv(SATURDAY_ONLY_CSV_PATH)
    training = build_training_rows(daily, sat, WINDOWS_Y, SKIP_FRIDAY, BIN_SIZE)
    seen = build_pattern_last_seen(sat)

    targets = [parse_date_any(x) for x in dates]
    hits = 0
    total = 0

    print("\n==============================")
    print("BACKTEST (explicit dates)")
    print("==============================")
    for td in targets:
        if td not in sat:
            print(f"{td} | missing in Saturday CSV -> skipped")
            continue
        ranked = score_patterns_for_target(td, daily, training, seen)
        actual = decade_pattern_6(sat[td])
        top_patterns = [p for p, _ in ranked[:TOPK]]
        hit = (actual in top_patterns)
        hits += int(hit)
        total += 1
        print(f"\nRESULT {td} | actual={actual} | top1={top_patterns[0] if top_patterns else None} | top{TOPK}_hit={int(hit)}")

    total = total or 1
    print("\n==============================")
    print("BACKTEST SUMMARY")
    print("==============================")
    print(f"Total tested Saturdays: {total}")
    print(f"Top{TOPK} exact-pattern hits: {hits}/{total} = {hits/total:.3f}")


def backtest_last_n(n: int):
    daily_all = load_daily_csv(DAILY_CSV_PATH)
    sat_all = load_sat_csv(SATURDAY_ONLY_CSV_PATH)

    all_sats = sorted(sat_all.keys())
    if not all_sats:
        raise RuntimeError("No Saturday draws loaded from Saturday-only CSV")

    targets = all_sats[-n:]

    hits = 0
    rows = []

    print("\n==============================")
    print("BACKTEST (STRICT, NO LOOKAHEAD)")
    print("==============================")

    for td in targets:
        # --- STRICT PAST-ONLY CUTS (no leakage) ---
        sat_hist = {d: nums for d, nums in sat_all.items() if d < td}
        if len(sat_hist) < 6:
            # not enough history to train reliably
            rows.append((td.strftime("%Y-%m-%d"), None, None, 0, "SKIP:not_enough_history"))
            continue

        daily_hist = {d: rec for d, rec in daily_all.items() if d < td}

        training = build_training_rows(daily_hist, sat_hist, WINDOWS_Y, SKIP_FRIDAY, BIN_SIZE)
        seen = build_pattern_last_seen(sat_hist)

        ranked = score_patterns_for_target(td, daily_hist, training, seen)

        actual = decade_pattern_6(sat_all[td])  # ground truth for that Saturday
        top_patterns = [p for p, _ in ranked[:TOPK]]
        hit = int(actual in top_patterns)

        hits += hit
        top1 = top_patterns[0] if top_patterns else None
        rows.append((td.strftime("%Y-%m-%d"), actual, top1, hit, ""))

    tested = sum(1 for r in rows if r[4] == "")
    print("\n==============================")
    print("BACKTEST SUMMARY (STRICT)")
    print("==============================")
    print(f"DAILY_CSV_PATH={DAILY_CSV_PATH}")
    print(f"SATURDAY_ONLY_CSV_PATH={SATURDAY_ONLY_CSV_PATH}")
    print(f"SKIP_FRIDAY={SKIP_FRIDAY} WINDOWS_Y={WINDOWS_Y} BIN_SIZE={BIN_SIZE} KNN_K={KNN_K} SMOOTHING={SMOOTHING}")
    print(f"HARD_AVOID_YEARS={HARD_AVOID_YEARS} RECENCY_STEP_TABLE={RECENCY_STEP_TABLE} TOPK={TOPK}")
    print(f"Total tested Saturdays: {tested}")
    if tested:
        print(f"Top{TOPK} exact-pattern hits: {hits}/{tested} = {hits/tested:.3f}")

    print("\nPer-date:")
    for d, actual, top1, hit, note in rows:
        if note:
            print(f"{d} | {note}")
        else:
            print(f"{d} | actual={actual} | top1={top1} | hit={hit}")



if __name__ == "__main__":
    if MODE.upper() == "PREDICT":
        predict_one(TARGET_DATE)
    else:
        if BACKTEST_DATES:
            backtest_dates(BACKTEST_DATES)
        else:
            backtest_last_n(BACKTEST_LAST_N)
            predict_one(TARGET_DATE)

