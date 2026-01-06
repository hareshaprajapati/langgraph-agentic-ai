import sys
import os
import json
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
    "siko_Tue_single_3_hits_logs.log"   # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple, Optional, Callable
import random
import math

# ============================================================
# USER CONFIG (edit only these)
# Oz Lotto: 7 main numbers, range 1-47
# ============================================================

CSV_PATH = "Oz_Lotto_transformed.csv"
# TARGET_DATE = "2025-12-27"
TARGET_DATE = "2025-12-30"

NUM_TICKETS = 20
NUMBERS_PER_TICKET = 7

MAIN_MIN = 1
MAIN_MAX = 47

LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "210"))
SEASON_WINDOW_DAYS = int(os.environ.get("SEASON_WINDOW_DAYS", "9"))
SEASON_LOOKBACK_YEARS = int(os.environ.get("SEASON_LOOKBACK_YEARS", "20"))

# Candidate pool
POOL_SIZE = int(os.environ.get("POOL_SIZE", "32"))
MID_POOL_SIZE = int(os.environ.get("MID_POOL_SIZE", "10"))
COLD_POOL_SIZE = int(os.environ.get("COLD_POOL_SIZE", "12"))
HOT_POOL_SIZE = int(os.environ.get("HOT_POOL_SIZE", "10"))
OVERDUE_POOL_SIZE = int(os.environ.get("OVERDUE_POOL_SIZE", "10"))
SEASON_POOL_SIZE = int(os.environ.get("SEASON_POOL_SIZE", "10"))
COLD_FORCE_COUNT = int(os.environ.get("COLD_FORCE_COUNT", "2"))

# Hard-force coverage mix
FORCE_COVERAGE = False
RANDOM_SEED = 0
DEBUG_PRINT = True
PRINT_ALL_SCORES_WHEN_REAL = True

# Score weights (date-agnostic)
W_RECENT = float(os.environ.get("W_RECENT", "0.55"))
W_LONG = float(os.environ.get("W_LONG", "0.20"))
W_SEASON = float(os.environ.get("W_SEASON", "0.15"))
W_RANK = float(os.environ.get("W_RANK", "0.10"))
COLD_BOOST = float(os.environ.get("COLD_BOOST", "0.25"))
# Overdue gap boost (date-agnostic)
W_GAP = float(os.environ.get("W_GAP", "0.25"))
GAP_CAP = float(os.environ.get("GAP_CAP", "0.30"))

# Ticket constraints (soft)
OVERLAP_CAP = int(os.environ.get("OVERLAP_CAP", "6"))
GLOBAL_MAX_USES = int(os.environ.get("GLOBAL_MAX_USES", "6"))

# Odd / low / sum preferences (learned from history)
ODD_BAND = (2, 5)
LOW_RANGE_MAX = 23
LOW_BAND = (2, 5)
SUM_BAND_QUANTILES = (0.25, 0.75)
CONSECUTIVE_MAX = 2

# Decade bands
DECADE_BANDS: List[Tuple[int, int, int]] = [
    (1, 1, 10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 40),
    (5, 41, 47),
]

# Seasonal decade weighting (date-agnostic, learned from history)
SEASON_DECADE_WEIGHT = 0.6

# Optional: enforce exact decade counts (set None to disable)
# Example: {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}
DECADE_TARGET_COUNTS = None

# Acceptance behavior
PENALTY_SCALE = float(os.environ.get("PENALTY_SCALE", "0.55"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "30000"))

# Cohesion strategy (3+ hit traits)
COHESION_ENABLED = True
COHESION_TOP_RANK = 30
COHESION_MAX_SCORE_SPREAD = 0.22
COHESION_MAX_RANK_GAP = 10
COHESION_PAIR_BASE_PCTL = 75
COHESION_ACCEPT_FLOOR = 0.70
COHESION_ACCEPT_SPAN = 0.60
COHESION_W_SPREAD = 0.30
COHESION_W_PAIR = 0.35
COHESION_W_RANK_CONT = 0.20
COHESION_W_CENTRAL = 0.15
# DEFAULT_STRATEGY_NAME = "RANK_CONT_HEAVY"
DEFAULT_STRATEGY_NAME = "COHESION_SOFT"
DEFAULT_PRESET = "OZ_BASELINE"
# DEFAULT_STRATEGY_NAME = "PAIR_HEAVY"

# Sweep mode (strategy backtest comparison)
SWEEP_MODE = False
SWEEP_BACKTEST_DRAWS = 12
VARIANT_SWEEP = False
VARIANT_BACKTEST_DRAWS = 12

# Oz Lotto tuner (auto-select best config on recent draws)
TUNER_MODE = os.environ.get("TUNER_MODE", "1").strip() == "1"
TUNER_BACKTEST_DRAWS = 10
TUNER_MIN_WEEKS_WITH_4 = 1

# Portfolio selection (20-ticket optimizer)
PORTFOLIO_MODE = False
PORTFOLIO_CANDIDATES = 50000
COHESIVE_TICKETS = 56
DIFFUSE_TICKETS = 24
CORE_SIZE = 20
CORE_MIN_USE = 14
CORE_MAX_USE = 26
BASE_TICKET_COUNT = 3
BASE_SWAP_VARIANTS = 6
SWAP_CANDIDATE_BAND = 20
STRICT_MAX_SPREAD = 0.16
STRICT_MAX_GAP = 6
DIFFUSE_MAX_SPREAD = 0.28
DIFFUSE_MAX_GAP = 12
COHESIVE_DIVERSITY_PENALTY = 0.30
DIFFUSE_DIVERSITY_PENALTY = 0.15

# Optional: verify against a known real draw (set [] to disable)
REAL_DRAW = []
# If TARGET_DATE is missing in CSV, optionally use REAL_DRAW for hit summary.
USE_REAL_DRAW_FALLBACK = False

# Supplementary targeting (Oz Lotto has 3 supplementary numbers)
SUPP_MIN_HIT = 0
SUPP_POOL_SIZE = 12
SUPP_W_RECENT = 0.60
SUPP_W_LONG = 0.30
SUPP_W_SEASON = 0.10
SUPP_SCORE_W = float(os.environ.get("SUPP_SCORE_W", "0.10"))
SUPP_SOFT_SWAP = True
SUPP_SWAP_TOP = int(os.environ.get("SUPP_SWAP_TOP", "24"))
SUPP_SWAP_SCORE_DELTA = float(os.environ.get("SUPP_SWAP_DELTA", "-0.02"))
SUPP_SWAP_TICKETS = int(os.environ.get("SUPP_SWAP_TICKETS", "0"))
SUPP_FOCUS_TICKETS = int(os.environ.get("SUPP_FOCUS_TICKETS", "2"))
SUPP_FOCUS_COUNT = int(os.environ.get("SUPP_FOCUS_COUNT", "2"))
SUPP_FOCUS_TOP = int(os.environ.get("SUPP_FOCUS_TOP", "47"))
SUPP_FOCUS_MODE = os.environ.get("SUPP_FOCUS_MODE", "HIGH").strip().upper()
USE_PROFILE_FILTER = False

# Filled at runtime after CSV load
SUPP_COLS: List[str] = []

# ============================================================
# INTERNALS
# ============================================================

@dataclass
class CandidateScore:
    n: int
    total_score: float
    freq_recent: int
    freq_long: int
    freq_season: int
    rank_recent: int
    gap_days: int


def _parse_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d, dayfirst=True, errors="coerce")


def _detect_main_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if "Winning Number" in c]
    if not cols:
        raise ValueError("Could not find main-number columns like 'Winning Number 1..6' in CSV.")
    import re
    def key(c):
        m = re.search(r"(\d+)$", c.strip())
        return int(m.group(1)) if m else 999
    cols = sorted(cols, key=key)
    if len(cols) < NUMBERS_PER_TICKET:
        raise ValueError(f"Expected at least {NUMBERS_PER_TICKET} main columns.")
    return cols[:NUMBERS_PER_TICKET]


def _detect_supp_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if "Supplementary" in c]
    if not cols:
        return []
    import re
    def key(c):
        m = re.search(r"(\d+)$", c.strip())
        return int(m.group(1)) if m else 999
    return sorted(cols, key=key)


def _load_ga_preset_file(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _load_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    date_col = "Date" if "Date" in df.columns else "Draw date"
    if date_col not in df.columns:
        raise ValueError("CSV must contain column: 'Date'")
    df = df.copy()
    df["Date"] = df[date_col].apply(_parse_date)
    df = df.dropna(subset=["Date"]).copy()

    main_cols = _detect_main_cols(df)
    for c in main_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=main_cols).copy()
    for c in main_cols:
        df[c] = df[c].astype(int)

    for c in main_cols:
        df = df[(df[c] >= MAIN_MIN) & (df[c] <= MAIN_MAX)]

    df = df.sort_values("Date").reset_index(drop=True)
    return df, main_cols


def _explode_mains(df: pd.DataFrame, main_cols: List[str]) -> pd.Series:
    return pd.concat([df[c] for c in main_cols], ignore_index=True)


def _counts_mains_in_window(df: pd.DataFrame, main_cols: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sub = df[(df["Date"] >= start) & (df["Date"] < end)]
    nums = _explode_mains(sub, main_cols)
    return nums.value_counts().reindex(range(MAIN_MIN, MAIN_MAX + 1), fill_value=0).sort_index()


def _explode_supps(df: pd.DataFrame, supp_cols: List[str]) -> pd.Series:
    return pd.concat([df[c] for c in supp_cols], ignore_index=True)


def _counts_supp_in_window(df: pd.DataFrame, supp_cols: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    sub = df[(df["Date"] >= start) & (df["Date"] < end)]
    if sub.empty:
        return pd.Series([0] * (MAIN_MAX - MAIN_MIN + 1), index=range(MAIN_MIN, MAIN_MAX + 1))
    nums = _explode_supps(sub, supp_cols)
    return nums.value_counts().reindex(range(MAIN_MIN, MAIN_MAX + 1), fill_value=0).sort_index()


def _supp_score_map(df: pd.DataFrame, supp_cols: List[str], target_date: str) -> Dict[int, float]:
    if not supp_cols:
        return {}
    t = pd.to_datetime(target_date)
    if pd.isna(t):
        return {}
    train = df[df["Date"] < t]
    if train.empty:
        return {}

    recent_start = t - pd.Timedelta(days=LOOKBACK_DAYS)
    recent_counts = _counts_supp_in_window(train, supp_cols, recent_start, t)
    long_counts = _explode_supps(train, supp_cols).value_counts().reindex(
        range(MAIN_MIN, MAIN_MAX + 1), fill_value=0
    ).sort_index()

    seasonal_counts = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
    for year in range(t.year - SEASON_LOOKBACK_YEARS, t.year + 1):
        anchor = date(year, t.month, t.day)
        win_start, win_end = _season_window_dates(anchor, SEASON_WINDOW_DAYS)
        rows = train[(train["Date"] >= win_start) & (train["Date"] <= win_end)]
        if rows.empty:
            continue
        for _, row in rows.iterrows():
            for c in supp_cols:
                val = row[c]
                if pd.isna(val):
                    continue
                n = int(val)
                seasonal_counts[n] += 1
    s_recent = _normalize_series(recent_counts)
    s_long = _normalize_series(long_counts)
    s_season = _normalize_series(pd.Series(seasonal_counts).sort_index())

    scores: Dict[int, float] = {}
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        scores[n] = (
            SUPP_W_RECENT * float(s_recent.get(n, 0.0)) +
            SUPP_W_LONG * float(s_long.get(n, 0.0)) +
            SUPP_W_SEASON * float(s_season.get(n, 0.0))
        )
    return scores


def _supp_candidate_set(df: pd.DataFrame, supp_cols: List[str], target_date: str) -> List[int]:
    if not supp_cols:
        return []
    scores = _supp_score_map(df, supp_cols, target_date)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [n for n, _ in ranked[:SUPP_POOL_SIZE]]


def _rank_from_counts(counts: pd.Series) -> Dict[int, int]:
    tmp = counts.sort_values(ascending=False)
    ranks = {}
    current_rank = 1
    last_freq = None
    for n, freq in tmp.items():
        n = int(n)
        if last_freq is None:
            ranks[n] = current_rank
            last_freq = freq
        else:
            if freq < last_freq:
                current_rank += 1
                last_freq = freq
            ranks[n] = current_rank
    return ranks


def _anchor_for_year(target: date, year: int) -> date:
    try:
        return date(year, target.month, target.day)
    except ValueError:
        return date(year, 2, 28)


def _season_window_dates(anchor: date, window_days: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(anchor) - pd.Timedelta(days=window_days)
    end = pd.Timestamp(anchor) + pd.Timedelta(days=window_days + 1)
    return start, end


DECADE_IDS = [d[0] for d in DECADE_BANDS]


def _decade_of(n: int) -> int:
    for did, lo, hi in DECADE_BANDS:
        if lo <= n <= hi:
            return did
    return -1


def _decade_vector(nums: List[int]) -> Dict[int, int]:
    v = {did: 0 for did in DECADE_IDS}
    for n in nums:
        did = _decade_of(n)
        if did in v:
            v[did] += 1
    return v


def _normalize_decade_target(target: Dict[int, int]) -> Dict[int, int]:
    if target is None:
        return None
    if not isinstance(target, dict):
        raise ValueError("DECADE_TARGET_COUNTS must be a dict like {decade_id: count}.")
    for k, v in target.items():
        if k not in DECADE_IDS:
            raise ValueError(f"DECADE_TARGET_COUNTS has unknown decade id: {k}")
        if not isinstance(v, int) or v < 0:
            raise ValueError("DECADE_TARGET_COUNTS values must be non-negative integers.")
    total = sum(int(v) for v in target.values())
    if total != NUMBERS_PER_TICKET:
        raise ValueError("DECADE_TARGET_COUNTS must sum to NUMBERS_PER_TICKET.")
    return {d: int(target.get(d, 0)) for d in DECADE_IDS}


DECADE_TARGET_COUNTS = _normalize_decade_target(DECADE_TARGET_COUNTS)


def _count_consecutive_pairs(nums: List[int]) -> int:
    consec = 0
    nums_sorted = sorted(nums)
    for i in range(len(nums_sorted) - 1):
        if nums_sorted[i + 1] - nums_sorted[i] == 1:
            consec += 1
    return consec


def _normalize_series(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    mn = float(s.min())
    mx = float(s.max())
    if mx <= mn:
        return s * 0.0
    return (s - mn) / (mx - mn)


def _weighted_sample_no_replace(items: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    keyed = []
    for x, w in zip(items, weights):
        w = max(1e-9, float(w))
        u = rng.random()
        key = math.log(u) / w
        keyed.append((key, x))
    keyed.sort(reverse=True)
    return [x for _, x in keyed[:k]]


# ============================================================
# Cohesion helpers
# ============================================================

def _percentile(values: List[int], p: float) -> int:
    if not values:
        return 1
    v = sorted(values)
    idx = int(round((p / 100.0) * (len(v) - 1)))
    return int(v[max(0, min(idx, len(v) - 1))])

def _percentile(values: List[int], p: float) -> int:
    if not values:
        return 1
    v = sorted(values)
    idx = int(round((p / 100.0) * (len(v) - 1)))
    return int(v[max(0, min(idx, len(v) - 1))])


def _build_pair_counts(df: pd.DataFrame, main_cols: List[str]) -> Dict[Tuple[int, int], int]:
    from itertools import combinations
    pair_counts: Dict[Tuple[int, int], int] = {}
    for _, row in df.iterrows():
        nums = sorted(int(row[c]) for c in main_cols)
        for a, b in combinations(nums, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    return pair_counts


def _pair_density(nums: List[int], pair_counts: Dict[Tuple[int, int], int], pair_base: int) -> float:
    from itertools import combinations
    if not pair_counts or pair_base <= 0:
        return 0.0
    pairs = list(combinations(sorted(nums), 2))
    if not pairs:
        return 0.0
    s = sum(pair_counts.get((a, b), 0) for a, b in pairs)
    avg = float(s) / float(len(pairs))
    return min(avg / float(pair_base), 1.0)


def _rank_mass_score(ranks: List[int], top_rank: int) -> float:
    if not ranks:
        return 0.0
    outside = sum(1 for r in ranks if r > top_rank)
    return 1.0 - min(float(outside) / float(len(ranks)) / 0.5, 1.0)


def _score_spread(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return max(scores) - min(scores)


def _ticket_quality(
    nums: List[int],
    score_map: Dict[int, float],
    rank_map: Dict[int, int],
    pair_counts: Dict[Tuple[int, int], int],
    pair_base: int,
    top_rank: int,
    max_spread: float,
    max_gap: int,
    allow_outside: int,
    weights: Dict[str, float],
) -> float:
    ranks = [rank_map.get(n, 999) for n in nums]
    outside = sum(1 for r in ranks if r > top_rank)
    if outside > allow_outside:
        return -1.0

    scores = [score_map.get(n, 0.0) for n in nums]
    spread = _score_spread(scores)
    if spread > max_spread:
        return -1.0

    r_sorted = sorted(ranks)
    gaps = [r_sorted[i + 1] - r_sorted[i] for i in range(len(r_sorted) - 1)]
    max_found = max(gaps) if gaps else 0
    if max_found > max_gap:
        return -1.0

    spread_score = 1.0 - min(spread / max_spread, 1.0)
    pair_score = _pair_density(nums, pair_counts, pair_base)
    cont_score = _rank_continuity(ranks, max_gap)
    central_score = _centrality_score(ranks, top_rank)
    rank_mass = _rank_mass_score(ranks, top_rank)

    return (
        weights["spread"] * spread_score +
        weights["pair"] * pair_score +
        weights["rank_cont"] * cont_score +
        weights["central"] * central_score +
        weights["rank_mass"] * rank_mass
    )

def _rank_continuity(ranks: List[int], max_gap: int) -> float:
    if not ranks:
        return 0.0
    r_sorted = sorted(ranks)
    gaps = [r_sorted[i + 1] - r_sorted[i] for i in range(len(r_sorted) - 1)]
    max_found = max(gaps) if gaps else 0
    return 1.0 - min(float(max_found) / float(max_gap), 1.0)


def _centrality_score(ranks: List[int], top_rank: int) -> float:
    if not ranks or top_rank <= 1:
        return 0.0
    avg_rank = sum(ranks) / float(len(ranks))
    return 1.0 - min((avg_rank - 1.0) / float(top_rank - 1), 1.0)


def _pair_count_map(df: pd.DataFrame, main_cols: List[str]) -> Dict[Tuple[int, int], int]:
    if df.empty:
        return {}
    return _build_pair_counts(df, main_cols)


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _ticket_pairs(nums: List[int]) -> List[Tuple[int, int]]:
    from itertools import combinations
    return [_pair_key(a, b) for a, b in combinations(sorted(nums), 2)]


def _cohesion_score(
    nums: List[int],
    score_map: Dict[int, float],
    rank_map: Dict[int, int],
    pair_counts: Dict[Tuple[int, int], int],
    pair_base: int,
    weights: Dict[str, float],
) -> float:
    ranks = [rank_map.get(n, 999) for n in nums]
    outside = sum(1 for r in ranks if r > COHESION_TOP_RANK)
    rank_mass_score = 1.0 - min(float(outside) / float(len(nums)) / 0.5, 1.0)

    scores = [score_map.get(n, 0.0) for n in nums]
    spread = max(scores) - min(scores) if scores else 0.0
    spread_score = 1.0 - min(spread / COHESION_MAX_SCORE_SPREAD, 1.0)

    pair_score = _pair_density(nums, pair_counts, pair_base)
    cont_score = _rank_continuity(ranks, COHESION_MAX_RANK_GAP)
    central_score = _centrality_score(ranks, COHESION_TOP_RANK)

    return (
        weights["spread"] * spread_score +
        weights["pair"] * pair_score +
        weights["rank_cont"] * cont_score +
        weights["central"] * central_score +
        weights["rank_mass"] * rank_mass_score
    )


def _build_adaptive_pool(
    scored: List[CandidateScore],
    pair_counts: Dict[Tuple[int, int], int],
    pool_target: int,
    cold_pool_size: int,
) -> List[int]:
    score_map = {c.n: c.total_score for c in scored}
    top_pairs = [p for p, _ in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:200]]
    top_pair_set = set(top_pairs)

    pool: List[int] = []
    for c in scored[:6]:
        pool.append(c.n)
    pool = list(dict.fromkeys(pool))

    while len(pool) < pool_target:
        best = None
        best_score = -1e9
        for c in scored:
            n = c.n
            if n in pool:
                continue
            new_pairs = sum(1 for p in _ticket_pairs(pool + [n]) if p in top_pair_set)
            score = (2.0 * new_pairs) + (0.8 * score_map.get(n, 0.0))
            if score > best_score:
                best_score = score
                best = n
        if best is None:
            break
        pool.append(best)

    cold_pool = [c.n for c in scored if c.freq_recent <= 1][:cold_pool_size]
    pool = list(dict.fromkeys(pool + cold_pool))
    return pool


# ============================================================
# Scoring + ticketing
# ============================================================

def score_numbers(
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
    debug: bool,
    show_all: bool = False,
    score_config: Dict[str, float] = None,
    supp_cols: Optional[List[str]] = None,
) -> List[CandidateScore]:
    t = pd.Timestamp(target_date)
    if pd.isna(t):
        raise ValueError("TARGET_DATE must be parseable (YYYY-MM-DD)")

    train = df[df["Date"] < t].copy()
    if train.empty:
        raise ValueError("No historical draws before TARGET_DATE")

    recent_start = t - pd.Timedelta(days=LOOKBACK_DAYS)
    recent_counts = _counts_mains_in_window(train, main_cols, recent_start, t)
    long_counts = _explode_mains(train, main_cols).value_counts().reindex(range(MAIN_MIN, MAIN_MAX + 1), fill_value=0).sort_index()
    ranks = _rank_from_counts(recent_counts)

    # days since last seen (gap)
    last_seen = {n: None for n in range(MAIN_MIN, MAIN_MAX + 1)}
    for _, row in train.sort_values("Date").iterrows():
        d = row["Date"]
        for c in main_cols:
            n = int(row[c])
            last_seen[n] = d
    gap_days = {}
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        if last_seen[n] is None:
            gap_days[n] = 0
        else:
            gap_days[n] = int((t - last_seen[n]).days)
    s_gap = _normalize_series(pd.Series(gap_days).sort_index())

    # seasonal counts around same month/day across years (date-agnostic learning)
    target_dt = t.date()
    start_year = target_dt.year - SEASON_LOOKBACK_YEARS
    end_year = target_dt.year - 1
    seasonal_counts = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_dt, y)
        win_start, win_end = _season_window_dates(anchor, SEASON_WINDOW_DAYS)
        near = train[(train["Date"] >= win_start) & (train["Date"] < win_end)]
        if near.empty:
            continue
        for _, row in near.iterrows():
            mains = [int(row[c]) for c in main_cols]
            for n in mains:
                if MAIN_MIN <= n <= MAIN_MAX:
                    seasonal_counts[n] += 1

    s_recent = _normalize_series(recent_counts)
    s_long = _normalize_series(long_counts)
    s_season = _normalize_series(pd.Series(seasonal_counts).sort_index())

    # supplementary score components (learned from supplementary history)
    supp_cols = supp_cols or SUPP_COLS
    if supp_cols and SUPP_SCORE_W > 0.0:
        supp_recent_counts = _counts_supp_in_window(train, supp_cols, recent_start, t)
        supp_long_counts = _explode_supps(train, supp_cols).value_counts().reindex(
            range(MAIN_MIN, MAIN_MAX + 1), fill_value=0
        ).sort_index()
        supp_season_counts = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
        for y in range(start_year, end_year + 1):
            anchor = _anchor_for_year(target_dt, y)
            win_start, win_end = _season_window_dates(anchor, SEASON_WINDOW_DAYS)
            near = train[(train["Date"] >= win_start) & (train["Date"] < win_end)]
            if near.empty:
                continue
            for _, row in near.iterrows():
                for c in supp_cols:
                    val = row[c]
                    if pd.isna(val):
                        continue
                    n = int(val)
                    if MAIN_MIN <= n <= MAIN_MAX:
                        supp_season_counts[n] += 1
        s_supp_recent = _normalize_series(supp_recent_counts)
        s_supp_long = _normalize_series(supp_long_counts)
        s_supp_season = _normalize_series(pd.Series(supp_season_counts).sort_index())
    else:
        s_supp_recent = {}
        s_supp_long = {}
        s_supp_season = {}

    inv_rank = pd.Series({n: 1.0 / float(ranks.get(n, 999)) for n in range(MAIN_MIN, MAIN_MAX + 1)})
    s_rank = _normalize_series(inv_rank)

    w_recent = score_config.get("w_recent", W_RECENT) if score_config else W_RECENT
    w_long = score_config.get("w_long", W_LONG) if score_config else W_LONG
    w_season = score_config.get("w_season", W_SEASON) if score_config else W_SEASON
    w_rank = score_config.get("w_rank", W_RANK) if score_config else W_RANK
    w_gap = score_config.get("w_gap", W_GAP) if score_config else W_GAP
    gap_cap = score_config.get("gap_cap", GAP_CAP) if score_config else GAP_CAP
    cold_boost = score_config.get("cold_boost", COLD_BOOST) if score_config else COLD_BOOST
    w_decay = score_config.get("w_decay", 0.0) if score_config else 0.0
    decay_half_life = score_config.get("decay_half_life", 60) if score_config else 60
    use_decay = score_config.get("use_decay", False) if score_config else False
    w_pairnum = score_config.get("w_pairnum", 0.0) if score_config else 0.0
    use_pairnum = score_config.get("use_pairnum", False) if score_config else False

    s_decay = pd.Series({n: 0.0 for n in range(MAIN_MIN, MAIN_MAX + 1)})
    if use_decay:
        recent = train[(train["Date"] >= recent_start) & (train["Date"] < t)]
        for _, row in recent.iterrows():
            d = row["Date"]
            age = max(0, int((t - d).days))
            w = math.exp(-age / float(max(decay_half_life, 1)))
            for c in main_cols:
                n = int(row[c])
                s_decay[n] = s_decay.get(n, 0.0) + w
        s_decay = _normalize_series(pd.Series(s_decay).sort_index())

    s_pairnum = pd.Series({n: 0.0 for n in range(MAIN_MIN, MAIN_MAX + 1)})
    if use_pairnum:
        pair_counts = _build_pair_counts(train, main_cols)
        for (a, b), cnt in pair_counts.items():
            s_pairnum[a] = s_pairnum.get(a, 0.0) + cnt
            s_pairnum[b] = s_pairnum.get(b, 0.0) + cnt
        s_pairnum = _normalize_series(pd.Series(s_pairnum).sort_index())

    scored: List[CandidateScore] = []
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        freq_recent = int(recent_counts.get(n, 0))
        freq_long = int(long_counts.get(n, 0))
        freq_season = int(seasonal_counts.get(n, 0))
        rank_recent = int(ranks.get(n, 999))

        total = (
            w_recent * float(s_recent.get(n, 0.0)) +
            w_long * float(s_long.get(n, 0.0)) +
            w_season * float(s_season.get(n, 0.0)) +
            w_rank * float(s_rank.get(n, 0.0))
        )
        total += min(gap_cap, w_gap * float(s_gap.get(n, 0.0)))
        if supp_cols and SUPP_SCORE_W > 0.0:
            supp_score = (
                SUPP_W_RECENT * float(s_supp_recent.get(n, 0.0)) +
                SUPP_W_LONG * float(s_supp_long.get(n, 0.0)) +
                SUPP_W_SEASON * float(s_supp_season.get(n, 0.0))
            )
            total += SUPP_SCORE_W * supp_score
        if use_decay and w_decay > 0.0:
            total += w_decay * float(s_decay.get(n, 0.0))
        if use_pairnum and w_pairnum > 0.0:
            total += w_pairnum * float(s_pairnum.get(n, 0.0))
        if freq_recent <= 1:
            total += cold_boost

        scored.append(CandidateScore(n=n, total_score=round(total, 6),
                                     freq_recent=freq_recent, freq_long=freq_long,
                                     freq_season=freq_season, rank_recent=rank_recent,
                                     gap_days=gap_days.get(n, 0)))

    scored.sort(key=lambda x: x.total_score, reverse=True)

    if debug:
        if show_all:
            print("\n=== ALL SCORED NUMBERS ===")
            for c in scored:
                print(c)
        else:
            print("\n=== TOP 20 SCORED NUMBERS ===")
            for c in scored[:20]:
                print(c)

    return scored


def _history_distributions(df: pd.DataFrame, main_cols: List[str], target_date: str):
    sums = []
    odd_counts = []
    low_counts = []
    consec_counts = []
    decade_vectors = []

    for _, row in df.iterrows():
        nums = [int(row[c]) for c in main_cols]
        sums.append(sum(nums))
        odd_counts.append(sum(1 for n in nums if n % 2 == 1))
        low_counts.append(sum(1 for n in nums if n <= LOW_RANGE_MAX))
        consec_counts.append(_count_consecutive_pairs(nums))
        decade_vectors.append(_decade_vector(nums))

    sum_series = pd.Series(sums)
    sum_lo = float(sum_series.quantile(SUM_BAND_QUANTILES[0]))
    sum_hi = float(sum_series.quantile(SUM_BAND_QUANTILES[1]))

    # average decade vector (global)
    decade_mean = {d: 0.0 for d in DECADE_IDS}
    for v in decade_vectors:
        for d in DECADE_IDS:
            decade_mean[d] += v[d]
    if decade_vectors:
        for d in DECADE_IDS:
            decade_mean[d] /= float(len(decade_vectors))

    # seasonal decade mean around target month/day across years
    t = pd.Timestamp(target_date)
    seasonal_vectors = []
    if not pd.isna(t):
        target_dt = t.date()
        start_year = target_dt.year - SEASON_LOOKBACK_YEARS
        end_year = target_dt.year - 1
        for y in range(start_year, end_year + 1):
            anchor = _anchor_for_year(target_dt, y)
            win_start, win_end = _season_window_dates(anchor, SEASON_WINDOW_DAYS)
            near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
            if near.empty:
                continue
            for _, row in near.iterrows():
                nums = [int(row[c]) for c in main_cols]
                seasonal_vectors.append(_decade_vector(nums))

    decade_season_mean = {d: 0.0 for d in DECADE_IDS}
    if seasonal_vectors:
        for v in seasonal_vectors:
            for d in DECADE_IDS:
                decade_season_mean[d] += v[d]
        for d in DECADE_IDS:
            decade_season_mean[d] /= float(len(seasonal_vectors))

    return {
        "sum_lo": int(round(sum_lo)),
        "sum_hi": int(round(sum_hi)),
        "odd_counts": pd.Series(odd_counts).value_counts().to_dict(),
        "low_counts": pd.Series(low_counts).value_counts().to_dict(),
        "consec_counts": pd.Series(consec_counts).value_counts().to_dict(),
        "decade_mean": decade_mean,
        "decade_season_mean": decade_season_mean,
        "sum_series": sums,
    }


def _learn_winner_profile(
    df: pd.DataFrame,
    main_cols: List[str],
    learn_dates: List[pd.Timestamp],
) -> Optional[Dict[str, object]]:
    ranks = []
    gaps = []
    scores = []
    for d in learn_dates:
        bt_date = d.strftime("%Y-%m-%d")
        scored = score_numbers(df, main_cols, bt_date, debug=False)
        score_map = {c.n: c for c in scored}
        row = df[df["Date"] == d].iloc[0]
        winners = [int(row[c]) for c in main_cols]
        for n in winners:
            c = score_map.get(n)
            if c is None:
                continue
            ranks.append(int(c.rank_recent))
            gaps.append(int(c.gap_days))
            scores.append(float(c.total_score))
    if not ranks:
        return None
    s_rank = pd.Series(ranks)
    s_gap = pd.Series(gaps) if gaps else pd.Series([0])
    s_score = pd.Series(scores) if scores else pd.Series([0.0])
    return {
        "rank_soft_max": int(round(s_rank.quantile(0.75))),
        "rank_hard_max": int(round(s_rank.quantile(0.90))),
        "gap_hard_max": int(round(s_gap.quantile(0.90))),
        "score_min": float(s_score.quantile(0.40)),
    }


def _ticket_passes_profile(
    ticket: List[int],
    score_map: Dict[int, CandidateScore],
    profile: Dict[str, object],
) -> bool:
    if not profile:
        return True
    rank_soft = int(profile.get("rank_soft_max", 7))
    rank_hard = int(profile.get("rank_hard_max", 10))
    gap_hard = int(profile.get("gap_hard_max", 150))
    score_min = float(profile.get("score_min", 0.0))

    rank_soft_ct = 0
    outlier_ct = 0
    for n in ticket:
        c = score_map.get(n)
        if c is None:
            return False
        if c.rank_recent <= rank_soft:
            rank_soft_ct += 1
        is_outlier = (c.rank_recent > rank_hard) or (c.gap_days > gap_hard) or (c.total_score < score_min)
        if is_outlier:
            outlier_ct += 1
    if rank_soft_ct < 4 or rank_soft_ct > 5:
        return False
    if outlier_ct > 1:
        return False
    return True


def _ticket_passes_profile_relaxed(
    ticket: List[int],
    score_map: Dict[int, CandidateScore],
    profile: Dict[str, object],
) -> bool:
    if not profile:
        return True
    rank_soft = int(profile.get("rank_soft_max", 7))
    rank_hard = int(profile.get("rank_hard_max", 10))
    gap_hard = int(profile.get("gap_hard_max", 150))
    score_min = float(profile.get("score_min", 0.0))

    rank_soft_ct = 0
    outlier_ct = 0
    for n in ticket:
        c = score_map.get(n)
        if c is None:
            return False
        if c.rank_recent <= rank_soft:
            rank_soft_ct += 1
        is_outlier = (c.rank_recent > rank_hard) or (c.gap_days > gap_hard) or (c.total_score < score_min)
        if is_outlier:
            outlier_ct += 1
    if rank_soft_ct < 3:
        return False
    if outlier_ct > 2:
        return False
    return True


def _dynamic_band(counts: Dict[int, int], low_pct: float, high_pct: float) -> Tuple[int, int]:
    if not counts:
        return (0, 0)
    items = []
    for k, v in counts.items():
        items.extend([int(k)] * int(v))
    if not items:
        return (0, 0)
    items.sort()
    lo_idx = int(round(low_pct * (len(items) - 1)))
    hi_idx = int(round(high_pct * (len(items) - 1)))
    lo = items[max(0, min(lo_idx, len(items) - 1))]
    hi = items[max(0, min(hi_idx, len(items) - 1))]
    return (int(lo), int(hi))


def _ticket_penalty(nums: List[int], dist: Dict[str, object], penalty_config: Dict[str, object] = None) -> float:
    penalty = 0.0

    # odd count band
    odd_ct = sum(1 for n in nums if n % 2 == 1)
    odd_band = ODD_BAND
    if penalty_config and penalty_config.get("dynamic_bands"):
        odd_band = _dynamic_band(dist.get("odd_counts", {}), 0.20, 0.80)
    if odd_ct < odd_band[0] or odd_ct > odd_band[1]:
        penalty += 0.8 * abs(odd_ct - max(min(odd_ct, odd_band[1]), odd_band[0]))

    # low count band
    low_ct = sum(1 for n in nums if n <= LOW_RANGE_MAX)
    low_band = LOW_BAND
    if penalty_config and penalty_config.get("dynamic_bands"):
        low_band = _dynamic_band(dist.get("low_counts", {}), 0.20, 0.80)
    if low_ct < low_band[0] or low_ct > low_band[1]:
        penalty += 0.8 * abs(low_ct - max(min(low_ct, low_band[1]), low_band[0]))

    # sum band
    s = sum(nums)
    sum_lo = dist["sum_lo"]
    sum_hi = dist["sum_hi"]
    if penalty_config and penalty_config.get("sum_quantiles"):
        qlo, qhi = penalty_config["sum_quantiles"]
        sum_lo = int(round(pd.Series(dist.get("sum_series", [])).quantile(qlo))) if dist.get("sum_series") else sum_lo
        sum_hi = int(round(pd.Series(dist.get("sum_series", [])).quantile(qhi))) if dist.get("sum_series") else sum_hi
    if s < sum_lo:
        penalty += 0.6 * (sum_lo - s) / 10.0
    elif s > sum_hi:
        penalty += 0.6 * (s - sum_hi) / 10.0

    # consecutive pairs
    consec = _count_consecutive_pairs(nums)
    consec_max = CONSECUTIVE_MAX
    if penalty_config and penalty_config.get("consec_max") is not None:
        consec_max = int(penalty_config["consec_max"])
    if consec > consec_max:
        penalty += 0.7 * (consec - consec_max)

    # decade balance (blend seasonal + global)
    vec = _decade_vector(nums)
    for d in DECADE_IDS:
        season_mean = dist.get("decade_season_mean", {}).get(d, 0.0)
        global_mean = dist["decade_mean"].get(d, 0.0)
        target_mean = (SEASON_DECADE_WEIGHT * season_mean) + ((1.0 - SEASON_DECADE_WEIGHT) * global_mean)
        penalty += 0.4 * abs(vec[d] - target_mean)

    return penalty


def _apply_supp_soft_swap(
    pick: List[int],
    supp_candidates: List[int],
    supp_scores: Dict[int, float],
    score_map: Dict[int, float],
    pool: List[int],
    global_use: Dict[int, int],
    global_max: int,
) -> List[int]:
    if not SUPP_SOFT_SWAP or not supp_candidates:
        return pick
    pick_set = set(pick)
    if pick_set.intersection(supp_candidates):
        return pick

    pool_set = set(pool)
    candidates = [
        n for n in supp_candidates[:SUPP_SWAP_TOP]
        if n in pool_set and n not in pick_set and global_use.get(n, 0) < global_max
    ]
    if not candidates:
        return pick

    candidates.sort(key=lambda n: (supp_scores.get(n, 0.0), score_map.get(n, 0.0)), reverse=True)
    removable = sorted(pick, key=lambda n: score_map.get(n, 0.0))
    for cand in candidates:
        cand_score = score_map.get(cand, 0.0)
        for out in removable:
            out_score = score_map.get(out, 0.0)
            if cand_score + SUPP_SWAP_SCORE_DELTA < out_score:
                continue
            new_pick = [n for n in pick if n != out] + [cand]
            return sorted(new_pick)
    return pick


def _apply_supp_soft_swaps(
    tickets: List[List[int]],
    supp_candidates: List[int],
    supp_scores: Dict[int, float],
    score_map: Dict[int, float],
    pool: List[int],
    global_use: Dict[int, int],
    global_max: int,
    overlap_cap: int,
) -> List[List[int]]:
    if not SUPP_SOFT_SWAP or not supp_candidates:
        return tickets
    if not tickets:
        return tickets

    pool_set = set(pool)
    candidates = [n for n in supp_candidates[:SUPP_SWAP_TOP] if n in pool_set]
    if not candidates:
        return tickets

    candidates.sort(key=lambda n: (supp_scores.get(n, 0.0), score_map.get(n, 0.0)), reverse=True)
    ticket_scores = []
    for i, t in enumerate(tickets):
        avg_score = sum(score_map.get(n, 0.0) for n in t) / float(len(t))
        ticket_scores.append((avg_score, i))
    if SUPP_FOCUS_MODE == "HIGH":
        ticket_scores.sort(reverse=True)
    else:
        ticket_scores.sort()

    swaps_done = 0
    for _, idx in ticket_scores:
        if swaps_done >= SUPP_SWAP_TICKETS:
            break
        pick = tickets[idx]
        pick_set = set(pick)
        if pick_set.intersection(candidates):
            continue

        removable = sorted(pick, key=lambda n: score_map.get(n, 0.0))
        for cand in candidates:
            if cand in pick_set or global_use.get(cand, 0) >= global_max:
                continue
            cand_score = score_map.get(cand, 0.0)
            for out in removable:
                out_score = score_map.get(out, 0.0)
                if cand_score + SUPP_SWAP_SCORE_DELTA < out_score:
                    continue
                new_pick = sorted([n for n in pick if n != out] + [cand])
                s_new = set(new_pick)
                ok = True
                for j, other in enumerate(tickets):
                    if j == idx:
                        continue
                    if len(s_new.intersection(other)) > overlap_cap:
                        ok = False
                        break
                if not ok:
                    continue
                global_use[out] -= 1
                global_use[cand] = global_use.get(cand, 0) + 1
                tickets[idx] = new_pick
                swaps_done += 1
                break
            if swaps_done >= SUPP_SWAP_TICKETS:
                break
    return tickets


def _apply_supp_focus_tickets(
    tickets: List[List[int]],
    scored: List[CandidateScore],
    supp_scores: Dict[int, float],
    score_map: Dict[int, float],
    global_use: Dict[int, int],
    global_max: int,
    overlap_cap: int,
) -> List[List[int]]:
    if SUPP_FOCUS_TICKETS <= 0 or SUPP_FOCUS_COUNT <= 0:
        return tickets
    if not tickets or not supp_scores:
        return tickets

    supp_ranked = sorted(supp_scores.items(), key=lambda kv: kv[1], reverse=True)
    supp_candidates = [n for n, _ in supp_ranked[:SUPP_FOCUS_TOP]]
    if not supp_candidates:
        return tickets

    main_ranked = [c.n for c in scored]
    ticket_scores = []
    for i, t in enumerate(tickets):
        avg_score = sum(score_map.get(n, 0.0) for n in t) / float(len(t))
        ticket_scores.append((avg_score, i))
    ticket_scores.sort()

    replaced = 0
    for _, idx in ticket_scores:
        if replaced >= SUPP_FOCUS_TICKETS:
            break

        pick = tickets[idx]
        pick_set = set(pick)
        supp_pick = []
        for n in supp_candidates:
            if n in pick_set:
                continue
            if global_use.get(n, 0) >= global_max:
                continue
            supp_pick.append(n)
            if len(supp_pick) >= SUPP_FOCUS_COUNT:
                break
        if len(supp_pick) < SUPP_FOCUS_COUNT:
            continue

        remaining = []
        for n in main_ranked:
            if n in pick_set or n in supp_pick:
                continue
            if global_use.get(n, 0) >= global_max:
                continue
            remaining.append(n)
            if len(remaining) >= NUMBERS_PER_TICKET - len(supp_pick):
                break
        if len(remaining) < NUMBERS_PER_TICKET - len(supp_pick):
            continue

        new_pick = sorted(supp_pick + remaining)
        s_new = set(new_pick)
        ok = True
        for j, other in enumerate(tickets):
            if j == idx:
                continue
            if len(s_new.intersection(other)) > overlap_cap:
                ok = False
                break
        if not ok:
            continue

        for n in pick:
            global_use[n] -= 1
        for n in new_pick:
            global_use[n] = global_use.get(n, 0) + 1
        tickets[idx] = new_pick
        replaced += 1

    return tickets


def generate_tickets(
    scored: List[CandidateScore],
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
    use_weights: bool = True,
    seed_hot_overdue: bool = True,
    penalty_scale: float = None,
    pool_override: List[int] = None,
    force_coverage: bool = False,
    fixed_seed: List[int] = None,
    overlap_cap_override: int = None,
    global_max_override: int = None,
    cohesion_config: Dict[str, object] = None,
    pool_config: Dict[str, object] = None,
    penalty_config: Dict[str, object] = None,
    constraint_fn: Optional[Callable[[List[int]], bool]] = None,
    supp_candidates: Optional[List[int]] = None,
    supp_min_hit: int = 0,
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    train = df[df["Date"] < pd.Timestamp(target_date)]
    dist = _history_distributions(train, main_cols, target_date)

    pool_cfg = pool_config or {}
    pool_size = int(pool_cfg.get("pool_size", POOL_SIZE))
    mid_pool_size = int(pool_cfg.get("mid_pool_size", MID_POOL_SIZE))
    cold_pool_size = int(pool_cfg.get("cold_pool_size", COLD_POOL_SIZE))
    hot_pool_size = int(pool_cfg.get("hot_pool_size", HOT_POOL_SIZE))
    overdue_pool_size = int(pool_cfg.get("overdue_pool_size", OVERDUE_POOL_SIZE))
    season_pool_size = int(pool_cfg.get("season_pool_size", SEASON_POOL_SIZE))
    cold_force_count = int(pool_cfg.get("cold_force_count", COLD_FORCE_COUNT))

    # build pool
    if pool_override is not None:
        pool = list(dict.fromkeys(pool_override))
    else:
        top_pool = [c.n for c in scored[:pool_size]]
        mid_pool = [c.n for c in scored[pool_size:pool_size + mid_pool_size]]
        cold_pool = [c.n for c in scored if c.freq_recent <= 1][:cold_pool_size]
        pool = list(dict.fromkeys(top_pool + mid_pool + cold_pool))

    # weights from scores
    score_map = {c.n: c.total_score for c in scored}
    rank_map = {c.n: c.rank_recent for c in scored}
    supp_scores: Dict[int, float] = {}
    if (SUPP_SOFT_SWAP or SUPP_FOCUS_TICKETS > 0) and SUPP_COLS:
        if supp_candidates or SUPP_FOCUS_TICKETS > 0:
            supp_scores = _supp_score_map(df, SUPP_COLS, target_date)
    min_score = min(score_map.values()) if score_map else 0.0
    if use_weights:
        weights = [(score_map.get(n, 0.0) - min_score) + 0.25 for n in pool]
    else:
        weights = [1.0 for _ in pool]

    # hot + overdue + seasonal + cold pools (for enforced mix)
    hot_pool = [c.n for c in scored[:hot_pool_size]]
    season_sorted = sorted(scored, key=lambda x: (x.freq_season, x.total_score), reverse=True)
    season_pool = [c.n for c in season_sorted[:season_pool_size]]
    overdue_sorted = sorted(scored, key=lambda x: (x.gap_days, x.total_score), reverse=True)
    overdue_pool = [c.n for c in overdue_sorted[:overdue_pool_size]]
    cold_sorted = sorted(scored, key=lambda x: (x.freq_recent, x.total_score))
    cold_pool_force = [c.n for c in cold_sorted[:cold_pool_size]]

    pair_counts = _build_pair_counts(train, main_cols)
    pair_base = _percentile(list(pair_counts.values()), COHESION_PAIR_BASE_PCTL) if pair_counts else 0

    if cohesion_config is None:
        cohesion_config = {
            "enabled": COHESION_ENABLED,
            "accept_floor": COHESION_ACCEPT_FLOOR,
            "accept_span": COHESION_ACCEPT_SPAN,
            "min_score": None,
            "weights": {
                "spread": COHESION_W_SPREAD,
                "pair": COHESION_W_PAIR,
                "rank_cont": COHESION_W_RANK_CONT,
                "central": COHESION_W_CENTRAL,
                "rank_mass": 0.05,
            },
        }

    tickets: List[List[int]] = []
    global_use = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
    overlap_cap = OVERLAP_CAP if overlap_cap_override is None else int(overlap_cap_override)
    global_max = GLOBAL_MAX_USES if global_max_override is None else int(global_max_override)

    attempts = 0
    while len(tickets) < NUM_TICKETS and attempts < MAX_ATTEMPTS:
        attempts += 1

        if fixed_seed:
            seed = list(dict.fromkeys(fixed_seed))[:NUMBERS_PER_TICKET]
            remaining_k = NUMBERS_PER_TICKET - len(seed)
            remaining_items = [n for n in pool if n not in seed]
            remaining_weights = [w for n, w in zip(pool, weights) if n not in seed]
            if remaining_k > 0 and remaining_items:
                rest = _weighted_sample_no_replace(remaining_items, remaining_weights, remaining_k, rng)
            else:
                rest = []
            pick = sorted(seed + rest)
        elif force_coverage:
            seed = []
            for pool_pick in (hot_pool, season_pool, overdue_pool, cold_pool_force):
                if not pool_pick:
                    continue
                pick_n = rng.choice(pool_pick)
                if pick_n not in seed:
                    seed.append(pick_n)
            # force additional cold numbers
            if cold_pool_force and cold_force_count > 1:
                extras = [n for n in cold_pool_force if n not in seed]
                rng.shuffle(extras)
                for n in extras[: max(0, cold_force_count - 1)]:
                    if n not in seed:
                        seed.append(n)
            # fallback: if we overfilled, trim to size while keeping at least one cold
            if len(seed) > NUMBERS_PER_TICKET:
                cold_keep = None
                for n in seed:
                    if n in cold_pool_force:
                        cold_keep = n
                        break
                if cold_keep is not None:
                    seed = [cold_keep] + [n for n in seed if n != cold_keep]
                seed = seed[:NUMBERS_PER_TICKET]
            remaining_k = NUMBERS_PER_TICKET - len(seed)
            remaining_items = [n for n in pool if n not in seed]
            remaining_weights = [w for n, w in zip(pool, weights) if n not in seed]
            if remaining_k > 0 and remaining_items:
                rest = _weighted_sample_no_replace(remaining_items, remaining_weights, remaining_k, rng)
            else:
                rest = []
            pick = sorted(seed + rest)
        elif seed_hot_overdue:
            # enforce mix: 1 hot + 1 overdue (if possible), rest from weighted pool
            hot_pick = rng.choice(hot_pool) if hot_pool else None
            overdue_pick = rng.choice(overdue_pool) if overdue_pool else None

            seed = []
            if hot_pick is not None:
                seed.append(hot_pick)
            if overdue_pick is not None and overdue_pick not in seed:
                seed.append(overdue_pick)

            remaining_k = NUMBERS_PER_TICKET - len(seed)
            remaining_items = [n for n in pool if n not in seed]
            remaining_weights = [w for n, w in zip(pool, weights) if n not in seed]
            if remaining_k > 0 and remaining_items:
                rest = _weighted_sample_no_replace(remaining_items, remaining_weights, remaining_k, rng)
            else:
                rest = []
            pick = sorted(seed + rest)
        else:
            pick = sorted(_weighted_sample_no_replace(pool, weights, NUMBERS_PER_TICKET, rng))
        if pick in tickets:
            continue
        if supp_candidates and supp_min_hit > 0:
            if len(set(pick).intersection(supp_candidates)) < supp_min_hit:
                continue

        if any(global_use[n] >= global_max for n in pick):
            continue

        # overlap cap
        ok = True
        s_pick = set(pick)
        for t in tickets:
            if len(s_pick.intersection(t)) > overlap_cap:
                ok = False
                break
        if not ok:
            continue

        if constraint_fn and not constraint_fn(pick):
            continue

        penalty = _ticket_penalty(pick, dist, penalty_config=penalty_config)
        if DECADE_TARGET_COUNTS is not None:
            vec = _decade_vector(pick)
            if any(vec.get(d, 0) != DECADE_TARGET_COUNTS.get(d, 0) for d in DECADE_IDS):
                continue
        scale = PENALTY_SCALE if penalty_scale is None else float(penalty_scale)
        accept_prob = math.exp(-penalty * scale)
        if cohesion_config.get("enabled"):
            cohesion = _cohesion_score(
                pick, score_map, rank_map, pair_counts, pair_base, cohesion_config["weights"]
            )
            min_score = cohesion_config.get("min_score")
            if min_score is not None and cohesion < float(min_score):
                continue
            accept_prob *= (cohesion_config["accept_floor"] + cohesion_config["accept_span"] * cohesion)
            if accept_prob > 1.0:
                accept_prob = 1.0
        if rng.random() > accept_prob:
            continue

        tickets.append(pick)
        for n in pick:
            global_use[n] += 1

    print(f"Generated {len(tickets)}/{NUM_TICKETS} tickets in {attempts} attempts")
    # if len(tickets) < NUM_TICKETS:
    #     raise RuntimeError("Could not generate enough tickets. Increase POOL_SIZE or relax constraints.")

    return tickets


def show_ticket_hits(real_draw: List[int], tickets: List[List[int]], supp_draw: List[int]):
    if not real_draw:
        return
    rd = sorted(real_draw)
    rd_set = set(rd)
    supp_set = set(supp_draw)
    print("\n=== REAL DRAW HIT SUMMARY ===")
    print(f"REAL_DRAW: {rd}")

    hit_counts: Dict[int, int] = {}
    any_ge3 = False
    best_near = (-1, None, [])
    for i, t in enumerate(tickets, 1):
        hits = sorted(set(t).intersection(rd_set))
        hit_n = len(hits)
        hit_counts[hit_n] = hit_counts.get(hit_n, 0) + 1
        near = []
        for n in t:
            if n in rd_set:
                continue
            if any(abs(n - r) == 1 for r in rd_set):
                near.append(n)
        if len(near) > best_near[0]:
            best_near = (len(near), i, sorted(set(near)))
        if hit_n >= 3:
            any_ge3 = True
            supp_hits = sorted(set(t).intersection(supp_set)) if supp_set else []
            print(
                f"Ticket #{i:02d}: hits={hit_n} nums={hits} "
                f"supp_hit={len(supp_hits)} supp_nums={supp_hits} "
                f"near_miss={len(near)} near_nums={sorted(set(near))}"
            )

    if not any_ge3:
        print("No tickets with 3+ hits.")
    if best_near[0] >= 1:
        print(f"Best near-miss: Ticket #{best_near[1]:02d} near_miss={best_near[0]} near_nums={best_near[2]}")


def _hit_summary(real_draw: List[int], tickets: List[List[int]]) -> Dict[str, int]:
    rd_set = set(real_draw)
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_hits = 0
    for t in tickets:
        hit_n = len(set(t).intersection(rd_set))
        counts[hit_n] = counts.get(hit_n, 0) + 1
        total_hits += hit_n
    return {
        "ge3": sum(counts[h] for h in counts if h >= 3),
        "ge4": sum(counts[h] for h in counts if h >= 4),
        "ge5": sum(counts[h] for h in counts if h >= 5),
        "ge2": sum(counts[h] for h in counts if h >= 2),
        "total_hits": total_hits,
    }


def _evaluate_strategy(
    df: pd.DataFrame,
    main_cols: List[str],
    bt_dates: List[pd.Timestamp],
    strategy_name: str,
    cohesion_config: Dict[str, object],
    debug: bool = False,
) -> Dict[str, object]:
    weeks_with_5 = 0
    weeks_with_4 = 0
    ge3_total = 0
    ge4_total = 0
    ge5_total = 0
    for d in bt_dates:
        bt_date = d.strftime("%Y-%m-%d")
        row = df[df["Date"] == d].iloc[0]
        bt_draw = [int(row[c]) for c in main_cols]
        bt_scored = score_numbers(df, main_cols, bt_date, debug)
        bt_tickets = generate_tickets(
            bt_scored,
            df,
            main_cols,
            bt_date,
            use_weights=True,
            seed_hot_overdue=False,
            force_coverage=FORCE_COVERAGE,
            cohesion_config=cohesion_config,
        )
        summary = _hit_summary(bt_draw, bt_tickets)
        ge3_total += summary.get("ge3", 0)
        ge4_total += summary.get("ge4", 0)
        ge5_total += summary.get("ge5", 0)
        if summary.get("ge4", 0) > 0:
            weeks_with_4 += 1
        if summary.get("ge5", 0) > 0:
            weeks_with_5 += 1
    return {
        "name": strategy_name,
        "hits5": weeks_with_5,
        "ge3_total": ge3_total,
        "ge4_total": ge4_total,
        "ge5_total": ge5_total,
        "weeks_with_4": weeks_with_4,
    }




def _tune_configs(
    df: pd.DataFrame,
    main_cols: List[str],
    bt_dates: List[pd.Timestamp],
    base_cohesion: Dict[str, object],
) -> Dict[str, object]:
    preset_name = os.environ.get("PRESET", "").strip().upper()
    ga_preset_path = os.environ.get("GA_PRESET_PATH", "ga_best_config.json")
    compare_only = os.environ.get("COMPARE_ONLY", "").strip() == "1"
    p36 = {
        "pool_size": 36, "mid_pool_size": 14, "cold_pool_size": 12,
        "hot_pool_size": 10, "overdue_pool_size": 10, "season_pool_size": 10,
        "cold_force_count": 2,
    }
    p40 = {
        "pool_size": 40, "mid_pool_size": 16, "cold_pool_size": 12,
        "hot_pool_size": 12, "overdue_pool_size": 12, "season_pool_size": 12,
        "cold_force_count": 2,
    }
    p44 = {
        "pool_size": 44, "mid_pool_size": 18, "cold_pool_size": 12,
        "hot_pool_size": 12, "overdue_pool_size": 12, "season_pool_size": 12,
        "cold_force_count": 2,
    }
    coh_soft = {
        "enabled": True,
        "accept_floor": 0.70,
        "accept_span": 0.65,
        "min_score": None,
        "weights": {"spread": 0.30, "pair": 0.35, "rank_cont": 0.20, "central": 0.10, "rank_mass": 0.05},
    }
    coh_pair = {
        "enabled": True,
        "accept_floor": 0.72,
        "accept_span": 0.60,
        "min_score": 0.52,
        "weights": {"spread": 0.22, "pair": 0.45, "rank_cont": 0.18, "central": 0.10, "rank_mass": 0.05},
    }
    coh_rank = {
        "enabled": True,
        "accept_floor": 0.68,
        "accept_span": 0.65,
        "min_score": None,
        "weights": {"spread": 0.22, "pair": 0.28, "rank_cont": 0.35, "central": 0.10, "rank_mass": 0.05},
    }
    coh_none = {
        "enabled": False,
        "accept_floor": 1.0,
        "accept_span": 0.0,
        "min_score": None,
        "weights": {"spread": 0.0, "pair": 0.0, "rank_cont": 0.0, "central": 0.0, "rank_mass": 0.0},
    }

    base_score = {"w_recent": 0.55, "w_long": 0.20, "w_season": 0.15, "w_rank": 0.10, "w_gap": 0.25, "gap_cap": 0.30, "cold_boost": 0.25}
    decay_score = {"w_recent": 0.40, "w_long": 0.15, "w_season": 0.15, "w_rank": 0.10, "w_gap": 0.30, "gap_cap": 0.35, "cold_boost": 0.20, "use_decay": True, "w_decay": 0.25, "decay_half_life": 60}
    pair_score = {"w_recent": 0.50, "w_long": 0.15, "w_season": 0.15, "w_rank": 0.10, "w_gap": 0.25, "gap_cap": 0.30, "cold_boost": 0.20, "use_pairnum": True, "w_pairnum": 0.20}
    decay_pair_score = {"w_recent": 0.35, "w_long": 0.15, "w_season": 0.15, "w_rank": 0.10, "w_gap": 0.25, "gap_cap": 0.30, "cold_boost": 0.20, "use_decay": True, "w_decay": 0.20, "decay_half_life": 60, "use_pairnum": True, "w_pairnum": 0.15}

    def _load_ga_preset() -> Dict[str, object]:
        if not os.path.exists(ga_preset_path):
            return None
        try:
            with open(ga_preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        data.setdefault("name", "GA_BEST")
        return data

    def _save_ga_preset(cfg: Dict[str, object]) -> None:
        try:
            with open(ga_preset_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, sort_keys=True)
        except OSError:
            pass

    configs = []
    base_cfg = {
        "name": "BASE",
        "pool": p36,
        "penalty_scale": 0.50,
        "cohesion": coh_soft,
        "score_config": base_score,
        "penalty_config": {"dynamic_bands": False},
        "generator": "standard",
    }

    # One-by-one variants (single changes from BASE)
    configs.extend([
        base_cfg,
        {**base_cfg, "name": "POOL_P40", "pool": p40},
        {**base_cfg, "name": "POOL_P44", "pool": p44},
        {**base_cfg, "name": "PENALTY_035", "penalty_scale": 0.35},
        {**base_cfg, "name": "PENALTY_065", "penalty_scale": 0.65},
        {**base_cfg, "name": "COH_PAIR", "cohesion": coh_pair},
        {**base_cfg, "name": "COH_RANK", "cohesion": coh_rank},
        {**base_cfg, "name": "COH_NONE", "cohesion": coh_none},
        {**base_cfg, "name": "SCORE_DECAY", "score_config": decay_score},
        {**base_cfg, "name": "SCORE_PAIR", "score_config": pair_score},
        {**base_cfg, "name": "SCORE_DECAY_PAIR", "score_config": decay_pair_score},
        {**base_cfg, "name": "DYN_BANDS", "penalty_config": {"dynamic_bands": True}},
        {**base_cfg, "name": "SUM_BAND_TIGHT", "penalty_config": {"dynamic_bands": False, "sum_quantiles": (0.25, 0.75)}},
        {**base_cfg, "name": "SUM_BAND_WIDE", "penalty_config": {"dynamic_bands": False, "sum_quantiles": (0.15, 0.85)}},
        {**base_cfg, "name": "CONSEC_1", "penalty_config": {"dynamic_bands": False, "consec_max": 1}},
        {**base_cfg, "name": "CONSEC_3", "penalty_config": {"dynamic_bands": False, "consec_max": 3}},
        {**base_cfg, "name": "PORTFOLIO", "generator": "portfolio"},
        {**base_cfg, "name": "PAIR_ANCHOR", "generator": "pair_anchored"},
        {**base_cfg, "name": "PAIR_CLUSTER", "generator": "pair_cluster"},
        {**base_cfg, "name": "GRAPH_COVER", "generator": "graph_cover"},
        {**base_cfg, "name": "ADAPT_POOL", "generator": "adaptive_pool"},
        {**base_cfg, "name": "ADAPT_POOL_FORCE", "generator": "adaptive_pool", "force_coverage": True},
    ])

    if preset_name == "BASE":
        print("Preset selected: BASE")
        return base_cfg
    if preset_name == "GA_BEST":
        cached = _load_ga_preset()
        if cached:
            print(f"Preset selected: GA_BEST (loaded from {ga_preset_path})")
            return cached
        print("Preset GA_BEST not found; running GA search.")

    if compare_only:
        configs = [base_cfg]
    # Combination grid (small, systematic cross-product)
    combo_generators = ["standard", "adaptive_pool"]
    combo_scores = [base_score, decay_pair_score]
    combo_pools = [p36]
    combo_cohesion = [coh_soft, coh_pair]
    combo_penalties = [
        {"dynamic_bands": False},
        {"dynamic_bands": True},
    ]
    combo_penalty_scale = [0.50]
    combo_force = [False, True]

    if not compare_only:
        for gen in combo_generators:
            for sc in combo_scores:
                for pool in combo_pools:
                    for coh in combo_cohesion:
                        for pen in combo_penalties:
                            for ps in combo_penalty_scale:
                                for force in combo_force:
                                    if gen == "standard" and force:
                                        continue
                                    if gen != "adaptive_pool" and force:
                                        continue
                                    name = f"COMBO_{gen}_sc{combo_scores.index(sc)}_p{combo_pools.index(pool)}_c{combo_cohesion.index(coh)}_d{combo_penalties.index(pen)}_s{combo_penalty_scale.index(ps)}_f{int(force)}"
                                    configs.append({
                                        "name": name,
                                        "pool": pool,
                                        "penalty_scale": ps,
                                        "cohesion": coh,
                                        "score_config": sc,
                                        "penalty_config": pen,
                                        "generator": gen,
                                        "force_coverage": force,
                                    })

    def _genetic_tune_config() -> Dict[str, object]:
        rng = random.Random(RANDOM_SEED + 17)
        ga_quick = os.environ.get("GA_QUICK", "").strip() == "1"
        pop_size = 6 if ga_quick else (10 if compare_only else 18)
        generations = 3 if ga_quick else (6 if compare_only else 10)

        def _rand_gene() -> Dict[str, object]:
            return {
                "w_recent": rng.uniform(0.25, 0.70),
                "w_long": rng.uniform(0.05, 0.30),
                "w_season": rng.uniform(0.05, 0.25),
                "w_rank": rng.uniform(0.05, 0.20),
                "w_gap": rng.uniform(0.10, 0.35),
                "gap_cap": rng.uniform(0.20, 0.40),
                "cold_boost": rng.uniform(0.00, 0.30),
                "use_decay": rng.choice([True, False]),
                "w_decay": rng.uniform(0.00, 0.30),
                "decay_half_life": rng.randint(40, 120),
                "use_pairnum": rng.choice([True, False]),
                "w_pairnum": rng.uniform(0.00, 0.30),
                "penalty_scale": rng.uniform(0.30, 0.70),
                "dynamic_bands": rng.choice([True, False]),
                "sum_quant": rng.choice([None, (0.20, 0.80), (0.25, 0.75)]),
                "consec_max": rng.choice([1, 2, 3]),
                "pool_size": rng.randint(32, 40),
                "mid_pool_size": rng.randint(10, 18),
                "cold_pool_size": rng.randint(8, 14),
            }

        def _decode_gene(gene: Dict[str, object]) -> Dict[str, object]:
            score_config = {
                "w_recent": gene["w_recent"],
                "w_long": gene["w_long"],
                "w_season": gene["w_season"],
                "w_rank": gene["w_rank"],
                "w_gap": gene["w_gap"],
                "gap_cap": gene["gap_cap"],
                "cold_boost": gene["cold_boost"],
                "use_decay": gene["use_decay"],
                "w_decay": gene["w_decay"],
                "decay_half_life": gene["decay_half_life"],
                "use_pairnum": gene["use_pairnum"],
                "w_pairnum": gene["w_pairnum"],
            }
            penalty_config = {"dynamic_bands": gene["dynamic_bands"], "consec_max": gene["consec_max"]}
            if gene["sum_quant"]:
                penalty_config["sum_quantiles"] = gene["sum_quant"]
            pool_cfg = {
                "pool_size": gene["pool_size"],
                "mid_pool_size": gene["mid_pool_size"],
                "cold_pool_size": gene["cold_pool_size"],
                "hot_pool_size": 10,
                "overdue_pool_size": 10,
                "season_pool_size": 10,
                "cold_force_count": 2,
            }
            return {
                "name": "GA_BEST",
                "pool": pool_cfg,
                "penalty_scale": gene["penalty_scale"],
                "cohesion": coh_soft,
                "score_config": score_config,
                "penalty_config": penalty_config,
                "generator": "standard",
            }

        def _mutate(gene: Dict[str, object]) -> Dict[str, object]:
            g = dict(gene)
            key = rng.choice(list(g.keys()))
            if key in ("use_decay", "use_pairnum", "dynamic_bands"):
                g[key] = not g[key]
                return g
            if key == "sum_quant":
                g[key] = rng.choice([None, (0.20, 0.80), (0.25, 0.75)])
                return g
            if key == "consec_max":
                g[key] = rng.choice([1, 2, 3])
                return g
            if key in ("decay_half_life", "pool_size", "mid_pool_size", "cold_pool_size"):
                if key == "decay_half_life":
                    g[key] = rng.randint(40, 120)
                elif key == "pool_size":
                    g[key] = rng.randint(32, 40)
                elif key == "mid_pool_size":
                    g[key] = rng.randint(10, 18)
                else:
                    g[key] = rng.randint(8, 14)
                return g
            lo, hi = {
                "w_recent": (0.25, 0.70),
                "w_long": (0.05, 0.30),
                "w_season": (0.05, 0.25),
                "w_rank": (0.05, 0.20),
                "w_gap": (0.10, 0.35),
                "gap_cap": (0.20, 0.40),
                "cold_boost": (0.00, 0.30),
                "w_decay": (0.00, 0.30),
                "w_pairnum": (0.00, 0.30),
                "penalty_scale": (0.30, 0.70),
            }[key]
            g[key] = rng.uniform(lo, hi)
            return g

        def _crossover(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
            child = {}
            for k in a.keys():
                child[k] = a[k] if rng.random() < 0.5 else b[k]
            return child

        pop = [_rand_gene() for _ in range(pop_size)]
        best_gene = None
        best_key = None
        for _ in range(generations):
            scored_pop = []
            for gene in pop:
                cfg = _decode_gene(gene)
                w4, minhit, g4, g5, g3, total_hits = _evaluate_cfg_with_profile(df, main_cols, bt_dates, cfg)
                key = (w4, minhit, g4, g5, g3, total_hits)
                scored_pop.append((key, gene))
                if best_key is None or key > best_key:
                    best_key = key
                    best_gene = gene
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            survivors = [g for _, g in scored_pop[: max(4, pop_size // 3)]]
            next_pop = survivors[:]
            while len(next_pop) < pop_size:
                p1 = rng.choice(survivors)
                p2 = rng.choice(survivors)
                child = _crossover(p1, p2)
                if rng.random() < 0.35:
                    child = _mutate(child)
                next_pop.append(child)
            pop = next_pop

        if best_gene is None:
            return base_cfg
        best_cfg = _decode_gene(best_gene)
        best_cfg["name"] = "GA_BEST"
        return best_cfg

    best = None
    best_key = None
    results = []
    target_weeks = len(bt_dates)
    ga_best = _genetic_tune_config()
    ga_key = _evaluate_cfg_with_profile(df, main_cols, bt_dates, ga_best)
    _save_ga_preset(ga_best)
    results.append((ga_best["name"], *ga_key))
    best = ga_best
    best_key = ga_key
    for cfg in configs:
        key = _evaluate_cfg_with_profile(df, main_cols, bt_dates, cfg)
        results.append((cfg["name"], *key))
        if best_key is None or key > best_key:
            best_key = key
            best = cfg
        if key[0] >= target_weeks:
            best = cfg
            break

    print("\n=== TUNER RESULTS ===")
    for name, w4, minhit, g4, g5, g3, total_hits in results:
        print(f"{name}: weeks_with_4={w4} min_week_maxhit={minhit} ge4_total={g4} ge5_total={g5} ge3_total={g3} total_hits={total_hits}")
    if ga_best:
        print("GA best parameters are included as GA_BEST.")
    if best:
        print(f"Selected strategy: {best['name']}")
    return best or configs[0]

def _strategy_configs() -> List[Dict[str, object]]:
    return [
        {
            "name": "BASELINE",
            "cohesion": {
                "enabled": False,
                "accept_floor": 1.0,
                "accept_span": 0.0,
                "min_score": None,
                "weights": {"spread": 0.0, "pair": 0.0, "rank_cont": 0.0, "central": 0.0, "rank_mass": 0.0},
            },
        },
        {
            "name": "COHESION_SOFT",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.70,
                "accept_span": 0.60,
                "min_score": None,
                "weights": {"spread": 0.30, "pair": 0.35, "rank_cont": 0.20, "central": 0.10, "rank_mass": 0.05},
            },
        },
        {
            "name": "SOFT_PAIR_PLUS",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.70,
                "accept_span": 0.60,
                "min_score": None,
                "weights": {"spread": 0.25, "pair": 0.40, "rank_cont": 0.20, "central": 0.10, "rank_mass": 0.05},
            },
        },
        {
            "name": "SOFT_RANK_PLUS",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.70,
                "accept_span": 0.60,
                "min_score": None,
                "weights": {"spread": 0.30, "pair": 0.33, "rank_cont": 0.25, "central": 0.07, "rank_mass": 0.05},
            },
        },
        {
            "name": "SOFT_SPREAD_PLUS",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.70,
                "accept_span": 0.60,
                "min_score": None,
                "weights": {"spread": 0.35, "pair": 0.32, "rank_cont": 0.18, "central": 0.10, "rank_mass": 0.05},
            },
        },
        {
            "name": "SOFT_CENTRAL_PLUS",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.70,
                "accept_span": 0.60,
                "min_score": None,
                "weights": {"spread": 0.28, "pair": 0.33, "rank_cont": 0.19, "central": 0.15, "rank_mass": 0.05},
            },
        },
        {
            "name": "SOFT_BALANCED",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.70,
                "accept_span": 0.60,
                "min_score": None,
                "weights": {"spread": 0.29, "pair": 0.34, "rank_cont": 0.21, "central": 0.11, "rank_mass": 0.05},
            },
        },
        {
            "name": "PAIR_HEAVY",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.72,
                "accept_span": 0.60,
                "min_score": 0.55,
                "weights": {"spread": 0.20, "pair": 0.50, "rank_cont": 0.15, "central": 0.10, "rank_mass": 0.05},
            },
        },
        {
            "name": "RANK_CONT_HEAVY",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.65,
                "accept_span": 0.70,
                "min_score": None,
                "weights": {"spread": 0.25, "pair": 0.30, "rank_cont": 0.30, "central": 0.10, "rank_mass": 0.05},
            },
        },
        {
            "name": "COHESION_GATED",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.75,
                "accept_span": 0.50,
                "min_score": 0.60,
                "weights": {"spread": 0.30, "pair": 0.35, "rank_cont": 0.20, "central": 0.10, "rank_mass": 0.05},
            },
        },
    ]


def _build_core_numbers(scored: List[CandidateScore], core_size: int) -> List[int]:
    return [c.n for c in scored[:core_size]]


def _build_sampling_pool(scored: List[CandidateScore], include_cold: bool) -> Tuple[List[int], List[float]]:
    top_pool = [c.n for c in scored[:POOL_SIZE]]
    mid_pool = [c.n for c in scored[POOL_SIZE:POOL_SIZE + MID_POOL_SIZE]]
    cold_pool = [c.n for c in scored if c.freq_recent <= 1][:COLD_POOL_SIZE] if include_cold else []
    pool = list(dict.fromkeys(top_pool + mid_pool + cold_pool))
    score_map = {c.n: c.total_score for c in scored}
    min_score = min(score_map.values()) if score_map else 0.0
    weights = [(score_map.get(n, 0.0) - min_score) + 0.25 for n in pool]
    return pool, weights


def _generate_candidate_pool(
    scored: List[CandidateScore],
    count: int,
    rng: random.Random,
    include_cold: bool,
) -> List[List[int]]:
    pool, weights = _build_sampling_pool(scored, include_cold=include_cold)
    candidates: List[List[int]] = []
    seen = set()
    attempts = 0
    max_attempts = max(count * 10, 5000)
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        pick = sorted(_weighted_sample_no_replace(pool, weights, NUMBERS_PER_TICKET, rng))
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(pick)
    return candidates


def _top_pairs(pair_counts: Dict[Tuple[int, int], int], top_k: int) -> List[Tuple[int, int]]:
    if not pair_counts:
        return []
    return [p for p, _ in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def _pair_anchored_candidates(
    scored: List[CandidateScore],
    pair_counts: Dict[Tuple[int, int], int],
    rng: random.Random,
    count: int,
) -> List[List[int]]:
    pool, weights = _build_sampling_pool(scored, include_cold=False)
    top_pairs = _top_pairs(pair_counts, top_k=min(60, len(pair_counts)))
    candidates: List[List[int]] = []
    seen = set()

    if not top_pairs:
        return _generate_candidate_pool(scored, count, rng, include_cold=False)

    attempts = 0
    max_attempts = max(count * 10, 5000)
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        pair = rng.choice(top_pairs)
        base = list(pair)
        remaining_k = NUMBERS_PER_TICKET - len(base)
        remaining_items = [n for n in pool if n not in base]
        remaining_weights = [w for n, w in zip(pool, weights) if n not in base]
        if remaining_k <= 0 or not remaining_items:
            continue
        rest = _weighted_sample_no_replace(remaining_items, remaining_weights, remaining_k, rng)
        pick = sorted(dict.fromkeys(base + rest))
        if len(pick) != NUMBERS_PER_TICKET:
            continue
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(pick)
    return candidates


def _swap_variants(
    base_ticket: List[int],
    scored: List[CandidateScore],
    rng: random.Random,
    variants: int,
) -> List[List[int]]:
    top_candidates = [c.n for c in scored[:SWAP_CANDIDATE_BAND]]
    out = []
    base_set = set(base_ticket)
    for _ in range(variants):
        t = list(base_ticket)
        idx = rng.randrange(len(t))
        replacements = [n for n in top_candidates if n not in base_set]
        if not replacements:
            break
        t[idx] = rng.choice(replacements)
        t = sorted(dict.fromkeys(t))
        if len(t) != NUMBERS_PER_TICKET:
            continue
        out.append(t)
    return out


def _select_portfolio(
    candidates: List[Tuple[List[int], float]],
    target_count: int,
    core_numbers: List[int],
    diversity_penalty: float,
    non_core_penalty: float,
) -> List[List[int]]:
    selected: List[List[int]] = []
    selected_set = set()
    core_set = set(core_numbers)
    core_use = {n: 0 for n in core_numbers}

    for _ in range(target_count):
        best = None
        best_score = -1e9
        for ticket, quality in candidates:
            key = tuple(ticket)
            if key in selected_set:
                continue
            if quality < 0.0:
                continue
            max_overlap = 0
            t_set = set(ticket)
            for s in selected:
                max_overlap = max(max_overlap, len(t_set.intersection(s)))
                if max_overlap == NUMBERS_PER_TICKET:
                    break
            diversity_pen = diversity_penalty * (max_overlap / float(NUMBERS_PER_TICKET))

            core_bonus = 0.0
            core_over_pen = 0.0
            for n in ticket:
                if n in core_set:
                    if core_use[n] < CORE_MIN_USE:
                        core_bonus += 0.05
                    elif core_use[n] >= CORE_MAX_USE:
                        core_over_pen += 0.05
            non_core = sum(1 for n in ticket if n not in core_set)
            score = quality + core_bonus - core_over_pen - diversity_pen - (non_core_penalty * non_core)
            if score > best_score:
                best_score = score
                best = ticket
        if best is None:
            break
        selected.append(best)
        selected_set.add(tuple(best))
        for n in best:
            if n in core_use:
                core_use[n] += 1
    return selected


def generate_portfolio_tickets(
    scored: List[CandidateScore],
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    train = df[df["Date"] < pd.Timestamp(target_date)]
    score_map = {c.n: c.total_score for c in scored}
    rank_map = {c.n: c.rank_recent for c in scored}
    pair_counts = _build_pair_counts(train, main_cols)
    pair_base = _percentile(list(pair_counts.values()), COHESION_PAIR_BASE_PCTL) if pair_counts else 0

    core_numbers = _build_core_numbers(scored, CORE_SIZE)

    strict_weights = {"spread": 0.30, "pair": 0.35, "rank_cont": 0.20, "central": 0.10, "rank_mass": 0.05}
    diffuse_weights = {"spread": 0.20, "pair": 0.25, "rank_cont": 0.20, "central": 0.15, "rank_mass": 0.20}

    cohesive_candidates = _generate_candidate_pool(scored, PORTFOLIO_CANDIDATES, rng, include_cold=False)
    diffuse_candidates = _generate_candidate_pool(scored, int(PORTFOLIO_CANDIDATES * 0.5), rng, include_cold=True)

    cohesive_scored: List[Tuple[List[int], float]] = []
    diffuse_scored: List[Tuple[List[int], float]] = []

    for t in cohesive_candidates:
        q = _ticket_quality(
            t, score_map, rank_map, pair_counts, pair_base,
            COHESION_TOP_RANK, STRICT_MAX_SPREAD, STRICT_MAX_GAP, 0, strict_weights
        )
        if q >= 0.0:
            cohesive_scored.append((t, q))

    for t in diffuse_candidates:
        q = _ticket_quality(
            t, score_map, rank_map, pair_counts, pair_base,
            COHESION_TOP_RANK, DIFFUSE_MAX_SPREAD, DIFFUSE_MAX_GAP, 2, diffuse_weights
        )
        if q >= 0.0:
            diffuse_scored.append((t, q))

    cohesive_scored.sort(key=lambda x: x[1], reverse=True)
    base_tickets = [t for t, _ in cohesive_scored[:BASE_TICKET_COUNT]]
    for base in base_tickets:
        for variant in _swap_variants(base, scored, rng, BASE_SWAP_VARIANTS):
            q = _ticket_quality(
                variant, score_map, rank_map, pair_counts, pair_base,
                COHESION_TOP_RANK, STRICT_MAX_SPREAD, STRICT_MAX_GAP, 0, strict_weights
            )
            if q >= 0.0:
                cohesive_scored.append((variant, q))

    cohesive_scored.sort(key=lambda x: x[1], reverse=True)
    diffuse_scored.sort(key=lambda x: x[1], reverse=True)

    cohesive_pick = _select_portfolio(
        cohesive_scored, COHESIVE_TICKETS, core_numbers, COHESIVE_DIVERSITY_PENALTY, 0.08
    )
    diffuse_pick = _select_portfolio(
        diffuse_scored, DIFFUSE_TICKETS, core_numbers, DIFFUSE_DIVERSITY_PENALTY, 0.02
    )

    tickets = cohesive_pick + diffuse_pick
    return tickets[:NUM_TICKETS]


def generate_pair_anchored_tickets(
    scored: List[CandidateScore],
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
    pair_weight: float = 0.45,
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    train = df[df["Date"] < pd.Timestamp(target_date)]
    score_map = {c.n: c.total_score for c in scored}
    rank_map = {c.n: c.rank_recent for c in scored}
    pair_counts = _build_pair_counts(train, main_cols)
    pair_base = _percentile(list(pair_counts.values()), COHESION_PAIR_BASE_PCTL) if pair_counts else 0
    core_numbers = _build_core_numbers(scored, CORE_SIZE)

    candidates = _pair_anchored_candidates(scored, pair_counts, rng, PORTFOLIO_CANDIDATES)
    weights = {"spread": 0.25, "pair": pair_weight, "rank_cont": 0.20, "central": 0.10, "rank_mass": 0.05}
    scored_candidates: List[Tuple[List[int], float]] = []
    for t in candidates:
        q = _ticket_quality(
            t, score_map, rank_map, pair_counts, pair_base,
            COHESION_TOP_RANK, STRICT_MAX_SPREAD, STRICT_MAX_GAP, 0, weights
        )
        if q >= 0.0:
            scored_candidates.append((t, q))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    selected = _select_portfolio(
        scored_candidates, NUM_TICKETS, core_numbers, COHESIVE_DIVERSITY_PENALTY, 0.08
    )
    return selected[:NUM_TICKETS]


def generate_pair_cluster_tickets(
    scored: List[CandidateScore],
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
    pool_config: Dict[str, object] = None,
    penalty_scale: float = None,
    penalty_config: Dict[str, object] = None,
    cohesion_config: Dict[str, object] = None,
    seed_recent_days: int = 120,
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    t = pd.Timestamp(target_date)
    train = df[df["Date"] < t]
    if train.empty:
        return []

    dist = _history_distributions(train, main_cols, target_date)
    score_map = {c.n: c.total_score for c in scored}
    rank_map = {c.n: c.rank_recent for c in scored}

    pool_cfg = pool_config or {}
    pool_size = int(pool_cfg.get("pool_size", POOL_SIZE))
    mid_pool_size = int(pool_cfg.get("mid_pool_size", MID_POOL_SIZE))
    cold_pool_size = int(pool_cfg.get("cold_pool_size", COLD_POOL_SIZE))

    top_pool = [c.n for c in scored[:pool_size]]
    mid_pool = [c.n for c in scored[pool_size:pool_size + mid_pool_size]]
    cold_pool = [c.n for c in scored if c.freq_recent <= 1][:cold_pool_size]
    pool = list(dict.fromkeys(top_pool + mid_pool + cold_pool))

    recent_start = t - pd.Timedelta(days=seed_recent_days)
    recent = train[(train["Date"] >= recent_start) & (train["Date"] < t)]
    pair_counts = _pair_count_map(train, main_cols)
    pair_counts_recent = _pair_count_map(recent, main_cols)

    # Build top pair list weighted by overall + recent counts
    pair_scores = []
    for (a, b), cnt in pair_counts.items():
        rc = pair_counts_recent.get((a, b), 0)
        score = cnt + (0.7 * rc)
        pair_scores.append(((a, b), score))
    pair_scores.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [p for p, _ in pair_scores[:200]] if pair_scores else []

    pair_base = _percentile(list(pair_counts.values()), COHESION_PAIR_BASE_PCTL) if pair_counts else 0
    if cohesion_config is None:
        cohesion_config = {
            "enabled": COHESION_ENABLED,
            "accept_floor": COHESION_ACCEPT_FLOOR,
            "accept_span": COHESION_ACCEPT_SPAN,
            "min_score": None,
            "weights": {
                "spread": COHESION_W_SPREAD,
                "pair": COHESION_W_PAIR,
                "rank_cont": COHESION_W_RANK_CONT,
                "central": COHESION_W_CENTRAL,
                "rank_mass": 0.05,
            },
        }

    tickets: List[List[int]] = []
    global_use = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
    overlap_cap = OVERLAP_CAP
    global_max = GLOBAL_MAX_USES
    attempts = 0
    pair_idx = 0

    while len(tickets) < NUM_TICKETS and attempts < MAX_ATTEMPTS:
        attempts += 1
        if not top_pairs:
            break
        a, b = top_pairs[pair_idx % len(top_pairs)]
        pair_idx += 1
        if a not in pool or b not in pool:
            continue

        pick = [a, b]
        candidates = [n for n in pool if n not in pick]

        while len(pick) < NUMBERS_PER_TICKET and candidates:
            scored_candidates = []
            for n in candidates:
                score = 0.0
                for p in pick:
                    score += pair_counts.get(_pair_key(p, n), 0)
                    score += 0.7 * pair_counts_recent.get(_pair_key(p, n), 0)
                score += 0.3 * score_map.get(n, 0.0)
                scored_candidates.append((n, score))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_k = [n for n, _ in scored_candidates[:12]]
            if not top_k:
                break
            n = rng.choice(top_k)
            pick.append(n)
            candidates.remove(n)

        if len(pick) != NUMBERS_PER_TICKET:
            continue
        pick = sorted(pick)
        if pick in tickets:
            continue
        if any(global_use[n] >= global_max for n in pick):
            continue
        s_pick = set(pick)
        if any(len(s_pick.intersection(t)) > overlap_cap for t in tickets):
            continue

        penalty = _ticket_penalty(pick, dist, penalty_config=penalty_config)
        scale = PENALTY_SCALE if penalty_scale is None else float(penalty_scale)
        if (penalty * scale) > 2.5:
            continue
        if cohesion_config.get("enabled"):
            cohesion = _cohesion_score(
                pick, score_map, rank_map, pair_counts, pair_base, cohesion_config["weights"]
            )
            min_score = cohesion_config.get("min_score")
            if min_score is not None and cohesion < float(min_score):
                continue

        tickets.append(pick)
        for n in pick:
            global_use[n] += 1

    if SUPP_SOFT_SWAP and supp_candidates:
        tickets = _apply_supp_soft_swaps(
            tickets, supp_candidates, supp_scores, score_map, pool, global_use, global_max, overlap_cap
        )
    if SUPP_FOCUS_TICKETS > 0:
        tickets = _apply_supp_focus_tickets(
            tickets, scored, supp_scores, score_map, global_use, global_max, overlap_cap
        )

    print(f"Generated {len(tickets)}/{NUM_TICKETS} tickets in {attempts} attempts")
    return tickets


def generate_adaptive_pool_tickets(
    scored: List[CandidateScore],
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
    pool_config: Dict[str, object] = None,
    penalty_scale: float = None,
    penalty_config: Dict[str, object] = None,
    cohesion_config: Dict[str, object] = None,
    force_coverage: bool = False,
) -> List[List[int]]:
    t = pd.Timestamp(target_date)
    train = df[df["Date"] < t]
    if train.empty:
        return []
    pair_counts = _pair_count_map(train, main_cols)
    pool_cfg = pool_config or {}
    pool_size = int(pool_cfg.get("pool_size", POOL_SIZE))
    mid_pool_size = int(pool_cfg.get("mid_pool_size", MID_POOL_SIZE))
    cold_pool_size = int(pool_cfg.get("cold_pool_size", COLD_POOL_SIZE))
    pool_target = pool_size + mid_pool_size
    pool_override = _build_adaptive_pool(scored, pair_counts, pool_target, cold_pool_size)
    return generate_tickets(
        scored,
        df,
        main_cols,
        target_date,
        use_weights=True,
        seed_hot_overdue=True,
        force_coverage=force_coverage,
        cohesion_config=cohesion_config,
        penalty_scale=penalty_scale,
        pool_override=pool_override,
        pool_config=pool_config,
        penalty_config=penalty_config,
    )


def _coverage_select(
    candidates: List[List[int]],
    num_tickets: int,
    top_numbers: set,
    top_pairs: set,
    pair_weight: float = 1.0,
    num_weight: float = 0.5,
    overlap_weight: float = 0.25,
) -> List[List[int]]:
    selected: List[List[int]] = []
    covered_nums = set()
    covered_pairs = set()
    use_count: Dict[int, int] = {}
    for _ in range(num_tickets):
        best = None
        best_score = -1e9
        for t in candidates:
            if t in selected:
                continue
            new_nums = sum(1 for n in t if n in top_numbers and n not in covered_nums)
            new_pairs = sum(1 for p in _ticket_pairs(t) if p in top_pairs and p not in covered_pairs)
            overlap = sum(max(0, use_count.get(n, 0) - 1) for n in t)
            score = (pair_weight * new_pairs) + (num_weight * new_nums) - (overlap_weight * overlap)
            if score > best_score:
                best_score = score
                best = t
        if best is None:
            break
        selected.append(best)
        for n in best:
            use_count[n] = use_count.get(n, 0) + 1
            if n in top_numbers:
                covered_nums.add(n)
        for p in _ticket_pairs(best):
            if p in top_pairs:
                covered_pairs.add(p)
    return selected


def generate_graph_coverage_tickets(
    scored: List[CandidateScore],
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
    pool_config: Dict[str, object] = None,
    penalty_scale: float = None,
    penalty_config: Dict[str, object] = None,
    cohesion_config: Dict[str, object] = None,
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    t = pd.Timestamp(target_date)
    train = df[df["Date"] < t]
    if train.empty:
        return []

    dist = _history_distributions(train, main_cols, target_date)
    score_map = {c.n: c.total_score for c in scored}
    rank_map = {c.n: c.rank_recent for c in scored}

    pool_cfg = pool_config or {}
    pool_size = int(pool_cfg.get("pool_size", POOL_SIZE))
    mid_pool_size = int(pool_cfg.get("mid_pool_size", MID_POOL_SIZE))
    cold_pool_size = int(pool_cfg.get("cold_pool_size", COLD_POOL_SIZE))

    top_pool = [c.n for c in scored[:pool_size]]
    mid_pool = [c.n for c in scored[pool_size:pool_size + mid_pool_size]]
    cold_pool = [c.n for c in scored if c.freq_recent <= 1][:cold_pool_size]
    pool = list(dict.fromkeys(top_pool + mid_pool + cold_pool))

    pair_counts = _pair_count_map(train, main_cols)
    pair_base = _percentile(list(pair_counts.values()), COHESION_PAIR_BASE_PCTL) if pair_counts else 0

    centrality = {n: 0.0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
    for (a, b), cnt in pair_counts.items():
        centrality[a] = centrality.get(a, 0.0) + cnt
        centrality[b] = centrality.get(b, 0.0) + cnt
    max_cent = max(centrality.values()) if centrality else 0.0
    if max_cent > 0:
        for n in centrality:
            centrality[n] = centrality[n] / max_cent

    if cohesion_config is None:
        cohesion_config = {
            "enabled": COHESION_ENABLED,
            "accept_floor": COHESION_ACCEPT_FLOOR,
            "accept_span": COHESION_ACCEPT_SPAN,
            "min_score": None,
            "weights": {
                "spread": COHESION_W_SPREAD,
                "pair": COHESION_W_PAIR,
                "rank_cont": COHESION_W_RANK_CONT,
                "central": COHESION_W_CENTRAL,
                "rank_mass": 0.05,
            },
        }

    top_numbers = set(n for n in [c.n for c in scored[:24]])
    top_pairs = set()
    if pair_counts:
        top_pairs = set([p for p, _ in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:120]])

    candidates: List[List[int]] = []
    candidate_target = min(PORTFOLIO_CANDIDATES, 1000)
    max_attempts = max(1200, candidate_target * 6)
    attempts = 0
    while len(candidates) < candidate_target and attempts < max_attempts:
        attempts += 1
        if not pool:
            break
        first_weights = [(score_map.get(n, 0.0) + centrality.get(n, 0.0) + 0.05) for n in pool]
        n1 = rng.choices(pool, weights=first_weights, k=1)[0]

        n1_dec = _decade_of(n1)
        alt_pool = [n for n in pool if _decade_of(n) != n1_dec]
        if not alt_pool:
            alt_pool = pool
        n2 = rng.choice(alt_pool)
        if n2 == n1:
            continue
        pick = [n1, n2]

        while len(pick) < NUMBERS_PER_TICKET:
            candidates_left = [n for n in pool if n not in pick]
            if not candidates_left:
                break
            weights = []
            for n in candidates_left:
                pair_aff = sum(pair_counts.get(_pair_key(n, p), 0) for p in pick)
                if pair_base:
                    pair_term = 0.15 * (pair_aff / float(pair_base))
                else:
                    pair_term = 0.15 * pair_aff
                w = (0.55 * score_map.get(n, 0.0)) + (0.30 * centrality.get(n, 0.0)) + pair_term
                weights.append(max(w, 0.01))
            pick.append(rng.choices(candidates_left, weights=weights, k=1)[0])

        pick = sorted(set(pick))
        if len(pick) != NUMBERS_PER_TICKET:
            continue

        penalty = _ticket_penalty(pick, dist, penalty_config=penalty_config)
        scale = PENALTY_SCALE if penalty_scale is None else float(penalty_scale)
        accept_prob = math.exp(-penalty * scale)
        if cohesion_config.get("enabled"):
            cohesion = _cohesion_score(
                pick, score_map, rank_map, pair_counts, pair_base, cohesion_config["weights"]
            )
            min_score = cohesion_config.get("min_score")
            if min_score is not None and cohesion < float(min_score):
                continue
            accept_prob *= (cohesion_config["accept_floor"] + cohesion_config["accept_span"] * cohesion)
            if accept_prob > 1.0:
                accept_prob = 1.0
        if rng.random() > accept_prob:
            continue
        if pick not in candidates:
            candidates.append(pick)

    selected = _coverage_select(
        candidates,
        NUM_TICKETS,
        top_numbers,
        top_pairs,
        pair_weight=1.0,
        num_weight=0.5,
        overlap_weight=0.25,
    )
    if len(selected) < NUM_TICKETS:
        extras = [t for t in candidates if t not in selected]
        rng.shuffle(extras)
        selected.extend(extras[: max(0, NUM_TICKETS - len(selected))])
    print(f"Generated {len(selected)}/{NUM_TICKETS} tickets in {attempts} attempts")
    return selected[:NUM_TICKETS]


def _evaluate_variant(
    df: pd.DataFrame,
    main_cols: List[str],
    bt_dates: List[pd.Timestamp],
    variant_name: str,
    variant_fn,
) -> Dict[str, object]:
    weeks_with_5 = 0
    weeks_with_4 = 0
    ge3_total = 0
    ge4_total = 0
    ge5_total = 0
    for d in bt_dates:
        bt_date = d.strftime("%Y-%m-%d")
        row = df[df["Date"] == d].iloc[0]
        bt_draw = [int(row[c]) for c in main_cols]
        bt_scored = score_numbers(df, main_cols, bt_date, debug=False)
        bt_tickets = variant_fn(bt_scored, df, main_cols, bt_date)
        summary = _hit_summary(bt_draw, bt_tickets)
        ge3_total += summary.get("ge3", 0)
        ge4_total += summary.get("ge4", 0)
        ge5_total += summary.get("ge5", 0)
        if summary.get("ge4", 0) > 0:
            weeks_with_4 += 1
        if summary.get("ge5", 0) > 0:
            weeks_with_5 += 1
    return {
        "name": variant_name,
        "hits5": weeks_with_5,
        "ge3_total": ge3_total,
        "ge4_total": ge4_total,
        "ge5_total": ge5_total,
        "weeks_with_4": weeks_with_4,
    }


def _evaluate_cfg_with_profile(
    df: pd.DataFrame,
    main_cols: List[str],
    bt_dates: List[pd.Timestamp],
    cfg: Dict[str, object],
) -> Tuple[int, int, int, int, int, int]:
    weeks_with_4 = 0
    ge3_total = 0
    ge4_total = 0
    ge5_total = 0
    total_hits = 0
    min_week_maxhit = 99
    if len(bt_dates) >= 10:
        start_idx = 5
    else:
        start_idx = 0
    for i in range(start_idx, len(bt_dates)):
        d = bt_dates[i]
        learn_dates = bt_dates[:i]
        profile = _learn_winner_profile(df, main_cols, learn_dates) if len(learn_dates) >= 5 else None
        bt_date = d.strftime("%Y-%m-%d")
        row = df[df["Date"] == d].iloc[0]
        bt_draw = [int(row[c]) for c in main_cols]
        bt_scored = score_numbers(df, main_cols, bt_date, debug=False, score_config=cfg.get("score_config"))
        score_map = {c.n: c for c in bt_scored}
        if cfg.get("generator") == "portfolio":
            bt_tickets = generate_portfolio_tickets(bt_scored, df, main_cols, bt_date)
        elif cfg.get("generator") == "pair_anchored":
            bt_tickets = generate_pair_anchored_tickets(bt_scored, df, main_cols, bt_date, pair_weight=0.50)
        elif cfg.get("generator") == "pair_cluster":
            bt_tickets = generate_pair_cluster_tickets(
                bt_scored,
                df,
                main_cols,
                bt_date,
                pool_config=cfg.get("pool"),
                penalty_scale=cfg.get("penalty_scale"),
                penalty_config=cfg.get("penalty_config"),
                cohesion_config=cfg.get("cohesion"),
            )
        elif cfg.get("generator") == "graph_cover":
            bt_tickets = generate_graph_coverage_tickets(
                bt_scored,
                df,
                main_cols,
                bt_date,
                pool_config=cfg.get("pool"),
                penalty_scale=cfg.get("penalty_scale"),
                penalty_config=cfg.get("penalty_config"),
                cohesion_config=cfg.get("cohesion"),
            )
        elif cfg.get("generator") == "adaptive_pool":
            bt_tickets = generate_adaptive_pool_tickets(
                bt_scored,
                df,
                main_cols,
                bt_date,
                pool_config=cfg.get("pool"),
                penalty_scale=cfg.get("penalty_scale"),
                penalty_config=cfg.get("penalty_config"),
                cohesion_config=cfg.get("cohesion"),
                force_coverage=cfg.get("force_coverage", False),
            )
        else:
            bt_tickets = generate_tickets(
                bt_scored,
                df,
                main_cols,
                bt_date,
                use_weights=True,
                seed_hot_overdue=cfg.get("seed_hot_overdue", False),
                force_coverage=cfg.get("force_coverage", False),
                cohesion_config=cfg.get("cohesion"),
                penalty_scale=cfg.get("penalty_scale"),
                pool_config=cfg.get("pool"),
                penalty_config=cfg.get("penalty_config"),
            )
        if profile and USE_PROFILE_FILTER:
            strict = [t for t in bt_tickets if _ticket_passes_profile(t, score_map, profile)]
            relaxed = [t for t in bt_tickets if _ticket_passes_profile_relaxed(t, score_map, profile) and t not in strict]
            remainder = [t for t in bt_tickets if t not in strict and t not in relaxed]
            bt_tickets = (strict + relaxed + remainder)[:NUM_TICKETS]
        summary = _hit_summary(bt_draw, bt_tickets)
        ge3_total += summary.get("ge3", 0)
        ge4_total += summary.get("ge4", 0)
        ge5_total += summary.get("ge5", 0)
        total_hits += summary.get("total_hits", 0)
        if summary.get("ge4", 0) > 0:
            weeks_with_4 += 1
        rd_set = set(bt_draw)
        max_hit = 0
        for t in bt_tickets:
            max_hit = max(max_hit, len(set(t).intersection(rd_set)))
        min_week_maxhit = min(min_week_maxhit, max_hit)
    return weeks_with_4, min_week_maxhit, ge4_total, ge5_total, ge3_total, total_hits


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, main_cols = _load_csv(CSV_PATH)
    supp_cols = _detect_supp_cols(df)
    globals()["SUPP_COLS"] = supp_cols

    # Run for the configured target date (prediction); if it exists in CSV, use it as a backtest draw.
    t_target = pd.Timestamp(TARGET_DATE)
    if pd.isna(t_target):
        raise ValueError("TARGET_DATE must be parseable (YYYY-MM-DD)")
    df_target = df[df["Date"] == t_target]

    if SWEEP_MODE:
        backtest_rows = df.sort_values("Date").tail(SWEEP_BACKTEST_DRAWS)
        bt_dates = [row["Date"] for _, row in backtest_rows.iterrows()]
        print(f"\n=== SWEEP (LAST {len(bt_dates)} DRAWS) ===")
        for s in _strategy_configs():
            result = _evaluate_strategy(
                df,
                main_cols,
                bt_dates,
                s["name"],
                s["cohesion"],
                debug=False,
            )
            print(
                f"{result['name']}:",
                f"weeks_with_4={result['weeks_with_4']}",
                f"weeks_with_5={result['hits5']}",
                f"ge3_total={result['ge3_total']}",
                f"ge4_total={result['ge4_total']}",
                f"ge5_total={result['ge5_total']}",
            )
        raise SystemExit

    if VARIANT_SWEEP:
        backtest_rows = df.sort_values("Date").tail(VARIANT_BACKTEST_DRAWS)
        bt_dates = [row["Date"] for _, row in backtest_rows.iterrows()]
        print(f"\n=== VARIANT SWEEP (LAST {len(bt_dates)} DRAWS) ===")
        strategies = _strategy_configs()
        cohesion_soft = next((s["cohesion"] for s in strategies if s["name"] == "COHESION_SOFT"), None)
        pair_heavy = next((s["cohesion"] for s in strategies if s["name"] == "PAIR_HEAVY"), None)
        variants = [
            ("GEN_BASELINE", lambda scored, dff, cols, d: generate_tickets(
                scored, dff, cols, d, use_weights=True, seed_hot_overdue=False,
                force_coverage=FORCE_COVERAGE, cohesion_config=None
            )),
            ("GEN_COHESION_SOFT", lambda scored, dff, cols, d: generate_tickets(
                scored, dff, cols, d, use_weights=True, seed_hot_overdue=False,
                force_coverage=FORCE_COVERAGE, cohesion_config=cohesion_soft
            )),
            ("GEN_PAIR_HEAVY", lambda scored, dff, cols, d: generate_tickets(
                scored, dff, cols, d, use_weights=True, seed_hot_overdue=False,
                force_coverage=FORCE_COVERAGE, cohesion_config=pair_heavy
            )),
            ("PORTFOLIO_DEFAULT", generate_portfolio_tickets),
            ("PAIR_ANCHORED", lambda scored, dff, cols, d: generate_pair_anchored_tickets(
                scored, dff, cols, d, pair_weight=0.50
            )),
        ]
        for name, fn in variants:
            result = _evaluate_variant(df, main_cols, bt_dates, name, fn)
            print(
                f"{result['name']}:",
                f"weeks_with_4={result['weeks_with_4']}",
                f"weeks_with_5={result['hits5']}",
                f"ge3_total={result['ge3_total']}",
                f"ge4_total={result['ge4_total']}",
                f"ge5_total={result['ge5_total']}",
            )
        raise SystemExit

    def _build_consensus_seed(scored_list: List[CandidateScore]) -> List[int]:
        if not scored_list:
            return []
        seed = []
        top_scores = [c.n for c in scored_list[:6]]
        top_gap = max(scored_list, key=lambda x: x.gap_days).n
        top_season = max(scored_list, key=lambda x: x.freq_season).n
        for n in top_scores[:3] + [top_gap, top_season]:
            if n not in seed:
                seed.append(n)
        if len(seed) < 4:
            for n in top_scores:
                if n not in seed:
                    seed.append(n)
                if len(seed) >= 4:
                    break
        return seed[:4]

    run_date = t_target.strftime("%Y-%m-%d")

    # Backtest: run on the last 10 available draws excluding TARGET_DATE (oldest to newest).
    backtest_rows = df[df["Date"] != t_target].sort_values("Date").tail(10)
    bt_dates = [row["Date"] for _, row in backtest_rows.iterrows()]
    learn_dates = bt_dates[:5] if len(bt_dates) >= 10 else []
    predict_dates = bt_dates[5:] if len(bt_dates) >= 10 else bt_dates
    profile = _learn_winner_profile(df, main_cols, learn_dates) if learn_dates else None

    strategies = _strategy_configs()
    main_cfg = next((s["cohesion"] for s in strategies if s["name"] == DEFAULT_STRATEGY_NAME), None)
    if main_cfg is None:
        raise ValueError(f"Unknown DEFAULT_STRATEGY_NAME: {DEFAULT_STRATEGY_NAME}")

    tuner_cfg = None
    use_ga_preset = os.environ.get("USE_GA_PRESET", "1").strip() == "1"
    if use_ga_preset:
        if DEFAULT_PRESET == "OZ_BASELINE":
            tuner_cfg = {
                "name": "OZ_BASELINE",
                "pool": {
                    "pool_size": 36, "mid_pool_size": 14, "cold_pool_size": 12,
                    "hot_pool_size": 10, "overdue_pool_size": 10, "season_pool_size": 10,
                    "cold_force_count": 2,
                },
                "penalty_scale": 0.55,
                "force_coverage": False,
                "cohesion": main_cfg,
            }
            print("Using OZ_BASELINE preset (no tuner).")
        else:
            ga_preset_path = os.environ.get("GA_PRESET_PATH", "ga_best_config.json")
            tuner_cfg = _load_ga_preset_file(ga_preset_path)
            if tuner_cfg:
                tuner_cfg.setdefault("name", DEFAULT_PRESET)
                print(f"Using {DEFAULT_PRESET} preset (no tuner): {ga_preset_path}")
                if tuner_cfg.get("cohesion"):
                    main_cfg = tuner_cfg["cohesion"]
            else:
                print(f"GA_BEST preset not found at {ga_preset_path}; falling back to tuner.")
                tuner_cfg = None

    if TUNER_MODE and tuner_cfg is None:
        tuner_rows = df[df["Date"] != t_target].sort_values("Date").tail(TUNER_BACKTEST_DRAWS)
        tuner_dates = [row["Date"] for _, row in tuner_rows.iterrows()]
        tuner_cfg = _tune_configs(df, main_cols, tuner_dates, main_cfg)
        if tuner_cfg.get("cohesion"):
            main_cfg = tuner_cfg["cohesion"]

    if len(bt_dates) >= 10:
        start_idx = 5
    else:
        start_idx = 0
    print(
        f"\n=== BACKTEST (LAST {len(bt_dates)} DRAWS; "
        f"PREDICT LAST {len(bt_dates) - start_idx}; INCREMENTAL LEARNING) ==="
    )
    for i in range(start_idx, len(bt_dates)):
        d = bt_dates[i]
        learn_dates = bt_dates[:i]
        profile = _learn_winner_profile(df, main_cols, learn_dates) if len(learn_dates) >= 5 else None
        if profile:
            print(f"\nLearned profile from last {len(learn_dates)} draws:")
            print(
                f"  rank_soft_max={profile['rank_soft_max']} "
                f"rank_hard_max={profile['rank_hard_max']} "
                f"gap_hard_max={profile['gap_hard_max']} "
                f"score_min={profile['score_min']:.3f}"
            )
        bt_date = d.strftime("%Y-%m-%d")
        row = df[df["Date"] == d].iloc[0]
        bt_draw = [int(row[c]) for c in main_cols]
        bt_supp = [int(row[c]) for c in supp_cols] if supp_cols else []
        bt_scored = score_numbers(
            df,
            main_cols,
            bt_date,
            DEBUG_PRINT,
            show_all=PRINT_ALL_SCORES_WHEN_REAL,
        )
        supp_candidates = _supp_candidate_set(
            df, supp_cols, bt_date
        ) if supp_cols and (SUPP_MIN_HIT > 0 or SUPP_SOFT_SWAP or SUPP_FOCUS_TICKETS > 0) else []
        score_map = {c.n: c for c in bt_scored}
        if PORTFOLIO_MODE:
            bt_tickets = generate_portfolio_tickets(bt_scored, df, main_cols, bt_date)
        else:
            if tuner_cfg and tuner_cfg.get("generator") == "portfolio":
                bt_tickets = generate_portfolio_tickets(bt_scored, df, main_cols, bt_date)
            elif tuner_cfg and tuner_cfg.get("generator") == "pair_anchored":
                bt_tickets = generate_pair_anchored_tickets(bt_scored, df, main_cols, bt_date, pair_weight=0.50)
            elif tuner_cfg and tuner_cfg.get("generator") == "pair_cluster":
                bt_tickets = generate_pair_cluster_tickets(
                    bt_scored,
                    df,
                    main_cols,
                    bt_date,
                    pool_config=tuner_cfg.get("pool"),
                    penalty_scale=tuner_cfg.get("penalty_scale"),
                    penalty_config=tuner_cfg.get("penalty_config"),
                    cohesion_config=main_cfg,
                )
            elif tuner_cfg and tuner_cfg.get("generator") == "graph_cover":
                bt_tickets = generate_graph_coverage_tickets(
                    bt_scored,
                    df,
                    main_cols,
                    bt_date,
                    pool_config=tuner_cfg.get("pool"),
                    penalty_scale=tuner_cfg.get("penalty_scale"),
                    penalty_config=tuner_cfg.get("penalty_config"),
                    cohesion_config=main_cfg,
                )
            elif tuner_cfg and tuner_cfg.get("generator") == "adaptive_pool":
                bt_tickets = generate_adaptive_pool_tickets(
                    bt_scored,
                    df,
                    main_cols,
                    bt_date,
                    pool_config=tuner_cfg.get("pool"),
                    penalty_scale=tuner_cfg.get("penalty_scale"),
                    penalty_config=tuner_cfg.get("penalty_config"),
                    cohesion_config=main_cfg,
                    force_coverage=tuner_cfg.get("force_coverage", False),
                )
            else:
                bt_tickets = generate_tickets(bt_scored, df, main_cols, bt_date,
                                              use_weights=True,
                                              seed_hot_overdue=(tuner_cfg.get("seed_hot_overdue") if tuner_cfg else False),
                                              force_coverage=(tuner_cfg.get("force_coverage") if tuner_cfg else FORCE_COVERAGE),
                                              cohesion_config=main_cfg,
                                              penalty_scale=(tuner_cfg.get("penalty_scale") if tuner_cfg else None),
                                              pool_config=(tuner_cfg.get("pool") if tuner_cfg else None),
                                              penalty_config=(tuner_cfg.get("penalty_config") if tuner_cfg else None),
                                              supp_candidates=supp_candidates,
                                              supp_min_hit=SUPP_MIN_HIT)
        if profile:
            strict = [t for t in bt_tickets if _ticket_passes_profile(t, score_map, profile)]
            relaxed = [t for t in bt_tickets if _ticket_passes_profile_relaxed(t, score_map, profile) and t not in strict]
            remainder = [t for t in bt_tickets if t not in strict and t not in relaxed]
            bt_tickets = (strict + relaxed + remainder)[:NUM_TICKETS]
        print(f"\nBacktest Target: {bt_date}")
        for i, t in enumerate(bt_tickets, 1):
            vec = _decade_vector(t)
            print(f"Ticket #{i:02d}: {t}  decades={vec}")
        show_ticket_hits(bt_draw, bt_tickets, bt_supp)

    if not df_target.empty:
        row = df_target.iloc[0]
        real_draw = [int(row[c]) for c in main_cols]
        real_supp = [int(row[c]) for c in supp_cols] if supp_cols else []
    else:
        real_draw = REAL_DRAW if (USE_REAL_DRAW_FALLBACK and REAL_DRAW) else []
        real_supp = []
        if real_draw:
            print("TARGET_DATE not found in CSV; using REAL_DRAW override.")
        else:
            print("TARGET_DATE not found in CSV; generating prediction without hit summary.")

    scored = score_numbers(
        df,
        main_cols,
        run_date,
        DEBUG_PRINT,
        show_all=(not df_target.empty) and PRINT_ALL_SCORES_WHEN_REAL,
        score_config=(tuner_cfg.get("score_config") if tuner_cfg else None),
    )
    target_supp_candidates = _supp_candidate_set(
        df, supp_cols, run_date
    ) if supp_cols and (SUPP_MIN_HIT > 0 or SUPP_SOFT_SWAP or SUPP_FOCUS_TICKETS > 0) else []
    target_learn_rows = df[df["Date"] < t_target].sort_values("Date").tail(5)
    target_learn_dates = [row["Date"] for _, row in target_learn_rows.iterrows()]
    target_profile = _learn_winner_profile(df, main_cols, target_learn_dates) if len(target_learn_dates) >= 5 else None
    target_score_map = {c.n: c for c in scored}

    if PORTFOLIO_MODE:
        tickets = generate_portfolio_tickets(scored, df, main_cols, run_date)
    else:
        if tuner_cfg and tuner_cfg.get("generator") == "portfolio":
            tickets = generate_portfolio_tickets(scored, df, main_cols, run_date)
        elif tuner_cfg and tuner_cfg.get("generator") == "pair_anchored":
            tickets = generate_pair_anchored_tickets(scored, df, main_cols, run_date, pair_weight=0.50)
        elif tuner_cfg and tuner_cfg.get("generator") == "pair_cluster":
            tickets = generate_pair_cluster_tickets(
                scored,
                df,
                main_cols,
                run_date,
                pool_config=tuner_cfg.get("pool"),
                penalty_scale=tuner_cfg.get("penalty_scale"),
                penalty_config=tuner_cfg.get("penalty_config"),
                cohesion_config=main_cfg,
            )
        elif tuner_cfg and tuner_cfg.get("generator") == "graph_cover":
            tickets = generate_graph_coverage_tickets(
                scored,
                df,
                main_cols,
                run_date,
                pool_config=tuner_cfg.get("pool"),
                penalty_scale=tuner_cfg.get("penalty_scale"),
                penalty_config=tuner_cfg.get("penalty_config"),
                cohesion_config=main_cfg,
            )
        elif tuner_cfg and tuner_cfg.get("generator") == "adaptive_pool":
            tickets = generate_adaptive_pool_tickets(
                scored,
                df,
                main_cols,
                run_date,
                pool_config=tuner_cfg.get("pool"),
                penalty_scale=tuner_cfg.get("penalty_scale"),
                penalty_config=tuner_cfg.get("penalty_config"),
                cohesion_config=main_cfg,
                force_coverage=tuner_cfg.get("force_coverage", False),
            )
        else:
            tickets = generate_tickets(scored, df, main_cols, run_date,
                                       use_weights=True,
                                       seed_hot_overdue=(tuner_cfg.get("seed_hot_overdue") if tuner_cfg else False),
                                       force_coverage=(tuner_cfg.get("force_coverage") if tuner_cfg else FORCE_COVERAGE),
                                       cohesion_config=main_cfg,
                                       penalty_scale=(tuner_cfg.get("penalty_scale") if tuner_cfg else None),
                                       pool_config=(tuner_cfg.get("pool") if tuner_cfg else None),
                                       penalty_config=(tuner_cfg.get("penalty_config") if tuner_cfg else None),
                                       supp_candidates=target_supp_candidates,
                                       supp_min_hit=SUPP_MIN_HIT)
    if target_profile and USE_PROFILE_FILTER:
        strict = [t for t in tickets if _ticket_passes_profile(t, target_score_map, target_profile)]
        relaxed = [t for t in tickets if _ticket_passes_profile_relaxed(t, target_score_map, target_profile) and t not in strict]
        remainder = [t for t in tickets if t not in strict and t not in relaxed]
        tickets = (strict + relaxed + remainder)[:NUM_TICKETS]

    mode_label = "HARD_FORCE" if FORCE_COVERAGE else "WEIGHTED"
    print(f"\n=== {mode_label} STRATEGY ===")
    print(f"Target: {run_date}")
    print(f"Tickets: {NUM_TICKETS} | Pool size: {POOL_SIZE} + mid {MID_POOL_SIZE} + cold {COLD_POOL_SIZE}")
    print(f"Decade bands: {DECADE_BANDS}")

    for i, t in enumerate(tickets, 1):
        vec = _decade_vector(t)
        print(f"Ticket #{i:02d}: {t}  decades={vec}")

    show_ticket_hits(real_draw, tickets, real_supp)
