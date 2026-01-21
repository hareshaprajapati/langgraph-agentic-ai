import sys
import os
from datetime import datetime

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# LOG_DIR = ""
# os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(
    ".",
    f"Siko_Power.log"   # single growing log file
)
CSV_PATH = "Powerball.csv"

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple
import random
import math

# ============================================================
# USER CONFIG (edit only these)
# ============================================================

N = 10
# 3 hit with without decade
TARGET_DATE = "2026-01-15"
REAL_DRAW_TARGET = [1, 2, 4, 24, 25, 27, 35]
# 4 hit without decade, with decade 3
# TARGET_DATE = "2026-01-08"
# REAL_DRAW_TARGET = [7, 15, 16, 17, 25, 26, 27]
# 3 hit with without exact decade
# TARGET_DATE = "2026-01-01"
# REAL_DRAW_TARGET = [30, 9, 7, 27, 18, 15, 29]
# Example: {1: 2, 2: 2, 3: 3, 4: 0}
DECADE_TARGET_COUNTS = {1: 3, 2: 0, 3: 3, 4: 1}

NUM_TICKETS = 10
NUMBERS_PER_TICKET = 7

MAIN_MIN = 1
MAIN_MAX = 35
POWERBALL_COL = "Powerball Number"
POWERBALL_MIN = 1
POWERBALL_MAX = 20
POWERBALL_TOP_N = 10

LOOKBACK_DAYS = 365
SEASON_WINDOW_DAYS = 7
SEASON_LOOKBACK_YEARS = 20
MIN_SEASON_SAMPLES = 50

RECENT_DRAWS_PENALTY_N = 6

# Ticket diversification
TOP_POOL = 35
OVERLAP_CAP = 5
RANDOM_SEED = 0
DEBUG_PRINT = True

# --- decade constraint strictness ---
DECADE_MODE = "soft"         # "hard" or "soft"
DECADE_MEDIAN_TOL = 1        # allow +/-1 per decade around seasonal p25..p75 (hard mode)
DECADE_SOFT_PENALTY = 0.6    # per unit distance outside tolerance (soft mode)
# Optional: exact decade counts to prefer per ticket (soft rule)

DECADE_TARGET_SOFT_PENALTY = 0.25

# Optional: verify against a known real draw (set [] to disable)
# REAL_DRAW = [30, 9, 7, 27, 18, 15, 29]
# REAL_DRAW = [7,23,29,20,11,16,17]
# ----------------------------
# Ticket generation knobs
# ----------------------------
  # generate this many valid candidate tickets, then pick best NUM_TICKETS
MAX_GEN_ATTEMPTS = 30000       # safety cap for candidate generation loop

WEIGHT_MODE = "exp"            # "exp" recommended, or "linear"
BREAKER_MODE = "hot"           # "hot" | "cold" | "mix"
# WEIGHT_TEMPERATURE = 0.55      # lower => more stacked tickets (try 0.45..0.80)
# BREAKER_PROB = 0.55            # probability per candidate ticket
# CANDIDATE_TICKET_POOL = 2500
# WEIGHT_TEMPERATURE = 0.45
# BREAKER_PROB = 0.35
# CANDIDATE_TICKET_POOL = 4000

WEIGHT_TEMPERATURE = 0.70
BREAKER_PROB = 0.30
CANDIDATE_TICKET_POOL = 8000


WEIGHT_FLOOR = 0.75            # keeps tail weights non-zero (stability)

BREAKER_ENABLED = True
BREAKER_TOP_K = 4              # breaker drawn from top-K by score
BREAKER_BOTTOM_K = 3           # and/or bottom-K of the pool (cold exploration)

BREAKER_BYPASS_DECADE_CHECK = False  # apply breaker AFTER decade checks (recommended)

TOTAL_TICKETS = 20

MODE_SPLIT = {
    "stack": 8,
    "structure": 8,
    "chaos": 4
}

MODE_PARAMS = {
    "stack": {
        "TEMP": 0.38,
        "BREAKERS": 0,
        "DECADE_MODE": "hard",
        "OVERLAP_CAP": 6
    },
    "structure": {
        "TEMP": 0.55,
        "BREAKERS": 1,
        "DECADE_MODE": "hard",
        "OVERLAP_CAP": 4
    },
    "chaos": {
        "TEMP": 0.85,
        "BREAKERS": 2,
        "DECADE_MODE": "ignore",
        "OVERLAP_CAP": 99
    }
}


# ============================================================
# INTERNALS
# ============================================================

@dataclass
class SeasonProfile:
    anchor_month: int
    anchor_day: int
    sample_years: int
    sample_draws: int
    ratio_low: float
    ratio_high: float
    ratio_ideal: float
    rank_low: int
    rank_high: int
    leader_rate: float


@dataclass
class SeasonDecadeProfile:
    anchor_month: int
    anchor_day: int
    sample_years: int
    sample_draws: int
    med: Dict[int, int]   # decade_id -> count
    p25: Dict[int, int]
    p75: Dict[int, int]


@dataclass
class CandidateScoreMain:
    n: int
    total_score: float
    freq_12mo: int
    rank_12mo: int
    ratio_12mo: float
    seasonal_count: int
    seasonal_success: int
    penalties: Dict[str, float]
    components: Dict[str, float]


# ---------- helpers ----------
def _parse_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d, dayfirst=True, errors="coerce")


def _detect_main_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if "Winning Number" in c]
    if not cols:
        raise ValueError("Could not find main-number columns like 'Winning Number 1..7' in CSV.")
    import re
    def key(c):
        m = re.search(r"(\d+)$", c.strip())
        return int(m.group(1)) if m else 999
    cols = sorted(cols, key=key)
    if len(cols) < 7:
        raise ValueError("Expected at least 7 main columns (Winning Number 1..7).")
    return cols[:7]


def _load_csv(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    if "Draw date" not in df.columns:
        raise ValueError("CSV must contain column: 'Draw date'")
    df = df.copy()
    df["Date"] = df["Draw date"].apply(_parse_date)
    df = df.dropna(subset=["Date"]).copy()

    main_cols = _detect_main_cols(df)
    for c in main_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=main_cols).copy()
    for c in main_cols:
        df[c] = df[c].astype(int)

    # enforce range 1..35
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


def _counts_single_in_window(
    df: pd.DataFrame,
    col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_val: int,
    max_val: int
) -> pd.Series:
    sub = df[(df["Date"] >= start) & (df["Date"] < end)]
    nums = pd.to_numeric(sub[col], errors="coerce")
    nums = nums[(nums >= min_val) & (nums <= max_val)]
    return nums.value_counts().reindex(range(min_val, max_val + 1), fill_value=0).sort_index()


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


# ============================================================
# DECADE DEFINITION (as you requested)
# decade 1 = 1-10, decade 2 = 11-20, decade 3 = 21-30, decade 4 = 31-35
# ============================================================

DECADE_BANDS: List[Tuple[int, int, int]] = [
    (1, 1, 10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 35),
]
DECADE_IDS = [d[0] for d in DECADE_BANDS]


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


def _decade_distance(vec: Dict[int, int], med: Dict[int, int]) -> int:
    return sum(abs(vec[d] - med[d]) for d in DECADE_IDS)


def _within_decade_band(vec: Dict[int, int], p25: Dict[int, int], p75: Dict[int, int], tol: int) -> bool:
    for d in DECADE_IDS:
        lo = max(0, p25[d] - tol)
        hi = p75[d] + tol
        if not (lo <= vec[d] <= hi):
            return False
    return True


def _decade_target_distance(vec: Dict[int, int], target: Dict[int, int]) -> int:
    return sum(abs(vec.get(d, 0) - target.get(d, 0)) for d in DECADE_IDS)


def _learn_season_decade_profile(
    df: pd.DataFrame,
    main_cols: List[str],
    target_ts: pd.Timestamp,
    season_window_days: int,
    season_lookback_years: int,
    min_samples: int
) -> SeasonDecadeProfile:
    target_d = target_ts.date()
    start_year = target_d.year - season_lookback_years
    end_year = target_d.year - 1

    vectors = []
    seen_years = set()
    draw_count = 0

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_d, y)
        win_start, win_end = _season_window_dates(anchor, season_window_days)
        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if near.empty:
            continue
        seen_years.add(y)

        for _, row in near.iterrows():
            mains = [int(row[c]) for c in main_cols]
            vectors.append(_decade_vector(mains))
            draw_count += 1

    # fallback if too few samples
    if len(vectors) < max(5, min_samples // 10):
        # neutral-ish fallback for 7 numbers
        med = {1: 2, 2: 2, 3: 2, 4: 1}  # sums to 7
        p25 = {1: 1, 2: 1, 3: 1, 4: 0}
        p75 = {1: 3, 2: 3, 3: 3, 4: 2}
        return SeasonDecadeProfile(
            anchor_month=target_d.month,
            anchor_day=target_d.day,
            sample_years=len(seen_years),
            sample_draws=draw_count,
            med=med, p25=p25, p75=p75
        )

    dfv = pd.DataFrame(vectors)  # columns are decade ids
    med = {d: int(dfv[d].median()) for d in DECADE_IDS}
    p25 = {d: int(dfv[d].quantile(0.25)) for d in DECADE_IDS}
    p75 = {d: int(dfv[d].quantile(0.75)) for d in DECADE_IDS}

    # ensure sums are close to NUMBERS_PER_TICKET (for sanity in printing)
    return SeasonDecadeProfile(
        anchor_month=target_d.month,
        anchor_day=target_d.day,
        sample_years=len(seen_years),
        sample_draws=draw_count,
        med=med, p25=p25, p75=p75
    )


# ---------- season profile (rank/ratio) ----------
def _learn_season_profile_mains(
    df: pd.DataFrame,
    main_cols: List[str],
    target_ts: pd.Timestamp,
    lookback_days: int,
    season_window_days: int,
    season_lookback_years: int,
    min_samples: int,
) -> SeasonProfile:
    target_d = target_ts.date()
    start_year = target_d.year - season_lookback_years
    end_year = target_d.year - 1

    ratios, ranks, leader_flags = [], [], []
    seen_years = set()
    draw_count = 0

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_d, y)
        win_start, win_end = _season_window_dates(anchor, season_window_days)
        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if near.empty:
            continue
        seen_years.add(y)

        for _, row in near.iterrows():
            d = row["Date"]
            mains = [int(row[c]) for c in main_cols]

            hist_train = df[df["Date"] < d]
            hist_start = d - pd.Timedelta(days=lookback_days)
            hist_counts = _counts_mains_in_window(hist_train, main_cols, hist_start, d)
            if hist_counts.empty:
                continue
            hist_max = int(hist_counts.max())
            if hist_max <= 0:
                continue
            hist_ranks = _rank_from_counts(hist_counts)

            for n in mains:
                freq = int(hist_counts.get(n, 0))
                rnk = int(hist_ranks.get(n, 999))
                ratio = (freq / hist_max) if hist_max > 0 else 0.0
                ratios.append(ratio)
                ranks.append(rnk)
                leader_flags.append(1 if rnk == 1 else 0)

            draw_count += 1

    if len(ratios) < min_samples:
        return SeasonProfile(
            anchor_month=target_d.month,
            anchor_day=target_d.day,
            sample_years=len(seen_years),
            sample_draws=draw_count,
            ratio_low=0.40,
            ratio_high=0.80,
            ratio_ideal=0.55,
            rank_low=2,
            rank_high=8,
            leader_rate=float(sum(leader_flags) / len(leader_flags)) if leader_flags else 0.0,
        )

    s_ratios = pd.Series(ratios)
    s_ranks = pd.Series(ranks)

    ratio_p25 = float(s_ratios.quantile(0.25))
    ratio_p75 = float(s_ratios.quantile(0.75))
    ratio_med = float(s_ratios.median())

    rank_p25 = float(s_ranks.quantile(0.25))
    rank_p75 = float(s_ranks.quantile(0.75))

    learned_ratio_low = max(0.0, min(1.0, ratio_p25))
    learned_ratio_high = max(0.0, min(1.0, ratio_p75))
    if learned_ratio_high < learned_ratio_low:
        learned_ratio_low, learned_ratio_high = learned_ratio_high, learned_ratio_low

    learned_rank_low = max(1, int(round(rank_p25)))
    learned_rank_high = max(learned_rank_low, int(round(rank_p75)))

    leader_rate = float(sum(leader_flags) / len(leader_flags))

    return SeasonProfile(
        anchor_month=target_d.month,
        anchor_day=target_d.day,
        sample_years=len(seen_years),
        sample_draws=draw_count,
        ratio_low=learned_ratio_low,
        ratio_high=learned_ratio_high,
        ratio_ideal=ratio_med,
        rank_low=learned_rank_low,
        rank_high=learned_rank_high,
        leader_rate=leader_rate,
    )


def _learn_season_profile_single(
    df: pd.DataFrame,
    col: str,
    target_ts: pd.Timestamp,
    lookback_days: int,
    season_window_days: int,
    season_lookback_years: int,
    min_samples: int,
    min_val: int,
    max_val: int,
) -> SeasonProfile:
    target_d = target_ts.date()
    start_year = target_d.year - season_lookback_years
    end_year = target_d.year - 1

    ratios, ranks, leader_flags = [], [], []
    seen_years = set()
    draw_count = 0

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_d, y)
        win_start, win_end = _season_window_dates(anchor, season_window_days)
        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if near.empty:
            continue
        seen_years.add(y)

        for _, row in near.iterrows():
            d = row["Date"]
            n = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(n) or not (min_val <= int(n) <= max_val):
                continue
            n = int(n)

            hist_train = df[df["Date"] < d]
            hist_start = d - pd.Timedelta(days=lookback_days)
            hist_counts = _counts_single_in_window(hist_train, col, hist_start, d, min_val, max_val)
            if hist_counts.empty:
                continue
            hist_max = int(hist_counts.max())
            if hist_max <= 0:
                continue
            hist_ranks = _rank_from_counts(hist_counts)

            freq = int(hist_counts.get(n, 0))
            rnk = int(hist_ranks.get(n, 999))
            ratio = (freq / hist_max) if hist_max > 0 else 0.0
            ratios.append(ratio)
            ranks.append(rnk)
            leader_flags.append(1 if rnk == 1 else 0)
            draw_count += 1

    if len(ratios) < min_samples:
        return SeasonProfile(
            anchor_month=target_d.month,
            anchor_day=target_d.day,
            sample_years=len(seen_years),
            sample_draws=draw_count,
            ratio_low=0.40,
            ratio_high=0.80,
            ratio_ideal=0.55,
            rank_low=2,
            rank_high=8,
            leader_rate=float(sum(leader_flags) / len(leader_flags)) if leader_flags else 0.0,
        )

    s_ratios = pd.Series(ratios)
    s_ranks = pd.Series(ranks)

    ratio_p25 = float(s_ratios.quantile(0.25))
    ratio_p75 = float(s_ratios.quantile(0.75))
    ratio_med = float(s_ratios.median())

    rank_p25 = float(s_ranks.quantile(0.25))
    rank_p75 = float(s_ranks.quantile(0.75))

    learned_ratio_low = max(0.0, min(1.0, ratio_p25))
    learned_ratio_high = max(0.0, min(1.0, ratio_p75))
    if learned_ratio_high < learned_ratio_low:
        learned_ratio_low, learned_ratio_high = learned_ratio_high, learned_ratio_low

    learned_rank_low = max(1, int(round(rank_p25)))
    learned_rank_high = max(learned_rank_low, int(round(rank_p75)))

    leader_rate = float(sum(leader_flags) / len(leader_flags)) if leader_flags else 0.0

    return SeasonProfile(
        anchor_month=target_d.month,
        anchor_day=target_d.day,
        sample_years=len(seen_years),
        sample_draws=draw_count,
        ratio_low=learned_ratio_low,
        ratio_high=learned_ratio_high,
        ratio_ideal=ratio_med,
        rank_low=learned_rank_low,
        rank_high=learned_rank_high,
        leader_rate=leader_rate,
    )


def score_main_numbers(target_date: str, csv_path: str, debug: bool):
    df, main_cols = _load_csv(csv_path)
    t = pd.Timestamp(target_date)
    if pd.isna(t):
        raise ValueError("TARGET_DATE must be parseable (YYYY-MM-DD)")

    train = df[df["Date"] < t].copy()
    if train.empty:
        raise ValueError("No historical draws before TARGET_DATE")

    season_profile = _learn_season_profile_mains(
        df=df, main_cols=main_cols, target_ts=t,
        lookback_days=LOOKBACK_DAYS,
        season_window_days=SEASON_WINDOW_DAYS,
        season_lookback_years=SEASON_LOOKBACK_YEARS,
        min_samples=MIN_SEASON_SAMPLES,
    )
    season_decades = _learn_season_decade_profile(
        df=df, main_cols=main_cols, target_ts=t,
        season_window_days=SEASON_WINDOW_DAYS,
        season_lookback_years=SEASON_LOOKBACK_YEARS,
        min_samples=MIN_SEASON_SAMPLES,
    )

    allow_leader = season_profile.leader_rate >= 0.20

    recent_start = t - pd.Timedelta(days=LOOKBACK_DAYS)
    recent_counts = _counts_mains_in_window(train, main_cols, recent_start, t)
    maxfreq = int(recent_counts.max()) if not recent_counts.empty else 0
    ranks = _rank_from_counts(recent_counts) if maxfreq > 0 else {}

    last_n_draws = train.sort_values("Date").tail(RECENT_DRAWS_PENALTY_N)
    last_n_counts = _explode_mains(last_n_draws, main_cols).value_counts()

    # seasonal count / success
    target_dt = t.date()
    start_year = target_dt.year - SEASON_LOOKBACK_YEARS
    end_year = target_dt.year - 1

    seasonal_counts = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
    seasonal_success = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_dt, y)
        win_start, win_end = _season_window_dates(anchor, SEASON_WINDOW_DAYS)
        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if near.empty:
            continue

        for _, row in near.iterrows():
            d = row["Date"]
            mains = [int(row[c]) for c in main_cols]

            for n in mains:
                if MAIN_MIN <= n <= MAIN_MAX:
                    seasonal_counts[n] += 1

            hist_train = df[df["Date"] < d]
            hist_start = d - pd.Timedelta(days=LOOKBACK_DAYS)
            hist_counts = _counts_mains_in_window(hist_train, main_cols, hist_start, d)
            if hist_counts.empty:
                continue
            hist_max = int(hist_counts.max())
            if hist_max <= 0:
                continue
            hist_ranks = _rank_from_counts(hist_counts)

            for n in mains:
                if not (MAIN_MIN <= n <= MAIN_MAX):
                    continue
                freq = int(hist_counts.get(n, 0))
                rnk = int(hist_ranks.get(n, 999))
                ratio = (freq / hist_max) if hist_max > 0 else 0.0
                if (
                    (season_profile.rank_low <= rnk <= season_profile.rank_high)
                    and (season_profile.ratio_low <= ratio <= season_profile.ratio_high)
                    and (freq >= 2)
                    and (allow_leader or rnk != 1)
                ):
                    seasonal_success[n] += 1

    scored: List[CandidateScoreMain] = []
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        freq = int(recent_counts.get(n, 0))
        rank = int(ranks.get(n, 999))
        ratio = (freq / maxfreq) if maxfreq > 0 else 0.0

        components: Dict[str, float] = {}
        penalties: Dict[str, float] = {}

        if (rank == 1) and (not allow_leader):
            penalties["leader_penalty"] = 2.0
        elif season_profile.rank_low <= rank <= season_profile.rank_high:
            width = max(1, (season_profile.rank_high - season_profile.rank_low))
            components["rank_bonus"] = 2.0 - (0.75 * (rank - season_profile.rank_low) / width)
        elif (season_profile.rank_high + 1) <= rank <= (season_profile.rank_high + 5):
            components["rank_bonus"] = 0.6
        else:
            components["rank_bonus"] = 0.0

        if ratio != 0.0:
            components["ratio_bonus"] = max(0.0, 1.5 - abs(ratio - season_profile.ratio_ideal) * 4.0)
        else:
            components["ratio_bonus"] = 0.0

        if freq <= 1:
            penalties["cold_penalty"] = 1.5
        elif freq == 2:
            penalties["low_freq_penalty"] = 0.5

        recent_hits = int(last_n_counts.get(n, 0))
        if recent_hits > 0:
            penalties["recent_repeat_penalty"] = 0.20 * recent_hits

        season_ct = int(seasonal_counts.get(n, 0))
        season_succ = int(seasonal_success.get(n, 0))
        components["seasonal_count_bonus"] = min(1.2, 0.20 * season_ct)
        components["seasonal_success_bonus"] = min(1.8, 0.45 * season_succ)

        total = sum(components.values()) - sum(penalties.values())
        scored.append(
            CandidateScoreMain(
                n=n,
                total_score=round(total, 4),
                freq_12mo=freq,
                rank_12mo=rank,
                ratio_12mo=round(ratio, 3),
                seasonal_count=season_ct,
                seasonal_success=season_succ,
                penalties=penalties,
                components=components,
            )
        )

    scored.sort(key=lambda x: x.total_score, reverse=True)

    if debug:
        print("\n=== SEASON PROFILE (rank/ratio) ===")
        print(
            f"Anchor: {season_profile.anchor_month:02}/{season_profile.anchor_day:02} | "
            f"Samples: years={season_profile.sample_years}, draws={season_profile.sample_draws}"
        )
        # print(
        #     f"Ratio band (p25..p75): {season_profile.ratio_low:.3f}..{season_profile.ratio_high:.3f} | "
        #     f"ideal={season_profile.ratio_ideal:.3f}"
        # )
        # print(
        #     f"Rank band (p25..p75): {season_profile.rank_low}..{season_profile.rank_high} | "
        #     f"leader_rate={season_profile.leader_rate:.3f} | allow_leader={allow_leader}"
        # )

        print("\n=== SEASON DECADE PROFILE (YOUR BANDS) ===")
        print(
            f"Anchor: {season_decades.anchor_month:02}/{season_decades.anchor_day:02} | "
            f"Samples: years={season_decades.sample_years}, draws={season_decades.sample_draws}"
        )
        print(f"Decade bands: {DECADE_BANDS}")
        print(f"Median decade mix: {season_decades.med}")
        print(f"P25 decade mix:    {season_decades.p25}")
        print(f"P75 decade mix:    {season_decades.p75}")

        print("\n=== SCORED NUMBERS ===")
        for c in scored[:35]:
            print(c)

    return scored, season_profile, season_decades


def score_powerball_numbers(target_date: str, csv_path: str, debug: bool):
    df = pd.read_csv(csv_path)
    if "Draw date" not in df.columns:
        raise ValueError("CSV must contain column: 'Draw date'")
    if POWERBALL_COL not in df.columns:
        raise ValueError(f"CSV must contain column: '{POWERBALL_COL}'")

    df = df.copy()
    df["Date"] = df["Draw date"].apply(_parse_date)
    df = df.dropna(subset=["Date"]).copy()
    df[POWERBALL_COL] = pd.to_numeric(df[POWERBALL_COL], errors="coerce")
    df = df.dropna(subset=[POWERBALL_COL]).copy()
    df[POWERBALL_COL] = df[POWERBALL_COL].astype(int)
    df = df[(df[POWERBALL_COL] >= POWERBALL_MIN) & (df[POWERBALL_COL] <= POWERBALL_MAX)]

    df = df.sort_values("Date").reset_index(drop=True)

    t = pd.Timestamp(target_date)
    if pd.isna(t):
        raise ValueError("TARGET_DATE must be parseable (YYYY-MM-DD)")

    train = df[df["Date"] < t].copy()
    if train.empty:
        raise ValueError("No historical draws before TARGET_DATE for Powerball")

    season_profile = _learn_season_profile_single(
        df=df,
        col=POWERBALL_COL,
        target_ts=t,
        lookback_days=LOOKBACK_DAYS,
        season_window_days=SEASON_WINDOW_DAYS,
        season_lookback_years=SEASON_LOOKBACK_YEARS,
        min_samples=MIN_SEASON_SAMPLES,
        min_val=POWERBALL_MIN,
        max_val=POWERBALL_MAX,
    )

    allow_leader = season_profile.leader_rate >= 0.20

    recent_start = t - pd.Timedelta(days=LOOKBACK_DAYS)
    recent_counts = _counts_single_in_window(train, POWERBALL_COL, recent_start, t, POWERBALL_MIN, POWERBALL_MAX)
    maxfreq = int(recent_counts.max()) if not recent_counts.empty else 0
    ranks = _rank_from_counts(recent_counts) if maxfreq > 0 else {}

    last_n_draws = train.sort_values("Date").tail(RECENT_DRAWS_PENALTY_N)
    last_n_counts = pd.to_numeric(last_n_draws[POWERBALL_COL], errors="coerce").value_counts()

    target_dt = t.date()
    start_year = target_dt.year - SEASON_LOOKBACK_YEARS
    end_year = target_dt.year - 1

    seasonal_counts = {n: 0 for n in range(POWERBALL_MIN, POWERBALL_MAX + 1)}
    seasonal_success = {n: 0 for n in range(POWERBALL_MIN, POWERBALL_MAX + 1)}

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_dt, y)
        win_start, win_end = _season_window_dates(anchor, SEASON_WINDOW_DAYS)
        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if near.empty:
            continue

        for _, row in near.iterrows():
            d = row["Date"]
            n = int(row[POWERBALL_COL])
            if not (POWERBALL_MIN <= n <= POWERBALL_MAX):
                continue
            seasonal_counts[n] += 1

            hist_train = df[df["Date"] < d]
            hist_start = d - pd.Timedelta(days=LOOKBACK_DAYS)
            hist_counts = _counts_single_in_window(hist_train, POWERBALL_COL, hist_start, d, POWERBALL_MIN, POWERBALL_MAX)
            if hist_counts.empty:
                continue
            hist_max = int(hist_counts.max())
            if hist_max <= 0:
                continue
            hist_ranks = _rank_from_counts(hist_counts)

            freq = int(hist_counts.get(n, 0))
            rnk = int(hist_ranks.get(n, 999))
            ratio = (freq / hist_max) if hist_max > 0 else 0.0
            if (
                (season_profile.rank_low <= rnk <= season_profile.rank_high)
                and (season_profile.ratio_low <= ratio <= season_profile.ratio_high)
                and (freq >= 2)
                and (allow_leader or rnk != 1)
            ):
                seasonal_success[n] += 1

    scored: List[CandidateScoreMain] = []
    for n in range(POWERBALL_MIN, POWERBALL_MAX + 1):
        freq = int(recent_counts.get(n, 0))
        rank = int(ranks.get(n, 999))
        ratio = (freq / maxfreq) if maxfreq > 0 else 0.0

        components: Dict[str, float] = {}
        penalties: Dict[str, float] = {}

        if (rank == 1) and (not allow_leader):
            penalties["leader_penalty"] = 2.0
        elif season_profile.rank_low <= rank <= season_profile.rank_high:
            width = max(1, (season_profile.rank_high - season_profile.rank_low))
            components["rank_bonus"] = 2.0 - (0.75 * (rank - season_profile.rank_low) / width)
        elif (season_profile.rank_high + 1) <= rank <= (season_profile.rank_high + 5):
            components["rank_bonus"] = 0.6
        else:
            components["rank_bonus"] = 0.0

        if ratio != 0.0:
            components["ratio_bonus"] = max(0.0, 1.5 - abs(ratio - season_profile.ratio_ideal) * 4.0)
        else:
            components["ratio_bonus"] = 0.0

        if freq <= 1:
            penalties["cold_penalty"] = 1.5
        elif freq == 2:
            penalties["low_freq_penalty"] = 0.5

        recent_hits = int(last_n_counts.get(n, 0))
        if recent_hits > 0:
            penalties["recent_repeat_penalty"] = 0.20 * recent_hits

        season_ct = int(seasonal_counts.get(n, 0))
        season_succ = int(seasonal_success.get(n, 0))
        components["seasonal_count_bonus"] = min(1.2, 0.20 * season_ct)
        components["seasonal_success_bonus"] = min(1.8, 0.45 * season_succ)

        total = sum(components.values()) - sum(penalties.values())
        scored.append(
            CandidateScoreMain(
                n=n,
                total_score=round(total, 4),
                freq_12mo=freq,
                rank_12mo=rank,
                ratio_12mo=round(ratio, 3),
                seasonal_count=season_ct,
                seasonal_success=season_succ,
                penalties=penalties,
                components=components,
            )
        )

    scored.sort(key=lambda x: x.total_score, reverse=True)

    if debug:
        print("\n=== POWERBALL SEASON PROFILE (rank/ratio) ===")
        print(
            f"Anchor: {season_profile.anchor_month:02}/{season_profile.anchor_day:02} | "
            f"Samples: years={season_profile.sample_years}, draws={season_profile.sample_draws}"
        )
        print(
            f"Ratio band (p25..p75): {season_profile.ratio_low:.3f}..{season_profile.ratio_high:.3f} | "
            f"ideal={season_profile.ratio_ideal:.3f}"
        )
        print(
            f"Rank band (p25..p75): {season_profile.rank_low}..{season_profile.rank_high} | "
            f"leader_rate={season_profile.leader_rate:.3f} | allow_leader={allow_leader}"
        )

        print("\n=== TOP SCORED POWERBALL NUMBERS ===")
        for c in scored[:min(10, len(scored))]:
            print(c)

    return scored, season_profile


import math
import random
from typing import List, Dict, Tuple, Set, Optional

def _weighted_sample_no_replace(items: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    """
    Weighted sampling without replacement using Efraimidis-Spirakis keys.
    Stable and fast enough for small pools.
    """
    if k > len(items):
        raise ValueError(f"_weighted_sample_no_replace: k={k} > len(items)={len(items)}")
    keyed = []
    for x, w in zip(items, weights):
        w = float(w)
        if w <= 0:
            w = 1e-12
        u = rng.random()
        # log(u)/w: larger (less negative) means more likely selected
        keyed.append((math.log(u) / w, x))
    keyed.sort(reverse=True)
    return [x for _, x in keyed[:k]]


def _ticket_overlap_ok(candidate: List[int], chosen: List[List[int]], overlap_cap: int) -> bool:
    s = set(candidate)
    for t in chosen:
        if len(s.intersection(t)) > overlap_cap:
            return False
    return True


def generate_tickets(scored, season_decades, band_quota_counts=None, band_quota_strict=False, band_min_requirements=None):
    """
    Production ticket generator:
      1) Sample MANY valid candidate tickets from the score pool (stacking enabled)
      2) Select the BEST NUM_TICKETS using greedy optimization under overlap cap

    This fixes the failure mode where good 4-hit ingredients are spread across different tickets.
    """
    if not scored:
        raise ValueError("generate_tickets(): scored list is empty")

    rng = random.Random(RANDOM_SEED)
    rank_map = {c.n: i + 1 for i, c in enumerate(scored)}

    # --- pool ---
    pool = scored[:min(TOP_POOL, len(scored))]
    if len(pool) < NUMBERS_PER_TICKET:
        raise RuntimeError(
            f"Pool too small: pool_size={len(pool)} TOP_POOL={TOP_POOL} "
            f"NUMBERS_PER_TICKET={NUMBERS_PER_TICKET}"
        )

    items = [c.n for c in pool]
    score_map = {c.n: float(c.total_score) for c in pool}
    min_score = min(float(c.total_score) for c in pool)

    # --- weights ---
    if WEIGHT_MODE == "exp":
        temp = max(1e-6, float(WEIGHT_TEMPERATURE))
        weights = [math.exp((float(c.total_score) - min_score) / temp) + float(WEIGHT_FLOOR) for c in pool]
    else:
        weights = [(float(c.total_score) - min_score) + float(WEIGHT_FLOOR) for c in pool]
        weights = [w if w > 0 else 1e-12 for w in weights]

    band_pools = None
    if band_quota_counts:
        band_pools = {"B_LEADER": {"items": [], "weights": []},
                      "A_CORE": {"items": [], "weights": []},
                      "C_TAIL": {"items": [], "weights": []},
                      "OTHER": {"items": [], "weights": []}}
        for n, w in zip(items, weights):
            rnk = rank_map.get(n, 9999)
            band = _band_of_rank(rnk)
            if band in band_pools:
                band_pools[band]["items"].append(n)
                if band == "A_CORE":
                    band_pools[band]["weights"].append(1.0)
                else:
                    band_pools[band]["weights"].append(w)

    # --- decade helpers: use your existing ones if present ---
    # We expect season_decades to provide med/p25/p75 dicts like {decade_id: count}
    def decade_vector(nums: List[int]) -> Dict[int, int]:
        # If your code already has _decade_vector, use it by name; otherwise fallback:
        if "_decade_vector" in globals():
            return globals()["_decade_vector"](nums)
        # fallback assumes DECADE_BANDS = [(id,lo,hi),...]
        v = {k: 0 for k in season_decades.med.keys()}
        for n in nums:
            did = None
            for band_id, lo, hi in DECADE_BANDS:
                if lo <= n <= hi:
                    did = band_id
                    break
            if did in v:
                v[did] += 1
        return v

    def decade_distance(vec: Dict[int, int], med: Dict[int, int]) -> int:
        if "_decade_distance" in globals():
            return globals()["_decade_distance"](vec, med)
        return sum(abs(int(vec.get(d, 0)) - int(med.get(d, 0))) for d in med.keys())

    def within_decade_band(vec: Dict[int, int], p25: Dict[int, int], p75: Dict[int, int], tol: int) -> bool:
        if "_within_decade_band" in globals():
            return globals()["_within_decade_band"](vec, p25, p75, tol=tol)
        for d in p25.keys():
            lo = max(0, int(p25[d]) - tol)
            hi = int(p75[d]) + tol
            if not (lo <= int(vec.get(d, 0)) <= hi):
                return False
        return True

    # optional: if you have DECADE_TARGET_COUNTS logic in your file
    def decade_target_distance(vec: Dict[int, int], target: Dict[int, int]) -> int:
        if "_decade_target_distance" in globals():
            return globals()["_decade_target_distance"](vec, target)
        return sum(abs(int(vec.get(d, 0)) - int(target.get(d, 0))) for d in target.keys())

    # --- breaker pools ---
    top_k = max(1, min(int(BREAKER_TOP_K), len(pool)))
    bot_k = max(1, min(int(BREAKER_BOTTOM_K), len(pool)))
    breaker_hot = [c.n for c in pool[:top_k]]
    breaker_cold = [c.n for c in pool[-bot_k:]]

    def apply_breaker(pick_sorted: List[int]) -> List[int]:
        if not BREAKER_ENABLED:
            return pick_sorted
        if rng.random() >= float(BREAKER_PROB):
            return pick_sorted

        pick_set = set(pick_sorted)

        mode = BREAKER_MODE
        if mode == "mix":
            mode = "hot" if rng.random() < 0.70 else "cold"

        candidates = breaker_hot if mode == "hot" else breaker_cold
        candidates = [n for n in candidates if n not in pick_set]
        if not candidates:
            return pick_sorted

        out = pick_sorted[:]
        out[rng.randrange(len(out))] = rng.choice(candidates)
        out = sorted(set(out))

        # refill if uniqueness shrink happened
        while len(out) < NUMBERS_PER_TICKET:
            add = rng.choice(breaker_hot)
            if add not in out:
                out.append(add)
                out.sort()

        return out

    # --- generate many candidates ---
    candidates: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = set()
    attempts = 0

    needed = int(CANDIDATE_TICKET_POOL)
    max_attempts = int(MAX_GEN_ATTEMPTS)

    while len(candidates) < needed and attempts < max_attempts:
        attempts += 1

        if band_quota_counts and band_pools:
            pick = []
            valid = True
            for band, count in band_quota_counts.items():
                cnt = int(count)
                if cnt <= 0:
                    continue
                bitems = band_pools.get(band, {}).get("items", [])
                bweights = band_pools.get(band, {}).get("weights", [])
                if len(bitems) < cnt:
                    valid = False
                    break
                pick.extend(_weighted_sample_no_replace(bitems, bweights, cnt, rng))
            if not valid or len(set(pick)) != NUMBERS_PER_TICKET:
                continue
            pick = sorted(pick)
        else:
            pick = _weighted_sample_no_replace(items, weights, NUMBERS_PER_TICKET, rng)
        pick = sorted(pick)
        tkey = tuple(pick)
        if tkey in seen:
            continue

        vec = decade_vector(pick)
        dist = decade_distance(vec, season_decades.med)
        in_band = within_decade_band(vec, season_decades.p25, season_decades.p75, tol=DECADE_MEDIAN_TOL)

        if DECADE_MODE == "hard":
            if not in_band:
                continue
            if dist > (2 * DECADE_MEDIAN_TOL):
                continue
        else:
            # soft: probabilistic rejection (keep your existing constants if present)
            soft_pen = globals().get("DECADE_SOFT_PENALTY", 1.0)
            reject_prob = min(0.85, dist * float(soft_pen) * 0.15)
            if rng.random() < reject_prob:
                continue

        # optional target decade counts (if you use it)
        if "DECADE_TARGET_COUNTS" in globals() and globals()["DECADE_TARGET_COUNTS"] is not None:
            target = globals()["DECADE_TARGET_COUNTS"]
            tdist = decade_target_distance(vec, target)
            if tdist > 0:
                tsoft = globals().get("DECADE_TARGET_SOFT_PENALTY", 0.25)
                if rng.random() < min(0.90, tdist * float(tsoft)):
                    continue

        # breaker after decade acceptance
        if BREAKER_ENABLED and BREAKER_BYPASS_DECADE_CHECK:
            pick = apply_breaker(pick)

        counts = None
        if band_quota_counts or band_min_requirements:
            counts = {"A_CORE": 0, "B_LEADER": 0, "C_TAIL": 0, "OTHER": 0}
            for n in pick:
                rnk = rank_map.get(n, 9999)
                band = _band_of_rank(rnk)
                counts[band] += 1

        if band_min_requirements and counts:
            meets_min = True
            for k, v in band_min_requirements.items():
                if counts.get(k, 0) < int(v):
                    meets_min = False
                    break
            if not meets_min:
                continue

        if band_quota_counts and counts:
            dist = sum(
                abs(int(counts.get(k, 0)) - int(band_quota_counts.get(k, 0)))
                for k in counts
            )
            if dist:
                if band_quota_strict:
                    continue
                reject_prob = min(0.90, dist * 0.35)
                if rng.random() < reject_prob:
                    continue

        # validate
        if len(set(pick)) != NUMBERS_PER_TICKET:
            continue

        seen.add(tuple(pick))
        candidates.append(pick)

    if len(candidates) < max(NUM_TICKETS, 50):
        raise RuntimeError(
            f"generate_tickets(): only {len(candidates)} candidates after {attempts} attempts. "
            f"Try DECADE_MODE='soft', increase DECADE_MEDIAN_TOL, or reduce constraints."
        )

        # --- choose best NUM_TICKETS (REGIME-AWARE) ---
    def ticket_value(t: List[int]) -> float:
        return float(sum(score_map.get(n, 0.0) for n in t))

    # Signal regime detection (uses ONLY scored list - no real draw)
    _scores = [float(c.total_score) for c in scored[:min(25, len(scored))]]
    if len(_scores) >= 20 and sum(_scores[:20]) > 0:
        dominance = sum(_scores[:5]) / sum(_scores[:20])
    elif _scores and sum(_scores) > 0:
        dominance = sum(_scores[:min(5, len(_scores))]) / sum(_scores)
    else:
        dominance = 0.0

    DOMINANCE_THRESHOLD = 0.38
    signal_regime = "CONCENTRATED" if dominance >= DOMINANCE_THRESHOLD else "FLAT"
    print(f"SIGNAL REGIME: {signal_regime} (dominance={dominance:.3f}, threshold={DOMINANCE_THRESHOLD})")

    candidates.sort(key=ticket_value, reverse=True)

    tickets: List[List[int]] = []

    if signal_regime == "CONCENTRATED":
        for t in candidates:
            if _ticket_overlap_ok(t, tickets, OVERLAP_CAP):
                tickets.append(t)
                if len(tickets) >= NUM_TICKETS:
                    break
    else:
        num_use = {}
        pair_use = {}

        MAX_PER_NUMBER = 6
        NUM_PENALTY = 0.25
        PAIR_PENALTY = 0.12

        remaining = list(candidates)
        while remaining and len(tickets) < NUM_TICKETS:
            best_idx = None
            best_score = -1e18

            for idx, t in enumerate(remaining):
                if not _ticket_overlap_ok(t, tickets, OVERLAP_CAP):
                    continue
                base = ticket_value(t)

                num_pen = 0.0
                for n in t:
                    u = num_use.get(n, 0)
                    num_pen += u
                    if u >= MAX_PER_NUMBER:
                        num_pen += 3.0 * (u - MAX_PER_NUMBER + 1)

                pair_pen = 0.0
                tt = sorted(t)
                for a, b in combinations(tt, 2):
                    pair_pen += pair_use.get((a, b), 0)

                adj = base - NUM_PENALTY * num_pen - PAIR_PENALTY * pair_pen
                if adj > best_score:
                    best_score = adj
                    best_idx = idx

            if best_idx is None:
                break

            chosen = remaining.pop(best_idx)
            tickets.append(chosen)
            for n in chosen:
                num_use[n] = num_use.get(n, 0) + 1
            chs = sorted(chosen)
            for a, b in combinations(chs, 2):
                pair_use[(a, b)] = pair_use.get((a, b), 0) + 1

    return tickets


def show_ticket_hits(real_draw: List[int], tickets: List[List[int]]):
    if not real_draw:
        return
    rd = sorted(real_draw)
    rd_set = set(rd)
    print("\n=== REAL DRAW HIT SUMMARY ===")
    print(f"REAL_DRAW: {rd}")

    hit_counts: Dict[int, int] = {}
    for i, t in enumerate(tickets, 1):
        hits = sorted(set(t).intersection(rd_set))
        hit_n = len(hits)
        hit_counts[hit_n] = hit_counts.get(hit_n, 0) + 1
        print(f"Ticket #{i:02d}: hits={hit_n} nums={hits}")

    print("Hit distribution:", dict(sorted(hit_counts.items())))

from collections import Counter
from itertools import combinations

def extract_pairs_triplets(draws, top_pairs=30, top_triplets=10):
    pair_cnt = Counter()
    trip_cnt = Counter()

    for d in draws:
        for p in combinations(sorted(d), 2):
            pair_cnt[p] += 1
        for t in combinations(sorted(d), 3):
            trip_cnt[t] += 1

    return (
        set(p for p,_ in pair_cnt.most_common(top_pairs)),
        set(t for t,_ in trip_cnt.most_common(top_triplets))
    )


def generate_mode_tickets(
    scored, season_decades, count,
    temp, breakers, decade_mode,
    overlap_cap, pairs, triplets
):
    rng = random.Random(RANDOM_SEED)
    pool = scored[:TOP_POOL]
    items = [c.n for c in pool]
    min_s = min(c.total_score for c in pool)

    weights = [
        math.exp((c.total_score - min_s) / temp) + 0.6
        for c in pool
    ]

    tickets = []
    seen = set()

    def valid_structure(t):
        s = set(t)
        return any(set(p).issubset(s) for p in pairs) or \
               any(set(tr).issubset(s) for tr in triplets)

    attempts = 0
    while len(tickets) < count and attempts < 40000:
        attempts += 1
        pick = sorted(_weighted_sample_no_replace(items, weights, NUMBERS_PER_TICKET, rng))

        if tuple(pick) in seen:
            continue

        # decade control
        if decade_mode != "ignore":
            vec = _decade_vector(pick)
            if not _within_decade_band(vec, season_decades.p25, season_decades.p75, DECADE_MEDIAN_TOL):
                continue

        # enforce structure
        if not valid_structure(pick):
            continue

        # breakers
        for _ in range(breakers):
            outlaw = rng.choice(pool[:3] + pool[-3:])
            pick[rng.randrange(len(pick))] = outlaw.n
            pick = sorted(set(pick))
            if len(pick) < NUMBERS_PER_TICKET:
                continue

        if any(len(set(pick)&set(t)) > overlap_cap for t in tickets):
            continue

        tickets.append(pick)
        seen.add(tuple(pick))

    return tickets

from dataclasses import dataclass
from typing import List, Dict, Tuple


# ----- CONFIG (LOCKED DEFAULTS) -----
BAND_B_LEADER = (1, 5)
BAND_A_CORE   = (6, 22)
BAND_C_TAIL   = (23, 27)


def _band_of_rank(rank: int,
                  band_b: Tuple[int, int] = BAND_B_LEADER,
                  band_a: Tuple[int, int] = BAND_A_CORE,
                  band_c: Tuple[int, int] = BAND_C_TAIL) -> str:
    if band_b[0] <= rank <= band_b[1]:
        return "B_LEADER"
    if band_a[0] <= rank <= band_a[1]:
        return "A_CORE"
    if band_c[0] <= rank <= band_c[1]:
        return "C_TAIL"
    return "OTHER"


@dataclass
class DrawBandStats:
    date: str
    counts: Dict[str, int]
    has_leader: bool
    has_tail: bool


def build_winner_rank_score_table_block(
    target_date: str,
    real_draw: List[int],
    scored: List["CandidateScoreMain"],
    band_b: Tuple[int, int] = BAND_B_LEADER,
    band_a: Tuple[int, int] = BAND_A_CORE,
    band_c: Tuple[int, int] = BAND_C_TAIL,
) -> Tuple[str, DrawBandStats]:
    """
    Returns:
      (1) formatted per-date table block (exact format)
      (2) per-date band stats for the end summary
    """
    # number -> (rank, score) where rank is position in scored list (1-indexed)
    rank_map: Dict[int, Tuple[int, float]] = {
        int(c.n): (i, float(c.total_score))
        for i, c in enumerate(scored, 1)
    }

    counts = {"A_CORE": 0, "B_LEADER": 0, "C_TAIL": 0, "OTHER": 0}

    lines = []
    lines.append("=== WINNER RANK+SCORE TABLE (by REAL_DRAW, based on SCORED list) ===")
    lines.append(f"Target: {target_date} | REAL_DRAW={sorted(real_draw)}")
    lines.append(f"Bands: B_LEADER={band_b}, A_CORE={band_a}, C_TAIL={band_c}")
    lines.append("n | rank | score | band")
    lines.append("--+------+-------+--------")

    for n in sorted(real_draw):
        if n in rank_map:
            rnk, sc = rank_map[n]
        else:
            rnk, sc = 999, 0.0  # winner not present in scored list
        band = _band_of_rank(rnk, band_b=band_b, band_a=band_a, band_c=band_c)
        counts[band] += 1
        lines.append(f"{n:>2} | {rnk:>4} | {sc:>5.4f} | {band}")

    has_leader = counts["B_LEADER"] > 0
    has_tail = counts["C_TAIL"] > 0

    return "\n".join(lines), DrawBandStats(
        date=target_date,
        counts=counts,
        has_leader=has_leader,
        has_tail=has_tail
    )


def collect_winner_tables_and_stats(
    blocks: List[str],
    stats: List[DrawBandStats],
    target_date: str,
    real_draw: List[int],
    scored: List["CandidateScoreMain"],
    band_b: Tuple[int, int] = BAND_B_LEADER,
    band_a: Tuple[int, int] = BAND_A_CORE,
    band_c: Tuple[int, int] = BAND_C_TAIL,
) -> None:
    """
    Call inside each backtest iteration AFTER `scored` exists.
    Stores block+stats; prints nothing.
    """
    block, st = build_winner_rank_score_table_block(
        target_date=target_date,
        real_draw=real_draw,
        scored=scored,
        band_b=band_b,
        band_a=band_a,
        band_c=band_c,
    )
    blocks.append(block)
    stats.append(st)


def print_all_winner_tables_at_end(blocks: List[str]) -> None:
    print("\n" + "=" * 78)
    print("=== ALL WINNER RANK+SCORE TABLES (GROUPED AT END) ===")
    print("=" * 78)
    for b in blocks:
        print("\n" + b)
        print("-" * 78)


def print_band_summary_at_end(
    stats: List[DrawBandStats],
    band_b: Tuple[int, int] = BAND_B_LEADER,
    band_a: Tuple[int, int] = BAND_A_CORE,
    band_c: Tuple[int, int] = BAND_C_TAIL,
    print_rank_sample_sorted: bool = False,  # keep False to avoid noise unless you want it
) -> None:
    if not stats:
        print("\n=== BAND SUMMARY (ACROSS BACKTEST DRAWS) ===")
        print("No backtest draws were collected.")
        return

    total_counts = {"A_CORE": 0, "B_LEADER": 0, "C_TAIL": 0, "OTHER": 0}
    leader_draws = 0
    tail_draws = 0
    both_draws = 0

    for s in stats:
        for k in total_counts:
            total_counts[k] += int(s.counts.get(k, 0))
        if s.has_leader:
            leader_draws += 1
        if s.has_tail:
            tail_draws += 1
        if s.has_leader and s.has_tail:
            both_draws += 1

    n_draws = len(stats)
    n_winners = sum(total_counts.values())

    print("\n" + "=" * 78)
    print("=== BAND SUMMARY (ACROSS BACKTEST DRAWS) ===")
    print(f"Draws analyzed: {n_draws} | Winner numbers total: {n_winners}")
    print(f"Bands: B_LEADER={band_b}, A_CORE={band_a}, C_TAIL={band_c}\n")

    print("Total winner-number counts by band:")
    for k, v in total_counts.items():
        pct = (100.0 * v / n_winners) if n_winners else 0.0
        print(f"  {k:8s}: {v:>3}  ({pct:>5.1f}%)")

    print("\nPer-draw presence:")
    print(f"  Draws with >=1 leader-band winner: {leader_draws}/{n_draws} ({100.0*leader_draws/n_draws:.1f}%)")
    print(f"  Draws with >=1 tail-band winner:   {tail_draws}/{n_draws} ({100.0*tail_draws/n_draws:.1f}%)")
    print(f"  Draws with BOTH leader+tail:       {both_draws}/{n_draws} ({100.0*both_draws/n_draws:.1f}%)")


def band_quota_from_stats(stats: List[DrawBandStats]) -> Dict[str, int] | None:
    if not stats:
        return None
    sums = {"B_LEADER": 0, "A_CORE": 0, "C_TAIL": 0, "OTHER": 0}
    for s in stats:
        sums["B_LEADER"] += int(s.counts.get("B_LEADER", 0))
        sums["A_CORE"] += int(s.counts.get("A_CORE", 0))
        sums["C_TAIL"] += int(s.counts.get("C_TAIL", 0))
        sums["OTHER"] += int(s.counts.get("OTHER", 0))

    n = len(stats)
    avgs = {k: (sums[k] / n) for k in sums}
    floors = {k: int(math.floor(avgs[k])) for k in avgs}
    remainder = max(0, int(NUMBERS_PER_TICKET) - sum(floors.values()))
    frac_order = sorted(avgs.keys(), key=lambda k: (avgs[k] - floors[k]), reverse=True)
    for k in frac_order:
        if remainder <= 0:
            break
        floors[k] += 1
        remainder -= 1

    total = sum(floors.values())
    if total != NUMBERS_PER_TICKET:
        k = max(floors, key=lambda x: floors[x])
        floors[k] += (NUMBERS_PER_TICKET - total)
    return floors

from datetime import datetime
from typing import List, Tuple, Dict

def print_date_by_date_band_counts_ascending(band_stats):
    """
    Prints date-by-date band counts in ascending order.

    Expected DrawBandStats shape (as per your traceback):
      - s.date
      - s.counts  (dict holding band counts)
      - s.has_leader (optional)
      - s.has_tail   (optional)

    bands expected in s.counts:
      B_LEADER, A_CORE, C_TAIL, OTHER
    """

    # Sort by date asc (YYYY-MM-DD sorts correctly as string)
    stats_sorted = sorted(band_stats, key=lambda s: str(s.date))

    def get_count(s, key, default=0):
        c = getattr(s, "counts", None)
        if isinstance(c, dict):
            # try exact key first
            if key in c:
                return int(c[key])
            # fallback: accept lowercase keys
            k2 = key.lower()
            if k2 in c:
                return int(c[k2])
        return int(default)

    print()
    print("Date        | B_LEADER | A_CORE | C_TAIL | OTHER")
    print("------------+----------+--------+--------+------")

    for s in stats_sorted:
        b = get_count(s, "B_LEADER")
        a = get_count(s, "A_CORE")
        t = get_count(s, "C_TAIL")
        o = get_count(s, "OTHER")

        print(
            f"{str(s.date):<12}|"
            f"{b:>10} |"
            f"{a:>7} |"
            f"{t:>7} |"
            f"{o:>6}"
        )

    print()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, main_cols = _load_csv(CSV_PATH)


    df_bt = df.sort_values("Date", ascending=False).head(N)
    results = []



    # backtest
    winner_blocks = []
    band_stats = []
    for i, (_, row) in enumerate(df_bt.iloc[::-1].iterrows(), 1):
        target_date = row["Date"].strftime("%Y-%m-%d")  # format expected by algo
        real_draw = [int(row[c]) for c in main_cols]  # ONLY 7 winning numbers

        print("\n" + "=" * 70)
        print(f"BACKTEST #{i}  TARGET_DATE={target_date}")
        print(f"REAL_DRAW={real_draw}  PB={int(row[POWERBALL_COL])}")

        try:

            scored, season_profile, season_decades = score_main_numbers(
                target_date=target_date,
                csv_path=CSV_PATH,
                debug=DEBUG_PRINT,
            )

            tickets = generate_tickets(scored, season_decades)

            collect_winner_tables_and_stats(
                blocks=winner_blocks,
                stats=band_stats,
                target_date=target_date,
                real_draw=real_draw,
                scored=scored
            )

            # pb_scored, pb_season_profile = score_powerball_numbers(
            #     target_date=target_date,
            #     csv_path=CSV_PATH,
            #     debug=DEBUG_PRINT,
            # )
            # top_pb_scored = pb_scored[:POWERBALL_TOP_N]

            print("\n=== FINAL TICKETS (season-aware decades) ===")
            print(f"Target: {target_date} | Tickets: {NUM_TICKETS} | Pool: top{TOP_POOL} | Overlap cap: {OVERLAP_CAP}")
            print(f"Decade mode: {DECADE_MODE} | tol={DECADE_MEDIAN_TOL}")
            print(f"Decade bands: {DECADE_BANDS}")
            if DECADE_TARGET_COUNTS is not None:
                print(f"Target decade counts (soft): {DECADE_TARGET_COUNTS}")
            print(f"Season median decades: {season_decades.med}")
            print(f"Season p25 decades:    {season_decades.p25}")
            print(f"Season p75 decades:    {season_decades.p75}")
            # if top_pb_scored:
            #     print(f"Top {POWERBALL_TOP_N} Powerball numbers (with scores):")
            #     for c in top_pb_scored:
            #         print(f"  PB {c.n:02d} score={c.total_score} freq_12mo={c.freq_12mo} rank_12mo={c.rank_12mo} ratio_12mo={c.ratio_12mo}")

            for i, t in enumerate(tickets, 1):
                vec = _decade_vector(t)
                print(f"Ticket #{i:02d}: {t}  decades={vec}")

            show_ticket_hits(real_draw, tickets)


            def hit_summary(tickets, real):
                real = set(real)
                hits = [len(set(t) & real) for t in tickets]
                return {
                    "max": max(hits) if hits else 0,
                    "n5": sum(h >= 5 for h in hits),
                    "n4": sum(h >= 4 for h in hits),
                    "hits": hits,
                }
            if real_draw:
                s = hit_summary(tickets, real_draw)
                print("\n=== HIT METRICS ===")
                print("max_hit =", s["max"], "| n>=5 =", s["n5"], "| n>=4 =", s["n4"])
                print("hits_per_ticket =", s["hits"])

        except Exception as e:
            print(f"ERROR on {target_date}: {e}")
            results.append({"date": target_date, "error": str(e)})

    print("\n" + "=" * 70)
    print_all_winner_tables_at_end(winner_blocks)
    print_date_by_date_band_counts_ascending(band_stats)

    print_band_summary_at_end(band_stats)
    print("\n" + "=" * 70)


    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    for r in results:
        print(r)
    print("\n Current draw prediction " + "=" * 70)
    # current draw prediction
    target_band_quota = band_quota_from_stats(band_stats)
    if target_band_quota:
        print(f"BAND_QUOTA (target from backtest): {target_band_quota}")
    scored, season_profile, season_decades = score_main_numbers(
        target_date=TARGET_DATE,
        csv_path=CSV_PATH,
        debug=DEBUG_PRINT,
    )

    tickets = generate_tickets(
        scored,
        season_decades,
        band_quota_counts=target_band_quota,
        band_quota_strict=False
    )
    # pb_scored, pb_season_profile = score_powerball_numbers(
    #     target_date=TARGET_DATE,
    #     csv_path=CSV_PATH,
    #     debug=DEBUG_PRINT,
    # )
    # top_pb_scored = pb_scored[:POWERBALL_TOP_N]

    print("\n=== FINAL TICKETS (season-aware decades) ===")
    print(f"Target: {TARGET_DATE} | Tickets: {NUM_TICKETS} | Pool: top{TOP_POOL} | Overlap cap: {OVERLAP_CAP}")
    print(f"Decade mode: {DECADE_MODE} | tol={DECADE_MEDIAN_TOL}")
    print(f"Decade bands: {DECADE_BANDS}")
    if DECADE_TARGET_COUNTS is not None:
        print(f"Target decade counts (soft): {DECADE_TARGET_COUNTS}")
    print(f"Season median decades: {season_decades.med}")
    print(f"Season p25 decades:    {season_decades.p25}")
    print(f"Season p75 decades:    {season_decades.p75}")

    for i, t in enumerate(tickets, 1):
        vec = _decade_vector(t)
        print(f"Ticket #{i:02d}: {t}  decades={vec}")
    show_ticket_hits(REAL_DRAW_TARGET, tickets)
    # if top_pb_scored:
    #     print(f"Top {POWERBALL_TOP_N} Powerball numbers (with scores):")
    #     for c in top_pb_scored:
    #         print(f"  PB {c.n:02d} score={c.total_score} freq_12mo={c.freq_12mo} rank_12mo={c.rank_12mo} ratio_12mo={c.ratio_12mo}")

    # scored, season_profile, season_decades = score_main_numbers(
    #     target_date=TARGET_DATE,
    #     csv_path=CSV_PATH,
    #     debug=DEBUG_PRINT,
    # )
    #
    # tickets = generate_tickets(scored, season_decades)
    # pb_scored, pb_season_profile = score_powerball_numbers(
    #     target_date=TARGET_DATE,
    #     csv_path=CSV_PATH,
    #     debug=DEBUG_PRINT,
    # )
    # top_pb_scored = pb_scored[:POWERBALL_TOP_N]
    #
    # print("\n=== FINAL TICKETS (season-aware decades) ===")
    # print(f"Target: {TARGET_DATE} | Tickets: {NUM_TICKETS} | Pool: top{TOP_POOL} | Overlap cap: {OVERLAP_CAP}")
    # print(f"Decade mode: {DECADE_MODE} | tol={DECADE_MEDIAN_TOL}")
    # print(f"Decade bands: {DECADE_BANDS}")
    # if DECADE_TARGET_COUNTS is not None:
    #     print(f"Target decade counts (soft): {DECADE_TARGET_COUNTS}")
    # print(f"Season median decades: {season_decades.med}")
    # print(f"Season p25 decades:    {season_decades.p25}")
    # print(f"Season p75 decades:    {season_decades.p75}")
    # if top_pb_scored:
    #     print(f"Top {POWERBALL_TOP_N} Powerball numbers (with scores):")
    #     for c in top_pb_scored:
    #         print(f"  PB {c.n:02d} score={c.total_score} freq_12mo={c.freq_12mo} rank_12mo={c.rank_12mo} ratio_12mo={c.ratio_12mo}")
    #
    # for i, t in enumerate(tickets, 1):
    #     vec = _decade_vector(t)
    #     print(f"Ticket #{i:02d}: {t}  decades={vec}")
    #
    # show_ticket_hits(REAL_DRAW, tickets)
    #
    #
    # def hit_summary(tickets, real):
    #     real = set(real)
    #     hits = [len(set(t) & real) for t in tickets]
    #     return {
    #         "max": max(hits) if hits else 0,
    #         "n5": sum(h >= 5 for h in hits),
    #         "n4": sum(h >= 4 for h in hits),
    #         "hits": hits,
    #     }
    #
    #
    # if REAL_DRAW:
    #     s = hit_summary(tickets, REAL_DRAW)
    #     print("\n=== HIT METRICS ===")
    #     print("max_hit =", s["max"], "| n>=5 =", s["n5"], "| n>=4 =", s["n4"])
    #     print("hits_per_ticket =", s["hits"])

