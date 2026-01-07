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
    f"siko_power_single_logs.log"   # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

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

CSV_PATH = "Powerball.csv"
# TARGET_DATE = "2025-12-25"   # any date, learns season around this month/day
# TARGET_DATE = "2026-01-01"   # any date, learns season around this month/day

NUM_TICKETS = 20
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
DECADE_MODE = "hard"         # "hard" or "soft"
DECADE_MEDIAN_TOL = 1        # allow +/-1 per decade around seasonal p25..p75 (hard mode)
DECADE_SOFT_PENALTY = 0.6    # per unit distance outside tolerance (soft mode)
# Optional: exact decade counts to prefer per ticket (soft rule)
# Example: {1: 2, 2: 2, 3: 3, 4: 0}
DECADE_TARGET_COUNTS = None
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
        print(
            f"Ratio band (p25..p75): {season_profile.ratio_low:.3f}..{season_profile.ratio_high:.3f} | "
            f"ideal={season_profile.ratio_ideal:.3f}"
        )
        print(
            f"Rank band (p25..p75): {season_profile.rank_low}..{season_profile.rank_high} | "
            f"leader_rate={season_profile.leader_rate:.3f} | allow_leader={allow_leader}"
        )

        print("\n=== SEASON DECADE PROFILE (YOUR BANDS) ===")
        print(
            f"Anchor: {season_decades.anchor_month:02}/{season_decades.anchor_day:02} | "
            f"Samples: years={season_decades.sample_years}, draws={season_decades.sample_draws}"
        )
        print(f"Decade bands: {DECADE_BANDS}")
        print(f"Median decade mix: {season_decades.med}")
        print(f"P25 decade mix:    {season_decades.p25}")
        print(f"P75 decade mix:    {season_decades.p75}")

        print("\n=== TOP 35 SCORED NUMBERS ===")
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


def generate_tickets(scored, season_decades):
    """
    Production ticket generator:
      1) Sample MANY valid candidate tickets from the score pool (stacking enabled)
      2) Select the BEST NUM_TICKETS using greedy optimization under overlap cap

    This fixes the failure mode where good 4-hit ingredients are spread across different tickets.
    """
    if not scored:
        raise ValueError("generate_tickets(): scored list is empty")

    rng = random.Random(RANDOM_SEED)

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

    # --- choose best NUM_TICKETS using greedy overlap cap ---
    def ticket_value(t: List[int]) -> float:
        return float(sum(score_map.get(n, 0.0) for n in t))

    candidates.sort(key=ticket_value, reverse=True)

    tickets: List[List[int]] = []
    for t in candidates:
        if _ticket_overlap_ok(t, tickets, OVERLAP_CAP):
            tickets.append(t)
            if len(tickets) >= NUM_TICKETS:
                break

    # fallback relax overlap once
    if len(tickets) < NUM_TICKETS:
        relaxed = OVERLAP_CAP + 1
        for t in candidates:
            if t in tickets:
                continue
            if _ticket_overlap_ok(t, tickets, relaxed):
                tickets.append(t)
                if len(tickets) >= NUM_TICKETS:
                    break

    if len(tickets) < NUM_TICKETS:
        raise RuntimeError(
            f"generate_tickets(): could only select {len(tickets)} tickets from {len(candidates)} candidates. "
            f"Increase CANDIDATE_TICKET_POOL or relax OVERLAP_CAP."
        )

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

def generate_20_ticket_portfolio(scored, season_decades, history_draws):
    pairs, triplets = extract_pairs_triplets(history_draws)

    final = []

    for mode, cnt in MODE_SPLIT.items():
        p = MODE_PARAMS[mode]
        tks = generate_mode_tickets(
            scored,
            season_decades,
            cnt,
            temp=p["TEMP"],
            breakers=p["BREAKERS"],
            decade_mode=p["DECADE_MODE"],
            overlap_cap=p["OVERLAP_CAP"],
            pairs=pairs,
            triplets=triplets
        )
        final.extend(tks)

    return final[:TOTAL_TICKETS]



# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, main_cols = _load_csv(CSV_PATH)
    N = 10

    # IMPORTANT CHOICE:
    # If you want the first N rows AS THEY APPEAR in the CSV (your CSV is latest-first),
    # then use descending date:
    df_bt = df.sort_values("Date", ascending=False).head(N)
    # If you instead want the earliest N chronologically, use ascending=True.
    # df_bt = df.sort_values("Date", ascending=True).head(N)

    results = []

    for i, (_, row) in enumerate(df_bt.iterrows(), 1):
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
            # tickets = generate_tickets(scored, season_decades)
            # Build past_main_numbers: ONLY draws before target_date
            past_df = df[df["Date"] < row["Date"]].sort_values("Date")

            past_main_numbers = [
                [int(past_row[c]) for c in main_cols]
                for _, past_row in past_df.iterrows()
            ]

            historical_draws = [list(row) for row in past_main_numbers]

            tickets = generate_20_ticket_portfolio(scored, season_decades, historical_draws)

            show_ticket_hits(real_draw, tickets)

            hits = [len(set(t) & set(real_draw)) for t in tickets]
            results.append({
                "date": target_date,
                "max_hit": max(hits) if hits else 0,
                "n>=5": sum(h >= 5 for h in hits),
                "n>=4": sum(h >= 4 for h in hits),
            })

        except Exception as e:
            print(f"ERROR on {target_date}: {e}")
            results.append({"date": target_date, "error": str(e)})

    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    for r in results:
        print(r)

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

