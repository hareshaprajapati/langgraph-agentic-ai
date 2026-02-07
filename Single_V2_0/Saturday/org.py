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
    "org.log"   # single growing log file
)

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

CSV_PATH = "Tattslotto.csv"
TARGET_DATE = "2026-2-07"
REAL_DRAW_TARGET = [3, 8, 9, 27, 33, 41]

# TARGET_DATE = "2026-1-31"
# REAL_DRAW_TARGET = [9, 20, 33, 34, 42, 45]

# Backtest: run on the last 5 available draws in the CSV.
N = 20

NUM_TICKETS = 20
NUMBERS_PER_TICKET = 6

MAIN_MIN = 1
MAIN_MAX = 45

LOOKBACK_DAYS = 210
LOOKBACK_DAYS_12MO = 365
SEASON_WINDOW_DAYS = 9
SEASON_LOOKBACK_YEARS = 20
MIN_SEASON_SAMPLES = 50
RECENT_DRAWS_PENALTY_N = 6

# Candidate pool
POOL_SIZE = 30
MID_POOL_SIZE = 10
COLD_POOL_SIZE = 10
HOT_POOL_SIZE = 8
OVERDUE_POOL_SIZE = 8
SEASON_POOL_SIZE = 8
COLD_FORCE_COUNT = 2

# Hard-force coverage mix
FORCE_COVERAGE = False
RANDOM_SEED = 0
DEBUG_PRINT = True

# Score weights (date-agnostic)
W_RECENT = 0.55
W_LONG = 0.20
W_SEASON = 0.15
W_RANK = 0.10
COLD_BOOST = 0.25
# Overdue gap boost (date-agnostic)
W_GAP = 0.25
GAP_CAP = 0.30

# Use Power-style CandidateScoreMain totals for ranking/selection
USE_POWER_STYLE_SCORE = False

# Ticket constraints (soft)
OVERLAP_CAP = 5
GLOBAL_MAX_USES = 5

# Odd / low / sum preferences (learned from history)
ODD_BAND = (1, 4)
LOW_RANGE_MAX = 22
LOW_BAND = (1, 4)
SUM_BAND_QUANTILES = (0.25, 0.75)
CONSECUTIVE_MAX = 2

# Decade bands
DECADE_BANDS: List[Tuple[int, int, int]] = [
    (1, 1, 10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 40),
    (5, 41, 45),
]

# Seasonal decade weighting (date-agnostic, learned from history)
SEASON_DECADE_WEIGHT = 0.6

# Optional: enforce exact decade counts (set None to disable)
# Example: {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}
DECADE_TARGET_COUNTS = None

# Acceptance behavior
PENALTY_SCALE = 0.65
MAX_ATTEMPTS = 30000


# Cohesion strategy (3+ hit traits)
COHESION_ENABLED = True
COHESION_TOP_RANK = 25
COHESION_MAX_SCORE_SPREAD = 0.18
COHESION_MAX_RANK_GAP = 8
COHESION_PAIR_BASE_PCTL = 75
COHESION_ACCEPT_FLOOR = 0.70
COHESION_ACCEPT_SPAN = 0.60
COHESION_W_SPREAD = 0.30
COHESION_W_PAIR = 0.35
COHESION_W_RANK_CONT = 0.20
COHESION_W_CENTRAL = 0.15
DEFAULT_STRATEGY_NAME = "COHESION_SOFT"

# Portfolio selection (20-ticket optimizer)
PORTFOLIO_MODE = False
PORTFOLIO_CANDIDATES = 20000
COHESIVE_TICKETS = 14
DIFFUSE_TICKETS = 6
CORE_SIZE = 16
CORE_MIN_USE = 6
CORE_MAX_USE = 10
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
REAL_DRAW = [3, 5, 20, 26, 28, 40]
# If TARGET_DATE is missing in CSV, optionally use REAL_DRAW for hit summary.
USE_REAL_DRAW_FALLBACK = False

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


# ----- Power-style CandidateScoreMain (logging) -----
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


# ----- WINNER BAND SUMMARY (Powerball-style logging) -----
BAND_B_LEADER = (1, 5)
BAND_A_CORE = (6, 22)
BAND_C_TAIL = (23, 27)


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
    scored: List["CandidateScore"],
    band_b: Tuple[int, int] = BAND_B_LEADER,
    band_a: Tuple[int, int] = BAND_A_CORE,
    band_c: Tuple[int, int] = BAND_C_TAIL,
) -> Tuple[str, DrawBandStats]:
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
            rnk, sc = 999, 0.0
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
    scored: List["CandidateScore"],
    band_b: Tuple[int, int] = BAND_B_LEADER,
    band_a: Tuple[int, int] = BAND_A_CORE,
    band_c: Tuple[int, int] = BAND_C_TAIL,
) -> None:
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


def print_date_by_date_band_counts_ascending(band_stats: List[DrawBandStats]) -> None:
    stats_sorted = sorted(band_stats, key=lambda s: str(s.date))

    def get_count(s, key, default=0):
        c = getattr(s, "counts", None)
        if isinstance(c, dict):
            if key in c:
                return int(c[key])
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


# ============================================================
# Scoring + ticketing
# ============================================================

def score_numbers(df: pd.DataFrame, main_cols: List[str], target_date: str, debug: bool) -> List[CandidateScore]:
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

    inv_rank = pd.Series({n: 1.0 / float(ranks.get(n, 999)) for n in range(MAIN_MIN, MAIN_MAX + 1)})
    s_rank = _normalize_series(inv_rank)

    scored: List[CandidateScore] = []
    for n in range(MAIN_MIN, MAIN_MAX + 1):
        freq_recent = int(recent_counts.get(n, 0))
        freq_long = int(long_counts.get(n, 0))
        freq_season = int(seasonal_counts.get(n, 0))
        rank_recent = int(ranks.get(n, 999))

        total = (
            W_RECENT * float(s_recent.get(n, 0.0)) +
            W_LONG * float(s_long.get(n, 0.0)) +
            W_SEASON * float(s_season.get(n, 0.0)) +
            W_RANK * float(s_rank.get(n, 0.0))
        )
        total += min(GAP_CAP, W_GAP * float(s_gap.get(n, 0.0)))
        if freq_recent <= 1:
            total += COLD_BOOST

        scored.append(CandidateScore(n=n, total_score=round(total, 6),
                                     freq_recent=freq_recent, freq_long=freq_long,
                                     freq_season=freq_season, rank_recent=rank_recent,
                                     gap_days=gap_days.get(n, 0)))

    scored.sort(key=lambda x: x.total_score, reverse=True)

    scored_main = compute_candidate_score_main(df, main_cols, target_date)
    if USE_POWER_STYLE_SCORE and scored_main:
        power_score_map = {c.n: float(c.total_score) for c in scored_main}
        for c in scored:
            if c.n in power_score_map:
                c.total_score = round(power_score_map[c.n], 6)
        scored.sort(key=lambda x: x.total_score, reverse=True)

    if scored_main:
        main_map = {c.n: c for c in scored_main}
        scored_main = [main_map[n.n] for n in scored if n.n in main_map]

    if debug:
        if scored_main:
            print("\n=== CANDIDATE SCORE MAIN (POWER-STYLE) ===")
            for c in scored_main[:45]:
                print(c)

    return scored


def compute_candidate_score_main(
    df: pd.DataFrame,
    main_cols: List[str],
    target_date: str,
) -> List[CandidateScoreMain]:
    t = pd.Timestamp(target_date)
    if pd.isna(t):
        raise ValueError("TARGET_DATE must be parseable (YYYY-MM-DD)")

    train = df[df["Date"] < t].copy()
    if train.empty:
        return []

    season_profile = _learn_season_profile_mains(
        df=df,
        main_cols=main_cols,
        target_ts=t,
        lookback_days=LOOKBACK_DAYS_12MO,
        season_window_days=SEASON_WINDOW_DAYS,
        season_lookback_years=SEASON_LOOKBACK_YEARS,
        min_samples=MIN_SEASON_SAMPLES,
    )
    allow_leader = season_profile.leader_rate >= 0.20

    recent_start = t - pd.Timedelta(days=LOOKBACK_DAYS_12MO)
    recent_counts = _counts_mains_in_window(train, main_cols, recent_start, t)
    maxfreq = int(recent_counts.max()) if not recent_counts.empty else 0
    ranks = _rank_from_counts(recent_counts) if maxfreq > 0 else {}

    last_n_draws = train.sort_values("Date").tail(RECENT_DRAWS_PENALTY_N)
    last_n_counts = _explode_mains(last_n_draws, main_cols).value_counts()

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
            hist_start = d - pd.Timedelta(days=LOOKBACK_DAYS_12MO)
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
    }


def _ticket_penalty(nums: List[int], dist: Dict[str, object]) -> float:
    penalty = 0.0

    # odd count band
    odd_ct = sum(1 for n in nums if n % 2 == 1)
    if odd_ct < ODD_BAND[0] or odd_ct > ODD_BAND[1]:
        penalty += 0.8 * abs(odd_ct - max(min(odd_ct, ODD_BAND[1]), ODD_BAND[0]))

    # low count band
    low_ct = sum(1 for n in nums if n <= LOW_RANGE_MAX)
    if low_ct < LOW_BAND[0] or low_ct > LOW_BAND[1]:
        penalty += 0.8 * abs(low_ct - max(min(low_ct, LOW_BAND[1]), LOW_BAND[0]))

    # sum band
    s = sum(nums)
    if s < dist["sum_lo"]:
        penalty += 0.6 * (dist["sum_lo"] - s) / 10.0
    elif s > dist["sum_hi"]:
        penalty += 0.6 * (s - dist["sum_hi"]) / 10.0

    # consecutive pairs
    consec = _count_consecutive_pairs(nums)
    if consec > CONSECUTIVE_MAX:
        penalty += 0.7 * (consec - CONSECUTIVE_MAX)

    # decade balance (blend seasonal + global)
    vec = _decade_vector(nums)
    for d in DECADE_IDS:
        season_mean = dist.get("decade_season_mean", {}).get(d, 0.0)
        global_mean = dist["decade_mean"].get(d, 0.0)
        target_mean = (SEASON_DECADE_WEIGHT * season_mean) + ((1.0 - SEASON_DECADE_WEIGHT) * global_mean)
        penalty += 0.4 * abs(vec[d] - target_mean)

    return penalty


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
    ticket_count: int = None,
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    train = df[df["Date"] < pd.Timestamp(target_date)]
    dist = _history_distributions(train, main_cols, target_date)

    # build pool
    if pool_override is not None:
        pool = list(dict.fromkeys(pool_override))
    else:
        top_pool = [c.n for c in scored[:POOL_SIZE]]
        mid_pool = [c.n for c in scored[POOL_SIZE:POOL_SIZE + MID_POOL_SIZE]]
        cold_pool = [c.n for c in scored if c.freq_recent <= 1][:COLD_POOL_SIZE]
        pool = list(dict.fromkeys(top_pool + mid_pool + cold_pool))

    # weights from scores
    score_map = {c.n: c.total_score for c in scored}
    rank_map = {c.n: c.rank_recent for c in scored}
    min_score = min(score_map.values()) if score_map else 0.0
    if use_weights:
        weights = [(score_map.get(n, 0.0) - min_score) + 0.25 for n in pool]
    else:
        weights = [1.0 for _ in pool]

    # hot + overdue + seasonal + cold pools (for enforced mix)
    hot_pool = [c.n for c in scored[:HOT_POOL_SIZE]]
    season_sorted = sorted(scored, key=lambda x: (x.freq_season, x.total_score), reverse=True)
    season_pool = [c.n for c in season_sorted[:SEASON_POOL_SIZE]]
    overdue_sorted = sorted(scored, key=lambda x: (x.gap_days, x.total_score), reverse=True)
    overdue_pool = [c.n for c in overdue_sorted[:OVERDUE_POOL_SIZE]]
    cold_sorted = sorted(scored, key=lambda x: (x.freq_recent, x.total_score))
    cold_pool_force = [c.n for c in cold_sorted[:COLD_POOL_SIZE]]

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

    if ticket_count is None:
        ticket_count = NUM_TICKETS
    tickets: List[List[int]] = []
    global_use = {n: 0 for n in range(MAIN_MIN, MAIN_MAX + 1)}
    overlap_cap = OVERLAP_CAP if overlap_cap_override is None else int(overlap_cap_override)
    global_max = GLOBAL_MAX_USES if global_max_override is None else int(global_max_override)

    attempts = 0
    while len(tickets) < ticket_count and attempts < MAX_ATTEMPTS:
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
            if cold_pool_force and COLD_FORCE_COUNT > 1:
                extras = [n for n in cold_pool_force if n not in seed]
                rng.shuffle(extras)
                for n in extras[: max(0, COLD_FORCE_COUNT - 1)]:
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

        penalty = _ticket_penalty(pick, dist)
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

    print(f"Generated {len(tickets)}/{ticket_count} tickets in {attempts} attempts")
    # if len(tickets) < NUM_TICKETS:
    #     raise RuntimeError("Could not generate enough tickets. Increase POOL_SIZE or relax constraints.")

    return tickets


def show_ticket_hits(real_draw: List[int], tickets: List[List[int]]):
    if not real_draw:
        return
    rd = sorted(real_draw)
    rd_set = set(rd)
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
            print(f"Ticket #{i:02d}: hits={hit_n} nums={hits} near_miss={len(near)} near_nums={sorted(set(near))}")

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
        "ge2": sum(counts[h] for h in counts if h >= 2),
        "total_hits": total_hits,
    }


def _evaluate_strategy(
    df: pd.DataFrame,
    main_cols: List[str],
    bt_dates: List[pd.Timestamp],
    strategy_name: str,
    cohesion_config: Dict[str, object],
) -> Dict[str, object]:
    weeks_with_5 = 0
    ge3_total = 0
    for d in bt_dates:
        bt_date = d.strftime("%Y-%m-%d")
        row = df[df["Date"] == d].iloc[0]
        bt_draw = [int(row[c]) for c in main_cols]
        bt_scored = score_numbers(df, main_cols, bt_date, DEBUG_PRINT)
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
        if any(len(set(t).intersection(set(bt_draw))) >= 5 for t in bt_tickets):
            weeks_with_5 += 1
    return {
        "name": strategy_name,
        "hits5": weeks_with_5,
        "ge3_total": ge3_total,
    }


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
            "name": "PAIR_HEAVY",
            "cohesion": {
                "enabled": True,
                "accept_floor": 0.65,
                "accept_span": 0.70,
                "min_score": None,
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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, main_cols = _load_csv(CSV_PATH)

    # Run for the configured target date (prediction); if it exists in CSV, use it as a backtest draw.
    t_target = pd.Timestamp(TARGET_DATE)
    if pd.isna(t_target):
        raise ValueError("TARGET_DATE must be parseable (YYYY-MM-DD)")
    df_target = df[df["Date"] == t_target]

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
    if not df_target.empty:
        row = df_target.iloc[0]
        real_draw = [int(row[c]) for c in main_cols]
    else:
        real_draw = REAL_DRAW if (USE_REAL_DRAW_FALLBACK and REAL_DRAW) else []
        if real_draw:
            print("TARGET_DATE not found in CSV; using REAL_DRAW override.")
        else:
            print("TARGET_DATE not found in CSV; generating prediction without hit summary.")

    scored = score_numbers(df, main_cols, run_date, DEBUG_PRINT)

    strategies = _strategy_configs()
    main_cfg = next((s["cohesion"] for s in strategies if s["name"] == DEFAULT_STRATEGY_NAME), None)
    if main_cfg is None:
        raise ValueError(f"Unknown DEFAULT_STRATEGY_NAME: {DEFAULT_STRATEGY_NAME}")

    if PORTFOLIO_MODE:
        tickets = generate_portfolio_tickets(scored, df, main_cols, run_date)
    else:
        tickets = generate_tickets(scored, df, main_cols, run_date,
                                   use_weights=True, seed_hot_overdue=False,
                                   force_coverage=FORCE_COVERAGE,
                                   cohesion_config=main_cfg)

    mode_label = "HARD_FORCE" if FORCE_COVERAGE else "WEIGHTED"
    print(f"\n=== {mode_label} STRATEGY ===")
    print(f"Target: {run_date}")
    print(f"Tickets: {NUM_TICKETS} | Pool size: {POOL_SIZE} + mid {MID_POOL_SIZE} + cold {COLD_POOL_SIZE}")
    print(f"Decade bands: {DECADE_BANDS}")

    for i, t in enumerate(tickets, 1):
        vec = _decade_vector(t)
        print(f"Ticket #{i:02d}: {t}  decades={vec}")

    show_ticket_hits(real_draw, tickets)
    show_ticket_hits(REAL_DRAW_TARGET, tickets)


    backtest_rows = df.sort_values("Date").tail(N)
    bt_dates = [row["Date"] for _, row in backtest_rows.iterrows()]

    strategies = _strategy_configs()

    winner_blocks = []
    band_stats = []

    print(f"\n=== BACKTEST (LAST {N} DRAWS) ===")
    bt_weeks_lt3 = 0
    bt_weeks_ge3 = 0
    bt_weeks_ge4 = 0
    bt_weeks_ge5 = 0
    bt_weeks_ge6 = 0
    bt_ticket_hit3 = 0
    bt_ticket_hit4 = 0
    bt_ticket_hit5 = 0
    bt_ticket_hit6p = 0
    bt_best_hit = 0
    for d in bt_dates:
        bt_date = d.strftime("%Y-%m-%d")
        row = df[df["Date"] == d].iloc[0]
        bt_draw = [int(row[c]) for c in main_cols]
        bt_scored = score_numbers(df, main_cols, bt_date, DEBUG_PRINT)
        if PORTFOLIO_MODE:
            bt_tickets = generate_portfolio_tickets(bt_scored, df, main_cols, bt_date)
        else:
            bt_tickets = generate_tickets(bt_scored, df, main_cols, bt_date,
                                          use_weights=True, seed_hot_overdue=False,
                                          force_coverage=FORCE_COVERAGE, cohesion_config=main_cfg)
        print(f"\nTarget: {bt_date}")
        for i, t in enumerate(bt_tickets, 1):
            vec = _decade_vector(t)
            print(f"Ticket #{i:02d}: {t}  decades={vec}")
        show_ticket_hits(bt_draw, bt_tickets)
        best = 0
        for t in bt_tickets:
            h = len(set(t).intersection(set(bt_draw)))
            if h > best:
                best = h
            if h == 3:
                bt_ticket_hit3 += 1
            if h == 4:
                bt_ticket_hit4 += 1
            if h == 5:
                bt_ticket_hit5 += 1
            if h >= 6:
                bt_ticket_hit6p += 1
        if best < 3:
            bt_weeks_lt3 += 1
        if best >= 3:
            bt_weeks_ge3 += 1
        if best >= 4:
            bt_weeks_ge4 += 1
        if best >= 5:
            bt_weeks_ge5 += 1
        if best >= 6:
            bt_weeks_ge6 += 1
        if best > bt_best_hit:
            bt_best_hit = best

        collect_winner_tables_and_stats(
            blocks=winner_blocks,
            stats=band_stats,
            target_date=bt_date,
            real_draw=bt_draw,
            scored=bt_scored
        )

    print_all_winner_tables_at_end(winner_blocks)
    print_date_by_date_band_counts_ascending(band_stats)
    print_band_summary_at_end(band_stats)

    print(f"\n=== BACKTEST SUMMARY (LAST {N} DRAWS) ===")
    print(f"Weeks with <3 hits: {bt_weeks_lt3}")
    print(f"Weeks with 3+ hits: {bt_weeks_ge3}")
    print(f"Weeks with 4+ hits: {bt_weeks_ge4}")
    print(f"Weeks with 5+ hits: {bt_weeks_ge5}")
    print(f"Weeks with 6+ hits: {bt_weeks_ge6}")
    print(f"Max hit observed : {bt_best_hit}")
    print(f"Total tickets (hits=3): {bt_ticket_hit3}")
    print(f"Total tickets (hits=4): {bt_ticket_hit4}")
    print(f"Total tickets (hits=5): {bt_ticket_hit5}")
    print(f"Total tickets (hits>=6): {bt_ticket_hit6p}")
    print(f"Total tickets (hits>=3): {bt_ticket_hit3 + bt_ticket_hit4 + bt_ticket_hit5 + bt_ticket_hit6p}")
