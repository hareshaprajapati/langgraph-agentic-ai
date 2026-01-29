import sys
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
from itertools import combinations
import random
from collections import Counter
import pandas as pd

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

log_file_path = os.path.join(".", "Siko_Sat.py.log")
log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ============================================================
# USER CONFIG
# ============================================================
CSV_PATH = "Tattslotto.csv"
TARGET_DATE = "2026-1-24"
REAL_DRAW_TARGET = [8, 22, 24, 28, 29, 33]
# REAL_DRAW_TARGET = None
N = 21

NUM_TICKETS = 20
NUMBERS_PER_TICKET = 6
BACKTEST_TICKET_COUNT = 20
CANDIDATE_MULTIPLIER = 8
RANDOM_SEED = 0

# Logging
VERBOSE_TARGET = True
VERBOSE_BACKTEST = False
LOG_TOP_TICKETS = 5

# History windows
RECENT_WINDOW_DRAWS = 52
RECENT_TRIPLET_WINDOW = 20
OVERLAP_RECENT_WINDOW = 20
SEASON_WINDOW_DAYS = 9
SEASON_LOOKBACK_YEARS = 20

# Scoring weights
W_RECENT = 0.50
W_LONG = 0.15
W_GAP = 0.15
W_SEASON = 0.10
W_TREND = 0.10

PAIR_BONUS_W = 0.15
TRIPLET_BONUS_W = 0.10

# Penalties
PEN_SUM_W = 0.30
PEN_ODD_W = 0.25
PEN_LOW_W = 0.25
PEN_CONSEC_W = 0.25
PEN_DECADE_W = 0.25
PEN_OVERLAP_LAST_W = 0.40
PEN_OVERLAP_RECENT_W = 0.25
PEN_TRIPLET_HOT_W = 0.25
PEN_TRIPLET_RECENT_W = 0.20

# Candidate mix
UGLY_MODE_SHARE = 0.30
UGLY_RANK_START = 36
UGLY_RANK_END = 45

# Bucket-template + decade-conditioned generation
USE_BUCKET_TEMPLATES = True
BUCKET_TEMPLATES = [
    {1: 3, 2: 2, 3: 1, 4: 0},  # top-heavy
    {1: 2, 2: 1, 3: 2, 4: 1},  # balanced
    {1: 1, 2: 1, 3: 2, 4: 2},  # bottom-heavy
]
HOT_BUCKET_TEMPLATE = {1: 3, 2: 2, 3: 1, 4: 0}
DECADE_TOP_K = 5
TICKETS_PER_COMBO = 2

# Regime classifier (data-driven from history windows)
REGIME_WINDOW_DRAWS = 13
REGIME_HOT_Q = 0.75
REGIME_COLD_Q = 0.25
REGIME_B4_LOW_Q = 0.25
REGIME_B4_HIGH_Q = 0.75
TEMPLATE_WEIGHTS_HOT = [1.00, 0.00, 0.00]
TEMPLATE_WEIGHTS_COLD = [0.00, 0.30, 0.70]
TEMPLATE_WEIGHTS_NEUTRAL = [0.70, 0.30, 0.00]
CANDIDATE_MULTIPLIER_BUCKET = 20
HOT_POOL_SIZE = 20
HOT_POOL_SCORE_CUTOFF = 3000
USE_COVER_WHEEL = False
COVER_POOL_SIZE = 18
COVER_SELECT_WEIGHT = 0.30

# Ensemble generator
USE_ENSEMBLE = False
ENSEMBLE_COUNTS = {
    "baseline": 6,
    "bucket_top": 5,
    "bucket_mid": 4,
    "bucket_bottom": 3,
    "overlap2": 2,
}

# Decade bands (1-45)
DECADE_BANDS: List[Tuple[int, int, int]] = [
    (1, 1, 10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 40),
    (5, 41, 45),
]


@dataclass
class NumberStats:
    recent_freq: int
    long_freq: int
    gap_days: int
    season_freq: int
    trend: float


def _find_main_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if "Winning Number" in c]
    cols = sorted(cols, key=lambda x: int(x.split()[-1]))
    return cols


def _load_data() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(CSV_PATH)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    main_cols = _find_main_cols(df)
    for c in main_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=main_cols)
    for c in main_cols:
        df[c] = df[c].astype(int)
    if "Date" in df.columns:
        df = df.sort_values("Date")
    return df, main_cols


def _decade_vector(nums: List[int]) -> Dict[int, int]:
    vec = {k: 0 for k, _, _ in DECADE_BANDS}
    for n in nums:
        for k, lo, hi in DECADE_BANDS:
            if lo <= n <= hi:
                vec[k] += 1
                break
    return vec


def _decade_vector_tuple(vec: Dict[int, int]) -> Tuple[int, int, int, int, int]:
    return tuple(vec.get(k, 0) for k, _, _ in DECADE_BANDS)


def _avg_decade_vector(vectors: List[Dict[int, int]]) -> Dict[int, float]:
    if not vectors:
        return {k: 0.0 for k, _, _ in DECADE_BANDS}
    sums = {k: 0.0 for k, _, _ in DECADE_BANDS}
    for v in vectors:
        for k, _, _ in DECADE_BANDS:
            sums[k] += v.get(k, 0)
    n = float(len(vectors))
    return {k: sums[k] / n for k, _, _ in DECADE_BANDS}


def _format_decade_vec(vec: Dict[int, int]) -> str:
    return "{" + ", ".join(f"{k}:{vec.get(k,0)}" for k, _, _ in DECADE_BANDS) + "}"


def _format_decade_avg(vec: Dict[int, float]) -> str:
    return "{" + ", ".join(f"{k}:{vec.get(k,0):.2f}" for k, _, _ in DECADE_BANDS) + "}"


def _decade_overlap_score(a: Dict[int, int], b: Dict[int, int]) -> int:
    return sum(min(a.get(k, 0), b.get(k, 0)) for k, _, _ in DECADE_BANDS)


def _count_consecutive_pairs(nums: List[int]) -> int:
    s = sorted(nums)
    consec = 0
    for i in range(1, len(s)):
        if s[i] - s[i - 1] == 1:
            consec += 1
    return consec


def _season_draws(df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    t = pd.Timestamp(target_date)
    if pd.isna(t):
        return df.iloc[0:0]
    target_dt = t.date()
    start_year = target_dt.year - SEASON_LOOKBACK_YEARS
    end_year = target_dt.year - 1
    out = []
    for y in range(start_year, end_year + 1):
        try:
            anchor = datetime(y, target_dt.month, target_dt.day)
        except ValueError:
            continue
        win_start = anchor - pd.Timedelta(days=SEASON_WINDOW_DAYS)
        win_end = anchor + pd.Timedelta(days=SEASON_WINDOW_DAYS + 1)
        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if not near.empty:
            out.append(near)
    if not out:
        return df.iloc[0:0]
    return pd.concat(out, ignore_index=True)


def _history_stats(df_hist: pd.DataFrame, main_cols: List[str], target_date: str) -> Tuple[Dict[int, NumberStats], Dict[Tuple[int, int], float], Dict[Tuple[int, int, int], int], Dict[str, object]]:
    # Frequencies
    nums_all = []
    for _, row in df_hist.iterrows():
        nums_all.extend(int(row[c]) for c in main_cols)
    long_counts = Counter(nums_all)

    recent = df_hist.tail(RECENT_WINDOW_DRAWS)
    nums_recent = []
    for _, row in recent.iterrows():
        nums_recent.extend(int(row[c]) for c in main_cols)
    recent_counts = Counter(nums_recent)

    # Gaps
    gap_days = {n: 9999 for n in range(1, 46)}
    if not df_hist.empty:
        last_date = df_hist["Date"].max()
        for n in range(1, 46):
            rows = df_hist[df_hist[main_cols].isin([n]).any(axis=1)]
            if not rows.empty:
                gap = (last_date - rows["Date"].max()).days
                gap_days[n] = int(gap)

    # Seasonal freq
    season = _season_draws(df_hist, target_date)
    season_counts = Counter()
    if not season.empty:
        nums_season = []
        for _, row in season.iterrows():
            nums_season.extend(int(row[c]) for c in main_cols)
        season_counts = Counter(nums_season)

    # Trend: recent vs long (normalized)
    stats = {}
    for n in range(1, 46):
        long_f = long_counts.get(n, 0)
        recent_f = recent_counts.get(n, 0)
        trend = (recent_f / max(1, len(recent))) - (long_f / max(1, len(df_hist)))
        stats[n] = NumberStats(
            recent_freq=recent_f,
            long_freq=long_f,
            gap_days=gap_days.get(n, 9999),
            season_freq=season_counts.get(n, 0),
            trend=trend,
        )

    # Pair PMI (recent window)
    pair_counts = Counter()
    total_pairs = 0
    for _, row in recent.iterrows():
        nums = sorted(int(row[c]) for c in main_cols)
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                a, b = nums[i], nums[j]
                pair_counts[(a, b)] += 1
                total_pairs += 1
    pair_pmi = {}
    if total_pairs > 0:
        total_nums = sum(recent_counts.values())
        for (a, b), cnt in pair_counts.items():
            p_ab = cnt / total_pairs
            p_a = recent_counts.get(a, 0) / max(1, total_nums)
            p_b = recent_counts.get(b, 0) / max(1, total_nums)
            if p_a > 0 and p_b > 0 and p_ab > 0:
                pair_pmi[(a, b)] = math.log(p_ab / (p_a * p_b))

    # Triplets (recent window for bonus)
    triplet_counts = Counter()
    for _, row in recent.iterrows():
        nums = sorted(int(row[c]) for c in main_cols)
        for comb in combinations(nums, 3):
            triplet_counts[comb] += 1

    # Triplets (short recent window for strict avoidance)
    recent_short = df_hist.tail(RECENT_TRIPLET_WINDOW)
    triplet_recent_short = Counter()
    for _, row in recent_short.iterrows():
        nums = sorted(int(row[c]) for c in main_cols)
        for comb in combinations(nums, 3):
            triplet_recent_short[comb] += 1

    # Triplets (full history for "hot" triplets)
    triplet_total = Counter()
    for _, row in df_hist.iterrows():
        nums = sorted(int(row[c]) for c in main_cols)
        for comb in combinations(nums, 3):
            triplet_total[comb] += 1

    # Distribution constraints
    sums = []
    odd_counts = []
    low_counts = []
    consec_counts = []
    decade_vectors = []
    for _, row in df_hist.iterrows():
        nums = [int(row[c]) for c in main_cols]
        sums.append(sum(nums))
        odd_counts.append(sum(1 for n in nums if n % 2 == 1))
        low_counts.append(sum(1 for n in nums if n <= 22))
        consec_counts.append(_count_consecutive_pairs(nums))
        decade_vectors.append(_decade_vector(nums))
    sum_lo = pd.Series(sums).quantile(0.25) if sums else 0
    sum_hi = pd.Series(sums).quantile(0.75) if sums else 0
    dist = {
        "sum_lo": float(sum_lo),
        "sum_hi": float(sum_hi),
        "odd_mean": float(pd.Series(odd_counts).mean()) if odd_counts else 0,
        "low_mean": float(pd.Series(low_counts).mean()) if low_counts else 0,
        "consec_mean": float(pd.Series(consec_counts).mean()) if consec_counts else 0,
        "decade_mean": {k: sum(v.get(k, 0) for v in decade_vectors) / max(1, len(decade_vectors)) for k, _, _ in DECADE_BANDS},
        "triplet_recent_short": triplet_recent_short,
        "triplet_total": triplet_total,
    }

    return stats, pair_pmi, triplet_counts, dist


def _normalize_map(m: Dict[int, float]) -> Dict[int, float]:
    vals = list(m.values())
    if not vals:
        return m
    mn = min(vals)
    mx = max(vals)
    if mx <= mn:
        return {k: 0.0 for k in m}
    return {k: (v - mn) / (mx - mn) for k, v in m.items()}


def _number_scores(stats: Dict[int, NumberStats]) -> Dict[int, float]:
    # Build component maps
    recent = {n: float(s.recent_freq) for n, s in stats.items()}
    long = {n: float(s.long_freq) for n, s in stats.items()}
    gap = {n: float(s.gap_days) for n, s in stats.items()}
    season = {n: float(s.season_freq) for n, s in stats.items()}
    trend = {n: float(s.trend) for n, s in stats.items()}

    recent_n = _normalize_map(recent)
    long_n = _normalize_map(long)
    gap_n = _normalize_map(gap)
    season_n = _normalize_map(season)
    trend_n = _normalize_map(trend)

    scores = {}
    for n in stats:
        scores[n] = (
            W_RECENT * recent_n[n]
            + W_LONG * long_n[n]
            + W_GAP * gap_n[n]
            + W_SEASON * season_n[n]
            + W_TREND * trend_n[n]
        )
    return scores


def _log_stats_snapshot(stats: Dict[int, NumberStats], dist: Dict[str, object]) -> None:
    recent = {n: float(s.recent_freq) for n, s in stats.items()}
    long = {n: float(s.long_freq) for n, s in stats.items()}
    gap = {n: float(s.gap_days) for n, s in stats.items()}
    season = {n: float(s.season_freq) for n, s in stats.items()}
    trend = {n: float(s.trend) for n, s in stats.items()}

    def topk(m: Dict[int, float], k: int = 10) -> str:
        items = sorted(m.items(), key=lambda x: x[1], reverse=True)[:k]
        return ", ".join(f"{n}:{v:.3f}" for n, v in items)

    print("\n=== STATS SNAPSHOT ===")
    print(f"Sum Q1/Q3: {dist['sum_lo']:.1f}/{dist['sum_hi']:.1f} | Odd mean: {dist['odd_mean']:.2f} | Low mean: {dist['low_mean']:.2f} | Consec mean: {dist['consec_mean']:.2f}")
    print("Top recent:", topk(recent))
    print("Top long  sees:", topk(long))
    print("Top gap (overdue):", topk(gap))
    print("Top season:", topk(season))
    print("Top trend:", topk(trend))


def _ticket_diagnostics(
    nums: List[int],
    num_scores: Dict[int, float],
    pair_pmi: Dict[Tuple[int, int], float],
    triplet_counts: Dict[Tuple[int, int, int], int],
    dist: Dict[str, object],
    last_draw: List[int],
    recent_draws: List[List[int]],
) -> Dict[str, float]:
    s = sum(nums)
    odd_ct = sum(1 for n in nums if n % 2 == 1)
    low_ct = sum(1 for n in nums if n <= 22)
    consec = _count_consecutive_pairs(nums)
    vec = _decade_vector(nums)

    pair_bonus = 0.0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            if a > b:
                a, b = b, a
            pair_bonus += pair_pmi.get((a, b), 0.0)

    trip_bonus = 0.0
    hot_triplets = 0
    recent_triplets = 0
    triplet_total = dist.get("triplet_total", triplet_counts)
    triplet_recent_short = dist.get("triplet_recent_short", triplet_counts)
    for comb in combinations(sorted(nums), 3):
        trip_bonus += min(2, triplet_counts.get(comb, 0))
        if triplet_total.get(comb, 0) >= 6:
            hot_triplets += 1
        if triplet_recent_short.get(comb, 0) >= 1:
            recent_triplets += 1

    last_ol = len(set(nums).intersection(last_draw)) if last_draw else 0
    max_ol = 0
    if recent_draws:
        for d in recent_draws:
            ol = len(set(nums).intersection(d))
            if ol > max_ol:
                max_ol = ol

    return {
        "score_sum": sum(num_scores.get(n, 0.0) for n in nums),
        "pair_bonus": pair_bonus,
        "trip_bonus": trip_bonus,
        "sum": s,
        "odd": odd_ct,
        "low": low_ct,
        "consec": consec,
        "last_ol": last_ol,
        "max_ol": max_ol,
        "hot_triplets": hot_triplets,
        "recent_triplets": recent_triplets,
        "decade_vec": vec,
    }


def _number_profile(nums: List[int], stats: Dict[int, NumberStats], num_scores: Dict[int, float]) -> List[str]:
    ranked = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {n: i + 1 for i, (n, _) in enumerate(ranked)}
    out = []
    for n in sorted(nums):
        s = stats.get(n)
        out.append(
            f"{n}(r{ranks.get(n, 999)},rec={s.recent_freq if s else 0},long={s.long_freq if s else 0},gap={s.gap_days if s else 0},season={s.season_freq if s else 0},trend={s.trend if s else 0:.3f})"
        )
    return out


def _rank_buckets(num_scores: Dict[int, float]) -> Dict[int, int]:
    ranked = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    buckets = {}
    for idx, (n, _) in enumerate(ranked, 1):
        if idx <= 10:
            b = 1
        elif idx <= 20:
            b = 2
        elif idx <= 30:
            b = 3
        else:
            b = 4
        buckets[n] = b
    return buckets


def _recent_decade_vectors(df_hist: pd.DataFrame, main_cols: List[str]) -> List[Tuple[Tuple[int, int, int, int, int], int]]:
    if df_hist.empty:
        return []
    recent = df_hist.tail(RECENT_WINDOW_DRAWS)
    counts = Counter()
    for _, row in recent.iterrows():
        nums = [int(row[c]) for c in main_cols]
        vec = _decade_vector(nums)
        counts[_decade_vector_tuple(vec)] += 1
    return counts.most_common(DECADE_TOP_K)


def _build_decade_bucket_pools(num_scores: Dict[int, float], buckets: Dict[int, int]) -> Dict[int, Dict[int, List[int]]]:
    pools: Dict[int, Dict[int, List[int]]] = {k: {1: [], 2: [], 3: [], 4: []} for k, _, _ in DECADE_BANDS}
    for n in range(1, 46):
        b = buckets.get(n, 4)
        for k, lo, hi in DECADE_BANDS:
            if lo <= n <= hi:
                pools[k][b].append(n)
                break
    # sort by score desc within each pool
    for k in pools:
        for b in pools[k]:
            pools[k][b].sort(key=lambda n: num_scores.get(n, 0.0), reverse=True)
    return pools


def _generate_tickets_bucket_only(
    stats: Dict[int, NumberStats],
    num_scores: Dict[int, float],
    pair_pmi: Dict[Tuple[int, int], float],
    triplet_counts: Dict[Tuple[int, int, int], int],
    dist: Dict[str, object],
    last_draw: List[int],
    recent_draws: List[List[int]],
    buckets: Dict[int, int],
    bucket_template: Dict[int, int],
    ticket_count: int,
    rng: random.Random,
    lock_decades: bool = True,
) -> List[List[int]]:
    candidates = []
    target_candidates = ticket_count * CANDIDATE_MULTIPLIER_BUCKET
    attempts = 0
    pool = list(range(1, 46))
    while len(candidates) < target_candidates and attempts < target_candidates * 50:
        attempts += 1
        bucket_limits = dict(bucket_template)
        chosen: List[int] = []
        if lock_decades:
            # lock top recent/long per decade if that decade has quota
            for dec_k, _, _ in DECADE_BANDS:
                if sum(bucket_limits.values()) <= 0:
                    break
                locked = _decade_lock_numbers(stats, dec_k)
                for n in locked:
                    b = buckets.get(n, 4)
                    if bucket_limits.get(b, 0) <= 0:
                        continue
                    if n not in chosen:
                        chosen.append(n)
                        bucket_limits[b] -= 1
        remaining = NUMBERS_PER_TICKET - len(chosen)
        if remaining <= 0:
            pick = sorted(chosen)[:NUMBERS_PER_TICKET]
            if pick not in candidates:
                candidates.append(pick)
            continue
        # sample remaining by score with bucket limits
        available = [n for n in pool if n not in chosen and bucket_limits.get(buckets.get(n, 4), 0) > 0]
        picks = _sample_from_pool_with_bucket_limits(available, bucket_limits, buckets, num_scores, rng, remaining)
        if len(picks) != remaining:
            continue
        chosen.extend(picks)
        pick = sorted(chosen)
        if pick in candidates:
            continue
        candidates.append(pick)

    scored = []
    for t in candidates:
        s = _ticket_score(t, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:ticket_count]]


def _generate_tickets_hot_pool(
    num_scores: Dict[int, float],
    pair_pmi: Dict[Tuple[int, int], float],
    triplet_counts: Dict[Tuple[int, int, int], int],
    dist: Dict[str, object],
    last_draw: List[int],
    recent_draws: List[List[int]],
    ticket_count: int,
) -> List[List[int]]:
    ranked = [n for n, _ in sorted(num_scores.items(), key=lambda x: x[1], reverse=True)]
    pool = ranked[:HOT_POOL_SIZE]
    scored = []
    for comb in combinations(pool, NUMBERS_PER_TICKET):
        t = list(comb)
        s = _ticket_score(t, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    if HOT_POOL_SCORE_CUTOFF and len(scored) > HOT_POOL_SCORE_CUTOFF:
        scored = scored[:HOT_POOL_SCORE_CUTOFF]
    return [t for _, t in scored[:ticket_count]]


def _generate_tickets_cover_wheel(
    stats: Dict[int, NumberStats],
    num_scores: Dict[int, float],
    pair_pmi: Dict[Tuple[int, int], float],
    triplet_counts: Dict[Tuple[int, int, int], int],
    dist: Dict[str, object],
    last_draw: List[int],
    recent_draws: List[List[int]],
    buckets: Dict[int, int],
    ticket_count: int,
) -> List[List[int]]:
    ranked = [n for n, _ in sorted(num_scores.items(), key=lambda x: x[1], reverse=True)]
    top_core = ranked[:COVER_POOL_SIZE - 3]
    # add 3 overdue bucket4 numbers to allow low-rank inclusion
    bucket4 = [n for n in ranked[30:]]
    bucket4 = sorted(bucket4, key=lambda n: stats[n].gap_days, reverse=True)[:3]
    pool = list(dict.fromkeys(top_core + bucket4))
    if len(pool) < NUMBERS_PER_TICKET:
        return []

    # precompute tickets and their 4-subsets
    candidates = []
    for comb in combinations(pool, NUMBERS_PER_TICKET):
        t = tuple(sorted(comb))
        score = _ticket_score(list(t), num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
        subs = [tuple(sorted(s)) for s in combinations(t, 4)]
        candidates.append((score, t, subs))

    selected = []
    covered = set()
    remaining = candidates[:]
    for _ in range(ticket_count):
        best_idx = None
        best_val = -1e9
        for i, (score, t, subs) in enumerate(remaining):
            new_cov = sum(1 for s in subs if s not in covered)
            val = score + COVER_SELECT_WEIGHT * new_cov
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx is None:
            break
        score, t, subs = remaining.pop(best_idx)
        selected.append(list(t))
        for s in subs:
            covered.add(s)
    return selected


def _generate_tickets_with_overlap(
    num_scores: Dict[int, float],
    pair_pmi: Dict[Tuple[int, int], float],
    triplet_counts: Dict[Tuple[int, int, int], int],
    dist: Dict[str, object],
    last_draw: List[int],
    recent_draws: List[List[int]],
    overlap_k: int,
    ticket_count: int,
    rng: random.Random,
) -> List[List[int]]:
    if not last_draw or overlap_k <= 0:
        return []
    overlap_k = min(overlap_k, len(last_draw))
    pool = list(range(1, 46))
    weights = [max(1e-9, num_scores.get(n, 0.0)) for n in pool]
    candidates = []
    target_candidates = ticket_count * CANDIDATE_MULTIPLIER
    attempts = 0
    while len(candidates) < target_candidates and attempts < target_candidates * 50:
        attempts += 1
        base = rng.sample(last_draw, overlap_k)
        remaining = NUMBERS_PER_TICKET - overlap_k
        available = [n for n in pool if n not in base]
        picks = _weighted_sample_no_replace(available, [weights[pool.index(n)] for n in available], remaining, rng)
        t = sorted(base + picks)
        if t in candidates:
            continue
        candidates.append(t)
    scored = []
    for t in candidates:
        s = _ticket_score(t, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:ticket_count]]


def _decade_focus_pool(stats: Dict[int, NumberStats], decade_k: int) -> List[int]:
    decade_nums = []
    lo = hi = None
    for k, d_lo, d_hi in DECADE_BANDS:
        if k == decade_k:
            lo, hi = d_lo, d_hi
            break
    if lo is None or hi is None:
        return []
    decade_nums = [n for n in range(lo, hi + 1)]
    top_recent = sorted(decade_nums, key=lambda n: stats[n].recent_freq, reverse=True)[:3]
    top_long = sorted(decade_nums, key=lambda n: stats[n].long_freq, reverse=True)[:3]
    top_gap = sorted(decade_nums, key=lambda n: stats[n].gap_days, reverse=True)[:3]
    pool = list(dict.fromkeys(top_recent + top_long + top_gap))
    return pool


def _decade_lock_numbers(stats: Dict[int, NumberStats], decade_k: int) -> List[int]:
    lo = hi = None
    for k, d_lo, d_hi in DECADE_BANDS:
        if k == decade_k:
            lo, hi = d_lo, d_hi
            break
    if lo is None or hi is None:
        return []
    decade_nums = [n for n in range(lo, hi + 1)]
    top_recent = max(decade_nums, key=lambda n: stats[n].recent_freq)
    top_long = max(decade_nums, key=lambda n: stats[n].long_freq)
    locked = [top_recent]
    if top_long != top_recent:
        locked.append(top_long)
    return locked


def _top10_share_for_window(df_win: pd.DataFrame, main_cols: List[str]) -> float:
    nums = []
    for _, row in df_win.iterrows():
        nums.extend(int(row[c]) for c in main_cols)
    counts = Counter(nums)
    if not counts:
        return 0.0
    total = sum(counts.values())
    top10 = [n for n, _ in counts.most_common(10)]
    return sum(counts.get(n, 0) for n in top10) / total if total else 0.0


def _regime_thresholds(df_hist: pd.DataFrame, main_cols: List[str], buckets: Dict[int, int]) -> Dict[str, float]:
    # top10 share distribution over rolling windows
    top_shares = []
    if len(df_hist) >= REGIME_WINDOW_DRAWS:
        for i in range(REGIME_WINDOW_DRAWS, len(df_hist) + 1):
            win = df_hist.iloc[i - REGIME_WINDOW_DRAWS:i]
            top_shares.append(_top10_share_for_window(win, main_cols))
    # bucket4 count distribution over historical draws (using current buckets)
    b4_counts = []
    for _, row in df_hist.iterrows():
        nums = [int(row[c]) for c in main_cols]
        b4_counts.append(sum(1 for n in nums if buckets.get(n, 4) == 4))

    def q(vals, quant):
        if not vals:
            return 0.0
        vals = sorted(vals)
        idx = int(round((len(vals) - 1) * quant))
        return vals[idx]

    return {
        "top_hot": q(top_shares, REGIME_HOT_Q),
        "top_cold": q(top_shares, REGIME_COLD_Q),
        "b4_low": q(b4_counts, REGIME_B4_LOW_Q),
        "b4_high": q(b4_counts, REGIME_B4_HIGH_Q),
        "top_share_now": top_shares[-1] if top_shares else 0.0,
    }


def _sample_from_pool_with_bucket_limits(
    pool: List[int],
    bucket_limits: Dict[int, int],
    buckets: Dict[int, int],
    num_scores: Dict[int, float],
    rng: random.Random,
    k: int,
) -> List[int]:
    chosen: List[int] = []
    available = [n for n in pool if bucket_limits.get(buckets.get(n, 4), 0) > 0]
    attempts = 0
    while len(chosen) < k and available and attempts < 200:
        attempts += 1
        weights = []
        for n in available:
            b = buckets.get(n, 4)
            if bucket_limits.get(b, 0) <= 0:
                weights.append(0.0)
            else:
                weights.append(max(1e-9, num_scores.get(n, 0.0)))
        pick = _weighted_sample_no_replace(available, weights, 1, rng)[0]
        b = buckets.get(pick, 4)
        if bucket_limits.get(b, 0) <= 0:
            continue
        chosen.append(pick)
        bucket_limits[b] -= 1
        available = [n for n in available if n not in chosen and bucket_limits.get(buckets.get(n, 4), 0) > 0]
    return chosen


def _generate_tickets_bucket_decade(
    stats: Dict[int, NumberStats],
    num_scores: Dict[int, float],
    pair_pmi: Dict[Tuple[int, int], float],
    triplet_counts: Dict[Tuple[int, int, int], int],
    dist: Dict[str, object],
    last_draw: List[int],
    recent_draws: List[List[int]],
    buckets: Dict[int, int],
    decade_vec: Dict[int, int],
    bucket_template: Dict[int, int],
    ticket_count: int,
    rng: random.Random,
) -> List[List[int]]:
    pools = _build_decade_bucket_pools(num_scores, buckets)
    candidates = []
    target_candidates = ticket_count * CANDIDATE_MULTIPLIER_BUCKET
    attempts = 0
    while len(candidates) < target_candidates and attempts < target_candidates * 50:
        attempts += 1
        bucket_limits = dict(bucket_template)
        chosen: List[int] = []
        ok = True
        for dec_k, _, _ in DECADE_BANDS:
            need = decade_vec.get(dec_k, 0)
            if need <= 0:
                continue
            # combine all buckets for this decade, but enforce bucket limits
            focus = _decade_focus_pool(stats, dec_k)
            dec_pool = focus + [n for n in (pools[dec_k][1] + pools[dec_k][2] + pools[dec_k][3] + pools[dec_k][4]) if n not in focus]
            locked = _decade_lock_numbers(stats, dec_k)
            locked = [n for n in locked if n in dec_pool and bucket_limits.get(buckets.get(n, 4), 0) > 0]
            # apply locks if possible
            for n in locked[:need]:
                b = buckets.get(n, 4)
                if bucket_limits.get(b, 0) <= 0:
                    continue
                chosen.append(n)
                bucket_limits[b] -= 1
            remaining = need - sum(1 for n in chosen if n in dec_pool)
            picks = _sample_from_pool_with_bucket_limits(dec_pool, bucket_limits, buckets, num_scores, rng, remaining)
            if len(picks) != need:
                ok = False
                break
            chosen.extend(picks)
        if not ok or len(chosen) != NUMBERS_PER_TICKET:
            continue
        pick = sorted(chosen)
        if pick in candidates:
            continue
        candidates.append(pick)

    scored = []
    for t in candidates:
        s = _ticket_score(t, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:ticket_count]]


def _weighted_sample_no_replace(items: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    keyed = []
    for x, w in zip(items, weights):
        w = max(1e-9, float(w))
        u = rng.random()
        key = math.log(u) / w
        keyed.append((key, x))
    keyed.sort(reverse=True)
    return [x for _, x in keyed[:k]]


def _ticket_penalty(nums: List[int], dist: Dict[str, object], last_draw: List[int], recent_draws: List[List[int]], triplet_counts: Dict[Tuple[int, int, int], int]) -> float:
    penalty = 0.0
    s = sum(nums)
    if s < dist["sum_lo"]:
        penalty += PEN_SUM_W * (dist["sum_lo"] - s) / 10.0
    elif s > dist["sum_hi"]:
        penalty += PEN_SUM_W * (s - dist["sum_hi"]) / 10.0

    odd_ct = sum(1 for n in nums if n % 2 == 1)
    penalty += PEN_ODD_W * abs(odd_ct - dist["odd_mean"])

    low_ct = sum(1 for n in nums if n <= 22)
    penalty += PEN_LOW_W * abs(low_ct - dist["low_mean"])

    consec = _count_consecutive_pairs(nums)
    penalty += PEN_CONSEC_W * abs(consec - dist["consec_mean"])

    vec = _decade_vector(nums)
    for k, _, _ in DECADE_BANDS:
        penalty += PEN_DECADE_W * abs(vec.get(k, 0) - dist["decade_mean"].get(k, 0.0))

    # Overlap penalties
    if last_draw:
        ol = len(set(nums).intersection(last_draw))
        if ol > 2:
            penalty += PEN_OVERLAP_LAST_W * (ol - 2)
    if recent_draws:
        max_ol = 0
        for d in recent_draws:
            ol = len(set(nums).intersection(d))
            if ol > max_ol:
                max_ol = ol
        if max_ol > 3:
            penalty += PEN_OVERLAP_RECENT_W * (max_ol - 3)

    # Triplet penalties
    hot_hits = 0
    recent_hits = 0
    triplet_total = dist.get("triplet_total", triplet_counts)
    triplet_recent_short = dist.get("triplet_recent_short", triplet_counts)
    for comb in combinations(sorted(nums), 3):
        if triplet_total.get(comb, 0) >= 6:
            hot_hits += 1
        if triplet_recent_short.get(comb, 0) >= 1:
            recent_hits += 1
    if hot_hits:
        penalty += PEN_TRIPLET_HOT_W * hot_hits
    if recent_hits:
        penalty += PEN_TRIPLET_RECENT_W * min(recent_hits, 3)

    return penalty


def _ticket_score(nums: List[int], num_scores: Dict[int, float], pair_pmi: Dict[Tuple[int, int], float], triplet_counts: Dict[Tuple[int, int, int], int], dist: Dict[str, object], last_draw: List[int], recent_draws: List[List[int]]) -> float:
    s = sum(num_scores.get(n, 0.0) for n in nums)

    # Pair bonus (recent PMI)
    pair_bonus = 0.0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            if a > b:
                a, b = b, a
            pair_bonus += pair_pmi.get((a, b), 0.0)

    # Triplet bonus (recent co-occurrence)
    trip_bonus = 0.0
    for comb in combinations(sorted(nums), 3):
        trip_bonus += min(2, triplet_counts.get(comb, 0))

    pen = _ticket_penalty(nums, dist, last_draw, recent_draws, triplet_counts)
    return s + PAIR_BONUS_W * pair_bonus + TRIPLET_BONUS_W * trip_bonus - pen


def _generate_tickets_baseline(df_hist: pd.DataFrame, main_cols: List[str], target_date: str, ticket_count: int, verbose: bool = False) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    stats, pair_pmi, triplet_counts, dist = _history_stats(df_hist, main_cols, target_date)
    num_scores = _number_scores(stats)
    buckets = _rank_buckets(num_scores)
    dist["buckets"] = buckets
    ranked = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    ugly_pool = [n for n, _ in ranked[UGLY_RANK_START - 1:UGLY_RANK_END]]

    # normalize to probability
    weights = {n: max(1e-9, num_scores.get(n, 0.0)) for n in range(1, 46)}
    w_norm = _normalize_map(weights)
    pool = list(range(1, 46))

    # last draw + recent draws for penalties
    last_draw = []
    recent_draws = []
    if not df_hist.empty:
        last_row = df_hist.iloc[-1]
        last_draw = [int(last_row[c]) for c in main_cols]
        recent = df_hist.tail(OVERLAP_RECENT_WINDOW)
        for _, row in recent.iterrows():
            recent_draws.append([int(row[c]) for c in main_cols])

    candidates = []
    target_candidates = ticket_count * CANDIDATE_MULTIPLIER
    attempts = 0
    while len(candidates) < target_candidates and attempts < target_candidates * 50:
        attempts += 1
        chosen = []
        if ugly_pool and rng.random() < UGLY_MODE_SHARE:
            chosen.append(rng.choice(ugly_pool))

        available = [n for n in pool if n not in chosen]
        for _ in range(NUMBERS_PER_TICKET):
            if not available:
                break
            weights_list = []
            for n in available:
                w = w_norm.get(n, 1e-9)
                if chosen:
                    # bias by pair PMI with chosen set
                    pair_score = 0.0
                    for m in chosen:
                        a, b2 = (m, n) if m < n else (n, m)
                        pair_score += pair_pmi.get((a, b2), 0.0)
                    w *= math.exp(0.10 * pair_score)
                weights_list.append(max(1e-9, w))
            pick = _weighted_sample_no_replace(available, weights_list, 1, rng)[0]
            chosen.append(pick)
            available = [n for n in available if n not in chosen]
        if len(chosen) != NUMBERS_PER_TICKET:
            continue
        pick = sorted(chosen)
        if pick in candidates:
            continue
        candidates.append(pick)

    # score and select
    scored = []
    for t in candidates:
        s = _ticket_score(t, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
        scored.append((s, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [t for _, t in scored[:ticket_count]]

    if verbose:
        print(f"Generated {len(out)}/{ticket_count} tickets from {len(candidates)} candidates in {attempts} attempts")
        print("Top ranked:", ", ".join(f"{n}:{v:.3f}" for n, v in sorted(num_scores.items(), key=lambda x: x[1], reverse=True)[:10]))
        print(f"Ugly pool (ranks {UGLY_RANK_START}-{UGLY_RANK_END}) size={len(ugly_pool)} share={UGLY_MODE_SHARE}")
        _log_stats_snapshot(stats, dist)
        print("\n=== TOP TICKET DIAGNOSTICS ===")
        for i, t in enumerate(out[:LOG_TOP_TICKETS], 1):
            d = _ticket_diagnostics(t, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws)
            print(
                f"#{i:02d} {t} score_sum={d['score_sum']:.3f} pair={d['pair_bonus']:.3f} "
                f"trip={d['trip_bonus']:.3f} sum={d['sum']} odd={d['odd']} low={d['low']} "
                f"consec={d['consec']} last_ol={d['last_ol']} max_ol={d['max_ol']} "
                f"hot_tr={d['hot_triplets']} rec_tr={d['recent_triplets']} decades={d['decade_vec']}"
            )

    return out


def _generate_tickets(df_hist: pd.DataFrame, main_cols: List[str], target_date: str, ticket_count: int, verbose: bool = False) -> List[List[int]]:
    if not USE_BUCKET_TEMPLATES:
        return _generate_tickets_baseline(df_hist, main_cols, target_date, ticket_count, verbose)

    rng = random.Random(RANDOM_SEED)
    stats, pair_pmi, triplet_counts, dist = _history_stats(df_hist, main_cols, target_date)
    num_scores = _number_scores(stats)
    buckets = _rank_buckets(num_scores)
    dist["buckets"] = buckets

    last_draw = []
    recent_draws = []
    if not df_hist.empty:
        last_row = df_hist.iloc[-1]
        last_draw = [int(last_row[c]) for c in main_cols]
        recent = df_hist.tail(OVERLAP_RECENT_WINDOW)
        for _, row in recent.iterrows():
            recent_draws.append([int(row[c]) for c in main_cols])

    decade_counts = _recent_decade_vectors(df_hist, main_cols)
    decade_vecs = []
    for vec, _cnt in decade_counts:
        decade_vecs.append({k: vec[i] for i, (k, _, _) in enumerate(DECADE_BANDS)})
    if not decade_vecs:
        decade_vecs = [{k: 1 for k, _, _ in DECADE_BANDS}]

    thresholds = _regime_thresholds(df_hist, main_cols, buckets)
    last_b4 = sum(1 for n in last_draw if buckets.get(n, 4) == 4) if last_draw else 0
    top_share_now = thresholds["top_share_now"]
    # Stronger regime signal: use last_draw bucket4 count as primary
    if last_b4 <= 1:
        weights = TEMPLATE_WEIGHTS_HOT
        regime = "hot_last_b4"
    elif last_b4 >= 3:
        weights = TEMPLATE_WEIGHTS_COLD
        regime = "cold_last_b4"
    elif top_share_now >= thresholds["top_hot"]:
        weights = TEMPLATE_WEIGHTS_HOT
        regime = "hot_share"
    elif top_share_now <= thresholds["top_cold"]:
        weights = TEMPLATE_WEIGHTS_COLD
        regime = "cold_share"
    else:
        weights = TEMPLATE_WEIGHTS_NEUTRAL
        regime = "neutral"

    # allocate ticket quotas per template
    template_counts = []
    remaining = ticket_count
    for w in weights:
        c = int(round(w * ticket_count))
        template_counts.append(c)
        remaining -= c
    # adjust to exact total
    idx = 0
    while remaining != 0:
        template_counts[idx % len(template_counts)] += 1 if remaining > 0 else -1
        remaining += -1 if remaining > 0 else 1
        idx += 1

    out: List[List[int]] = []
    if USE_COVER_WHEEL:
        out.extend(
            _generate_tickets_cover_wheel(
                stats, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws, buckets, ticket_count
            )
        )
        if verbose:
            print(f"Cover wheel used: pool_size={COVER_POOL_SIZE} select_weight={COVER_SELECT_WEIGHT}")
        return out

    if regime.startswith("hot"):
        out.extend(
            _generate_tickets_bucket_only(
                stats,
                num_scores,
                pair_pmi,
                triplet_counts,
                dist,
                last_draw,
                recent_draws,
                buckets,
                HOT_BUCKET_TEMPLATE,
                ticket_count,
                rng,
                lock_decades=False,
            )
        )
        if verbose:
            print("Hot regime: bucket-only top-heavy (no decade locks)")
        return out

    if USE_ENSEMBLE:
        out = []
        # baseline
        out.extend(_generate_tickets_baseline(df_hist, main_cols, target_date, ENSEMBLE_COUNTS["baseline"], verbose=False))
        # bucket templates with decade conditioning
        out.extend(
            _generate_tickets_bucket_decade(
                stats, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws, buckets,
                decade_vecs[0] if decade_vecs else {k:1 for k,_,_ in DECADE_BANDS},
                BUCKET_TEMPLATES[0], ENSEMBLE_COUNTS["bucket_top"], rng
            )
        )
        out.extend(
            _generate_tickets_bucket_decade(
                stats, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws, buckets,
                decade_vecs[0] if decade_vecs else {k:1 for k,_,_ in DECADE_BANDS},
                BUCKET_TEMPLATES[1], ENSEMBLE_COUNTS["bucket_mid"], rng
            )
        )
        out.extend(
            _generate_tickets_bucket_decade(
                stats, num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws, buckets,
                decade_vecs[0] if decade_vecs else {k:1 for k,_,_ in DECADE_BANDS},
                BUCKET_TEMPLATES[2], ENSEMBLE_COUNTS["bucket_bottom"], rng
            )
        )
        # overlap-based
        out.extend(
            _generate_tickets_with_overlap(
                num_scores, pair_pmi, triplet_counts, dist, last_draw, recent_draws, 2, ENSEMBLE_COUNTS["overlap2"], rng
            )
        )
        # de-dup and trim
        uniq = []
        seen = set()
        for t in out:
            key = tuple(t)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        out = uniq[:ticket_count]
        if len(out) < ticket_count:
            fill = ticket_count - len(out)
            out.extend(_generate_tickets_baseline(df_hist, main_cols, target_date, fill, verbose=False))
        if verbose:
            print(f"Ensemble used: counts={ENSEMBLE_COUNTS} total={len(out)}")
        return out
    decade_vecs_use = decade_vecs[:1] if decade_vecs else decade_vecs
    for bucket_template, tcount in zip(BUCKET_TEMPLATES, template_counts):
        if tcount <= 0:
            continue
        per_combo = max(1, int(round(tcount / max(1, len(decade_vecs_use)))))
        for decade_vec in decade_vecs_use:
            out.extend(
                _generate_tickets_bucket_decade(
                    stats,
                    num_scores,
                    pair_pmi,
                    triplet_counts,
                    dist,
                    last_draw,
                    recent_draws,
                    buckets,
                    decade_vec,
                    bucket_template,
                    per_combo,
                    rng,
                )
            )

    # de-dup and fill if short
    uniq = []
    seen = set()
    for t in out:
        key = tuple(t)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    out = uniq[:ticket_count]
    if len(out) < ticket_count:
        fill = ticket_count - len(out)
        out.extend(_generate_tickets_baseline(df_hist, main_cols, target_date, fill, verbose=False))

    if verbose:
        print("Decade vectors (recent top):", ", ".join(f"{vec}:{cnt}" for vec, cnt in decade_counts))
        if decade_vecs_use != decade_vecs:
            print("Decade vectors used:", ", ".join(str(_decade_vector_tuple(v)) for v in decade_vecs_use))
        print("Bucket templates:", BUCKET_TEMPLATES)
        print(f"Regime={regime} top_share_now={top_share_now:.3f} thresholds={thresholds} last_b4={last_b4} template_counts={template_counts}")
    return out


def _hit_summary(real: List[int], tickets: List[List[int]]) -> Dict[str, int]:
    rd = set(real)
    max_hit = 0
    ge3 = 0
    for t in tickets:
        hits = len(set(t).intersection(rd))
        if hits >= 3:
            ge3 += 1
        if hits > max_hit:
            max_hit = hits
    return {"ge3": ge3, "max_hit": max_hit}


def show_ticket_hits(real_draw: List[int], tickets: List[List[int]], draw_date: str = None, strategy_name: str = None) -> None:
    if not real_draw:
        return
    rd = sorted(real_draw)
    rd_set = set(rd)
    label_parts = []
    if strategy_name:
        label_parts.append(str(strategy_name))
    if draw_date:
        label_parts.append(str(draw_date))
    suffix = f" ({' | '.join(label_parts)})" if label_parts else ""
    print(f"\n=== REAL DRAW HIT SUMMARY{suffix} ===")
    print(f"REAL_DRAW: {rd}")

    hit_counts: Dict[int, int] = {}
    best = (-1, None)
    for i, t in enumerate(tickets, 1):
        hits = sorted(set(t).intersection(rd_set))
        hit_n = len(hits)
        hit_counts[hit_n] = hit_counts.get(hit_n, 0) + 1
        if hit_n > best[0]:
            best = (hit_n, i)
        if hit_n >= 3:
            print(f"Ticket #{i:02d}: hits={hit_n} nums={hits}")
    if best[0] < 3:
        print("No tickets with 3+ hits.")
    if best[1] is not None:
        print(f"Best ticket: #{best[1]:02d} hits={best[0]}")


def _rank_hits(real: List[int], num_scores: Dict[int, float]) -> Dict[str, int]:
    ranked = sorted(num_scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {n: i + 1 for i, (n, _) in enumerate(ranked)}
    top10 = sum(1 for n in real if ranks.get(n, 999) <= 10)
    top20 = sum(1 for n in real if ranks.get(n, 999) <= 20)
    top30 = sum(1 for n in real if ranks.get(n, 999) <= 30)
    return {"top10": top10, "top20": top20, "top30": top30}


def _cold_hits(real: List[int], stats: Dict[int, NumberStats], k: int = 12) -> int:
    gaps = sorted([(n, stats[n].gap_days) for n in stats], key=lambda x: x[1], reverse=True)
    cold = set(n for n, _ in gaps[:k])
    return sum(1 for n in real if n in cold)


def _max_overlap_with_recent(real: List[int], recent_draws: List[List[int]]) -> int:
    if not recent_draws:
        return 0
    rd = set(real)
    max_ol = 0
    for d in recent_draws:
        ol = len(rd.intersection(d))
        if ol > max_ol:
            max_ol = ol
    return max_ol


def main() -> None:
    df, main_cols = _load_data()
    run_date = TARGET_DATE

    if "Date" in df.columns:
        t_target = pd.Timestamp(TARGET_DATE)
        if pd.isna(t_target):
            raise ValueError("TARGET_DATE must be parseable")

    # Backtest window
    print(f"\n=== BACKTEST (LAST {N} DRAWS) ===")
    bt_rows = df.sort_values("Date").tail(N)
    bt_dates = [row["Date"] for _, row in bt_rows.iterrows()]

    bt_total_ge3 = 0
    bt_weeks_with_5 = 0
    bt_max_hit = 0
    print("Date        | ge3 | max_hit")
    print("------------+-----+--------")

    for d in bt_dates:
        bt_date = d.strftime("%Y-%m-%d")
        hist = df[df["Date"] < d]
        row = df[df["Date"] == d].iloc[0]
        real = [int(row[c]) for c in main_cols]
        stats, pair_pmi, triplet_counts, dist = _history_stats(hist, main_cols, bt_date)
        num_scores = _number_scores(stats)
        buckets = _rank_buckets(num_scores)
        dist["buckets"] = buckets
        rh = _rank_hits(real, num_scores)
        ch = _cold_hits(real, stats, k=12)
        bucket_counts = Counter(buckets.get(n, 4) for n in real)
        print(
            f"{bt_date} | {rh['top10']}t10 {rh['top20']}t20 {rh['top30']}t30 cold12={ch} "
            f"bkt={dict(bucket_counts)}",
        )

        tickets = _generate_tickets(hist, main_cols, bt_date, BACKTEST_TICKET_COUNT, verbose=VERBOSE_BACKTEST)
        summary = _hit_summary(real, tickets)
        bt_total_ge3 += summary["ge3"]
        if summary["max_hit"] >= 5:
            bt_weeks_with_5 += 1
        if summary["max_hit"] > bt_max_hit:
            bt_max_hit = summary["max_hit"]
        print(f"{bt_date} | {summary['ge3']:>3} | {summary['max_hit']:>7}")

        # Decade diagnostics for prediction
        real_dec = _decade_vector(real)
        last_draw = []
        recent_draws = []
        if not hist.empty:
            last_row = hist.iloc[-1]
            last_draw = [int(last_row[c]) for c in main_cols]
            recent = hist.tail(RECENT_WINDOW_DRAWS)
            for _, row2 in recent.iterrows():
                recent_draws.append([int(row2[c]) for c in main_cols])
        last_dec = _decade_vector(last_draw) if last_draw else {k: 0 for k, _, _ in DECADE_BANDS}
        recent_dec_avg = _avg_decade_vector([_decade_vector(r) for r in recent_draws])
        ticket_dec_vecs = [_decade_vector(t) for t in tickets]
        ticket_dec_avg = _avg_decade_vector(ticket_dec_vecs)
        mode_counter = Counter(_decade_vector_tuple(v) for v in ticket_dec_vecs)
        mode_dec_tuple = mode_counter.most_common(1)[0][0] if mode_counter else _decade_vector_tuple({k: 0 for k, _, _ in DECADE_BANDS})
        mode_dec = {k: mode_dec_tuple[i] for i, (k, _, _) in enumerate(DECADE_BANDS)}
        best_dec_match = max((_decade_overlap_score(real_dec, v) for v in ticket_dec_vecs), default=0)
        print(
            f"  DECADE real={_format_decade_vec(real_dec)} last={_format_decade_vec(last_dec)} "
            f"recent_avg={_format_decade_avg(recent_dec_avg)} ticket_avg={_format_decade_avg(ticket_dec_avg)} "
            f"mode={_format_decade_vec(mode_dec)} best_match={best_dec_match}"
        )
        recent_dec_top = _recent_decade_vectors(hist, main_cols)
        if recent_dec_top:
            print("  DECADE top_recent:", ", ".join(f"{vec}:{cnt}" for vec, cnt in recent_dec_top))

        # Detailed log: real draw profile + best ticket(s)
        print(f"  REAL: {_number_profile(real, stats, num_scores)}")
        best_hit = -1
        best_tickets = []
        for t in tickets:
            hits = len(set(t).intersection(real))
            if hits > best_hit:
                best_hit = hits
                best_tickets = [t]
            elif hits == best_hit:
                best_tickets.append(t)
        for idx, t in enumerate(best_tickets[:3], 1):
            d = _ticket_diagnostics(t, num_scores, pair_pmi if 'pair_pmi' in locals() else {}, triplet_counts if 'triplet_counts' in locals() else Counter(), dist, last_draw, recent_draws)
            hit_nums = sorted(set(t).intersection(real))
            diag_str = f"sum={d['sum']} odd={d['odd']} low={d['low']} consec={d['consec']} last_ol={d['last_ol']} max_ol={d['max_ol']}"
            print(f"  BEST#{idx}: hits={best_hit} nums={t} hitnums={hit_nums} diag=({diag_str})")

    print("\n=== BACKTEST HIT SUMMARY ===")
    print(f"Total ge3 tickets: {bt_total_ge3}")
    print(f"Weeks with 5+ hits: {bt_weeks_with_5}")
    print(f"Max hit observed: {bt_max_hit}")

    # Target
    print("\n=== TARGET (NEW ALGO) ===")
    if pd.Timestamp(TARGET_DATE) not in set(df["Date"]):
        print("TARGET_DATE not found in CSV; generating prediction without hit summary.")
    hist = df[df["Date"] < pd.Timestamp(TARGET_DATE)]
    # Decade priors for target
    last_draw = []
    recent_draws = []
    if not hist.empty:
        last_row = hist.iloc[-1]
        last_draw = [int(last_row[c]) for c in main_cols]
        recent = hist.tail(RECENT_WINDOW_DRAWS)
        for _, row2 in recent.iterrows():
            recent_draws.append([int(row2[c]) for c in main_cols])
    last_dec = _decade_vector(last_draw) if last_draw else {k: 0 for k, _, _ in DECADE_BANDS}
    recent_dec_avg = _avg_decade_vector([_decade_vector(r) for r in recent_draws])
    print(f"Target decade priors: last={_format_decade_vec(last_dec)} recent_avg={_format_decade_avg(recent_dec_avg)}")
    # Regime preview for target
    stats_t, pair_pmi_t, triplet_counts_t, dist_t = _history_stats(hist, main_cols, run_date)
    num_scores_t = _number_scores(stats_t)
    buckets_t = _rank_buckets(num_scores_t)
    thresholds_t = _regime_thresholds(hist, main_cols, buckets_t)
    last_b4_t = sum(1 for n in last_draw if buckets_t.get(n, 4) == 4) if last_draw else 0
    top_share_now_t = thresholds_t["top_share_now"]
    print(f"Target regime: top_share_now={top_share_now_t:.3f} thresholds={thresholds_t} last_b4={last_b4_t}")
    tickets = _generate_tickets(hist, main_cols, run_date, NUM_TICKETS, verbose=VERBOSE_TARGET)
    print(f"Target: {run_date}")
    for i, t in enumerate(tickets, 1):
        vec = _decade_vector(t)
        print(f"Ticket #{i:02d}: {t} decades={vec}")
    show_ticket_hits(REAL_DRAW_TARGET, tickets, draw_date=run_date, strategy_name="NEW_ALGO")


if __name__ == "__main__":
    main()
