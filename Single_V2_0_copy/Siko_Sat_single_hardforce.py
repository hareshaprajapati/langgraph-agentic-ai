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
    "siko_sat_single_logs.log"   # single growing log file
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

CSV_PATH = "Tattslotto.csv"
TARGET_DATE = "2025-12-27"

NUM_TICKETS = 20
NUMBERS_PER_TICKET = 6

MAIN_MIN = 1
MAIN_MAX = 45

LOOKBACK_DAYS = 150
SEASON_WINDOW_DAYS = 1
SEASON_LOOKBACK_YEARS = 20

# Candidate pool
POOL_SIZE = 40
MID_POOL_SIZE = 15
COLD_POOL_SIZE = 15
HOT_POOL_SIZE = 12
OVERDUE_POOL_SIZE = 12
SEASON_POOL_SIZE = 12
COLD_FORCE_COUNT = 1

# Hard-force coverage mix
FORCE_COVERAGE = True
RANDOM_SEED = 0
DEBUG_PRINT = True

# Score weights (date-agnostic)
W_RECENT = 0.45
W_LONG = 0.25
W_SEASON = 0.20
W_RANK = 0.10
COLD_BOOST = 0.35
# Overdue gap boost (date-agnostic)
W_GAP = 0.35
GAP_CAP = 0.40

# Ticket constraints (soft)
OVERLAP_CAP = 4
GLOBAL_MAX_USES = 4

# Odd / low / sum preferences (learned from history)
ODD_BAND = (2, 4)
LOW_RANGE_MAX = 22
LOW_BAND = (2, 4)
SUM_BAND_QUANTILES = (0.25, 0.75)
CONSECUTIVE_MAX = 1

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
PENALTY_SCALE = 0.5
MAX_ATTEMPTS = 30000

# Optional: verify against a known real draw (set [] to disable)
REAL_DRAW = [3, 5, 20, 26, 28, 40]

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

    if debug:
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
) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    dist = _history_distributions(df[df["Date"] < pd.Timestamp(target_date)], main_cols, target_date)

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
        if rng.random() > accept_prob:
            continue

        tickets.append(pick)
        for n in pick:
            global_use[n] += 1

    print(f"Generated {len(tickets)}/{NUM_TICKETS} tickets in {attempts} attempts")
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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df, main_cols = _load_csv(CSV_PATH)

    # Run for target date and previous two draws (by Date in CSV)
    t_target = pd.Timestamp(TARGET_DATE)
    df_dates = df[df["Date"] <= t_target].sort_values("Date")
    last_three = df_dates.tail(3)

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

    for _, row in last_three.iterrows():
        run_date = row["Date"].strftime("%Y-%m-%d")
        real_draw = [int(row[c]) for c in main_cols]
        scored = score_numbers(df, main_cols, run_date, DEBUG_PRINT)

        tickets = generate_tickets(scored, df, main_cols, run_date,
                                   use_weights=True, seed_hot_overdue=False,
                                   force_coverage=True)

        print("\n=== HARD_FORCE STRATEGY ===")
        print(f"Target: {run_date}")
        print(f"Tickets: {NUM_TICKETS} | Pool size: {POOL_SIZE} + mid {MID_POOL_SIZE} + cold {COLD_POOL_SIZE}")
        print(f"Decade bands: {DECADE_BANDS}")

        for i, t in enumerate(tickets, 1):
            vec = _decade_vector(t)
            print(f"Ticket #{i:02d}: {t}  decades={vec}")

        show_ticket_hits(real_draw, tickets)
