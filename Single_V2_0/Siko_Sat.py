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
TARGET_DATE = "2026-1-17"
REAL_DRAW_TARGET = [8, 9, 19, 35, 38, 44]
N = 10

NUM_TICKETS = 20
NUMBERS_PER_TICKET = 6
BACKTEST_TICKET_COUNT = 20
CANDIDATE_MULTIPLIER = 8
RANDOM_SEED = 0

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

    # Triplets (recent window)
    triplet_counts = Counter()
    for _, row in recent.iterrows():
        nums = sorted(int(row[c]) for c in main_cols)
        for comb in combinations(nums, 3):
            triplet_counts[comb] += 1

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
    for comb in combinations(sorted(nums), 3):
        if triplet_counts.get(comb, 0) >= 6:
            hot_hits += 1
        if triplet_counts.get(comb, 0) >= 1:
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


def _generate_tickets(df_hist: pd.DataFrame, main_cols: List[str], target_date: str, ticket_count: int) -> List[List[int]]:
    rng = random.Random(RANDOM_SEED)
    stats, pair_pmi, triplet_counts, dist = _history_stats(df_hist, main_cols, target_date)
    num_scores = _number_scores(stats)

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
                        a, b = (m, n) if m < n else (n, m)
                        pair_score += pair_pmi.get((a, b), 0.0)
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
    return [t for _, t in scored[:ticket_count]]


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
        tickets = _generate_tickets(hist, main_cols, bt_date, BACKTEST_TICKET_COUNT)
        summary = _hit_summary(real, tickets)
        bt_total_ge3 += summary["ge3"]
        if summary["max_hit"] >= 5:
            bt_weeks_with_5 += 1
        if summary["max_hit"] > bt_max_hit:
            bt_max_hit = summary["max_hit"]
        print(f"{bt_date} | {summary['ge3']:>3} | {summary['max_hit']:>7}")

    print("\n=== BACKTEST HIT SUMMARY ===")
    print(f"Total ge3 tickets: {bt_total_ge3}")
    print(f"Weeks with 5+ hits: {bt_weeks_with_5}")
    print(f"Max hit observed: {bt_max_hit}")

    # Target
    print("\n=== TARGET (NEW ALGO) ===")
    if pd.Timestamp(TARGET_DATE) not in set(df["Date"]):
        print("TARGET_DATE not found in CSV; generating prediction without hit summary.")
    hist = df[df["Date"] < pd.Timestamp(TARGET_DATE)]
    tickets = _generate_tickets(hist, main_cols, run_date, NUM_TICKETS)
    print(f"Target: {run_date}")
    for i, t in enumerate(tickets, 1):
        vec = _decade_vector(t)
        print(f"Ticket #{i:02d}: {t} decades={vec}")
    show_ticket_hits(REAL_DRAW_TARGET, tickets, draw_date=run_date, strategy_name="NEW_ALGO")


if __name__ == "__main__":
    main()
