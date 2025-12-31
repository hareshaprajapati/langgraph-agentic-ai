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
TARGET_DATE = "2026-01-01"   # any date, learns season around this month/day

NUM_TICKETS = 10
NUMBERS_PER_TICKET = 7

MAIN_MIN = 1
MAIN_MAX = 35

LOOKBACK_DAYS = 365
SEASON_WINDOW_DAYS = 6
SEASON_LOOKBACK_YEARS = 20
MIN_SEASON_SAMPLES = 50

RECENT_DRAWS_PENALTY_N = 6

# Ticket diversification
TOP_POOL = 26
OVERLAP_CAP = 5
RANDOM_SEED = 42
DEBUG_PRINT = True

# --- decade constraint strictness ---
DECADE_MODE = "hard"         # "hard" or "soft"
DECADE_MEDIAN_TOL = 1        # allow +/-1 per decade around seasonal p25..p75 (hard mode)
DECADE_SOFT_PENALTY = 0.6    # per unit distance outside tolerance (soft mode)

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

        print("\n=== TOP 20 SCORED NUMBERS ===")
        for c in scored[:20]:
            print(c)

    return scored, season_profile, season_decades


def _weighted_sample_no_replace(items: List[int], weights: List[float], k: int, rng: random.Random) -> List[int]:
    keyed = []
    for x, w in zip(items, weights):
        w = max(1e-9, float(w))
        u = rng.random()
        key = math.log(u) / w
        keyed.append((key, x))
    keyed.sort(reverse=True)
    return [x for _, x in keyed[:k]]


def generate_tickets(scored: List[CandidateScoreMain], season_decades: SeasonDecadeProfile):
    rng = random.Random(RANDOM_SEED)

    pool = scored[:TOP_POOL]
    items = [c.n for c in pool]

    min_score = min(c.total_score for c in pool)
    base_weights = [(c.total_score - min_score) + 0.25 for c in pool]

    tickets: List[List[int]] = []
    attempts = 0
    max_attempts = 12000

    while len(tickets) < NUM_TICKETS and attempts < max_attempts:
        attempts += 1

        pick = _weighted_sample_no_replace(items, base_weights, NUMBERS_PER_TICKET, rng)
        pick = sorted(pick)

        # overlap cap
        ok = True
        s_pick = set(pick)
        for t in tickets:
            if len(s_pick.intersection(t)) > OVERLAP_CAP:
                ok = False
                break
        if not ok:
            continue

        # decade constraint
        vec = _decade_vector(pick)
        dist = _decade_distance(vec, season_decades.med)
        in_band = _within_decade_band(vec, season_decades.p25, season_decades.p75, tol=DECADE_MEDIAN_TOL)

        if DECADE_MODE == "hard":
            if not in_band:
                continue
            if dist > (2 * DECADE_MEDIAN_TOL):
                continue
        else:
            reject_prob = min(0.85, dist * DECADE_SOFT_PENALTY * 0.15)
            if rng.random() < reject_prob:
                continue

        tickets.append(pick)

    if len(tickets) < NUM_TICKETS:
        raise RuntimeError(
            f"Could only generate {len(tickets)} tickets. "
            f"Try increasing TOP_POOL or increasing DECADE_MEDIAN_TOL or switching DECADE_MODE='soft'."
        )

    return tickets


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    scored, season_profile, season_decades = score_main_numbers(
        target_date=TARGET_DATE,
        csv_path=CSV_PATH,
        debug=DEBUG_PRINT,
    )

    tickets = generate_tickets(scored, season_decades)

    print("\n=== FINAL TICKETS (season-aware decades) ===")
    print(f"Target: {TARGET_DATE} | Tickets: {NUM_TICKETS} | Pool: top{TOP_POOL} | Overlap cap: {OVERLAP_CAP}")
    print(f"Decade mode: {DECADE_MODE} | tol={DECADE_MEDIAN_TOL}")
    print(f"Decade bands: {DECADE_BANDS}")
    print(f"Season median decades: {season_decades.med}")
    print(f"Season p25 decades:    {season_decades.p25}")
    print(f"Season p75 decades:    {season_decades.p75}")

    for i, t in enumerate(tickets, 1):
        vec = _decade_vector(t)
        print(f"Ticket #{i:02d}: {t}  decades={vec}")
