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
    f"siko_pb_logs.log"   # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import pandas as pd
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Tuple, Optional

# ----------------------------
# Config / knobs (tuneable)
# ----------------------------

PB_MIN = 1
PB_MAX = 20

DEFAULT_LOOKBACK_DAYS = 365          # "last 12 months"
DEFAULT_SEASON_WINDOW_DAYS = 6       # +/- days around target month/day
DEFAULT_SEASON_LOOKBACK_YEARS = 20   # how many past years to consult
DEFAULT_RECENT_DRAWS_PENALTY_N = 6   # penalize PBs seen in last N draws
DEFAULT_SAME_MONTH_CLUSTER_THRESHOLD = 2  # if PB appears >=2 times in target month, penalize/exclude

# Target ratio band from our analysis:
# Seasonal/Xmas PB often sits around ~60–85% of the max 12mo frequency,
# and tends to be rank 2–5 rather than rank 1.
RATIO_IDEAL = 0.72
RATIO_BAND_LOW = 0.60
RATIO_BAND_HIGH = 0.85


@dataclass
class CandidateScore:
    pb: int
    total_score: float
    recent_12mo_freq: int
    recent_12mo_rank: int
    recent_12mo_ratio: float
    seasonal_count: int
    seasonal_success_count: int
    penalties: Dict[str, float]
    components: Dict[str, float]

@dataclass
class SeasonProfile:
    """Learned, date-specific behavior profile based on historical draws near the same month/day."""
    anchor_month: int
    anchor_day: int
    sample_years: int
    sample_draws: int

    # Learned bands for "typical" winners in this seasonal window (based on each draw's prior 12 months)
    ratio_low: float
    ratio_high: float
    ratio_ideal: float

    rank_low: int
    rank_high: int

    # How often the winner was the #1 leader in its prior-12mo table (season-specific)
    leader_rate: float

    # Debug summary stats
    ratio_median: float
    ratio_p25: float
    ratio_p75: float
    rank_median: float
    rank_p25: float
    rank_p75: float


def _learn_season_profile(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    lookback_days: int,
    season_window_days: int,
    season_lookback_years: int,
    min_samples: int = 5,
) -> SeasonProfile:
    """
    Learns the seasonal PB behavior for the target month/day by scanning past years around that calendar date.

    For each historical draw date d within +/- season_window_days around the anchor (same month/day) for year y,
    compute the winner PB's (freq, rank, ratio) in the prior 12 months window ending at d.
    Then learn typical bands using percentiles.

    If insufficient samples, falls back to global Xmas-tuned defaults (RATIO_BAND_LOW/HIGH and rank 2..5).
    """
    target_d = target_ts.date()
    start_year = target_d.year - season_lookback_years
    end_year = target_d.year - 1

    ratios = []
    ranks = []
    leader_flags = []
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
            pb = int(row["PB"])
            # Prior-12mo window ending at d
            hist_train = df[df["Date"] < d]
            hist_start = d - pd.Timedelta(days=lookback_days)
            hist_counts = _counts_in_window(hist_train, hist_start, d)
            if hist_counts.empty:
                continue
            hist_counts = hist_counts.reindex(range(PB_MIN, PB_MAX + 1), fill_value=0)
            hist_max = int(hist_counts.max())
            if hist_max <= 0:
                continue
            hist_ranks = _rank_from_counts(hist_counts)
            hist_freq = int(hist_counts.get(pb, 0))
            hist_rank = int(hist_ranks.get(pb, 999))
            hist_ratio = (hist_freq / hist_max) if hist_max > 0 else 0.0

            ratios.append(hist_ratio)
            ranks.append(hist_rank)
            leader_flags.append(1 if hist_rank == 1 else 0)
            draw_count += 1

    # Fallback if too few samples
    if len(ratios) < min_samples or len(ranks) < min_samples:
        return SeasonProfile(
            anchor_month=target_d.month,
            anchor_day=target_d.day,
            sample_years=len(seen_years),
            sample_draws=draw_count,
            ratio_low=RATIO_BAND_LOW,
            ratio_high=RATIO_BAND_HIGH,
            ratio_ideal=RATIO_IDEAL,
            rank_low=2,
            rank_high=5,
            leader_rate=float(sum(leader_flags) / len(leader_flags)) if leader_flags else 0.0,
            ratio_median=float(pd.Series(ratios).median()) if ratios else RATIO_IDEAL,
            ratio_p25=float(pd.Series(ratios).quantile(0.25)) if ratios else RATIO_BAND_LOW,
            ratio_p75=float(pd.Series(ratios).quantile(0.75)) if ratios else RATIO_BAND_HIGH,
            rank_median=float(pd.Series(ranks).median()) if ranks else 3.0,
            rank_p25=float(pd.Series(ranks).quantile(0.25)) if ranks else 2.0,
            rank_p75=float(pd.Series(ranks).quantile(0.75)) if ranks else 5.0,
        )

    s_ratios = pd.Series(ratios)
    s_ranks = pd.Series(ranks)

    ratio_p25 = float(s_ratios.quantile(0.25))
    ratio_p75 = float(s_ratios.quantile(0.75))
    ratio_med = float(s_ratios.median())

    rank_p25 = float(s_ranks.quantile(0.25))
    rank_p75 = float(s_ranks.quantile(0.75))
    rank_med = float(s_ranks.median())

    # Learned bands (clamped)
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
        ratio_median=ratio_med,
        ratio_p25=ratio_p25,
        ratio_p75=ratio_p75,
        rank_median=rank_med,
        rank_p25=rank_p25,
        rank_p75=rank_p75,
    )



def _parse_date(d: str) -> pd.Timestamp:
    # CSV uses "DD/MM/YYYY" in Lotterywest export
    return pd.to_datetime(d, dayfirst=True, errors="coerce")


def _load_powerball_csv(csv_path: str) -> pd.DataFrame:
    """
    ✅ Source-of-truth column mapping:
      - Reads 'Powerball Number' from CSV and stores into df['PB'] (int)
      - Reads 'Draw date' from CSV and stores into df['Date'] (datetime)
    """
    df = pd.read_csv(csv_path)
    if "Draw date" not in df.columns or "Powerball Number" not in df.columns:
        raise ValueError("CSV must contain columns: 'Draw date' and 'Powerball Number'")

    df = df.copy()
    df["Date"] = df["Draw date"].apply(_parse_date)
    df = df.dropna(subset=["Date"])

    # ✅ Uses Powerball Number column
    df["PB"] = df["Powerball Number"].astype(int)

    df = df[(df["PB"] >= PB_MIN) & (df["PB"] <= PB_MAX)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _counts_in_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    # inclusive start, exclusive end
    sub = df[(df["Date"] >= start) & (df["Date"] < end)]
    return sub["PB"].value_counts().sort_index()


def _rank_from_counts(counts: pd.Series) -> Dict[int, int]:
    """
    Dense ranking: highest freq => rank 1; ties share same rank.
    Returns dict pb->rank. Missing PBs have rank very large (handled later).
    """
    if counts.empty:
        return {}

    tmp = counts.sort_values(ascending=False)
    ranks: Dict[int, int] = {}
    current_rank = 1
    last_freq = None

    for pb, freq in tmp.items():
        if last_freq is None:
            ranks[pb] = current_rank
            last_freq = freq
        else:
            if freq < last_freq:
                current_rank += 1
                last_freq = freq
            ranks[pb] = current_rank

    return ranks


def _anchor_for_year(target: date, year: int) -> date:
    """
    Build the same month/day in a given year.
    Handles Feb 29 by falling back to Feb 28.
    """
    try:
        return date(year, target.month, target.day)
    except ValueError:
        return date(year, 2, 28)


def _season_window_dates(anchor: date, window_days: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(anchor) - pd.Timedelta(days=window_days)
    end = pd.Timestamp(anchor) + pd.Timedelta(days=window_days + 1)  # inclusive by day
    return start, end


# -------------------------------------------------------------------
# Debug/Print helpers (what you requested)
# -------------------------------------------------------------------

def print_12mo_pb_table_and_ranks(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Tuple[pd.Series, Dict[int, int], int]:
    """
    Prints:
      - Full PB frequency table for the last 12 months before target_ts (PB 1..20)
      - Ratio to max
      - Rank list (dense ranks, ties share rank)

    Returns: (counts_series, ranks_dict, maxfreq)
    """
    train = df[df["Date"] < target_ts].copy()
    recent_start = target_ts - pd.Timedelta(days=lookback_days)
    recent = train[(train["Date"] >= recent_start) & (train["Date"] < target_ts)]

    counts = (
        recent["PB"]
        .value_counts()
        .reindex(range(PB_MIN, PB_MAX + 1), fill_value=0)
        .sort_index()
    )
    maxfreq = int(counts.max()) if not counts.empty else 0
    ranks = _rank_from_counts(counts)

    print("\n=== LAST 12 MONTHS PB TABLE ===")
    print(f"Target date: {target_ts.date()} | Window: [{recent_start.date()} .. {target_ts.date()})")
    print(f"Max frequency in window: {maxfreq}\n")
    print("PB | Freq | Ratio_to_Max | Rank")
    print("---+------+-------------+-----")
    order = sorted(range(PB_MIN, PB_MAX + 1), key=lambda x: (-int(counts.loc[x]), x))
    for pb in order:
        freq = int(counts.loc[pb])
        ratio = (freq / maxfreq) if maxfreq > 0 else 0.0
        rnk = int(ranks.get(pb, 999))
        print(f"{pb:>2} | {freq:>4} | {ratio:>11.3f} | {rnk:>3}")

    print("\n=== RANK LIST (ties grouped) ===")
    # Group by rank
    rank_groups: Dict[int, List[int]] = {}
    for pb, rnk in ranks.items():
        rank_groups.setdefault(rnk, []).append(pb)

    for rnk in sorted(rank_groups.keys()):
        pbs = sorted(rank_groups[rnk], key=lambda x: (-counts.loc[x], x))
        items = ", ".join([f"{pb}({int(counts.loc[pb])})" for pb in pbs])
        print(f"Rank {rnk}: {items}")

    return counts, ranks, maxfreq


def explain_exclusions(
    df: pd.DataFrame,
    target_ts: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    recent_draws_penalty_n: int = DEFAULT_RECENT_DRAWS_PENALTY_N,
    same_month_cluster_threshold: int = DEFAULT_SAME_MONTH_CLUSTER_THRESHOLD,
    exclude_if_same_month_clustered: bool = True,
) -> None:
    """
    Prints which PBs are excluded/flagged and why:
      - leader (#1 rank)
      - cold (<=1 hits in 12 months)
      - recent repeats (in last N draws)
      - same-month clustering (>= threshold in target month within lookback window)
    """
    train = df[df["Date"] < target_ts].copy()
    recent_start = target_ts - pd.Timedelta(days=lookback_days)
    recent = train[(train["Date"] >= recent_start) & (train["Date"] < target_ts)]
    counts = (
        recent["PB"]
        .value_counts()
        .reindex(range(PB_MIN, PB_MAX + 1), fill_value=0)
    )
    maxfreq = int(counts.max()) if not counts.empty else 0
    ranks = _rank_from_counts(counts)

    # last N draws
    last_n = train.sort_values("Date").tail(recent_draws_penalty_n)["PB"].tolist()
    last_n_counts = pd.Series(last_n).value_counts()

    # same-month clustering in target month (within lookback window)
    target_month = target_ts.month
    recent_in_target_month = train[
        (train["Date"] >= recent_start) &
        (train["Date"] < target_ts) &
        (train["Date"].dt.month == target_month)
    ]
    month_counts = recent_in_target_month["PB"].value_counts()

    print("\n=== EXCLUSIONS / FLAGS ===")
    print(f"Target date: {target_ts.date()} | Lookback: {lookback_days}d | Target month: {target_month}")
    print(f"Same-month cluster threshold: {same_month_cluster_threshold} | Hard-exclude: {exclude_if_same_month_clustered}\n")

    any_flag = False
    for pb in range(PB_MIN, PB_MAX + 1):
        freq = int(counts.loc[pb])
        rnk = int(ranks.get(pb, 999))
        reasons: List[str] = []

        if maxfreq > 0 and rnk == 1:
            reasons.append("LEADER (#1 rank) → usually avoided by our seasonal rule")

        if freq <= 1:
            reasons.append("COLD in last 12mo (freq<=1)")

        recent_hits = int(last_n_counts.get(pb, 0))
        if recent_hits > 0:
            reasons.append(f"Recent repeat: {recent_hits} hit(s) in last {recent_draws_penalty_n} draws")

        same_month_hits = int(month_counts.get(pb, 0))
        if same_month_hits >= same_month_cluster_threshold:
            if exclude_if_same_month_clustered:
                reasons.append(f"EXCLUDED: clustered in target month (hits={same_month_hits})")
            else:
                reasons.append(f"Clustered in target month (hits={same_month_hits})")

        if reasons:
            any_flag = True
            print(f"PB {pb:>2}: " + " | ".join(reasons))

    if not any_flag:
        print("(No exclusions/flags triggered.)")


# def seasonal_window_hits_last_n_years(
#     df: pd.DataFrame,
#     target_ts: pd.Timestamp,
#     window_days: int = DEFAULT_SEASON_WINDOW_DAYS,
#     n_years: int = 4,
# ) -> None:
#     """
#     Prints seasonal window hits by year for the last n_years:
#     e.g. for target=Jan 1, shows draws within +/- window_days around Jan 1
#     for each of the last n_years (target_year-1 downwards).
#     """
#     target_d = target_ts.date()
#     last_year = target_d.year - 1
#     first_year = last_year - (n_years - 1)
#
#     print(f"\n=== SEASONAL WINDOW HITS (last {n_years} years) ===")
#     print(f"Target month/day: {target_d.month:02}/{target_d.day:02} | Window: +/-{window_days} days\n")
#
#     for y in range(first_year, last_year + 1):
#         anchor = _anchor_for_year(target_d, y)
#         win_start, win_end = _season_window_dates(anchor, window_days)
#
#         near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)].sort_values("Date")
#         if near.empty:
#             print(f"{y}: (no draws found in window)")
#             continue
#
#         hits = []
#         for _, row in near.iterrows():
#             hits.append(f"{row['Date'].date()} → PB {int(row['PB'])}")
#         print(f"{y}: " + " | ".join(hits))


# -------------------------------------------------------------------
# Main predictor (your code) + optional debug prints
# -------------------------------------------------------------------

def predict_powerball(
    target_date: str,
    csv_path: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    season_window_days: int = DEFAULT_SEASON_WINDOW_DAYS,
    season_lookback_years: int = DEFAULT_SEASON_LOOKBACK_YEARS,
    recent_draws_penalty_n: int = DEFAULT_RECENT_DRAWS_PENALTY_N,
    same_month_cluster_threshold: int = DEFAULT_SAME_MONTH_CLUSTER_THRESHOLD,
    exclude_if_same_month_clustered: bool = True,
    top_k: int = 5,
    debug: bool = False,
    seasonal_hits_last_years: int = 4,
) -> List[CandidateScore]:
    """
    Predict Powerball for a target date using:
      - last 12 months PB frequency/rank/ratio
      - historical seasonality around the same month/day (+/- season_window_days) across past years
      - penalties for over-exposure (rank #1), cold PBs, recent draw repeats, and same-month clustering

    target_date: 'YYYY-MM-DD' (e.g., '2025-12-25' or '2026-01-01')
    """
    df = _load_powerball_csv(csv_path)
    t = pd.Timestamp(target_date)
    if pd.isna(t):
        raise ValueError("target_date must be parseable, e.g. '2025-12-25'")

    # Use only draws strictly before target date for training
    train = df[df["Date"] < t].copy()
    if train.empty:
        raise ValueError("No historical draws before target_date in this CSV.")
    # ----------------------------
    # Season-aware learning (date-specific)
    # ----------------------------
    season_profile = _learn_season_profile(
        df=df,
        target_ts=t,
        lookback_days=lookback_days,
        season_window_days=season_window_days,
        season_lookback_years=season_lookback_years,
    )
    allow_leader = season_profile.leader_rate >= 0.20  # if leaders frequently win in this season, don't hard-penalize

    if debug:
        print("\n=== SEASON PROFILE (learned from history near the same month/day) ===")
        print(f"Anchor: {season_profile.anchor_month:02}/{season_profile.anchor_day:02} | "
              f"Samples: years={season_profile.sample_years}, draws={season_profile.sample_draws}")
        print(f"Learned ratio band (p25..p75): {season_profile.ratio_low:.3f} .. {season_profile.ratio_high:.3f} "
              f"| ideal(median)={season_profile.ratio_ideal:.3f}")
        print(f"Learned rank band  (p25..p75): {season_profile.rank_low} .. {season_profile.rank_high} "
              f"| leader_rate={season_profile.leader_rate:.3f} | allow_leader={allow_leader}")

    # ----------------------------
    # A) Current "last 12 months" regime (relative to target)
    # ----------------------------
    recent_start = t - pd.Timedelta(days=lookback_days)
    recent_counts = _counts_in_window(train, recent_start, t).reindex(range(PB_MIN, PB_MAX + 1), fill_value=0)
    maxfreq = int(recent_counts.max()) if not recent_counts.empty else 0
    ranks = _rank_from_counts(recent_counts)

    # last N draws for recency penalty
    last_n = train.sort_values("Date").tail(recent_draws_penalty_n)["PB"].tolist()
    last_n_counts = pd.Series(last_n).value_counts()

    # same-month clustering in the target month, within the recent window
    target_month = t.month
    recent_in_target_month = train[
        (train["Date"] >= recent_start) &
        (train["Date"] < t) &
        (train["Date"].dt.month == target_month)
    ]
    month_counts = recent_in_target_month["PB"].value_counts()

    # Candidate pool: PBs 1..20
    candidates = list(range(PB_MIN, PB_MAX + 1))

    # ----------------------------
    # B) Seasonal history around month/day (e.g. Christmas, Jan 1, etc.)
    # For each past year: take draws within +/- season_window_days of anchor.
    # Also evaluate whether those PBs were "runner-up band" in THEIR OWN prior 12 months.
    # ----------------------------
    target_dt = t.date()
    start_year = target_dt.year - season_lookback_years
    end_year = target_dt.year - 1

    seasonal_pb_counts = {pb: 0 for pb in candidates}
    seasonal_success_counts = {pb: 0 for pb in candidates}

    for y in range(start_year, end_year + 1):
        anchor = _anchor_for_year(target_dt, y)
        win_start, win_end = _season_window_dates(anchor, season_window_days)

        near = df[(df["Date"] >= win_start) & (df["Date"] < win_end)]
        if near.empty:
            continue

        for _, row in near.iterrows():
            pb = int(row["PB"])
            if pb < PB_MIN or pb > PB_MAX:
                continue
            seasonal_pb_counts[pb] += 1

            # "success" = at that historical draw date, pb was in rank 2–5 AND ratio band in its prior 12 months
            d = row["Date"]
            hist_train = df[df["Date"] < d]
            hist_start = d - pd.Timedelta(days=lookback_days)
            hist_counts = _counts_in_window(hist_train, hist_start, d)
            if hist_counts.empty:
                continue

            # Ensure full PB space for ranks consistency
            hist_counts = hist_counts.reindex(range(PB_MIN, PB_MAX + 1), fill_value=0)

            hist_max = int(hist_counts.max())
            hist_ranks = _rank_from_counts(hist_counts)
            hist_freq = int(hist_counts.get(pb, 0))
            hist_rank = int(hist_ranks.get(pb, 999))
            hist_ratio = (hist_freq / hist_max) if hist_max > 0 else 0.0

            if (
                (season_profile.rank_low <= hist_rank <= season_profile.rank_high)
                and (season_profile.ratio_low <= hist_ratio <= season_profile.ratio_high)
                and (hist_freq >= 2)
                and (allow_leader or hist_rank != 1)
            ):
                seasonal_success_counts[pb] += 1

    # ----------------------------
    # Debug prints (requested)
    # ----------------------------
    if debug:
        # 12mo table + ranks
        print_12mo_pb_table_and_ranks(df, t, lookback_days=lookback_days)
        # exclusions and why
        explain_exclusions(
            df,
            t,
            lookback_days=lookback_days,
            recent_draws_penalty_n=recent_draws_penalty_n,
            same_month_cluster_threshold=same_month_cluster_threshold,
            exclude_if_same_month_clustered=exclude_if_same_month_clustered,
        )
        # seasonal window hits for last N years
        # seasonal_window_hits_last_n_years(
        #     df,
        #     t,
        #     window_days=season_window_days,
        #     n_years=seasonal_hits_last_years,
        # )

    # ----------------------------
    # C) Scoring (implements our analysis)
    # ----------------------------
    scored: List[CandidateScore] = []

    for pb in candidates:
        freq = int(recent_counts.get(pb, 0))
        rank = int(ranks.get(pb, 999))
        ratio = (freq / maxfreq) if maxfreq > 0 else 0.0

        components: Dict[str, float] = {}
        penalties: Dict[str, float] = {}
        # 1) Rank preference (season-aware): prefer the learned "typical winner" rank band for this calendar date
        if (rank == 1) and (not allow_leader):
            components["rank_bonus"] = 0.0
            penalties["leader_penalty"] = 2.0
        elif season_profile.rank_low <= rank <= season_profile.rank_high:
            # Highest bonus at the low end of the band, slightly declining within the band
            width = max(1, (season_profile.rank_high - season_profile.rank_low))
            components["rank_bonus"] = 2.0 - (0.75 * (rank - season_profile.rank_low) / width)
        elif (season_profile.rank_high + 1) <= rank <= (season_profile.rank_high + 5):
            components["rank_bonus"] = 0.6
        else:
            components["rank_bonus"] = 0.0

        # 2) Ratio closeness to season-aware ideal (median of historical winners near this date)
        if ratio == 0.0:
            components["ratio_bonus"] = 0.0
        else:
            components["ratio_bonus"] = max(0.0, 1.5 - abs(ratio - season_profile.ratio_ideal) * 4.0)


        # 3) Cold penalty (avoid one-offs)
        if freq <= 1:
            penalties["cold_penalty"] = 1.5
        elif freq == 2:
            penalties["low_freq_penalty"] = 0.5

        # 4) Recent draw repeat penalty (over-exposure)
        recent_hits = int(last_n_counts.get(pb, 0))
        if recent_hits > 0:
            penalties["recent_repeat_penalty"] = 0.3 * recent_hits

        # 5) Same-month clustering penalty / exclusion
        same_month_hits = int(month_counts.get(pb, 0))
        if same_month_hits >= same_month_cluster_threshold:
            if exclude_if_same_month_clustered:
                penalties["month_cluster_exclusion"] = 10.0  # effectively removes it
            else:
                penalties["month_cluster_penalty"] = 1.2 + 0.3 * (same_month_hits - same_month_cluster_threshold)

        # 6) Seasonal prior (more occurrences near that calendar date)
        season_ct = seasonal_pb_counts.get(pb, 0)
        season_succ = seasonal_success_counts.get(pb, 0)

        components["seasonal_count_bonus"] = min(1.2, 0.25 * season_ct)
        components["seasonal_success_bonus"] = min(1.8, 0.45 * season_succ)

        total = sum(components.values()) - sum(penalties.values())

        scored.append(
            CandidateScore(
                pb=pb,
                total_score=round(total, 4),
                recent_12mo_freq=freq,
                recent_12mo_rank=rank,
                recent_12mo_ratio=round(ratio, 3),
                seasonal_count=season_ct,
                seasonal_success_count=season_succ,
                penalties=penalties,
                components=components,
            )
        )

    scored.sort(key=lambda x: x.total_score, reverse=True)
    return scored[:top_k]


def run_predictions_for_first_n_csv_rows(
    csv_path: str,
    n: int,
    top_k: int = 5,
    debug: bool = False,
):
    """
    Uses the FIRST N rows exactly as they appear in the CSV file (no sorting),
    """
    raw = pd.read_csv(csv_path)

    if "Draw date" not in raw.columns or "Powerball Number" not in raw.columns:
        raise ValueError("CSV must contain columns: 'Draw date' and 'Powerball Number'")

    raw = raw.copy()
    raw["Date"] = raw["Draw date"].apply(_parse_date)
    raw = raw.dropna(subset=["Date"])
    raw["PB"] = raw["Powerball Number"].astype(int)

    # FIRST N rows as-is (file order)
    target_rows = raw.head(n).reset_index(drop=True)

    print(f"CSV rows total: {len(raw)}")
    print(f"Running FIRST {len(target_rows)} rows AS-IS from CSV (no date sorting)\n")
    print(f"First target row date: {target_rows.iloc[0]['Date'].date()} | "
          f"Last target row date: {target_rows.iloc[-1]['Date'].date()}")

    for i, row in target_rows.iterrows():
        target_date = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
        real_pb = int(row["PB"])

        print("\n" + "=" * 70)
        print(f"[CSV ROW #{i+1}] TARGET DATE: {target_date} | REAL PB: {real_pb}")
        print("=" * 70)

        preds = predict_powerball(
            target_date,
            csv_path,
            top_k=top_k,
            debug=True,
            seasonal_hits_last_years=4
        )

        print("Top predictions:")
        for r in preds:
            hit = "🎯 HIT" if r.pb == real_pb else ""
            print(f"  PB {r.pb:>2} | score={r.total_score:>6} {hit}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    csv_path = "Powerball.csv"  # change to your path

    # Christmas target (debug prints ON)
    # res = predict_powerball(
    #     "2025-12-25",
    #     csv_path,
    #     top_k=10,
    #     debug=True,                 # ✅ prints 12mo table, ranks, exclusions, seasonal hits last 4 yrs
    #     seasonal_hits_last_years=4  # ✅ last 4 years only
    # )
    # print("\nTop candidates for 2025-12-25:")
    # for r in res:
    #     print(r)

    # New year target (debug prints ON)
    # res2 = predict_powerball(
    #     "2026-01-01",
    #     csv_path,
    #     top_k=10,
    #     debug=True,
    #     seasonal_hits_last_years=4
    # )
    # print("\nTop candidates for 2026-01-01:")
    # for r in res2:
    #     print(r)
    run_predictions_for_first_n_csv_rows(
        csv_path=csv_path,
        n=10,
        top_k=20,
        debug=False
    )
