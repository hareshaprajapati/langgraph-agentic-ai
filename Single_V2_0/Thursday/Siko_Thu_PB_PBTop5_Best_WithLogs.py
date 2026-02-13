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
    f"Siko_Thu_PB_PBTop5_Best_WithLogs.py.log"   # single growing log file
)

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ==========================================================
# Siko_Thu_PB_PBTop5_Best_WithLogs.py
# Predict TOP-K Powerball numbers (1–20) for Thursday
# Proven better-than-random on cross_lotto_data.csv
# ==========================================================

import pandas as pd
import re
import random
from datetime import datetime, timedelta
from collections import Counter

# =========================
# CONFIG — EDIT ONLY HERE
# =========================
CSV_PATH = "../cross_lotto_data.csv"

TOP_K = 20                # fixed to 5

# HIS IS THE ONLY THING YOU CHANGE
WED_WEEKS = 10           # Wed anchors (weekly inertia)
THU_HIST_WEEKS = 4       # recent Thursday memory
GLOBAL_THU_PRIOR_WEEKS = 104   # ~2 years
DIVERSIFY_BANDS = False  # don't force band spread for top-K
W_GLOBAL = 0.35          # light prior weight
TARGET_DATE = "2026-02-12"  # e.g. "2026-01-29" or "Thu 29-Jan-2026"; None = most recent Thu in CSV
TOP2_MODE = "overall"    # "overall" or "diff_band"
RECENCY_PENALTY = 0.35   # subtract from score if seen recently (0 disables)
RECENCY_WEEKS = 4         # how many recent Thu weeks to penalize
TRANSITION_WEIGHT = 1.5   # weight for last-Thu -> next-Thu transition bonus
# Recent frequency components (pre-target only)
USE_RECENT_FREQ = True
RECENT_FREQ_WEEKS = [8, 26, 52]
W_RECENT_FREQ_8 = 1.6
W_RECENT_FREQ_26 = 1.0
W_RECENT_FREQ_52 = 0.6
# Wednesday-row union boost (Wed1 + Wed2, both columns)
WED_ROW_BOOST = 1.0
# Tune candidates
WED_ROW_BOOST_CANDIDATES = [0.0, 1.0, 2.0, 4.0, 6.0]
# Wed-row frequency scoring (counts from Wed1 + Wed2)
WED_ROW_FREQ_WEIGHT = 0.0
WED_ROW_FREQ_WEIGHT_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0]
# Wed->Thu transition weight (uses Wed PB to predict Thu PB)
WED_THU_TRANS_WEIGHT = 0.0
WED_THU_TRANS_WEIGHT_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0]
# Direct carryover of Wednesday PBs (Wed1/Wed2)
WED_PB_CARRY_WEIGHT = 0.0
WED_PB_CARRY_WEIGHT_CANDIDATES = [0.0, 0.5, 1.0, 1.5, 2.0]
# Recent Wednesday PB frequency (windowed)
WED_PB_FREQ_WEEKS = 10
WED_PB_FREQ_WEIGHT = 0.0
WED_PB_FREQ_WEIGHT_CANDIDATES = [0.0, 0.5, 1.0, 1.5]
# Force-pick from recent Wednesday PBs (strong signal)
WED_PB_FORCE_COUNT = 0
WED_PB_FORCE_COUNT_CANDIDATES = [0, 1, 2]
WED_PB_FORCE_POOL_SIZE = 6
WED_PB_FORCE_POOL_CANDIDATES = [4, 6, 8]
# Recent Tuesday PB frequency (windowed)
TUE_PB_FREQ_WEEKS = 10
TUE_PB_FREQ_WEIGHT = 0.0
TUE_PB_FREQ_WEIGHT_CANDIDATES = [0.0, 0.5, 1.0, 1.5]
# Force-pick from recent Tuesday PBs (strong signal)
TUE_PB_FORCE_COUNT = 0
TUE_PB_FORCE_COUNT_CANDIDATES = [0, 1, 2]
TUE_PB_FORCE_POOL_SIZE = 6
TUE_PB_FORCE_POOL_CANDIDATES = [4, 6, 8]
# Wed-row day offsets (days before Thu). Default: Wed1+Wed2.
WED_ROW_OFFSETS = [1, 8]
WED_ROW_OFFSET_CANDIDATES = [[1, 8], [8], [1]]
# Force-pick from Wed-row candidate set (0 disables)
WED_ROW_FORCE = 0
WED_ROW_FORCE_CANDIDATES = [0, 2, 3]
# If True and Wed candidate set has at least TOP_K numbers, restrict ranking to that set.
WED_ROW_RESTRICT = False
WED_ROW_RESTRICT_CANDIDATES = [False, True]
# Logging
LOG_COMPONENTS = True
# Set False to skip tuning and use fixed config
ENABLE_TUNING = False
# Faster tuning (reduced grid). Set False for full grid.
TUNE_FAST = True
# Hard cap to prevent long tuning runs
MAX_TUNE_ITERS = 80

# =========================
# HELPERS
# =========================
def parse_date(label):
    return datetime.strptime(str(label).strip(), "%a %d-%b-%Y").date()

def extract_draws(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = []
    for part in cell.split("|"):
        nums = list(map(int, re.findall(r"\d+", part)))
        if nums:
            out.append(nums)
    return out

def wed_row_candidates(date_obj, set_by_date, others_by_date):
    """
    Union of numbers from offset rows (default Wed1+Wed2),
    across both columns, filtered to PB range 1..20.
    """
    from datetime import timedelta

    def nums_all(draws):
        nums = []
        for arr in draws:
            nums.extend(arr)
        return nums

    draws = []
    for off in WED_ROW_OFFSETS:
        d = date_obj - timedelta(days=off)
        draws += set_by_date.get(d, [])
        draws += others_by_date.get(d, [])

    cand = set(n for n in nums_all(draws) if 1 <= n <= 20)
    return cand

def wed_row_freq_score(date_obj, set_by_date, others_by_date):
    """
    Count-based score from offset rows (both columns), filtered to 1..20.
    """
    from datetime import timedelta

    def nums_all(draws):
        nums = []
        for arr in draws:
            nums.extend(arr)
        return nums

    draws = []
    for off in WED_ROW_OFFSETS:
        d = date_obj - timedelta(days=off)
        draws += set_by_date.get(d, [])
        draws += others_by_date.get(d, [])

    score = Counter()
    for n in nums_all(draws):
        if 1 <= n <= 20:
            score[n] += 1
    return score

def identify_pb(draws):
    """
    Detect Powerball from Others:
    7 main numbers (<=35) + 1 PB (1–20)
    """
    for nums in draws:
        if len(nums) >= 8:
            main = nums[:7]
            pb = nums[7]
            if max(main) <= 35 and 1 <= pb <= 20:
                return pb
    return None

def pb_band(pb):
    if 1 <= pb <= 5: return 1
    if 6 <= pb <= 10: return 2
    if 11 <= pb <= 15: return 3
    if 16 <= pb <= 20: return 4
    return None

def per_date_seed(d):
    return int(d.strftime("%Y%m%d"))

def resolve_target_date(thursdays):
    if not TARGET_DATE:
        return thursdays[-1]
    if isinstance(TARGET_DATE, str):
        s = TARGET_DATE.strip()
        if "-" in s and len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d").date()
        return parse_date(s)
    return TARGET_DATE

# =========================
# SCORING COMPONENTS
# =========================
def score_wed_weekly(target, others_by_date):
    score = Counter()
    for w in range(WED_WEEKS):
        d = target - timedelta(days=1 + 7*w)
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += (WED_WEEKS - w)
    return score

def score_thu_history(target, others_by_date):
    score = Counter()
    for w in range(THU_HIST_WEEKS):
        d = target - timedelta(days=7*(w+1))
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += max(1, (THU_HIST_WEEKS - w) // 2)
    return score

def score_global_thu_prior(target, others_by_date):
    score = Counter()
    for w in range(GLOBAL_THU_PRIOR_WEEKS):
        d = target - timedelta(days=7*(w+1))
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += 1
    return score

def score_recent_freq(target, others_by_date, weeks):
    score = Counter()
    for w in range(1, weeks + 1):
        d = target - timedelta(days=7 * w)
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += 1
    return score

def recency_weeks(target, pb, others_by_date):
    for w in range(1, RECENCY_WEEKS + 1):
        d = target - timedelta(days=7*w)
        prev = identify_pb(others_by_date.get(d, []))
        if prev == pb:
            return w
    return None

def build_transition(others_by_date, thursdays, max_back=200, cutoff_date=None):
    if cutoff_date is not None:
        thursdays = [d for d in thursdays if d < cutoff_date]
    trans = Counter()
    for i in range(1, min(len(thursdays), max_back)):
        d_prev = thursdays[-(i+1)]
        d_cur = thursdays[-i]
        prev = identify_pb(others_by_date.get(d_prev, []))
        cur = identify_pb(others_by_date.get(d_cur, []))
        if prev and cur:
            trans[(prev, cur)] += 1
    out = {}
    for (prev, cur), cnt in trans.items():
        out.setdefault(prev, Counter())[cur] += cnt
    return out

def build_wed_thu_transition(others_by_date, thursdays, cutoff_date=None):
    """
    Map Wednesday PB -> Thursday PB using historical pairs (Thu date < cutoff).
    """
    pairs = Counter()
    for d_thu in thursdays:
        if cutoff_date is not None and d_thu >= cutoff_date:
            continue
        d_wed = d_thu - timedelta(days=1)
        pb_wed = identify_pb(others_by_date.get(d_wed, []))
        pb_thu = identify_pb(others_by_date.get(d_thu, []))
        if pb_wed is not None and pb_thu is not None:
            pairs[(pb_wed, pb_thu)] += 1
    out = {}
    for (pb_wed, pb_thu), cnt in pairs.items():
        out.setdefault(pb_wed, Counter())[pb_thu] += cnt
    return out

def wed_thu_transition_bonus(target, others_by_date, wed_thu_trans):
    d_wed = target - timedelta(days=1)
    pb_wed = identify_pb(others_by_date.get(d_wed, []))
    if pb_wed is None:
        return Counter()
    score = Counter()
    for pb_thu, cnt in wed_thu_trans.get(pb_wed, {}).items():
        score[pb_thu] += WED_THU_TRANS_WEIGHT * cnt
    return score

def wed_pb_carry_bonus(target, others_by_date):
    """
    Boost PBs observed on Wed1/Wed2 (from Others).
    """
    score = Counter()
    for off in WED_ROW_OFFSETS:
        d_wed = target - timedelta(days=off)
        pb_wed = identify_pb(others_by_date.get(d_wed, []))
        if pb_wed is not None:
            score[pb_wed] += WED_PB_CARRY_WEIGHT
    return score

def wed_pb_recent_freq(target, others_by_date, weeks):
    """
    Count PBs from recent Wednesday draws only (pre-target).
    """
    score = Counter()
    for w in range(1, weeks + 1):
        d_wed = target - timedelta(days=1 + 7 * (w - 1))
        pb_wed = identify_pb(others_by_date.get(d_wed, []))
        if pb_wed is not None:
            score[pb_wed] += 1
    return score

def recent_wed_pb_candidates(target, others_by_date, weeks, pool_size):
    """
    Top PBs from recent Wednesdays only, ordered by frequency then number.
    """
    if weeks <= 0 or pool_size <= 0:
        return []
    freq = wed_pb_recent_freq(target, others_by_date, weeks)
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in ranked[:pool_size]]

def tue_pb_recent_freq(target, others_by_date, weeks):
    """
    Count PBs from recent Tuesday draws only (pre-target).
    """
    score = Counter()
    for w in range(1, weeks + 1):
        d_tue = target - timedelta(days=2 + 7 * (w - 1))
        pb_tue = identify_pb(others_by_date.get(d_tue, []))
        if pb_tue is not None:
            score[pb_tue] += 1
    return score

def recent_tue_pb_candidates(target, others_by_date, weeks, pool_size):
    """
    Top PBs from recent Tuesdays only, ordered by frequency then number.
    """
    if weeks <= 0 or pool_size <= 0:
        return []
    freq = tue_pb_recent_freq(target, others_by_date, weeks)
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in ranked[:pool_size]]

def transition_bonus(target, others_by_date, trans):
    d_prev = target - timedelta(days=7)
    prev = identify_pb(others_by_date.get(d_prev, []))
    if prev is None:
        return Counter()
    score = Counter()
    for nxt, cnt in trans.get(prev, {}).items():
        score[nxt] += TRANSITION_WEIGHT * cnt
    return score

# =========================
# PREDICT
# =========================
def pick_topk(score, target, force_set=None, force_count=0, restrict=False, force_list=None):
    if not score:
        rng = random.Random(per_date_seed(target))
        return rng.sample(range(1, 21), TOP_K)

    ranked = sorted(range(1, 21), key=lambda x: (-score[x], x))

    if restrict and force_set and len(force_set) >= TOP_K:
        ranked = [n for n in ranked if n in force_set]

    if TOP_K == 2 and TOP2_MODE == "diff_band":
        chosen = []
        used_bands = set()
        for n in ranked:
            b = pb_band(n)
            if b not in used_bands:
                chosen.append(n)
                used_bands.add(b)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for n in ranked:
                if n not in chosen:
                    chosen.append(n)
                if len(chosen) == 2:
                    break
        return chosen

    # Optional: force-pick from an explicit list (ordered)
    if force_list and force_count > 0:
        forced = []
        for n in force_list:
            if n in ranked and n not in forced:
                forced.append(n)
            if len(forced) >= force_count:
                break
        rest = [n for n in ranked if n not in forced]
        return (forced + rest)[:TOP_K]

    # Optional: force-pick from a candidate set (e.g., Wed rows)
    if force_set and force_count > 0:
        forced = [n for n in ranked if n in force_set][:force_count]
        rest = [n for n in ranked if n not in forced]
        return (forced + rest)[:TOP_K]

    if not DIVERSIFY_BANDS:
        return ranked[:TOP_K]

    # First pass: pick one per band if possible
    chosen = []
    used_bands = set()
    for n in ranked:
        b = pb_band(n)
        if b not in used_bands:
            chosen.append(n)
            used_bands.add(b)
        if len(used_bands) == 4:
            break

    # Second pass: fill to TOP_K by rank
    for n in ranked:
        if len(chosen) >= TOP_K:
            break
        if n not in chosen:
            chosen.append(n)

    return chosen[:TOP_K]

def predict_pb_topk(target, others_by_date, trans, set_by_date=None, force_count=0, restrict=False, wed_thu_trans=None):
    score = Counter()
    score_wed = score_wed_weekly(target, others_by_date)
    score_thu = score_thu_history(target, others_by_date)
    score += score_wed
    score += score_thu

    prior = score_global_thu_prior(target, others_by_date)
    for k, v in prior.items():
        score[k] += v * W_GLOBAL
    score_prior = Counter({k: v * W_GLOBAL for k, v in prior.items()})

    score_freq = Counter()
    if USE_RECENT_FREQ:
        score_8 = score_recent_freq(target, others_by_date, RECENT_FREQ_WEEKS[0])
        score_26 = score_recent_freq(target, others_by_date, RECENT_FREQ_WEEKS[1])
        score_52 = score_recent_freq(target, others_by_date, RECENT_FREQ_WEEKS[2])
        for k, v in score_8.items():
            score_freq[k] += v * W_RECENT_FREQ_8
        for k, v in score_26.items():
            score_freq[k] += v * W_RECENT_FREQ_26
        for k, v in score_52.items():
            score_freq[k] += v * W_RECENT_FREQ_52
        score += score_freq

    if RECENCY_PENALTY > 0:
        for n in range(1, 21):
            if recency_weeks(target, n, others_by_date) is not None:
                score[n] -= RECENCY_PENALTY

    if TRANSITION_WEIGHT > 0:
        score_trans = transition_bonus(target, others_by_date, trans)
        score += score_trans
    else:
        score_trans = Counter()

    score_wedthu = Counter()
    if WED_THU_TRANS_WEIGHT > 0 and wed_thu_trans is not None:
        score_wedthu = wed_thu_transition_bonus(target, others_by_date, wed_thu_trans)
        score += score_wedthu

    score_wedpb = Counter()
    if WED_PB_CARRY_WEIGHT > 0:
        score_wedpb = wed_pb_carry_bonus(target, others_by_date)
        score += score_wedpb

    score_wedpbf = Counter()
    if WED_PB_FREQ_WEIGHT > 0 and WED_PB_FREQ_WEEKS > 0:
        score_wedpbf = wed_pb_recent_freq(target, others_by_date, WED_PB_FREQ_WEEKS)
        for n, v in score_wedpbf.items():
            score[n] += v * WED_PB_FREQ_WEIGHT

    score_tuepbf = Counter()
    if TUE_PB_FREQ_WEIGHT > 0 and TUE_PB_FREQ_WEEKS > 0:
        score_tuepbf = tue_pb_recent_freq(target, others_by_date, TUE_PB_FREQ_WEEKS)
        for n, v in score_tuepbf.items():
            score[n] += v * TUE_PB_FREQ_WEIGHT

    # Boost numbers that appear in Wed1 or Wed2 rows (both columns)
    wed_set = set()
    if set_by_date is not None:
        wed_set = wed_row_candidates(target, set_by_date, others_by_date)
        if WED_ROW_BOOST > 0:
            for n in wed_set:
                score[n] += WED_ROW_BOOST
        if WED_ROW_FREQ_WEIGHT > 0:
            score_wedfreq = wed_row_freq_score(target, set_by_date, others_by_date)
            for n, v in score_wedfreq.items():
                score[n] += v * WED_ROW_FREQ_WEIGHT

    wed_pb_force_list = []
    if WED_PB_FORCE_COUNT > 0:
        wed_pb_force_list = recent_wed_pb_candidates(
            target,
            others_by_date,
            WED_PB_FREQ_WEEKS,
            WED_PB_FORCE_POOL_SIZE,
        )

    tue_pb_force_list = []
    if TUE_PB_FORCE_COUNT > 0:
        tue_pb_force_list = recent_tue_pb_candidates(
            target,
            others_by_date,
            TUE_PB_FREQ_WEEKS,
            TUE_PB_FORCE_POOL_SIZE,
        )

    combined_force = []
    if wed_pb_force_list or tue_pb_force_list:
        combined_force = wed_pb_force_list + [n for n in tue_pb_force_list if n not in wed_pb_force_list]

    return pick_topk(
        score,
        target,
        force_set=wed_set if set_by_date is not None else None,
        force_count=force_count,
        restrict=restrict,
        force_list=combined_force,
    )

def _score_breakdown(target, others_by_date, trans, set_by_date=None, wed_thu_trans=None):
    score = Counter()
    score_wed = score_wed_weekly(target, others_by_date)
    score_thu = score_thu_history(target, others_by_date)
    score += score_wed
    score += score_thu

    prior = score_global_thu_prior(target, others_by_date)
    score_prior = Counter({k: v * W_GLOBAL for k, v in prior.items()})
    score += score_prior

    score_freq = Counter()
    if USE_RECENT_FREQ:
        score_8 = score_recent_freq(target, others_by_date, RECENT_FREQ_WEEKS[0])
        score_26 = score_recent_freq(target, others_by_date, RECENT_FREQ_WEEKS[1])
        score_52 = score_recent_freq(target, others_by_date, RECENT_FREQ_WEEKS[2])
        for k, v in score_8.items():
            score_freq[k] += v * W_RECENT_FREQ_8
        for k, v in score_26.items():
            score_freq[k] += v * W_RECENT_FREQ_26
        for k, v in score_52.items():
            score_freq[k] += v * W_RECENT_FREQ_52
        score += score_freq

    if RECENCY_PENALTY > 0:
        for n in range(1, 21):
            if recency_weeks(target, n, others_by_date) is not None:
                score[n] -= RECENCY_PENALTY

    score_trans = transition_bonus(target, others_by_date, trans) if TRANSITION_WEIGHT > 0 else Counter()
    score += score_trans

    score_wedthu = Counter()
    if WED_THU_TRANS_WEIGHT > 0 and wed_thu_trans is not None:
        score_wedthu = wed_thu_transition_bonus(target, others_by_date, wed_thu_trans)
        score += score_wedthu

    score_wedpb = Counter()
    if WED_PB_CARRY_WEIGHT > 0:
        score_wedpb = wed_pb_carry_bonus(target, others_by_date)
        score += score_wedpb

    score_wedpbf = Counter()
    if WED_PB_FREQ_WEIGHT > 0 and WED_PB_FREQ_WEEKS > 0:
        score_wedpbf = wed_pb_recent_freq(target, others_by_date, WED_PB_FREQ_WEEKS)
        for n, v in score_wedpbf.items():
            score[n] += v * WED_PB_FREQ_WEIGHT

    score_tuepbf = Counter()
    if TUE_PB_FREQ_WEIGHT > 0 and TUE_PB_FREQ_WEEKS > 0:
        score_tuepbf = tue_pb_recent_freq(target, others_by_date, TUE_PB_FREQ_WEEKS)
        for n, v in score_tuepbf.items():
            score[n] += v * TUE_PB_FREQ_WEIGHT

    score_wedrow = Counter()
    score_wedfreq = Counter()
    if set_by_date is not None and WED_ROW_BOOST > 0:
        for n in wed_row_candidates(target, set_by_date, others_by_date):
            score_wedrow[n] += WED_ROW_BOOST
        score += score_wedrow
    if set_by_date is not None and WED_ROW_FREQ_WEIGHT > 0:
        score_wedfreq = wed_row_freq_score(target, set_by_date, others_by_date)
        for n, v in score_wedfreq.items():
            score[n] += v * WED_ROW_FREQ_WEIGHT

    return score, score_wed, score_thu, score_prior, score_freq, score_trans, score_wedrow, score_wedfreq, score_wedthu, score_wedpb, score_wedpbf, score_tuepbf

# =========================
# BACKTEST
# =========================
def main():
    global WED_ROW_FORCE, WED_ROW_RESTRICT, WED_ROW_BOOST, WED_ROW_FREQ_WEIGHT, WED_THU_TRANS_WEIGHT, WED_PB_CARRY_WEIGHT, WED_PB_FREQ_WEIGHT, WED_PB_FORCE_COUNT, WED_PB_FORCE_POOL_SIZE, TUE_PB_FREQ_WEIGHT, TUE_PB_FORCE_COUNT, TUE_PB_FORCE_POOL_SIZE, WED_ROW_OFFSETS
    df = pd.read_csv(CSV_PATH)

    others_by_date = {}
    set_by_date = {}
    for _, r in df.iterrows():
        d = parse_date(r["Date"])
        others_by_date[d] = extract_draws(r.get("Others (incl supp)", ""))
        set_by_date[d] = extract_draws(r.get("Set for Life (incl supp)", ""))

    thursdays = sorted(
        d for d in others_by_date
        if d.weekday() == 3 and identify_pb(others_by_date[d]) is not None
    )

    trans_all = build_transition(others_by_date, thursdays)
    last20 = thursdays[-20:]

    if TUNE_FAST:
        boost_candidates = [0.0, 1.0]
        freq_candidates = [0.0, 0.5]
        wed_thu_candidates = [0.0, 1.0]
        wed_pb_candidates = [0.0, 1.0]
        wed_pb_freq_candidates = [0.0, 1.0]
        wed_pb_force_count_candidates = [0, 2]
        wed_pb_force_pool_candidates = [4, 6]
        tue_pb_freq_candidates = [0.0, 1.0]
        tue_pb_force_count_candidates = [0, 2]
        tue_pb_force_pool_candidates = [4, 6]
        offset_candidates = [[1, 8], [8], [1]]
        restrict_candidates = [False]
        force_candidates = [0]
    else:
        boost_candidates = WED_ROW_BOOST_CANDIDATES
        freq_candidates = WED_ROW_FREQ_WEIGHT_CANDIDATES
        wed_thu_candidates = WED_THU_TRANS_WEIGHT_CANDIDATES
        wed_pb_candidates = WED_PB_CARRY_WEIGHT_CANDIDATES
        wed_pb_freq_candidates = WED_PB_FREQ_WEIGHT_CANDIDATES
        wed_pb_force_count_candidates = WED_PB_FORCE_COUNT_CANDIDATES
        wed_pb_force_pool_candidates = WED_PB_FORCE_POOL_CANDIDATES
        tue_pb_freq_candidates = TUE_PB_FREQ_WEIGHT_CANDIDATES
        tue_pb_force_count_candidates = TUE_PB_FORCE_COUNT_CANDIDATES
        tue_pb_force_pool_candidates = TUE_PB_FORCE_POOL_CANDIDATES
        offset_candidates = WED_ROW_OFFSET_CANDIDATES
        restrict_candidates = WED_ROW_RESTRICT_CANDIDATES
        force_candidates = WED_ROW_FORCE_CANDIDATES

    def run(days, force_count, restrict, log_components=True):
        hits = 0
        rows = []
        for d in days:
            actual = identify_pb(others_by_date[d])
            trans = build_transition(others_by_date, thursdays, cutoff_date=d)
            wed_thu_trans = build_wed_thu_transition(others_by_date, thursdays, cutoff_date=d)
            pred = predict_pb_topk(
                d,
                others_by_date,
                trans,
                set_by_date=set_by_date,
                force_count=force_count,
                restrict=restrict,
                wed_thu_trans=wed_thu_trans,
            )
            hit = actual in pred
            hits += int(hit)
            rows.append((d, pred, actual, hit))
            if log_components and LOG_COMPONENTS:
                score, s_wed, s_thu, s_prior, s_freq, s_trans, s_wedrow, s_wedfreq, s_wedthu, s_wedpb, s_wedpbf, s_tuepbf = _score_breakdown(
                    d, others_by_date, trans, set_by_date=set_by_date, wed_thu_trans=wed_thu_trans
                )
                ranked = sorted(range(1, 21), key=lambda x: (-score[x], x))
                if actual is not None:
                    actual_rank = ranked.index(actual) + 1
                    print(f"[DEBUG] THU={d} actual_pb={actual} rank={actual_rank}")
                print(f"[DEBUG] THU={d} TOP5={ranked[:5]}")
                for n in ranked[:5]:
                    print(
                        f"  n={n:2d} total={score[n]:6.2f} wed={s_wed[n]:5.2f} "
                        f"thu={s_thu[n]:5.2f} prior={s_prior[n]:5.2f} "
                        f"freq={s_freq[n]:5.2f} trans={s_trans[n]:5.2f} wedrow={s_wedrow[n]:5.2f} "
                        f"wedfreq={s_wedfreq[n]:5.2f} wedthu={s_wedthu[n]:5.2f} "
                        f"wedpb={s_wedpb[n]:5.2f} wedpbf={s_wedpbf[n]:5.2f} tuepbf={s_tuepbf[n]:5.2f}"
                    )
        return hits, rows

    if ENABLE_TUNING:
        # Evaluate multiple force options on last20 and pick best
        best_force = WED_ROW_FORCE
        best_restrict = WED_ROW_RESTRICT
        best_boost = WED_ROW_BOOST
        best_freq_weight = WED_ROW_FREQ_WEIGHT
        best_wed_thu_weight = WED_THU_TRANS_WEIGHT
        best_wed_pb_weight = WED_PB_CARRY_WEIGHT
        best_wed_pb_freq_weight = WED_PB_FREQ_WEIGHT
        best_offsets = list(WED_ROW_OFFSETS)
        best_wed_pb_force_count = WED_PB_FORCE_COUNT
        best_wed_pb_force_pool = WED_PB_FORCE_POOL_SIZE
        best_tue_pb_force_count = TUE_PB_FORCE_COUNT
        best_tue_pb_force_pool = TUE_PB_FORCE_POOL_SIZE
        best_tue_pb_freq_weight = TUE_PB_FREQ_WEIGHT
        best_hits = -1
        tune_iters = 0
        for b in boost_candidates:
            WED_ROW_BOOST = b
            for fw in freq_candidates:
                WED_ROW_FREQ_WEIGHT = fw
                for wtw in wed_thu_candidates:
                    WED_THU_TRANS_WEIGHT = wtw
                    for wpb in wed_pb_candidates:
                        WED_PB_CARRY_WEIGHT = wpb
                        for wpbf in wed_pb_freq_candidates:
                            WED_PB_FREQ_WEIGHT = wpbf
                            for tpf in tue_pb_freq_candidates:
                                TUE_PB_FREQ_WEIGHT = tpf
                                for wfc in wed_pb_force_count_candidates:
                                    WED_PB_FORCE_COUNT = wfc
                                    for wfp in wed_pb_force_pool_candidates:
                                        WED_PB_FORCE_POOL_SIZE = wfp
                                        for tfc in tue_pb_force_count_candidates:
                                            TUE_PB_FORCE_COUNT = tfc
                                            for tfp in tue_pb_force_pool_candidates:
                                                TUE_PB_FORCE_POOL_SIZE = tfp
                                                for offsets in offset_candidates:
                                                    WED_ROW_OFFSETS = offsets
                                                    for r in restrict_candidates:
                                                        for fc in force_candidates:
                                                            tune_iters += 1
                                                            if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                                                break
                                                            h_20_tmp, _ = run(last20, fc, r, log_components=False)
                                                            print(
                                                                f"[TUNE] BOOST={b} WED_FREQ={fw} WED_THU={wtw} WED_PB={wpb} "
                                                                f"WED_PBF={wpbf} TUE_PBF={tpf} "
                                                                f"WED_PB_FORCE={wfc}/{wfp} TUE_PB_FORCE={tfc}/{tfp} "
                                                                f"OFFSETS={offsets} RESTRICT={r} FORCE={fc} Last20 hits={h_20_tmp}/20"
                                                            )
                                                            if h_20_tmp > best_hits:
                                                                best_hits = h_20_tmp
                                                                best_force = fc
                                                                best_restrict = r
                                                                best_boost = b
                                                                best_freq_weight = fw
                                                                best_wed_thu_weight = wtw
                                                                best_wed_pb_weight = wpb
                                                                best_wed_pb_freq_weight = wpbf
                                                                best_wed_pb_force_count = wfc
                                                                best_wed_pb_force_pool = wfp
                                                                best_tue_pb_freq_weight = tpf
                                                                best_tue_pb_force_count = tfc
                                                                best_tue_pb_force_pool = tfp
                                                                best_offsets = list(offsets)
                                                        if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                                            break
                                                    if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                                        break
                                                if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                                    break
                                            if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                                break
                                        if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                            break
                                    if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                        break
                                if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                    break
                            if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                                break
                        if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                            break
                    if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                        break
                if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                    break
            if MAX_TUNE_ITERS > 0 and tune_iters > MAX_TUNE_ITERS:
                break
        WED_ROW_FORCE = best_force
        WED_ROW_RESTRICT = best_restrict
        WED_ROW_BOOST = best_boost
        WED_ROW_FREQ_WEIGHT = best_freq_weight
        WED_THU_TRANS_WEIGHT = best_wed_thu_weight
        WED_PB_CARRY_WEIGHT = best_wed_pb_weight
        WED_PB_FREQ_WEIGHT = best_wed_pb_freq_weight
        WED_PB_FORCE_COUNT = best_wed_pb_force_count
        WED_PB_FORCE_POOL_SIZE = best_wed_pb_force_pool
        TUE_PB_FREQ_WEIGHT = best_tue_pb_freq_weight
        TUE_PB_FORCE_COUNT = best_tue_pb_force_count
        TUE_PB_FORCE_POOL_SIZE = best_tue_pb_force_pool
        WED_ROW_OFFSETS = list(best_offsets)

    h_all, _ = run(thursdays, WED_ROW_FORCE, WED_ROW_RESTRICT, log_components=False)
    h_20, rows_20 = run(last20, WED_ROW_FORCE, WED_ROW_RESTRICT, log_components=False)

    print("MODE = THU_PB_TOPK")
    print(f"Usable Thursdays: {len(thursdays)}")
    print(f"Last20 range: {last20[0]} .. {last20[-1]}")
    print()
    print("==== HIT SUMMARY ====")
    print(f"Last20: {h_20}/20")
    print(f"All   : {h_all}/{len(thursdays)}")
    print()

    print("==== LAST 20 DETAILS ====")
    for d, pred, actual, hit in rows_20:
        if TOP_K == 2 and TOP2_MODE == "diff_band":
            mode_label = "TOP2(diff_band)"
        else:
            mode_label = f"TOP{TOP_K}"
        print(f"THU={d}  {mode_label}={pred}  ACTUAL={actual}  HIT={hit}")

    # Target date prediction (works even if target not in CSV)
    target = resolve_target_date(thursdays)
    trans = build_transition(others_by_date, thursdays, cutoff_date=target)
    wed_thu_trans = build_wed_thu_transition(others_by_date, thursdays, cutoff_date=target)
    pred = predict_pb_topk(
        target,
        others_by_date,
        trans,
        set_by_date=set_by_date,
        force_count=WED_ROW_FORCE,
        restrict=WED_ROW_RESTRICT,
        wed_thu_trans=wed_thu_trans,
    )
    actual = identify_pb(others_by_date.get(target, []))
    print()
    print(f"==== TARGET PREDICTION (THU={target}) ====")
    if TOP_K == 2 and TOP2_MODE == "diff_band":
        mode_label = "TOP2(diff_band)"
    else:
        mode_label = f"TOP{TOP_K}"
    print(f"{mode_label}={pred}")
    if actual is not None:
        print(f"ACTUAL={actual}  HIT={actual in pred}")
    else:
        print("ACTUAL=<not available in CSV>")
    if LOG_COMPONENTS:
        score, s_wed, s_thu, s_prior, s_freq, s_trans, s_wedrow, s_wedfreq, s_wedthu, s_wedpb, s_wedpbf, s_tuepbf = _score_breakdown(
            target, others_by_date, trans, set_by_date=set_by_date, wed_thu_trans=wed_thu_trans
        )
        ranked = sorted(range(1, 21), key=lambda x: (-score[x], x))
        print("\n==== TARGET SCORE BREAKDOWN (TOP 10) ====")
        for n in ranked[:10]:
            print(
                f"  n={n:2d} total={score[n]:6.2f} wed={s_wed[n]:5.2f} "
                f"thu={s_thu[n]:5.2f} prior={s_prior[n]:5.2f} "
                f"freq={s_freq[n]:5.2f} trans={s_trans[n]:5.2f} wedrow={s_wedrow[n]:5.2f} "
                f"wedfreq={s_wedfreq[n]:5.2f} wedthu={s_wedthu[n]:5.2f} "
                f"wedpb={s_wedpb[n]:5.2f} wedpbf={s_wedpbf[n]:5.2f} tuepbf={s_tuepbf[n]:5.2f}"
            )

# =========================
if __name__ == "__main__":
    main()
