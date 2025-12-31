# -*- coding: utf-8 -*-
"""
Saturday Lotto prediction for 22 Nov 2025
Implements the full spec with:
- Calibration of WINDOW_SIZE_CAT* and WINDOW_SIZE_DEC*
- Main run with learning
- Final prediction (no learning)
- Cross-lottery hop (SFL-centred)
- Cold resurgence / late fire
No shortcuts.
"""

import math
import random
import statistics
import datetime
from collections import defaultdict, Counter
from itertools import combinations

# -----------------------------
# 0) DEFINITIONS & CONSTANTS
# -----------------------------

NUMBER_RANGE = range(1, 45)  # 1..45 inclusive
LOG_SCORE_MAX = 4.0
BASE_LEARNING_RATE_DECADE = 0.10

GAMMA_HOP = 0.30   # cross-lottery hop strength (Step 4E → Step E)
THETA_RESURGE = 0.25  # cold resurgence strength (Step E)

WINDOW_SIZE_CANDIDATES = [6, 7, 8, 9, 10]


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def decade(n):
    if 1 <= n <= 9:
        return 1
    if 10 <= n <= 19:
        return 2
    if 20 <= n <= 29:
        return 3
    if 30 <= n <= 39:
        return 4
    if 40 <= n <= 45:
        return 5
    return None  # ignored for decade-based calculations


class Draw:
    def __init__(self, date, lottery, main, supp=None, powerball=None):
        self.date = date  # datetime.date
        self.lottery = lottery  # str
        self.main = [n for n in main if 1 <= n <= 45]
        self.supp = [n for n in (supp or []) if 1 <= n <= 45]
        self.powerball = list(powerball or [])


def make_date(dmy_str):
    """
    Helper to build dates from 'Fri, 21 Nov 2025' style strings.
    """
    # Example: 'Fri, 21 Nov 2025'
    parts = dmy_str.split(',')
    rest = parts[1].strip()  # '21 Nov 2025'
    d_str, m_str, y_str = rest.split()
    day = int(d_str)
    year = int(y_str)
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    month = month_map[m_str]
    return datetime.date(year, month, day)


# ----------------------------------------
# HISTORICAL INPUT (GAME-AGNOSTIC)
# ----------------------------------------

ALL_DRAWS = []

# Set for Life
ALL_DRAWS += [
    Draw(make_date("Fri, 5 Dec 2025"), "Set for Life", [5, 25, 21, 17, 31, 1, 15], [24, 22]),
    Draw(make_date("Fri, 4 Dec 2025"), "Set for Life", [35, 2, 25, 8, 6, 17, 28], [3, 31]),
    Draw(make_date("Fri, 3 Dec 2025"), "Set for Life", [22, 29, 44, 31, 10, 25, 30], [8, 14]),
    Draw(make_date("Fri, 2 Dec 2025"), "Set for Life", [37, 13, 15, 19, 25, 39, 26], [3, 5]),
    Draw(make_date("Fri, 1 Dec 2025"), "Set for Life", [18, 1, 10, 41, 24, 11, 3], [25, 2]),
    Draw(make_date("Fri, 30 Nov 2025"), "Set for Life", [7, 44, 18, 27, 32, 22, 11], [38, 9]),
    Draw(make_date("Fri, 29 Nov 2025"), "Set for Life", [8, 31, 4, 6, 42, 16, 14], [13, 19]),
    Draw(make_date("Fri, 28 Nov 2025"), "Set for Life", [15, 27, 8, 39, 5, 43, 20], [19, 29]),
    Draw(make_date("Thu, 27 Nov 2025"), "Set for Life", [12, 36, 6, 7, 37, 41, 29], [8, 43]),
    Draw(make_date("Wed, 26 Nov 2025"), "Set for Life", [29, 37, 34, 14, 5, 21, 20], [18, 19]),
    Draw(make_date("Tue, 25 Nov 2025"), "Set for Life", [26, 16, 23, 15, 31, 1, 27], [8, 41]),
    Draw(make_date("Mon, 24 Nov 2025"), "Set for Life", [41, 1, 17, 29, 14, 40, 22], [35, 31]),
    Draw(make_date("Sun, 23 Nov 2025"), "Set for Life", [25, 27, 42, 18, 26, 9, 33], [22, 19]),
    Draw(make_date("Sat, 22 Nov 2025"), "Set for Life", [24, 23, 31, 30, 26, 5, 17], [6, 27]),
    Draw(make_date("Fri, 21 Nov 2025"), "Set for Life", [27, 32, 10, 42, 38, 33, 17], [19, 39]),
    Draw(make_date("Thu, 20 Nov 2025"), "Set for Life", [28, 10, 11, 35, 34, 41, 23], [30, 26]),
    Draw(make_date("Wed, 19 Nov 2025"), "Set for Life", [4, 44, 5, 33, 21, 30, 39], [9, 18]),
    Draw(make_date("Tue, 18 Nov 2025"), "Set for Life", [33, 35, 44, 32, 20, 29, 39], [5, 41]),
    Draw(make_date("Mon, 17 Nov 2025"), "Set for Life", [15, 23, 40, 43, 28, 1, 37], [18, 34]),
    Draw(make_date("Sun, 16 Nov 2025"), "Set for Life", [8, 19, 21, 27, 40, 14, 7], [20, 44]),
    Draw(make_date("Sat, 15 Nov 2025"), "Set for Life", [13, 4, 27, 14, 2, 5, 42], [33, 39]),
    Draw(make_date("Fri, 14 Nov 2025"), "Set for Life", [7, 25, 23, 35, 13, 18, 6], [3, 39]),
    Draw(make_date("Thu, 13 Nov 2025"), "Set for Life", [25, 24, 3, 21, 5, 33, 36], [22, 11]),
    Draw(make_date("Wed, 12 Nov 2025"), "Set for Life", [15, 20, 29, 21, 5, 10, 6], [32, 17]),
]


# Weekday Windfall
ALL_DRAWS += [
    Draw(make_date("Fri, 5 Dec 2025"), "Weekday Windfall", [9, 23, 8, 16, 11, 33], [34, 1]),
    Draw(make_date("Fri, 3 Dec 2025"), "Weekday Windfall", [15, 2, 38, 37, 22, 35], [39, 6]),
    Draw(make_date("Fri, 1 Dec 2025"), "Weekday Windfall", [8, 6, 30, 38, 36, 1], [43, 5]),
    Draw(make_date("Fri, 28 Nov 2025"), "Weekday Windfall", [30, 8, 25, 43, 39, 24], [21, 1]),
    Draw(make_date("Wed, 26 Nov 2025"), "Weekday Windfall", [44, 43, 8, 36, 16, 27], [31, 30]),
    Draw(make_date("Mon, 24 Nov 2025"), "Weekday Windfall", [44, 15, 20, 17, 4, 18], [7, 11]),
    Draw(make_date("Fri, 21 Nov 2025"), "Weekday Windfall", [4, 5, 26, 10, 40, 20], [14, 24]),
    Draw(make_date("Wed, 19 Nov 2025"), "Weekday Windfall", [43, 26, 35, 25, 42, 13], [24, 5]),
    Draw(make_date("Mon, 17 Nov 2025"), "Weekday Windfall", [37, 11, 4, 2, 5, 7], [30, 22]),
    Draw(make_date("Fri, 14 Nov 2025"), "Weekday Windfall", [34, 11, 28, 15, 44, 31], [9, 20]),
    Draw(make_date("Wed, 12 Nov 2025"), "Weekday Windfall", [35, 11, 33, 15, 34, 45], [8, 37]),
]


# OZ Lotto Tuesday
ALL_DRAWS += [
    Draw(make_date("Tue, 2 Dec 2025"), "OZ Lotto", [40, 26, 43, 28, 22, 42, 7], [29, 6, 47]),
    Draw(make_date("Tue, 25 Nov 2025"), "OZ Lotto", [12, 43, 28, 1, 47, 35, 14], [15, 16, 46]),
    Draw(make_date("Tue, 18 Nov 2025"), "OZ Lotto", [39, 2, 22, 8, 27, 6, 4], [47, 5, 24]),
]


# Powerball Thursday
ALL_DRAWS += [
    Draw(make_date("Thu, 4 Dec 2025"), "Powerball", [19, 23, 32, 12, 11, 15, 9], [], [14]),
    Draw(make_date("Thu, 27 Nov 2025"), "Powerball", [2, 17, 11, 9, 19, 28, 24], [], [1]),
    Draw(make_date("Thu, 20 Nov 2025"), "Powerball", [19, 11, 12, 4, 29, 13, 27], [], [20]),
    Draw(make_date("Thu, 13 Nov 2025"), "Powerball", [22, 10, 6, 15, 2, 8, 7], [], [13]),
]


# Saturday Lotto
ALL_DRAWS += [
    Draw(make_date("Sat, 29 Nov 2025"), "Saturday Lotto", [22, 10, 17, 5, 44, 36], [3, 11]),
    Draw(make_date("Sat, 22 Nov 2025"), "Saturday Lotto", [7, 31, 15, 39, 42, 12], [5, 8]),
    Draw(make_date("Sat, 15 Nov 2025"), "Saturday Lotto", [36, 19, 33, 41, 39, 1], [25, 20]),
]


# Map date -> draws list
DRAWS_BY_DATE = defaultdict(list)
for d in ALL_DRAWS:
    DRAWS_BY_DATE[d.date].append(d)

ALL_DATES = sorted(DRAWS_BY_DATE.keys())

# Target and prediction info
PREDICTION_DATE = make_date("Sat, 6 Dec 2025")

# Target draws for analysis (chronological learning), by lottery name + date
TARGET_DRAW_SPECS = [
    ("Saturday Lotto", make_date("Sat, 29 Nov 2025")),
    ("Weekday Windfall", make_date("Mon, 1 Dec 2025")),
    ("OZ Lotto", make_date("Tue, 2 Dec 2025")),
    ("Weekday Windfall", make_date("Wed, 3 Dec 2025")),
    ("Powerball", make_date("Thu, 4 Dec 2025")),
    ("Weekday Windfall", make_date("Fri, 5 Dec 2025")),
]


def find_actual_draw(lottery_name, date_obj):
    for d in DRAWS_BY_DATE.get(date_obj, []):
        if d.lottery == lottery_name:
            return d
    return None


# ---------------------------------------
# LEARNING STATE TEMPLATE
# ---------------------------------------

def reset_learning_state():
    return {
        "Delta_hot": 0.0,
        "Delta_warm": 0.0,
        "Delta_cold": 0.0,
        "cluster_priority_score_global": {},  # cluster tuple -> score
        "Delta_decade": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        "last_dom_decade": None,
    }


# ---------------------------------------
# STEP A: BUILD WINDOWS
# ---------------------------------------

def build_windows(target_date, window_size_cat, window_size_dec):
    # Category window dates and draws
    start_cat = target_date - datetime.timedelta(days=window_size_cat)
    end_cat = target_date - datetime.timedelta(days=1)
    seed_dates_cat = [start_cat + datetime.timedelta(days=i) for i in range((end_cat - start_cat).days + 1)]

    # Decade window dates and draws
    start_dec = target_date - datetime.timedelta(days=window_size_dec)
    end_dec = target_date - datetime.timedelta(days=1)
    seed_dates_dec = [start_dec + datetime.timedelta(days=i) for i in range((end_dec - start_dec).days + 1)]

    # If any required date has no draws → abort for that Dt
    for dt in seed_dates_cat:
        if dt not in DRAWS_BY_DATE:
            return None
    for dt in seed_dates_dec:
        if dt not in DRAWS_BY_DATE:
            return None

    seed_draws_cat = []
    for dt in seed_dates_cat:
        seed_draws_cat.extend(DRAWS_BY_DATE[dt])

    seed_draws_dec = []
    for dt in seed_dates_dec:
        seed_draws_dec.extend(DRAWS_BY_DATE[dt])

    return seed_dates_cat, seed_draws_cat, seed_dates_dec, seed_draws_dec


# ---------------------------------------
# STEP B: CATEGORY & DECADE STATS
# + 4E HOP
# ---------------------------------------

def compute_category_stats(seed_dates_cat, seed_draws_cat):
    # Step 4 (category window): f_main, f_supp, L(n), last_main_date(n), M
    seed_numbers_main = []
    seed_numbers_supp = []
    last_main_date = {n: None for n in NUMBER_RANGE}

    lotteries_seen_per_number = {n: set() for n in NUMBER_RANGE}

    for d in seed_draws_cat:
        for n in d.main:
            if n in NUMBER_RANGE:
                seed_numbers_main.append(n)
                if (last_main_date[n] is None) or (d.date > last_main_date[n]):
                    last_main_date[n] = d.date
                lotteries_seen_per_number[n].add(d.lottery)
        for s in d.supp:
            if s in NUMBER_RANGE:
                seed_numbers_supp.append(s)
                lotteries_seen_per_number[s].add(d.lottery)
        # powerballs do not count for 1..45 in this spec except via supp/extra if used

    f_main = Counter(seed_numbers_main)
    f_supp = Counter(seed_numbers_supp)
    L_num = {n: len(lotteries_seen_per_number[n]) for n in NUMBER_RANGE}

    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)

    return f_main, f_supp, L_num, last_main_date, M, seed_numbers_main


def compute_decade_window_stats(seed_draws_dec):
    # Step 4B: decade-side main frequencies
    dec_seed_numbers_main = []
    for d in seed_draws_dec:
        for n in d.main:
            if n in NUMBER_RANGE:
                dec_seed_numbers_main.append(n)

    f_main_dec = Counter(dec_seed_numbers_main)
    dec_main_count = {d: 0 for d in range(1, 6)}
    for n in NUMBER_RANGE:
        d = decade(n)
        if d is not None:
            dec_main_count[d] += f_main_dec[n]

    total_main = sum(dec_main_count.values())

    dec_w_window = {d: 1.0 for d in range(1, 6)}

    if total_main == 0:
        # all weights stay at 1.0
        return dec_main_count, dec_w_window

    dec_freq = {d: dec_main_count[d] / float(total_main) for d in range(1, 6)}
    max_dec_freq = max(dec_freq.values()) if dec_freq else 1.0
    if max_dec_freq == 0:
        max_dec_freq = 1.0

    for d in range(1, 6):
        if dec_main_count[d] == 0:
            dec_w_window[d] = 0.20
        else:
            dec_w_raw = 0.70 + 0.90 * (dec_freq[d] / max_dec_freq)
            dec_w_tmp = 0.75 * dec_w_raw + 0.25 * 1.0
            dec_w_window[d] = clamp(dec_w_tmp, 0.50, 1.70)

    return dec_main_count, dec_w_window


def apply_global_decade_learning(dec_w_window, learning_state):
    # Step 4C
    delta_dec = learning_state["Delta_decade"]

    dec_w_final = {}
    for d in range(1, 6):
        dec_w_final[d] = dec_w_window[d] * math.exp(delta_dec[d])

    avg_dec_w = sum(dec_w_final.values()) / 5.0 if dec_w_final else 1.0
    if avg_dec_w > 0:
        for d in dec_w_final:
            dec_w_final[d] = dec_w_final[d] / avg_dec_w

    return dec_w_final


def apply_last_3_days_momentum(target_date, seed_dates_dec, seed_draws_dec, dec_main_count, total_main_dec, dec_w_final):
    # Step 4D
    earliest = seed_dates_dec[0]
    recent_start_dec = max(target_date - datetime.timedelta(days=3), earliest)
    recent_dates = [recent_start_dec + datetime.timedelta(days=i)
                    for i in range((target_date - recent_start_dec).days)]

    dec_recent_count = {d: 0 for d in range(1, 6)}
    for d in seed_draws_dec:
        if d.date in recent_dates:
            for n in d.main:
                if n in NUMBER_RANGE:
                    dec_recent_count[decade(n)] += 1

    total_recent = sum(dec_recent_count.values())
    rec_w = {d: 1.0 for d in range(1, 6)}

    if total_recent > 0:
        dec_recent_freq = {d: dec_recent_count[d] / float(total_recent) for d in range(1, 6)}
        dec_freq_window = {d: (dec_main_count[d] / float(max(total_main_dec, 1)))
                           for d in range(1, 6)}
        eps = 1e-6
        for d in range(1, 6):
            ratio = dec_recent_freq[d] / (dec_freq_window[d] + eps)
            rec_w_raw = clamp(ratio, 0.8, 1.3)
            rec_w[d] = 0.7 * rec_w_raw + 0.3 * 1.0

    dec_w_recent = {}
    for d in range(1, 6):
        dec_w_recent[d] = dec_w_final[d] * rec_w[d]

    avg_dec_recent = sum(dec_w_recent.values()) / 5.0
    if avg_dec_recent > 0:
        for d in range(1, 6):
            dec_w_recent[d] = dec_w_recent[d] / avg_dec_recent

    return dec_w_recent


def compute_cross_lottery_hop(seed_dates_cat, seed_draws_cat, window_size_cat):
    """
    Step 4E: cross-lottery hop metrics, SFL-centered and Saturday destination biased.
    Returns cross_hop_log per n.
    """
    appearances = {n: [] for n in NUMBER_RANGE}  # list of (date, lottery)
    for d in seed_draws_cat:
        for n in d.main:
            if n in NUMBER_RANGE:
                appearances[n].append((d.date, d.lottery))

    cross_pair_sum = {n: 0.0 for n in NUMBER_RANGE}
    sfl_count = {n: 0 for n in NUMBER_RANGE}
    non_sfl_count = {n: 0 for n in NUMBER_RANGE}

    for n in NUMBER_RANGE:
        for (dt, lot) in appearances[n]:
            if lot == "Set for Life":
                sfl_count[n] += 1
            else:
                non_sfl_count[n] += 1

    for n in NUMBER_RANGE:
        pairs = appearances[n]
        if len(pairs) < 2:
            continue
        # consider ordered pairs (d1,L1) -> (d2,L2)
        for (d1, L1) in pairs:
            for (d2, L2) in pairs:
                if d2 <= d1 or L2 == L1:
                    continue
                lag_days = (d2 - d1).days
                if lag_days <= 0 or lag_days > window_size_cat:
                    continue
                lag_days = max(1, lag_days)
                base_pair = 1.0 / lag_days

                w_dir = 1.0
                if L1 == "Set for Life" and L2 != "Set for Life":
                    w_dir *= 1.5
                if L2 == "Saturday Lotto" and L1 != "Saturday Lotto":
                    w_dir *= 1.3

                pair_weight = base_pair * w_dir
                cross_pair_sum[n] += pair_weight

    base_hop_score = {}
    for n in NUMBER_RANGE:
        base_hop_score[n] = sfl_count[n] * non_sfl_count[n] + cross_pair_sum[n]

    max_hop = max(base_hop_score.values()) if base_hop_score else 0.0
    cross_hop_log = {n: 0.0 for n in NUMBER_RANGE}

    if max_hop > 0:
        for n in NUMBER_RANGE:
            score = base_hop_score[n] / max_hop
            cross_hop_log[n] = math.log(1.0 + GAMMA_HOP * score)
    # else all stay zero

    return cross_hop_log, base_hop_score


# ---------------------------------------
# STEP C: HOT / WARM / COLD CLASSIFICATION
# ---------------------------------------

def classify_hot_warm_cold(f_main, L_num):
    # Determine M and K
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)
    K = max(1, round(max(3, M * 0.15)))

    # sort numbers by (f_main desc, L desc, n asc)
    nums_sorted = sorted(
        NUMBER_RANGE,
        key=lambda x: (-f_main[x], -L_num[x], x)
    )

    hot_set = []
    for n in nums_sorted:
        if f_main[n] > 0 and len(hot_set) < K:
            hot_set.append(n)
    if len(hot_set) < K:
        remaining = [n for n in nums_sorted if f_main[n] == 0 and n not in hot_set]
        for n in remaining:
            if len(hot_set) < K:
                hot_set.append(n)

    hot_set = set(hot_set)

    warm_set = set()
    for n in NUMBER_RANGE:
        if 1 <= f_main[n] <= 2 and n not in hot_set:
            warm_set.add(n)

    cold_set = set()
    for n in NUMBER_RANGE:
        if f_main[n] == 0:
            cold_set.add(n)

    return hot_set, warm_set, cold_set, M, K


def compute_hw_c_bias(seed_draws_cat, hot_set, warm_set, cold_set):
    p_hot_list = []
    p_warm_list = []
    p_cold_list = []

    for d in seed_draws_cat:
        nums = [n for n in d.main if n in NUMBER_RANGE]
        if not nums:
            continue
        sz = len(nums)
        h = sum(1 for n in nums if n in hot_set)
        w = sum(1 for n in nums if n in warm_set)
        c = sum(1 for n in nums if n in cold_set)
        p_hot_list.append(h / sz)
        p_warm_list.append(w / sz)
        p_cold_list.append(c / sz)

    if not p_hot_list:
        avg_hot = avg_warm = avg_cold = 1.0 / 3.0
    else:
        avg_hot = sum(p_hot_list) / len(p_hot_list)
        avg_warm = sum(p_warm_list) / len(p_warm_list)
        avg_cold = sum(p_cold_list) / len(p_cold_list)

    if avg_warm >= avg_hot and avg_warm >= avg_cold:
        bias = "warm-heavy"
    elif avg_hot > avg_warm and avg_hot > avg_cold:
        bias = "hot-heavy"
    elif avg_cold > avg_hot and avg_cold > avg_warm:
        bias = "cold-heavy"
    else:
        bias = "balanced"

    return avg_hot, avg_warm, avg_cold, bias


# ---------------------------------------
# STEP D: CATEGORY WEIGHTS + LEARNING
# ---------------------------------------

def compute_category_weights(avg_hot, avg_warm, avg_cold, bias, learning_state):
    if bias == "hot-heavy":
        base_hot, base_warm, base_cold = 1.4, 1.15, 0.6
    elif bias == "warm-heavy":
        base_hot, base_warm, base_cold = 0.95, 1.35, 1.05
    elif bias == "cold-heavy":
        base_hot, base_warm, base_cold = 0.8, 1.0, 1.4
    else:
        base_hot, base_warm, base_cold = 1.0, 1.0, 0.95

    hot_w = base_hot * (1 + (avg_hot - 1/3) * 0.25)
    warm_w = base_warm * (1 + (avg_warm - 1/3) * 0.25)
    cold_w = base_cold * (1 + (avg_cold - 1/3) * 0.25)

    hot_w += learning_state["Delta_hot"]
    warm_w += learning_state["Delta_warm"]
    cold_w += learning_state["Delta_cold"]

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    s = hot_w + warm_w + cold_w
    if s > 0:
        hot_w /= s / 3.0
        warm_w /= s / 3.0
        cold_w /= s / 3.0

    # centralise 25% toward 1.0
    hot_w = 0.75 * hot_w + 0.25 * 1.0
    warm_w = 0.75 * warm_w + 0.25 * 1.0
    cold_w = 0.75 * cold_w + 0.25 * 1.0

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    return hot_w, warm_w, cold_w


# ---------------------------------------
# STEP E: PER-NUMBER LOG SCORES
# (including hop + resurgence)
# ---------------------------------------

def compute_per_number_scores(
    target_date,
    seed_dates_cat,
    seed_draws_cat,
    seed_numbers_main,
    f_main,
    f_supp,
    L_num,
    last_main_date,
    hot_set,
    warm_set,
    cold_set,
    hot_w,
    warm_w,
    cold_w,
    dec_w_recent,
    cross_hop_log,
    window_size_cat
):
    # adjacency
    adj_count = {n: 0 for n in NUMBER_RANGE}
    # treat seed_numbers_main as bag; adjacency counts neighbors in that bag
    for n in NUMBER_RANGE:
        c = 0
        c += seed_numbers_main.count(n - 1) if (n - 1) in NUMBER_RANGE else 0
        c += seed_numbers_main.count(n + 1) if (n + 1) in NUMBER_RANGE else 0
        adj_count[n] = c
    max_adj = max(adj_count.values()) if adj_count else 1
    if max_adj <= 0:
        max_adj = 1

    adj_log = {}
    for n in NUMBER_RANGE:
        adj_score_raw = 0.05 + 0.25 * (adj_count[n] / max_adj)
        adj_log[n] = math.log(1 + adj_score_raw)

    # delta last 3 days
    delta_log = {}
    recent_start = target_date - datetime.timedelta(days=3)
    recent_dates = [recent_start + datetime.timedelta(days=i) for i in range(3)]
    # note: only days strictly before target_date are in seed, but that's OK
    main_recent = set()
    supp_recent = set()
    for d in seed_draws_cat:
        if d.date in recent_dates:
            for x in d.main:
                if x in NUMBER_RANGE:
                    main_recent.add(x)
            for x in d.supp:
                if x in NUMBER_RANGE:
                    supp_recent.add(x)

    for n in NUMBER_RANGE:
        if n in main_recent:
            multiplier = 1.4
        elif (n not in main_recent) and (n in supp_recent):
            multiplier = 1.2
        else:
            multiplier = 1.0
        delta_log[n] = math.log(multiplier)

    # cross-lottery density
    cross_log = {}
    for n in NUMBER_RANGE:
        cross_log[n] = math.log(1 + 0.08 * L_num[n])

    # supp-only bonus
    supp_log = {}
    for n in NUMBER_RANGE:
        if f_main[n] == 0 and f_supp[n] > 0:
            supp_log[n] = math.log(1.05)
        else:
            supp_log[n] = 0.0

    # category weight log
    category_weight_log = {}
    for n in NUMBER_RANGE:
        if n in hot_set:
            category_weight_log[n] = math.log(hot_w)
        elif n in warm_set:
            category_weight_log[n] = math.log(warm_w)
        else:
            category_weight_log[n] = math.log(cold_w)

    # decade weight log
    decade_weight_log = {}
    for n in NUMBER_RANGE:
        d = decade(n)
        if d is not None:
            decade_weight_log[n] = math.log(dec_w_recent[d])
        else:
            decade_weight_log[n] = 0.0

    # cold resurgence
    resurge_raw = {n: 0.0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        lm = last_main_date[n]
        if lm is None:
            resurge_raw[n] = 0.0
        else:
            gap_days = (target_date - lm).days
            if 4 <= gap_days <= window_size_cat:
                resurge_raw[n] = 1.0 / gap_days
            else:
                resurge_raw[n] = 0.0

    max_resurge = max(resurge_raw.values()) if resurge_raw else 0.0
    cold_resurge_score = {n: 0.0 for n in NUMBER_RANGE}
    cold_resurge_log = {n: 0.0 for n in NUMBER_RANGE}
    if max_resurge > 0:
        for n in NUMBER_RANGE:
            cold_resurge_score[n] = resurge_raw[n] / max_resurge
            if n in hot_set:
                cold_resurge_log[n] = 0.0
            else:
                cold_resurge_log[n] = math.log(1.0 + THETA_RESURGE * cold_resurge_score[n])

    # total log_score
    log_score_raw = {}
    log_score = {}
    rawP = {}
    for n in NUMBER_RANGE:
        ls = (
            adj_log[n]
            + delta_log[n]
            + cross_log[n]
            + supp_log[n]
            + category_weight_log[n]
            + decade_weight_log[n]
            + cross_hop_log[n]
            + cold_resurge_log[n]
        )
        ls = min(ls, LOG_SCORE_MAX)
        log_score_raw[n] = ls
        val = math.exp(ls)
        rawP[n] = val

    total_rawP = sum(rawP.values())
    if total_rawP <= 0:
        P = {n: 1.0 / len(NUMBER_RANGE) for n in NUMBER_RANGE}
    else:
        P = {n: rawP[n] / total_rawP for n in NUMBER_RANGE}

    return {
        "log_score_raw": log_score_raw,
        "P": P,
        "cold_resurge_score": cold_resurge_score,
        "cold_resurge_log": cold_resurge_log,
    }


# ---------------------------------------
# STEP F: CLUSTER DETECTION
# ---------------------------------------

def detect_clusters(seed_draws_cat, cluster_priority_score_global):
    """
    Find clusters (size 2,3,4) appearing in >=2 draws.
    Return:
        clusters_info: dict cluster -> (freq, cluster_priority_value)
    """
    cluster_counts = Counter()
    # build clusters on main numbers only, per draw
    for d in seed_draws_cat:
        nums = sorted(set(n for n in d.main if n in NUMBER_RANGE))
        for size in (2, 3, 4):
            if len(nums) >= size:
                for comb in combinations(nums, size):
                    cluster_counts[comb] += 1

    clusters_info = {}
    for cluster, freq in cluster_counts.items():
        if freq >= 2:
            base = 1 + 0.2 * (freq - 1)
            global_off = cluster_priority_score_global.get(cluster, 0.0)
            cluster_priority = base * (1 + global_off)
            clusters_info[cluster] = (freq, cluster_priority)

    return clusters_info


# ---------------------------------------
# STEP G: COMPOSITION TARGETS
# ---------------------------------------

def get_draw_size_for_lottery(lottery):
    if lottery == "Saturday Lotto":
        return 6
    elif lottery == "Weekday Windfall":
        return 6
    elif lottery == "Powerball":
        return 7
    elif lottery == "OZ Lotto":
        return 7
    elif lottery == "Set for Life":
        return 7
    else:
        return 6  # default safeguard


def compute_composition_targets(draw_size, avg_hot, avg_warm, avg_cold, hot_set, warm_set, cold_set):
    h_target = round(draw_size * avg_hot)
    w_target = round(draw_size * avg_warm)
    c_target = draw_size - h_target - w_target

    # clamp to non-negative and not exceeding set sizes
    h_target = max(0, min(h_target, len(hot_set)))
    w_target = max(0, min(w_target, len(warm_set)))
    c_target = max(0, min(c_target, len(cold_set)))

    # repair if total < draw_size
    total = h_target + w_target + c_target
    all_pool_size = len(hot_set | warm_set | cold_set)
    if total > all_pool_size:
        total = all_pool_size

    while total < draw_size:
        # try to add to the largest available category that still has capacity
        options = []
        if h_target < len(hot_set):
            options.append("H")
        if w_target < len(warm_set):
            options.append("W")
        if c_target < len(cold_set):
            options.append("C")
        if not options:
            break
        choice = random.choice(options)
        if choice == "H":
            h_target += 1
        elif choice == "W":
            w_target += 1
        else:
            c_target += 1
        total = h_target + w_target + c_target

    return h_target, w_target, c_target


# ---------------------------------------
# STEP H: MONTE CARLO SAMPLING
# ---------------------------------------

def monte_carlo_sampling(
    lottery_name,
    P,
    hot_set,
    warm_set,
    cold_set,
    avg_hot,
    avg_warm,
    avg_cold,
    clusters_info,
    draw_size,
    M
):
    # TRIALS = clamp(50000 * sqrt(max(1, M/20)), 10000, 50000)
    complexity = math.sqrt(max(1.0, M / 20.0))
    TRIALS = int(clamp(200000 * complexity, 10000, 200000))

    max_avg = max(avg_hot, avg_warm, avg_cold)
    # EXPLORE_FRAC in [0.10,0.20]
    EXPLORE_FRAC = clamp(0.10 + 0.20 * max(0.0, max_avg - 1/3), 0.10, 0.20)

    h_target, w_target, c_target = compute_composition_targets(
        draw_size, avg_hot, avg_warm, avg_cold, hot_set, warm_set, cold_set
    )

    # Precompute cluster boost per number
    cluster_boost = {n: 1.0 for n in NUMBER_RANGE}
    for cluster, (freq, priority) in clusters_info.items():
        for n in cluster:
            # simple max priority
            cluster_boost[n] = max(cluster_boost[n], priority)

    def select_from_pool(pool, already, required, explore_mode):
        """
        pool: set of n (hot/warm/cold)
        already: current chosen numbers
        required: how many to pick
        explore_mode: bool
        returns: list of chosen numbers
        """
        chosen = []
        available = list(pool - already)
        for _ in range(required):
            if not available:
                break
            weights = []
            for n in available:
                base_p = P[n]
                # cluster factor (soft)
                cb = cluster_boost.get(n, 1.0)
                # compress cluster boost a bit
                cluster_factor = 1.0 + 0.1 * (cb - 1.0)
                # exploratory boost for cold numbers globally
                cold_factor = 1.0
                if explore_mode and n in cold_set:
                    cold_factor = 1.2
                w = base_p * cluster_factor * cold_factor
                weights.append(w)
            s = sum(weights)
            if s <= 0:
                idx = random.randrange(len(available))
                pick = available[idx]
            else:
                r = random.random() * s
                acc = 0.0
                pick = available[-1]
                for n, w in zip(available, weights):
                    acc += w
                    if acc >= r:
                        pick = n
                        break
            chosen.append(pick)
            already.add(pick)
            available.remove(pick)
        return chosen

    freq_tuples = Counter()
    hot_total = 0.0
    warm_total = 0.0
    cold_total = 0.0
    dec_total = {d: 0.0 for d in range(1, 6)}

    for _ in range(TRIALS):
        explore_mode = (random.random() < EXPLORE_FRAC)
        chosen = set()

        # build composition sequence (H/W/C) and shuffle
        labels = ["H"] * h_target + ["W"] * w_target + ["C"] * c_target
        random.shuffle(labels)

        for label in labels:
            if label == "H":
                chosen_list = select_from_pool(hot_set, chosen, 1, explore_mode)
            elif label == "W":
                chosen_list = select_from_pool(warm_set, chosen, 1, explore_mode)
            else:
                chosen_list = select_from_pool(cold_set, chosen, 1, explore_mode)

            if not chosen_list:
                # try from all sets
                pool_all = (hot_set | warm_set | cold_set) - chosen
                if not pool_all:
                    continue
                chosen_list = select_from_pool(pool_all, chosen, 1, explore_mode)

        # if still not enough, fill from global
        while len(chosen) < draw_size:
            pool_all = (hot_set | warm_set | cold_set) - chosen
            if not pool_all:
                break
            chosen_list = select_from_pool(pool_all, chosen, 1, explore_mode)

        if len(chosen) == 0:
            continue

        tup = tuple(sorted(chosen))
        freq_tuples[tup] += 1

        # accumulate diagnostics
        h_count = sum(1 for n in chosen if n in hot_set)
        w_count = sum(1 for n in chosen if n in warm_set)
        c_count = sum(1 for n in chosen if n in cold_set)
        hot_total += h_count
        warm_total += w_count
        cold_total += c_count

        for n in chosen:
            d = decade(n)
            if d is not None:
                dec_total[d] += 1

    if TRIALS == 0:
        hot_pred = warm_pred = cold_pred = 0.0
        dec_pred = {d: 0.0 for d in range(1, 6)}
    else:
        hot_pred = hot_total / TRIALS
        warm_pred = warm_total / TRIALS
        cold_pred = cold_total / TRIALS
        dec_pred = {d: dec_total[d] / TRIALS for d in range(1, 6)}

    # sort top-10 tuples
    top10 = freq_tuples.most_common(20)
    top10_with_prob = [(tup, cnt / TRIALS) for tup, cnt in top10]

    return {
        "TRIALS": TRIALS,
        "EXPLORE_FRAC": EXPLORE_FRAC,
        "h_target": h_target,
        "w_target": w_target,
        "c_target": c_target,
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "dec_pred": dec_pred,
        "top10": top10_with_prob,
        "freq_tuples": freq_tuples,
    }


# ---------------------------------------
# STEP I: DIAGNOSTICS
# (built as a dict returned by pipeline)
# ---------------------------------------

# We will build diagnostics inside pipeline_for_draw()


# ---------------------------------------
# STEP J: LEARNING
# ---------------------------------------

def learning_step(learning_state, diag, actual_draw_main, hot_set, warm_set, cold_set, clusters_info):
    # Category learning
    hot_actual = sum(1 for n in actual_draw_main if n in hot_set)
    warm_actual = sum(1 for n in actual_draw_main if n in warm_set)
    cold_actual = sum(1 for n in actual_draw_main if n in cold_set)

    hot_pred = diag["hot_pred"]
    cold_pred = diag["cold_pred"]

    hot_error = hot_actual - hot_pred
    cold_error = cold_actual - cold_pred

    # learning_rate based on variance of f_main
    f_values = diag["f_main_values"]
    if len(f_values) > 1:
        var_f = statistics.pvariance(f_values)
    else:
        var_f = 0.0
    learning_rate = clamp(0.02 + 0.02 * var_f, 0.02, 0.10)

    step_hot = clamp(math.copysign(learning_rate * abs(hot_error), hot_error), -0.1, 0.1)
    step_cold = clamp(math.copysign(learning_rate * abs(cold_error), cold_error), -0.1, 0.1)

    learning_state["Delta_hot"] += step_hot
    learning_state["Delta_cold"] += step_cold
    # warm is opposite to keep sum stable
    learning_state["Delta_warm"] = - (learning_state["Delta_hot"] + learning_state["Delta_cold"]) / 2.0

    learning_state["Delta_hot"] = clamp(learning_state["Delta_hot"], -0.5, 0.5)
    learning_state["Delta_warm"] = clamp(learning_state["Delta_warm"], -0.5, 0.5)
    learning_state["Delta_cold"] = clamp(learning_state["Delta_cold"], -0.5, 0.5)

    # Cluster learning
    actual_set = set(actual_draw_main)
    cpsg = learning_state["cluster_priority_score_global"]
    for cluster in clusters_info.keys():
        cluster_set = set(cluster)
        if cluster_set.issubset(actual_set):
            cpsg[cluster] = cpsg.get(cluster, 0.0) + 0.05
        else:
            cpsg[cluster] = cpsg.get(cluster, 0.0) - 0.02

    for c in cpsg:
        cpsg[c] = clamp(cpsg[c], -0.5, 0.5)

    # Decade learning
    dec_actual = {d: 0 for d in range(1, 6)}
    for n in actual_draw_main:
        d = decade(n)
        if d is not None:
            dec_actual[d] += 1

    dec_pred = diag["dec_pred"]
    delta_dec = learning_state["Delta_decade"]

    for d in range(1, 6):
        dec_error = dec_actual[d] - dec_pred[d]
        step_d = BASE_LEARNING_RATE_DECADE * dec_error
        step_d = clamp(step_d, -0.15, 0.15)
        delta_dec[d] += step_d

    # oscillation regulariser
    LOW = {1, 2}
    HIGH = {3, 4, 5}

    dom_decade_t = max(range(1, 6), key=lambda d: dec_actual[d])
    last_dom = learning_state["last_dom_decade"]
    if last_dom is not None and dom_decade_t == last_dom:
        if dom_decade_t in LOW:
            for dh in HIGH:
                delta_dec[dh] += 0.03
            for dl in LOW:
                delta_dec[dl] -= 0.03
        else:
            for dl in LOW:
                delta_dec[dl] += 0.03
            for dh in HIGH:
                delta_dec[dh] -= 0.03

    learning_state["last_dom_decade"] = dom_decade_t

    for d in range(1, 6):
        delta_dec[d] = clamp(delta_dec[d], -0.8, 0.8)

    mean_delta = sum(delta_dec.values()) / 5.0
    for d in range(1, 6):
        delta_dec[d] -= mean_delta


# ---------------------------------------
# PIPELINE FOR ONE DRAW (Steps 3–10+11)
# ---------------------------------------

def pipeline_for_draw(lottery_name, target_date, window_size_cat, window_size_dec, learning_state, is_prediction=False):
    # Step A: build windows
    win = build_windows(target_date, window_size_cat, window_size_dec)
    if win is None:
        return None  # abort for that Dt
    seed_dates_cat, seed_draws_cat, seed_dates_dec, seed_draws_dec = win

    # Step B (category side)
    f_main, f_supp, L_num, last_main_date, M, seed_numbers_main = compute_category_stats(
        seed_dates_cat, seed_draws_cat
    )

    # Step 4B,4C,4D (decade side)
    dec_main_count, dec_w_window = compute_decade_window_stats(seed_draws_dec)
    total_main_dec = sum(dec_main_count.values())
    dec_w_final = apply_global_decade_learning(dec_w_window, learning_state)
    dec_w_recent = apply_last_3_days_momentum(
        target_date, seed_dates_dec, seed_draws_dec, dec_main_count, total_main_dec, dec_w_final
    )

    # Step 4E: cross-lottery hop
    cross_hop_log, base_hop_score = compute_cross_lottery_hop(
        seed_dates_cat, seed_draws_cat, window_size_cat
    )

    # Step C: hot/warm/cold
    hot_set, warm_set, cold_set, M, K = classify_hot_warm_cold(f_main, L_num)
    avg_hot, avg_warm, avg_cold, bias = compute_hw_c_bias(seed_draws_cat, hot_set, warm_set, cold_set)

    # Step D: category weights
    hot_w, warm_w, cold_w = compute_category_weights(avg_hot, avg_warm, avg_cold, bias, learning_state)

    # Step E: per-number log scores
    per_num = compute_per_number_scores(
        target_date,
        seed_dates_cat,
        seed_draws_cat,
        seed_numbers_main,
        f_main,
        f_supp,
        L_num,
        last_main_date,
        hot_set,
        warm_set,
        cold_set,
        hot_w,
        warm_w,
        cold_w,
        dec_w_recent,
        cross_hop_log,
        window_size_cat
    )
    P = per_num["P"]

    # Step F: cluster detection
    clusters_info = detect_clusters(seed_draws_cat, learning_state["cluster_priority_score_global"])

    # Step G: composition targets
    draw_size = get_draw_size_for_lottery(lottery_name)

    # Step H: Monte Carlo sampling
    mc = monte_carlo_sampling(
        lottery_name,
        P,
        hot_set,
        warm_set,
        cold_set,
        avg_hot,
        avg_warm,
        avg_cold,
        clusters_info,
        draw_size,
        M
    )

    # Step I: diagnostics
    diag = {
        "seed_dates_cat": seed_dates_cat,
        "seed_dates_dec": seed_dates_dec,
        "hot_set": hot_set,
        "warm_set": warm_set,
        "cold_set": cold_set,
        "M": M,
        "K": K,
        "avg_hot": avg_hot,
        "avg_warm": avg_warm,
        "avg_cold": avg_cold,
        "bias": bias,
        "hot_w": hot_w,
        "warm_w": warm_w,
        "cold_w": cold_w,
        "dec_main_count": dec_main_count,
        "dec_w_window": dec_w_window,
        "dec_w_final": dec_w_final,
        "dec_w_recent": dec_w_recent,
        "cross_hop_score": base_hop_score,
        "cross_hop_log": cross_hop_log,
        "cold_resurge_score": per_num["cold_resurge_score"],
        "cold_resurge_log": per_num["cold_resurge_log"],
        "TRIALS": mc["TRIALS"],
        "EXPLORE_FRAC": mc["EXPLORE_FRAC"],
        "h_target": mc["h_target"],
        "w_target": mc["w_target"],
        "c_target": mc["c_target"],
        "hot_pred": mc["hot_pred"],
        "warm_pred": mc["warm_pred"],
        "cold_pred": mc["cold_pred"],
        "dec_pred": mc["dec_pred"],
        "top10": mc["top10"],
        "clusters_info": clusters_info,
        "f_main_values": [f_main[n] for n in NUMBER_RANGE],
    }

    return diag


# ---------------------------------------
# CALIBRATION & MAIN RUN
# ---------------------------------------

def calibrate_category_window():
    best_W_cat = None
    best_MSE = None

    for W_cat in WINDOW_SIZE_CANDIDATES:
        ls = reset_learning_state()
        errors = []

        for (lot_name, dt) in sorted(TARGET_DRAW_SPECS, key=lambda x: x[1]):
            actual_draw = find_actual_draw(lot_name, dt)
            if actual_draw is None:
                continue
            diag = pipeline_for_draw(lot_name, dt, W_cat, W_cat, ls, is_prediction=False)
            if diag is None:
                # aborted for this Dt
                continue

            # compute hot_actual, cold_actual
            hot_set = diag["hot_set"]
            warm_set = diag["warm_set"]
            cold_set = diag["cold_set"]

            actual = actual_draw.main
            hot_actual = sum(1 for n in actual if n in hot_set)
            cold_actual = sum(1 for n in actual if n in cold_set)

            hot_pred = diag["hot_pred"]
            cold_pred = diag["cold_pred"]

            errors.append((hot_actual - hot_pred) ** 2 + (cold_actual - cold_pred) ** 2)

            # apply learning
            learning_step(ls, diag, actual, hot_set, warm_set, cold_set, diag["clusters_info"])

        if not errors:
            continue

        mse = sum(errors) / len(errors)
        if best_MSE is None or mse < best_MSE:
            best_MSE = mse
            best_W_cat = W_cat

    return best_W_cat


def calibrate_decade_window(W_CAT_STAR):
    best_W_dec = None
    best_score = None

    for W_dec in WINDOW_SIZE_CANDIDATES:
        ls = reset_learning_state()
        dec_errors = []
        z_values = []

        for (lot_name, dt) in sorted(TARGET_DRAW_SPECS, key=lambda x: x[1]):
            actual_draw = find_actual_draw(lot_name, dt)
            if actual_draw is None:
                continue
            diag = pipeline_for_draw(lot_name, dt, W_CAT_STAR, W_dec, ls, is_prediction=False)
            if diag is None:
                continue

            # dec_actual
            dec_actual = {d: 0 for d in range(1, 6)}
            for n in actual_draw.main:
                d = decade(n)
                if d is not None:
                    dec_actual[d] += 1

            dec_pred = diag["dec_pred"]
            for d in range(1, 6):
                dec_errors.append((dec_actual[d] - dec_pred[d]) ** 2)

            low_t = dec_pred[1] + dec_pred[2]
            high_t = dec_pred[3] + dec_pred[4] + dec_pred[5]
            z_t = low_t - high_t
            z_values.append(z_t)

            # learning
            hot_set = diag["hot_set"]
            warm_set = diag["warm_set"]
            cold_set = diag["cold_set"]
            learning_step(ls, diag, actual_draw.main, hot_set, warm_set, cold_set, diag["clusters_info"])

        if not dec_errors:
            continue

        decade_MSE = sum(dec_errors) / len(dec_errors)
        if len(z_values) > 1:
            stability_penalty = statistics.pvariance(z_values)
        else:
            stability_penalty = 0.0

        alpha = 1.0
        beta = 0.5
        score_dec = alpha * decade_MSE + beta * stability_penalty

        if best_score is None or score_dec < best_score:
            best_score = score_dec
            best_W_dec = W_dec

    return best_W_dec


def main():
    random.seed(0)

    # 1) Calibration for CATEGORY
    W_CAT_STAR = calibrate_category_window()
    if W_CAT_STAR is None:
        print("Failed to calibrate WINDOW_SIZE_CAT*")
        return

    # 2) Calibration for DECADE
    W_DEC_STAR = calibrate_decade_window(W_CAT_STAR)
    if W_DEC_STAR is None:
        print("Failed to calibrate WINDOW_SIZE_DEC*")
        return

    print(f"WINDOW_SIZE_CAT* = {W_CAT_STAR}")
    print(f"WINDOW_SIZE_DEC* = {W_DEC_STAR}")
    print()

    # 3) Main run with chosen windows
    ls = reset_learning_state()
    # process all target draws in chronological order
    for (lot_name, dt) in sorted(TARGET_DRAW_SPECS, key=lambda x: x[1]):
        actual_draw = find_actual_draw(lot_name, dt)
        if actual_draw is None:
            continue
        diag = pipeline_for_draw(lot_name, dt, W_CAT_STAR, W_DEC_STAR, ls, is_prediction=False)
        if diag is None:
            continue

        hot_set = diag["hot_set"]
        warm_set = diag["warm_set"]
        cold_set = diag["cold_set"]
        learning_step(ls, diag, actual_draw.main, hot_set, warm_set, cold_set, diag["clusters_info"])

    # 4) Final prediction for Saturday Lotto, 22 Nov 2025
    pred_lot = "Saturday Lotto"
    diag_pred = pipeline_for_draw(pred_lot, PREDICTION_DATE, W_CAT_STAR, W_DEC_STAR, ls, is_prediction=True)
    if diag_pred is None:
        print("Prediction aborted due to window gaps.")
        return

    print("=== H/W/C + CATEGORY DIAGNOSTICS ===")
    print(f"avg_hot  = {diag_pred['avg_hot']}")
    print(f"avg_warm = {diag_pred['avg_warm']}")
    print(f"avg_cold = {diag_pred['avg_cold']}")
    print(f"bias = {diag_pred['bias']}")
    print(f"Hot_set  = {sorted(list(diag_pred['hot_set']))}")
    print(f"Warm_set size = {len(diag_pred['warm_set'])}")
    print(f"Cold_set size = {len(diag_pred['cold_set'])}")
    print(f"hot_w = {diag_pred['hot_w']}")
    print(f"warm_w = {diag_pred['warm_w']}")
    print(f"cold_w = {diag_pred['cold_w']}")
    print()

    print("=== TARGET COMPOSITION FROM MONTE CARLO ===")
    print(f"h_target = {diag_pred['h_target']}")
    print(f"w_target = {diag_pred['w_target']}")
    print(f"c_target = {diag_pred['c_target']}")
    print(f"hot_pred  = {diag_pred['hot_pred']}")
    print(f"warm_pred = {diag_pred['warm_pred']}")
    print(f"cold_pred = {diag_pred['cold_pred']}")
    print()

    print("=== DECADE STATS ===")
    print(f"dec_main_count = {diag_pred['dec_main_count']}")
    print(f"dec_w_window = {diag_pred['dec_w_window']}")
    print(f"dec_w_final = {diag_pred['dec_w_final']}")
    print(f"dec_w_recent = {diag_pred['dec_w_recent']}")
    print(f"dec_pred = {diag_pred['dec_pred']}")
    print()

    print("=== MONTE CARLO PARAMS ===")
    print(f"TRIALS = {diag_pred['TRIALS']}")
    print(f"EXPLORE_FRAC = {diag_pred['EXPLORE_FRAC']}")
    print()

    print("=== CLUSTERS DETECTED ===")
    clusters_info = diag_pred["clusters_info"]
    for cluster, (freq, priority) in list(clusters_info.items())[:20]:
        print(f"{cluster} -> freq={freq}, priority={priority}")
    print(f"... total clusters: {len(clusters_info)}")
    print()

    print("Top-10 predicted 6-number tuples for Saturday 22 Nov 2025:")
    for tup, prob in diag_pred["top10"]:
        if len(tup) == 6:
            print(f"{tup}  ->  {prob:.6f}")


if __name__ == "__main__":
    main()
