#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Powerball prediction pipeline implementing the strict algorithm spec.

Prediction target:
- Powerball Thursday, 27 Nov 2025
- 7 main numbers from 1..35
- 1 Powerball extra from 1..20

This script follows the user's algorithm specification step-by-step.
No steps are skipped; calibration, main run, and prediction are all executed.
"""

import math
import random
import datetime
from collections import defaultdict, Counter
import statistics

# -------------------------
# 0) DEFINITIONS & HELPERS
# -------------------------

NUMBER_MIN = 1
NUMBER_MAX = 35
EXTRA_MIN = 1
EXTRA_MAX = 20

NUMBER_RANGE = range(NUMBER_MIN, NUMBER_MAX + 1)
EXTRA_RANGE = range(EXTRA_MIN, EXTRA_MAX + 1)

LOG_SCORE_MAX = 4.0
BASE_LEARNING_RATE_DECADE = 0.10  # for Δ_decade
EPSILON = 1e-6

WINDOW_SIZE_CAT_CANDIDATES = [6, 7, 8, 9, 10]
WINDOW_SIZE_DEC_CANDIDATES = [6, 7, 8, 9, 10]

random.seed(0)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def decade(n):
    if NUMBER_MIN <= n <= 10:
        return 1
    if 11 <= n <= 20:
        return 2
    if 21 <= n <= 30:
        return 3
    if 31 <= n <= 35:
        return 4
    return None  # ignore for decade-based stats


# -------------------------
# Historical Input
# -------------------------

date = datetime.date

# Each draw: {"date": date, "lottery": str, "main": [...], "supp": [...], "extra": int or None}
DRAWS = []

def add_draw(d, lottery, main, supp=None, extra=None):
    if supp is None:
        supp = []
    DRAWS.append({
        "date": d,
        "lottery": lottery,
        "main": list(main),
        "supp": list(supp),
        "extra": extra,
    })

# --- Set for Life Draw Results ---
add_draw(date(2025, 11, 26), "Set for Life", [29, 37, 34, 14, 5, 21, 20], [18, 19])
add_draw(date(2025, 11, 25), "Set for Life", [26, 16, 23, 15, 31, 1, 27], [8, 41])
add_draw(date(2025, 11, 24), "Set for Life", [41, 1, 17, 29, 14, 40, 22], [35, 31])
add_draw(date(2025, 11, 23), "Set for Life", [25, 27, 42, 18, 26, 9, 33], [22, 19])
add_draw(date(2025, 11, 22), "Set for Life", [24, 23, 31, 30, 26, 5, 17], [6, 27])
add_draw(date(2025, 11, 21), "Set for Life", [27, 32, 10, 42, 38, 33, 17], [19, 39])
add_draw(date(2025, 11, 20), "Set for Life", [28, 10, 11, 35, 34, 41, 23], [30, 26])
add_draw(date(2025, 11, 19), "Set for Life", [4, 44, 5, 33, 21, 30, 39], [9, 18])
add_draw(date(2025, 11, 18), "Set for Life", [33, 35, 44, 32, 20, 29, 39], [5, 41])
add_draw(date(2025, 11, 17), "Set for Life", [15, 23, 40, 43, 28, 1, 37], [18, 34])
add_draw(date(2025, 11, 16), "Set for Life", [8, 19, 21, 27, 40, 14, 7], [20, 44])
add_draw(date(2025, 11, 15), "Set for Life", [13, 4, 27, 14, 2, 5, 42], [33, 39])
add_draw(date(2025, 11, 14), "Set for Life", [7, 25, 23, 35, 13, 18, 6], [3, 39])
add_draw(date(2025, 11, 13), "Set for Life", [25, 24, 3, 21, 5, 33, 36], [22, 11])
add_draw(date(2025, 11, 12), "Set for Life", [15, 20, 29, 21, 5, 10, 6], [32, 17])
add_draw(date(2025, 11, 11), "Set for Life", [4, 7, 10, 44, 32, 30, 26], [5, 18])
add_draw(date(2025, 11, 10), "Set for Life", [5, 36, 13, 23, 39, 3, 9], [35, 6])
add_draw(date(2025, 11, 9),  "Set for Life", [11, 4, 44, 26, 6, 31, 40], [21, 33])
add_draw(date(2025, 11, 8),  "Set for Life", [7, 31, 5, 37, 43, 38, 2], [42, 10])
add_draw(date(2025, 11, 7),  "Set for Life", [30, 18, 6, 28, 33, 41, 14], [38, 29])
add_draw(date(2025, 11, 6),  "Set for Life", [12, 20, 35, 42, 41, 10, 18], [33, 32])
add_draw(date(2025, 11, 5),  "Set for Life", [16, 22, 13, 34, 25, 3, 18], [33, 43])
add_draw(date(2025, 11, 4),  "Set for Life", [38, 9, 27, 25, 10, 23, 37], [13, 17])
add_draw(date(2025, 11, 3),  "Set for Life", [8, 15, 25, 26, 13, 24, 23], [4, 2])

# --- Weekday Windfall Draw Results ---
add_draw(date(2025, 11, 26), "Weekday Windfall", [44, 43, 8, 36, 16, 27], [31, 30])
add_draw(date(2025, 11, 24), "Weekday Windfall", [44, 15, 20, 17, 4, 18], [7, 11])
add_draw(date(2025, 11, 21), "Weekday Windfall", [4, 5, 26, 10, 40, 20], [14, 24])
add_draw(date(2025, 11, 19), "Weekday Windfall", [43, 26, 35, 25, 42, 13], [24, 5])
add_draw(date(2025, 11, 17), "Weekday Windfall", [37, 11, 4, 2, 5, 7], [30, 22])
add_draw(date(2025, 11, 14), "Weekday Windfall", [34, 11, 28, 15, 44, 31], [9, 20])
add_draw(date(2025, 11, 12), "Weekday Windfall", [35, 11, 33, 15, 34, 45], [8, 37])
add_draw(date(2025, 11, 10), "Weekday Windfall", [38, 3, 31, 22, 28, 5], [26, 14])
add_draw(date(2025, 11, 7),  "Weekday Windfall", [31, 16, 23, 30, 6, 3], [13, 18])
add_draw(date(2025, 11, 5),  "Weekday Windfall", [26, 15, 18, 27, 7, 37], [19, 44])
add_draw(date(2025, 11, 3),  "Weekday Windfall", [25, 14, 29, 23, 45, 13], [31, 8])

# --- OZ Lotto Tuesday Draw Results ---
add_draw(date(2025, 11, 25), "OZ Lotto", [12, 43, 28, 1, 47, 35, 14], [15, 16, 46])
add_draw(date(2025, 11, 18), "OZ Lotto", [39, 2, 22, 8, 27, 6, 4], [47, 5, 24])
add_draw(date(2025, 11, 11), "OZ Lotto", [44, 30, 7, 28, 17, 34, 42], [20, 32, 3])
add_draw(date(2025, 11, 4),  "OZ Lotto", [21, 17, 43, 25, 12, 18, 14], [15, 42, 24])

# --- Powerball Thursday Draw Results ---
add_draw(date(2025, 11, 20), "Powerball", [19, 11, 12, 4, 29, 13, 27], [], 20)
add_draw(date(2025, 11, 13), "Powerball", [22, 10, 6, 15, 2, 8, 7], [], 13)
add_draw(date(2025, 11, 6),  "Powerball", [11, 34, 7, 33, 15, 22, 16], [], 13)

# --- Saturday Lotto Draw Results ---
add_draw(date(2025, 11, 22), "Saturday Lotto", [7, 31, 15, 39, 42, 12], [5, 8])
add_draw(date(2025, 11, 15), "Saturday Lotto", [36, 19, 33, 41, 39, 1], [25, 20])
add_draw(date(2025, 11, 8),  "Saturday Lotto", [28, 13, 1, 41, 14, 16], [39, 34])

# Build a quick index by date
DRAWS_BY_DATE = defaultdict(list)
for dr in DRAWS:
    DRAWS_BY_DATE[dr["date"]].append(dr)


# Target draws for analysis (calibration phase)
CALIBRATION_TARGETS = [
    (date(2025, 11, 20), "Powerball"),
    (date(2025, 11, 21), "Weekday Windfall"),
    (date(2025, 11, 22), "Saturday Lotto"),
    (date(2025, 11, 24), "Weekday Windfall"),
    (date(2025, 11, 25), "OZ Lotto"),
    (date(2025, 11, 26), "Weekday Windfall"),
]

# Main run targets (learning phase)
MAIN_RUN_TARGETS = [
    (date(2025, 11, 13), "Powerball"),
    (date(2025, 11, 14), "Weekday Windfall"),
    (date(2025, 11, 15), "Saturday Lotto"),
    (date(2025, 11, 17), "Weekday Windfall"),
    (date(2025, 11, 18), "OZ Lotto"),
    (date(2025, 11, 19), "Weekday Windfall"),
]

# Prediction date
PREDICTION_DATE = date(2025, 11, 27)
PREDICTION_LOTTERY = "Powerball"


# -------------------------
# Learning state template
# -------------------------

def init_learning_state():
    return {
        "Delta_hot": 0.0,
        "Delta_warm": 0.0,
        "Delta_cold": 0.0,
        "Delta_decade": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        "cluster_scores": defaultdict(float),  # cluster -> offset
        "last_dom_decade": None,
    }


# -------------------------
# Utility: draw access
# -------------------------

def get_draw_by_date_and_lottery(d, lottery):
    for dr in DRAWS_BY_DATE.get(d, []):
        if dr["lottery"] == lottery:
            return dr
    return None


def get_all_draws_on_dates(dates):
    res = []
    for dd in dates:
        res.extend(DRAWS_BY_DATE.get(dd, []))
    return res


def ensure_all_dates_have_draws(dates):
    for dd in dates:
        if len(DRAWS_BY_DATE.get(dd, [])) == 0:
            return False
    return True


# -------------------------
# STEP A: build windows
# -------------------------

def build_seed_windows(target_date, window_cat, window_dec):
    # Category window dates
    cat_dates = [
        target_date - datetime.timedelta(days=k)
        for k in range(1, window_cat + 1)
    ]
    cat_dates.reverse()  # chronological
    # Decade window dates
    dec_dates = [
        target_date - datetime.timedelta(days=k)
        for k in range(1, window_dec + 1)
    ]
    dec_dates.reverse()

    if not ensure_all_dates_have_draws(cat_dates):
        return None
    if not ensure_all_dates_have_draws(dec_dates):
        return None

    seed_draws_cat = get_all_draws_on_dates(cat_dates)
    seed_draws_dec = get_all_draws_on_dates(dec_dates)
    return {
        "cat_dates": cat_dates,
        "dec_dates": dec_dates,
        "seed_draws_cat": seed_draws_cat,
        "seed_draws_dec": seed_draws_dec,
    }


# -------------------------
# STEP B: frequencies & decades
# -------------------------

def compute_category_frequencies(seed_draws_cat):
    seed_numbers_main_cat = []
    seed_numbers_supp_cat = []

    lotteries_seen_by_num = {n: set() for n in NUMBER_RANGE}

    for dr in seed_draws_cat:
        lotto = dr["lottery"]
        # main numbers
        for n in dr["main"]:
            if NUMBER_MIN <= n <= NUMBER_MAX:
                seed_numbers_main_cat.append(n)
                lotteries_seen_by_num[n].add(lotto)
        # supp + PB extra
        for s in dr["supp"]:
            if EXTRA_MIN <= s <= EXTRA_MAX:
                seed_numbers_supp_cat.append(s)
                lotteries_seen_by_num.setdefault(s, set()).add(lotto)
        if dr["extra"] is not None:
            x = dr["extra"]
            if EXTRA_MIN <= x <= EXTRA_MAX:
                seed_numbers_supp_cat.append(x)
                lotteries_seen_by_num.setdefault(x, set()).add(lotto)

    f_main = {n: 0 for n in NUMBER_RANGE}
    f_supp = {n: 0 for n in NUMBER_RANGE}
    L = {n: 0 for n in NUMBER_RANGE}

    for n in seed_numbers_main_cat:
        f_main[n] += 1
    for s in seed_numbers_supp_cat:
        if s in f_supp:
            f_supp[s] += 1

    for n in NUMBER_RANGE:
        L[n] = len(lotteries_seen_by_num.get(n, set()))

    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)

    # Extra-ball raw frequency over 1..20 for STEP E-EXTRA
    f_extra_total = {x: 0 for x in EXTRA_RANGE}
    lotteries_seen_by_extra = {x: set() for x in EXTRA_RANGE}
    for dr in seed_draws_cat:
        lotto = dr["lottery"]
        for s in dr["supp"]:
            if EXTRA_MIN <= s <= EXTRA_MAX:
                f_extra_total[s] += 1
                lotteries_seen_by_extra[s].add(lotto)
        if dr["extra"] is not None:
            x = dr["extra"]
            if EXTRA_MIN <= x <= EXTRA_MAX:
                f_extra_total[x] += 1
                lotteries_seen_by_extra[x].add(lotto)

    return {
        "seed_numbers_main_cat": seed_numbers_main_cat,
        "seed_numbers_supp_cat": seed_numbers_supp_cat,
        "f_main": f_main,
        "f_supp": f_supp,
        "L": L,
        "M": M,
        "f_extra_total": f_extra_total,
        "lotteries_seen_by_extra": lotteries_seen_by_extra,
    }


def compute_decade_weights(seed_draws_dec, target_date, delta_decade):
    # Step 4B: f_main_dec, dec_main_count, dec_w_window
    f_main_dec = {n: 0 for n in NUMBER_RANGE}
    for dr in seed_draws_dec:
        for n in dr["main"]:
            if NUMBER_MIN <= n <= NUMBER_MAX:
                f_main_dec[n] += 1

    dec_main_count = {1: 0, 2: 0, 3: 0, 4: 0}
    for n in NUMBER_RANGE:
        d = decade(n)
        if d is not None:
            dec_main_count[d] += f_main_dec[n]

    total_main_in_range = sum(dec_main_count.values())
    dec_w_window = {}

    if total_main_in_range == 0:
        for d in range(1, 5):
            dec_w_window[d] = 1.0
    else:
        dec_freq = {}
        for d in range(1, 5):
            dec_freq[d] = dec_main_count[d] / total_main_in_range
        max_dec_freq = max(dec_freq.values()) if dec_freq else 1.0
        if max_dec_freq == 0:
            max_dec_freq = 1.0
        for d in range(1, 5):
            if dec_main_count[d] == 0:
                dec_w_window[d] = 0.20
            else:
                dec_w_raw = 0.70 + 0.90 * (dec_freq[d] / max_dec_freq)
                dec_w_tmp = 0.75 * dec_w_raw + 0.25 * 1.0
                dec_w_window[d] = clamp(dec_w_tmp, 0.50, 1.70)

    # Step 4C: apply global learning
    dec_w_final = {}
    for d in range(1, 5):
        dec_w_final[d] = dec_w_window[d] * math.exp(delta_decade[d])

    avg_dec_w = sum(dec_w_final.values()) / 4.0
    if avg_dec_w > 0:
        for d in range(1, 5):
            dec_w_final[d] /= avg_dec_w

    # Step 4D: last-3-days decade momentum
    dec_dates = sorted({dr["date"] for dr in seed_draws_dec})
    if not dec_dates:
        # no data
        dec_w_recent = {d: dec_w_final[d] for d in range(1, 5)}
        return {
            "dec_main_count": dec_main_count,
            "dec_w_window": dec_w_window,
            "dec_w_final": dec_w_final,
            "dec_w_recent": dec_w_recent,
            "decade_weight_log": {n: 0.0 for n in NUMBER_RANGE},
        }

    earliest_in_window = dec_dates[0]
    recent_start = max(target_date - datetime.timedelta(days=3), earliest_in_window)
    recent_dates = [d for d in dec_dates if recent_start <= d <= target_date - datetime.timedelta(days=1)]

    dec_recent_count = {1: 0, 2: 0, 3: 0, 4: 0}
    for rd in recent_dates:
        for dr in DRAWS_BY_DATE[rd]:
            for n in dr["main"]:
                if NUMBER_MIN <= n <= NUMBER_MAX:
                    d = decade(n)
                    if d is not None:
                        dec_recent_count[d] += 1

    total_recent = sum(dec_recent_count.values())
    rec_w = {}
    if total_recent == 0:
        for d in range(1, 5):
            rec_w[d] = 1.0
    else:
        for d in range(1, 5):
            dec_recent_freq = dec_recent_count[d] / total_recent
            dec_freq_window = dec_main_count[d] / max(total_main_in_range, 1)
            ratio = dec_recent_freq / (dec_freq_window + EPSILON)
            rec_w_raw = clamp(ratio, 0.8, 1.3)
            rec_w[d] = 0.7 * rec_w_raw + 0.3 * 1.0

    dec_w_recent = {}
    for d in range(1, 5):
        dec_w_recent[d] = dec_w_final[d] * rec_w[d]
    avg_dec_recent = sum(dec_w_recent.values()) / 4.0
    if avg_dec_recent > 0:
        for d in range(1, 5):
            dec_w_recent[d] /= avg_dec_recent

    decade_weight_log = {}
    for n in NUMBER_RANGE:
        d = decade(n)
        if d is None:
            decade_weight_log[n] = 0.0
        else:
            decade_weight_log[n] = math.log(dec_w_recent[d])

    return {
        "dec_main_count": dec_main_count,
        "dec_w_window": dec_w_window,
        "dec_w_final": dec_w_final,
        "dec_w_recent": dec_w_recent,
        "decade_weight_log": decade_weight_log,
    }


# -------------------------
# STEP C: Hot/Warm/Cold
# -------------------------

def classify_hot_warm_cold(f_main, L, seed_draws_cat):
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)
    K = max(1, round(max(3, M * 0.15)))

    # sort by (-f_main, -L, +n)
    nums_sorted = sorted(
        NUMBER_RANGE,
        key=lambda n: (-f_main[n], -L[n], n)
    )
    hot_candidates = [n for n in nums_sorted if f_main[n] > 0]
    Hot_set = []
    for n in hot_candidates:
        if len(Hot_set) < K:
            Hot_set.append(n)
    if len(Hot_set) < K:
        zero_freq = [n for n in nums_sorted if f_main[n] == 0]
        for n in zero_freq:
            if len(Hot_set) < K and n not in Hot_set:
                Hot_set.append(n)

    Hot_set = set(Hot_set)
    Warm_set = set(n for n in NUMBER_RANGE if 1 <= f_main[n] <= 2) - Hot_set
    Cold_set = set(n for n in NUMBER_RANGE if f_main[n] == 0)

    # per-draw classification
    p_hot_list = []
    p_warm_list = []
    p_cold_list = []

    for dr in seed_draws_cat:
        draw_nums = [n for n in dr["main"] if NUMBER_MIN <= n <= NUMBER_MAX]
        if not draw_nums:
            continue
        sz = len(draw_nums)
        c_hot = sum(1 for n in draw_nums if n in Hot_set)
        c_warm = sum(1 for n in draw_nums if n in Warm_set)
        c_cold = sum(1 for n in draw_nums if n in Cold_set)
        p_hot_list.append(c_hot / sz)
        p_warm_list.append(c_warm / sz)
        p_cold_list.append(c_cold / sz)

    if p_hot_list:
        avg_hot = sum(p_hot_list) / len(p_hot_list)
        avg_warm = sum(p_warm_list) / len(p_warm_list)
        avg_cold = sum(p_cold_list) / len(p_cold_list)
    else:
        avg_hot = avg_warm = avg_cold = 1.0 / 3.0

    # bias decision: warm > hot > cold > balanced
    if avg_warm >= avg_hot and avg_warm >= avg_cold:
        bias = "warm-heavy"
    elif avg_hot > avg_warm and avg_hot > avg_cold:
        bias = "hot-heavy"
    elif avg_cold > avg_hot and avg_cold > avg_warm:
        bias = "cold-heavy"
    else:
        bias = "balanced"

    return {
        "Hot_set": Hot_set,
        "Warm_set": Warm_set,
        "Cold_set": Cold_set,
        "M": M,
        "K": K,
        "avg_hot": avg_hot,
        "avg_warm": avg_warm,
        "avg_cold": avg_cold,
        "bias": bias,
    }


# -------------------------
# STEP D: Category weights
# -------------------------

def compute_category_weights(avg_hot, avg_warm, avg_cold, bias, state):
    if bias == "hot-heavy":
        hot_base, warm_base, cold_base = (1.4, 1.15, 0.6)
    elif bias == "warm-heavy":
        hot_base, warm_base, cold_base = (0.95, 1.35, 1.05)
    elif bias == "cold-heavy":
        hot_base, warm_base, cold_base = (0.8, 1.0, 1.4)
    else:  # balanced
        hot_base, warm_base, cold_base = (1.0, 1.0, 0.95)

    hot_w = hot_base * (1 + (avg_hot - 1/3) * 0.25)
    warm_w = warm_base * (1 + (avg_warm - 1/3) * 0.25)
    cold_w = cold_base * (1 + (avg_cold - 1/3) * 0.25)

    hot_w += state["Delta_hot"]
    warm_w += state["Delta_warm"]
    cold_w += state["Delta_cold"]

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    sum_w = hot_w + warm_w + cold_w
    if sum_w > 0:
        scale = 3.0 / sum_w
        hot_w *= scale
        warm_w *= scale
        cold_w *= scale

    # centralise 25% toward 1.0
    hot_w = 1.0 + 0.75 * (hot_w - 1.0)
    warm_w = 1.0 + 0.75 * (warm_w - 1.0)
    cold_w = 1.0 + 0.75 * (cold_w - 1.0)

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    return hot_w, warm_w, cold_w


# -------------------------
# STEP E: per-number log scores
# -------------------------

def compute_delta_multiplier_for_number(n, seed_draws_cat, seed_dates_cat):
    # Use last 3 calendar days in seed_dates_cat
    if not seed_dates_cat:
        return 1.0
    last_date = seed_dates_cat[-1]
    recent_start = max(seed_dates_cat[0], last_date - datetime.timedelta(days=2))
    recent_dates = [d for d in seed_dates_cat if recent_start <= d <= last_date]

    seen_main = False
    seen_supp_only = False
    for d in recent_dates:
        for dr in DRAWS_BY_DATE[d]:
            # main
            if n in dr["main"]:
                if NUMBER_MIN <= n <= NUMBER_MAX:
                    seen_main = True
            # supp + extra
            if dr["supp"]:
                if n in dr["supp"]:
                    seen_supp_only = True
            if dr["extra"] is not None and dr["extra"] == n:
                seen_supp_only = True

    if seen_main:
        return 1.4
    elif seen_supp_only:
        return 1.2
    else:
        return 1.0


def compute_per_number_log_scores(seed_numbers_main_cat,
                                  seed_numbers_supp_cat,
                                  f_main, f_supp, L,
                                  Hot_set, Warm_set, Cold_set,
                                  hot_w, warm_w, cold_w,
                                  decade_weight_log,
                                  seed_dates_cat):
    # adjacency
    adj_count = {n: 0 for n in NUMBER_RANGE}
    seed_main = [n for n in seed_numbers_main_cat if NUMBER_MIN <= n <= NUMBER_MAX]
    seed_main_set = set(seed_main)
    for n in NUMBER_RANGE:
        cnt = 0
        if (n - 1) in seed_main_set:
            cnt += seed_main.count(n - 1)
        if (n + 1) in seed_main_set:
            cnt += seed_main.count(n + 1)
        adj_count[n] = cnt
    max_adj = max(adj_count.values()) if adj_count else 1
    if max_adj <= 0:
        max_adj = 1

    P = {}
    log_score = {}
    for n in NUMBER_RANGE:
        # adjacency
        adj_score_raw = 0.05 + 0.25 * (adj_count[n] / max_adj)
        adj_log = math.log(1 + adj_score_raw)

        # delta last 3 days
        delta_mult = compute_delta_multiplier_for_number(n, seed_numbers_main_cat, seed_dates_cat)
        delta_log = math.log(delta_mult)

        # cross-lottery density
        cross_log = math.log(1 + 0.08 * L[n])

        # supp-only bonus
        if f_main[n] == 0 and f_supp[n] > 0:
            supp_log = math.log(1.05)
        else:
            supp_log = 0.0

        # category weight
        if n in Hot_set:
            category_weight_log = math.log(hot_w)
        elif n in Warm_set:
            category_weight_log = math.log(warm_w)
        else:
            category_weight_log = math.log(cold_w)

        dlog = decade_weight_log.get(n, 0.0)

        log_score_raw = adj_log + delta_log + cross_log + supp_log + category_weight_log + dlog
        log_score[n] = min(log_score_raw, LOG_SCORE_MAX)

    rawP = {n: math.exp(log_score[n]) for n in NUMBER_RANGE}
    total = sum(rawP.values())
    if total <= 0:
        # fallback uniform
        for n in NUMBER_RANGE:
            P[n] = 1.0 / len(NUMBER_RANGE)
    else:
        for n in NUMBER_RANGE:
            P[n] = rawP[n] / total

    return P, log_score


# -------------------------
# STEP E-EXTRA: extra ball 1..20
# -------------------------

def compute_extra_probabilities(seed_draws_cat, info_cat):
    f_extra_total = info_cat["f_extra_total"]
    lotteries_seen_by_extra = info_cat["lotteries_seen_by_extra"]

    # adjacency in supp/extra
    seed_supp_extra_all = []
    for dr in seed_draws_cat:
        for s in dr["supp"]:
            if EXTRA_MIN <= s <= EXTRA_MAX:
                seed_supp_extra_all.append(s)
        if dr["extra"] is not None:
            x = dr["extra"]
            if EXTRA_MIN <= x <= EXTRA_MAX:
                seed_supp_extra_all.append(x)
    seed_supp_set = set(seed_supp_extra_all)

    adj_count_extra = {x: 0 for x in EXTRA_RANGE}
    for x in EXTRA_RANGE:
        cnt = 0
        if (x - 1) in seed_supp_set:
            cnt += seed_supp_extra_all.count(x - 1)
        if (x + 1) in seed_supp_set:
            cnt += seed_supp_extra_all.count(x + 1)
        adj_count_extra[x] = cnt
    max_adj_extra = max(adj_count_extra.values()) if adj_count_extra else 1
    if max_adj_extra <= 0:
        max_adj_extra = 1

    # last-3-days presence
    dates = sorted({dr["date"] for dr in seed_draws_cat})
    if dates:
        last_date = dates[-1]
        recent_start = max(dates[0], last_date - datetime.timedelta(days=2))
        recent_dates = [d for d in dates if recent_start <= d <= last_date]
    else:
        recent_dates = []

    last3_seen = {x: False for x in EXTRA_RANGE}
    for d in recent_dates:
        for dr in DRAWS_BY_DATE[d]:
            for s in dr["supp"]:
                if EXTRA_MIN <= s <= EXTRA_MAX:
                    last3_seen[s] = True
            if dr["extra"] is not None and EXTRA_MIN <= dr["extra"] <= EXTRA_MAX:
                last3_seen[dr["extra"]] = True

    # cross-lottery density for extras
    P_extra_raw = {}
    for x in EXTRA_RANGE:
        base = 1.0 + 0.10 * f_extra_total[x]
        density = 1.0 + 0.05 * len(lotteries_seen_by_extra[x])
        adj = 1.0 + 0.20 * (adj_count_extra[x] / max_adj_extra)
        last3_mult = 1.3 if last3_seen[x] else 1.0
        P_extra_raw[x] = base * density * adj * last3_mult

    total = sum(P_extra_raw.values())
    if total <= 0:
        P_extra = {x: 1.0 / len(EXTRA_RANGE) for x in EXTRA_RANGE}
    else:
        P_extra = {x: P_extra_raw[x] / total for x in EXTRA_RANGE}

    predicted_extra = max(P_extra.items(), key=lambda kv: kv[1])[0]

    return P_extra, predicted_extra


# -------------------------
# STEP F: cluster detection
# -------------------------

def find_clusters(seed_numbers_main_cat):
    # Use only n in NUMBER_RANGE
    # Build per-draw unique sets for combinations
    draws_sets = []
    # But we need draws; seed_numbers_main_cat is flattened, so instead use DRAWS_BY_DATE
    # For cluster detection, we operate per draw in category window. Caller will pass the actual draws.
    # We'll handle that in the pipeline, here we just define logic for a list of per-draw sets.

    # This function expects a list of sets, each draw = set of main numbers in NUMBER_RANGE.
    raise NotImplementedError("Use find_clusters_for_draws instead")


def find_clusters_for_draws(draw_sets):
    # draw_sets: list[set of main numbers in NUMBER_RANGE] per draw
    cluster_freq = Counter()
    for draw_set in draw_sets:
        nums = sorted(draw_set)
        # clusters of size 2,3,4
        for k in (2, 3, 4):
            if len(nums) < k:
                continue
            # local combinations
            # simple manual combinations to avoid importing itertools excessively in hot loops
            if k == 2:
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        cluster = (nums[i], nums[j])
                        cluster_freq[cluster] += 1
            elif k == 3:
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        for l in range(j+1, len(nums)):
                            cluster = (nums[i], nums[j], nums[l])
                            cluster_freq[cluster] += 1
            elif k == 4:
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        for l in range(j+1, len(nums)):
                            for m in range(l+1, len(nums)):
                                cluster = (nums[i], nums[j], nums[l], nums[m])
                                cluster_freq[cluster] += 1

    # keep only freq >= 2
    clusters = {C: f for C, f in cluster_freq.items() if f >= 2}
    return clusters


def compute_cluster_priorities(clusters, cluster_scores_global):
    priorities = {}
    for C, freq in clusters.items():
        base = 1 + 0.2 * (freq - 1)
        score = cluster_scores_global.get(C, 0.0)
        priorities[C] = base * (1 + score)
    return priorities


# -------------------------
# STEP G: composition targets
# -------------------------

def draw_size_for_lottery(lottery):
    if lottery in ("Saturday Lotto", "Weekday Windfall"):
        return 6
    elif lottery == "Powerball":
        return 7
    elif lottery in ("OZ Lotto", "Set for Life"):
        return 7
    else:
        # default to 7
        return 7


def compute_composition_targets(draw_size, avg_hot, avg_warm, avg_cold,
                                Hot_set, Warm_set, Cold_set):
    h_target = round(draw_size * avg_hot)
    w_target = round(draw_size * avg_warm)
    c_target = draw_size - h_target - w_target

    # clamp non-negative
    h_target = max(0, h_target)
    w_target = max(0, w_target)
    c_target = max(0, c_target)

    # clamp to set sizes
    h_target = min(h_target, len(Hot_set))
    w_target = min(w_target, len(Warm_set))
    c_target = min(c_target, len(Cold_set))

    # if still short of draw_size, borrow from others
    while h_target + w_target + c_target < draw_size:
        # borrow from the largest remaining pool
        candidates = []
        if h_target < len(Hot_set):
            candidates.append("hot")
        if w_target < len(Warm_set):
            candidates.append("warm")
        if c_target < len(Cold_set):
            candidates.append("cold")
        if not candidates:
            break
        choice = random.choice(candidates)
        if choice == "hot":
            h_target += 1
        elif choice == "warm":
            w_target += 1
        else:
            c_target += 1

    return h_target, w_target, c_target


# -------------------------
# STEP H: Monte Carlo sampling
# -------------------------

def build_draw_sets_for_clusters(seed_draws_cat):
    draw_sets = []
    for dr in seed_draws_cat:
        s = set(n for n in dr["main"] if NUMBER_MIN <= n <= NUMBER_MAX)
        if s:
            draw_sets.append(s)
    return draw_sets


def choose_with_weights(candidates, weights):
    # candidates: list of numbers
    # weights: dict number -> weight
    arr = []
    ws = []
    for n in candidates:
        w = max(weights.get(n, 0.0), 0.0)
        arr.append(n)
        ws.append(w)
    if not arr or sum(ws) <= 0:
        return random.choice(candidates)
    return random.choices(arr, weights=ws, k=1)[0]


def build_clusters_by_number(clusters):
    # clusters: dict cluster_tuple -> priority
    by_num = defaultdict(list)
    for C in clusters.keys():
        for n in C:
            by_num[n].append(C)
    return by_num


def compute_cluster_bias_for_number(n, chosen, clusters, cluster_priorities):
    # Softly favour numbers that complete or extend clusters
    bias = 1.0
    for C in clusters:
        if n not in C:
            continue
        already = sum(1 for x in C if x in chosen)
        if already > 0:
            pr = cluster_priorities.get(C, 1.0)
            bias += 0.1 * pr * already / len(C)
    return bias


def monte_carlo_sampling(lottery,
                         P, dec_w_recent,
                         Hot_set, Warm_set, Cold_set,
                         h_target, w_target, c_target,
                         clusters, cluster_priorities):
    M = sum(1 for n in NUMBER_RANGE if P[n] > 0)
    complexity = math.sqrt(max(1, M / 20.0))
    TRIALS = int(clamp(50000 * complexity, 10000, 50000))

    # EXPLORE_FRAC based on dominance
    # approximate: if max category share > 0.45 -> more exploration
    # (note: we will receive avg_hot, avg_warm, avg_cold separately if needed;
    # here we derive a rough measure from category set sizes)
    total_cat = len(Hot_set) + len(Warm_set) + len(Cold_set)
    if total_cat == 0:
        explore_frac = 0.15
    else:
        # simple heuristic on sets (not perfect but deterministic)
        share_hot = len(Hot_set) / total_cat
        share_warm = len(Warm_set) / total_cat
        share_cold = len(Cold_set) / total_cat
        max_share = max(share_hot, share_warm, share_cold)
        if max_share > 0.5:
            explore_frac = 0.20
        elif max_share > 0.4:
            explore_frac = 0.15
        else:
            explore_frac = 0.10

    draw_size = draw_size_for_lottery(lottery)

    freq_tuples = Counter()
    hot_count_sum = 0.0
    warm_count_sum = 0.0
    cold_count_sum = 0.0
    dec_count_sum = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}

    # Precompute exploration distribution base
    # P'(n): P(n) boosted for Cold_set and under-used decades
    P_explore = {}
    for n in NUMBER_RANGE:
        w = P[n]
        if n in Cold_set:
            w *= 1.20
        d = decade(n)
        if d is not None and dec_w_recent[d] < 1.0:
            w *= 1.10
        P_explore[n] = w
    total_explore = sum(P_explore.values())
    if total_explore > 0:
        for n in NUMBER_RANGE:
            P_explore[n] /= total_explore
    else:
        P_explore = dict(P)

    clusters_by_number = build_clusters_by_number(clusters)

    for _ in range(TRIALS):
        is_explore = (random.random() < explore_frac)
        base_P = P_explore if is_explore else P

        chosen = []
        chosen_set = set()

        def sample_from_set(set_name, target_count, base_set):
            nonlocal chosen, chosen_set
            for _i in range(target_count):
                candidates = [n for n in base_set if n not in chosen_set]
                if not candidates:
                    return
                # weights with cluster bias
                weights = {}
                for n in candidates:
                    w = base_P[n]
                    if clusters:
                        bias = compute_cluster_bias_for_number(
                            n, chosen_set, clusters_by_number[n], cluster_priorities
                        )
                        w *= bias
                    weights[n] = w
                picked = choose_with_weights(candidates, weights)
                chosen.append(picked)
                chosen_set.add(picked)

        # sample hot/warm/cold
        sample_from_set("hot", h_target, Hot_set)
        sample_from_set("warm", w_target, Warm_set)
        sample_from_set("cold", c_target, Cold_set)

        # If still not enough, fill using global P
        while len(chosen) < draw_size:
            candidates = [n for n in NUMBER_RANGE if n not in chosen_set]
            if not candidates:
                break
            picked = choose_with_weights(candidates, base_P)
            chosen.append(picked)
            chosen_set.add(picked)

        chosen_sorted = tuple(sorted(chosen[:draw_size]))
        freq_tuples[chosen_sorted] += 1

        # diagnostics counters
        c_hot = sum(1 for n in chosen_sorted if n in Hot_set)
        c_warm = sum(1 for n in chosen_sorted if n in Warm_set)
        c_cold = sum(1 for n in chosen_sorted if n in Cold_set)
        hot_count_sum += c_hot
        warm_count_sum += c_warm
        cold_count_sum += c_cold

        dec_cnt = {1: 0, 2: 0, 3: 0, 4: 0}
        for n in chosen_sorted:
            d = decade(n)
            if d is not None:
                dec_cnt[d] += 1
        for d in range(1, 5):
            dec_count_sum[d] += dec_cnt[d]

    hot_pred = hot_count_sum / TRIALS
    warm_pred = warm_count_sum / TRIALS
    cold_pred = cold_count_sum / TRIALS
    dec_pred = {d: dec_count_sum[d] / TRIALS for d in range(1, 5)}

    # convert to probabilities
    prob_tuples = {T: c / TRIALS for T, c in freq_tuples.items()}
    top_tuples = sorted(prob_tuples.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "TRIALS": TRIALS,
        "EXPLORE_FRAC": explore_frac,
        "prob_tuples": prob_tuples,
        "top_tuples": top_tuples,
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "dec_pred": dec_pred,
    }


# -------------------------
# STEP 11/12: Learning
# -------------------------

def compute_learning_rate_from_f_main(f_main):
    vals = [f_main[n] for n in NUMBER_RANGE]
    if len(vals) <= 1:
        return 0.02
    var = statistics.pvariance(vals)
    # keep bounded
    return 0.02 + 0.01 * min(var, 5.0)


def apply_learning(state, diagnostics, actual_main):
    """
    Applies category, cluster, and decade learning for a historical target.
    Also applies decade oscillation regulariser and clamp/de-mean.
    """
    Hot_set = diagnostics["Hot_set"]
    Warm_set = diagnostics["Warm_set"]
    Cold_set = diagnostics["Cold_set"]
    hot_pred = diagnostics["hot_pred"]
    warm_pred = diagnostics["warm_pred"]
    cold_pred = diagnostics["cold_pred"]
    f_main = diagnostics["f_main"]
    dec_pred = diagnostics["dec_pred"]
    dec_actual = diagnostics["dec_actual"]
    clusters = diagnostics["clusters"]

    # Category actual counts
    hot_actual = sum(1 for n in actual_main if n in Hot_set)
    warm_actual = sum(1 for n in actual_main if n in Warm_set)
    cold_actual = sum(1 for n in actual_main if n in Cold_set)

    hot_error = hot_actual - hot_pred
    cold_error = cold_actual - cold_pred

    learning_rate = compute_learning_rate_from_f_main(f_main)

    delta_hot_step = clamp(math.copysign(learning_rate * abs(hot_error), hot_error), -0.1, 0.1) if hot_error != 0 else 0.0
    delta_cold_step = clamp(math.copysign(learning_rate * abs(cold_error), cold_error), -0.1, 0.1) if cold_error != 0 else 0.0

    state["Delta_hot"] += delta_hot_step
    state["Delta_cold"] += delta_cold_step
    state["Delta_warm"] = - (state["Delta_hot"] + state["Delta_cold"]) / 2.0

    state["Delta_hot"] = clamp(state["Delta_hot"], -0.5, 0.5)
    state["Delta_warm"] = clamp(state["Delta_warm"], -0.5, 0.5)
    state["Delta_cold"] = clamp(state["Delta_cold"], -0.5, 0.5)

    # Cluster learning
    actual_set = set(actual_main)
    for C in clusters.keys():
        if all(n in actual_set for n in C):
            state["cluster_scores"][C] += 0.05
        else:
            state["cluster_scores"][C] -= 0.02
        state["cluster_scores"][C] = clamp(state["cluster_scores"][C], -0.5, 0.5)

    # Decade learning
    for d in range(1, 5):
        error_d = dec_actual[d] - dec_pred[d]
        step_d = BASE_LEARNING_RATE_DECADE * error_d
        step_d = clamp(step_d, -0.15, 0.15)
        state["Delta_decade"][d] += step_d

    # Decade oscillation regulariser
    LOW = {1, 2}
    HIGH = {3, 4}

    # dominant decade based on actual
    dom_decade = max(range(1, 5), key=lambda dd: (dec_actual[dd], -dd))
    last_dom = state["last_dom_decade"]
    if last_dom is not None and dom_decade == last_dom:
        if dom_decade in LOW:
            for dh in HIGH:
                state["Delta_decade"][dh] += 0.03
            for dl in LOW:
                state["Delta_decade"][dl] -= 0.03
        else:
            for dl in LOW:
                state["Delta_decade"][dl] += 0.03
            for dh in HIGH:
                state["Delta_decade"][dh] -= 0.03
    state["last_dom_decade"] = dom_decade

    # clamp & de-mean Δ_decade
    for d in range(1, 5):
        state["Delta_decade"][d] = clamp(state["Delta_decade"][d], -0.8, 0.8)

    mean_delta_dec = sum(state["Delta_decade"][d] for d in range(1, 5)) / 4.0
    for d in range(1, 5):
        state["Delta_decade"][d] -= mean_delta_dec

    diagnostics["hot_actual"] = hot_actual
    diagnostics["warm_actual"] = warm_actual
    diagnostics["cold_actual"] = cold_actual
    diagnostics["dec_actual"] = dec_actual
    diagnostics["dom_decade"] = dom_decade


# -------------------------
# PIPELINE FOR ONE TARGET
# -------------------------

def run_pipeline_for_target(target_date,
                            lottery,
                            window_cat,
                            window_dec,
                            state,
                            do_learning,
                            is_prediction=False):
    """
    Runs Steps 3→10 (and learning if do_learning=True and not prediction)
    for a single target date+lottery.
    Returns diagnostics dict or None if window infeasible.
    """
    sw = build_seed_windows(target_date, window_cat, window_dec)
    if sw is None:
        return None

    cat_dates = sw["cat_dates"]
    dec_dates = sw["dec_dates"]
    seed_draws_cat = sw["seed_draws_cat"]
    seed_draws_dec = sw["seed_draws_dec"]

    # STEP B category
    info_cat = compute_category_frequencies(seed_draws_cat)
    f_main = info_cat["f_main"]
    f_supp = info_cat["f_supp"]
    L = info_cat["L"]
    M = info_cat["M"]
    seed_numbers_main_cat = info_cat["seed_numbers_main_cat"]
    seed_numbers_supp_cat = info_cat["seed_numbers_supp_cat"]

    # STEP B decades
    dec_info = compute_decade_weights(seed_draws_dec, target_date, state["Delta_decade"])
    dec_main_count = dec_info["dec_main_count"]
    dec_w_window = dec_info["dec_w_window"]
    dec_w_final = dec_info["dec_w_final"]
    dec_w_recent = dec_info["dec_w_recent"]
    decade_weight_log = dec_info["decade_weight_log"]

    # STEP C hot/warm/cold
    hwhc = classify_hot_warm_cold(f_main, L, seed_draws_cat)
    Hot_set = hwhc["Hot_set"]
    Warm_set = hwhc["Warm_set"]
    Cold_set = hwhc["Cold_set"]
    K = hwhc["K"]
    avg_hot = hwhc["avg_hot"]
    avg_warm = hwhc["avg_warm"]
    avg_cold = hwhc["avg_cold"]
    bias = hwhc["bias"]

    # STEP D category weights
    hot_w, warm_w, cold_w = compute_category_weights(avg_hot, avg_warm, avg_cold, bias, state)

    # STEP E per-number log scores
    P_main, log_score = compute_per_number_log_scores(
        seed_numbers_main_cat,
        seed_numbers_supp_cat,
        f_main, f_supp, L,
        Hot_set, Warm_set, Cold_set,
        hot_w, warm_w, cold_w,
        decade_weight_log,
        cat_dates,
    )

    # STEP E-EXTRA: extra ball
    P_extra, predicted_extra = compute_extra_probabilities(seed_draws_cat, info_cat)

    # STEP F clusters
    draw_sets_for_clusters = build_draw_sets_for_clusters(seed_draws_cat)
    clusters = find_clusters_for_draws(draw_sets_for_clusters)
    cluster_priorities = compute_cluster_priorities(clusters, state["cluster_scores"])

    # STEP G composition
    draw_size = draw_size_for_lottery(lottery)
    h_target, w_target, c_target = compute_composition_targets(
        draw_size, avg_hot, avg_warm, avg_cold,
        Hot_set, Warm_set, Cold_set
    )

    # STEP H Monte Carlo
    mc = monte_carlo_sampling(
        lottery,
        P_main, dec_w_recent,
        Hot_set, Warm_set, Cold_set,
        h_target, w_target, c_target,
        clusters, cluster_priorities
    )

    prob_tuples = mc["prob_tuples"]
    top_tuples = mc["top_tuples"]
    hot_pred = mc["hot_pred"]
    warm_pred = mc["warm_pred"]
    cold_pred = mc["cold_pred"]
    dec_pred = mc["dec_pred"]
    TRIALS = mc["TRIALS"]
    EXPLORE_FRAC = mc["EXPLORE_FRAC"]

    # STEP I diagnostics
    # actual draw for historical targets
    dec_actual = {1: 0, 2: 0, 3: 0, 4: 0}
    actual_draw = get_draw_by_date_and_lottery(target_date, lottery)
    actual_main = actual_draw["main"] if actual_draw else []

    for n in actual_main:
        if NUMBER_MIN <= n <= NUMBER_MAX:
            d = decade(n)
            if d is not None:
                dec_actual[d] += 1

    # top numbers by P
    top_numbers = sorted(NUMBER_RANGE, key=lambda n: P_main[n], reverse=True)

    # annotate top tuples with H/W/C & decades composition
    annotated_top = []
    for T, p in top_tuples[:20]:
        hc = sum(1 for n in T if n in Hot_set)
        wc = sum(1 for n in T if n in Warm_set)
        cc = sum(1 for n in T if n in Cold_set)
        dec_comp = {1: 0, 2: 0, 3: 0, 4: 0}
        for n in T:
            d = decade(n)
            if d is not None:
                dec_comp[d] += 1
        annotated_top.append({
            "tuple": T,
            "prob": p,
            "hot_count": hc,
            "warm_count": wc,
            "cold_count": cc,
            "decades": dec_comp,
        })

    diagnostics = {
        "date": target_date,
        "lottery": lottery,
        "WINDOW_SIZE_CAT": window_cat,
        "WINDOW_SIZE_DEC": window_dec,
        "cat_dates": cat_dates,
        "dec_dates": dec_dates,
        "M": M,
        "K": K,
        "Hot_set": sorted(Hot_set),
        "Warm_set": sorted(Warm_set),
        "Cold_set": sorted(Cold_set),
        "avg_hot": avg_hot,
        "avg_warm": avg_warm,
        "avg_cold": avg_cold,
        "bias": bias,
        "hot_w": hot_w,
        "warm_w": warm_w,
        "cold_w": cold_w,
        "top_numbers": top_numbers[:15],
        "clusters": clusters,
        "cluster_priorities": cluster_priorities,
        "h_target": h_target,
        "w_target": w_target,
        "c_target": c_target,
        "TRIALS": TRIALS,
        "EXPLORE_FRAC": EXPLORE_FRAC,
        "top_tuples_annotated": annotated_top,
        "P_main": P_main,
        "P_extra": P_extra,
        "predicted_extra": predicted_extra,
        "dec_main_count": dec_main_count,
        "dec_w_window": dec_w_window,
        "dec_w_final": dec_w_final,
        "dec_w_recent": dec_w_recent,
        "dec_pred": dec_pred,
        "dec_actual": dec_actual,
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "f_main": f_main,
    }

    # Learning for historical targets
    if do_learning and not is_prediction and actual_draw is not None:
        apply_learning(state, diagnostics, actual_main)

    return diagnostics


# -------------------------
# CALIBRATION
# -------------------------

def calibrate_window_size_cat():
    best_W = None
    best_MSE = None

    for W_cat in WINDOW_SIZE_CAT_CANDIDATES:
        state = init_learning_state()
        errors = []

        for (d, lottery) in CALIBRATION_TARGETS:
            diag = run_pipeline_for_target(
                d, lottery,
                window_cat=W_cat,
                window_dec=W_cat,  # shared for this phase
                state=state,
                do_learning=True,  # includes decade oscillation and clamp/de-mean
                is_prediction=False,
            )
            if diag is None:
                continue
            if "hot_pred" not in diag:
                continue
            hot_pred = diag["hot_pred"]
            cold_pred = diag["cold_pred"]
            hot_actual = diag.get("hot_actual")
            cold_actual = diag.get("cold_actual")
            if hot_actual is None or cold_actual is None:
                continue
            err = (hot_actual - hot_pred) ** 2 + (cold_actual - cold_pred) ** 2
            errors.append(err)

        if not errors:
            continue
        mse = sum(errors) / len(errors)
        if best_MSE is None or mse < best_MSE or (mse == best_MSE and (best_W is None or W_cat < best_W)):
            best_MSE = mse
            best_W = W_cat

    return best_W, best_MSE


def calibrate_window_size_dec(window_cat_star):
    best_W = None
    best_score = None

    for W_dec in WINDOW_SIZE_DEC_CANDIDATES:
        state = init_learning_state()
        all_dec_errors = []
        z_values = []

        for (d, lottery) in CALIBRATION_TARGETS:
            diag = run_pipeline_for_target(
                d, lottery,
                window_cat=window_cat_star,
                window_dec=W_dec,
                state=state,
                do_learning=True,
                is_prediction=False,
            )
            if diag is None:
                continue
            dec_pred = diag["dec_pred"]
            dec_actual = diag["dec_actual"]
            if dec_pred is None or dec_actual is None:
                continue
            # per-decade errors
            for dd in range(1, 5):
                err = (dec_actual[dd] - dec_pred[dd]) ** 2
                all_dec_errors.append(err)
            # low-high balance
            low_pred = dec_pred[1] + dec_pred[2]
            high_pred = dec_pred[3] + dec_pred[4]
            z_values.append(low_pred - high_pred)

        if not all_dec_errors:
            continue
        decade_MSE = sum(all_dec_errors) / len(all_dec_errors)
        if len(z_values) > 1:
            stability_penalty = statistics.pvariance(z_values)
        else:
            stability_penalty = 0.0

        beta = 1.0
        gamma = 0.5
        score_dec = beta * decade_MSE + gamma * stability_penalty

        if best_score is None or score_dec < best_score or (score_dec == best_score and (best_W is None or W_dec < best_W)):
            best_score = score_dec
            best_W = W_dec

    return best_W, best_score


# -------------------------
# MAIN RUN + PREDICTION
# -------------------------

def run_main_and_prediction(window_cat_star, window_dec_star):
    # Main run
    state = init_learning_state()
    main_diagnostics = []

    for (d, lottery) in MAIN_RUN_TARGETS:
        diag = run_pipeline_for_target(
            d, lottery,
            window_cat=window_cat_star,
            window_dec=window_dec_star,
            state=state,
            do_learning=True,
            is_prediction=False,
        )
        if diag is not None:
            main_diagnostics.append(diag)

    # Prediction
    pred_diag = run_pipeline_for_target(
        PREDICTION_DATE,
        PREDICTION_LOTTERY,
        window_cat=window_cat_star,
        window_dec=window_dec_star,
        state=state,
        do_learning=False,
        is_prediction=True,
    )

    return state, main_diagnostics, pred_diag


# -------------------------
# ENTRY POINT
# -------------------------

def main():
    print("=== Calibration: WINDOW_SIZE_CAT* ===")
    W_cat_star, mse_cat = calibrate_window_size_cat()
    print(f"Chosen WINDOW_SIZE_CAT* = {W_cat_star} (category_MSE = {mse_cat})")

    print("\n=== Calibration: WINDOW_SIZE_DEC* ===")
    W_dec_star, score_dec = calibrate_window_size_dec(W_cat_star)
    print(f"Chosen WINDOW_SIZE_DEC* = {W_dec_star} (score_dec = {score_dec})")

    print("\n=== Main run + Prediction ===")
    state, main_diags, pred_diag = run_main_and_prediction(W_cat_star, W_dec_star)

    print("\nFinal learning state:")
    print("  Δ_hot  =", state["Delta_hot"])
    print("  Δ_warm =", state["Delta_warm"])
    print("  Δ_cold =", state["Delta_cold"])
    print("  Δ_decade =", state["Delta_decade"])
    print("  last_dom_decade =", state["last_dom_decade"])

    print("\nPrediction diagnostics for Powerball Thu 27 Nov 2025:")
    if pred_diag is None:
        print("  Prediction aborted: infeasible windows (missing dates).")
        return

    print(f"  WINDOW_SIZE_CAT* = {pred_diag['WINDOW_SIZE_CAT']}")
    print(f"  WINDOW_SIZE_DEC* = {pred_diag['WINDOW_SIZE_DEC']}")
    print(f"  TRIALS = {pred_diag['TRIALS']}")
    print(f"  EXPLORE_FRAC = {pred_diag['EXPLORE_FRAC']:.3f}")
    print(f"  hot_w, warm_w, cold_w = {pred_diag['hot_w']:.3f}, {pred_diag['warm_w']:.3f}, {pred_diag['cold_w']:.3f}")
    print(f"  avg_hot, avg_warm, avg_cold = {pred_diag['avg_hot']:.3f}, {pred_diag['avg_warm']:.3f}, {pred_diag['avg_cold']:.3f}")
    print(f"  bias = {pred_diag['bias']}")

    print("\n  Top-20 predicted main tuples (7 numbers) with probabilities and H/W/C + decades:")
    for item in pred_diag["top_tuples_annotated"]:
        T = item["tuple"]
        p = item["prob"]
        hc = item["hot_count"]
        wc = item["warm_count"]
        cc = item["cold_count"]
        dec_comp = item["decades"]
        print(f"    {T}  prob={p:.6f}  H/W/C=({hc},{wc},{cc})  decades={dec_comp}")

    print("\n  Extra-ball probabilities (1..20):")
    top_extras = sorted(pred_diag["P_extra"].items(), key=lambda kv: kv[1], reverse=True)[:20]
    for x, px in top_extras:
        print(f"    extra {x}: {px:.4f}")
    print(f"\n  predicted_extra (deterministic argmax) = {pred_diag['predicted_extra']}")

    print("\n  Decade diagnostics (prediction):")
    print("    dec_main_count =", pred_diag["dec_main_count"])
    print("    dec_w_window   =", {k: round(v, 3) for k, v in pred_diag["dec_w_window"].items()})
    print("    dec_w_final    =", {k: round(v, 3) for k, v in pred_diag["dec_w_final"].items()})
    print("    dec_w_recent   =", {k: round(v, 3) for k, v in pred_diag["dec_w_recent"].items()})
    print("    dec_pred       =", {k: round(v, 3) for k, v in pred_diag["dec_pred"].items()})

    print("\nDone.")


if __name__ == "__main__":
    main()
