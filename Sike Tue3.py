#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siko 3 — Tuesday OZ Lotto Monte Carlo predictor (strict single-file implementation)

Implements the algorithm spec from:
"Siko 3 Tue fix near miss window size"

Major features:
- Calibration of WINDOW_SIZE_CAT* and WINDOW_SIZE_DEC*
- H/W/C classification and category weights
- Decade weights with learning
- Last-3-days effects, adjacency, centre vs neighbour
- Cross-lottery hop
- Cold resurgence (excluding hot numbers)
- Cluster detection and cluster learning feedback
- Monte Carlo with H/W/C category quotas
- EXPLORE_FRAC driven by avg_hot, avg_warm, avg_cold
- Category + decade learning + oscillation regulariser
- Centre-biased ranking for final prediction

Prediction target: OZ Lotto, Tue 25 Nov 2025, 7 main numbers from 1..47
"""

import math
import random
import datetime
from collections import Counter, defaultdict
import itertools

# ------------------------------
# A) Core numeric config
# ------------------------------

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX = 47
NUMBER_RANGE = list(range(MAIN_NUMBER_MIN, MAIN_NUMBER_MAX + 1))

LOG_SCORE_MAX = 4.0  # Cap for per-number log_score

base_learning_rate_decade = 0.10  # decade learning rate

gamma_hop = 0.30      # cross-lottery hop strength
theta_resurge = 0.25  # cold resurgence strength

# ------------------------------
# B) Decade bands
# ------------------------------

DECADE_BANDS = [
    (1, 1, 9),
    (2, 10, 19),
    (3, 20, 29),
    (4, 30, 39),
    (5, 40, 47),
]

DECADES = {d_id for (d_id, start, end) in DECADE_BANDS}
N_DECADES = len(DECADES)


def decade(n: int):
    """Return decade id for n, or None if out of range."""
    if not (MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX):
        return None
    for d_id, start, end in DECADE_BANDS:
        if start <= n <= end:
            return d_id
    return None


# ------------------------------
# C) Lottery types and sizes
# ------------------------------

LOTTERIES = {
    "Set for Life":     dict(main_draw_size=7, uses_supp=True,  uses_powerball=False),
    "Weekday Windfall": dict(main_draw_size=6, uses_supp=True,  uses_powerball=False),
    "OZ Lotto":         dict(main_draw_size=7, uses_supp=True,  uses_powerball=False),
    "Powerball":        dict(main_draw_size=7, uses_supp=False, uses_powerball=True),
    "Saturday Lotto":   dict(main_draw_size=6, uses_supp=True,  uses_powerball=False),
}


def main_draw_size(lottery_name: str) -> int:
    return LOTTERIES[lottery_name]["main_draw_size"]


# ------------------------------
# D) Cross-lottery hop roles
# ------------------------------

HOP_SOURCE_LOTTERY = "Set for Life"
HOP_DESTINATION_LOTTERY = "OZ Lotto"


# ------------------------------
# Helper utilities
# ------------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def parse_date(s: str) -> datetime.date:
    # Example: "Mon, 24 Nov 2025"
    return datetime.datetime.strptime(s, "%a, %d %b %Y").date()


# ------------------------------
# E) Historical input
# ------------------------------

def make_draw(lottery, date_str, mains, supp=None, powerball=None):
    return {
        "lottery": lottery,
        "date": parse_date(date_str),
        "main": list(mains),
        "supp": list(supp or []),
        "powerball": list(powerball or []),
    }


GLOBAL_DRAWS = []

# --- Set for Life ---
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Mon, 1 Dec 2025', [18, 1, 10, 41, 24, 11, 3], [25, 2], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sun, 30 Nov 2025', [7, 44, 18, 27, 32, 22, 11], [38, 9], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sat, 29 Nov 2025', [8, 31, 4, 6, 42, 16, 14], [13, 19], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Fri, 28 Nov 2025', [15, 27, 8, 39, 5, 43, 20], [19, 29], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Thu, 27 Nov 2025', [12, 36, 6, 7, 37, 41, 29], [8, 43], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Wed, 26 Nov 2025', [29, 37, 34, 14, 5, 21, 20], [18, 19], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Tue, 25 Nov 2025', [26, 16, 23, 15, 31, 1, 27], [8, 41], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Mon, 24 Nov 2025', [41, 1, 17, 29, 14, 40, 22], [35, 31], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sun, 23 Nov 2025', [25, 27, 42, 18, 26, 9, 33], [22, 19], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sat, 22 Nov 2025', [24, 23, 31, 30, 26, 5, 17], [6, 27], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Fri, 21 Nov 2025', [27, 32, 10, 42, 38, 33, 17], [19, 39], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Thu, 20 Nov 2025', [28, 10, 11, 35, 34, 41, 23], [30, 26], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Wed, 19 Nov 2025', [4, 44, 5, 33, 21, 30, 39], [9, 18], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Tue, 18 Nov 2025', [33, 35, 44, 32, 20, 29, 39], [5, 41], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Mon, 17 Nov 2025', [15, 23, 40, 43, 28, 1, 37], [18, 34], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sun, 16 Nov 2025', [8, 19, 21, 27, 40, 14, 7], [20, 44], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sat, 15 Nov 2025', [13, 4, 27, 14, 2, 5, 42], [33, 39], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Fri, 14 Nov 2025', [7, 25, 23, 35, 13, 18, 6], [3, 39], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Thu, 13 Nov 2025', [25, 24, 3, 21, 5, 33, 36], [22, 11], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Wed, 12 Nov 2025', [15, 20, 29, 21, 5, 10, 6], [32, 17], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Tue, 11 Nov 2025', [4, 7, 10, 44, 32, 30, 26], [5, 18], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Mon, 10 Nov 2025', [5, 36, 13, 23, 39, 3, 9], [35, 6], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sun, 9 Nov 2025', [11, 4, 44, 26, 6, 31, 40], [21, 33], None))
GLOBAL_DRAWS.append(make_draw('Set for Life', 'Sat, 8 Nov 2025', [7, 31, 5, 37, 43, 38, 2], [42, 10], None))

# --- Weekday Windfall ---
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Mon, 1 Dec 2025', [8, 6, 30, 38, 36, 1], [43, 5], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Fri, 28 Nov 2025', [30, 8, 25, 43, 39, 24], [21, 1], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Wed, 26 Nov 2025', [44, 43, 8, 36, 16, 27], [31, 30], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Mon, 24 Nov 2025', [44, 15, 20, 17, 4, 18], [7, 11], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Fri, 21 Nov 2025', [4, 5, 26, 10, 40, 20], [14, 24], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Wed, 19 Nov 2025', [43, 26, 35, 25, 42, 13], [24, 5], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Mon, 17 Nov 2025', [37, 11, 4, 2, 5, 7], [30, 22], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Fri, 14 Nov 2025', [34, 11, 28, 15, 44, 31], [9, 20], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Wed, 12 Nov 2025', [35, 11, 33, 15, 34, 45], [8, 37], None))
GLOBAL_DRAWS.append(make_draw('Weekday Windfall', 'Mon, 10 Nov 2025', [38, 3, 31, 22, 28, 5], [26, 14], None))

# --- OZ Lotto ---
GLOBAL_DRAWS.append(make_draw('OZ Lotto', 'Tue, 25 Nov 2025', [12, 43, 28, 1, 47, 35, 14], [15, 16, 46], None))
GLOBAL_DRAWS.append(make_draw('OZ Lotto', 'Tue, 18 Nov 2025', [39, 2, 22, 8, 27, 6, 4], [47, 5, 24], None))
GLOBAL_DRAWS.append(make_draw('OZ Lotto', 'Tue, 11 Nov 2025', [44, 30, 7, 28, 17, 34, 42], [20, 32, 3], None))

# --- Powerball ---
GLOBAL_DRAWS.append(make_draw('Powerball', 'Thu, 27 Nov 2025', [2, 17, 11, 9, 19, 28, 24], None, [1]))
GLOBAL_DRAWS.append(make_draw('Powerball', 'Thu, 20 Nov 2025', [19, 11, 12, 4, 29, 13, 27], None, [20]))
GLOBAL_DRAWS.append(make_draw('Powerball', 'Thu, 13 Nov 2025', [22, 10, 6, 15, 2, 8, 7], None, [13]))

# --- Saturday Lotto ---
GLOBAL_DRAWS.append(make_draw('Saturday Lotto', 'Sat, 29 Nov 2025', [22, 10, 17, 5, 44, 36], [3, 11], None))
GLOBAL_DRAWS.append(make_draw('Saturday Lotto', 'Sat, 22 Nov 2025', [7, 31, 15, 39, 42, 12], [5, 8], None))
GLOBAL_DRAWS.append(make_draw('Saturday Lotto', 'Sat, 15 Nov 2025', [36, 19, 33, 41, 39, 1], [25, 20], None))
GLOBAL_DRAWS.append(make_draw('Saturday Lotto', 'Sat, 8 Nov 2025', [28, 13, 1, 41, 14, 16], [39, 34], None))

GLOBAL_DRAWS.sort(key=lambda d: d["date"])

# ------------------------------
# F) Target draws & prediction target
# ------------------------------

TARGET_DRAWS_FOR_LEARNING = [
    ("OZ Lotto",         parse_date("Tue, 25 Nov 2025")),
    ("Weekday Windfall", parse_date("Wed, 26 Nov 2025")),
    ("Powerball",        parse_date("Thu, 27 Nov 2025")),
    ("Weekday Windfall", parse_date("Fri, 28 Nov 2025")),
    ("Saturday Lotto",   parse_date("Sat, 29 Nov 2025")),
    ("Weekday Windfall", parse_date("Mon, 1 Dec 2025")),
]

PREDICTION_TARGET = (
    "OZ Lotto",
    parse_date("Tue, 2 Dec 2025"),
    7,
)

# ------------------------------
# G) Candidate window sizes
# ------------------------------

WINDOW_SIZE_CANDIDATES = [6, 7, 8, 9, 10]

# ------------------------------
# Global learning state
# ------------------------------

delta_hot = 0.0
delta_warm = 0.0
delta_cold = 0.0
delta_decade = {d: 0.0 for d in DECADES}
last_dom_decade = None
cluster_priority_score_global = {}  # cluster -> score offset


def reset_learning_state():
    global delta_hot, delta_warm, delta_cold, delta_decade, last_dom_decade, cluster_priority_score_global
    delta_hot = 0.0
    delta_warm = 0.0
    delta_cold = 0.0
    delta_decade = {d: 0.0 for d in DECADES}
    last_dom_decade = None
    cluster_priority_score_global = {}


# ------------------------------
# Helpers on GLOBAL_DRAWS
# ------------------------------

def get_draw(lottery_name, date_obj):
    for d in GLOBAL_DRAWS:
        if d["lottery"] == lottery_name and d["date"] == date_obj:
            return d
    return None


# ------------------------------
# STEP A — windows
# ------------------------------

def build_windows(D_t, W_cat, W_dec):
    seed_dates_cat = [D_t - datetime.timedelta(days=i) for i in range(1, W_cat + 1)]
    seed_dates_cat = sorted(seed_dates_cat)
    seed_dates_dec = [D_t - datetime.timedelta(days=i) for i in range(1, W_dec + 1)]
    seed_dates_dec = sorted(seed_dates_dec)

    seed_draws_cat = []
    seed_draws_dec = []

    for d in GLOBAL_DRAWS:
        if d["date"] in seed_dates_cat:
            seed_draws_cat.append(d)
        if d["date"] in seed_dates_dec:
            seed_draws_dec.append(d)

    if not seed_draws_cat or not seed_draws_dec:
        return None
    return seed_dates_cat, seed_dates_dec, seed_draws_cat, seed_draws_dec


# ------------------------------
# STEP B — category & decade freqs
# ------------------------------

def step_B_category(seed_draws_cat):
    seed_numbers_main = []
    seed_numbers_supp = []
    lotto_presence = {n: set() for n in NUMBER_RANGE}
    last_main_date = {n: None for n in NUMBER_RANGE}

    for d in seed_draws_cat:
        lot = d["lottery"]
        dt = d["date"]
        for n in d["main"]:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                seed_numbers_main.append(n)
                lotto_presence[n].add(lot)
                last_main_date[n] = max(last_main_date[n], dt) if last_main_date[n] else dt
        for n in (d["supp"] + d["powerball"]):
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                seed_numbers_supp.append(n)
                lotto_presence[n].add(lot)

    f_main = Counter(seed_numbers_main)
    f_supp = Counter(seed_numbers_supp)
    L = {n: len(lotto_presence[n]) for n in NUMBER_RANGE}
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)

    return f_main, f_supp, L, last_main_date, M


def step_B_decade(seed_draws_dec):
    dec_seed_numbers_main = []
    for d in seed_draws_dec:
        for n in d["main"]:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                dec_seed_numbers_main.append(n)

    dec_main_count = {d: 0 for d in DECADES}
    for n in dec_seed_numbers_main:
        d_id = decade(n)
        if d_id is not None:
            dec_main_count[d_id] += 1

    total_main_in_range_dec = sum(dec_main_count.values())
    if total_main_in_range_dec == 0:
        dec_w_window = {d: 1.0 for d in DECADES}
        return dec_main_count, total_main_in_range_dec, dec_w_window

    dec_freq = {d: dec_main_count[d] / total_main_in_range_dec for d in DECADES}
    max_dec_freq = max(dec_freq.values()) if dec_freq else 1.0
    if max_dec_freq == 0:
        max_dec_freq = 1.0

    dec_w_window = {}
    for d in DECADES:
        if dec_main_count[d] == 0:
            dec_w_window[d] = 0.20
        else:
            dec_w_raw = 0.70 + 0.90 * (dec_freq[d] / max_dec_freq)
            dec_w_tmp = 0.75 * dec_w_raw + 0.25 * 1.0
            dec_w_window[d] = clamp(dec_w_tmp, 0.50, 1.70)

    return dec_main_count, total_main_in_range_dec, dec_w_window


def apply_global_decade_learning(dec_w_window):
    dec_w_final = {d: dec_w_window[d] * math.exp(delta_decade[d]) for d in DECADES}
    avg_dec_w = sum(dec_w_final.values()) / N_DECADES if N_DECADES > 0 else 1.0
    if avg_dec_w > 0:
        for d in DECADES:
            dec_w_final[d] /= avg_dec_w
    return dec_w_final


def last_3_days_decade_momentum(D_t, W_dec, dec_main_count, total_main_in_range_dec, dec_w_final):
    recent_start_dec = D_t - datetime.timedelta(days=3)
    recent_dates_dec = [recent_start_dec + datetime.timedelta(days=i) for i in range(3)]
    dec_recent_count = {d: 0 for d in DECADES}

    for d in GLOBAL_DRAWS:
        if d["date"] in recent_dates_dec:
            for n in d["main"]:
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    dd = decade(n)
                    if dd is not None:
                        dec_recent_count[dd] += 1

    total_recent = sum(dec_recent_count.values())
    if total_recent == 0:
        rec_w = {d: 1.0 for d in DECADES}
    else:
        dec_recent_freq = {d: dec_recent_count[d] / total_recent for d in DECADES}
        dec_freq_window = {
            d: dec_main_count[d] / max(total_main_in_range_dec, 1)
            for d in DECADES
        }
        eps = 1e-6
        rec_w_raw = {}
        rec_w = {}
        for d in DECADES:
            ratio = dec_recent_freq[d] / (dec_freq_window[d] + eps)
            rec_w_raw[d] = clamp(ratio, 0.8, 1.3)
            rec_w[d] = 0.7 * rec_w_raw[d] + 0.3 * 1.0

    dec_w_recent = {d: dec_w_final[d] * rec_w[d] for d in DECADES}
    avg_dec_recent = sum(dec_w_recent.values()) / N_DECADES if N_DECADES > 0 else 1.0
    if avg_dec_recent > 0:
        for d in DECADES:
            dec_w_recent[d] /= avg_dec_recent

    decade_weight_log = {}
    for n in NUMBER_RANGE:
        dd = decade(n)
        if dd is not None:
            decade_weight_log[n] = math.log(dec_w_recent[dd])
        else:
            decade_weight_log[n] = 0.0

    return decade_weight_log


# ------------------------------
# 4E) CROSS-LOTTERY HOP METRICS
# ------------------------------

def cross_lottery_hop(seed_draws_cat):
    appearances = {n: [] for n in NUMBER_RANGE}
    for d in seed_draws_cat:
        dt = d["date"]
        lot = d["lottery"]
        for n in d["main"]:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                appearances[n].append((dt, lot))

    cross_hop_score = {n: 0.0 for n in NUMBER_RANGE}

    for n in NUMBER_RANGE:
        sfl_count = 0
        non_sfl_count = 0
        app = appearances[n]
        for _, lot in app:
            if lot == HOP_SOURCE_LOTTERY:
                sfl_count += 1
            else:
                non_sfl_count += 1

        cross_pair_sum = 0.0
        for (d1, L1), (d2, L2) in itertools.combinations(app, 2):
            if d2 < d1:
                d1, d2 = d2, d1
                L1, L2 = L2, L1
            lag_days = (d2 - d1).days
            if lag_days <= 0:
                continue
            base_pair = 1.0 / lag_days
            w_dir = 1.0
            if L1 == HOP_SOURCE_LOTTERY and L2 != HOP_SOURCE_LOTTERY:
                w_dir *= 1.5
            if L2 == HOP_DESTINATION_LOTTERY and L1 != HOP_DESTINATION_LOTTERY:
                w_dir *= 1.3
            cross_pair_sum += base_pair * w_dir

        base_hop_score = sfl_count * non_sfl_count + cross_pair_sum
        cross_hop_score[n] = base_hop_score

    max_hop_score = max(cross_hop_score.values()) if cross_hop_score else 0.0
    if max_hop_score <= 0:
        cross_hop_log = {n: 0.0 for n in NUMBER_RANGE}
    else:
        cross_hop_log = {}
        for n in NUMBER_RANGE:
            score = cross_hop_score[n] / max_hop_score
            cross_hop_log[n] = math.log(1.0 + gamma_hop * score) if score > 0 else 0.0

    return cross_hop_log


# ------------------------------
# STEP C — HOT / WARM / COLD
# ------------------------------

def step_C_hw_cold(f_main, L):
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)
    K = max(1, round(max(3, M * 0.15)))

    all_nums = list(NUMBER_RANGE)
    all_nums.sort(key=lambda n: (-f_main[n], -L[n], n))
    hot_set = set(all_nums[:K])

    warm_set = {n for n in NUMBER_RANGE if 1 <= f_main[n] <= 2} - hot_set
    cold_set = set(NUMBER_RANGE) - hot_set - warm_set

    return hot_set, warm_set, cold_set, M, K


# ------------------------------
# STEP D — category weights
# ------------------------------

def step_D_category_weights(seed_draws_cat, hot_set, warm_set, cold_set):
    p_hot_list = []
    p_warm_list = []
    p_cold_list = []

    for d in seed_draws_cat:
        main_nums = d["main"]
        size = len(main_nums)
        if size == 0:
            continue
        hot_cnt = sum(1 for n in main_nums if n in hot_set)
        warm_cnt = sum(1 for n in main_nums if n in warm_set)
        cold_cnt = sum(1 for n in main_nums if n in cold_set)
        p_hot_list.append(hot_cnt / size)
        p_warm_list.append(warm_cnt / size)
        p_cold_list.append(cold_cnt / size)

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    avg_hot = avg(p_hot_list)
    avg_warm = avg(p_warm_list)
    avg_cold = avg(p_cold_list)

    if avg_warm >= avg_hot and avg_warm >= avg_cold:
        bias = "warm-heavy"
    elif avg_hot > avg_warm and avg_hot > avg_cold:
        bias = "hot-heavy"
    elif avg_cold > avg_hot and avg_cold > avg_warm:
        bias = "cold-heavy"
    else:
        bias = "balanced"

    if bias == "hot-heavy":
        base_hot, base_warm, base_cold = (1.4, 1.15, 0.6)
    elif bias == "warm-heavy":
        base_hot, base_warm, base_cold = (0.95, 1.35, 1.05)
    elif bias == "cold-heavy":
        base_hot, base_warm, base_cold = (0.8, 1.0, 1.4)
    else:
        base_hot, base_warm, base_cold = (1.0, 1.0, 0.95)

    hot_w = base_hot * (1 + (avg_hot - 1/3) * 0.25)
    warm_w = base_warm * (1 + (avg_warm - 1/3) * 0.25)
    cold_w = base_cold * (1 + (avg_cold - 1/3) * 0.25)

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    s = hot_w + warm_w + cold_w
    if s > 0:
        hot_w /= s / 3.0
        warm_w /= s / 3.0
        cold_w /= s / 3.0

    hot_w = 0.75 * hot_w + 0.25 * 1.0
    warm_w = 0.75 * warm_w + 0.25 * 1.0
    cold_w = 0.75 * cold_w + 0.25 * 1.0

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    return avg_hot, avg_warm, avg_cold, bias, hot_w, warm_w, cold_w


# ------------------------------
# STEP E — per-number log scores
# ------------------------------

def step_E_logs(D_t, W_cat, seed_dates_cat, seed_draws_cat,
                f_main, f_supp, L, last_main_date,
                hot_set, warm_set, cold_set,
                decade_weight_log, cross_hop_log):

    # adjacency
    seed_numbers_main = []
    for d in seed_draws_cat:
        seed_numbers_main.extend([n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX])

    main_set_all = set(seed_numbers_main)
    adj_count = {n: 0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        if (n - 1 in main_set_all) or (n + 1 in main_set_all):
            adj_count[n] = sum(1 for x in seed_numbers_main if x in (n-1, n+1))
    max_adj = max(adj_count.values()) if adj_count else 1
    max_adj = max(max_adj, 1)

    adj_log = {}
    for n in NUMBER_RANGE:
        adj_score_raw = 0.05 + 0.25 * (adj_count[n] / max_adj)
        adj_log[n] = math.log(1 + adj_score_raw)

    # last-3-days (main / supp / powerball) in category window
    last3_dates = [D_t - datetime.timedelta(days=i) for i in range(1, 4)]
    last3_draws = [d for d in GLOBAL_DRAWS if d["date"] in last3_dates]

    hit_main_last3 = {n: 0 for n in NUMBER_RANGE}
    hit_supp_last3 = {n: 0 for n in NUMBER_RANGE}
    for d in last3_draws:
        for n in d["main"]:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                hit_main_last3[n] += 1
        for n in (d["supp"] + d["powerball"]):
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                hit_supp_last3[n] += 1

    delta_log = {}
    for n in NUMBER_RANGE:
        if hit_main_last3[n] > 0:
            mult = 1.4
        elif hit_supp_last3[n] > 0:
            mult = 1.2
        else:
            mult = 1.0
        delta_log[n] = math.log(mult)

    # centre vs neighbour recency
    neigh_hits_last3 = {n: 0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        for d in last3_draws:
            for x in d["main"]:
                if x in (n-1, n+1) and MAIN_NUMBER_MIN <= x <= MAIN_NUMBER_MAX:
                    neigh_hits_last3[n] += 1

    centre_raw = {}
    for n in NUMBER_RANGE:
        centre_raw[n] = hit_main_last3[n] - 0.7 * neigh_hits_last3[n]

    centre_min = min(centre_raw.values())
    centre_max = max(centre_raw.values())
    centre_score = {}
    if centre_max == centre_min:
        for n in NUMBER_RANGE:
            centre_score[n] = 0.5
    else:
        for n in NUMBER_RANGE:
            centre_score[n] = (centre_raw[n] - centre_min) / (centre_max - centre_min)

    LAMBDA_CENTRE = 0.35
    centre_log = {}
    for n in NUMBER_RANGE:
        mult = 1.0 + LAMBDA_CENTRE * (centre_score[n] - 0.5)
        mult = clamp(mult, 0.75, 1.25)
        centre_log[n] = math.log(mult)

    # cross-lottery density
    cross_log = {n: math.log(1 + 0.08 * L[n]) for n in NUMBER_RANGE}

    # supp-only bonus
    supp_log = {}
    for n in NUMBER_RANGE:
        if f_main[n] == 0 and f_supp[n] > 0:
            supp_log[n] = math.log(1.05)
        else:
            supp_log[n] = 0.0

    # cold resurgence (exclude hot_set)
    cold_resurge_raw = {}
    for n in NUMBER_RANGE:
        if last_main_date[n] is None:
            cold_resurge_raw[n] = 0.0
        else:
            gap_days = (D_t - last_main_date[n]).days
            if 4 <= gap_days <= W_cat:
                cold_resurge_raw[n] = 1.0 / gap_days
            else:
                cold_resurge_raw[n] = 0.0
    max_resurge = max(cold_resurge_raw.values()) if cold_resurge_raw else 0.0
    cold_resurge_score = {}
    cold_resurge_log = {}
    if max_resurge <= 0:
        for n in NUMBER_RANGE:
            cold_resurge_score[n] = 0.0
            cold_resurge_log[n] = 0.0
    else:
        for n in NUMBER_RANGE:
            s = cold_resurge_raw[n] / max_resurge
            cold_resurge_score[n] = s
            if n in hot_set:
                cold_resurge_log[n] = 0.0
            else:
                mult = 1.0 + theta_resurge * s
                cold_resurge_log[n] = math.log(mult)

    return (adj_log, delta_log, centre_log, centre_score,
            cross_log, supp_log, cold_resurge_log)


# ------------------------------
# STEP F — clusters
# ------------------------------

def step_F_clusters(seed_draws_cat):
    counts = Counter()
    for d in seed_draws_cat:
        nums = sorted(set(n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX))
        for size in (2, 3, 4):
            if len(nums) >= size:
                for combo in itertools.combinations(nums, size):
                    counts[combo] += 1

    clusters = {C: freq for C, freq in counts.items() if freq >= 2}

    cluster_priority = {}
    for C, freq in clusters.items():
        base = 1 + 0.2 * (freq - 1)
        base *= (1 + cluster_priority_score_global.get(C, 0.0))
        cluster_priority[C] = base

    return clusters, cluster_priority


# ------------------------------
# STEP G — composition targets
# ------------------------------

def step_G_targets(draw_size, avg_hot, avg_warm, avg_cold,
                   hot_set, warm_set, cold_set):

    h_target = round(draw_size * avg_hot)
    w_target = round(draw_size * avg_warm)
    c_target = draw_size - h_target - w_target

    h_target = max(0, min(h_target, len(hot_set)))
    w_target = max(0, min(w_target, len(warm_set)))
    c_target = max(0, min(c_target, len(cold_set)))

    total = h_target + w_target + c_target
    while total < draw_size:
        caps = []
        if len(hot_set) > h_target:
            caps.append(("H", len(hot_set) - h_target))
        if len(warm_set) > w_target:
            caps.append(("W", len(warm_set) - w_target))
        if len(cold_set) > c_target:
            caps.append(("C", len(cold_set) - c_target))
        if not caps:
            break
        caps.sort(key=lambda x: (-x[1], {"W": 0, "H": 1, "C": 2}[x[0]]))
        cat = caps[0][0]
        if cat == "H":
            h_target += 1
        elif cat == "W":
            w_target += 1
        else:
            c_target += 1
        total = h_target + w_target + c_target

    return h_target, w_target, c_target


# ------------------------------
# STEP H — Monte Carlo (with quotas)
# ------------------------------

def weighted_sample_without_replacement(candidates, weights_dict, k):
    """Sample k items (or fewer if not enough) from candidates using weights_dict, without replacement."""
    candidates = list(candidates)
    chosen = set()
    if not candidates or k <= 0:
        return chosen

    weights = [max(0.0, weights_dict.get(n, 0.0)) for n in candidates]

    for _ in range(min(k, len(candidates))):
        Z = sum(weights)
        if Z <= 0:
            idx = random.randrange(len(candidates))
        else:
            r = random.random() * Z
            acc = 0.0
            idx = 0
            for i, w in enumerate(weights):
                acc += w
                if r <= acc:
                    idx = i
                    break
        n = candidates[idx]
        chosen.add(n)
        del candidates[idx]
        del weights[idx]
        if not candidates:
            break

    return chosen


def monte_carlo_for_draw(D_t, mode_learning_or_prediction,
                         current_lottery_name, draw_size,
                         seed_dates_cat, seed_draws_cat,
                         f_main, L, hot_set, warm_set, cold_set,
                         hot_w, warm_w, cold_w,
                         avg_hot, avg_warm, avg_cold,
                         decade_weight_log, cross_hop_log,
                         adj_log, delta_log, centre_log, centre_score,
                         cross_log, supp_log, cold_resurge_log,
                         clusters, cluster_priority):

    # category weight log
    category_weight_log = {}
    for n in NUMBER_RANGE:
        if n in hot_set:
            category_weight_log[n] = math.log(hot_w)
        elif n in warm_set:
            category_weight_log[n] = math.log(warm_w)
        else:
            category_weight_log[n] = math.log(cold_w)

    # total log score per number
    log_score = {}
    rawP = {}
    for n in NUMBER_RANGE:
        ls = (
            adj_log[n]
            + delta_log[n]
            + centre_log[n]
            + cross_log[n]
            + supp_log[n]
            + category_weight_log[n]
            + decade_weight_log[n]
            + cross_hop_log[n]
            + cold_resurge_log[n]
        )
        ls = min(ls, LOG_SCORE_MAX)
        log_score[n] = ls
        rawP[n] = math.exp(ls)

    Z = sum(rawP.values())
    if Z <= 0:
        P = {n: 1.0 / len(NUMBER_RANGE) for n in NUMBER_RANGE}
    else:
        P = {n: rawP[n] / Z for n in NUMBER_RANGE}

    # cluster effect
    CLUSTER_LAMBDA = 0.15

    # category quotas (Step G)
    h_target, w_target, c_target = step_G_targets(
        draw_size, avg_hot, avg_warm, avg_cold,
        hot_set, warm_set, cold_set
    )

    # trial count
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)
    complexity = math.sqrt(max(1, M / 20.0))
    if mode_learning_or_prediction == "learning":
        TRIALS = int(clamp(40000 * complexity, 15000, 60000))
    else:
        TRIALS = int(clamp(150000 * complexity, 60000, 200000))

    # EXPLORE_FRAC from avg_hot/warm/cold
    max_avg = max(avg_hot, avg_warm, avg_cold)
    EXPLORE_FRAC = clamp(0.10 + 0.20 * max(0.0, max_avg - 1/3), 0.10, 0.20)
    COLD_EXPLORE_MULT = 1.25

    baseP = P.copy()
    hot_list = list(hot_set)
    warm_list = list(warm_set)
    cold_list = list(cold_set)

    freq = Counter()
    hot_count_sum = 0.0
    warm_count_sum = 0.0
    cold_count_sum = 0.0
    dec_count_sum = {d: 0.0 for d in DECADES}

    for _ in range(TRIALS):
        mode = "explore" if random.random() < EXPLORE_FRAC else "normal"

        W = {}
        for n in NUMBER_RANGE:
            w = baseP[n]
            if mode == "explore" and n in cold_set:
                w *= COLD_EXPLORE_MULT
            C_n = [C for C in clusters if n in C]
            if C_n:
                S_n = sum(cluster_priority[C] - 1.0 for C in C_n)
                w *= math.exp(CLUSTER_LAMBDA * S_n)
            W[n] = w

        chosen = set()
        chosen |= weighted_sample_without_replacement(hot_list, W, h_target)
        chosen |= weighted_sample_without_replacement(warm_list, W, w_target)
        chosen |= weighted_sample_without_replacement(cold_list, W, c_target)

        # if still fewer than draw_size due to small sets, fill from all numbers by global P
        while len(chosen) < draw_size:
            r = random.random()
            acc = 0.0
            for n in NUMBER_RANGE:
                acc += P[n]
                if r <= acc:
                    if n not in chosen:
                        chosen.add(n)
                    break

        T = tuple(sorted(chosen))
        freq[T] += 1

        hot_count_sum += sum(1 for n in T if n in hot_set)
        warm_count_sum += sum(1 for n in T if n in warm_set)
        cold_count_sum += sum(1 for n in T if n in cold_set)
        for n in T:
            d_id = decade(n)
            if d_id is not None:
                dec_count_sum[d_id] += 1

    prob = {T: freq[T] / TRIALS for T in freq}

    hot_pred = hot_count_sum / TRIALS
    warm_pred = warm_count_sum / TRIALS
    cold_pred = cold_count_sum / TRIALS
    dec_pred = {d: dec_count_sum[d] / TRIALS for d in DECADES}

    return prob, TRIALS, hot_pred, warm_pred, cold_pred, dec_pred, centre_score


# ------------------------------
# Diagnostics (compact)
# ------------------------------

def print_diagnostics(label, M, K, hot_set, warm_set, cold_set,
                      avg_hot, avg_warm, avg_cold, bias,
                      hot_w, warm_w, cold_w,
                      TRIALS, hot_pred, warm_pred, cold_pred):
    print(f"\n=== Diagnostics for {label} ===")
    print(f"M={M}, K={K}")
    print(f"Hot/Warm/Cold sizes = {len(hot_set)}/{len(warm_set)}/{len(cold_set)}")
    print(f"avg_hot={avg_hot:.3f}, avg_warm={avg_warm:.3f}, avg_cold={avg_cold:.3f}, bias={bias}")
    print(f"hot_w={hot_w:.3f}, warm_w={warm_w:.3f}, cold_w={cold_w:.3f}")
    print(f"TRIALS={TRIALS}, hot_pred={hot_pred:.3f}, warm_pred={warm_pred:.3f}, cold_pred={cold_pred:.3f}")


# ------------------------------
# Learning feedback (STEP J)
# ------------------------------

def variance_f_main(f_main):
    vals = [f_main[n] for n in NUMBER_RANGE]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return sum((v - mean) ** 2 for v in vals) / len(vals)


def apply_learning(actual_main, hot_set, warm_set, cold_set,
                   f_main, hot_pred, cold_pred, dec_pred,
                   clusters):
    global delta_hot, delta_warm, delta_cold, delta_decade, last_dom_decade, cluster_priority_score_global

    # category learning
    hot_actual = sum(1 for n in actual_main if n in hot_set)
    warm_actual = sum(1 for n in actual_main if n in warm_set)
    cold_actual = sum(1 for n in actual_main if n in cold_set)

    hot_error = hot_actual - hot_pred
    cold_error = cold_actual - cold_pred

    var_f = variance_f_main(f_main)
    learning_rate = clamp(0.02 + 0.02 * var_f, 0.02, 0.10)

    delta_hot_step = clamp(math.copysign(learning_rate * abs(hot_error), hot_error), -0.1, 0.1)
    delta_cold_step = clamp(math.copysign(learning_rate * abs(cold_error), cold_error), -0.1, 0.1)

    delta_hot += delta_hot_step
    delta_cold += delta_cold_step
    delta_warm = - (delta_hot + delta_cold) / 2.0

    delta_hot = clamp(delta_hot, -0.5, 0.5)
    delta_warm = clamp(delta_warm, -0.5, 0.5)
    delta_cold = clamp(delta_cold, -0.5, 0.5)

    # decade learning
    dec_actual = {d: 0 for d in DECADES}
    for n in actual_main:
        d_id = decade(n)
        if d_id is not None:
            dec_actual[d_id] += 1

    for d in DECADES:
        dec_error = dec_actual[d] - dec_pred[d]
        step_d = clamp(base_learning_rate_decade * dec_error, -0.15, 0.15)
        delta_decade[d] += step_d

    # oscillation regulariser
    decade_ids_sorted = [d_id for (d_id, _, _) in DECADE_BANDS]
    N = len(decade_ids_sorted)
    k = N // 2
    LOW = set(decade_ids_sorted[:k])
    HIGH = set(decade_ids_sorted[k:])

    dom_decade = max(DECADES, key=lambda d: (dec_actual[d], -d))
    if last_dom_decade is not None and dom_decade == last_dom_decade:
        if dom_decade in LOW:
            for dh in HIGH:
                delta_decade[dh] += 0.03
            for dl in LOW:
                delta_decade[dl] -= 0.03
        else:
            for dl in LOW:
                delta_decade[dl] += 0.03
            for dh in HIGH:
                delta_decade[dh] -= 0.03

    last_dom_decade = dom_decade

    for d in DECADES:
        delta_decade[d] = clamp(delta_decade[d], -0.8, 0.8)

    mean_delta_dec = sum(delta_decade.values()) / N_DECADES if N_DECADES > 0 else 0.0
    for d in DECADES:
        delta_decade[d] -= mean_delta_dec

    # cluster learning feedback
    actual_set = set(actual_main)
    for C in clusters:
        if set(C).issubset(actual_set):
            delta = 0.05
        else:
            delta = -0.02
        old = cluster_priority_score_global.get(C, 0.0)
        cluster_priority_score_global[C] = clamp(old + delta, -0.5, 0.5)


# ------------------------------
# Single-draw pipeline (for calib & main & prediction)
# ------------------------------

def run_full_steps_for(D_t, lottery_name, W_cat, W_dec, mode_learning_or_prediction):
    draw = get_draw(lottery_name, D_t)
    if draw is None and mode_learning_or_prediction == "learning":
        return None

        # For prediction, we don't have the result yet – create a dummy shell.
    if draw is None and mode_learning_or_prediction == "prediction":
        draw = {
            "lottery": lottery_name,
            "date": D_t,
            "main": [],
            "supp": [],
            "powerball": [],
        }

    windows = build_windows(D_t, W_cat, W_dec)
    if windows is None:
        return None
    seed_dates_cat, seed_dates_dec, seed_draws_cat, seed_draws_dec = windows

    f_main, f_supp, L, last_main_date, M = step_B_category(seed_draws_cat)
    dec_main_count, total_main_dec, dec_w_window = step_B_decade(seed_draws_dec)
    dec_w_final = apply_global_decade_learning(dec_w_window)
    decade_weight_log = last_3_days_decade_momentum(D_t, W_dec, dec_main_count, total_main_dec, dec_w_final)
    cross_hop_log = cross_lottery_hop(seed_draws_cat)

    hot_set, warm_set, cold_set, M, K = step_C_hw_cold(f_main, L)
    avg_hot, avg_warm, avg_cold, bias, hot_w, warm_w, cold_w = step_D_category_weights(
        seed_draws_cat, hot_set, warm_set, cold_set
    )

    (adj_log, delta_log, centre_log, centre_score,
     cross_log, supp_log, cold_resurge_log) = step_E_logs(
        D_t, W_cat, seed_dates_cat, seed_draws_cat,
        f_main, f_supp, L, last_main_date,
        hot_set, warm_set, cold_set,
        decade_weight_log, cross_hop_log
    )

    clusters, cluster_priority = step_F_clusters(seed_draws_cat)

    draw_size = main_draw_size(lottery_name)

    prob, TRIALS, hot_pred, warm_pred, cold_pred, dec_pred, centre_score = monte_carlo_for_draw(
        D_t, mode_learning_or_prediction, lottery_name, draw_size,
        seed_dates_cat, seed_draws_cat,
        f_main, L, hot_set, warm_set, cold_set,
        hot_w, warm_w, cold_w,
        avg_hot, avg_warm, avg_cold,
        decade_weight_log, cross_hop_log,
        adj_log, delta_log, centre_log, centre_score,
        cross_log, supp_log, cold_resurge_log,
        clusters, cluster_priority
    )

    return {
        "draw": draw,
        "f_main": f_main,
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
        "TRIALS": TRIALS,
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "dec_pred": dec_pred,
        "prob": prob,
        "centre_score": centre_score,
        "clusters": clusters,
    }


# ------------------------------
# Calibration 0C-1 and 0C-2
# ------------------------------

def calibrate_window_size_cat():
    results = {}
    for W_cat in WINDOW_SIZE_CANDIDATES:
        reset_learning_state()
        mse_terms = []
        for lottery_name, D_t in TARGET_DRAWS_FOR_LEARNING:
            W_dec = W_cat
            res = run_full_steps_for(D_t, lottery_name, W_cat, W_dec, "learning")
            if res is None:
                continue
            draw = res["draw"]
            actual_main = [n for n in draw["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
            hot_set, warm_set, cold_set = res["hot_set"], res["warm_set"], res["cold_set"]
            hot_pred, cold_pred = res["hot_pred"], res["cold_pred"]

            hot_actual = sum(1 for n in actual_main if n in hot_set)
            cold_actual = sum(1 for n in actual_main if n in cold_set)

            mse = ((hot_actual - hot_pred) ** 2 +
                   (cold_actual - cold_pred) ** 2)
            mse_terms.append(mse)

            apply_learning(actual_main, hot_set, warm_set, cold_set,
                           res["f_main"], hot_pred, cold_pred, res["dec_pred"],
                           res["clusters"])

        category_MSE = sum(mse_terms) / len(mse_terms) if mse_terms else float("inf")
        results[W_cat] = category_MSE
        print(f"Calibration WINDOW_SIZE_CAT W={W_cat}, category_MSE={category_MSE}")

    best_W = min(results, key=lambda w: (results[w], w))
    print(f"\nChosen WINDOW_SIZE_CAT* = {best_W} (category_MSE = {results[best_W]})")
    return best_W, results


def calibrate_window_size_dec(W_cat_star):
    results = {}
    for W_dec in WINDOW_SIZE_CANDIDATES:
        reset_learning_state()
        mse_terms = []
        z_values = []
        for lottery_name, D_t in TARGET_DRAWS_FOR_LEARNING:
            res = run_full_steps_for(D_t, lottery_name, W_cat_star, W_dec, "learning")
            if res is None:
                continue
            draw = res["draw"]
            actual_main = [n for n in draw["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
            hot_set, warm_set, cold_set = res["hot_set"], res["warm_set"], res["cold_set"]
            hot_pred, cold_pred = res["hot_pred"], res["cold_pred"]
            dec_pred = res["dec_pred"]

            dec_actual = {d: 0 for d in DECADES}
            for n in actual_main:
                d_id = decade(n)
                if d_id is not None:
                    dec_actual[d_id] += 1

            for d in DECADES:
                mse_terms.append((dec_actual[d] - dec_pred[d]) ** 2)

            decade_ids_sorted = [d_id for (d_id, _, _) in DECADE_BANDS]
            N = len(decade_ids_sorted)
            k = N // 2
            LOW = set(decade_ids_sorted[:k])
            HIGH = set(decade_ids_sorted[k:])
            low_t = sum(dec_pred[d] for d in LOW)
            high_t = sum(dec_pred[d] for d in HIGH)
            z_t = low_t - high_t
            z_values.append(z_t)

            apply_learning(actual_main, hot_set, warm_set, cold_set,
                           res["f_main"], hot_pred, cold_pred, dec_pred,
                           res["clusters"])

        if mse_terms:
            decade_MSE = sum(mse_terms) / len(mse_terms)
        else:
            decade_MSE = float("inf")
        if z_values:
            mean_z = sum(z_values) / len(z_values)
            var_z = sum((z - mean_z) ** 2 for z in z_values) / len(z_values)
        else:
            var_z = 0.0

        alpha, beta = 1.0, 0.5
        score_dec = alpha * decade_MSE + beta * var_z
        results[W_dec] = score_dec
        print(f"Calibration WINDOW_SIZE_DEC W={W_dec}, score_dec={score_dec}")

    best_W = min(results, key=lambda w: (results[w], w))
    print(f"\nChosen WINDOW_SIZE_DEC* = {best_W} (score_dec = {results[best_W]})")
    return best_W, results


# ------------------------------
# Main run + final prediction
# ------------------------------

def main():
    random.seed(0)

    # Calibration
    W_cat_star, _ = calibrate_window_size_cat()
    W_dec_star, _ = calibrate_window_size_dec(W_cat_star)

    print("\n=== Main run with chosen windows ===")
    reset_learning_state()
    for lottery_name, D_t in TARGET_DRAWS_FOR_LEARNING:
        res = run_full_steps_for(D_t, lottery_name, W_cat_star, W_dec_star, "learning")
        if res is None:
            continue
        draw = res["draw"]
        actual_main = [n for n in draw["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        hot_set, warm_set, cold_set = res["hot_set"], res["warm_set"], res["cold_set"]
        hot_pred, warm_pred, cold_pred = res["hot_pred"], res["warm_pred"], res["cold_pred"]

        print_diagnostics(
            f"{lottery_name} {D_t.isoformat()}",
            res["M"], res["K"],
            hot_set, warm_set, cold_set,
            res["avg_hot"], res["avg_warm"], res["avg_cold"], res["bias"],
            res["hot_w"], res["warm_w"], res["cold_w"],
            res["TRIALS"], hot_pred, warm_pred, cold_pred,
        )

        apply_learning(actual_main, hot_set, warm_set, cold_set,
                       res["f_main"], hot_pred, cold_pred, res["dec_pred"],
                       res["clusters"])

    # Final prediction (Step 13)
    prediction_lottery_name, prediction_date, prediction_draw_size = PREDICTION_TARGET
    assert prediction_draw_size == main_draw_size(prediction_lottery_name)

    print("\n=== Final prediction ===")
    res_p = run_full_steps_for(prediction_date, prediction_lottery_name,
                               W_cat_star, W_dec_star, "prediction")
    if res_p is None:
        print("No data for prediction date (windows empty).")
        return

    prob = res_p["prob"]
    centre_score = res_p["centre_score"]
    alpha_centre = 0.25

    # Centre-biased tuple scores
    tuple_scores = {}
    for T, p in prob.items():
        avg_centre = sum(centre_score.get(n, 0.5) for n in T) / len(T)
        mult = 1.0 + alpha_centre * (avg_centre - 0.5)
        mult = clamp(mult, 0.8, 1.2)
        tuple_scores[T] = p * mult

    top10 = sorted(tuple_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    print(f"\nTop-10 predicted tuples for prediction ({prediction_lottery_name}, {prediction_date}):")
    for T, score in top10:
        print(f"  {T}  score={score:.6f}  prob={prob[T]:.6f}")


if __name__ == "__main__":
    main()
