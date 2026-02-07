#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
You can adjust only:
  - DATA section
  - TARGET_DRAWS_FOR_LEARNING
  - PREDICTION_TARGET
  - TOP_N_PREDICTIONS, MC/cluster tunables in CONFIG
"""

import math
import random
import datetime
import itertools
import os
import csv
import ast
import re
from collections import Counter, defaultdict
from statistics import mean, median

# BASE_TRIALS = 3000         # base TRIALS factor before scaling
# MIN_TRIALS =  9000          # minimum MC trials
# MAX_TRIALS = 12000         # maximum MC trials

BASE_TRIALS = 3         # base TRIALS factor before scaling
MIN_TRIALS = 6          # minimum MC trials
MAX_TRIALS = 9         # maximum MC trials

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX = 45
NUMBER_RANGE = list(range(MAIN_NUMBER_MIN, MAIN_NUMBER_MAX + 1))
LOG_SCORE_MAX = 4.0

gamma_hop = 0.55     # γ_hop
theta_resurge = 0.25 # θ_resurge

# ---- V4.1 tuning ----
ALPHA_CENTRE = 0.22         # strength of centre-bias in tuple ranking
BETA_CLUSTER = 0.30         # strength of joint cluster bias in tuple ranking
WINDOW_LENGTH = (6, 7, 8, 9)
PREDICTION_CONFIG = {
    # Example (uncomment when needed):
    # "BASE_TRIALS": 60000,
    # "MIN_TRIALS": 100000,
    # "MAX_TRIALS": 200000,
    # "CLUSTER_TRIAL_FRAC": 0.25,
}

CLUSTER_TRIAL_FRAC = 0.25

STRUCT_CONFIDENCE_WEIGHT = 1.2  # how much structure confidence boosts TRIALS

CLUSTER_LAMBDA_BASE = 0.25  # stronger per-number cluster boost


TOP_N_PREDICTIONS = 10     # you can change this to print more/less predictions

DECADE_BANDS = []

DECADES = [band[0] for band in DECADE_BANDS]
N_DECADES = len(DECADES)

# How far back we look for SFL → other-lottery momentum
SFL_MOMENTUM_DAYS = 7

# Strength of SFL momentum (0.2–0.5 is reasonable; start mild)
SFL_MOMENTUM_K = 0.35
SFL_MOMENTUM_W_MAX = 1.30   # max multiplicative boost from SFL momentum

# Focused recent-draw adjacency (±1) from last N SFL draws
SFL_LASTN_DRAWS = 3
SFL_LASTN_PM1_K = 0.15      # per-hit multiplicative boost
SFL_LASTN_PM1_W_MAX = 1.60
SFL_PM1_STRICT_COHORT = False
SFL_PM1_STRICT_LEADER = False
SFL_PM1_COMBO_ENABLED = False
SFL_PM1_COMBO_TOP_N = 12
SFL_PM1_COMBO_MAX = 20
TOP_P_COMBO_ENABLED = False
TOP_P_COMBO_N = 14
TOP_P_COMBO_MAX = 20
PM1_WEIGHTED_COMBO_ENABLED = False
PM1_WEIGHTED_COMBO_TOP_N = 18
PM1_WEIGHTED_COMBO_MAX = 20
PM1_WEIGHTED_SCORE_W = 3.0
PM1_GREEDY_ENABLED = False
PM1_GREEDY_TOP_N = 12
PM1_GREEDY_MAX = 20
PM1_ONLY_MODE = False
PM1_ONLY_TOP_N = 16

# Last-draw suppression for HOT numbers
RECENCY_LASTDRAW_SUPPRESS = 0.85  # factor < 1.0 to slightly punish "just hit" HOT numbers

RECENT_DECADE_DAYS = 2          # short recency window for decades (auto suppression/boost)
RECENT_DECADE_W_MIN = 0.5       # strongest suppression factor
RECENT_DECADE_W_MAX = 1.8       # strongest boost factor

# Short vs long pressure windows
RECENT_DECADE_DAYS_SHORT = 2    # very reactive (what you're already using)
RECENT_DECADE_DAYS_LONG  = 7    # slower background pressure

# Blend between short and long pressure
RECENT_DECADE_SHORT_WEIGHT = 0.65
RECENT_DECADE_LONG_WEIGHT  = 0.35  # must satisfy SHORT_WEIGHT + LONG_WEIGHT = 1.0

TARGET_DRAWS_FOR_LEARNING = []

# Non-linear pressure → boost curve (higher = more aggressive)
RECENT_DECADE_NONLIN_K = 1.1

LOTTERIES = {
    "Set for Life":     {"main_draw_size": 7, "uses_supp": True,  "uses_powerball": False},
    "Weekday Windfall": {"main_draw_size": 6, "uses_supp": True,  "uses_powerball": False},
    "OZ Lotto":         {"main_draw_size": 7, "uses_supp": True,  "uses_powerball": False},
    "Powerball":        {"main_draw_size": 7, "uses_supp": False, "uses_powerball": True},
    "Saturday Lotto":   {"main_draw_size": 6, "uses_supp": True,  "uses_powerball": False},
}

HOP_SOURCE_LOTTERY = "Set for Life"
HOP_DESTINATION_LOTTERY = ""

WINDOW_SIZE_CANDIDATES = [6, 7, 8, 9]

HWC_OVERRIDE = None              # e.g. (h_target, w_target, c_target)
DECADE_FACTORS_OVERRIDE = None   # e.g. {1: 0.9, 2: 1.1, 3: 1.0, 4: 1.05, 5: 0.95}

RUNNING_PB_ONLY = False

# Optional locked-regime override (set by runners)
LOCKED_REGIME_DATES = None
LOCKED_REGIME_LOTTERY = None
LOCKED_REGIME_SNAPSHOT_CACHE = {}

# Optional cohort diversification cap (fraction of max_tickets_to_print)
COHORT_USAGE_CAP_FRAC = None
# Auto predictor evaluation window for cohort envelopes
COHORT_AUTOPRED_EVAL_LAST_N = 3
# Optional multi-shape allowance for initial ticket build
COHORT_ALLOWED_HWC_TOP_K = 1
COHORT_ALLOWED_DEC_TOP_K = 1
# Optional leader repetition cap (per leader value)
LEADER_USAGE_CAP = None
# Optional ticket selection diversity penalty (higher => more unique numbers)
TICKET_DIVERSITY_LAMBDA = None
# Optional coverage-optimized ticket selection (tries to avoid <3-hit weeks)
COVERAGE_MODE = False
COVERAGE_ALPHA = 1.0

def apply_prediction_config_overrides():
    """
    Apply overrides from PREDICTION_CONFIG to global config variables.
    This only runs in prediction mode. Defaults in the file remain unchanged.

    Supported keys in PREDICTION_CONFIG:
      - BASE_TRIALS, MIN_TRIALS, MAX_TRIALS, CLUSTER_TRIAL_FRAC
      - HWC_OVERRIDE:           (h_target, w_target, c_target)
      - DECADE_FACTORS_OVERRIDE: {decade_id: factor, ...}
    """
    global BASE_TRIALS, MIN_TRIALS, MAX_TRIALS
    global CLUSTER_TRIAL_FRAC
    global HWC_OVERRIDE, DECADE_FACTORS_OVERRIDE
    global EXPLORE_FRAC

    if not PREDICTION_CONFIG:
        return  # nothing to override

    if "EXPLORE_FRAC" in PREDICTION_CONFIG:
        EXPLORE_FRAC = float(PREDICTION_CONFIG["EXPLORE_FRAC"])
    # Trial-count related knobs
    if "BASE_TRIALS" in PREDICTION_CONFIG:
        BASE_TRIALS = PREDICTION_CONFIG["BASE_TRIALS"]
    if "MIN_TRIALS" in PREDICTION_CONFIG:
        MIN_TRIALS = PREDICTION_CONFIG["MIN_TRIALS"]
    if "MAX_TRIALS" in PREDICTION_CONFIG:
        MAX_TRIALS = PREDICTION_CONFIG["MAX_TRIALS"]
    if "CLUSTER_TRIAL_FRAC" in PREDICTION_CONFIG:
        CLUSTER_TRIAL_FRAC = PREDICTION_CONFIG["CLUSTER_TRIAL_FRAC"]

    if "HWC_OVERRIDE" in PREDICTION_CONFIG:
        try:
            hwc = PREDICTION_CONFIG["HWC_OVERRIDE"]
            if hwc is not None and len(hwc) == 3:
                HWC_OVERRIDE = (int(hwc[0]), int(hwc[1]), int(hwc[2]))
        except Exception:
            # If mis-specified, ignore and fall back to learned targets
            HWC_OVERRIDE = None

    # Decade-factor manual override (optional)
    # Expected shape: {1: factor_for_1_10, 2: factor_for_11_20, ...}
    if "DECADE_FACTORS_OVERRIDE" in PREDICTION_CONFIG:
        try:
            dct = dict(PREDICTION_CONFIG["DECADE_FACTORS_OVERRIDE"])
            # Only keep positive factors
            DECADE_FACTORS_OVERRIDE = {
                int(k): float(v)
                for k, v in dct.items()
                if float(v) > 0.0
            }
        except Exception:
            DECADE_FACTORS_OVERRIDE = None

# ===========================
# Helper math functions
# ===========================

def clamp(x, a, b):
    return min(max(x, a), b)

def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

def variance_population(values):
    k = len(values)
    if k == 0:
        return 0.0
    m = sum(values) / k
    return sum((v - m) ** 2 for v in values) / k

def decade_of(n):
    if n < MAIN_NUMBER_MIN or n > MAIN_NUMBER_MAX:
        return None
    for d_id, start, end in DECADE_BANDS:
        if start <= n <= end:
            return d_id
    return None

def main_draw_size(lottery_name):
    return LOTTERIES[lottery_name]["main_draw_size"]

def hwc_counts(numbers, Hot_set, Warm_set, Cold_set):
    h = sum(1 for n in numbers if n in Hot_set)
    w = sum(1 for n in numbers if n in Warm_set)
    c = sum(1 for n in numbers if n in Cold_set)
    return h, w, c

def decade_counts(numbers):
    dec = {d_id: 0 for d_id in DECADES}
    for n in numbers:
        d_id = decade_of(n)
        if d_id is not None:
            dec[d_id] += 1
    return dec

def compute_recent_decade_weights(target_date,
                                  recent_days=RECENT_DECADE_DAYS):
    # ===============================
    # STRICT DECADE OVERRIDE MODE
    # ===============================
    if DECADE_FACTORS_OVERRIDE:
        dec_factors = {d_id: 1.0 for d_id in DECADES}

        for d_id, fac in DECADE_FACTORS_OVERRIDE.items():
            if d_id in dec_factors and fac > 0:
                dec_factors[d_id] = float(fac)

        decade_weight_log = {}
        for n in NUMBER_RANGE:
            d_id = decade_of(n)
            decade_weight_log[n] = 0.0 if d_id is None else math.log(dec_factors[d_id])

        # No learning, no pressure — return immediately
        return decade_weight_log, {}, dec_factors

    # --- helper: count decade hits for a date range ---
    def count_decades_over_range(start_date, end_date):
        counts = {d_id: 0 for d_id in DECADES}
        for dt, draws in draws_by_date.items():
            if not (start_date <= dt <= end_date):
                continue
            for dr in draws:
                nums = []
                # main
                nums.extend(n for n in dr.main
                            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX)
                # supp if lottery uses them
                if LOTTERIES[dr.lottery]["uses_supp"] and dr.supp:
                    nums.extend(n for n in dr.supp
                                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX)
                # powerball as a normal number if in range
                if dr.powerball and RUNNING_PB_ONLY:
                    nums.extend(n for n in dr.powerball
                                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX)
                for n in nums:
                    d_id = decade_of(n)
                    if d_id is not None:
                        counts[d_id] += 1
        return counts

    # --- window setup ---
    short_days = recent_days               # usually 2 (RECENT_DECADE_DAYS)
    long_days  = RECENT_DECADE_DAYS_LONG  # e.g. 7

    short_start = target_date - datetime.timedelta(days=short_days)
    long_start  = target_date - datetime.timedelta(days=long_days)
    end_date    = target_date - datetime.timedelta(days=1)

    # --- 1) short & long counts ---
    dec_short_counts = count_decades_over_range(short_start, end_date)
    dec_long_counts  = count_decades_over_range(long_start,  end_date)

    total_short = sum(dec_short_counts.values())
    total_long  = sum(dec_long_counts.values())

    # If there is absolutely no information in either window -> neutral
    if total_short == 0 and total_long == 0:
        decade_weight_log = {n: 0.0 for n in NUMBER_RANGE}
        dec_factors = {d_id: 1.0 for d_id in DECADES}
        return decade_weight_log, dec_short_counts, dec_factors

    eps = 1e-9

    # --- 2) normalise short window to [0,1] pressure ---
    short_min = min(dec_short_counts.values())
    short_max = max(dec_short_counts.values())
    if short_max == short_min:
        # flat short window -> treat as neutral mid pressure
        p_short = {d_id: 0.5 for d_id in DECADES}
    else:
        p_short = {}
        for d_id, C_d in dec_short_counts.items():
            p_short[d_id] = (C_d - short_min) / (short_max - short_min + eps)

    # --- 3) normalise long window to [0,1] pressure ---
    long_min = min(dec_long_counts.values())
    long_max = max(dec_long_counts.values())
    if long_max == long_min:
        # flat long window -> neutral mid pressure
        p_long = {d_id: 0.5 for d_id in DECADES}
    else:
        p_long = {}
        for d_id, C_d in dec_long_counts.items():
            p_long[d_id] = (C_d - long_min) / (long_max - long_min + eps)

    # ensure weights are sane
    a = RECENT_DECADE_SHORT_WEIGHT
    b = RECENT_DECADE_LONG_WEIGHT
    if abs(a + b - 1.0) > 1e-6:
        # safety: renormalise if user accidentally changes them
        total_ab = a + b
        a /= total_ab
        b /= total_ab

    # --- 4) combined pressure + non-linear inversion ---
    # pressure[d] in [0,1]; 0.5 is "neutral".
    # use pressure deviation from 0.5 in an exponential curve
    raw_factors = {}
    for d_id in DECADES:
        p_s = p_short[d_id]
        p_l = p_long[d_id]
        pressure = a * p_s + b * p_l  # combined pressure

        # centre at 0.5 (neutral), positive = high pressure, negative = low
        x = pressure - 0.5
        # non-linear inversion: high pressure → factor < 1, low → > 1
        factor = math.exp(-RECENT_DECADE_NONLIN_K * x)

        # clip to configured min/max first (soft safety)
        factor = clamp(factor, RECENT_DECADE_W_MIN, RECENT_DECADE_W_MAX)
        raw_factors[d_id] = factor

    # --- 5) renormalise so the average factor is exactly 1.0 ---
    avg_raw = sum(raw_factors.values()) / len(raw_factors)
    if avg_raw <= 0:
        dec_factors = {d_id: 1.0 for d_id in DECADES}
    else:
        dec_factors = {d_id: raw_factors[d_id] / avg_raw for d_id in DECADES}

    if DECADE_FACTORS_OVERRIDE:
        # Overwrite only known decades, keep others as learned.
        for d_id, fac in DECADE_FACTORS_OVERRIDE.items():
            if d_id in dec_factors and fac > 0.0:
                dec_factors[d_id] = float(fac)


    # --- 6) map to per-number log weights ---
    decade_weight_log = {}
    for n in NUMBER_RANGE:
        d_id = decade_of(n)
        if d_id is None:
            decade_weight_log[n] = 0.0
        else:
            decade_weight_log[n] = math.log(dec_factors[d_id])

    return decade_weight_log, dec_short_counts, dec_factors

# ===========================
# Data setup: historical draws
# ===========================

class Draw:
    __slots__ = ("date", "lottery", "main", "supp", "powerball")
    def __init__(self, date, lottery, main, supp=None, powerball=None):
        self.date = date
        self.lottery = lottery
        self.main = list(main)
        self.supp = list(supp) if supp is not None else []
        self.powerball = list(powerball) if powerball is not None else []

def d(day, month, year=2025):
    return datetime.date(year, month, day)

global_draws = []

# Index draws by date (built later via finalize_data)
draws_by_date = defaultdict(list)

def finalize_data():
    global draws_by_date
    # Sort draws by date then lottery for determinism
    global_draws.sort(key=lambda dr: (dr.date, dr.lottery))

    # Rebuild the index
    draws_by_date = defaultdict(list)
    for dr in global_draws:
        draws_by_date[dr.date].append(dr)


def get_actual_main(lottery_name, date):
    for dr in draws_by_date.get(date, []):
        if dr.lottery == lottery_name:
            return [n for n in dr.main if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
    return None

def get_actual_pb(lottery_name, date):
    for dr in draws_by_date.get(date, []):
        if dr.lottery == lottery_name:
            return [n for n in dr.powerball if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
    return None

# ===========================
# Learning state
# ===========================

class LearningState:
    def __init__(self):
        self.delta_hot = 0.0
        self.delta_warm = 0.0
        self.delta_cold = 0.0
        self.cluster_priority_score_global = {}  # cluster tuple -> offset

    def reset(self):
        self.__init__()

# ===========================
# Step A: build windows
# ===========================

def build_windows(target_date, window_cat):
    seed_dates_cat = [target_date - datetime.timedelta(days=delta)
                      for delta in range(window_cat, 0, -1)]

    for dt in seed_dates_cat:
        if dt not in draws_by_date:
            return None

    seed_draws_cat = []
    for dt in seed_dates_cat:
        seed_draws_cat.extend(draws_by_date[dt])

    return seed_dates_cat, seed_draws_cat

# ===========================
# Step B: category frequencies
# ===========================

def step_B_category(seed_draws_cat, seed_dates_cat):
    seed_numbers_main = []
    seed_numbers_supp = []
    last_main_date = {n: None for n in NUMBER_RANGE}
    lotteries_seen_by_n = {n: set() for n in NUMBER_RANGE}

    for dr in seed_draws_cat:
        # main
        for n in dr.main:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                seed_numbers_main.append(n)
                if (last_main_date[n] is None) or (dr.date > last_main_date[n]):
                    last_main_date[n] = dr.date
                lotteries_seen_by_n[n].add(dr.lottery)
        # supp + powerball
        supp_nums = []
        if dr.supp:
            supp_nums.extend(dr.supp)
        if dr.powerball:
            supp_nums.extend(dr.powerball)
        for n in supp_nums:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                seed_numbers_supp.append(n)
                lotteries_seen_by_n[n].add(dr.lottery)

    f_main = {n: 0 for n in NUMBER_RANGE}
    f_supp = {n: 0 for n in NUMBER_RANGE}
    for n in seed_numbers_main:
        f_main[n] += 1
    for n in seed_numbers_supp:
        f_supp[n] += 1

    L_counts = {n: len(lotteries_seen_by_n[n]) for n in NUMBER_RANGE}
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)

    return f_main, f_supp, L_counts, last_main_date, M, seed_numbers_main



# ===========================
# Step 4E: cross-lottery hop
# ===========================

def cross_lottery_hop(seed_draws_cat, window_size_cat):
    appearances = {n: [] for n in NUMBER_RANGE}
    sfl_count = {n: 0 for n in NUMBER_RANGE}
    non_sfl_count = {n: 0 for n in NUMBER_RANGE}

    for dr in seed_draws_cat:
        date = dr.date
        lot = dr.lottery
        for n in dr.main:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                appearances[n].append((date, lot))
                if lot == HOP_SOURCE_LOTTERY:
                    sfl_count[n] += 1
                else:
                    non_sfl_count[n] += 1

    cross_pair_sum = {n: 0.0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        app = appearances[n]
        if len(app) < 2:
            continue
        for (d1, L1) in app:
            for (d2, L2) in app:
                if d2 > d1 and L2 != L1:
                    lag_days = (d2 - d1).days
                    if lag_days <= 0 or lag_days > window_size_cat:
                        continue
                    lag_days = max(1, lag_days)
                    base_pair = 1.0 / lag_days
                    w_dir = 1.0

                    # Strong direct SFL -> OZ Lotto hop,
                    # extra boost if the lag is very short (<= 3 days)
                    if L1 == HOP_SOURCE_LOTTERY and L2 == HOP_DESTINATION_LOTTERY:
                        w_dir *= 2.2
                        if lag_days <= 3:
                            w_dir *= 1.5
                    # Other SFL -> non-SFL hops (still useful but weaker)
                    elif L1 == HOP_SOURCE_LOTTERY and L2 != HOP_SOURCE_LOTTERY:
                        w_dir *= 1.4
                    # Non-SFL -> OZ hops
                    elif L2 == HOP_DESTINATION_LOTTERY and L1 != HOP_SOURCE_LOTTERY:
                        w_dir *= 1.2
                    cross_pair_sum[n] += base_pair * w_dir

    base_hop_score = {n: sfl_count[n] * non_sfl_count[n] + cross_pair_sum[n]
                      for n in NUMBER_RANGE}
    max_hop = max(base_hop_score.values()) if base_hop_score else 0.0

    cross_hop_score = {}
    cross_hop_log = {}
    if max_hop <= 0:
        for n in NUMBER_RANGE:
            cross_hop_score[n] = 0.0
            cross_hop_log[n] = 0.0
    else:
        for n in NUMBER_RANGE:
            score = base_hop_score[n] / max_hop
            cross_hop_score[n] = score
            if score == 0:
                cross_hop_log[n] = 0.0
            else:
                cross_hop_log[n] = math.log(1.0 + gamma_hop * score)

    return cross_hop_score, cross_hop_log

# ===========================
# Step 5: H/W/C classification
# ===========================

def classify_hot_warm_cold(f_main, L_counts):
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)
    inner = max(3.0, M * 0.15)
    K = max(1, round(inner))

    nums_with_freq = [n for n in NUMBER_RANGE if f_main[n] > 0]
    nums_with_zero = [n for n in NUMBER_RANGE if f_main[n] == 0]

    def hot_key(n):
        return (-f_main[n], -L_counts[n], n)

    nums_with_freq.sort(key=hot_key)
    nums_with_zero.sort(key=lambda n: (-L_counts[n], n))

    Hot_list = nums_with_freq[:K]
    if len(Hot_list) < K:
        remaining = K - len(Hot_list)
        Hot_list.extend(nums_with_zero[:remaining])

    Hot_set = set(Hot_list)
    Warm_set = {n for n in NUMBER_RANGE if f_main[n] > 1 and n not in Hot_set}
    Cold_set = {n for n in NUMBER_RANGE if f_main[n] <= 1 and n not in Hot_set}

    union = Hot_set | Warm_set | Cold_set
    missing = set(NUMBER_RANGE) - union
    if missing:
        raise RuntimeError(
            f"H/W/C partition error: missing numbers {sorted(missing)}; "
            f"check classification logic."
        )
    return Hot_set, Warm_set, Cold_set, M, K

def compute_category_bias(seed_draws_cat, Hot_set, Warm_set, Cold_set):
    p_hot_list = []
    p_warm_list = []
    p_cold_list = []
    for dr in seed_draws_cat:
        main_nums = [n for n in dr.main if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        if not main_nums:
            continue
        size = len(main_nums)
        h = sum(1 for n in main_nums if n in Hot_set)
        w = sum(1 for n in main_nums if n in Warm_set)
        c = sum(1 for n in main_nums if n in Cold_set)
        p_hot_list.append(h / size)
        p_warm_list.append(w / size)
        p_cold_list.append(c / size)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

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

    return avg_hot, avg_warm, avg_cold, bias

def category_weights(avg_hot, avg_warm, avg_cold, state: LearningState):
    # Base patterns – close to V4.1, but slightly kinder to HOT
    if avg_hot > avg_warm and avg_hot > avg_cold:
        base_hot, base_warm, base_cold = (1.4, 1.15, 0.6)
    elif avg_warm >= avg_hot and avg_warm >= avg_cold:
        # was (0.95, 1.35, 1.05) – soften warm bias, lift hot a touch
        base_hot, base_warm, base_cold = (1.00, 1.30, 1.00)
    elif avg_cold > avg_hot and avg_cold > avg_warm:
        # slight bump to hot vs warm in cold-heavy regimes
        base_hot, base_warm, base_cold = (0.85, 0.95, 1.40)
    else:
        # balanced regime, nudge hot a bit
        base_hot, base_warm, base_cold = (1.05, 1.00, 0.95)

    hot_w  = base_hot  * (1 + (avg_hot  - 1/3) * 0.25)
    warm_w = base_warm * (1 + (avg_warm - 1/3) * 0.25)
    cold_w = base_cold * (1 + (avg_cold - 1/3) * 0.25)

    # V4.2: mild extra tilt towards HOT, without breaking learning
    hot_w  *= 1.06
    warm_w *= 0.99
    cold_w *= 0.95

    # apply learned offsets
    hot_w  += state.delta_hot
    warm_w += state.delta_warm
    cold_w += state.delta_cold

    hot_w  = clamp(hot_w,  0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    # normalise so average weight ~1.0
    s = hot_w + warm_w + cold_w
    if s > 0:
        factor = s / 3.0
        hot_w  /= factor
        warm_w /= factor
        cold_w /= factor

    # blend back towards 1.0 to avoid runaway
    hot_w  = 0.75 * hot_w  + 0.25 * 1.0
    warm_w = 0.75 * warm_w + 0.25 * 1.0
    cold_w = 0.75 * cold_w + 0.25 * 1.0

    hot_w  = clamp(hot_w,  0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    return hot_w, warm_w, cold_w

# ===========================
# Step 7: per-number log scores
# ===========================

def per_number_log_scores(seed_numbers_main, f_main, f_supp, L_counts,
                          last_main_date, target_date, current_lottery_name,
                          Hot_set, Warm_set, Cold_set,
                          hot_w, warm_w, cold_w,
                          decade_weight_log, cross_hop_log,
                          window_size_cat):
    # adjacency
    freq_main_all = Counter(seed_numbers_main)
    adj_count = {}
    for n in NUMBER_RANGE:
        count = 0
        if n - 1 in freq_main_all:
            count += freq_main_all[n - 1]
        if n + 1 in freq_main_all:
            count += freq_main_all[n + 1]
        adj_count[n] = count
    max_adj = max(adj_count.values()) if adj_count else 1
    if max_adj <= 0:
        max_adj = 1
    adj_log = {}
    for n in NUMBER_RANGE:
        adj_score_raw = 0.15 + 0.40 * (adj_count[n] / max_adj)
        adj_log[n] = math.log(1 + adj_score_raw)

    # last-3-days delta (all lotteries) + SFL-specific momentum
    last3_start = target_date - datetime.timedelta(days=3)
    main_hits_last3 = {n: False for n in NUMBER_RANGE}
    supp_hits_last3 = {n: False for n in NUMBER_RANGE}

    # V4.3: SFL-specific repeat + adjacency
    sfl_hits_last3 = {n: 0 for n in NUMBER_RANGE}
    sfl_adj_count = {n: 0 for n in NUMBER_RANGE}

    for dt, draws in draws_by_date.items():
        if last3_start <= dt <= target_date - datetime.timedelta(days=1):
            for dr in draws:
                # general last-3-days flags (all lotteries)
                for x in dr.main:
                    if MAIN_NUMBER_MIN <= x <= MAIN_NUMBER_MAX:
                        main_hits_last3[x] = True
                supp_nums = []
                if dr.supp:
                    supp_nums.extend(dr.supp)
                if dr.powerball:
                    supp_nums.extend(dr.powerball)
                for x in supp_nums:
                    if MAIN_NUMBER_MIN <= x <= MAIN_NUMBER_MAX and not main_hits_last3[x]:
                        supp_hits_last3[x] = True

                # SFL-only repeat + adjacency
                if dr.lottery == HOP_SOURCE_LOTTERY:  # "Set for Life"
                    sfl_main = [x for x in dr.main
                                if MAIN_NUMBER_MIN <= x <= MAIN_NUMBER_MAX]
                    for x in sfl_main:
                        sfl_hits_last3[x] += 1
                        # adjacency in ±1, ±2 band
                        for n in (x - 2, x - 1, x + 1, x + 2):
                            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                                sfl_adj_count[n] += 1

    # Base last-3-days (all lotteries)
    delta_log = {}
    for n in NUMBER_RANGE:
        if main_hits_last3[n]:
            mult = 1.4
        elif supp_hits_last3[n]:
            mult = 1.2
        else:
            mult = 1.0
        delta_log[n] = math.log(mult)

    # V4.3: SFL repeat boost (recent exact repeats)
    sfl_repeat_log = {}
    for n in NUMBER_RANGE:
        c = sfl_hits_last3[n]
        if c >= 2:
            mult = 1.6  # strong boost: hit at least twice in last 3 SFL draws
        elif c == 1:
            mult = 1.25  # mild boost: hit once in last 3 SFL draws
        else:
            mult = 1.0
        sfl_repeat_log[n] = math.log(mult)

    # V4.3: SFL adjacency boost from last 3 days
    max_sfl_adj = max(sfl_adj_count.values()) if sfl_adj_count else 0
    if max_sfl_adj <= 0:
        max_sfl_adj = 1
    sfl_adj_log = {}
    for n in NUMBER_RANGE:
        # normalised SFL adjacency intensity for n
        adj_ratio = sfl_adj_count[n] / max_sfl_adj
        # a bit stronger than global adj, but only for SFL window
        adj_score_raw = 0.10 + 0.60 * adj_ratio
        sfl_adj_log[n] = math.log(1 + adj_score_raw)

    # Focused SFL ±1 boost from last N SFL draws (by draw count, not days)
    sfl_pm1_log = {n: 0.0 for n in NUMBER_RANGE}
    if current_lottery_name != HOP_SOURCE_LOTTERY:
        sfl_draws = []
        for dt, draws in draws_by_date.items():
            if dt >= target_date:
                continue
            for dr in draws:
                if dr.lottery == HOP_SOURCE_LOTTERY:
                    sfl_draws.append((dt, dr.main))
        sfl_draws.sort(key=lambda x: x[0], reverse=True)
        sfl_draws = sfl_draws[:SFL_LASTN_DRAWS]
        pm1_counts = {n: 0 for n in NUMBER_RANGE}
        for _, main_nums in sfl_draws:
            for x in main_nums:
                for n in (x - 1, x + 1):
                    if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                        pm1_counts[n] += 1
        for n in NUMBER_RANGE:
            c = pm1_counts[n]
            if c <= 0:
                continue
            factor = min(1.0 + SFL_LASTN_PM1_K * c, SFL_LASTN_PM1_W_MAX)
            sfl_pm1_log[n] = math.log(factor)

    # cross-lottery density
    cross_log = {n: math.log(1 + 0.08 * L_counts[n]) for n in NUMBER_RANGE}

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
        if n in Hot_set:
            category_weight_log[n] = math.log(hot_w)
        elif n in Warm_set:
            category_weight_log[n] = math.log(warm_w)
        else:
            category_weight_log[n] = math.log(cold_w)

    # cold resurgence
    resurge_raw = {n: 0.0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        lm = last_main_date[n]
        if lm is not None:
            gap_days = (target_date - lm).days
            if 4 <= gap_days <= window_size_cat:
                resurge_raw[n] = 1.0 / gap_days
            else:
                resurge_raw[n] = 0.0
        else:
            resurge_raw[n] = 0.0
    max_resurge = max(resurge_raw.values()) if resurge_raw else 0.0
    cold_resurge_score = {}
    cold_resurge_log = {}
    if max_resurge <= 0:
        for n in NUMBER_RANGE:
            cold_resurge_score[n] = 0.0
            cold_resurge_log[n] = 0.0
    else:
        for n in NUMBER_RANGE:
            score = resurge_raw[n] / max_resurge
            cold_resurge_score[n] = score
            if n in Hot_set:
                cold_resurge_log[n] = 0.0
            else:
                cold_resurge_log[n] = math.log(1.0 + theta_resurge * score)

    # --- V6: SFL → other-lottery momentum (7-day window) ---
    sfl_momentum_log = {n: 0.0 for n in NUMBER_RANGE}

    if current_lottery_name != HOP_SOURCE_LOTTERY:
        start_sfl = target_date - datetime.timedelta(days=SFL_MOMENTUM_DAYS)
        end_sfl = target_date - datetime.timedelta(days=1)
        sfl_counts = {n: 0 for n in NUMBER_RANGE}

        for dt, draws in draws_by_date.items():
            if not (start_sfl <= dt <= end_sfl):
                continue
            for dr in draws:
                if dr.lottery != HOP_SOURCE_LOTTERY:
                    continue
                for x in dr.main:
                    if MAIN_NUMBER_MIN <= x <= MAIN_NUMBER_MAX:
                        sfl_counts[x] += 1

        max_sfl = max(sfl_counts.values()) if sfl_counts else 0
        if max_sfl > 0:
            for n in NUMBER_RANGE:
                c = sfl_counts[n]
                if c <= 0:
                    sfl_momentum_log[n] = 0.0
                    continue
                # normalised activity in [0,1]
                r = c / max_sfl
                # non-linear gentle boost from SFL
                factor = math.exp(SFL_MOMENTUM_K * r)
                factor = clamp(factor, 1.0, SFL_MOMENTUM_W_MAX)
                sfl_momentum_log[n] = math.log(factor)

    # --- V6: last-draw suppression for HOT numbers ---
    recency_log = {n: 0.0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        lm = last_main_date.get(n)
        if lm is None:
            continue
        gap_days = (target_date - lm).days
        # If a HOT number just hit yesterday, lightly suppress it
        if gap_days == 1 and n in Hot_set:
            recency_log[n] = math.log(RECENCY_LASTDRAW_SUPPRESS)
        else:
            recency_log[n] = 0.0

    # total log score
    log_score = {}
    rawP = {}
    for n in NUMBER_RANGE:
        ls = (
                adj_log[n] +  # global adjacency in the full window
                delta_log[n] +  # last-3-days all lotteries (base)
                sfl_repeat_log[n] +  # SFL exact-repeat momentum (last 3 days)
                sfl_adj_log[n] +  # SFL adjacency momentum (±1, ±2)
                sfl_pm1_log[n] +  # SFL last-N draws ±1 boost
                cross_log[n] +  # cross-lottery density (L_counts)
                supp_log[n] +  # supp-only bonus
                category_weight_log[n] +  # H/W/C weighting
                decade_weight_log[n] +  # V6 decade pressure/depletion
                cross_hop_log[n] +  # SFL → OZ hop structure
                cold_resurge_log[n] +  # cold resurgence (mid-window for non-hot)
                sfl_momentum_log[n] +  # NEW: 7-day SFL → other-lottery momentum
                recency_log[n]  # NEW: HOT last-draw suppression
        )
        ls = min(ls, LOG_SCORE_MAX)
        log_score[n] = ls
        rawP[n] = math.exp(ls)
    total_rawP = sum(rawP.values())
    if total_rawP <= 0:
        P = {n: 1.0 / len(NUMBER_RANGE) for n in NUMBER_RANGE}
    else:
        P = {n: rawP[n] / total_rawP for n in NUMBER_RANGE}

    return P, log_score, cold_resurge_score

def _sfl_lastn_pm1_set(target_date, n_draws):
    """
    Build a set of numbers that are ±1 of the last N SFL draws before target_date.
    """
    sfl_draws = []
    for dt, draws in draws_by_date.items():
        if dt >= target_date:
            continue
        for dr in draws:
            if dr.lottery == HOP_SOURCE_LOTTERY:
                sfl_draws.append((dt, dr.main))
    sfl_draws.sort(key=lambda x: x[0], reverse=True)
    sfl_draws = sfl_draws[:n_draws]
    pm1 = set()
    for _, main_nums in sfl_draws:
        for x in main_nums:
            for n in (x - 1, x + 1):
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    pm1.add(n)
    return pm1

def _sfl_lastn_pm1_counts(target_date, n_draws):
    counts = {n: 0 for n in NUMBER_RANGE}
    sfl_draws = []
    for dt, draws in draws_by_date.items():
        if dt >= target_date:
            continue
        for dr in draws:
            if dr.lottery == HOP_SOURCE_LOTTERY:
                sfl_draws.append((dt, dr.main))
    sfl_draws.sort(key=lambda x: x[0], reverse=True)
    sfl_draws = sfl_draws[:n_draws]
    for _, main_nums in sfl_draws:
        for x in main_nums:
            for n in (x - 1, x + 1):
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    counts[n] += 1
    return counts

def _build_pm1_combo_tickets(pred, meta_map, sfl_pm1, *, top_n, max_tickets, decade_ids):
    if not sfl_pm1:
        return []
    pm1_nums = [n for n in NUMBER_RANGE if n in sfl_pm1]
    pm1_nums.sort(key=lambda n: meta_map[n]["P"], reverse=True)
    pm1_nums = pm1_nums[:top_n]
    if len(pm1_nums) < pred["K"]:
        return []

    tickets = []
    for combo in itertools.combinations(pm1_nums, pred["K"]):
        leader_n = max(combo, key=lambda n: meta_map[n]["P"])
        cohort = [n for n in combo if n != leader_n]
        hwc = {"H": 0, "W": 0, "C": 0}
        dec = {d: 0 for d in decade_ids}
        ps = []
        ranks = []
        log_scores = []
        max_p = -1.0
        for n in cohort:
            m = meta_map[n]
            hwc[m["category"]] += 1
            dec[m["decade"]] += 1
            ps.append(m["P"])
            ranks.append(m["rank"])
            log_scores.append(m["log_score"])
            max_p = max(max_p, m["P"])
        leader_meta = meta_map[leader_n]
        pm1_hits = sum(1 for n in combo if n in sfl_pm1)
        tickets.append({
            "leader": leader_n,
            "leader_rank": leader_meta["rank"],
            "leader_P": leader_meta["P"],
            "cohort": tuple(cohort),
            "hwc": (hwc["H"], hwc["W"], hwc["C"]),
            "decades": dec,
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "rank_span": max(ranks) - min(ranks),
            "P_sum": sum(ps),
            "P_mean": sum(ps) / len(ps),
            "P_min": min(ps),
            "P_max": max(ps),
            "sum_log_score": sum(log_scores) + leader_meta["log_score"],
            "product_P": math.prod(ps) * leader_meta["P"],
            "pm1_hits": pm1_hits,
            "P_total": sum(ps) + leader_meta["P"],
        })

    tickets.sort(key=lambda t: (t["P_total"], t["sum_log_score"], t["product_P"]), reverse=True)
    return tickets[:max_tickets]

def _build_top_p_combo_tickets(pred, meta_map, *, top_n, max_tickets, decade_ids):
    ranked = sorted(NUMBER_RANGE, key=lambda n: meta_map[n]["P"], reverse=True)
    top_nums = ranked[:top_n]
    if len(top_nums) < pred["K"]:
        return []
    tickets = []
    for combo in itertools.combinations(top_nums, pred["K"]):
        leader_n = max(combo, key=lambda n: meta_map[n]["P"])
        cohort = [n for n in combo if n != leader_n]
        hwc = {"H": 0, "W": 0, "C": 0}
        dec = {d: 0 for d in decade_ids}
        ps = []
        ranks = []
        log_scores = []
        max_p = -1.0
        for n in cohort:
            m = meta_map[n]
            hwc[m["category"]] += 1
            dec[m["decade"]] += 1
            ps.append(m["P"])
            ranks.append(m["rank"])
            log_scores.append(m["log_score"])
            max_p = max(max_p, m["P"])
        leader_meta = meta_map[leader_n]
        tickets.append({
            "leader": leader_n,
            "leader_rank": leader_meta["rank"],
            "leader_P": leader_meta["P"],
            "cohort": tuple(cohort),
            "hwc": (hwc["H"], hwc["W"], hwc["C"]),
            "decades": dec,
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "rank_span": max(ranks) - min(ranks),
            "P_sum": sum(ps),
            "P_mean": sum(ps) / len(ps),
            "P_min": min(ps),
            "P_max": max(ps),
            "sum_log_score": sum(log_scores) + leader_meta["log_score"],
            "product_P": math.prod(ps) * leader_meta["P"],
            "pm1_hits": 0,
            "P_total": sum(ps) + leader_meta["P"],
        })
    tickets.sort(key=lambda t: (t["P_sum"] + t["leader_P"], t["sum_log_score"], t["product_P"]), reverse=True)
    return tickets[:max_tickets]

def _build_pm1_weighted_combo_tickets(pred, meta_map, pm1_counts, *, top_n, max_tickets, decade_ids, score_w):
    ranked = sorted(NUMBER_RANGE, key=lambda n: (pm1_counts.get(n, 0), meta_map[n]["P"]), reverse=True)
    top_nums = ranked[:top_n]
    if len(top_nums) < pred["K"]:
        return []
    tickets = []
    for combo in itertools.combinations(top_nums, pred["K"]):
        leader_n = max(combo, key=lambda n: meta_map[n]["P"])
        cohort = [n for n in combo if n != leader_n]
        hwc = {"H": 0, "W": 0, "C": 0}
        dec = {d: 0 for d in decade_ids}
        ps = []
        ranks = []
        log_scores = []
        pm1_sum = 0
        for n in cohort:
            m = meta_map[n]
            hwc[m["category"]] += 1
            dec[m["decade"]] += 1
            ps.append(m["P"])
            ranks.append(m["rank"])
            log_scores.append(m["log_score"])
            pm1_sum += pm1_counts.get(n, 0)
        leader_meta = meta_map[leader_n]
        pm1_sum += pm1_counts.get(leader_n, 0)
        tickets.append({
            "leader": leader_n,
            "leader_rank": leader_meta["rank"],
            "leader_P": leader_meta["P"],
            "cohort": tuple(cohort),
            "hwc": (hwc["H"], hwc["W"], hwc["C"]),
            "decades": dec,
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "rank_span": max(ranks) - min(ranks),
            "P_sum": sum(ps),
            "P_mean": sum(ps) / len(ps),
            "P_min": min(ps),
            "P_max": max(ps),
            "sum_log_score": sum(log_scores) + leader_meta["log_score"],
            "product_P": math.prod(ps) * leader_meta["P"],
            "pm1_hits": 0,
            "P_total": sum(ps) + leader_meta["P"],
            "pm1_sum": pm1_sum,
        })
    tickets.sort(
        key=lambda t: (t["pm1_sum"] * score_w + t["P_total"], t["sum_log_score"], t["product_P"]),
        reverse=True,
    )
    return tickets[:max_tickets]

def _build_pm1_greedy_tickets(pred, meta_map, pm1_counts, *, top_n, max_tickets, decade_ids):
    ranked = sorted(NUMBER_RANGE, key=lambda n: (pm1_counts.get(n, 0), meta_map[n]["P"]), reverse=True)
    top_nums = ranked[:top_n]
    if len(top_nums) < pred["K"]:
        return []
    base = top_nums[:pred["K"]]
    base_sorted = sorted(base, key=lambda n: (pm1_counts.get(n, 0), meta_map[n]["P"]))
    replace_pool = top_nums[pred["K"]:]

    def build_ticket(combo):
        leader_n = max(combo, key=lambda n: meta_map[n]["P"])
        cohort = [n for n in combo if n != leader_n]
        hwc = {"H": 0, "W": 0, "C": 0}
        dec = {d: 0 for d in decade_ids}
        ps = []
        ranks = []
        log_scores = []
        for n in cohort:
            m = meta_map[n]
            hwc[m["category"]] += 1
            dec[m["decade"]] += 1
            ps.append(m["P"])
            ranks.append(m["rank"])
            log_scores.append(m["log_score"])
        leader_meta = meta_map[leader_n]
        return {
            "leader": leader_n,
            "leader_rank": leader_meta["rank"],
            "leader_P": leader_meta["P"],
            "cohort": tuple(cohort),
            "hwc": (hwc["H"], hwc["W"], hwc["C"]),
            "decades": dec,
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "rank_span": max(ranks) - min(ranks),
            "P_sum": sum(ps),
            "P_mean": sum(ps) / len(ps),
            "P_min": min(ps),
            "P_max": max(ps),
            "sum_log_score": sum(log_scores) + leader_meta["log_score"],
            "product_P": math.prod(ps) * leader_meta["P"],
            "pm1_hits": 0,
            "P_total": sum(ps) + leader_meta["P"],
            "pm1_sum": sum(pm1_counts.get(n, 0) for n in combo),
        }

    tickets = [build_ticket(base)]
    # Single swaps (replace weakest 1-2 numbers)
    weak = base_sorted[:2]
    for r in replace_pool:
        for w in weak:
            combo = [n for n in base if n != w] + [r]
            tickets.append(build_ticket(combo))
            if len(tickets) >= max_tickets:
                break
        if len(tickets) >= max_tickets:
            break

    # Double swaps for remaining slots
    if len(tickets) < max_tickets and len(replace_pool) >= 2:
        for a, b in itertools.combinations(replace_pool, 2):
            combo = [n for n in base if n not in weak] + [a, b]
            tickets.append(build_ticket(combo))
            if len(tickets) >= max_tickets:
                break

    tickets.sort(
        key=lambda t: (t["pm1_sum"], t["P_total"], t["sum_log_score"], t["product_P"]),
        reverse=True,
    )
    return tickets[:max_tickets]

# ===========================
# V4: centre score computation
# ===========================

def compute_centre_score(f_main):
    """
    For each n, compute how much the centre is favoured vs its neighbours (n-1, n+1).
    centre_score ~ 0.5 => neutral
    > 0.5 => centre stronger than neighbours
    < 0.5 => neighbours stronger
    """
    centre_score = {}
    for n in NUMBER_RANGE:
        centre_hits = f_main.get(n, 0)
        neigh_hits = 0
        if n - 1 >= MAIN_NUMBER_MIN:
            neigh_hits += f_main.get(n - 1, 0)
        if n + 1 <= MAIN_NUMBER_MAX:
            neigh_hits += f_main.get(n + 1, 0)
        denom = centre_hits + neigh_hits
        if denom == 0:
            centre_score[n] = 0.5
        else:
            centre_score[n] = centre_hits / denom
    return centre_score

# ===========================
# Step 8: cluster detection
# ===========================

def detect_clusters(seed_draws_cat, state: LearningState):
    cluster_counter = Counter()
    for dr in seed_draws_cat:
        nums = sorted(set(n for n in dr.main if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX))
        for size in (2, 3, 4):
            if len(nums) >= size:
                for comb in itertools.combinations(nums, size):
                    cluster_counter[comb] += 1
    clusters = {}
    for C, freq in cluster_counter.items():
        if freq >= 2:
            base = 1 + 0.35 * (freq - 1)
            offset = state.cluster_priority_score_global.get(C, 0.0)
            clusters[C] = base * (1 + offset)
    return clusters

# ===========================
# Step 9: composition targets
# ===========================

def composition_targets(current_lottery_name, avg_hot, avg_warm, avg_cold,
                        Hot_set, Warm_set, Cold_set):
    draw_size = main_draw_size(current_lottery_name)
    h_target = round(draw_size * avg_hot)
    w_target = round(draw_size * avg_warm)
    c_target = draw_size - h_target - w_target

    if HWC_OVERRIDE is not None:
        try:
            h_o, w_o, c_o = HWC_OVERRIDE
            h_target = int(h_o)
            w_target = int(w_o)
            c_target = int(c_o)
        except Exception:
            # If override is malformed, ignore it and keep learned targets.
            pass

    h_target = max(0, h_target)
    w_target = max(0, w_target)
    c_target = max(0, c_target)

    h_target = min(h_target, len(Hot_set))
    w_target = min(w_target, len(Warm_set))
    c_target = min(c_target, len(Cold_set))

    h, w, c = h_target, w_target, c_target

    def total():
        return h + w + c

    while total() < draw_size:
        cap_hot = len(Hot_set) - h
        cap_warm = len(Warm_set) - w
        cap_cold = len(Cold_set) - c
        caps = {
            "Warm": cap_warm,
            "Hot": cap_hot,
            "Cold": cap_cold,
        }
        cat = max(caps.items(), key=lambda kv: kv[1])[0]
        if caps[cat] <= 0:
            break
        if cat == "Warm":
            w += 1
        elif cat == "Hot":
            h += 1
        else:
            c += 1

    h_target, w_target, c_target = h, w, c

    if h_target > len(Hot_set) or w_target > len(Warm_set) or c_target > len(Cold_set):
        raise RuntimeError("Composition targets exceed category capacities")

    return h_target, w_target, c_target, draw_size

# ===========================
# Step 10: Monte Carlo sampling (V4.1)
# ===========================

def sample_from_category(category_list, probs_dict, k, chosen_set):
    """
    Sample up to k numbers from a category without replacement.
    If we run out of candidates, we just return what we have and let
    the caller decide whether to use/skip that trial.
    """
    selected = []
    while len(selected) < k:
        candidates = [n for n in category_list if n not in chosen_set and n not in selected]
        if not candidates:
            # Not enough remaining candidates to fulfil k – return partial.
            break
        weights = [probs_dict.get(n, 0.0) for n in candidates]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0] * len(candidates)
            total_w = float(len(candidates))
        r = random.random() * total_w
        acc = 0.0
        for n, w in zip(candidates, weights):
            acc += w
            if acc >= r:
                selected.append(n)
                break
    return selected

def monte_carlo_sampling(P, Hot_set, Warm_set, Cold_set,
                         h_target, w_target, c_target, draw_size,
                         clusters, M, avg_hot, avg_warm, avg_cold,
                         centre_score, top_n):
    # --- 1. TRIALS AND EXPLORE_FRAC UNLOCK ---
    complexity = math.sqrt(max(1.0, M / 20.0))
    structure_conf = max(abs(avg_hot - 1 / 3), abs(avg_warm - 1 / 3), abs(avg_cold - 1 / 3))
    trials_factor = 1.0 + STRUCT_CONFIDENCE_WEIGHT * structure_conf
    TRIALS = int(clamp(BASE_TRIALS * complexity * trials_factor, MIN_TRIALS, MAX_TRIALS))

    max_avg = max(avg_hot, avg_warm, avg_cold)

    # FIX: Prioritize config override to bypass the 0.10 clamp
    if PREDICTION_CONFIG and "EXPLORE_FRAC" in PREDICTION_CONFIG:
        EXPLORE_FRAC = float(PREDICTION_CONFIG["EXPLORE_FRAC"])
    else:
        EXPLORE_FRAC = clamp(0.05 + 0.10 * max(0.0, max_avg - 1 / 3), 0.05, 0.10)

    COLD_EXPLORE_MULT = 1.35
    CLUSTER_LAMBDA = CLUSTER_LAMBDA_BASE

    # --- 2. SPEED PATCH: PRE-CALCULATIONS ---
    # Move all repetitive lookups outside the 100,000+ trial loop
    dec_mults = PREDICTION_CONFIG.get("DECADE_FACTORS_OVERRIDE", {})

    # Pre-calculate combined weight (Decade + Base Probability)
    # This is where your 2.2x, 3.0x etc. decade boosts are applied
    W_base = {}
    for n in NUMBER_RANGE:
        d_id = decade_of(n)
        d_weight = dec_mults.get(d_id, 1.0) if d_id is not None else 1.0
        W_base[n] = P.get(n, 1e-6) * d_weight

    # Pre-calculate Cluster Boosts
    clusters_by_n = {n: [] for n in NUMBER_RANGE}
    for C, priority in clusters.items():
        for n in C:
            clusters_by_n[n].append((C, priority))

    cluster_boost_map = {}
    for n in NUMBER_RANGE:
        C_n = clusters_by_n.get(n, [])
        if not C_n:
            cluster_boost_map[n] = 1.0
        else:
            S_n = sum(priority - 1.0 for (_, priority) in C_n)
            cluster_boost_map[n] = math.exp(CLUSTER_LAMBDA * S_n)

    # Pre-lists for sampling
    Hot_list, Warm_list, Cold_list = list(Hot_set), list(Warm_set), list(Cold_set)
    cluster_items = list(clusters.items())
    freq = Counter()
    sum_hot = sum_warm = sum_cold = 0.0
    sum_decade = {d_id: 0.0 for d_id in DECADES}
    tuple_centre_bias = {}

    # --- 3. THE OPTIMIZED SIMULATION LOOP ---
    for _ in range(TRIALS):
        u = random.random()
        is_exploring = (u < EXPLORE_FRAC)

        # Apply exploration and cluster boosts
        W = {}
        for n in NUMBER_RANGE:
            w = W_base[n] * cluster_boost_map[n]
            if is_exploring and n in Cold_set:
                w *= COLD_EXPLORE_MULT
            W[n] = w

        # Per-category probabilities (Z-Normalization)
        def get_probs(lst):
            if not lst: return {}
            Z = sum(W[n] for n in lst)
            return {n: W[n] / Z for n in lst} if Z > 0 else {n: 1.0 / len(lst) for n in lst}

        p_H = get_probs(Hot_list)
        p_W = get_probs(Warm_list)
        p_C = get_probs(Cold_list)

        chosen = set()
        h_rem, w_rem, c_rem = h_target, w_target, c_target

        # Cluster-first sampling
        if cluster_items and random.random() < CLUSTER_TRIAL_FRAC:
            total_pr = sum(pr for (_, pr) in cluster_items)
            r = random.random() * total_pr
            acc = 0.0
            for C, pr in cluster_items:
                acc += pr
                if acc >= r:
                    for n in C:
                        if n in chosen: continue
                        if n in Hot_set and h_rem > 0:
                            chosen.add(n); h_rem -= 1
                        elif n in Warm_set and w_rem > 0:
                            chosen.add(n); w_rem -= 1
                        elif n in Cold_set and c_rem > 0:
                            chosen.add(n); c_rem -= 1
                    break

        # Category sampling fill
        if h_rem > 0: chosen.update(sample_from_category(Hot_list, p_H, h_rem, chosen))
        if w_rem > 0: chosen.update(sample_from_category(Warm_list, p_W, w_rem, chosen))
        if c_rem > 0: chosen.update(sample_from_category(Cold_list, p_C, c_rem, chosen))

        if len(chosen) != draw_size: continue

        T = tuple(sorted(chosen))
        freq[T] += 1

        # Stats tracking
        sum_hot += sum(1 for n in chosen if n in Hot_set)
        sum_warm += sum(1 for n in chosen if n in Warm_set)
        sum_cold += sum(1 for n in chosen if n in Cold_set)
        for n in chosen:
            d_id = decade_of(n)
            if d_id is not None: sum_decade[d_id] += 1
        tuple_centre_bias[T] = mean(centre_score[n] for n in chosen)

    # --- 4. RANKING AND DIAGNOSTICS ---
    # (Same as your original logic, using the now-accurate TRIALS and EXPLORE_FRAC)
    prob = {T: c / TRIALS for T, c in freq.items()}

    # Final Tuple Scoring (Centre + Cluster bias)
    cluster_scores = {T: sum(prio for C, prio in clusters.items() if set(T).issuperset(C)) for T in prob}
    max_cs = max(cluster_scores.values()) if cluster_scores else 0

    tuple_score = {}
    for T, p in prob.items():
        cb = tuple_centre_bias.get(T, 0.5)
        f_centre = max(0.5, 1.0 + ALPHA_CENTRE * (cb - 0.5))
        f_cluster = 1.0 + BETA_CLUSTER * (cluster_scores[T] / max_cs if max_cs > 0 else 0)
        tuple_score[T] = p * f_centre * f_cluster

    topN_tuples = sorted(prob.keys(), key=lambda x: tuple_score[x], reverse=True)[:top_n]

    return {
        "TRIALS": TRIALS,
        "EXPLORE_FRAC": EXPLORE_FRAC,
        "hot_pred": sum_hot / TRIALS, "warm_pred": sum_warm / TRIALS, "cold_pred": sum_cold / TRIALS,
        "dec_pred": {d: v / TRIALS for d, v in sum_decade.items()},
        "topN": [(T, tuple_score[T], prob[T]) for T in topN_tuples]
    }

# ===========================
# Step 12: learning feedback
# ===========================

def learning_step(state: LearningState,
                  Hot_set, Warm_set, Cold_set,
                  f_main,
                  actual_main):
    hot_actual  = sum(1 for n in actual_main if n in Hot_set)
    warm_actual = sum(1 for n in actual_main if n in Warm_set)
    cold_actual = sum(1 for n in actual_main if n in Cold_set)

    f_values = [f_main[n] for n in NUMBER_RANGE]
    var_f = variance_population(f_values)
    learning_rate = clamp(0.02 + 0.02 * var_f, 0.02, 0.10)

    return {
        "hot_actual": hot_actual,
        "warm_actual": warm_actual,
        "cold_actual": cold_actual,
        "learning_rate": learning_rate,
    }

def apply_learning(state: LearningState,
                   hot_actual, cold_actual,
                   hot_pred, cold_pred,
                   learning_rate,
                   clusters, actual_main):
    # category learning
    hot_error  = hot_actual  - hot_pred
    cold_error = cold_actual - cold_pred

    delta_hot_step  = clamp(sign(hot_error)  * learning_rate * abs(hot_error),  -0.1, 0.1)
    delta_cold_step = clamp(sign(cold_error) * learning_rate * abs(cold_error), -0.1, 0.1)

    state.delta_hot  += delta_hot_step
    state.delta_cold += delta_cold_step
    state.delta_warm  = - (state.delta_hot + state.delta_cold) / 2.0

    state.delta_hot  = clamp(state.delta_hot,  -0.5, 0.5)
    state.delta_warm = clamp(state.delta_warm, -0.5, 0.5)
    state.delta_cold = clamp(state.delta_cold, -0.5, 0.5)

    # cluster learning
    actual_set = set(actual_main)
    for C in clusters.keys():
        if all(n in actual_set for n in C):
            new_val = state.cluster_priority_score_global.get(C, 0.0) + 0.05
        else:
            new_val = state.cluster_priority_score_global.get(C, 0.0) - 0.02
        new_val = clamp(new_val, -0.5, 0.5)
        state.cluster_priority_score_global[C] = new_val



# ===========================
# Per-target full processing
# ===========================

def process_target(lottery_name, target_date,
                   window_cat,
                   state: LearningState,
                   do_learning=True,
                   top_n=TOP_N_PREDICTIONS):
    built = build_windows(target_date, window_cat)
    if built is None:
        return None
    seed_dates_cat, seed_draws_cat = built

    # Step 4: frequencies
    f_main, f_supp, L_counts, last_main_date, M, seed_numbers_main = step_B_category(seed_draws_cat, seed_dates_cat)
    # NEW: short-window decade suppression/boost from last RECENT_DECADE_DAYS
    decade_weight_log, dec_short_counts, dec_factors = compute_recent_decade_weights(
        target_date, RECENT_DECADE_DAYS
    )

    cross_hop_score, cross_hop_log = cross_lottery_hop(seed_draws_cat, window_cat)

    # Step 5: H/W/C
    Hot_set, Warm_set, Cold_set, M_val, K = classify_hot_warm_cold(f_main, L_counts)
    avg_hot, avg_warm, avg_cold, bias = compute_category_bias(seed_draws_cat, Hot_set, Warm_set, Cold_set)

    # Step 6: category weights
    hot_w, warm_w, cold_w = category_weights(avg_hot, avg_warm, avg_cold, state)

    # Step 7: per-number log scores => P
    P, log_score, cold_resurge_score = per_number_log_scores(
        seed_numbers_main, f_main, f_supp, L_counts,
        last_main_date, target_date, lottery_name,
        Hot_set, Warm_set, Cold_set,
        hot_w, warm_w, cold_w,
        decade_weight_log, cross_hop_log,
        window_cat
    )

    # V4: centre score
    centre_score = compute_centre_score(f_main)

    # Step 8: clusters
    clusters = detect_clusters(seed_draws_cat, state)

    # Step 9: composition
    h_target, w_target, c_target, draw_size = composition_targets(
        lottery_name, avg_hot, avg_warm, avg_cold,
        Hot_set, Warm_set, Cold_set
    )


    # Step 10: Monte Carlo
    mc_diag = monte_carlo_sampling(
        P, Hot_set, Warm_set, Cold_set,
        h_target, w_target, c_target, draw_size,
        clusters, M_val, avg_hot, avg_warm, avg_cold,
        centre_score, top_n
    )

    hot_pred = mc_diag["hot_pred"]
    warm_pred = mc_diag["warm_pred"]
    cold_pred = mc_diag["cold_pred"]
    dec_pred = mc_diag["dec_pred"]
    topN = mc_diag["topN"]

    result = {
        "seed_dates_cat": seed_dates_cat,
        "Hot_set": Hot_set,
        "Warm_set": Warm_set,
        "Cold_set": Cold_set,
        "avg_hot": avg_hot,
        "avg_warm": avg_warm,
        "avg_cold": avg_cold,
        "bias": bias,
        "hot_w": hot_w,
        "warm_w": warm_w,
        "cold_w": cold_w,
        "h_target": h_target,
        "w_target": w_target,
        "c_target": c_target,
        "TRIALS": mc_diag["TRIALS"],
        "EXPLORE_FRAC": mc_diag["EXPLORE_FRAC"],
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "dec_pred": dec_pred,
        "topN": topN,
        "dec_short_counts": dec_short_counts,   # how many hits per decade in last RECENT_DECADE_DAYS
        "dec_factors": dec_factors,             # final decade multipliers applied
        "cross_hop_score": cross_hop_score,
        "cold_resurge_score": cold_resurge_score,
        "f_main": f_main,
        "clusters": clusters,
        "P": P,
        "log_score": log_score,
    }

    if not do_learning:
        return result

    # Step 12 learning
    actual_main = get_actual_main(lottery_name, target_date)
    if actual_main is None:
        return result

    learn_meta = learning_step(state, Hot_set, Warm_set, Cold_set, f_main, actual_main)
    hot_actual = learn_meta["hot_actual"]
    cold_actual = learn_meta["cold_actual"]
    learning_rate = learn_meta["learning_rate"]

    apply_learning(
        state,
        hot_actual, cold_actual,
        hot_pred, cold_pred,
        learning_rate,
        clusters, actual_main,
    )

    return result

# ===========================
# Calibration 0C-1 and 0C-2
# ===========================

def calibration_category_window():
    best_W = None
    best_mse = None
    for W_cat in WINDOW_SIZE_CANDIDATES:
        state = LearningState()
        errors = []
        for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
            res = process_target(lottery_name, target_date,
                                 W_cat, state,
                                 do_learning=False)
            if res is None:
                continue
            actual_main = get_actual_main(lottery_name, target_date)
            if actual_main is None:
                continue
            Hot_set = res["Hot_set"]
            Cold_set = res["Cold_set"]
            hot_actual = sum(1 for n in actual_main if n in Hot_set)
            cold_actual = sum(1 for n in actual_main if n in Cold_set)
            hot_pred = res["hot_pred"]
            cold_pred = res["cold_pred"]
            errors.append((hot_actual - hot_pred) ** 2 + (cold_actual - cold_pred) ** 2)
        if not errors:
            continue
        mse = sum(errors) / len(errors)
        if best_mse is None or mse < best_mse or (mse == best_mse and (best_W is None or W_cat < best_W)):
            best_mse = mse
            best_W = W_cat
    if best_W is None:
        raise RuntimeError("No feasible WINDOW_SIZE_CAT found during calibration")
    return best_W, best_mse

# ============================================================
# Override suggestion helpers (MAPPING ONLY / NO MEANS)
# ============================================================

def _pick_mode(counter: Counter):
    """Return most frequent item from Counter. Deterministic tie-break by key."""
    if not counter:
        return None
    return max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]

def _decade_shape_key_from_mc(dec_pred: dict):
    """
    Discrete MODEL decade signature from MC averages.
    Uses rank order (robust to float noise).
    Key = (top2_decades_by_mc, bottom1_decade_by_mc)
    """
    order = tuple(sorted(dec_pred.keys(), key=lambda d: dec_pred[d], reverse=True))
    top2 = order[:2]
    low1 = order[-1]
    return (top2, low1)

def _decade_signature_from_actual(dec_a: dict):
    """
    Discrete ACTUAL decade signature from actual decade counts.
    Signature = (dominant_decades, missing_decades)
      - dominant_decades: decade(s) with max count (can be multiple)
      - missing_decades: decades with count == 0
    NOTE: Your decade_counts() already includes zeros for all decades. Good.
    """
    if not dec_a:
        return (tuple(), tuple())

    maxv = max(dec_a.values())
    dominant = tuple(sorted([d for d, v in dec_a.items() if v == maxv and maxv > 0]))
    missing  = tuple(sorted([d for d, v in dec_a.items() if v == 0]))
    return (dominant, missing)

def suggest_overrides_from_mappings(
    *,
    model_hwc: tuple,              # (h_target, w_target, c_target) for FINAL prediction
    model_dec_pred: dict,          # dec_pred for FINAL prediction
    hwc_map: dict,                 # dict[model_hwc] -> Counter(actual_hwc)
    dec_map: dict,                 # dict[model_dec_key] -> Counter(actual_dec_sig)
    boost_factor: float = 1.35,    # you can tweak
    suppress_factor: float = 0.70  # you can tweak
):
    """
    Pure mapping (mode). NO means, NO averaging.
    Returns:
      - suggested_hwc_override (tuple)
      - suggested_decade_factors_override (dict)
      - evidence dict for printing
    """
    # --- HWC ---
    hwc_counter = hwc_map.get(model_hwc, Counter())
    suggested_hwc = _pick_mode(hwc_counter)

    # If we have no mapping for this model_hwc, do not override it
    if suggested_hwc is None:
        suggested_hwc = model_hwc

    # --- Decades ---
    dec_key = _decade_shape_key_from_mc(model_dec_pred)
    dec_counter = dec_map.get(dec_key, Counter())
    chosen_sig = _pick_mode(dec_counter)  # (dominant, missing)

    # Default neutral factors
    decades = sorted(model_dec_pred.keys())
    dec_factors = {d: 1.00 for d in decades}

    # Convert signature -> factors
    if chosen_sig is not None:
        dominant, missing = chosen_sig
        for d in dominant:
            if d in dec_factors:
                dec_factors[d] = float(boost_factor)
        for d in missing:
            if d in dec_factors:
                dec_factors[d] = float(suppress_factor)

    evidence = {
        "model_hwc": model_hwc,
        "hwc_counts": dict(hwc_counter),

        "dec_key": dec_key,
        "dec_counts": {k: v for k, v in dec_counter.items()},
        "chosen_sig": chosen_sig,
    }

    return suggested_hwc, dec_factors, evidence

def _dec_state_from_counts(dec_counts: dict, draw_size: int):
    """
    Returns a robust state summary for cross-lottery decade tape.
    - clusters: decades with count >= 2 (works for 6 and 7)
    - heavy: decades with count >= 3
    - missing: decades with 0
    - dominant: decade(s) with max count (ties allowed)
    - shape: sorted counts desc, e.g. (3,2,1,0,0)
    - label: DIFFUSE / ONECLUSTER / TWOCLUSTER / HEAVY
    """
    keys = sorted(dec_counts.keys())
    vec = [dec_counts.get(d, 0) for d in keys]
    mx = max(vec) if vec else 0
    dominant = tuple(d for d in keys if dec_counts.get(d, 0) == mx)

    clusters = tuple(d for d in keys if dec_counts.get(d, 0) >= 2)
    heavy = tuple(d for d in keys if dec_counts.get(d, 0) >= 3)
    missing = tuple(d for d in keys if dec_counts.get(d, 0) == 0)

    shape = tuple(sorted(vec, reverse=True))

    if heavy:
        label = "HEAVY"
    elif len(clusters) >= 2:
        label = "TWOCLUSTER"
    elif len(clusters) == 1:
        label = "ONECLUSTER"
    else:
        label = "DIFFUSE"

    return {
        "dominant": dominant,
        "clusters": clusters,
        "heavy": heavy,
        "missing": missing,
        "shape": shape,
        "label": label,
    }


def build_cross_lottery_decade_transitions(tape_rows):
    """
    tape_rows: list of dicts with:
      {
        "lottery": str,
        "date": str,
        "dec": {1:...,2:...,3:...,4:...,5:...},
        "draw_size": int
      }
    Returns transition tables.
    """
    trans_state = defaultdict(Counter)      # label -> next_label
    trans_dom = defaultdict(Counter)        # dominant decade -> next dominant decade
    trans_cluster = defaultdict(Counter)    # cluster decade -> next cluster decade
    trans_missing = defaultdict(Counter)    # missing decade -> next missing decade

    states = []
    for r in tape_rows:
        st = _dec_state_from_counts(r["dec"], r["draw_size"])
        states.append(st)

    for i in range(len(tape_rows) - 1):
        st = states[i]
        st2 = states[i + 1]

        trans_state[st["label"]][st2["label"]] += 1

        # dominant movement
        for d in st["dominant"]:
            for d2 in st2["dominant"]:
                trans_dom[d][d2] += 1

        # cluster movement (>=2)
        for c in st["clusters"]:
            for c2 in st2["clusters"]:
                trans_cluster[c][c2] += 1

        # missing persistence/rotation
        for m in st["missing"]:
            for m2 in st2["missing"]:
                trans_missing[m][m2] += 1

    return {
        "trans_state": trans_state,
        "trans_dom": trans_dom,
        "trans_cluster": trans_cluster,
        "trans_missing": trans_missing,
        "states": states,
    }


def suggest_next_decade_bias_from_transitions(tape_rows, prediction_draw_size=6):
    """
    Prints a cross-lottery suggestion for the NEXT draw, based only on actual decade tape transitions.
    Outputs:
      - Last actual state
      - Most likely next state label
      - Likely dominant move(s) and cluster drift (adjacent / same / jump)
      - Suggested decades to slightly boost/suppress (wrapper override guidance)
    """
    if len(tape_rows) < 3:
        print("\n[DECADE SUGGESTION] Not enough tape rows to learn transitions.")
        return None

    T = build_cross_lottery_decade_transitions(tape_rows)

    # ============================
    # DEBUG: DECADE TAPE & TRANSITIONS
    # ============================

    print("\n[DEBUG] Learned tape order:")
    for r in tape_rows:
        print(f"  {r['date']}  {r['lottery']}  {r['dec']}")


    trans_state = T["trans_state"]
    trans_dom = T["trans_dom"]
    trans_cluster = T["trans_cluster"]
    trans_missing = T["trans_missing"]
    states = T["states"]

    last = tape_rows[-1]
    last_state = states[-1]

    # 1) Predict next state label
    next_label = None
    if trans_state.get(last_state["label"]):
        next_label = trans_state[last_state["label"]].most_common(1)[0][0]

    # 2) Predict next dominant decade(s) via dominant transition
    # If dominant is tied, we vote.
    dom_votes = Counter()
    for d in last_state["dominant"]:
        if trans_dom.get(d):
            dom_votes.update(trans_dom[d])

    predicted_dominant = tuple([dom_votes.most_common(1)[0][0]]) if dom_votes else ()

    # 3) Predict next cluster decade(s) via cluster transition
    cl_votes = Counter()
    for c in last_state["clusters"]:
        if trans_cluster.get(c):
            cl_votes.update(trans_cluster[c])

    predicted_clusters = tuple([x for x, _ in cl_votes.most_common(2)]) if cl_votes else ()

    # 4) Predict missing decades stickiness/rotation
    miss_votes = Counter()
    for m in last_state["missing"]:
        if trans_missing.get(m):
            miss_votes.update(trans_missing[m])

    predicted_missing = ()
    if miss_votes:
        top2 = miss_votes.most_common(2)
        # require at least 2 total votes AND a strict winner
        if sum(miss_votes.values()) >= 2 and (len(top2) == 1 or top2[0][1] > top2[1][1]):
            predicted_missing = (top2[0][0],)

    def _move_kind(a, b):
        if a == b:
            return "same"
        if abs(a - b) == 1:
            return "adjacent"
        return "jump"

    # Movement diagnostics for dominant
    dom_move_diag = []
    for d in last_state["dominant"]:
        if predicted_dominant:
            dom_move_diag.append((d, predicted_dominant[0], _move_kind(d, predicted_dominant[0])))

    # 5) Convert to a very light “bias” suggestion (NOT hard enforcement)
    # We keep it mild to avoid overfit:
    # - boost predicted dominant/cluster decades slightly
    # - suppress predicted missing slightly
    # - keep others 1.00
    factors = {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00}

    for d in predicted_dominant:
        factors[d] = max(factors[d], 1.15)
    for d in predicted_clusters:
        factors[d] = max(factors[d], 1.10)

    if last_state["label"] == "TWOCLUSTER" and len(last_state["clusters"]) == 2:
        a, b = sorted(last_state["clusters"])
        if abs(a - b) == 2:
            mid = (a + b) // 2
            factors[mid] = max(factors[mid], 1.12)

    protected = set(predicted_dominant) | set(predicted_clusters)

    for d in predicted_missing:
        if d in protected:
            continue  # never suppress what we also boost
        factors[d] = min(factors[d], 0.90)

    return {
        "next_label": next_label,
        "predicted_dominant": predicted_dominant,
        "predicted_clusters": predicted_clusters,
        "predicted_missing": predicted_missing,
        "factors": factors,
    }

import datetime

def build_targets_for_learning():
    """
    Build TARGET_DRAWS_FOR_LEARNING dynamically from PREDICTION_TARGET.
    Range: [prediction_date - 14 days, prediction_date - 1 day] inclusive.
    Uses cross-lottery schedule and skips missing-data dates.
    """
    pred_lottery, pred_date, _ = PREDICTION_TARGET

    finalize_data()

    start_date = pred_date - datetime.timedelta(days=7)
    end_date   = pred_date - datetime.timedelta(days=1)

    # Day-of-week schedule (AU typical):
    # Mon=0 Tue=1 Wed=2 Thu=3 Fri=4 Sat=5 Sun=6
    def lottery_for_date(d: datetime.date):
        wd = d.weekday()
        if wd == 5:
            return "Saturday Lotto"
        if wd in (0, 2, 4):
            return "Weekday Windfall"
        if wd == 1:
            return "OZ Lotto"
        if wd == 3:
            return "Powerball"
        return None  # Sunday / no draw

    targets = []
    d = start_date
    while d <= end_date:
        lot = lottery_for_date(d)
        if lot is not None:
            # Only include if data exists (prevents "incomplete window" surprises)
            if get_actual_main(lot, d) is not None:
                targets.append((lot, d))
        d += datetime.timedelta(days=1)
    return targets

def exact_hits(pred_nums, actual_set):
    return sum(1 for n in pred_nums if n in actual_set)

def near_miss_pm1_hits(pred_nums, actual_set):
    pm1 = set()
    for x in actual_set:
        pm1.add(x - 1)
        pm1.add(x + 1)
    # Count numbers within ±1 but not exact
    return sum(1 for n in pred_nums if (n in pm1) and (n not in actual_set))

def print_actual_number_scores(actual_main, log_score, P,res_pred, title="ACTUAL numbers (with score)"):
    """
    Prints log_score and P for each actual number.
    Works if log_score/P are dicts or list-like indexed by number.
    """
    if actual_main is None:
        return

    actual_sorted = sorted(actual_main)
    print(f"\n{title}")
    ps = []
    print(f"\n=== RANKS of {title} ===")
    ranked = sorted(NUMBER_RANGE, key=lambda n: P[n], reverse=True)
    for r, n in enumerate(ranked[:MAIN_NUMBER_MAX], 1):
        if n in actual_sorted:
            d = decade_of(n)
            cat = _cat_of(n, res_pred)
            p = P[n]
            ps.append(p)
            print(f"  r{r:02d}  {n:2d}  D{d}  {cat}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    print(f"\n  P_sum={sum(ps):.8f}  P_mean={sum(ps)/len(ps):.8f}  P_min={min(ps):.8f}  P_max={max(ps):.8f}")

def print_actual_if_exists(lottery_name, date, Hot_set, Warm_set, Cold_set):
    """
    Returns sorted actual main list if found, else None.
    Also prints Actual main + H/W/C + decades.
    """
    if RUNNING_PB_ONLY:
        actual = get_actual_pb(lottery_name, date)

        if actual is None:
            print("\n[ACTUAL] No actual draw found in GLOBAL_DRAWS for this prediction date.")
            return None

        actual_sorted = sorted(actual)
        h_a, w_a, c_a = hwc_counts(actual_sorted, Hot_set, Warm_set, Cold_set)
        dec_a = decade_counts(actual_sorted)

        print("\n=== ACTUAL RESULT (FOUND IN GLOBAL_DRAWS) ===")
        print(f"Lottery: {lottery_name}")
        print(f"Date:    {date}")
        print(f"Actual powerball: {actual_sorted}")
        print(f"Actual H/W/C: H={h_a}, W={w_a}, C={c_a}")
        print(f"Actual decades: {dec_a}")

        return actual_sorted
    else:
        actual = get_actual_main(lottery_name, date)

        if actual is None:
            print("\n[ACTUAL] No actual draw found in GLOBAL_DRAWS for this prediction date.")
            return None

        actual_sorted = sorted(actual)
        h_a, w_a, c_a = hwc_counts(actual_sorted, Hot_set, Warm_set, Cold_set)
        dec_a = decade_counts(actual_sorted)

        print("\n=== ACTUAL RESULT (FOUND IN GLOBAL_DRAWS) ===")
        print(f"Lottery: {lottery_name}")
        print(f"Date:    {date}")
        print(f"Actual main: {actual_sorted}")
        print(f"Actual H/W/C: H={h_a}, W={w_a}, C={c_a}")
        print(f"Actual decades: {dec_a}")

        return actual_sorted

def _cat_of(n: int,res_pred=[]) -> str:
    if n in res_pred["Hot_set"]:
        return "H"
    if n in res_pred["Warm_set"]:
        return "W"
    if n in res_pred["Cold_set"]:
        return "C"
    return "?"

def _rank_map_from_P(P):
    ranked = sorted(NUMBER_RANGE, key=lambda n: P[n], reverse=True)
    return {n: r for r, n in enumerate(ranked, 1)}

def _build_actual_draw_snapshot(lottery_name, target_date, actual_main, res):
    """
    Build per-number metadata for an actual draw from res (P/log_score + sets).
    """
    if actual_main is None:
        return None
    P = res["P"]
    log_score = res["log_score"]
    rank_map = _rank_map_from_P(P)
    number_meta = {}
    for n in actual_main:
        number_meta[n] = {
            "P": P[n],
            "log_score": log_score[n],
            "category": _cat_of(n, res),
            "decade": decade_of(n),
            "rank": rank_map[n],
        }
    return {
        "lottery": lottery_name,
        "date": target_date,
        "draw_size": len(actual_main),
        "actual_numbers": list(actual_main),
        "number_meta": number_meta,
    }

def _build_prediction_snapshot(lottery_name, target_date, res):
    P = res["P"]
    log_score = res["log_score"]
    rank_map = _rank_map_from_P(P)
    numbers = []
    for n in NUMBER_RANGE:
        numbers.append({
            "n": n,
            "rank": rank_map[n],
            "P": P[n],
            "log_score": log_score[n],
            "category": _cat_of(n, res),
            "decade": decade_of(n),
        })
    return {
        "lottery": lottery_name,
        "date": target_date,
        "draw_size": main_draw_size(lottery_name),
        "numbers": numbers,
    }

def build_last_n_targets(lottery_name, pred_date, n):
    draws = [
        d for d in global_draws
        if d.lottery == lottery_name and d.date < pred_date and d.main
    ]
    draws.sort(key=lambda d: d.date, reverse=True)
    return [(lottery_name, d.date) for d in draws[:n]]

def _is_adjacent_shape_3(a, b):
    if sum(a) != sum(b):
        return False
    diffs = [abs(a[i] - b[i]) for i in range(3)]
    if max(diffs) > 1:
        return False
    if sum(diffs) == 0:
        return True
    return sum(diffs) == 2 and diffs.count(1) == 2 and diffs.count(0) == 1

def _is_adjacent_shape_vec(a, b):
    if sum(a) != sum(b):
        return False
    diffs = [abs(a[i] - b[i]) for i in range(len(a))]
    if max(diffs) > 1:
        return False
    if sum(diffs) == 0:
        return True
    return sum(diffs) == 2 and diffs.count(1) == 2

def _choose_anchor_shape(shapes, is_adjacent_fn):
    counts = {}
    for s in shapes:
        counts[s] = counts.get(s, 0) + 1
    best_shape = None
    best_support = None
    best_exact = None
    for anchor in counts:
        support = sum(cnt for s, cnt in counts.items() if is_adjacent_fn(anchor, s))
        exact = counts[anchor]
        if (best_support is None or support > best_support or
            (support == best_support and (best_exact is None or exact > best_exact)) or
            (support == best_support and exact == best_exact and anchor < best_shape)):
            best_shape = anchor
            best_support = support
            best_exact = exact
    evidence = [(s, counts[s]) for s in counts if is_adjacent_fn(best_shape, s)]
    evidence.sort(key=lambda x: (-x[1], x[0]))
    return best_shape, evidence

def _shape_to_decade_dict(shape, decade_ids):
    return {d: shape[i] for i, d in enumerate(decade_ids)}

def _build_relaxed_decade_shapes(target_shape, decade_ids, decade_counts):
    shapes = [target_shape]
    # Order required decades by scarcity in the candidate pool.
    required = []
    for i, d in enumerate(decade_ids):
        if target_shape[i] > 0:
            required.append((decade_counts.get(d, 0), i))
    required.sort(key=lambda x: (x[0], x[1]))
    # Order destination decades by abundance.
    dest_order = sorted(range(len(decade_ids)),
                        key=lambda i: (-decade_counts.get(decade_ids[i], 0), i))
    for _, src_idx in required:
        if target_shape[src_idx] <= 0:
            continue
        src_count = target_shape[src_idx]
        for dst_idx in dest_order:
            if dst_idx == src_idx:
                continue
            shape = list(target_shape)
            shape[dst_idx] += src_count
            shape[src_idx] = 0
            shapes.append(tuple(shape))
        # Stop after first (rarest) decade is dropped to keep relaxation minimal.
        break
    # Deduplicate while preserving order.
    seen = set()
    ordered = []
    for s in shapes:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered

def _predict_median(values):
    return float(median(values))

def _predict_trimmed_mean(values):
    if len(values) <= 2:
        return float(sum(values) / len(values))
    vals = sorted(values)[1:-1]
    return float(sum(vals) / len(vals))

def _predict_weighted_mean(values):
    weights = list(range(1, len(values) + 1))
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

def _predict_trend_damped(values, alpha=0.5):
    n = len(values)
    if n == 1:
        return float(values[0])
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        trend = float(values[-1])
    else:
        slope = num / den
        intercept = mean_y - slope * mean_x
        trend = intercept + slope * n
    return alpha * trend + (1.0 - alpha) * float(values[-1])

def _choose_auto_predictor(rows, *, metric, eval_last_n=3):
    predictors = {
        "median": _predict_median,
        "trimmed_mean": _predict_trimmed_mean,
        "weighted_mean": _predict_weighted_mean,
        "trend_damped": _predict_trend_damped,
    }
    if len(rows) < 2:
        return _predict_median
    # Evaluate on the last N rows using only past rows for each prediction.
    n_rows = len(rows)
    start_idx = max(1, n_rows - eval_last_n)
    best_name = None
    best_err = None
    for name, fn in predictors.items():
        errs = []
        for i in range(start_idx, n_rows):
            past = rows[:i]
            vals = [r[metric] for r in past]
            if not vals:
                continue
            pred = fn(vals)
            errs.append(abs(pred - rows[i][metric]))
        if not errs:
            continue
        mean_err = sum(errs) / len(errs)
        if best_err is None or mean_err < best_err:
            best_err = mean_err
            best_name = name
    return predictors.get(best_name, _predict_median)

def _filter_learning_draws(draws, *, allowed_dates=None, allowed_lottery=None):
    if allowed_dates is None and allowed_lottery is None:
        return draws
    filtered = []
    for d in draws:
        if allowed_lottery is not None and d["lottery"] != allowed_lottery:
            continue
        if allowed_dates is not None and d["date"] not in allowed_dates:
            continue
        filtered.append(d)
    return filtered

def _compute_cohort_regime(run_data, *, allowed_dates=None, allowed_lottery=None):
    learning_draws = list(run_data.get("locked_regime_draws", []))
    if not learning_draws:
        learning_draws = list(run_data.get("learning_draws", []))
        pred_actual = run_data.get("prediction_actual")
        if pred_actual is not None:
            learning_draws.insert(0, pred_actual)
    learning_draws = _filter_learning_draws(
        learning_draws,
        allowed_dates=allowed_dates,
        allowed_lottery=allowed_lottery,
    )
    if not learning_draws:
        return None
    decade_ids = DECADES
    regime_rows = []
    for d in learning_draws:
        actual_numbers = d["actual_numbers"]
        meta = d["number_meta"]
        ordered = sorted(
            actual_numbers,
            key=lambda n: (-meta[n]["P"], meta[n]["rank"], n),
        )
        leader = ordered[0]
        cohort = ordered[1:]
        cohort_meta = [meta[n] for n in cohort]
        hwc = {"H": 0, "W": 0, "C": 0}
        decades = {dec_id: 0 for dec_id in decade_ids}
        ranks = []
        ps = []
        for m in cohort_meta:
            hwc[m["category"]] += 1
            decades[m["decade"]] += 1
            ranks.append(m["rank"])
            ps.append(m["P"])
        row = {
            "lottery": d["lottery"],
            "date": d["date"],
            "leader": leader,
            "cohort": cohort,
            "hwc": (hwc["H"], hwc["W"], hwc["C"]),
            "decades": decades,
            "rank_min": min(ranks),
            "rank_max": max(ranks),
            "rank_span": max(ranks) - min(ranks),
            "P_sum": sum(ps),
            "P_mean": sum(ps) / len(ps),
            "P_min": min(ps),
            "P_max": max(ps),
        }
        regime_rows.append(row)
    return {
        "rows": regime_rows,
        "decade_ids": decade_ids,
        "K": learning_draws[0]["draw_size"],
        "learning_draws": learning_draws,
        "pred_actual": run_data.get("prediction_actual"),
    }

def _prepare_prediction_numbers(run_data):
    pred = run_data["prediction"]
    numbers = list(pred["numbers"])
    numbers.sort(key=lambda x: x["rank"])
    return {
        "lottery": pred["lottery"],
        "date": pred["date"],
        "K": pred["draw_size"],
        "numbers": numbers,
    }

def _format_decade_dict(decade_ids, dec_dict):
    return "{" + ", ".join(f"D{d}:{dec_dict.get(d, 0)}" for d in decade_ids) + "}"

def print_locked_prediction_steps(
    run_data,
    *,
    leader_pool_rank_max=5,
    max_tickets_to_print=20,
    include_learning_scores=True,
    allowed_dates=None,
    allowed_lottery=None,
    override_cohort_hwc=None,
    override_cohort_decades=None,
    override_rank_min=None,
    override_rank_max=None,
    override_p_min=None,
    override_p_max=None,
):
    regime = _compute_cohort_regime(
        run_data,
        allowed_dates=allowed_dates,
        allowed_lottery=allowed_lottery,
    )
    if regime is None:
        print("\n[LOCKED] No learning draws available to build regime.")
        return
    pred = _prepare_prediction_numbers(run_data)
    pred_actual = regime["pred_actual"]
    learning_draws = regime["learning_draws"]

    print("\n" + "=" * 80)
    print("LOCKED PREDICTION STEPS (COHORT REGIME)")
    print("=" * 80)

    rows = regime["rows"]
    decade_ids = regime["decade_ids"]
    print("\n=== COHORT REGIME TABLE ===")
    for r in rows:
        print(
            f"{r['date']} {r['lottery']} | leader={r['leader']} "
            f"| cohort={sorted(r['cohort'])} "
            f"| HWC={r['hwc']} "
            f"| decades={_format_decade_dict(decade_ids, r['decades'])} "
            f"| rank_min={r['rank_min']} rank_max={r['rank_max']} span={r['rank_span']} "
            f"| P_sum={r['P_sum']:.8f} P_mean={r['P_mean']:.8f} "
            f"P_min={r['P_min']:.8f} P_max={r['P_max']:.8f}"
        )

    # if include_learning_scores and learning_draws:
    #     print("\n=== LEARNING DRAW ACTUAL SCORES ===")
    #     for d in learning_draws:
    #         meta = d["number_meta"]
    #         print(f"{d['date']} {d['lottery']} actual={sorted(d['actual_numbers'])}")
    #         nums = sorted(d["actual_numbers"], key=lambda n: meta[n]["rank"])
    #         ps = []
    #         for n in nums:
    #             m = meta[n]
    #             ps.append(m["P"])
    #             print(
    #                 f"  r{m['rank']:02d} {n:2d} D{m['decade']} {m['category']} "
    #                 f"log_score={m['log_score']:+.6f} P={m['P']:.8f}"
    #             )
    #         if ps:
    #             print(
    #                 f"  P_sum={sum(ps):.8f} P_mean={sum(ps)/len(ps):.8f} "
    #                 f"P_min={min(ps):.8f} P_max={max(ps):.8f}"
    #             )

    hwc_shapes = [r["hwc"] for r in rows]
    target_hwc, hwc_evidence = _choose_anchor_shape(hwc_shapes, _is_adjacent_shape_3)
    if override_cohort_hwc is not None and len(override_cohort_hwc) == 3:
        target_hwc = (int(override_cohort_hwc[0]), int(override_cohort_hwc[1]), int(override_cohort_hwc[2]))
        if sum(target_hwc) != pred["K"] - 1:
            raise ValueError("override_cohort_hwc must sum to K-1")
    print("\n=== TARGET COHORT HWC ===")
    print(f"TARGET_COHORT_HWC = {target_hwc}")
    print("Evidence (shape -> count, adjacent cluster):")
    for s, cnt in hwc_evidence:
        print(f"  {s} -> {cnt}")

    dec_shapes = [tuple(r["decades"][d] for d in decade_ids) for r in rows]
    target_dec_shape, dec_evidence = _choose_anchor_shape(dec_shapes, _is_adjacent_shape_vec)
    target_decades = _shape_to_decade_dict(target_dec_shape, decade_ids)
    if override_cohort_decades is not None:
        target_decades = {int(k): int(v) for k, v in override_cohort_decades.items()}
        target_dec_shape = tuple(target_decades.get(d, 0) for d in decade_ids)
        if sum(target_dec_shape) != pred["K"] - 1:
            raise ValueError("override_cohort_decades must sum to K-1")
    print("\n=== TARGET COHORT DECADES ===")
    print(f"TARGET_COHORT_DECADES = {_format_decade_dict(decade_ids, target_decades)}")
    print("Evidence (template -> count, adjacent cluster):")
    for s, cnt in dec_evidence:
        print(f"  {_format_decade_dict(decade_ids, _shape_to_decade_dict(s, decade_ids))} -> {cnt}")

    rows_sorted = sorted(rows, key=lambda r: r["date"])
    eval_last_n = COHORT_AUTOPRED_EVAL_LAST_N or 3
    rank_min_fn = _choose_auto_predictor(rows_sorted, metric="rank_min", eval_last_n=eval_last_n)
    rank_max_fn = _choose_auto_predictor(rows_sorted, metric="rank_max", eval_last_n=eval_last_n)
    p_min_fn = _choose_auto_predictor(rows_sorted, metric="P_min", eval_last_n=eval_last_n)
    p_max_fn = _choose_auto_predictor(rows_sorted, metric="P_max", eval_last_n=eval_last_n)
    rank_min_star = rank_min_fn([r["rank_min"] for r in rows_sorted])
    rank_max_star = rank_max_fn([r["rank_max"] for r in rows_sorted])
    p_min_star = p_min_fn([r["P_min"] for r in rows_sorted])
    p_max_star = p_max_fn([r["P_max"] for r in rows_sorted])
    rank_span_star = rank_max_star - rank_min_star

    if override_rank_min is not None:
        rank_min_star = int(override_rank_min)
    if override_rank_max is not None:
        rank_max_star = int(override_rank_max)
    if override_p_min is not None:
        p_min_star = float(override_p_min)
    if override_p_max is not None:
        p_max_star = float(override_p_max)

    print("\n=== COHORT ENVELOPES ===")
    print(f"rank_min*={rank_min_star} rank_max*={rank_max_star} rank_span*={rank_span_star}")
    print(f"P_min*={p_min_star:.8f} P_max*={p_max_star:.8f}")

    if pred_actual is not None:
        print("\n[LOCKED] Actual draw exists for this date; skipping ticket construction.")
        return

    max_rank = len(pred["numbers"])
    cohort_pool = [
        n for n in pred["numbers"]
        if rank_min_star <= n["rank"] <= rank_max_star and p_min_star <= n["P"] <= p_max_star
    ]
    leader_pool = [n for n in pred["numbers"] if n["rank"] <= leader_pool_rank_max]

    print("\n=== COHORT CANDIDATE POOL ===")
    for n in cohort_pool:
        print(
            f"  r{n['rank']:02d} {n['n']:2d} D{n['decade']} {n['category']} "
            f"log_score={n['log_score']:+.6f} P={n['P']:.8f}"
        )

    print("\n=== LEADER POOL ===")
    for n in leader_pool:
        print(
            f"  r{n['rank']:02d} {n['n']:2d} D{n['decade']} {n['category']} "
            f"log_score={n['log_score']:+.6f} P={n['P']:.8f}"
        )

    cohort_candidates = [n["n"] for n in cohort_pool]
    meta_map = {n["n"]: n for n in pred["numbers"]}
    target_h, target_w, target_c = target_hwc
    target_dec_tuple = target_dec_shape
    hwc_shapes = [r["hwc"] for r in rows]
    dec_shapes = [tuple(r["decades"][d] for d in decade_ids) for r in rows]
    sfl_pm1 = set()
    if pred["lottery"] != HOP_SOURCE_LOTTERY:
        sfl_pm1 = _sfl_lastn_pm1_set(pred["date"], SFL_LASTN_DRAWS)
    pm1_counts = None
    cohort_candidates_build = cohort_candidates
    leader_pool_build = leader_pool
    if PM1_ONLY_MODE and pred["lottery"] != HOP_SOURCE_LOTTERY:
        pm1_counts = _sfl_lastn_pm1_counts(pred["date"], SFL_LASTN_DRAWS)
        ranked_pm1 = sorted(
            [n for n in NUMBER_RANGE if pm1_counts.get(n, 0) > 0],
            key=lambda n: (pm1_counts.get(n, 0), meta_map[n]["P"]),
            reverse=True,
        )
        pm1_pool = ranked_pm1[:PM1_ONLY_TOP_N]
        if len(pm1_pool) >= pred["K"]:
            cohort_candidates = pm1_pool
            cohort_candidates_build = pm1_pool
            leader_pool_build = [n for n in leader_pool if n["n"] in pm1_pool]
            if not leader_pool_build:
                leader_pool_build = leader_pool
        else:
            pm1_counts = None
    if sfl_pm1 and SFL_PM1_STRICT_COHORT:
        strict_cohort = [n for n in cohort_candidates if n in sfl_pm1]
        if len(strict_cohort) >= pred["K"] - 1:
            cohort_candidates_build = strict_cohort
    if sfl_pm1 and SFL_PM1_STRICT_LEADER:
        strict_leaders = [n for n in leader_pool if n["n"] in sfl_pm1]
        if strict_leaders:
            leader_pool_build = strict_leaders

    def _build_tickets(allowed_hwc_shapes, allowed_dec_shapes, candidates, leaders):
        cohort_combos = []
        for combo in itertools.combinations(candidates, pred["K"] - 1):
            hwc = {"H": 0, "W": 0, "C": 0}
            dec = {d: 0 for d in decade_ids}
            ps = []
            ranks = []
            log_scores = []
            max_p = -1.0
            for n in combo:
                m = meta_map[n]
                hwc[m["category"]] += 1
                dec[m["decade"]] += 1
                ps.append(m["P"])
                ranks.append(m["rank"])
                log_scores.append(m["log_score"])
                max_p = max(max_p, m["P"])
            if allowed_hwc_shapes is not None and (hwc["H"], hwc["W"], hwc["C"]) not in allowed_hwc_shapes:
                continue
            dec_tuple = tuple(dec[d] for d in decade_ids)
            if allowed_dec_shapes is not None and dec_tuple not in allowed_dec_shapes:
                continue
            cohort_combos.append({
                "cohort": combo,
                "hwc": (hwc["H"], hwc["W"], hwc["C"]),
                "decades": dec,
                "rank_min": min(ranks),
                "rank_max": max(ranks),
                "rank_span": max(ranks) - min(ranks),
                "P_sum": sum(ps),
                "P_mean": sum(ps) / len(ps),
                "P_min": min(ps),
                "P_max": max(ps),
                "max_P": max_p,
                "sum_log_score": sum(log_scores),
                "product_P": math.prod(ps),
            })

        tickets = []
        for leader in leaders:
            leader_n = leader["n"]
            for c in cohort_combos:
                if leader_n in c["cohort"]:
                    continue
                if leader["P"] < c["max_P"]:
                    continue
                pm1_hits = sum(1 for n in c["cohort"] if n in sfl_pm1)
                if leader_n in sfl_pm1:
                    pm1_hits += 1
                tickets.append({
                    "leader": leader_n,
                    "leader_rank": leader["rank"],
                    "leader_P": leader["P"],
                    "cohort": c["cohort"],
                    "hwc": c["hwc"],
                    "decades": c["decades"],
                    "rank_min": c["rank_min"],
                    "rank_max": c["rank_max"],
                    "rank_span": c["rank_span"],
                    "P_sum": c["P_sum"],
                    "P_mean": c["P_mean"],
                    "P_min": c["P_min"],
                    "P_max": c["P_max"],
                    "sum_log_score": c["sum_log_score"] + leader["log_score"],
                    "product_P": c["product_P"] * leader["P"],
                    "pm1_hits": pm1_hits,
                    "P_total": c["P_sum"] + leader["P"],
                })
        return tickets

    def _dedup_tickets(items):
        seen = set()
        out = []
        for t in items:
            key = (t["leader"], tuple(sorted(t["cohort"])))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    allowed_hwc_shapes = {target_hwc}
    allowed_dec_shapes = {target_dec_tuple}
    tickets = _build_tickets(allowed_hwc_shapes, allowed_dec_shapes, cohort_candidates_build, leader_pool_build)
    relaxed_candidates = None
    relaxed_hwc_shapes = None
    relaxed_dec_shapes = None
    if max_tickets_to_print:
        # Joint relaxation: widen rank/P envelopes to cover regime rows and
        # expand HWC/decade shapes to the adjacent clusters around the target.
        if override_rank_min is None:
            rank_min_relaxed = min(r["rank_min"] for r in rows)
        else:
            rank_min_relaxed = rank_min_star
        if override_rank_max is None:
            rank_max_relaxed = max(r["rank_max"] for r in rows)
        else:
            rank_max_relaxed = rank_max_star
        if override_p_min is None:
            p_min_relaxed = min(r["P_min"] for r in rows)
        else:
            p_min_relaxed = p_min_star
        if override_p_max is None:
            p_max_relaxed = max(r["P_max"] for r in rows)
        else:
            p_max_relaxed = p_max_star

        rank_min_relaxed = max(1, int(math.floor(rank_min_relaxed)))
        rank_max_relaxed = min(max_rank, int(math.ceil(rank_max_relaxed)))
        p_min_relaxed = max(0.0, float(p_min_relaxed))
        p_max_relaxed = float(p_max_relaxed)

        relaxed_pool = [
            n for n in pred["numbers"]
            if rank_min_relaxed <= n["rank"] <= rank_max_relaxed and p_min_relaxed <= n["P"] <= p_max_relaxed
        ]
        relaxed_candidates = [n["n"] for n in relaxed_pool]

        if override_cohort_hwc is None:
            relaxed_hwc_shapes = {s for s in hwc_shapes if _is_adjacent_shape_3(s, target_hwc)}
            relaxed_hwc_shapes.add(target_hwc)
        else:
            relaxed_hwc_shapes = allowed_hwc_shapes
        if override_cohort_decades is None:
            relaxed_dec_shapes = {s for s in dec_shapes if _is_adjacent_shape_vec(s, target_dec_tuple)}
            relaxed_dec_shapes.add(target_dec_tuple)
        else:
            relaxed_dec_shapes = allowed_dec_shapes

        print(
            "[LOCKED] Joint relaxation available: "
            f"rank[{rank_min_relaxed}..{rank_max_relaxed}] "
            f"P[{p_min_relaxed:.8f}..{p_max_relaxed:.8f}] "
            f"HWC_shapes={len(relaxed_hwc_shapes)} "
            f"dec_shapes={len(relaxed_dec_shapes)}"
        )

        if len(tickets) < max_tickets_to_print:
            tickets = _build_tickets(relaxed_hwc_shapes, relaxed_dec_shapes, relaxed_candidates, leader_pool_build)

    if SFL_PM1_COMBO_ENABLED and sfl_pm1:
        pm1_tickets = _build_pm1_combo_tickets(
            pred,
            meta_map,
            sfl_pm1,
            top_n=SFL_PM1_COMBO_TOP_N,
            max_tickets=(max_tickets_to_print or SFL_PM1_COMBO_MAX),
            decade_ids=decade_ids,
        )
        tickets = _dedup_tickets(tickets + pm1_tickets)

    if TOP_P_COMBO_ENABLED:
        if TOP_P_COMBO_MAX:
            max_top_p = TOP_P_COMBO_MAX
            if max_tickets_to_print:
                max_top_p = max(max_top_p, max_tickets_to_print)
        else:
            max_top_p = max_tickets_to_print
        top_p_tickets = _build_top_p_combo_tickets(
            pred,
            meta_map,
            top_n=TOP_P_COMBO_N,
            max_tickets=max_top_p,
            decade_ids=decade_ids,
        )
        tickets = _dedup_tickets(tickets + top_p_tickets)

    if PM1_WEIGHTED_COMBO_ENABLED and pred["lottery"] != HOP_SOURCE_LOTTERY:
        pm1_counts = _sfl_lastn_pm1_counts(pred["date"], SFL_LASTN_DRAWS)
        weighted_tickets = _build_pm1_weighted_combo_tickets(
            pred,
            meta_map,
            pm1_counts,
            top_n=PM1_WEIGHTED_COMBO_TOP_N,
            max_tickets=(max_tickets_to_print or PM1_WEIGHTED_COMBO_MAX),
            decade_ids=decade_ids,
            score_w=PM1_WEIGHTED_SCORE_W,
        )
        tickets = _dedup_tickets(tickets + weighted_tickets)

    if PM1_GREEDY_ENABLED and pred["lottery"] != HOP_SOURCE_LOTTERY:
        pm1_counts = _sfl_lastn_pm1_counts(pred["date"], SFL_LASTN_DRAWS)
        greedy_tickets = _build_pm1_greedy_tickets(
            pred,
            meta_map,
            pm1_counts,
            top_n=PM1_GREEDY_TOP_N,
            max_tickets=(max_tickets_to_print or PM1_GREEDY_MAX),
            decade_ids=decade_ids,
        )
        tickets = _dedup_tickets(tickets + greedy_tickets)

    tickets.sort(key=lambda t: (t["P_total"], t["sum_log_score"], t["product_P"]), reverse=True)

    print("\n=== TICKETS ===")
    if not tickets:
        print("No valid tickets.")
        return
    # Enforce a per-number usage cap for cohort numbers to diversify tickets.
    usage_cap = None
    if COHORT_USAGE_CAP_FRAC and max_tickets_to_print:
        usage_cap = max(1, math.ceil(max_tickets_to_print * float(COHORT_USAGE_CAP_FRAC)))
    # Hard cap: any number can appear in at most 2 tickets.
    if usage_cap is None:
        usage_cap = 2
    else:
        usage_cap = min(usage_cap, 2)
    if usage_cap is not None:
        usage_counts = Counter()
        diversified = []
        for t in tickets:
            nums = list(t["cohort"])
            if any(usage_counts[n] >= usage_cap for n in nums):
                continue
            diversified.append(t)
            for n in nums:
                usage_counts[n] += 1
            if max_tickets_to_print and len(diversified) >= max_tickets_to_print:
                break

        if (
            max_tickets_to_print
            and len(diversified) < max_tickets_to_print
            and relaxed_candidates is not None
        ):
            print("[LOCKED] Diversification shortfall; expanding ticket pool.")
            expanded = _build_tickets(relaxed_hwc_shapes, relaxed_dec_shapes, relaxed_candidates, leader_pool_build)
            combined = _dedup_tickets(tickets + expanded)
            combined.sort(key=lambda t: (t["P_total"], t["sum_log_score"], t["product_P"]), reverse=True)
            usage_counts = Counter()
            diversified = []
            for t in combined:
                nums = list(t["cohort"])
                if any(usage_counts[n] >= usage_cap for n in nums):
                    continue
                diversified.append(t)
                for n in nums:
                    usage_counts[n] += 1
                if max_tickets_to_print and len(diversified) >= max_tickets_to_print:
                    break
        tickets = diversified if diversified else tickets
    if max_tickets_to_print:
        tickets = tickets[:max_tickets_to_print]
    for i, t in enumerate(tickets, 1):
        cohort_sorted = sorted(t["cohort"])
        cohort_ranks = sorted(meta_map[n]["rank"] for n in cohort_sorted)
        print(
            f"#{i} leader={t['leader']} (r{t['leader_rank']:02d}) "
            f"| cohort={cohort_sorted} "
            f"| cohort_hwc={t['hwc']} "
            f"| cohort_decades={_format_decade_dict(decade_ids, t['decades'])} "
            f"| cohort_Psum={t['P_sum']:.8f} Pmean={t['P_mean']:.8f} "
            f"Pmin={t['P_min']:.8f} Pmax={t['P_max']:.8f} "
            f"| cohort_ranks={cohort_ranks}"
        )

def build_locked_tickets(
    run_data,
    *,
    leader_pool_rank_max=5,
    max_tickets_to_print=20,
    allowed_dates=None,
    allowed_lottery=None,
    override_cohort_hwc=None,
    override_cohort_decades=None,
    override_rank_min=None,
    override_rank_max=None,
    override_p_min=None,
    override_p_max=None,
):
    """
    Build locked-regime tickets without printing. Returns list of ticket dicts or None.
    """
    regime = _compute_cohort_regime(
        run_data,
        allowed_dates=allowed_dates,
        allowed_lottery=allowed_lottery,
    )
    if regime is None:
        return None

    pred = _prepare_prediction_numbers(run_data)
    rows = regime["rows"]
    decade_ids = regime["decade_ids"]

    hwc_shapes = [r["hwc"] for r in rows]
    target_hwc, _ = _choose_anchor_shape(hwc_shapes, _is_adjacent_shape_3)
    if override_cohort_hwc is not None and len(override_cohort_hwc) == 3:
        target_hwc = (int(override_cohort_hwc[0]), int(override_cohort_hwc[1]), int(override_cohort_hwc[2]))
        if sum(target_hwc) != pred["K"] - 1:
            raise ValueError("override_cohort_hwc must sum to K-1")

    dec_shapes = [tuple(r["decades"][d] for d in decade_ids) for r in rows]
    target_dec_shape, _ = _choose_anchor_shape(dec_shapes, _is_adjacent_shape_vec)
    target_decades = _shape_to_decade_dict(target_dec_shape, decade_ids)
    if override_cohort_decades is not None:
        target_decades = {int(k): int(v) for k, v in override_cohort_decades.items()}
        target_dec_shape = tuple(target_decades.get(d, 0) for d in decade_ids)
        if sum(target_dec_shape) != pred["K"] - 1:
            raise ValueError("override_cohort_decades must sum to K-1")

    rows_sorted = sorted(rows, key=lambda r: r["date"])
    eval_last_n = COHORT_AUTOPRED_EVAL_LAST_N or 3
    rank_min_fn = _choose_auto_predictor(rows_sorted, metric="rank_min", eval_last_n=eval_last_n)
    rank_max_fn = _choose_auto_predictor(rows_sorted, metric="rank_max", eval_last_n=eval_last_n)
    p_min_fn = _choose_auto_predictor(rows_sorted, metric="P_min", eval_last_n=eval_last_n)
    p_max_fn = _choose_auto_predictor(rows_sorted, metric="P_max", eval_last_n=eval_last_n)
    rank_min_star = rank_min_fn([r["rank_min"] for r in rows_sorted])
    rank_max_star = rank_max_fn([r["rank_max"] for r in rows_sorted])
    p_min_star = p_min_fn([r["P_min"] for r in rows_sorted])
    p_max_star = p_max_fn([r["P_max"] for r in rows_sorted])

    if override_rank_min is not None:
        rank_min_star = int(override_rank_min)
    if override_rank_max is not None:
        rank_max_star = int(override_rank_max)
    if override_p_min is not None:
        p_min_star = float(override_p_min)
    if override_p_max is not None:
        p_max_star = float(override_p_max)

    cohort_pool = [
        n for n in pred["numbers"]
        if rank_min_star <= n["rank"] <= rank_max_star and p_min_star <= n["P"] <= p_max_star
    ]
    leader_pool = [n for n in pred["numbers"] if n["rank"] <= leader_pool_rank_max]

    cohort_candidates = [n["n"] for n in cohort_pool]
    meta_map = {n["n"]: n for n in pred["numbers"]}
    target_dec_tuple = target_dec_shape
    target_h, target_w, target_c = target_hwc
    sfl_pm1 = set()
    if pred["lottery"] != HOP_SOURCE_LOTTERY:
        sfl_pm1 = _sfl_lastn_pm1_set(pred["date"], SFL_LASTN_DRAWS)
    pm1_counts = None
    cohort_candidates_build = cohort_candidates
    leader_pool_build = leader_pool
    if PM1_ONLY_MODE and pred["lottery"] != HOP_SOURCE_LOTTERY:
        pm1_counts = _sfl_lastn_pm1_counts(pred["date"], SFL_LASTN_DRAWS)
        ranked_pm1 = sorted(
            [n for n in NUMBER_RANGE if pm1_counts.get(n, 0) > 0],
            key=lambda n: (pm1_counts.get(n, 0), meta_map[n]["P"]),
            reverse=True,
        )
        pm1_pool = ranked_pm1[:PM1_ONLY_TOP_N]
        if len(pm1_pool) >= pred["K"]:
            cohort_candidates_build = pm1_pool
            leader_pool_build = [n for n in leader_pool if n["n"] in pm1_pool]
            if not leader_pool_build:
                leader_pool_build = leader_pool
        else:
            pm1_counts = None
    if sfl_pm1 and SFL_PM1_STRICT_COHORT:
        strict_cohort = [n for n in cohort_candidates if n in sfl_pm1]
        if len(strict_cohort) >= pred["K"] - 1:
            cohort_candidates_build = strict_cohort
    if sfl_pm1 and SFL_PM1_STRICT_LEADER:
        strict_leaders = [n for n in leader_pool if n["n"] in sfl_pm1]
        if strict_leaders:
            leader_pool_build = strict_leaders

    def _build_tickets(allowed_hwc_shapes, allowed_dec_shapes, candidates, leaders):
        cohort_combos = []
        for combo in itertools.combinations(candidates, pred["K"] - 1):
            hwc = {"H": 0, "W": 0, "C": 0}
            dec = {d: 0 for d in decade_ids}
            ps = []
            ranks = []
            log_scores = []
            max_p = -1.0
            for n in combo:
                m = meta_map[n]
                hwc[m["category"]] += 1
                dec[m["decade"]] += 1
                ps.append(m["P"])
                ranks.append(m["rank"])
                log_scores.append(m["log_score"])
                max_p = max(max_p, m["P"])
            if allowed_hwc_shapes is not None and (hwc["H"], hwc["W"], hwc["C"]) not in allowed_hwc_shapes:
                continue
            dec_tuple = tuple(dec[d] for d in decade_ids)
            if allowed_dec_shapes is not None and dec_tuple not in allowed_dec_shapes:
                continue
            cohort_combos.append({
                "cohort": combo,
                "hwc": (hwc["H"], hwc["W"], hwc["C"]),
                "decades": dec,
                "rank_min": min(ranks),
                "rank_max": max(ranks),
                "rank_span": max(ranks) - min(ranks),
                "P_sum": sum(ps),
                "P_mean": sum(ps) / len(ps),
                "P_min": min(ps),
                "P_max": max(ps),
                "max_P": max_p,
                "sum_log_score": sum(log_scores),
                "product_P": math.prod(ps),
            })

        tickets = []
        for leader in leaders:
            leader_n = leader["n"]
            for c in cohort_combos:
                if leader_n in c["cohort"]:
                    continue
                if leader["P"] < c["max_P"]:
                    continue
                pm1_hits = sum(1 for n in c["cohort"] if n in sfl_pm1)
                if leader_n in sfl_pm1:
                    pm1_hits += 1
                tickets.append({
                    "leader": leader_n,
                    "leader_rank": leader["rank"],
                    "leader_P": leader["P"],
                    "cohort": c["cohort"],
                    "hwc": c["hwc"],
                    "decades": c["decades"],
                    "rank_min": c["rank_min"],
                    "rank_max": c["rank_max"],
                    "rank_span": c["rank_span"],
                    "P_sum": c["P_sum"],
                    "P_mean": c["P_mean"],
                    "P_min": c["P_min"],
                    "P_max": c["P_max"],
                    "sum_log_score": c["sum_log_score"] + leader["log_score"],
                    "product_P": c["product_P"] * leader["P"],
                    "pm1_hits": pm1_hits,
                    "P_total": c["P_sum"] + leader["P"],
                })
        return tickets

    def _dedup_tickets(items):
        seen = set()
        out = []
        for t in items:
            key = (t["leader"], tuple(sorted(t["cohort"])))
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    def _top_k_shapes(counter, k):
        return {shape for shape, _ in counter.most_common(k)} if k and k > 0 else None

    allowed_hwc_shapes = {(target_h, target_w, target_c)}
    allowed_dec_shapes = {target_dec_tuple}
    if COHORT_ALLOWED_HWC_TOP_K and COHORT_ALLOWED_HWC_TOP_K > 1:
        hwc_counter = Counter(hwc_shapes)
        allowed_hwc_shapes = _top_k_shapes(hwc_counter, COHORT_ALLOWED_HWC_TOP_K)
        allowed_hwc_shapes.add((target_h, target_w, target_c))
    if COHORT_ALLOWED_DEC_TOP_K and COHORT_ALLOWED_DEC_TOP_K > 1:
        dec_counter = Counter(dec_shapes)
        allowed_dec_shapes = _top_k_shapes(dec_counter, COHORT_ALLOWED_DEC_TOP_K)
        allowed_dec_shapes.add(target_dec_tuple)
    if PM1_ONLY_MODE and pm1_counts is not None:
        allowed_hwc_shapes = None
        allowed_dec_shapes = None
    if PM1_ONLY_MODE and pm1_counts is not None:
        allowed_hwc_shapes = None
        allowed_dec_shapes = None
    tickets = _build_tickets(allowed_hwc_shapes, allowed_dec_shapes, cohort_candidates_build, leader_pool_build)

    relaxed_candidates = None
    relaxed_hwc_shapes = None
    relaxed_dec_shapes = None
    if max_tickets_to_print:
        if override_rank_min is None:
            rank_min_relaxed = min(r["rank_min"] for r in rows)
        else:
            rank_min_relaxed = rank_min_star
        if override_rank_max is None:
            rank_max_relaxed = max(r["rank_max"] for r in rows)
        else:
            rank_max_relaxed = rank_max_star
        if override_p_min is None:
            p_min_relaxed = min(r["P_min"] for r in rows)
        else:
            p_min_relaxed = p_min_star
        if override_p_max is None:
            p_max_relaxed = max(r["P_max"] for r in rows)
        else:
            p_max_relaxed = p_max_star

        relaxed_candidates = [
            n["n"] for n in pred["numbers"]
            if rank_min_relaxed <= n["rank"] <= rank_max_relaxed and p_min_relaxed <= n["P"] <= p_max_relaxed
        ]
        relaxed_hwc_shapes = {s for s in hwc_shapes if _is_adjacent_shape_3(s, target_hwc)}
        relaxed_dec_shapes = {s for s in dec_shapes if _is_adjacent_shape_vec(s, target_dec_tuple)}

        if len(tickets) < max_tickets_to_print:
            tickets = _build_tickets(relaxed_hwc_shapes, relaxed_dec_shapes, relaxed_candidates, leader_pool_build)

    if SFL_PM1_COMBO_ENABLED and sfl_pm1:
        pm1_tickets = _build_pm1_combo_tickets(
            pred,
            meta_map,
            sfl_pm1,
            top_n=SFL_PM1_COMBO_TOP_N,
            max_tickets=(max_tickets_to_print or SFL_PM1_COMBO_MAX),
            decade_ids=decade_ids,
        )
        tickets = _dedup_tickets(tickets + pm1_tickets)

    if TOP_P_COMBO_ENABLED:
        if TOP_P_COMBO_MAX:
            max_top_p = TOP_P_COMBO_MAX
            if max_tickets_to_print:
                max_top_p = max(max_top_p, max_tickets_to_print)
        else:
            max_top_p = max_tickets_to_print
        top_p_tickets = _build_top_p_combo_tickets(
            pred,
            meta_map,
            top_n=TOP_P_COMBO_N,
            max_tickets=max_top_p,
            decade_ids=decade_ids,
        )
        tickets = _dedup_tickets(tickets + top_p_tickets)

    tickets.sort(key=lambda t: (t["P_total"], t["sum_log_score"], t["product_P"]), reverse=True)
    tickets_ranked = list(tickets)
    if not tickets:
        return {
            "tickets": [],
            "meta_map": meta_map,
            "decade_ids": decade_ids,
        }

    usage_cap = None
    if COHORT_USAGE_CAP_FRAC and max_tickets_to_print:
        usage_cap = max(1, math.ceil(max_tickets_to_print * float(COHORT_USAGE_CAP_FRAC)))
    if usage_cap is None:
        usage_cap = 2
    else:
        usage_cap = min(usage_cap, 2)

    if usage_cap is not None:
        usage_counts = Counter()
        diversified = []
        for t in tickets:
            nums = list(t["cohort"])
            if any(usage_counts[n] >= usage_cap for n in nums):
                continue
            diversified.append(t)
            for n in nums:
                usage_counts[n] += 1
            if max_tickets_to_print and len(diversified) >= max_tickets_to_print:
                break

        if (
            max_tickets_to_print
            and len(diversified) < max_tickets_to_print
            and relaxed_candidates is not None
        ):
            expanded = _build_tickets(relaxed_hwc_shapes, relaxed_dec_shapes, relaxed_candidates, leader_pool_build)
            combined = _dedup_tickets(tickets + expanded)
            combined.sort(key=lambda t: (t["P_total"], t["sum_log_score"], t["product_P"]), reverse=True)
            usage_counts = Counter()
            diversified = []
            for t in combined:
                nums = list(t["cohort"])
                if any(usage_counts[n] >= usage_cap for n in nums):
                    continue
                diversified.append(t)
                for n in nums:
                    usage_counts[n] += 1
                if max_tickets_to_print and len(diversified) >= max_tickets_to_print:
                    break
        tickets = diversified if diversified else tickets

        if max_tickets_to_print and len(tickets) < max_tickets_to_print:
            existing = set((t["leader"], tuple(sorted(t["cohort"]))) for t in tickets)
            for t in tickets_ranked:
                key = (t["leader"], tuple(sorted(t["cohort"])))
                if key in existing:
                    continue
                tickets.append(t)
                existing.add(key)
                if len(tickets) >= max_tickets_to_print:
                    break

    if LEADER_USAGE_CAP:
        leader_counts = Counter()
        capped = []
        seen = set()
        for t in tickets:
            key = (t["leader"], tuple(sorted(t["cohort"])))
            if key in seen:
                continue
            if leader_counts[t["leader"]] >= LEADER_USAGE_CAP:
                continue
            capped.append(t)
            seen.add(key)
            leader_counts[t["leader"]] += 1
            if max_tickets_to_print and len(capped) >= max_tickets_to_print:
                break
        if max_tickets_to_print and len(capped) < max_tickets_to_print:
            for t in tickets:
                key = (t["leader"], tuple(sorted(t["cohort"])))
                if key in seen:
                    continue
                capped.append(t)
                seen.add(key)
                if len(capped) >= max_tickets_to_print:
                    break
        tickets = capped if capped else tickets

    if max_tickets_to_print and TICKET_DIVERSITY_LAMBDA:
        selected = []
        used_counts = Counter()
        remaining = list(tickets)
        while remaining and len(selected) < max_tickets_to_print:
            best_idx = None
            best_score = None
            for i, t in enumerate(remaining):
                nums = [t["leader"]] + list(t["cohort"])
                penalty = sum(used_counts[n] for n in nums)
                score = t["P_total"] - (float(TICKET_DIVERSITY_LAMBDA) * penalty)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break
            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            for n in [chosen["leader"]] + list(chosen["cohort"]):
                used_counts[n] += 1
        tickets = selected if selected else tickets

    if max_tickets_to_print and COVERAGE_MODE:
        weights = {n: meta_map[n]["P"] for n in meta_map}
        selected = []
        used_counts = Counter()
        remaining = list(tickets_ranked)
        seen = set()
        while remaining and len(selected) < max_tickets_to_print:
            best_idx = None
            best_score = None
            for i, t in enumerate(remaining):
                key = (t["leader"], tuple(sorted(t["cohort"])))
                if key in seen:
                    continue
                nums = [t["leader"]] + list(t["cohort"])
                score = 0.0
                for n in nums:
                    w = weights.get(n, 0.0)
                    score += w / (1.0 + (float(COVERAGE_ALPHA) * used_counts[n]))
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is None:
                break
            chosen = remaining.pop(best_idx)
            key = (chosen["leader"], tuple(sorted(chosen["cohort"])))
            if key in seen:
                continue
            selected.append(chosen)
            seen.add(key)
            for n in [chosen["leader"]] + list(chosen["cohort"]):
                used_counts[n] += 1
        tickets = selected if selected else tickets

    if max_tickets_to_print:
        tickets = tickets[:max_tickets_to_print]

    return {
        "tickets": tickets,
        "meta_map": meta_map,
        "decade_ids": decade_ids,
    }

def main():
    random.seed(0)

    # --- NEW: error tracking accumulators (HWC + decade) ---
    err_hwc_target_rows = []   # list of (eH, eW, eC)
    err_hwc_mc_rows = []       # list of (eH, eW, eC)
    err_dec_rows = []          # list of dicts: decade -> err (actual - mc_avg)
    decade_tape_rows = []

    # --- NEW: mapping accumulators (MODEL -> ACTUAL, NO MEANS) ---
    # model_hwc -> Counter(actual_hwc)
    hwc_map = defaultdict(Counter)

    # model_dec_shape_key -> Counter(actual_dec_signature)
    dec_map = defaultdict(Counter)

    # Calibration
    W_cat_star, mse_cat = calibration_category_window()
    # print(f"Chosen WINDOW_SIZE_CAT* = {W_cat_star} (category MSE = {mse_cat:.6f})")

    # Main run with chosen windows
    state = LearningState()
    # print("\n=== MAIN RUN: TARGET DRAWS LEARNING DIAGNOSTICS ===")
    learning_snapshots = []
    for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
        res = process_target(lottery_name, target_date,
                             W_cat_star,
                             state, do_learning=True,
                             top_n=TOP_N_PREDICTIONS)
        if res is None:
            print(f"[MAIN RUN] Skipped {lottery_name} on {target_date} (incomplete window)")
            continue

        actual_main = get_actual_main(lottery_name, target_date)
        Hot_set = res["Hot_set"]
        Warm_set = res["Warm_set"]
        Cold_set = res["Cold_set"]
        dec_pred = res["dec_pred"]

        # print(f"\n[MAIN RUN] {lottery_name} on {target_date}")
        if actual_main is not None:
            snap = _build_actual_draw_snapshot(lottery_name, target_date, actual_main, res)
            if snap is not None:
                learning_snapshots.append(snap)
            h_a, w_a, c_a = hwc_counts(actual_main, Hot_set, Warm_set, Cold_set)
            dec_a = decade_counts(actual_main)
            decade_tape_rows.append({
                "lottery": lottery_name,
                "date": target_date,
                "dec": dec_a,
                "draw_size": len(actual_main),
            })
            model_hwc_key = (res['h_target'], res['w_target'], res['c_target'])
            actual_hwc_val = (h_a, w_a, c_a)
            hwc_map[model_hwc_key][actual_hwc_val] += 1

            model_dec_key = _decade_shape_key_from_mc(dec_pred)
            actual_dec_sig = _decade_signature_from_actual(dec_a)
            dec_map[model_dec_key][actual_dec_sig] += 1

        else:
            print("  Actual main: None (no draw found)")

        if actual_main is not None:
            # HWC error vs model target (direction: + means model under-shot that count)
            err_hwc_target = (
                h_a - res['h_target'],
                w_a - res['w_target'],
                c_a - res['c_target'],
            )

            # HWC error vs MC averages (direction: + means MC under-produced that category)
            err_hwc_mc = (
                h_a - res['hot_pred'],
                w_a - res['warm_pred'],
                c_a - res['cold_pred'],
            )

            # Decade error (Actual - MC avg)
            # IMPORTANT: use the union of decades seen in actual + predicted, so this works across lotteries
            dec_keys = sorted(set(list(dec_a.keys()) + list(dec_pred.keys())))
            err_dec = {d: dec_a[d] - dec_pred.get(d, 0.0) for d in dec_keys}

            # Store for summary
            err_hwc_target_rows.append(err_hwc_target)
            err_hwc_mc_rows.append(err_hwc_mc)
            err_dec_rows.append(err_dec)

            # Optional: show biggest under/over decade (helps decide DECADE_FACTORS_OVERRIDE)
            if dec_keys:
                max_under_d = max(dec_keys, key=lambda d: err_dec[d])  # most positive
                max_over_d  = min(dec_keys, key=lambda d: err_dec[d])  # most negative
                # print(f"    max_under_decade={max_under_d} ({err_dec[max_under_d]:+.3f}), max_over_decade={max_over_d} ({err_dec[max_over_d]:+.3f})")


    # Apply prediction-only overrides (trials, H/W/C, decades)
    if PREDICTION_CONFIG and PREDICTION_CONFIG.get('APPLY_PREDICTION_OVERRIDES') :
        apply_prediction_config_overrides()

    # Final prediction
    prediction_lottery_name, prediction_date, prediction_draw_size = PREDICTION_TARGET
    if prediction_draw_size != main_draw_size(prediction_lottery_name):
        raise RuntimeError("Config error: prediction_draw_size != main_draw_size")

    res_pred = process_target(prediction_lottery_name, prediction_date,
                              W_cat_star,
                              state, do_learning=False,
                              top_n=TOP_N_PREDICTIONS)
    if res_pred is None:
        raise RuntimeError("Prediction aborted due to incomplete window")

    Hot_set = res_pred["Hot_set"]
    Warm_set = res_pred["Warm_set"]
    Cold_set = res_pred["Cold_set"]
    dec_pred = res_pred["dec_pred"]
    actual_main = get_actual_main(prediction_lottery_name, prediction_date)
    prediction_snapshot = _build_prediction_snapshot(prediction_lottery_name, prediction_date, res_pred)
    prediction_actual_snapshot = _build_actual_draw_snapshot(
        prediction_lottery_name,
        prediction_date,
        actual_main,
        res_pred,
    )

    locked_regime_draws = []
    if LOCKED_REGIME_DATES:
        lot = LOCKED_REGIME_LOTTERY or prediction_lottery_name
        for d in LOCKED_REGIME_DATES:
            cached = LOCKED_REGIME_SNAPSHOT_CACHE.get(d)
            if cached is not None:
                locked_regime_draws.append(cached)
                continue
            if d == prediction_date and prediction_actual_snapshot is not None:
                locked_regime_draws.append(prediction_actual_snapshot)
                continue
            res_locked = process_target(
                lot,
                d,
                W_cat_star,
                state,
                do_learning=False,
                top_n=TOP_N_PREDICTIONS,
            )
            if res_locked is None:
                continue
            actual_locked = get_actual_main(lot, d)
            snap = _build_actual_draw_snapshot(lot, d, actual_locked, res_locked)
            if snap is not None:
                locked_regime_draws.append(snap)

    if actual_main is None:
        print("\n=== FINAL PREDICTION ===")
        print(f"Lottery:   {prediction_lottery_name}")
        print(f"Date:      {prediction_date}")
        print(f"Draw size: {prediction_draw_size}")
        print(f"TRIALS:    {res_pred['TRIALS']}, EXPLORE_FRAC={res_pred['EXPLORE_FRAC']:.3f}")

        # Extra diagnostics for manual tuning (no effect on logic)
        print(f"  Category bias: avg_hot={res_pred['avg_hot']:.3f}, "
              f"avg_warm={res_pred['avg_warm']:.3f}, "
              f"avg_cold={res_pred['avg_cold']:.3f}, "
              f"bias={res_pred['bias']}")
        print(f"  Category weights: hot_w={res_pred['hot_w']:.3f}, "
              f"warm_w={res_pred['warm_w']:.3f}, "
              f"cold_w={res_pred['cold_w']:.3f}")

        print("\n[DEBUG] Learned tape order:")
        for r in decade_tape_rows:
            print(f"  {r['date']}  {r['lottery']}  {r['dec']}")

        # Print Hot/Warm/Cold number lists
        Hot_list = sorted(list(res_pred["Hot_set"]))
        Warm_list = sorted(list(res_pred["Warm_set"]))
        Cold_list = sorted(list(res_pred["Cold_set"]))

        print(f"\nHot numbers:  {Hot_list}")
        print(f"Warm numbers: {Warm_list}")
        print(f"Cold numbers: {Cold_list}")

    def decade_of_num(n: int):
        for d, lo, hi in DECADE_BANDS:
            if lo <= n <= hi:
                return d
        return None

    def p_rank_windows(P_map, window_size: int, top_k: int = 5, decade_bonus: float = 0.01):
        """
        Slides a consecutive rank window over numbers sorted by descending P.
        Ranks windows by: sum(P) + decade_bonus * decade_coverage
        Returns list of dicts: {start_rank,end_rank,sumP,decades,nums}
        """
        ranked = sorted(NUMBER_RANGE, key=lambda n: P_map[n], reverse=True)

        windows = []
        for start in range(0, len(ranked) - window_size + 1):
            end = start + window_size
            nums = ranked[start:end]
            sumP = sum(P_map[n] for n in nums)
            decades = {decade_of_num(n) for n in nums}
            decades.discard(None)
            decade_cov = len(decades)

            score = sumP + decade_bonus * decade_cov
            windows.append({
                "score": score,
                "sumP": sumP,
                "decade_cov": decade_cov,
                "start_rank": start + 1,      # 1-based
                "end_rank": end,              # 1-based inclusive
                "nums": nums,
            })

        windows.sort(key=lambda w: w["score"], reverse=True)
        return windows[:top_k]

    def window_hit_stats(window_nums, actual_nums):
        actual = set(actual_nums)
        exact = len(actual.intersection(window_nums))
        near = 0
        for a in actual:
            if any(abs(a - w) <= 1 for w in window_nums):
                near += 1
        return exact, near

    # Pull P/log_score from prediction result (you added them above)
    P_map = res_pred.get("P")
    log_score_map = res_pred.get("log_score")
    top_k = 50

    P = res_pred["P"]
    log_score = res_pred["log_score"]

    def print_scores(title, nums):
        print(f"\n{title}")
        for n in nums:
            print(f"  {n:2d}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    # ------------------------------------------------------------
    # Rank-window view (7–9 consecutive ranks) to make picking easy
    # ------------------------------------------------------------


    def print_rank_windows_by_P(
        *,
        p_min: float,
        p_max: float,
        window_lens=WINDOW_LENGTH,
        max_start_rank: int = 35,
        exclude_hot: bool = True,
        title: str = None,
    ):
        """
        Shows consecutive rank windows (by descending P) where numbers fall in [p_min, p_max].
        This is the cleanest way to see a 'tight band' to pick from.
        """
        ranked = sorted(NUMBER_RANGE, key=lambda n: P[n], reverse=True)

        # optional filter (but keep ranks stable by filtering AFTER ranking)
        def allowed(n: int) -> bool:
            if exclude_hot and n in res_pred["Hot_set"]:
                return False
            return True

        hits = []
        for wlen in window_lens:
            for start_rank in range(1, max_start_rank + 1):
                window = ranked[start_rank - 1 : start_rank - 1 + wlen]
                if len(window) < wlen:
                    continue

                # apply allowed filter (must keep exact consecutive ranks; so reject if any disallowed)
                if not all(allowed(n) for n in window):
                    continue

                # all P inside band?
                if all(p_min <= P[n] <= p_max for n in window):
                    hits.append((wlen, start_rank, window))

        hdr = title or f"Rank windows by P within [{p_min:.3f} .. {p_max:.3f}] (exclude_hot={exclude_hot})"
        print("\n" + "=" * len(hdr))
        print(hdr)
        print("=" * len(hdr))

        if not hits:
            print(f"(none found up to start_rank={max_start_rank})")
            return

        # print all hits (usually a small number)
        for (wlen, start_rank, window) in hits:
            end_rank = start_rank + wlen - 1
            print(f"\nWindow len={wlen}, ranks {start_rank}..{end_rank}")
            for i, n in enumerate(window, start=start_rank):
                d = decade_of(n)
                c = _cat_of(n,res_pred)
                print(f"  r{i:02d}  {n:2d}  D{d}  {c}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    def print_rank_window_explicit(
        start_rank: int,
        window_len: int,
        *,
        exclude_hot: bool = False,
        title: str = None,
    ):
        """
        Print one explicit consecutive rank window (by descending P).
        Useful when you decide: "show me ranks 11..18" etc.
        """
        ranked = sorted(NUMBER_RANGE, key=lambda n: P[n], reverse=True)
        window = ranked[start_rank - 1 : start_rank - 1 + window_len]

        hdr = title or f"Rank window ranks {start_rank}..{start_rank + window_len - 1} (exclude_hot={exclude_hot})"
        print("\n" + "=" * len(hdr))
        print(hdr)
        print("=" * len(hdr))

        for i, n in enumerate(window, start=start_rank):
            if exclude_hot and n in res_pred["Hot_set"]:
                continue
            d = decade_of(n)
            c = _cat_of(n, res_pred)
            print(f"  r{i:02d}  {n:2d}  D{d}  {c}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    if actual_main is None:
        print_scores("Hot numbers (scored)", res_pred["Hot_set"])
        print_scores("Warm numbers (scored)", res_pred["Warm_set"])
        print_scores("Cold numbers (scored)", res_pred["Cold_set"])

    P = res_pred["P"]
    log_score = res_pred["log_score"]

    if actual_main is None:
        print("\n=== TOP P RANKS ===")
        ranked = sorted(NUMBER_RANGE, key=lambda n: P[n], reverse=True)
        for r, n in enumerate(ranked[:MAIN_NUMBER_MAX], 1):
            d = decade_of(n)
            cat = _cat_of(n, res_pred)
            print(f"  r{r:02d}  {n:2d}  D{d}  {cat}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")
    # if actual_main is None:

    def _decade_counts(nums):
        dc = {}
        for x in nums:
            d = decade_of(x)
            dc[d] = dc.get(d, 0) + 1
        return dc

    def _cat_counts(nums, res_pred):
        cc = {"H": 0, "W": 0, "C": 0}
        for x in nums:
            cc[_cat_of(x, res_pred)] += 1
        return cc

    def p_rank_windows(P_map, window_size: int, top_k: int = 5, decade_bonus: float = 0.01):
        ranked_local = sorted(NUMBER_RANGE, key=lambda n: P_map[n], reverse=True)
        windows = []
        for start in range(0, len(ranked_local) - window_size + 1):
            end = start + window_size
            nums = ranked_local[start:end]
            sumP = sum(P_map[n] for n in nums)
            decades = {decade_of(n) for n in nums}
            decades.discard(None)
            decade_cov = len(decades)
            score = sumP + decade_bonus * decade_cov
            windows.append((score, sumP, decade_cov, start + 1, end, nums))

        windows.sort(key=lambda t: t[0], reverse=True)
        return windows[:top_k]

    # if actual_main is None:
    #     for wsize in WINDOW_LENGTH:
    #         print(f"\n=== P-RANK WINDOWS (size={wsize}) ===")
    #         print("Top windows by sum(P) + decade_coverage bonus:")
    #         top_ws = p_rank_windows(P, wsize, top_k=top_k, decade_bonus=0.01)
    #
    #         for i, (score, sumP, dec_cov, start_rank, end_rank, nums) in enumerate(top_ws, 1):
    #             dc = _decade_counts(nums)
    #             cc = _cat_counts(nums,res_pred)
    #             print(
    #                 f"  W#{i} ranks[{start_rank}..{end_rank}]  "
    #                 f"sumP={sumP:.3f}  decades={dec_cov}  nums={nums}  dc={dc}  hwc={cc}"
    #             )

    # =========================================================
    # ACTUAL (if exists) + ACTUAL number scores + TOP-N comparison
    # =========================================================
    actual_pred_main = print_actual_if_exists(
        prediction_lottery_name,
        prediction_date,
        Hot_set, Warm_set, Cold_set
    )

    if actual_pred_main is not None:
        # 1) Print P/log_score for actual numbers
        print_actual_number_scores(actual_pred_main, log_score, P, res_pred)

        actual_set = set(actual_pred_main)

        best_exact = (-1, None, None)  # hits, tuple, idx
        best_pm1 = (-1, None, None)

    return {
        "learning_draws": learning_snapshots,
        "prediction": prediction_snapshot,
        "prediction_actual": prediction_actual_snapshot,
        "locked_regime_draws": locked_regime_draws,
        "prediction_target": {
            "lottery": prediction_lottery_name,
            "date": prediction_date,
            "draw_size": prediction_draw_size,
        },
    }

if __name__ == "__main__":
    main()

def add_draw(date, lottery, main, supp=None, powerball=None):
    global_draws.append(Draw(date, lottery, main, supp, powerball))

def addDraws():
    # Prefer loading draws from shared CSV to avoid manual updates.
    def _parse_list_cells(cell_text):
        if cell_text is None:
            return []
        text = str(cell_text).strip()
        if not text:
            return []
        lists = []
        for token in re.findall(r"\[[^\]]*\]", text):
            try:
                lists.append(ast.literal_eval(token))
            except Exception:
                continue
        return lists

    def _load_draws_from_csv(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_text = (row.get("Date") or "").strip()
                if not date_text:
                    continue
                dt = datetime.datetime.strptime(date_text, "%a %d-%b-%Y").date()

                # Set for Life (always present in column)
                sfl_lists = _parse_list_cells(row.get("Set for Life (incl supp)"))
                if sfl_lists:
                    sfl_main = sfl_lists[0]
                    sfl_supp = sfl_lists[1] if len(sfl_lists) > 1 else None
                    add_draw(dt, "Set for Life", sfl_main, sfl_supp)

                # Others column: map by weekday
                weekday = dt.weekday()  # Mon=0 Tue=1 Wed=2 Thu=3 Fri=4 Sat=5 Sun=6
                if weekday in (0, 2):
                    other_lottery = "Weekday Windfall"
                elif weekday == 1:
                    other_lottery = "OZ Lotto"
                elif weekday == 3:
                    other_lottery = "Powerball"
                elif weekday == 5:
                    other_lottery = "Saturday Lotto"
                else:
                    other_lottery = None

                if other_lottery:
                    other_lists = _parse_list_cells(row.get("Others (incl supp)"))
                    if not other_lists:
                        continue
                    other_main = other_lists[0]
                    other_tail = other_lists[1] if len(other_lists) > 1 else None
                    if other_lottery == "Powerball":
                        add_draw(dt, "Powerball", other_main, None, other_tail or [])
                    else:
                        add_draw(dt, other_lottery, other_main, other_tail)

    csv_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Single_V2_0", "cross_lotto_data.csv")
    )
    if os.path.exists(csv_path):
        try:
            _load_draws_from_csv(csv_path)
            return
        except Exception as e:
            print(f"[addDraws] CSV load failed ({csv_path}): {e}. Falling back to hardcoded draws.")

