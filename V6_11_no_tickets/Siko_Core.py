#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import datetime
import itertools
from collections import Counter, defaultdict
from statistics import mean

# ===========================
# CONFIG
# ===========================

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

# BASE_TRIALS = 3000         # base TRIALS factor before scaling
# MIN_TRIALS =  9000          # minimum MC trials
# MAX_TRIALS = 12000         # maximum MC trials

BASE_TRIALS = 30         # base TRIALS factor before scaling
MIN_TRIALS = 60          # minimum MC trials
MAX_TRIALS = 90         # maximum MC trials

CLUSTER_TRIAL_FRAC = 0.25

# CLUSTER_TRIAL_FRAC = 0.20   # fraction of trials that are cluster-first mode

STRUCT_CONFIDENCE_WEIGHT = 1.2  # how much structure confidence boosts TRIALS

CLUSTER_LAMBDA_BASE = 0.25  # stronger per-number cluster boost


TOP_N_PREDICTIONS = 10     # you can change this to print more/less predictions

DECADE_BANDS = []

DECADES = [band[0] for band in DECADE_BANDS]
N_DECADES = len(DECADES)

SFL_MOMENTUM_DAYS = 7

# Strength of SFL momentum (0.2–0.5 is reasonable; start mild)
SFL_MOMENTUM_K = 0.35
SFL_MOMENTUM_W_MAX = 1.30   # max multiplicative boost from SFL momentum

# Last-draw suppression for HOT numbers
RECENCY_LASTDRAW_SUPPRESS = 0.85  # factor < 1.0 to slightly punish "just hit" HOT numbers

RECENT_DECADE_DAYS = 2          # short recency window for decades (auto suppression/boost)
RECENT_DECADE_W_MIN = 0.5       # strongest suppression factor
RECENT_DECADE_W_MAX = 1.8       # strongest boost factor

RECENT_DECADE_DAYS_SHORT = 2    # very reactive (what you're already using)
RECENT_DECADE_DAYS_LONG  = 7    # slower background pressure

# Blend between short and long pressure
RECENT_DECADE_SHORT_WEIGHT = 0.65
RECENT_DECADE_LONG_WEIGHT  = 0.35  # must satisfy SHORT_WEIGHT + LONG_WEIGHT = 1.0

TARGET_DRAWS_FOR_LEARNING = []

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

    # H/W/C manual target override (optional)
    # Expected shape: (h_target, w_target, c_target) or [h, w, c]
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

        # Renormalise to average 1.0 again for stability
        # avg_manual = sum(dec_factors.values()) / len(dec_factors)
        # if avg_manual > 0:
        #     for d_id in dec_factors:
        #         dec_factors[d_id] /= avg_manual

    # --- 6) map to per-number log weights ---
    decade_weight_log = {}
    for n in NUMBER_RANGE:
        d_id = decade_of(n)
        if d_id is None:
            decade_weight_log[n] = 0.0
        else:
            decade_weight_log[n] = math.log(dec_factors[d_id])

    return decade_weight_log, dec_short_counts, dec_factors

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

class LearningState:
    def __init__(self):
        self.delta_hot = 0.0
        self.delta_warm = 0.0
        self.delta_cold = 0.0
        self.cluster_priority_score_global = {}  # cluster tuple -> offset

    def reset(self):
        self.__init__()

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
        "P_map": P,
        "log_score_map": log_score,

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

    print("\n[DEBUG] Top dominant transitions:")
    for d in [1, 2, 3, 4, 5]:
        if T["trans_dom"].get(d):
            print(f"  D{d} -> {T['trans_dom'][d].most_common(3)}")

    print("\n[DEBUG] Top cluster transitions:")
    for d in [1, 2, 3, 4, 5]:
        if T["trans_cluster"].get(d):
            print(f"  D{d} -> {T['trans_cluster'][d].most_common(3)}")

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

    print("\n=== CROSS-LOTTERY DECADE TRANSITION SUGGESTION (ACTUAL TAPE ONLY) ===")
    print(f"Last tape row: {last['lottery']} on {last['date']}  draw_size={last['draw_size']}")
    print(f"  Last actual decades: {last['dec']}")
    print(f"  Last state: label={last_state['label']}, dominant={last_state['dominant']}, "
          f"clusters={last_state['clusters']}, missing={last_state['missing']}, shape={last_state['shape']}")

    if next_label:
        print(f"Predicted next STATE label (from transitions): {next_label}  "
              f"(counts={dict(trans_state[last_state['label']])})")
    else:
        print("Predicted next STATE label: None (no transitions recorded for this label)")

    if dom_votes:
        print(f"Predicted dominant decade (vote): {predicted_dominant}  votes={dict(dom_votes)}")
        if dom_move_diag:
            print(f"Dominant movement: {dom_move_diag}")
    else:
        print("Predicted dominant decade: None (no dominant transitions learned)")

    if cl_votes:
        print(f"Predicted cluster decades (vote top2): {predicted_clusters}  votes={dict(cl_votes)}")
    else:
        print("Predicted cluster decades: None (no cluster transitions learned)")

    if miss_votes:
        print(f"Predicted missing decade (vote): {predicted_missing}  votes={dict(miss_votes)}")
    else:
        print("Predicted missing decade: None (no missing transitions learned)")

    print("\nSuggested *mild* DECADE_FACTORS_OVERRIDE (use only if you want a nudge, not a force):")
    print(factors)

    return {
        "next_label": next_label,
        "predicted_dominant": predicted_dominant,
        "predicted_clusters": predicted_clusters,
        "predicted_missing": predicted_missing,
        "factors": factors,
    }

import datetime

def build_targets_for_learning():
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

def print_actual_number_scores(actual_main, log_score, P, title="ACTUAL numbers (with score)"):
    """
    Prints log_score and P for each actual number.
    Works if log_score/P are dicts or list-like indexed by number.
    """
    if actual_main is None:
        return

    actual_sorted = sorted(actual_main)
    print(f"\n{title}")
    ps = []
    for n in actual_sorted:
        ls = log_score[n]
        p = P[n]
        ps.append(p)
        print(f"  {n:2d}  log_score={ls:+.6f}  P={p:.8f}")

    print(f"  P_sum={sum(ps):.8f}  P_mean={sum(ps)/len(ps):.8f}  P_min={min(ps):.8f}  P_max={max(ps):.8f}")

def print_actual_if_exists(lottery_name, date, Hot_set, Warm_set, Cold_set):
    """
    Returns sorted actual main list if found, else None.
    Also prints Actual main + H/W/C + decades.
    """
    actual_main = get_actual_main(lottery_name, date)
    if actual_main is None:
        print("\n[ACTUAL] No actual draw found in GLOBAL_DRAWS for this prediction date.")
        return None

    actual_sorted = sorted(actual_main)
    h_a, w_a, c_a = hwc_counts(actual_sorted, Hot_set, Warm_set, Cold_set)
    dec_a = decade_counts(actual_sorted)

    print("\n=== ACTUAL RESULT (FOUND IN GLOBAL_DRAWS) ===")
    print(f"Lottery: {lottery_name}")
    print(f"Date:    {date}")
    print(f"Actual main: {actual_sorted}")
    print(f"Actual H/W/C: H={h_a}, W={w_a}, C={c_a}")
    print(f"Actual decades: {dec_a}")

    return actual_sorted

# ============================================================
# P-window + rank-row helpers (standalone, pure functions)
# ============================================================

import random
from collections import Counter

def decade_of(n: int) -> int:
    """
    Uses core.DECADE_BANDS structure:
      (1,  1, 10),
      (2, 11, 20),
      (3, 21, 30),
      (4, 31, 40),
      (5, 41, 45),
    """
    for did, lo, hi in DECADE_BANDS:
        if lo <= n <= hi:
            return did
    # If your game max > 45, adjust bands. For safety:
    return -1


def _cat_of_row(row) -> str:
    """
    Row is dict with key 'cat' or a tuple with cat at index 2.
    We standardize to dict rows in build_rank_rows_from_res_pred().
    """
    return row.get("cat", "?")


def build_rank_rows_from_res_pred(res_pred: dict, exclude_hot: bool = False):
    """
    Builds a unified list of ranked rows sorted by P desc.
    Expects res_pred to contain:
      - "P" : {n: P}
      - "log_score" : {n: log_score}
      - "Hot_set", "Warm_set", "Cold_set" (sets)
    Returns list of dict rows:
      {rank, n, decade, cat, log_score, P}
    """
    if "P" not in res_pred or "log_score" not in res_pred:
        raise KeyError("res_pred must include 'P' and 'log_score' maps")

    P_map = res_pred["P"]
    log_map = res_pred["log_score"]

    hot = set(res_pred.get("Hot_set", set()))
    warm = set(res_pred.get("Warm_set", set()))
    cold = set(res_pred.get("Cold_set", set()))

    rows = []
    for n, p in P_map.items():
        if exclude_hot and n in hot:
            continue
        if n in hot:
            cat = "H"
        elif n in warm:
            cat = "W"
        elif n in cold:
            cat = "C"
        else:
            cat = "?"

        rows.append({
            "n": int(n),
            "decade": decade_of(int(n)),
            "cat": cat,
            "log_score": float(log_map.get(n, 0.0)),
            "P": float(p),
        })

    # Sort by P desc, then log_score desc, then number asc for determinism
    rows.sort(key=lambda r: (-r["P"], -r["log_score"], r["n"]))

    # Add rank (1-based)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    return rows


def pick_best_p_window(rank_rows, window_len: int = 9, bonus: float = 0.001):
    """
    Slide a window over rank_rows (already sorted by P desc).
    Score = sum(P) + bonus * decade_coverage
    Returns dict:
      {start_rank, end_rank, nums, sumP, decade_coverage, decade_counts, hwc_counts, rows}
    """
    if not rank_rows:
        raise ValueError("rank_rows is empty")
    if window_len <= 0 or window_len > len(rank_rows):
        raise ValueError(f"window_len must be 1..{len(rank_rows)}")

    best = None
    N = len(rank_rows)

    for i in range(0, N - window_len + 1):
        win = rank_rows[i:i + window_len]
        sumP = sum(r["P"] for r in win)
        decades = [r["decade"] for r in win if r["decade"] != -1]
        decade_coverage = len(set(decades))
        score = sumP + bonus * decade_coverage

        decade_counts = dict(Counter(decades))
        hwc_counts = dict(Counter(r["cat"] for r in win))

        cand = {
            "start_rank": win[0]["rank"],
            "end_rank": win[-1]["rank"],
            "nums": [r["n"] for r in win],
            "sumP": sumP,
            "decade_coverage": decade_coverage,
            "decade_counts": decade_counts,
            "hwc_counts": hwc_counts,
            "rows": win,
            "score": score,
        }

        if best is None or cand["score"] > best["score"]:
            best = cand

    return best


def _counts_hwc(nums, hot_set, warm_set, cold_set):
    h = sum(1 for n in nums if n in hot_set)
    w = sum(1 for n in nums if n in warm_set)
    c = sum(1 for n in nums if n in cold_set)
    return (h, w, c)


def _counts_decades(nums):
    dc = Counter(decade_of(n) for n in nums)
    # remove -1 if present
    if -1 in dc:
        del dc[-1]
    return dict(dc)


def generate_tickets_from_window(
    window_nums,
    *,
    draw_size: int,
    n_tickets: int = 10,
    seed: int = 0,
    target_hwc=None,   # tuple like (H,W,C) or None
    target_dec=None,   # dict like {1:1,2:2,...} or None
    hot_set=None,
    warm_set=None,
    cold_set=None,
    max_tries: int = 200000,
):
    """
    Randomly generate unique tickets from window_nums that satisfy optional constraints.
    Constraints:
      - target_hwc: exact category counts
      - target_dec: exact decade counts
    Returns list of sorted lists (tickets).
    """
    rng = random.Random(seed)
    window_nums = list(dict.fromkeys(int(x) for x in window_nums))  # unique preserve
    if draw_size > len(window_nums):
        raise ValueError(f"draw_size={draw_size} > window size={len(window_nums)}")

    hot_set = set(hot_set or [])
    warm_set = set(warm_set or [])
    cold_set = set(cold_set or [])

    tickets = []
    seen = set()

    tries = 0
    while len(tickets) < n_tickets and tries < max_tries:
        tries += 1
        pick = rng.sample(window_nums, draw_size)
        pick.sort()

        key = tuple(pick)
        if key in seen:
            continue

        # constraints
        if target_hwc is not None:
            if _counts_hwc(pick, hot_set, warm_set, cold_set) != tuple(target_hwc):
                continue

        if target_dec is not None:
            dc = _counts_decades(pick)
            # Ensure missing decades count as 0
            ok = True
            for d, cnt in target_dec.items():
                if dc.get(d, 0) != cnt:
                    ok = False
                    break
            # Also ensure no extra decades beyond target_dec
            if ok:
                for d, cnt in dc.items():
                    if d not in target_dec and cnt != 0:
                        ok = False
                        break
            if not ok:
                continue

        seen.add(key)
        tickets.append(pick)

    return tickets


def build_tickets_from_run_payload(
    run_payload: dict,
    *,
    window_len: int = 9,
    n_tickets: int = 10,
    seed: int = 0,
    exclude_hot: bool = False,
    target_hwc=None,
    target_dec=None,
    bonus: float = 0.001,
):
    """
    Takes the payload returned from core.main() and generates tickets using:
      - rank rows by P
      - best sliding P-window of length window_len
      - random ticket sampling inside that window with optional constraints
    Returns:
      {
        "best_window": {...},
        "tickets": [...],
      }
    """
    if "res_pred" not in run_payload:
        raise KeyError("run_payload missing 'res_pred'")
    if "draw_size" not in run_payload:
        raise KeyError("run_payload missing 'draw_size'")

    res_pred = run_payload["res_pred"]
    draw_size = int(run_payload["draw_size"])

    # Build rank rows (P-desc)
    rank_rows = build_rank_rows_from_res_pred(res_pred, exclude_hot=exclude_hot)

    # Pick best window
    best = pick_best_p_window(rank_rows, window_len=window_len, bonus=bonus)

    # Generate tickets from that window
    tickets = generate_tickets_from_window(
        best["nums"],
        draw_size=draw_size,
        n_tickets=n_tickets,
        seed=seed,
        target_hwc=target_hwc,
        target_dec=target_dec,
        hot_set=res_pred.get("Hot_set", set()),
        warm_set=res_pred.get("Warm_set", set()),
        cold_set=res_pred.get("Cold_set", set()),
    )

    return {
        "best_window": best,
        "tickets": tickets,
    }

def learn_pwindow_mappings_from_actuals(
    runs,
    *,
    window_lens=(6, 7, 8, 9),
    near_miss_delta=1,
    bonus=0.001,
):
    """
    From historical runs that HAVE actual_main, learn which P-rank start windows
    tend to contain actual hits (exact and near±delta).
    Returns votes per window_len and also a compact "band" summary.
    """
    # Defensive: allow single int passed accidentally
    if isinstance(window_lens, int):
        window_lens = (window_lens,)

    learned = {
        "votes": {L: {} for L in window_lens},     # L -> {start_rank: count}
        "bands": [],                               # per-run band info
        "used_runs": 0,
    }

    for run in runs:
        actual = run.get("actual_main")
        if not actual:
            continue

        rank_rows = run.get("rank_rows")
        if not rank_rows:
            # rebuild if missing
            rank_rows = build_rank_rows_from_res_pred(run["res_pred"], exclude_hot=False)

        # map number -> rank row
        by_n = {r["n"]: r for r in rank_rows}

        # Collect actual rank rows (exact only)
        actual_rows = [by_n[n] for n in actual if n in by_n]
        if not actual_rows:
            continue

        learned["used_runs"] += 1

        # Band summary (P min/max among actual numbers)
        ps = sorted(r["P"] for r in actual_rows)
        learned["bands"].append({
            "date": run.get("target_date"),
            "p_min": ps[0],
            "p_max": ps[-1],
            "ranks": sorted(r["rank"] for r in actual_rows),
        })

        actual_set = set(actual)

        # exact + near hits helper
        def hits_in_window(win_rows):
            nums = [r["n"] for r in win_rows]
            win_set = set(nums)
            exact = len(actual_set & win_set)

            # near ±delta
            near = 0
            for a in actual_set:
                if any((a + d) in win_set for d in range(-near_miss_delta, near_miss_delta + 1)):
                    near += 1
            return exact, near

        for L in window_lens:
            best = None
            for i in range(0, len(rank_rows) - L + 1):
                win = rank_rows[i:i + L]
                sumP = sum(r["P"] for r in win)
                decades = {r["decade"] for r in win if r["decade"] != -1}
                score = sumP + bonus * len(decades)

                exact, near = hits_in_window(win)

                cand = {
                    "start_rank": win[0]["rank"],
                    "score": score,
                    "exact": exact,
                    "near": near,
                }

                # Prefer higher exact, then near, then score
                if (best is None or
                    (cand["exact"], cand["near"], cand["score"]) >
                    (best["exact"], best["near"], best["score"])):
                    best = cand

            if best:
                votes = learned["votes"][L]
                votes[best["start_rank"]] = votes.get(best["start_rank"], 0) + 1

    return learned

def build_tickets_from_history_and_final(
    history_runs,
    final_run_payload,
    *,
    window_len=9,
    n_tickets=10,
    seed=0,
    exclude_hot=False,
    target_hwc=None,
    target_dec=None,
):
    """
    Learn window start-rank tendencies from history actuals,
    then apply to final run by choosing a start_rank with highest vote.
    If no votes, fallback to best window by sumP+bonus.
    """
    learned = learn_pwindow_mappings_from_actuals(history_runs, window_lens=(window_len,))

    # Build rank rows for final
    rank_rows = build_rank_rows_from_res_pred(final_run_payload["res_pred"], exclude_hot=exclude_hot)

    votes = learned["votes"].get(window_len, {})
    picked_window = None

    if votes:
        # pick most voted start rank
        best_start = max(votes.items(), key=lambda kv: kv[1])[0]
        # slice window
        start_idx = best_start - 1  # rank is 1-based
        end_idx = start_idx + window_len
        if 0 <= start_idx < len(rank_rows) and end_idx <= len(rank_rows):
            win_rows = rank_rows[start_idx:end_idx]
            picked_window = {
                "start_rank": win_rows[0]["rank"],
                "end_rank": win_rows[-1]["rank"],
                "sumP": sum(r["P"] for r in win_rows),
                "decades": len({r["decade"] for r in win_rows if r["decade"] != -1}),
                "nums": [r["n"] for r in win_rows],
                "rows": win_rows,
                "score": None,
                "learned_votes": votes,
                "picked_by": "history_votes",
            }

    if picked_window is None:
        # fallback
        picked_window = pick_best_p_window(rank_rows, window_len=window_len, bonus=0.001)
        if picked_window:
            picked_window["learned_votes"] = votes
            picked_window["picked_by"] = "sumP_bonus_fallback"

    tickets = []
    if picked_window:
        tickets = generate_tickets_from_window(
            window_nums=picked_window["rows"],
            draw_size=int(final_run_payload["draw_size"]),
            n_tickets=n_tickets,
            seed=seed,
            target_hwc=target_hwc,
            target_dec=target_dec,
        )

    return {
        "learned": learned,
        "picked_window": picked_window,
        "tickets": tickets,
    }

from collections import Counter
import math

def _decade_id_from_num(n, decade_bands):
    # decade_bands: [(id, lo, hi), ...]
    for did, lo, hi in decade_bands:
        if lo <= n <= hi:
            return did
    return None

def _actual_hwc_from_run(run):
    res = run["res_pred"]
    hot = set(res.get("Hot_set", []))
    warm = set(res.get("Warm_set", []))
    cold = set(res.get("Cold_set", []))
    H = W = C = 0
    for x in run["actual_main"]:
        if x in hot:
            H += 1
        elif x in warm:
            W += 1
        elif x in cold:
            C += 1
        else:
            # if some number isn’t in any set (shouldn’t happen), treat as warm-ish
            W += 1
    return (H, W, C)

def _actual_dec_counts(run, decade_bands):
    dc = {did: 0 for did, _, _ in decade_bands}
    for x in run["actual_main"]:
        did = _decade_id_from_num(x, decade_bands)
        if did is not None:
            dc[did] += 1
    return dc

def _rank_map_from_rank_rows(rank_rows):
    # rank_rows entries in your logs look like: r01 42 D5 C log_score=.. P=..
    # In payload, I assume they are dicts with at least {"rank":int,"n":int} OR tuples.
    m = {}
    for row in rank_rows:
        if isinstance(row, dict):
            m[row["n"]] = row["rank"]
        else:
            # tuple-like fallback: (rank, n, decade, cat, log_score, P) OR similar
            # adjust indices if your tuple format differs
            rank = row[0]
            n = row[1]
            m[n] = rank
    return m

def _best_window_for_actual(rank_rows, actual_main, window_len):
    """
    Find the start-rank whose window [s..s+L-1] captures max count of actual numbers.
    Returns: (best_start_rank, best_exact_hits)
    """
    if not rank_rows:
        return (None, 0)

    rank_map = _rank_map_from_rank_rows(rank_rows)
    ranks = []
    for x in actual_main:
        r = rank_map.get(x)
        if r is not None:
            ranks.append(r)

    if not ranks:
        return (None, 0)

    max_rank = max(r["rank"] if isinstance(r, dict) else r[0] for r in rank_rows)
    best_s = None
    best_hits = -1

    # Try all possible start ranks
    for s in range(1, max_rank - window_len + 2):
        e = s + window_len - 1
        hits = sum(1 for rr in ranks if s <= rr <= e)
        if hits > best_hits:
            best_hits = hits
            best_s = s

    return (best_s, best_hits)

def derive_learning_from_history(history_runs, decade_bands, window_lens=(6,7,8,9)):
    """
    Returns:
      chosen_window_len, learned_start_rank, learned_hwc, learned_dec
    """

    if not history_runs:
        return (9, 1, None, None)  # fallback

    # ---- Learn HWC (mode over actual draws) ----
    hwc_list = []
    for run in history_runs:
        if run.get("actual_main"):
            hwc_list.append(_actual_hwc_from_run(run))
    learned_hwc = Counter(hwc_list).most_common(1)[0][0] if hwc_list else None

    # ---- Learn decades (mode over actual draws) ----
    dec_shapes = []
    for run in history_runs:
        if run.get("actual_main"):
            dc = _actual_dec_counts(run, decade_bands)
            # store as a tuple in D1..D5 order
            shape = tuple(dc[did] for did, _, _ in decade_bands)
            dec_shapes.append(shape)
    learned_shape = Counter(dec_shapes).most_common(1)[0][0] if dec_shapes else None

    learned_dec = None
    if learned_shape is not None:
        learned_dec = {did: learned_shape[i] for i, (did, _, _) in enumerate(decade_bands)}

    # ---- Learn window_len + start_rank from rank_rows vs actuals ----
    # For each L, find best start rank per run (max exact hits). Then choose:
    # - window_len that gives highest average exact hits
    # - learned_start_rank = median of best starts for that L
    L_scores = []
    per_L_best_starts = {}

    for L in window_lens:
        starts = []
        hits_list = []
        for run in history_runs:
            if not run.get("actual_main"):
                continue
            rank_rows = run.get("rank_rows", [])
            s, hits = _best_window_for_actual(rank_rows, run["actual_main"], L)
            if s is not None:
                starts.append(s)
                hits_list.append(hits)

        per_L_best_starts[L] = starts
        avg_hits = (sum(hits_list) / len(hits_list)) if hits_list else -1
        L_scores.append((avg_hits, L))

    # pick best L by avg exact hits
    L_scores.sort(reverse=True)
    chosen_window_len = L_scores[0][1] if L_scores and L_scores[0][0] >= 0 else 9

    starts = per_L_best_starts.get(chosen_window_len, [])
    if starts:
        starts_sorted = sorted(starts)
        learned_start_rank = starts_sorted[len(starts_sorted)//2]  # median
    else:
        learned_start_rank = 1

    return (chosen_window_len, learned_start_rank, learned_hwc, learned_dec)

def main():
    if PREDICTION_TARGET is None:
        raise RuntimeError(
            "PREDICTION_TARGET is not set. Set core.PREDICTION_TARGET=(lottery, date, draw_size) in wrapper before calling core.main()."
        )

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
    print(f"Chosen WINDOW_SIZE_CAT* = {W_cat_star} (category MSE = {mse_cat:.6f})")

    # Main run with chosen windows
    state = LearningState()
    print("\n=== MAIN RUN: TARGET DRAWS LEARNING DIAGNOSTICS ===")
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

        print(f"\n[MAIN RUN] {lottery_name} on {target_date}")
        if actual_main is not None:
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

            print(f"  Actual main: {sorted(actual_main)}")
            print(f"  Actual H/W/C: H={h_a}, W={w_a}, C={c_a}")
            print(f"  Actual decades: {dec_a}")
        else:
            print("  Actual main: None (no draw found)")

        print(f"  Model H/W/C target: h_target={res['h_target']}, w_target={res['w_target']}, c_target={res['c_target']}")
        print(f"  MC H/W/C avg: hot={res['hot_pred']:.3f}, warm={res['warm_pred']:.3f}, cold={res['cold_pred']:.3f}")
        print(f"  MC decades predicted (avg counts): { {d: round(v,3) for d,v in dec_pred.items()} }")

        # --- NEW: error diagnostics (only when we have actual draw) ---
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

            # Print per-draw errors
            print(f"  err_hwc_target (Actual-Target): H={err_hwc_target[0]:+.3f}, W={err_hwc_target[1]:+.3f}, C={err_hwc_target[2]:+.3f}")
            print(f"  err_hwc_mc     (Actual-MC):     H={err_hwc_mc[0]:+.3f}, W={err_hwc_mc[1]:+.3f}, C={err_hwc_mc[2]:+.3f}")

            # Decade error pretty print (rounded)
            # print(f"  err_dec (Actual-MC): { {d: round(err_dec[d], 3) for d in dec_keys} }")

            # Optional: show biggest under/over decade (helps decide DECADE_FACTORS_OVERRIDE)
            if dec_keys:
                max_under_d = max(dec_keys, key=lambda d: err_dec[d])  # most positive
                max_over_d  = min(dec_keys, key=lambda d: err_dec[d])  # most negative
                print(f"    max_under_decade={max_under_d} ({err_dec[max_under_d]:+.3f}), max_over_decade={max_over_d} ({err_dec[max_over_d]:+.3f})")


    print("\n=== LEARNING SUMMARY ===")
    print(f"  Learned Δ_hot={state.delta_hot:.3f}, "
          f"Δ_warm={state.delta_warm:.3f}, "
          f"Δ_cold={state.delta_cold:.3f}")

    # --- NEW: aggregate error summary across TARGET_DRAWS_FOR_LEARNING ---
    # if err_hwc_target_rows or err_hwc_mc_rows or err_dec_rows:
    #     print("\n=== LEARNING ERROR SUMMARY ===")

    # if err_hwc_target_rows:
    #     n = len(err_hwc_target_rows)
    #     mean_eH = sum(r[0] for r in err_hwc_target_rows) / n
    #     mean_eW = sum(r[1] for r in err_hwc_target_rows) / n
    #     mean_eC = sum(r[2] for r in err_hwc_target_rows) / n
    #     print(f"  Mean err_hwc_target (Actual-Target) over {n} draws: H={mean_eH:+.3f}, W={mean_eW:+.3f}, C={mean_eC:+.3f}")

    # if err_hwc_mc_rows:
    #     n = len(err_hwc_mc_rows)
    #     mean_eH = sum(r[0] for r in err_hwc_mc_rows) / n
    #     mean_eW = sum(r[1] for r in err_hwc_mc_rows) / n
    #     mean_eC = sum(r[2] for r in err_hwc_mc_rows) / n
    #     print(f"  Mean err_hwc_mc     (Actual-MC)     over {n} draws: H={mean_eH:+.3f}, W={mean_eW:+.3f}, C={mean_eC:+.3f}")

    # if err_dec_rows:
    #     # union of all decade keys observed
    #     all_dec_keys = sorted({d for row in err_dec_rows for d in row.keys()})
    #     n = len(err_dec_rows)
    #     mean_err_dec = {d: (sum(row.get(d, 0.0) for row in err_dec_rows) / n) for d in all_dec_keys}
    #     print(f"  Mean err_dec (Actual-MC) over {n} draws: { {d: round(mean_err_dec[d], 3) for d in all_dec_keys} }")
    #
    #     # show biggest consistent under/over decades
    #     if all_dec_keys:
    #         max_under_d = max(all_dec_keys, key=lambda d: mean_err_dec[d])
    #         max_over_d = min(all_dec_keys, key=lambda d: mean_err_dec[d])
    #         print(f"    mean max_under_decade={max_under_d} ({mean_err_dec[max_under_d]:+.3f}), mean max_over_decade={max_over_d} ({mean_err_dec[max_over_d]:+.3f})")

    # Optional: see top learned clusters
    if state.cluster_priority_score_global:
        top_clusters = sorted(
            state.cluster_priority_score_global.items(),
            key=lambda kv: kv[1],
            reverse=True
        )[:5]
        print(f"  Top learned clusters (cluster -> score): {top_clusters}")

    if not (PREDICTION_CONFIG and PREDICTION_CONFIG.get('APPLY_PREDICTION_OVERRIDES')):
        suggest_next_decade_bias_from_transitions(
            decade_tape_rows,
            prediction_draw_size=main_draw_size(PREDICTION_TARGET[0])
        )

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
    print(f"  Recent decade hits (short window): {res_pred['dec_short_counts']}")
    print(f"  Final decade multipliers: "
          f"{ {d: round(f, 3) for d, f in res_pred['dec_factors'].items()} }")

    # Print Hot/Warm/Cold number lists
    Hot_list = sorted(list(res_pred["Hot_set"]))
    Warm_list = sorted(list(res_pred["Warm_set"]))
    Cold_list = sorted(list(res_pred["Cold_set"]))

    print(f"\nHot numbers:  {Hot_list}")
    print(f"Warm numbers: {Warm_list}")
    print(f"Cold numbers: {Cold_list}")

    # ============================================================
    # P-RANK WINDOWS (actionable view of the P table)
    # ============================================================
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

    if P_map and log_score_map:
        for wsize in WINDOW_LENGTH:
            print(f"\n=== P-RANK WINDOWS (size={wsize}) ===")
            print("Top 5 windows by sum(P) + decade_coverage bonus:")

            top_windows = p_rank_windows(P_map, window_size=wsize, top_k=5, decade_bonus=0.01)
            for i, w in enumerate(top_windows, 1):
                print(f"  W#{i} ranks[{w['start_rank']}..{w['end_rank']}]  "
                      f"sumP={w['sumP']:.3f}  decades={w['decade_cov']}  nums={w['nums']}")

            # Optional: overlay actual draw if it exists in GLOBAL_DRAWS
            actual_main = get_actual_main(PREDICTION_TARGET[0], PREDICTION_TARGET[1])
            if actual_main is not None and top_windows:
                best_nums = top_windows[0]["nums"]
                exact, near = window_hit_stats(best_nums, actual_main)
                print(f"Actual hits in best window: exact={exact} near±1={near}")
    else:
        print("\n[WARN] P/log_score not found in res_pred. Add them to the prediction result dict.")


    P = res_pred["P"]
    log_score = res_pred["log_score"]

    def print_scores(title, nums):
        print(f"\n{title}")
        for n in nums:
            print(f"  {n:2d}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    # ------------------------------------------------------------
    # Rank-window view (7–9 consecutive ranks) to make picking easy
    # ------------------------------------------------------------
    def _cat_of(n: int) -> str:
        if n in res_pred["Hot_set"]:
            return "H"
        if n in res_pred["Warm_set"]:
            return "W"
        if n in res_pred["Cold_set"]:
            return "C"
        return "?"

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
                c = _cat_of(n)
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
            c = _cat_of(n)
            print(f"  r{i:02d}  {n:2d}  D{d}  {c}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    print_scores("Hot numbers (scored)", res_pred["Hot_set"])
    print_scores("Warm numbers (scored)", res_pred["Warm_set"])
    print_scores("Cold numbers (scored)", res_pred["Cold_set"])

    # ============================================================
    # (2) TOP P RANKS (easy manual ticket picking)
    # ============================================================
    P = res_pred["P"]
    log_score = res_pred["log_score"]

    def _cat_of(n: int) -> str:
        if n in res_pred["Hot_set"]:
            return "H"
        if n in res_pred["Warm_set"]:
            return "W"
        if n in res_pred["Cold_set"]:
            return "C"
        return "?"

    print("\n=== TOP P RANKS (first 30) ===")
    ranked = sorted(NUMBER_RANGE, key=lambda n: P[n], reverse=True)
    for r, n in enumerate(ranked[:30], 1):
        d = decade_of(n)
        cat = _cat_of(n)
        print(f"  r{r:02d}  {n:2d}  D{d}  {cat}  log_score={log_score[n]:+.6f}  P={P[n]:.8f}")

    # ============================================================
    # (3) WINDOW DIAGNOSTICS: decade counts + H/W/C counts per window
    # ============================================================
    def _decade_counts(nums):
        dc = {}
        for x in nums:
            d = decade_of(x)
            dc[d] = dc.get(d, 0) + 1
        return dc

    def _cat_counts(nums):
        cc = {"H": 0, "W": 0, "C": 0}
        for x in nums:
            cc[_cat_of(x)] += 1
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

    for wsize in WINDOW_LENGTH:
        print(f"\n=== P-RANK WINDOWS (size={wsize}) ===")
        print("Top 5 windows by sum(P) + decade_coverage bonus:")
        top_ws = p_rank_windows(P, wsize, top_k=5, decade_bonus=0.01)

        for i, (score, sumP, dec_cov, start_rank, end_rank, nums) in enumerate(top_ws, 1):
            dc = _decade_counts(nums)
            cc = _cat_counts(nums)
            print(
                f"  W#{i} ranks[{start_rank}..{end_rank}]  "
                f"sumP={sumP:.3f}  decades={dec_cov}  nums={nums}  dc={dc}  hwc={cc}"
            )

    # Example: your band you mentioned
    print_rank_windows_by_P(p_min=0.015, p_max=0.030, window_lens=WINDOW_LENGTH, max_start_rank=40, exclude_hot=True)

    # Or if you want to inspect a specific slice: ranks 11..18
    print_rank_window_explicit(11, 8, exclude_hot=True)


    print(f"MC H/W/C avg: hot={res_pred['hot_pred']:.3f}, warm={res_pred['warm_pred']:.3f}, cold={res_pred['cold_pred']:.3f}")
    print(f"MC decades predicted (avg counts): { {d: round(v ,3) for d,v in dec_pred.items()} }")

    print(f"\nTop-{TOP_N_PREDICTIONS} predicted tuples (joint score, base prob, H/W/C, decades):")
    for T, score, p in res_pred["topN"]:
        nums = list(T)
        h_p, w_p, c_p = hwc_counts(nums, Hot_set, Warm_set, Cold_set)
        dec_p = decade_counts(nums)
        print(f"{nums}  score={score:.6f}  prob={p:.6f}  H/W/C=({h_p},{w_p},{c_p})  decades={dec_p}")

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
        print_actual_number_scores(actual_pred_main, log_score, P)

        # 2) Compare Top-N predicted tuples vs actual
        print("\n=== TOP-TUPLE COMPARISON VS ACTUAL ===")
        actual_set = set(actual_pred_main)

        best_exact = (-1, None, None)  # hits, tuple, idx
        best_pm1 = (-1, None, None)

        for idx, (T, score, p) in enumerate(res_pred["topN"], start=1):
            nums = list(T)
            ex = exact_hits(nums, actual_set)
            pm1 = near_miss_pm1_hits(nums, actual_set)

            if ex > best_exact[0]:
                best_exact = (ex, nums, idx)
            if pm1 > best_pm1[0]:
                best_pm1 = (pm1, nums, idx)

            print(f"#{idx:02d} pred={nums}  exact={ex}  near_miss_pm1={pm1}  score={score:.6f}  prob={p:.6f}")

        print("\n=== BEST OF TOP-N ===")
        print(f"Best exact-hit tuple: hits={best_exact[0]}  idx=#{best_exact[2]:02d}  nums={best_exact[1]}")
        print(f"Best ±1 near-miss tuple: hits={best_pm1[0]}  idx=#{best_pm1[2]:02d}  nums={best_pm1[1]}")

    rank_rows = build_rank_rows_from_res_pred(res_pred, exclude_hot=False)
    top_tuples = res_pred.get("topN", [])

    return {
        "lottery": prediction_lottery_name,
        "target_date": prediction_date,
        "draw_size": prediction_draw_size,
        "res_pred": res_pred,
        "top_tuples": top_tuples,
        "actual_main": actual_main,
        "rank_rows": rank_rows,
    }


if __name__ == "__main__":
    main()

def add_draw(date, lottery, main, supp=None, powerball=None):
    global_draws.append(Draw(date, lottery, main, supp, powerball))

def addDraws():
    # Set for Life
    def setForLife():
        add_draw(d(19, 12), "Set for Life", [29, 21, 2, 32, 37, 20, 36], [17, 19])
        add_draw(d(18, 12), "Set for Life", [14, 13, 27, 37, 30, 36, 42], [35, 15])
        add_draw(d(17, 12), "Set for Life", [27, 34, 26, 23, 1, 13, 7], [9, 12])
        add_draw(d(16, 12), "Set for Life", [8, 27, 34, 2, 38, 29, 1], [5, 31])
        add_draw(d(15, 12), "Set for Life", [21, 22, 43, 16, 17, 31, 25], [3, 30])
        add_draw(d(14, 12), "Set for Life", [40, 28, 30, 17, 7, 4, 24], [41, 44])
        add_draw(d(13, 12), "Set for Life", [27, 33, 31, 20, 30, 4, 41], [12, 7])
        add_draw(d(12, 12), "Set for Life", [22, 2, 36, 29, 10, 23, 13], [15, 27])
        add_draw(d(11, 12), "Set for Life", [27, 12, 16, 3, 30, 8, 29], [41, 31])
        add_draw(d(10, 12), "Set for Life", [1, 36, 40, 28, 37, 10, 3], [12, 19])
        add_draw(d(9, 12), "Set for Life", [12, 20, 16, 38, 26, 13, 39], [22, 40])
        add_draw(d(8, 12), "Set for Life", [39, 4, 42, 11, 16, 43, 37], [21, 32])
        add_draw(d(7, 12), "Set for Life", [4, 34, 30, 21, 23, 35, 15], [22, 18])
        add_draw(d(6, 12), "Set for Life", [42, 15, 24, 31, 5, 40, 39], [19, 1])
        add_draw(d(5, 12), "Set for Life", [5, 25, 21, 17, 31, 1, 15], [24, 22])
        add_draw(d(4, 12), "Set for Life", [35, 2, 25, 8, 6, 17, 28], [3, 31])
        add_draw(d(3, 12), "Set for Life", [22, 29, 44, 31, 10, 25, 30], [8, 14])
        add_draw(d(2, 12), "Set for Life", [37, 13, 15, 19, 25, 39, 26], [3, 5])
        add_draw(d(1, 12), "Set for Life", [18, 1, 10, 41, 24, 11, 3], [25, 2])
        add_draw(d(30, 11), "Set for Life", [7, 44, 18, 27, 32, 22, 11], [38, 9])
        add_draw(d(29, 11), "Set for Life", [8, 31, 4, 6, 42, 16, 14], [13, 19])
        add_draw(d(28, 11), "Set for Life", [15, 27, 8, 39, 5, 43, 20], [19, 29])
        add_draw(d(27, 11), "Set for Life", [12, 36, 6, 7, 37, 41, 29], [8, 43])
        add_draw(d(26, 11), "Set for Life", [29, 37, 34, 14, 5, 21, 20], [18, 19])
        add_draw(d(25, 11), "Set for Life", [26, 16, 23, 15, 31, 1, 27], [8, 41])
        add_draw(d(24, 11), "Set for Life", [41, 1, 17, 29, 14, 40, 22], [35, 31])
        add_draw(d(23, 11), "Set for Life", [25, 27, 42, 18, 26, 9, 33], [22, 19])
        add_draw(d(22, 11), "Set for Life", [24, 23, 31, 30, 26, 5, 17], [6, 27])
        add_draw(d(21, 11), "Set for Life", [27, 32, 10, 42, 38, 33, 17], [19, 39])
        add_draw(d(20, 11), "Set for Life", [28, 10, 11, 35, 34, 41, 23], [30, 26])
        add_draw(d(19, 11), "Set for Life", [4, 44, 5, 33, 21, 30, 39], [9, 18])
        add_draw(d(18, 11), "Set for Life", [33, 35, 44, 32, 20, 29, 39], [5, 41])
        add_draw(d(17, 11), "Set for Life", [15, 23, 40, 43, 28, 1, 37], [18, 34])
        add_draw(d(16, 11), "Set for Life", [8, 19, 21, 27, 40, 14, 7], [20, 44])
        add_draw(d(15, 11), "Set for Life", [13, 4, 27, 14, 2, 5, 42], [33, 39])
        add_draw(d(14, 11), "Set for Life", [7, 25, 23, 35, 13, 18, 6], [3, 39])
        add_draw(d(13, 11), "Set for Life", [25, 24, 3, 21, 5, 33, 36], [22, 11])
        add_draw(d(12, 11), "Set for Life", [15, 20, 29, 21, 5, 10, 6], [32, 17])
        add_draw(d(11, 11), "Set for Life", [4, 7, 10, 44, 32, 30, 26], [5, 18])
        add_draw(d(10, 11), "Set for Life", [5, 36, 13, 23, 39, 3, 9], [35, 6])
        add_draw(d(9, 11), "Set for Life", [11, 4, 44, 26, 6, 31, 40], [21, 33])
        add_draw(d(8, 11), "Set for Life", [7, 31, 5, 37, 43, 38, 2], [42, 10])
        add_draw(d(7, 11), "Set for Life", [30, 18, 6, 28, 33, 41, 14], [38, 29])
        add_draw(d(6, 11), "Set for Life", [12, 20, 35, 42, 41, 10, 18], [33, 32])
        add_draw(d(5, 11), "Set for Life", [16, 22, 13, 34, 25, 3, 18], [33, 43])
        add_draw(d(4, 11), "Set for Life", [38, 9, 27, 25, 10, 23, 37], [13, 17])
        add_draw(d(3, 11), "Set for Life", [8, 15, 25, 26, 13, 24, 23], [4, 2])
        add_draw(d(2, 11), "Set for Life", [6, 28, 26, 24, 13, 11, 19], [22, 12])
        add_draw(d(1, 11), "Set for Life", [8, 31, 42, 24, 15, 7, 4], [19, 18])

    setForLife()

    # Weekday Windfall
    def weekdayWindfall():
        add_draw(d(19, 12), "Weekday Windfall", [17, 3, 25, 5, 38, 2], [9, 7])
        add_draw(d(17, 12), "Weekday Windfall", [36, 14, 7, 17, 45, 29], [12, 32])
        add_draw(d(15, 12), "Weekday Windfall", [33, 14, 8, 23, 9, 27], [4, 3])
        add_draw(d(12, 12), "Weekday Windfall", [10, 36, 45, 2, 15, 39], [25, 38])
        add_draw(d(10, 12), "Weekday Windfall", [15, 2, 10, 33, 38, 26], [19, 14])
        add_draw(d(8, 12), "Weekday Windfall", [26, 40, 6, 39, 37, 12], [24, 7])
        add_draw(d(5, 12), "Weekday Windfall", [9, 23, 8, 16, 11, 33], [34, 1])
        add_draw(d(3, 12), "Weekday Windfall", [15, 2, 38, 37, 22, 35], [39, 6])
        add_draw(d(1, 12), "Weekday Windfall", [8, 6, 30, 38, 36, 1], [43, 5])
        add_draw(d(28, 11), "Weekday Windfall", [30, 8, 25, 43, 39, 24], [21, 1])
        add_draw(d(26, 11), "Weekday Windfall", [44, 43, 8, 36, 16, 27], [31, 30])
        add_draw(d(24, 11), "Weekday Windfall", [44, 15, 20, 17, 4, 18], [7, 11])
        add_draw(d(21, 11), "Weekday Windfall", [4, 5, 26, 10, 40, 20], [14, 24])
        add_draw(d(19, 11), "Weekday Windfall", [43, 26, 35, 25, 42, 13], [24, 5])
        add_draw(d(17, 11), "Weekday Windfall", [37, 11, 4, 2, 5, 7], [30, 22])
        add_draw(d(14, 11), "Weekday Windfall", [34, 11, 28, 15, 44, 31], [9, 20])
        add_draw(d(12, 11), "Weekday Windfall", [35, 11, 33, 15, 34, 45], [8, 37])
        add_draw(d(10, 11), "Weekday Windfall", [38, 3, 31, 22, 28, 5], [26, 14])
        add_draw(d(7, 11), "Weekday Windfall", [31, 16, 23, 30, 6, 3], [13, 18])
        add_draw(d(5, 11), "Weekday Windfall", [26, 15, 18, 27, 7, 37], [19, 44])
        add_draw(d(3, 11), "Weekday Windfall", [25, 14, 29, 23, 45, 13], [31, 8])

    weekdayWindfall()

    # OZ Lotto
    def ozLott():
        add_draw(d(16, 12), "OZ Lotto", [43, 41, 20, 9, 46, 4, 19], [45, 8, 21])
        add_draw(d(9, 12), "OZ Lotto", [21, 15, 3, 6, 9, 33, 19], [31, 14, 7])
        add_draw(d(2, 12), "OZ Lotto", [40, 26, 43, 28, 22, 42, 7], [29, 6, 47])
        add_draw(d(25, 11), "OZ Lotto", [12, 43, 28, 1, 47, 35, 14], [15, 16, 46])
        add_draw(d(18, 11), "OZ Lotto", [39, 2, 22, 8, 27, 6, 4], [47, 5, 24])
        add_draw(d(11, 11), "OZ Lotto", [44, 30, 7, 28, 17, 34, 42], [20, 32, 3])
        add_draw(d(4, 11), "OZ Lotto", [21, 17, 43, 25, 12, 18, 14], [15, 42, 24])

    ozLott()

    # Powerball
    def powerball():
        add_draw(d(18, 12), "Powerball", [31, 25, 19, 2, 35, 15, 16], None, [14])
        add_draw(d(11, 12), "Powerball", [12, 23, 25, 16, 10, 5, 4], None, [10])
        add_draw(d(4, 12), "Powerball", [19, 23, 32, 12, 11, 15, 9], None, [14])
        add_draw(d(27, 11), "Powerball", [2, 17, 11, 9, 19, 28, 24], None, [1])
        add_draw(d(20, 11), "Powerball", [19, 11, 12, 4, 29, 13, 27], None, [20])
        add_draw(d(13, 11), "Powerball", [22, 10, 6, 15, 2, 8, 7], None, [13])
        add_draw(d(6, 11), "Powerball", [11, 34, 7, 33, 15, 22, 16], None, [13])

    powerball()

    # Saturday Lotto
    def saturdayLotto():
        add_draw(d(13, 12), "Saturday Lotto", [28, 20, 35, 17, 32, 6],[41, 25])
        add_draw(d(6, 12), "Saturday Lotto", [17, 42, 5, 10, 33, 45], [31, 44])
        add_draw(d(29, 11), "Saturday Lotto", [22, 10, 17, 5, 44, 36], [3, 11])
        add_draw(d(22, 11), "Saturday Lotto", [7, 31, 15, 39, 42, 12], [5, 8])
        add_draw(d(15, 11), "Saturday Lotto", [36, 19, 33, 41, 39, 1], [25, 20])
        add_draw(d(8, 11), "Saturday Lotto", [28, 13, 1, 41, 14, 16], [39, 34])
        add_draw(d(1, 11), "Saturday Lotto", [42, 31, 21, 28, 17, 13], [36, 15])

    saturdayLotto()

