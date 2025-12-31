import math
import random
import datetime
import itertools
from collections import Counter, defaultdict

# ===========================
# CONFIG (direct from spec)
# ===========================

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX = 47
NUMBER_RANGE = list(range(MAIN_NUMBER_MIN, MAIN_NUMBER_MAX + 1))
LOG_SCORE_MAX = 4.0

base_learning_rate_decade = 0.10
gamma_hop = 0.30     # γ_hop
theta_resurge = 0.25 # θ_resurge

DECADE_BANDS = [
    (1,  1,  9),
    (2, 10, 19),
    (3, 20, 29),
    (4, 30, 39),
    (5, 40, 47),
]

DECADES = [band[0] for band in DECADE_BANDS]
N_DECADES = len(DECADES)

LOTTERIES = {
    "Set for Life":     {"main_draw_size": 7, "uses_supp": True,  "uses_powerball": False},
    "Weekday Windfall": {"main_draw_size": 6, "uses_supp": True,  "uses_powerball": False},
    "OZ Lotto":         {"main_draw_size": 7, "uses_supp": True,  "uses_powerball": False},
    "Powerball":        {"main_draw_size": 7, "uses_supp": False, "uses_powerball": True},
    "Saturday Lotto":   {"main_draw_size": 6, "uses_supp": True,  "uses_powerball": False},
}

HOP_SOURCE_LOTTERY = "Set for Life"
HOP_DESTINATION_LOTTERY = "OZ Lotto"

WINDOW_SIZE_CANDIDATES = [6, 7, 8, 9, 10]

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
    mean = sum(values) / k
    return sum((v - mean) ** 2 for v in values) / k

def decade_of(n):
    if n < MAIN_NUMBER_MIN or n > MAIN_NUMBER_MAX:
        return None
    for d_id, start, end in DECADE_BANDS:
        if start <= n <= end:
            return d_id
    return None

def main_draw_size(lottery_name):
    return LOTTERIES[lottery_name]["main_draw_size"]

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

def add_draw(date, lottery, main, supp=None, powerball=None):
    global_draws.append(Draw(date, lottery, main, supp, powerball))

# Set for Life
add_draw(d(1,12), "Set for Life", [18, 1, 10, 41, 24, 11, 3], [25, 2])
add_draw(d(30,11), "Set for Life", [7, 44, 18, 27, 32, 22, 11], [38, 9])
add_draw(d(29,11), "Set for Life", [8, 31, 4, 6, 42, 16, 14], [13, 19])
add_draw(d(28,11), "Set for Life", [15, 27, 8, 39, 5, 43, 20], [19, 29])
add_draw(d(27,11), "Set for Life", [12, 36, 6, 7, 37, 41, 29], [8, 43])
add_draw(d(26,11), "Set for Life", [29, 37, 34, 14, 5, 21, 20], [18, 19])
add_draw(d(25,11), "Set for Life", [26, 16, 23, 15, 31, 1, 27], [8, 41])
add_draw(d(24,11), "Set for Life", [41, 1, 17, 29, 14, 40, 22], [35, 31])
add_draw(d(23,11), "Set for Life", [25, 27, 42, 18, 26, 9, 33], [22, 19])
add_draw(d(22,11), "Set for Life", [24, 23, 31, 30, 26, 5, 17], [6, 27])
add_draw(d(21,11), "Set for Life", [27, 32, 10, 42, 38, 33, 17], [19, 39])
add_draw(d(20,11), "Set for Life", [28, 10, 11, 35, 34, 41, 23], [30, 26])
add_draw(d(19,11), "Set for Life", [4, 44, 5, 33, 21, 30, 39], [9, 18])
add_draw(d(18,11), "Set for Life", [33, 35, 44, 32, 20, 29, 39], [5, 41])
add_draw(d(17,11), "Set for Life", [15, 23, 40, 43, 28, 1, 37], [18, 34])
add_draw(d(16,11), "Set for Life", [8, 19, 21, 27, 40, 14, 7], [20, 44])
add_draw(d(15,11), "Set for Life", [13, 4, 27, 14, 2, 5, 42], [33, 39])
add_draw(d(14,11), "Set for Life", [7, 25, 23, 35, 13, 18, 6], [3, 39])
add_draw(d(13,11), "Set for Life", [25, 24, 3, 21, 5, 33, 36], [22, 11])
add_draw(d(12,11), "Set for Life", [15, 20, 29, 21, 5, 10, 6], [32, 17])
add_draw(d(11,11), "Set for Life", [4, 7, 10, 44, 32, 30, 26], [5, 18])
add_draw(d(10,11), "Set for Life", [5, 36, 13, 23, 39, 3, 9], [35, 6])
add_draw(d(9,11),  "Set for Life", [11, 4, 44, 26, 6, 31, 40], [21, 33])
add_draw(d(8,11),  "Set for Life", [7, 31, 5, 37, 43, 38, 2], [42, 10])
add_draw(d(7,11),  "Set for Life", [30, 18, 6, 28, 33, 41, 14], [38, 29])
add_draw(d(6,11),  "Set for Life", [12, 20, 35, 42, 41, 10, 18], [33, 32])
add_draw(d(5,11),  "Set for Life", [16, 22, 13, 34, 25, 3, 18], [33, 43])
add_draw(d(4,11),  "Set for Life", [38, 9, 27, 25, 10, 23, 37], [13, 17])
add_draw(d(3,11),  "Set for Life", [8, 15, 25, 26, 13, 24, 23], [4, 2])
add_draw(d(2,11),  "Set for Life", [6, 28, 26, 24, 13, 11, 19], [22, 12])
add_draw(d(1,11),  "Set for Life", [8, 31, 42, 24, 15, 7, 4], [19, 18])

# Weekday Windfall
add_draw(d(1,12), "Weekday Windfall", [8, 6, 30, 38, 36, 1], [43, 5])
add_draw(d(28,11), "Weekday Windfall", [30, 8, 25, 43, 39, 24], [21, 1])
add_draw(d(26,11), "Weekday Windfall", [44, 43, 8, 36, 16, 27], [31, 30])
add_draw(d(24,11), "Weekday Windfall", [44, 15, 20, 17, 4, 18], [7, 11])
add_draw(d(21,11), "Weekday Windfall", [4, 5, 26, 10, 40, 20], [14, 24])
add_draw(d(19,11), "Weekday Windfall", [43, 26, 35, 25, 42, 13], [24, 5])
add_draw(d(17,11), "Weekday Windfall", [37, 11, 4, 2, 5, 7], [30, 22])
add_draw(d(14,11), "Weekday Windfall", [34, 11, 28, 15, 44, 31], [9, 20])
add_draw(d(12,11), "Weekday Windfall", [35, 11, 33, 15, 34, 45], [8, 37])
add_draw(d(10,11), "Weekday Windfall", [38, 3, 31, 22, 28, 5], [26, 14])
add_draw(d(7,11),  "Weekday Windfall", [31, 16, 23, 30, 6, 3], [13, 18])
add_draw(d(5,11),  "Weekday Windfall", [26, 15, 18, 27, 7, 37], [19, 44])
add_draw(d(3,11),  "Weekday Windfall", [25, 14, 29, 23, 45, 13], [31, 8])

# OZ Lotto
add_draw(d(25,11), "OZ Lotto", [12, 43, 28, 1, 47, 35, 14], [15, 16, 46])
add_draw(d(18,11), "OZ Lotto", [39, 2, 22, 8, 27, 6, 4], [47, 5, 24])
add_draw(d(11,11), "OZ Lotto", [44, 30, 7, 28, 17, 34, 42], [20, 32, 3])
add_draw(d(4,11),  "OZ Lotto", [21, 17, 43, 25, 12, 18, 14], [15, 42, 24])

# Powerball
add_draw(d(27,11), "Powerball", [2, 17, 11, 9, 19, 28, 24], None, [1])
add_draw(d(20,11), "Powerball", [19, 11, 12, 4, 29, 13, 27], None, [20])
add_draw(d(13,11), "Powerball", [22, 10, 6, 15, 2, 8, 7], None, [13])
add_draw(d(6,11),  "Powerball", [11, 34, 7, 33, 15, 22, 16], None, [13])

# Saturday Lotto
add_draw(d(29,11), "Saturday Lotto", [22, 10, 17, 5, 44, 36], [3, 11])
add_draw(d(22,11), "Saturday Lotto", [7, 31, 15, 39, 42, 12], [5, 8])
add_draw(d(15,11), "Saturday Lotto", [36, 19, 33, 41, 39, 1], [25, 20])
add_draw(d(8,11),  "Saturday Lotto", [28, 13, 1, 41, 14, 16], [39, 34])
add_draw(d(1,11),  "Saturday Lotto", [42, 31, 21, 28, 17, 13], [36, 15])

# Sort draws by date then lottery for determinism
global_draws.sort(key=lambda dr: (dr.date, dr.lottery))

# Index draws by date
draws_by_date = defaultdict(list)
for dr in global_draws:
    draws_by_date[dr.date].append(dr)

# Target draws and prediction
TARGET_DRAWS_FOR_LEARNING = [
    ("OZ Lotto",         d(25,11)),
    ("Weekday Windfall", d(26,11)),
    ("Powerball",        d(27,11)),
    ("Weekday Windfall", d(28,11)),
    ("Saturday Lotto",   d(29,11)),
    ("Weekday Windfall", d(1,12)),
]

PREDICTION_TARGET = ("OZ Lotto", d(2,12), 7)

def get_actual_main(lottery_name, date):
    for dr in draws_by_date.get(date, []):
        if dr.lottery == lottery_name:
            return [n for n in dr.main if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
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
        self.delta_decade = {d_id: 0.0 for d_id in DECADES}
        self.last_dom_decade = None

    def reset(self):
        self.__init__()

# ===========================
# Step A: build windows
# ===========================

def build_windows(target_date, window_cat, window_dec):
    seed_dates_cat = [target_date - datetime.timedelta(days=delta)
                      for delta in range(window_cat, 0, -1)]
    seed_dates_dec = [target_date - datetime.timedelta(days=delta)
                      for delta in range(window_dec, 0, -1)]

    # abort if any date lacks draws
    for dt in seed_dates_cat:
        if dt not in draws_by_date:
            return None
    for dt in seed_dates_dec:
        if dt not in draws_by_date:
            return None

    seed_draws_cat = []
    for dt in seed_dates_cat:
        seed_draws_cat.extend(draws_by_date[dt])

    seed_draws_dec = []
    for dt in seed_dates_dec:
        seed_draws_dec.extend(draws_by_date[dt])

    return seed_dates_cat, seed_dates_dec, seed_draws_cat, seed_draws_dec

# ===========================
# Step B: category frequencies
# ===========================

def step_B_category(seed_draws_cat, seed_dates_cat):
    seed_numbers_main = []
    seed_numbers_supp = []
    last_main_date = {n: None for n in NUMBER_RANGE}
    lotteries_seen_by_n = {n: set() for n in NUMBER_RANGE}

    for dr in seed_draws_cat:
        # main numbers
        for n in dr.main:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                seed_numbers_main.append(n)
                if (last_main_date[n] is None) or (dr.date > last_main_date[n]):
                    last_main_date[n] = dr.date
                lotteries_seen_by_n[n].add(dr.lottery)
        # supp (including powerball)
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
# Step B: decade frequencies
# ===========================

def step_B_decade(seed_draws_dec):
    dec_seed_numbers_main = []
    for dr in seed_draws_dec:
        for n in dr.main:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                dec_seed_numbers_main.append(n)

    f_main_dec = {n: 0 for n in NUMBER_RANGE}
    for n in dec_seed_numbers_main:
        f_main_dec[n] += 1

    dec_main_count = {d_id: 0 for d_id in DECADES}
    for n, cnt in f_main_dec.items():
        if cnt > 0:
            d_id = decade_of(n)
            if d_id is not None:
                dec_main_count[d_id] += cnt

    total_main_in_range_dec = sum(dec_main_count.values())
    dec_w_window = {}

    if total_main_in_range_dec == 0:
        for d_id in DECADES:
            dec_w_window[d_id] = 1.0
    else:
        dec_freq = {}
        for d_id in DECADES:
            dec_freq[d_id] = dec_main_count[d_id] / total_main_in_range_dec if total_main_in_range_dec > 0 else 0.0
        max_dec_freq = max(dec_freq.values()) if dec_freq else 1.0
        if max_dec_freq == 0:
            max_dec_freq = 1.0
        for d_id in DECADES:
            if dec_main_count[d_id] == 0:
                dec_w_window[d_id] = 0.20
            else:
                dec_w_raw = 0.70 + 0.90 * (dec_freq[d_id] / max_dec_freq)
                dec_w_tmp = 0.75 * dec_w_raw + 0.25 * 1.0
                dec_w_window[d_id] = clamp(dec_w_tmp, 0.50, 1.70)

    return dec_w_window, dec_main_count, total_main_in_range_dec, f_main_dec

def apply_global_decade_learning(dec_w_window, state: LearningState):
    dec_w_final = {}
    for d_id in DECADES:
        dec_w_final[d_id] = dec_w_window[d_id] * math.exp(state.delta_decade[d_id])
    avg_dec_w = sum(dec_w_final.values()) / N_DECADES if N_DECADES > 0 else 1.0
    if avg_dec_w > 0:
        for d_id in DECADES:
            dec_w_final[d_id] /= avg_dec_w
    return dec_w_final

def decade_momentum(dec_w_final, dec_main_count, total_main_in_range_dec,
                    seed_draws_dec, seed_dates_dec, target_date):
    earliest_date = min(seed_dates_dec)
    recent_start_dec = max(target_date - datetime.timedelta(days=3), earliest_date)
    recent_dates_dec = [dt for dt in seed_dates_dec
                        if recent_start_dec <= dt <= target_date - datetime.timedelta(days=1)]

    dec_recent_count = {d_id: 0 for d_id in DECADES}
    for dr in seed_draws_dec:
        if dr.date in recent_dates_dec:
            for n in dr.main:
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    d_id = decade_of(n)
                    if d_id is not None:
                        dec_recent_count[d_id] += 1
    total_recent = sum(dec_recent_count.values())
    rec_w = {d_id: 1.0 for d_id in DECADES}
    if total_recent > 0:
        epsilon = 1e-6
        for d_id in DECADES:
            dec_recent_freq = dec_recent_count[d_id] / total_recent if total_recent > 0 else 0.0
            dec_freq_window = dec_main_count[d_id] / max(total_main_in_range_dec, 1)
            ratio = dec_recent_freq / (dec_freq_window + epsilon)
            rec_w_raw = clamp(ratio, 0.8, 1.3)
            rec_w[d_id] = 0.7 * rec_w_raw + 0.3 * 1.0

    dec_w_recent = {}
    for d_id in DECADES:
        dec_w_recent[d_id] = dec_w_final[d_id] * rec_w[d_id]
    avg_dec_recent = sum(dec_w_recent.values()) / N_DECADES if N_DECADES > 0 else 1.0
    if avg_dec_recent > 0:
        for d_id in DECADES:
            dec_w_recent[d_id] /= avg_dec_recent

    decade_weight_log = {}
    for n in NUMBER_RANGE:
        d_id = decade_of(n)
        if d_id is not None:
            decade_weight_log[n] = math.log(dec_w_recent[d_id])
        else:
            decade_weight_log[n] = 0.0

    return decade_weight_log, dec_w_recent, dec_recent_count

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
                    if L1 == HOP_SOURCE_LOTTERY and L2 != HOP_SOURCE_LOTTERY:
                        w_dir *= 1.5
                    if L2 == HOP_DESTINATION_LOTTERY and L1 != HOP_SOURCE_LOTTERY:
                        w_dir *= 1.3
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
    Warm_set = set(n for n in NUMBER_RANGE if 1 <= f_main[n] <= 2) - Hot_set
    Cold_set = set(n for n in NUMBER_RANGE if f_main[n] == 0)

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
    if avg_hot > avg_warm and avg_hot > avg_cold:
        base_hot, base_warm, base_cold = (1.4, 1.15, 0.6)
    elif avg_warm >= avg_hot and avg_warm >= avg_cold:
        base_hot, base_warm, base_cold = (0.95, 1.35, 1.05)
    elif avg_cold > avg_hot and avg_cold > avg_warm:
        base_hot, base_warm, base_cold = (0.8, 1.0, 1.4)
    else:
        base_hot, base_warm, base_cold = (1.0, 1.0, 0.95)

    hot_w  = base_hot  * (1 + (avg_hot  - 1/3) * 0.25)
    warm_w = base_warm * (1 + (avg_warm - 1/3) * 0.25)
    cold_w = base_cold * (1 + (avg_cold - 1/3) * 0.25)

    hot_w  += state.delta_hot
    warm_w += state.delta_warm
    cold_w += state.delta_cold

    hot_w  = clamp(hot_w,  0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    s = hot_w + warm_w + cold_w
    if s > 0:
        factor = s / 3.0
        hot_w  /= factor
        warm_w /= factor
        cold_w /= factor

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
                          last_main_date, target_date,
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
        adj_score_raw = 0.05 + 0.25 * (adj_count[n] / max_adj)
        adj_log[n] = math.log(1 + adj_score_raw)

    # last-3-days delta
    last3_start = target_date - datetime.timedelta(days=3)
    main_hits_last3 = {n: False for n in NUMBER_RANGE}
    supp_hits_last3 = {n: False for n in NUMBER_RANGE}
    for dt, draws in draws_by_date.items():
        if last3_start <= dt <= target_date - datetime.timedelta(days=1):
            for dr in draws:
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
    delta_log = {}
    for n in NUMBER_RANGE:
        if main_hits_last3[n]:
            mult = 1.4
        elif supp_hits_last3[n]:
            mult = 1.2
        else:
            mult = 1.0
        delta_log[n] = math.log(mult)

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

    # total log score
    log_score = {}
    rawP = {}
    for n in NUMBER_RANGE:
        ls = (adj_log[n] +
              delta_log[n] +
              cross_log[n] +
              supp_log[n] +
              category_weight_log[n] +
              decade_weight_log[n] +
              cross_hop_log[n] +
              cold_resurge_log[n])
        ls = min(ls, LOG_SCORE_MAX)
        log_score[n] = ls
        rawP[n] = math.exp(ls)
    total_rawP = sum(rawP.values())
    if total_rawP <= 0:
        P = {n: 1.0 / len(NUMBER_RANGE) for n in NUMBER_RANGE}
    else:
        P = {n: rawP[n] / total_rawP for n in NUMBER_RANGE}

    return P, log_score, cold_resurge_score

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
            base = 1 + 0.2 * (freq - 1)
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

    # clamp >=0
    h_target = max(0, h_target)
    w_target = max(0, w_target)
    c_target = max(0, c_target)

    # clamp by capacity
    h_target = min(h_target, len(Hot_set))
    w_target = min(w_target, len(Warm_set))
    c_target = min(c_target, len(Cold_set))

    def total():
        return h_target + w_target + c_target

    # mutable using list
    h, w, c = h_target, w_target, c_target
    while total() < draw_size:
        cap_hot = len(Hot_set) - h
        cap_warm = len(WarmSet) - w if False else len(Warm_set) - w  # keep logic explicit
        cap_warm = len(Warm_set) - w
        cap_cold = len(Cold_set) - c
        caps = {
            "Warm": cap_warm,
            "Hot": cap_hot,
            "Cold": cap_cold,
        }
        # choose category with largest remaining capacity (Warm > Hot > Cold tie-break via dict order)
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

    # final sanity
    if h_target > len(Hot_set) or w_target > len(Warm_set) or c_target > len(Cold_set):
        raise RuntimeError("Composition targets exceed category capacities")

    return h_target, w_target, c_target, draw_size

# ===========================
# Step 10: Monte Carlo sampling
# ===========================

def sample_from_category(category_list, probs_dict, k, chosen_set):
    """Sample k distinct numbers from category_list chosen_set according to probs_dict."""
    selected = []
    while len(selected) < k:
        candidates = [n for n in category_list if n not in chosen_set and n not in selected]
        if not candidates:
            raise RuntimeError("Not enough candidates to sample from category")
        weights = [probs_dict.get(n, 0.0) for n in candidates]
        total_w = sum(weights)
        if total_w <= 0:
            # uniform fallback
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
                         clusters, M, avg_hot, avg_warm, avg_cold):
    # TRIALS computation
    complexity = math.sqrt(max(1.0, M / 20.0))
    TRIALS = int(clamp(50000 * complexity, 10000, 50000))

    max_avg = max(avg_hot, avg_warm, avg_cold)
    EXPLORE_FRAC = clamp(0.10 + 0.20 * max(0.0, max_avg - 1/3), 0.10, 0.20)

    COLD_EXPLORE_MULT = 1.25
    CLUSTER_LAMBDA = 0.15

    baseP = dict(P)

    # Precompute clusters per number
    clusters_by_n = {n: [] for n in NUMBER_RANGE}
    for C, priority in clusters.items():
        for n in C:
            clusters_by_n[n].append((C, priority))

    freq = Counter()
    sum_hot = 0.0
    sum_warm = 0.0
    sum_cold = 0.0
    sum_decade = {d_id: 0.0 for d_id in DECADES}

    Hot_list = list(Hot_set)
    Warm_list = list(Warm_set)
    Cold_list = list(Cold_set)

    for _ in range(TRIALS):
        u = random.random()
        mode = "explore" if u < EXPLORE_FRAC else "normal"

        # 2) W(n)
        W = {}
        for n in NUMBER_RANGE:
            w = baseP[n]
            if mode == "explore" and n in Cold_set:
                w *= COLD_EXPLORE_MULT
            # cluster boost
            C_n = clusters_by_n.get(n, [])
            if not C_n:
                cluster_boost = 1.0
            else:
                S_n = sum(priority - 1.0 for (_, priority) in C_n)
                cluster_boost = math.exp(CLUSTER_LAMBDA * S_n)
            w *= cluster_boost
            W[n] = w

        # 4) per-category normalised probabilities
        p_H = {}
        p_W = {}
        p_C = {}

        if Hot_list:
            Z_H = sum(W[n] for n in Hot_list)
            if Z_H > 0:
                for n in Hot_list:
                    p_H[n] = W[n] / Z_H
            else:
                for n in Hot_list:
                    p_H[n] = 1.0 / len(Hot_list)
        if Warm_list:
            Z_W = sum(W[n] for n in Warm_list)
            if Z_W > 0:
                for n in Warm_list:
                    p_W[n] = W[n] / Z_W
            else:
                for n in Warm_list:
                    p_W[n] = 1.0 / len(Warm_list)
        if Cold_list:
            Z_C = sum(W[n] for n in Cold_list)
            if Z_C > 0:
                for n in Cold_list:
                    p_C[n] = W[n] / Z_C
            else:
                for n in Cold_list:
                    p_C[n] = 1.0 / len(Cold_list)

        # 5) capacities check already done in composition_targets

        chosen = set()

        if h_target > 0:
            selected_hot = sample_from_category(Hot_list, p_H, h_target, chosen)
            chosen.update(selected_hot)
        if w_target > 0:
            selected_warm = sample_from_category(Warm_list, p_W, w_target, chosen)
            chosen.update(selected_warm)
        if c_target > 0:
            selected_cold = sample_from_category(Cold_list, p_C, c_target, chosen)
            chosen.update(selected_cold)

        if len(chosen) != draw_size:
            raise RuntimeError("Implementation error: chosen size mismatch draw_size")

        T = tuple(sorted(chosen))
        freq[T] += 1

        # stats
        sum_hot  += sum(1 for n in chosen if n in Hot_set)
        sum_warm += sum(1 for n in chosen if n in Warm_set)
        sum_cold += sum(1 for n in chosen if n in Cold_set)
        for n in chosen:
            d_id = decade_of(n)
            if d_id is not None:
                sum_decade[d_id] += 1

    prob = {T: c / TRIALS for T, c in freq.items()}
    hot_pred = sum_hot / TRIALS
    warm_pred = sum_warm / TRIALS
    cold_pred = sum_cold / TRIALS
    dec_pred = {d_id: sum_decade[d_id] / TRIALS for d_id in DECADES}

    # Top-10 tuples
    top10 = sorted(prob.items(), key=lambda kv: kv[1], reverse=True)[:10]

    diagnostics = {
        "TRIALS": TRIALS,
        "EXPLORE_FRAC": EXPLORE_FRAC,
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "dec_pred": dec_pred,
        "prob": prob,
        "top10": top10,
    }

    return diagnostics

# ===========================
# Step 12: learning feedback
# ===========================

def learning_step(state: LearningState,
                  Hot_set, Warm_set, Cold_set,
                  f_main, dec_pred,
                  actual_main):
    actual_set = set(actual_main)

    hot_actual  = sum(1 for n in actual_main if n in Hot_set)
    warm_actual = sum(1 for n in actual_main if n in Warm_set)
    cold_actual = sum(1 for n in actual_main if n in Cold_set)

    # variance over f_main including zeros
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
                   clusters, actual_main,
                   dec_pred):
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

    # decade learning
    dec_actual = {d_id: 0 for d_id in DECADES}
    for n in actual_main:
        d_id = decade_of(n)
        if d_id is not None:
            dec_actual[d_id] += 1

    for d_id in DECADES:
        dec_error = dec_actual[d_id] - dec_pred[d_id]
        step_d = clamp(base_learning_rate_decade * dec_error, -0.15, 0.15)
        state.delta_decade[d_id] += step_d

    # oscillation regulariser
    decades_sorted = sorted(DECADES)
    k = len(decades_sorted) // 2
    LOW = set(decades_sorted[:k])
    HIGH = set(decades_sorted[k:])

    # dominant decade by actual count
    dom_decade = min(DECADES, key=lambda d_id: (-dec_actual[d_id], d_id))

    if state.last_dom_decade is not None and dom_decade == state.last_dom_decade:
        if dom_decade in LOW:
            for dh in HIGH:
                state.delta_decade[dh] += 0.03
            for dl in LOW:
                state.delta_decade[dl] -= 0.03
        else:
            for dl in LOW:
                state.delta_decade[dl] += 0.03
            for dh in HIGH:
                state.delta_decade[dh] -= 0.03

    state.last_dom_decade = dom_decade

    # clamp and de-mean
    for d_id in DECADES:
        state.delta_decade[d_id] = clamp(state.delta_decade[d_id], -0.8, 0.8)
    mean_delta = sum(state.delta_decade[d_id] for d_id in DECADES) / N_DECADES
    for d_id in DECADES:
        state.delta_decade[d_id] -= mean_delta

# ===========================
# Per-target full processing
# ===========================

def process_target(lottery_name, target_date,
                   window_cat, window_dec,
                   state: LearningState,
                   do_learning=True):
    built = build_windows(target_date, window_cat, window_dec)
    if built is None:
        return None  # aborted
    seed_dates_cat, seed_dates_dec, seed_draws_cat, seed_draws_dec = built

    # Step 4: frequencies
    f_main, f_supp, L_counts, last_main_date, M, seed_numbers_main = step_B_category(seed_draws_cat, seed_dates_cat)
    dec_w_window, dec_main_count, total_main_in_range_dec, f_main_dec = step_B_decade(seed_draws_dec)
    dec_w_final = apply_global_decade_learning(dec_w_window, state)
    decade_weight_log, dec_w_recent, dec_recent_count = decade_momentum(
        dec_w_final, dec_main_count, total_main_in_range_dec,
        seed_draws_dec, seed_dates_dec, target_date
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
        last_main_date, target_date,
        Hot_set, Warm_set, Cold_set,
        hot_w, warm_w, cold_w,
        decade_weight_log, cross_hop_log,
        window_cat
    )

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
        clusters, M_val, avg_hot, avg_warm, avg_cold
    )

    # Diagnostics of interest
    hot_pred = mc_diag["hot_pred"]
    warm_pred = mc_diag["warm_pred"]
    cold_pred = mc_diag["cold_pred"]
    dec_pred = mc_diag["dec_pred"]
    top10 = mc_diag["top10"]

    result = {
        "seed_dates_cat": seed_dates_cat,
        "seed_dates_dec": seed_dates_dec,
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
        "top10": top10,
        "dec_main_count": dec_main_count,
        "dec_w_window": dec_w_window,
        "dec_w_final": dec_w_final,
        "dec_w_recent": dec_w_recent,
        "cross_hop_score": cross_hop_score,
        "cold_resurge_score": cold_resurge_score,
        "f_main": f_main,
        "clusters": clusters,
    }

    if not do_learning:
        return result

    # Step 12 learning
    actual_main = get_actual_main(lottery_name, target_date)
    if actual_main is None:
        return result  # nothing to learn from

    learn_meta = learning_step(state, Hot_set, Warm_set, Cold_set, f_main, dec_pred, actual_main)
    hot_actual = learn_meta["hot_actual"]
    cold_actual = learn_meta["cold_actual"]
    learning_rate = learn_meta["learning_rate"]

    apply_learning(
        state,
        hot_actual, cold_actual,
        hot_pred, cold_pred,
        learning_rate,
        clusters, actual_main,
        dec_pred
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
                                 W_cat, W_cat, state,
                                 do_learning=True)
            if res is None:
                continue
            actual_main = get_actual_main(lottery_name, target_date)
            if actual_main is None:
                continue
            Hot_set = res["Hot_set"]
            Warm_set = res["Warm_set"]
            Cold_set = res["Cold_set"]
            hot_actual = sum(1 for n in actual_main if n in Hot_set)
            warm_actual = sum(1 for n in actual_main if n in Warm_set)
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

def calibration_decade_window(WINDOW_SIZE_CAT_star):
    alpha = 1.0
    beta = 0.5
    best_W = None
    best_score = None
    for W_dec in WINDOW_SIZE_CANDIDATES:
        state = LearningState()
        dec_errors_all = []
        z_values = []
        for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
            res = process_target(lottery_name, target_date,
                                 WINDOW_SIZE_CAT_star, W_dec,
                                 state, do_learning=True)
            if res is None:
                continue
            actual_main = get_actual_main(lottery_name, target_date)
            if actual_main is None:
                continue
            dec_pred = res["dec_pred"]
            dec_actual = {d_id: 0 for d_id in DECADES}
            for n in actual_main:
                d_id = decade_of(n)
                if d_id is not None:
                    dec_actual[d_id] += 1
            for d_id in DECADES:
                dec_errors_all.append((dec_actual[d_id] - dec_pred[d_id]) ** 2)

            decades_sorted = sorted(DECADES)
            k = len(decades_sorted) // 2
            LOW = set(decades_sorted[:k])
            HIGH = set(decades_sorted[k:])
            low_t = sum(dec_pred[d_id] for d_id in LOW)
            high_t = sum(dec_pred[d_id] for d_id in HIGH)
            z_values.append(low_t - high_t)
        if not dec_errors_all:
            continue
        decade_mse = sum(dec_errors_all) / len(dec_errors_all)
        stability_penalty = variance_population(z_values) if z_values else 0.0
        score = alpha * decade_mse + beta * stability_penalty
        if best_score is None or score < best_score or (score == best_score and (best_W is None or W_dec < best_W)):
            best_score = score
            best_W = W_dec
    if best_W is None:
        raise RuntimeError("No feasible WINDOW_SIZE_DEC found during calibration")
    return best_W, best_score

# ===========================
# Main run + final prediction
# ===========================

def main():
    random.seed(0)

    # Calibration
    W_cat_star, mse_cat = calibration_category_window()
    print(f"Chosen WINDOW_SIZE_CAT* = {W_cat_star} (category MSE = {mse_cat:.6f})")

    W_dec_star, score_dec = calibration_decade_window(W_cat_star)
    print(f"Chosen WINDOW_SIZE_DEC* = {W_dec_star} (score_dec = {score_dec:.6f})")

    # Main run with chosen windows
    state = LearningState()
    for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
        res = process_target(lottery_name, target_date,
                             W_cat_star, W_dec_star,
                             state, do_learning=True)
        if res is None:
            print(f"[MAIN RUN] Skipped {lottery_name} on {target_date} (incomplete window)")
        else:
            print(f"[MAIN RUN] Processed {lottery_name} on {target_date}")

    # Final prediction
    prediction_lottery_name, prediction_date, prediction_draw_size = PREDICTION_TARGET
    if prediction_draw_size != main_draw_size(prediction_lottery_name):
        raise RuntimeError("Config error: prediction_draw_size != main_draw_size")

    res_pred = process_target(prediction_lottery_name, prediction_date,
                              W_cat_star, W_dec_star,
                              state, do_learning=False)
    if res_pred is None:
        raise RuntimeError("Prediction aborted due to incomplete window")

    print("\n=== FINAL PREDICTION ===")
    print(f"Lottery: {prediction_lottery_name}")
    print(f"Date:    {prediction_date}")
    print(f"Draw size: {prediction_draw_size}")
    print("\nTop-10 predicted tuples (numbers, probability):")
    for T, p in res_pred["top10"]:
        print(f"{list(T)}  prob={p:.6f}")

if __name__ == "__main__":
    main()
