import math
import random
import itertools
import collections
import statistics
from datetime import datetime, date, timedelta
import numpy as np

# FIXED SEED — deterministic Monte Carlo
SEED_MAIN = 29112025
random.seed(SEED_MAIN)
np.random.seed(SEED_MAIN)
print("USING FIXED SEED:", SEED_MAIN)

# =========================
# CONFIG (Section A–D)
# =========================

MAIN_NUMBER_MIN = 1
MAIN_NUMBER_MAX = 45
NUMBER_RANGE = list(range(MAIN_NUMBER_MIN, MAIN_NUMBER_MAX + 1))

LOG_SCORE_MAX = 4.0  # Cap for per-number log_score


base_learning_rate_decade = 0.10   # Step 12 decade learning rate

gamma_hop = 0.30   # γ_hop
theta_resurge = 0.25  # θ_resurge

# Decade bands (Section B)
DECADE_BANDS = [
    (1, 1, 9),
    (2, 10, 19),
    (3, 20, 29),
    (4, 30, 39),
    (5, 40, 45),
]

DECADES = [band[0] for band in DECADE_BANDS]
N_DECADES = len(DECADES)

def decade(n):
    if n < MAIN_NUMBER_MIN or n > MAIN_NUMBER_MAX:
        return None
    for d_id, start, end in DECADE_BANDS:
        if start <= n <= end:
            return d_id
    return None

# Lottery types (Section C)
LOTTERIES = [
    {"name": "Set for Life",     "main_draw_size": 7, "uses_supp": True,  "uses_powerball": False},
    {"name": "Weekday Windfall", "main_draw_size": 6, "uses_supp": True,  "uses_powerball": False},
    {"name": "OZ Lotto",         "main_draw_size": 7, "uses_supp": True,  "uses_powerball": False},
    {"name": "Powerball",        "main_draw_size": 7, "uses_supp": False, "uses_powerball": True},
    {"name": "Saturday Lotto",   "main_draw_size": 6, "uses_supp": True,  "uses_powerball": False},
]

LOTTERY_INFO = {lt["name"]: lt for lt in LOTTERIES}

def main_draw_size(lottery_name):
    return LOTTERY_INFO[lottery_name]["main_draw_size"]

# Cross-lottery hop roles (Section D)
HOP_SOURCE_LOTTERY = "Set for Life"
HOP_DESTINATION_LOTTERY = "Saturday Lotto"

# =========================
# Helper math functions
# =========================

def clamp(x, a, b):
    return max(a, min(b, x))

def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0

def variance(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)

# =========================
# Historical draws (Section E)
# =========================

def parse_date(s):
    # Examples: "Fri, 28 Nov 2025"
    return datetime.strptime(s, "%a, %d %b %Y").date()

def make_draw(lottery, date_str, mains, supp=None, powerball=None):
    return {
        "lottery": lottery,
        "date": parse_date(date_str),
        "main": [n for n in mains if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX],
        "supp": [n for n in (supp or []) if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX],
        "powerball": [n for n in (powerball or []) if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX],
    }

GLOBAL_DRAWS = []

# Set for Life
GLOBAL_DRAWS += [
    make_draw("Set for Life", "Fri, 28 Nov 2025", [15, 27, 8, 39, 5, 43, 20], [19, 29]),
    make_draw("Set for Life", "Thu, 27 Nov 2025", [12, 36, 6, 7, 37, 41, 29], [8, 43]),
    make_draw("Set for Life", "Wed, 26 Nov 2025", [29, 37, 34, 14, 5, 21, 20], [18, 19]),
    make_draw("Set for Life", "Tue, 25 Nov 2025", [26, 16, 23, 15, 31, 1, 27], [8, 41]),
    make_draw("Set for Life", "Mon, 24 Nov 2025", [41, 1, 17, 29, 14, 40, 22], [35, 31]),
    make_draw("Set for Life", "Sun, 23 Nov 2025", [25, 27, 42, 18, 26, 9, 33], [22, 19]),
    make_draw("Set for Life", "Sat, 22 Nov 2025", [24, 23, 31, 30, 26, 5, 17], [6, 27]),
    make_draw("Set for Life", "Fri, 21 Nov 2025", [27, 32, 10, 42, 38, 33, 17], [19, 39]),
    make_draw("Set for Life", "Thu, 20 Nov 2025", [28, 10, 11, 35, 34, 41, 23], [30, 26]),
    make_draw("Set for Life", "Wed, 19 Nov 2025", [4, 44, 5, 33, 21, 30, 39], [9, 18]),
    make_draw("Set for Life", "Tue, 18 Nov 2025", [33, 35, 44, 32, 20, 29, 39], [5, 41]),
    make_draw("Set for Life", "Mon, 17 Nov 2025", [15, 23, 40, 43, 28, 1, 37], [18, 34]),
    make_draw("Set for Life", "Sun, 16 Nov 2025", [8, 19, 21, 27, 40, 14, 7], [20, 44]),
    make_draw("Set for Life", "Sat, 15 Nov 2025", [13, 4, 27, 14, 2, 5, 42], [33, 39]),
    make_draw("Set for Life", "Fri, 14 Nov 2025", [7, 25, 23, 35, 13, 18, 6], [3, 39]),
    make_draw("Set for Life", "Thu, 13 Nov 2025", [25, 24, 3, 21, 5, 33, 36], [22, 11]),
    make_draw("Set for Life", "Wed, 12 Nov 2025", [15, 20, 29, 21, 5, 10, 6], [32, 17]),
]

# Weekday Windfall
GLOBAL_DRAWS += [
    make_draw("Weekday Windfall", "Fri, 28 Nov 2025", [30, 8, 25, 43, 39, 24], [21, 1]),
    make_draw("Weekday Windfall", "Wed, 26 Nov 2025", [44, 43, 8, 36, 16, 27], [31, 30]),
    make_draw("Weekday Windfall", "Mon, 24 Nov 2025", [44, 15, 20, 17, 4, 18], [7, 11]),
    make_draw("Weekday Windfall", "Fri, 21 Nov 2025", [4, 5, 26, 10, 40, 20], [14, 24]),
    make_draw("Weekday Windfall", "Wed, 19 Nov 2025", [43, 26, 35, 25, 42, 13], [24, 5]),
    make_draw("Weekday Windfall", "Mon, 17 Nov 2025", [37, 11, 4, 2, 5, 7], [30, 22]),
    make_draw("Weekday Windfall", "Fri, 14 Nov 2025", [34, 11, 28, 15, 44, 31], [9, 20]),
    make_draw("Weekday Windfall", "Wed, 12 Nov 2025", [35, 11, 33, 15, 34, 45], [8, 37]),
]

# OZ Lotto
GLOBAL_DRAWS += [
    make_draw("OZ Lotto", "Tue, 25 Nov 2025", [12, 43, 28, 1, 47, 35, 14], [15, 16, 46]),
    make_draw("OZ Lotto", "Tue, 18 Nov 2025", [39, 2, 22, 8, 27, 6, 4], [47, 5, 24]),
]

# Powerball
GLOBAL_DRAWS += [
    make_draw("Powerball", "Thu, 27 Nov 2025", [2, 17, 11, 9, 19, 28, 24], [], [1]),
    make_draw("Powerball", "Thu, 20 Nov 2025", [19, 11, 12, 4, 29, 13, 27], [], [20]),
    make_draw("Powerball", "Thu, 13 Nov 2025", [22, 10, 6, 15, 2, 8, 7], [], [13]),
]

# Saturday Lotto
GLOBAL_DRAWS += [
    make_draw("Saturday Lotto", "Sat, 22 Nov 2025", [7, 31, 15, 39, 42, 12], [5, 8]),
    make_draw("Saturday Lotto", "Sat, 15 Nov 2025", [36, 19, 33, 41, 39, 1], [25, 20]),
]

# Sort all draws by date
GLOBAL_DRAWS.sort(key=lambda d: d["date"])

# Build date index
DRAWS_BY_DATE = collections.defaultdict(list)
for d in GLOBAL_DRAWS:
    DRAWS_BY_DATE[d["date"]].append(d)

# =========================
# Target draws & prediction (Sections F, G, H)
# =========================

TARGET_DRAWS_FOR_LEARNING = [
    ("Saturday Lotto",   parse_date("Sat, 22 Nov 2025")),
    ("Weekday Windfall", parse_date("Mon, 24 Nov 2025")),
    ("OZ Lotto",         parse_date("Tue, 25 Nov 2025")),
    ("Weekday Windfall", parse_date("Wed, 26 Nov 2025")),
    ("Powerball",        parse_date("Thu, 27 Nov 2025")),
    ("Weekday Windfall", parse_date("Fri, 28 Nov 2025")),
]

PREDICTION_TARGET = (
    "Saturday Lotto",
    parse_date("Sat, 29 Nov 2025"),
    6,
)

WINDOW_SIZE_CANDIDATES = [6, 7, 8, 9, 10]

# =========================
# Core window-building and per-draw processing
# =========================

class LearningState:
    def __init__(self):
        self.delta_hot = 0.0
        self.delta_warm = 0.0
        self.delta_cold = 0.0
        self.cluster_priority_score_global = {}  # cluster tuple -> score
        self.delta_decade = {d: 0.0 for d in DECADES}
        self.last_dom_decade = None

    def reset(self):
        self.__init__()

def get_seed_dates(target_date, window_size):
    # inclusive range [D_t - W, D_t - 1]
    return [target_date - timedelta(days=i) for i in range(window_size, 0, -1)]

def get_seed_draws(seed_dates):
    draws = []
    for dt in seed_dates:
        day_draws = DRAWS_BY_DATE.get(dt, [])
        if not day_draws:
            return None  # abort: missing date with no draws
        draws.extend(day_draws)
    return draws

def build_category_and_decade_windows(target_date, window_size_cat, window_size_dec):
    seed_dates_cat = get_seed_dates(target_date, window_size_cat)
    seed_dates_dec = get_seed_dates(target_date, window_size_dec)
    seed_draws_cat = get_seed_draws(seed_dates_cat)
    seed_draws_dec = get_seed_draws(seed_dates_dec)
    if seed_draws_cat is None or seed_draws_dec is None:
        return None
    return seed_dates_cat, seed_draws_cat, seed_dates_dec, seed_draws_dec

def stepB_category_frequencies(seed_draws_cat, seed_dates_cat):
    seed_numbers_main = []
    seed_numbers_supp = []
    L_map = {n: set() for n in NUMBER_RANGE}
    last_main_date = {n: None for n in NUMBER_RANGE}

    for d in seed_draws_cat:
        lot = d["lottery"]
        dt = d["date"]
        mains = [n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        supps = [n for n in d["supp"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        # Include Powerball extras as supp if within NUMBER_RANGE
        pballs = [n for n in d["powerball"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        seed_numbers_main.extend(mains)
        seed_numbers_supp.extend(supps)
        seed_numbers_supp.extend(pballs)
        for n in set(mains + supps + pballs):
            if n in L_map:
                L_map[n].add(lot)
        for n in mains:
            if n in last_main_date:
                if last_main_date[n] is None or dt > last_main_date[n]:
                    last_main_date[n] = dt

    f_main = {n: 0 for n in NUMBER_RANGE}
    f_supp = {n: 0 for n in NUMBER_RANGE}
    for n in seed_numbers_main:
        if n in f_main:
            f_main[n] += 1
    for n in seed_numbers_supp:
        if n in f_supp:
            f_supp[n] += 1

    L = {n: len(L_map[n]) for n in NUMBER_RANGE}
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)

    return {
        "seed_numbers_main": seed_numbers_main,
        "seed_numbers_supp": seed_numbers_supp,
        "f_main": f_main,
        "f_supp": f_supp,
        "L": L,
        "M": M,
        "last_main_date": last_main_date,
    }

def stepB_decade_window(seed_draws_dec):
    dec_seed_numbers_main = []
    for d in seed_draws_dec:
        mains = [n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        dec_seed_numbers_main.extend(mains)

    f_main_dec = {n: 0 for n in NUMBER_RANGE}
    for n in dec_seed_numbers_main:
        f_main_dec[n] += 1

    dec_main_count = {d: 0 for d in DECADES}
    for n, cnt in f_main_dec.items():
        d_id = decade(n)
        if d_id is not None:
            dec_main_count[d_id] += cnt
    total_main_in_range_dec = sum(dec_main_count.values())

    if total_main_in_range_dec == 0:
        dec_w_window = {d: 1.0 for d in DECADES}
    else:
        dec_freq = {d: dec_main_count[d] / total_main_in_range_dec for d in DECADES}
        max_dec_freq = max(dec_freq.values())
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

    return {
        "dec_seed_numbers_main": dec_seed_numbers_main,
        "f_main_dec": f_main_dec,
        "dec_main_count": dec_main_count,
        "total_main_in_range_dec": total_main_in_range_dec,
        "dec_w_window": dec_w_window,
    }

def stepB_apply_global_decade_learning(dec_w_window, learning_state):
    dec_w_final = {}
    for d in DECADES:
        dec_w_final[d] = dec_w_window[d] * math.exp(learning_state.delta_decade[d])
    avg_dec_w = sum(dec_w_final.values()) / N_DECADES if N_DECADES > 0 else 1.0
    if avg_dec_w > 0:
        for d in DECADES:
            dec_w_final[d] /= avg_dec_w
    return dec_w_final

def stepB_last3_momentum(target_date, seed_dates_dec, dec_main_count, total_main_in_range_dec, dec_w_final):
    if not seed_dates_dec:
        dec_w_recent = dec_w_final.copy()
        return dec_w_recent, {d: 0 for d in DECADES}, {d: 0.0 for d in DECADES}
    earliest = min(seed_dates_dec)
    recent_start_dec = max(target_date - timedelta(days=3), earliest)
    recent_dates_dec = [recent_start_dec + timedelta(days=i)
                        for i in range((target_date - recent_start_dec).days)]
    dec_recent_count = {d: 0 for d in DECADES}
    for dt in recent_dates_dec:
        for draw in DRAWS_BY_DATE.get(dt, []):
            for n in draw["main"]:
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    d_id = decade(n)
                    if d_id is not None:
                        dec_recent_count[d_id] += 1
    total_recent = sum(dec_recent_count.values())
    if total_recent == 0:
        rec_w = {d: 1.0 for d in DECADES}
    else:
        dec_recent_freq = {d: dec_recent_count[d] / total_recent for d in DECADES}
        dec_freq_window = {d: dec_main_count[d] / max(total_main_in_range_dec, 1)
                           for d in DECADES}
        eps = 1e-6
        ratio = {d: dec_recent_freq[d] / (dec_freq_window[d] + eps) for d in DECADES}
        rec_w_raw = {d: clamp(ratio[d], 0.8, 1.3) for d in DECADES}
        rec_w = {d: 0.7 * rec_w_raw[d] + 0.3 * 1.0 for d in DECADES}

    dec_w_recent = {}
    for d in DECADES:
        dec_w_recent[d] = dec_w_final[d] * rec_w[d]
    avg_dec_recent = sum(dec_w_recent.values()) / N_DECADES if N_DECADES > 0 else 1.0
    if avg_dec_recent > 0:
        for d in DECADES:
            dec_w_recent[d] /= avg_dec_recent
    return dec_w_recent, dec_recent_count, rec_w

def stepB_deacde_weight_log(dec_w_recent):
    decade_weight_log = {n: 0.0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        d_id = decade(n)
        if d_id is not None:
            decade_weight_log[n] = math.log(dec_w_recent[d_id])
    return decade_weight_log

def stepB_cross_hop(seed_draws_cat, window_size_cat):
    # appearances: n -> list of (date, lottery)
    appearances = {n: [] for n in NUMBER_RANGE}
    for d in seed_draws_cat:
        dt = d["date"]
        lot = d["lottery"]
        for n in d["main"]:
            if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                appearances[n].append((dt, lot))

    cross_pair_sum = {n: 0.0 for n in NUMBER_RANGE}
    sfl_count = {n: 0 for n in NUMBER_RANGE}
    non_sfl_count = {n: 0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        aps = appearances[n]
        if not aps:
            continue
        for dt, lot in aps:
            if lot == HOP_SOURCE_LOTTERY:
                sfl_count[n] += 1
            else:
                non_sfl_count[n] += 1
        # ordered pairs
        for (d1, L1), (d2, L2) in itertools.permutations(aps, 2):
            if d2 <= d1:
                continue
            lag_days = (d2 - d1).days
            if lag_days <= 0 or lag_days > window_size_cat:
                continue
            base_pair = 1.0 / max(1, lag_days)
            w_dir = 1.0
            if L1 == HOP_SOURCE_LOTTERY and L2 != HOP_SOURCE_LOTTERY:
                w_dir *= 1.5
            if L2 == HOP_DESTINATION_LOTTERY and L1 != HOP_SOURCE_LOTTERY:
                w_dir *= 1.3
            cross_pair_sum[n] += base_pair * w_dir

    base_hop_score = {}
    for n in NUMBER_RANGE:
        base_hop_score[n] = sfl_count[n] * non_sfl_count[n] + cross_pair_sum[n]
    max_hop_score = max(base_hop_score.values()) if base_hop_score else 0.0
    cross_hop_score = {n: 0.0 for n in NUMBER_RANGE}
    cross_hop_log = {n: 0.0 for n in NUMBER_RANGE}
    if max_hop_score > 0:
        for n in NUMBER_RANGE:
            s = base_hop_score[n] / max_hop_score
            cross_hop_score[n] = s
            if s > 0:
                cross_hop_log[n] = math.log(1.0 + gamma_hop * s)
    return cross_hop_score, cross_hop_log

def stepC_hot_warm_cold(f_main, L, seed_draws_cat):
    M = sum(1 for n in NUMBER_RANGE if f_main[n] > 0)
    K = max(1, round(max(3, M * 0.15)))
    # Sort for Hot_set
    candidates = list(NUMBER_RANGE)
    candidates.sort(key=lambda n: (-f_main[n], -L[n], n))
    Hot_set = set()
    for n in candidates:
        if f_main[n] > 0 and len(Hot_set) < K:
            Hot_set.add(n)
    if len(Hot_set) < K:
        zeros = [n for n in candidates if f_main[n] == 0 and n not in Hot_set]
        for n in zeros:
            if len(Hot_set) < K:
                Hot_set.add(n)
            else:
                break
    Warm_set = set(n for n in NUMBER_RANGE if 1 <= f_main[n] <= 2) - Hot_set
    Cold_set = set(NUMBER_RANGE) - Hot_set - Warm_set

    # Per-draw ratios
    p_hot_list = []
    p_warm_list = []
    p_cold_list = []
    for d in seed_draws_cat:
        mains = [n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
        if not mains:
            continue
        sz = len(mains)
        c_hot = sum(1 for n in mains if n in Hot_set)
        c_warm = sum(1 for n in mains if n in Warm_set)
        c_cold = sum(1 for n in mains if n in Cold_set)
        p_hot_list.append(c_hot / sz)
        p_warm_list.append(c_warm / sz)
        p_cold_list.append(c_cold / sz)
    if not p_hot_list:
        avg_hot = avg_warm = avg_cold = 1.0 / 3.0
    else:
        avg_hot = sum(p_hot_list) / len(p_hot_list)
        avg_warm = sum(p_warm_list) / len(p_warm_list)
        avg_cold = sum(p_cold_list) / len(p_cold_list)

    # Bias
    if avg_warm >= avg_hot and avg_warm >= avg_cold:
        bias = "warm-heavy"
    elif avg_hot > avg_warm and avg_hot > avg_cold:
        bias = "hot-heavy"
    elif avg_cold > avg_hot and avg_cold > avg_warm:
        bias = "cold-heavy"
    else:
        bias = "balanced"

    return Hot_set, Warm_set, Cold_set, avg_hot, avg_warm, avg_cold, bias, K, M

def stepD_category_weights(avg_hot, avg_warm, avg_cold, bias, learning_state):
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

    hot_w += learning_state.delta_hot
    warm_w += learning_state.delta_warm
    cold_w += learning_state.delta_cold

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    s = hot_w + warm_w + cold_w
    if s > 0:
        hot_w /= (s / 3.0)
        warm_w /= (s / 3.0)
        cold_w /= (s / 3.0)

    hot_w = 0.75 * hot_w + 0.25 * 1.0
    warm_w = 0.75 * warm_w + 0.25 * 1.0
    cold_w = 0.75 * cold_w + 0.25 * 1.0

    hot_w = clamp(hot_w, 0.6, 1.8)
    warm_w = clamp(warm_w, 0.6, 1.8)
    cold_w = clamp(cold_w, 0.6, 1.8)

    return hot_w, warm_w, cold_w

def stepE_log_scores(target_date, seed_dates_cat, category_data, decade_weight_log,
                     cross_hop_log, Hot_set, Warm_set, Cold_set,
                     hot_w, warm_w, cold_w, window_size_cat):
    seed_numbers_main = category_data["seed_numbers_main"]
    seed_numbers_supp = category_data["seed_numbers_supp"]
    f_main = category_data["f_main"]
    f_supp = category_data["f_supp"]
    L = category_data["L"]
    last_main_date = category_data["last_main_date"]

    # Adjacency
    adj_count = {n: 0 for n in NUMBER_RANGE}
    main_set = seed_numbers_main
    main_set = [n for n in main_set if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
    main_counter = collections.Counter(main_set)
    for n in NUMBER_RANGE:
        adj_count[n] = main_counter.get(n - 1, 0) + main_counter.get(n + 1, 0)
    max_adj = max(adj_count.values()) if adj_count else 1
    if max_adj <= 0:
        max_adj = 1

    # Last-3-days delta
    if seed_dates_cat:
        earliest = min(seed_dates_cat)
        recent_start = max(target_date - timedelta(days=3), earliest)
        recent_dates = [recent_start + timedelta(days=i)
                        for i in range((target_date - recent_start).days)]
    else:
        recent_dates = []
    hits_main_last3 = {n: 0 for n in NUMBER_RANGE}
    hits_supp_last3 = {n: 0 for n in NUMBER_RANGE}
    for dt in recent_dates:
        for d in DRAWS_BY_DATE.get(dt, []):
            for n in d["main"]:
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    hits_main_last3[n] += 1
            for n in d["supp"]:
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    hits_supp_last3[n] += 1
            for n in d["powerball"]:
                if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX:
                    hits_supp_last3[n] += 1

    # Centre vs neighbour recency
    centre_raw = {n: 0.0 for n in NUMBER_RANGE}
    for n in NUMBER_RANGE:
        hm = hits_main_last3[n]
        neigh = 0
        if n - 1 >= MAIN_NUMBER_MIN:
            neigh += hits_main_last3[n - 1]
        if n + 1 <= MAIN_NUMBER_MAX:
            neigh += hits_main_last3[n + 1]
        centre_raw[n] = hm - 0.7 * neigh

    if centre_raw:
        centre_min = min(centre_raw.values())
        centre_max = max(centre_raw.values())
    else:
        centre_min = centre_max = 0.0
    centre_score = {}
    if centre_max == centre_min:
        for n in NUMBER_RANGE:
            centre_score[n] = 0.5
    else:
        for n in NUMBER_RANGE:
            centre_score[n] = (centre_raw[n] - centre_min) / (centre_max - centre_min)
    LAMBDA_CENTRE = 0.35

    log_score_raw = {}
    log_score = {}
    rawP = {}
    P = {}

    cold_resurge_score = {}
    cold_resurge_log = {}

    cross_log = {n: math.log(1 + 0.08 * L[n]) for n in NUMBER_RANGE}

    category_weight_log = {}
    for n in NUMBER_RANGE:
        if n in Hot_set:
            category_weight_log[n] = math.log(hot_w)
        elif n in Warm_set:
            category_weight_log[n] = math.log(warm_w)
        else:
            category_weight_log[n] = math.log(cold_w)

    for n in NUMBER_RANGE:
        # adjacency
        adj_score_raw = 0.05 + 0.25 * (adj_count[n] / max_adj)
        adj_log = math.log(1 + adj_score_raw)

        # delta
        if hits_main_last3[n] > 0:
            mult = 1.4
        elif hits_supp_last3[n] > 0:
            mult = 1.2
        else:
            mult = 1.0
        delta_log = math.log(mult)

        # centre
        cm = 1.0 + LAMBDA_CENTRE * (centre_score[n] - 0.5)
        cm = clamp(cm, 0.75, 1.25)
        centre_log = math.log(cm)

        # supp-only bonus
        if f_main[n] == 0 and f_supp[n] > 0:
            supp_log = math.log(1.05)
        else:
            supp_log = 0.0

        # decade log
        dec_log = decade_weight_log.get(n, 0.0)

        # cross-hop log already given
        ch_log = cross_hop_log.get(n, 0.0)

        # cold resurgence
        lmd = last_main_date[n]
        if lmd is None:
            resurge_raw = 0.0
        else:
            gap_days = (target_date - lmd).days
            if 4 <= gap_days <= window_size_cat:
                resurge_raw = 1.0 / max(1, gap_days)
            else:
                resurge_raw = 0.0
        cold_resurge_score[n] = resurge_raw
        cold_resurge_log[n] = 0.0

    # normalise cold resurgence
    max_resurge = max(cold_resurge_score.values()) if cold_resurge_score else 0.0
    if max_resurge > 0:
        for n in NUMBER_RANGE:
            s = cold_resurge_score[n] / max_resurge
            cold_resurge_score[n] = s
            if n in Hot_set:
                cold_resurge_log[n] = 0.0
            else:
                cold_resurge_log[n] = math.log(1.0 + theta_resurge * s)
    else:
        for n in NUMBER_RANGE:
            cold_resurge_score[n] = 0.0
            cold_resurge_log[n] = 0.0

    # now final total log-scores
    for n in NUMBER_RANGE:
        adj_score_raw = 0.05 + 0.25 * (adj_count[n] / max_adj)
        adj_log = math.log(1 + adj_score_raw)

        if hits_main_last3[n] > 0:
            mult = 1.4
        elif hits_supp_last3[n] > 0:
            mult = 1.2
        else:
            mult = 1.0
        delta_log = math.log(mult)

        cm = 1.0 + LAMBDA_CENTRE * (centre_score[n] - 0.5)
        cm = clamp(cm, 0.75, 1.25)
        centre_log = math.log(cm)

        if f_main[n] == 0 and f_supp[n] > 0:
            supp_log = math.log(1.05)
        else:
            supp_log = 0.0

        dec_log = decade_weight_log.get(n, 0.0)
        ch_log = cross_hop_log.get(n, 0.0)
        cr_log = cold_resurge_log[n]
        cat_log = category_weight_log[n]

        total = adj_log + delta_log + centre_log + cross_log[n] + supp_log + cat_log + dec_log + ch_log + cr_log
        total = min(total, LOG_SCORE_MAX)
        log_score_raw[n] = total
        log_score[n] = total
        rawP[n] = math.exp(total)

    Z = sum(rawP.values())
    if Z <= 0:
        for n in NUMBER_RANGE:
            P[n] = 1.0 / len(NUMBER_RANGE)
    else:
        for n in NUMBER_RANGE:
            P[n] = rawP[n] / Z

    return {
        "log_score_raw": log_score_raw,
        "log_score": log_score,
        "P": P,
        "centre_score": centre_score,
        "cold_resurge_score": cold_resurge_score,
        "cold_resurge_log": cold_resurge_log,
    }

def stepF_clusters(seed_draws_cat, learning_state):
    # cluster detection on main numbers in category window
    cluster_counts = collections.Counter()
    for d in seed_draws_cat:
        nums = sorted(set(n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX))
        for r in (2, 3, 4):
            if len(nums) >= r:
                for comb in itertools.combinations(nums, r):
                    cluster_counts[comb] += 1
    clusters = {}
    for C, freq in cluster_counts.items():
        if freq >= 2:
            base = 1 + 0.2 * (freq - 1)
            prior = learning_state.cluster_priority_score_global.get(C, 0.0)
            pr = base * (1 + prior)
            clusters[C] = {"freq": freq, "priority": pr}
    return clusters

def stepG_composition_targets(current_lottery_name, avg_hot, avg_warm, avg_cold,
                              Hot_set, Warm_set, Cold_set):
    draw_size = main_draw_size(current_lottery_name)
    h_target = round(draw_size * avg_hot)
    w_target = round(draw_size * avg_warm)
    c_target = draw_size - h_target - w_target

    h_target = max(0, min(h_target, len(Hot_set)))
    w_target = max(0, min(w_target, len(Warm_set)))
    c_target = max(0, min(c_target, len(Cold_set)))

    total = h_target + w_target + c_target
    sets = [("Warm", len(Warm_set)), ("Hot", len(Hot_set)), ("Cold", len(Cold_set))]
    while total < draw_size:
        best_cat = None
        best_cap = -1
        for name, size in sets:
            if name == "Hot":
                cur = h_target
            elif name == "Warm":
                cur = w_target
            else:
                cur = c_target
            cap = size - cur
            if cap > best_cap:
                best_cap = cap
                best_cat = name
        if best_cat is None or best_cap <= 0:
            break
        if best_cat == "Warm":
            w_target += 1
        elif best_cat == "Hot":
            h_target += 1
        else:
            c_target += 1
        total = h_target + w_target + c_target

    return h_target, w_target, c_target, draw_size

def monte_carlo_sampling(current_lottery_name, target_date, mode, M,
                         Hot_set, Warm_set, Cold_set,
                         avg_hot, avg_warm, avg_cold,
                         P, clusters, centre_score):
    import numpy as np

    draw_size = main_draw_size(current_lottery_name)

    complexity = math.sqrt(max(1, M / 20))
    if mode == "learning":
        TRIALS = int(clamp(40000 * complexity, 15000, 60000))
    else:
        TRIALS = int(clamp(150000 * complexity, 60000, 200000))

    max_avg = max(avg_hot, avg_warm, avg_cold)
    EXPLORE_FRAC = clamp(0.10 + 0.20 * max(0, max_avg - 1/3), 0.10, 0.20)

    COLD_EXPLORE_MULT = 1.25
    CLUSTER_LAMBDA = 0.15

    baseP = P.copy()

    hot_list = sorted(Hot_set)
    warm_list = sorted(Warm_set)
    cold_list = sorted(Cold_set)

    # composition
    h_target, w_target, c_target, _ = stepG_composition_targets(
        current_lottery_name, avg_hot, avg_warm, avg_cold,
        Hot_set, Warm_set, Cold_set
    )

    if h_target > len(hot_list) or w_target > len(warm_list) or c_target > len(cold_list):
        raise RuntimeError("Composition targets exceed available set sizes")

    # Precompute cluster mapping: number -> sum of (priority - 1)
    cluster_Sn = {n: 0.0 for n in NUMBER_RANGE}
    for C, info in clusters.items():
        pri = info["priority"]
        for n in C:
            if n in cluster_Sn:
                cluster_Sn[n] += (pri - 1.0)

    freq = collections.Counter()

    hot_count_total = 0.0
    warm_count_total = 0.0
    cold_count_total = 0.0
    dec_count_total = {d: 0.0 for d in DECADES}

    for _ in range(TRIALS):
        u = random.random()
        mode_trial = "explore" if u < EXPLORE_FRAC else "normal"

        W = {}
        for n in NUMBER_RANGE:
            w = baseP[n]
            if mode_trial == "explore" and n in Cold_set:
                w *= COLD_EXPLORE_MULT
            # cluster boost
            S_n = cluster_Sn.get(n, 0.0)
            cluster_boost = math.exp(CLUSTER_LAMBDA * S_n) if S_n != 0 else 1.0
            w *= cluster_boost
            W[n] = w

        # Build per-category probability
        def build_probs(num_list):
            if not num_list:
                return []
            weights = [W[n] for n in num_list]
            Z = sum(weights)
            if Z <= 0:
                return [1.0 / len(num_list)] * len(num_list)
            return [w / Z for w in weights]

        chosen = set()

        # Hot
        if h_target > 0 and hot_list:
            probs = build_probs(hot_list)
            if h_target == len(hot_list):
                hot_pick = list(hot_list)
            else:
                hot_pick = list(np.random.choice(hot_list, size=h_target, replace=False, p=probs))
            chosen.update(hot_pick)
        else:
            hot_pick = []

        # Warm
        if w_target > 0 and warm_list:
            available = [n for n in warm_list if n not in chosen]
            probs = build_probs(available)
            if w_target == len(available):
                warm_pick = list(available)
            else:
                warm_pick = list(np.random.choice(available, size=w_target, replace=False, p=probs))
            chosen.update(warm_pick)
        else:
            warm_pick = []

        # Cold
        if c_target > 0 and cold_list:
            available = [n for n in cold_list if n not in chosen]
            probs = build_probs(available)
            if c_target == len(available):
                cold_pick = list(available)
            else:
                cold_pick = list(np.random.choice(available, size=c_target, replace=False, p=probs))
            chosen.update(cold_pick)
        else:
            cold_pick = []

        if len(chosen) != (h_target + w_target + c_target):
            raise RuntimeError("Sampling error: chosen size mismatch")

        T = tuple(sorted(chosen))
        freq[T] += 1

        # update diagnostics counts
        hot_count_total += sum(1 for n in chosen if n in Hot_set)
        warm_count_total += sum(1 for n in chosen if n in Warm_set)
        cold_count_total += sum(1 for n in chosen if n in Cold_set)
        for n in chosen:
            d_id = decade(n)
            if d_id is not None:
                dec_count_total[d_id] += 1

    hot_pred = hot_count_total / TRIALS
    warm_pred = warm_count_total / TRIALS
    cold_pred = cold_count_total / TRIALS
    dec_pred = {d: dec_count_total[d] / TRIALS for d in DECADES}

    prob_T = {T: c / TRIALS for T, c in freq.items()}

    if mode == "learning":
        # top-10 by prob(T)
        top_tuples = sorted(prob_T.items(), key=lambda kv: kv[1], reverse=True)[:20]
        tuple_scores = {T: prob for T, prob in prob_T.items()}
    else:
        # prediction mode: centre-biased ranking
        alpha_centre = 0.5
        tuple_scores = {}
        for T, p in prob_T.items():
            avg_centre = sum(centre_score.get(n, 0.5) for n in T) / len(T)
            factor = 1 + alpha_centre * (avg_centre - 0.5)
            tuple_scores[T] = p * factor
        top_tuples = sorted(tuple_scores.items(), key=lambda kv: kv[1], reverse=True)[:20]

    return {
        "TRIALS": TRIALS,
        "EXPLORE_FRAC": EXPLORE_FRAC,
        "freq": freq,
        "prob_T": prob_T,
        "tuple_scores": tuple_scores,
        "top_tuples": top_tuples,
        "hot_pred": hot_pred,
        "warm_pred": warm_pred,
        "cold_pred": cold_pred,
        "dec_pred": dec_pred,
        "h_target": h_target,
        "w_target": w_target,
        "c_target": c_target,
    }

def process_draw(lottery_name, target_date, window_size_cat, window_size_dec,
                 learning_state, mode):
    # mode: "learning" for calibration + main, "prediction" for final
    windows = build_category_and_decade_windows(target_date, window_size_cat, window_size_dec)
    if windows is None:
        return None  # aborted
    seed_dates_cat, seed_draws_cat, seed_dates_dec, seed_draws_dec = windows

    # Step B: category and decade frequencies
    category_data = stepB_category_frequencies(seed_draws_cat, seed_dates_cat)
    decade_data = stepB_decade_window(seed_draws_dec)
    dec_w_final = stepB_apply_global_decade_learning(decade_data["dec_w_window"], learning_state)
    dec_w_recent, dec_recent_count, rec_w = stepB_last3_momentum(
        target_date,
        seed_dates_dec,
        decade_data["dec_main_count"],
        decade_data["total_main_in_range_dec"],
        dec_w_final,
    )
    decade_weight_log = stepB_deacde_weight_log(dec_w_recent)
    cross_hop_score, cross_hop_log = stepB_cross_hop(seed_draws_cat, window_size_cat)

    # Step C: H/W/C
    Hot_set, Warm_set, Cold_set, avg_hot, avg_warm, avg_cold, bias, K, M = stepC_hot_warm_cold(
        category_data["f_main"], category_data["L"], seed_draws_cat
    )

    # Step D: category weights
    hot_w, warm_w, cold_w = stepD_category_weights(avg_hot, avg_warm, avg_cold, bias, learning_state)

    # Step E: per-number log scores
    log_score_data = stepE_log_scores(
        target_date, seed_dates_cat, category_data, decade_weight_log,
        cross_hop_log, Hot_set, Warm_set, Cold_set,
        hot_w, warm_w, cold_w, window_size_cat
    )

    # Step F: clusters
    clusters = stepF_clusters(seed_draws_cat, learning_state)

    # Step G + H: composition targets + Monte Carlo
    mc_results = monte_carlo_sampling(
        lottery_name, target_date,
        "learning" if mode == "learning" else "prediction",
        category_data["M"],
        Hot_set, Warm_set, Cold_set,
        avg_hot, avg_warm, avg_cold,
        log_score_data["P"],
        clusters,
        log_score_data["centre_score"],
    )

    # Build diagnostics structure (Step I)
    diagnostics = {
        "lottery": lottery_name,
        "date": target_date,
        "seed_dates_cat": seed_dates_cat,
        "seed_dates_dec": seed_dates_dec,
        "M": category_data["M"],
        "K": K,
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
        "top_numbers_by_P": sorted(log_score_data["P"].items(), key=lambda kv: kv[1], reverse=True)[:15],
        "clusters": clusters,
        "TRIALS": mc_results["TRIALS"],
        "EXPLORE_FRAC": mc_results["EXPLORE_FRAC"],
        "h_target": mc_results["h_target"],
        "w_target": mc_results["w_target"],
        "c_target": mc_results["c_target"],
        "top_tuples": mc_results["top_tuples"],
        "dec_main_count": decade_data["dec_main_count"],
        "dec_w_window": decade_data["dec_w_window"],
        "dec_w_final": dec_w_final,
        "dec_w_recent": dec_w_recent,
        "dec_pred": mc_results["dec_pred"],
        "dec_recent_count": dec_recent_count,
        "cross_hop_score": cross_hop_score,
        "cross_hop_log": cross_hop_log,
        "cold_resurge_score": log_score_data["cold_resurge_score"],
        "cold_resurge_log": log_score_data["cold_resurge_log"],
        "centre_score": log_score_data["centre_score"],
        "log_score": log_score_data["log_score"],
        "prob_T": mc_results["prob_T"],
        "tuple_scores": mc_results["tuple_scores"],
        "hot_pred": mc_results["hot_pred"],
        "warm_pred": mc_results["warm_pred"],
        "cold_pred": mc_results["cold_pred"],
    }

    return {
        "category_data": category_data,
        "decade_data": decade_data,
        "dec_w_final": dec_w_final,
        "dec_w_recent": dec_w_recent,
        "log_score_data": log_score_data,
        "clusters": clusters,
        "mc_results": mc_results,
        "diagnostics": diagnostics,
    }

# =========================
# Step 12: Learning feedback
# =========================

def get_actual_main_draw(lottery_name, target_date):
    for d in GLOBAL_DRAWS:
        if d["lottery"] == lottery_name and d["date"] == target_date:
            return [n for n in d["main"] if MAIN_NUMBER_MIN <= n <= MAIN_NUMBER_MAX]
    return None

def learning_feedback(lottery_name, target_date, diagnostics, category_data, learning_state):
    actual_main = get_actual_main_draw(lottery_name, target_date)
    if not actual_main:
        return  # no learning if we don't have the actual draw
    actual_set = set(actual_main)
    Hot_set = diagnostics["Hot_set"]
    Warm_set = diagnostics["Warm_set"]
    Cold_set = diagnostics["Cold_set"]

    hot_actual = sum(1 for n in actual_main if n in Hot_set)
    warm_actual = sum(1 for n in actual_main if n in Warm_set)
    cold_actual = sum(1 for n in actual_main if n in Cold_set)

    hot_pred = diagnostics["hot_pred"]
    cold_pred = diagnostics["cold_pred"]

    hot_error = hot_actual - hot_pred
    cold_error = cold_actual - cold_pred

    f_main = category_data["f_main"]
    f_values = [f_main[n] for n in NUMBER_RANGE]
    var_f_main = variance(f_values)
    learning_rate = clamp(0.02 + 0.02 * var_f_main, 0.02, 0.10)

    delta_hot_step = clamp(sign(hot_error) * learning_rate * abs(hot_error), -0.1, 0.1)
    delta_cold_step = clamp(sign(cold_error) * learning_rate * abs(cold_error), -0.1, 0.1)

    learning_state.delta_hot += delta_hot_step
    learning_state.delta_cold += delta_cold_step
    learning_state.delta_warm = - (learning_state.delta_hot + learning_state.delta_cold) / 2

    learning_state.delta_hot = clamp(learning_state.delta_hot, -0.5, 0.5)
    learning_state.delta_warm = clamp(learning_state.delta_warm, -0.5, 0.5)
    learning_state.delta_cold = clamp(learning_state.delta_cold, -0.5, 0.5)

    # Cluster learning
    for C in diagnostics["clusters"].keys():
        if set(C).issubset(actual_set):
            learning_state.cluster_priority_score_global[C] = learning_state.cluster_priority_score_global.get(C, 0.0) + 0.05
        else:
            learning_state.cluster_priority_score_global[C] = learning_state.cluster_priority_score_global.get(C, 0.0) - 0.02
        learning_state.cluster_priority_score_global[C] = clamp(learning_state.cluster_priority_score_global[C], -0.5, 0.5)

    # Decade learning
    dec_actual = {d: 0 for d in DECADES}
    for n in actual_main:
        d_id = decade(n)
        if d_id is not None:
            dec_actual[d_id] += 1

    dec_pred = diagnostics["dec_pred"]
    for d in DECADES:
        dec_error = dec_actual[d] - dec_pred[d]
        step_d = clamp(base_learning_rate_decade * dec_error, -0.15, 0.15)
        learning_state.delta_decade[d] += step_d

    # Oscillation regulariser
    # LOW/HIGH groups dynamically from sorted decades
    sorted_decades = sorted(DECADES)
    k = len(sorted_decades) // 2
    LOW = set(sorted_decades[:k])
    HIGH = set(sorted_decades[k:])

    dom_decade_t = max(DECADES, key=lambda d: (dec_actual[d], -d))
    if learning_state.last_dom_decade is not None and dom_decade_t == learning_state.last_dom_decade:
        if dom_decade_t in LOW:
            for dh in HIGH:
                learning_state.delta_decade[dh] += 0.03
            for dl in LOW:
                learning_state.delta_decade[dl] -= 0.03
        else:
            for dl in LOW:
                learning_state.delta_decade[dl] += 0.03
            for dh in HIGH:
                learning_state.delta_decade[dh] -= 0.03

    learning_state.last_dom_decade = dom_decade_t

    for d in DECADES:
        learning_state.delta_decade[d] = clamp(learning_state.delta_decade[d], -0.8, 0.8)

    mean_delta_dec = sum(learning_state.delta_decade[d] for d in DECADES) / N_DECADES
    for d in DECADES:
        learning_state.delta_decade[d] -= mean_delta_dec

# =========================
# Calibration
# =========================

def calibration_window_cat():
    best_W = None
    best_mse = None
    for W_cat in WINDOW_SIZE_CANDIDATES:
        ls = LearningState()
        sq_errors = []
        for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
            result = process_draw(lottery_name, target_date, W_cat, W_cat, ls, mode="learning")
            if result is None:
                continue
            diag = result["diagnostics"]
            hot_pred = diag["hot_pred"]
            cold_pred = diag["cold_pred"]
            actual_main = get_actual_main_draw(lottery_name, target_date)
            if not actual_main:
                continue
            Hot_set = diag["Hot_set"]
            Cold_set = diag["Cold_set"]
            hot_actual = sum(1 for n in actual_main if n in Hot_set)
            cold_actual = sum(1 for n in actual_main if n in Cold_set)
            sq_errors.append((hot_actual - hot_pred) ** 2 + (cold_actual - cold_pred) ** 2)
            learning_feedback(lottery_name, target_date, diag, result["category_data"], ls)
        if not sq_errors:
            continue
        mse = sum(sq_errors) / len(sq_errors)
        if best_mse is None or mse < best_mse or (mse == best_mse and (best_W is None or W_cat < best_W)):
            best_mse = mse
            best_W = W_cat
    return best_W, best_mse

def calibration_window_dec(W_cat_star):
    best_W = None
    best_score = None
    for W_dec in WINDOW_SIZE_CANDIDATES:
        ls = LearningState()
        dec_sq_errors = []
        z_values = []
        for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
            result = process_draw(lottery_name, target_date, W_cat_star, W_dec, ls, mode="learning")
            if result is None:
                continue
            diag = result["diagnostics"]
            dec_pred = diag["dec_pred"]
            actual_main = get_actual_main_draw(lottery_name, target_date)
            if not actual_main:
                continue
            dec_actual = {d: 0 for d in DECADES}
            for n in actual_main:
                d_id = decade(n)
                if d_id is not None:
                    dec_actual[d_id] += 1
            for d in DECADES:
                dec_sq_errors.append((dec_actual[d] - dec_pred[d]) ** 2)

            # LOW/HIGH groups
            sorted_decades = sorted(DECADES)
            k = len(sorted_decades) // 2
            LOW = set(sorted_decades[:k])
            HIGH = set(sorted_decades[k:])
            low_t = sum(dec_pred[d] for d in LOW)
            high_t = sum(dec_pred[d] for d in HIGH)
            z_values.append(low_t - high_t)

            learning_feedback(lottery_name, target_date, diag, result["category_data"], ls)

        if not dec_sq_errors:
            continue
        decade_mse = sum(dec_sq_errors) / len(dec_sq_errors)
        stab_pen = variance(z_values) if z_values else 0.0
        alpha = 1.0
        beta = 0.5
        score = alpha * decade_mse + beta * stab_pen

        if best_score is None or score < best_score or (score == best_score and (best_W is None or W_dec < best_W)):
            best_score = score
            best_W = W_dec
    return best_W, best_score

# =========================
# Pretty-print helpers
# =========================

def print_draw_diagnostics(diag, mode, learning_state):
    print("=" * 80)
    print(f"{mode.upper()} MODE DRAW: {diag['lottery']} on {diag['date']}")
    print("- Window CAT dates:", diag["seed_dates_cat"][0], "to", diag["seed_dates_cat"][-1])
    print("- Window DEC dates:", diag["seed_dates_dec"][0], "to", diag["seed_dates_dec"][-1])
    print(f"- M={diag['M']}, K={diag['K']}")
    print(f"- Hot/Warm/Cold sizes: {len(diag['Hot_set'])}/{len(diag['Warm_set'])}/{len(diag['Cold_set'])}")
    print(f"- avg_hot={diag['avg_hot']:.3f}, avg_warm={diag['avg_warm']:.3f}, avg_cold={diag['avg_cold']:.3f}, bias={diag['bias']}")
    print(f"- category weights: hot={diag['hot_w']:.3f}, warm={diag['warm_w']:.3f}, cold={diag['cold_w']:.3f}")
    print(f"- composition targets (h,w,c)={diag['h_target']},{diag['w_target']},{diag['c_target']}")
    print(f"- TRIALS={diag['TRIALS']}, EXPLORE_FRAC={diag['EXPLORE_FRAC']:.3f}")
    print("- Top 20 numbers by P(n):")
    for n, p in diag["top_numbers_by_P"][:20]:
        print(f"   n={n:2d}  P={p:.5f}")
    print("- Top 5 clusters:")
    clusters_sorted = sorted(diag["clusters"].items(),
                             key=lambda kv: kv[1]["priority"],
                             reverse=True)[:5]
    for C, info in clusters_sorted:
        print(f"   C={C}, freq={info['freq']}, priority={info['priority']:.3f}")
    print("- Decade diagnostics:")
    for d in DECADES:
        print(f"   Decade {d}: count={diag['dec_main_count'][d]}, "
              f"w_window={diag['dec_w_window'][d]:.3f}, "
              f"w_final={diag['dec_w_final'][d]:.3f}, "
              f"w_recent={diag['dec_w_recent'][d]:.3f}, "
              f"pred={diag['dec_pred'][d]:.3f}")
    print("- Hop diagnostics (top 5 by cross_hop_score):")
    hop_sorted = sorted(diag["cross_hop_score"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    for n, s in hop_sorted:
        print(f"   n={n:2d} score={s:.3f} log={diag['cross_hop_log'][n]:.3f}")
    print("- Cold resurgence diagnostics (top 5):")
    cold_sorted = sorted(diag["cold_resurge_score"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    for n, s in cold_sorted:
        print(f"   n={n:2d} score={s:.3f} log={diag['cold_resurge_log'][n]:.3f}")
    print("- Top 20 tuples:")
    for T, score in diag["top_tuples"]:
        prob = diag["prob_T"].get(T, 0.0)
        print(f"   T={T} prob={prob:.6f} score={score:.6f}")
    print("- Category prediction vs actual (if available):")
    actual_main = get_actual_main_draw(diag["lottery"], diag["date"])
    if actual_main:
        hot_act = sum(1 for n in actual_main if n in diag["Hot_set"])
        warm_act = sum(1 for n in actual_main if n in diag["Warm_set"])
        cold_act = sum(1 for n in actual_main if n in diag["Cold_set"])
        print(f"   hot_pred={diag['hot_pred']:.3f}, warm_pred={diag['warm_pred']:.3f}, cold_pred={diag['cold_pred']:.3f}")
        print(f"   hot_actual={hot_act}, warm_actual={warm_act}, cold_actual={cold_act}")
    else:
        print("   (no actual draw in history for this date; prediction only)")
    print("- Learning state snapshot (Δ_hot, Δ_warm, Δ_cold):",
          f"{learning_state.delta_hot:.3f}, {learning_state.delta_warm:.3f}, {learning_state.delta_cold:.3f}")
    print("- Decade learning offsets Δ_decade:")
    for d in DECADES:
        print(f"   d={d}: Δ={learning_state.delta_decade[d]:.3f}")
    print("=" * 80)
    print()

# =========================
# Main driver
# =========================

def main():
    SEED_MAIN = 0
    random.seed(SEED_MAIN)
    np.random.seed(SEED_MAIN)

    print("=== Calibration: WINDOW_SIZE_CAT* ===")
    W_cat_star, cat_mse = calibration_window_cat()
    print(f"Chosen WINDOW_SIZE_CAT* = {W_cat_star} (category_MSE = {cat_mse})")
    print()

    print("=== Calibration: WINDOW_SIZE_DEC* ===")
    W_dec_star, dec_score = calibration_window_dec(W_cat_star)
    print(f"Chosen WINDOW_SIZE_DEC* = {W_dec_star} (score_dec = {dec_score})")
    print()

    # Main run with chosen windows
    print("=== Main run with chosen windows ===")
    ls = LearningState()
    for lottery_name, target_date in TARGET_DRAWS_FOR_LEARNING:
        result = process_draw(lottery_name, target_date, W_cat_star, W_dec_star, ls, mode="learning")
        if result is None:
            print(f"[SKIP] No valid windows for {lottery_name} on {target_date}")
            continue
        diag = result["diagnostics"]
        print_draw_diagnostics(diag, mode="learning", learning_state=ls)
        learning_feedback(lottery_name, target_date, diag, result["category_data"], ls)

    # Final prediction
    print("=== Final prediction ===")
    prediction_lottery, prediction_date, prediction_draw_size = PREDICTION_TARGET
    if prediction_draw_size != main_draw_size(prediction_lottery):
        print("Configuration error: prediction_draw_size != main_draw_size")
        return

    pred_result = process_draw(prediction_lottery, prediction_date, W_cat_star, W_dec_star, ls, mode="prediction")
    if pred_result is None:
        print("Prediction aborted: no valid window for prediction date")
        return
    pred_diag = pred_result["diagnostics"]
    print_draw_diagnostics(pred_diag, mode="prediction", learning_state=ls)
    print("Top-20 predicted tuples for prediction:")
    for T, score in pred_diag["top_tuples"]:
        prob = pred_diag["prob_T"].get(T, 0.0)
        print(f"  {T}  prob={prob:.6f}  score={score:.6f}")

if __name__ == "__main__":
    main()
