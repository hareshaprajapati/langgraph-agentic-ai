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
    f"siko_sat_logs.log"   # single growing log file
)

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# print("\n" + "="*80)
# print(f"RUN START @ {datetime.now()}")
# print("="*80)

import Siko_Core as core
import time
import datetime
import random
from collections import Counter

core.global_draws.clear()

core.DECADE_BANDS = [
    (1,  1,  10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 40),
    (5, 41, 45),
]

for dec_id, start, end in core.DECADE_BANDS:
    print(f"Decade {dec_id}: {start}–{end}")

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)


# --- Number range for Saturday Lotto mains (1..45) ---
core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 45
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))
core.WINDOW_LENGTH = (6, 7, 8, 9)

# SFL → Saturday hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Saturday Lotto"

core.addDraws()
core.finalize_data()

# can change below code
# *****************
# today = datetime.date.today() # - datetime.timedelta(days=1)
today = datetime.date(2026, 2, 7)  # keep explicit & reproducible
# today = datetime.date(2025, 12, 27)  # keep explicit & reproducible
real_draw_date = today
real_draw_result = [3, 8, 9, 27, 33, 41]
N = 21

def last_n_saturdays(today, n):
    dates = []
    d = today
    while len(dates) < n:
        if d.weekday() == 5:  # Saturday
            dates.append(d)
        d -= datetime.timedelta(days=1)
    return dates

LEADER_POOL_RANK_MAX = 5
LEARNING_DRAW_COUNT = 14
MAX_TICKETS_TO_PRINT = 20
COHORT_USAGE_CAP_FRAC = 0.40      # e.g. 0.40 to cap cohort repeats
COHORT_AUTOPRED_EVAL_LAST_N = None  # e.g. 2 or 3 for auto predictor window
core.COHORT_USAGE_CAP_FRAC = COHORT_USAGE_CAP_FRAC
core.COHORT_AUTOPRED_EVAL_LAST_N = COHORT_AUTOPRED_EVAL_LAST_N
core.COHORT_ALLOWED_HWC_TOP_K = 3
core.COHORT_ALLOWED_DEC_TOP_K = 3
core.LEADER_USAGE_CAP = None
core.TICKET_DIVERSITY_LAMBDA = None
core.TOP_P_COMBO_ENABLED = True
core.TOP_P_COMBO_N = 14
core.TOP_P_COMBO_MAX = 80
core.COVERAGE_MODE = True
core.COVERAGE_ALPHA = 0.8
OVERRIDE_COHORT_HWC = None         # e.g. (0, 2, 4)
# OVERRIDE_COHORT_HWC = (0, 4, 1)          # e.g. (0, 2, 4)
OVERRIDE_COHORT_DECADES = None      # e.g. {1:1, 2:2, 3:0, 4:2, 5:1}
# OVERRIDE_COHORT_DECADES = {1:2, 2:1, 3:1, 4:1, 5:0}      # e.g. {1:1, 2:2, 3:0, 4:2, 5:1}
# OVERRIDE_RANK_MIN = 5            # e.g. 12
# OVERRIDE_RANK_MAX = 39            # e.g. 40
OVERRIDE_RANK_MIN = None           # e.g. 12
OVERRIDE_RANK_MAX = None            # e.g. 40
OVERRIDE_P_MIN = None               # e.g. 0.010
OVERRIDE_P_MAX = None               # e.g. 0.030

USE_NEW_ALGO = True

NEW_ALGO_RECENT_DRAWS = 14
NEW_ALGO_POOL_SIZE = 24
NEW_ALGO_WARM_SIZE = 14
NEW_ALGO_HOT_SIZE = 16
NEW_ALGO_RECENCY_DECAY = 0.15
NEW_ALGO_COMPOSITIONS = [
    (3, 2, 1),  # hot, warm, cold
    (3, 1, 2),
    (2, 3, 1),
    (2, 2, 2),
]

def _recent_saturday_draws(target_date, count):
    draws = []
    for d in sorted(core.draws_by_date.keys()):
        if d >= target_date:
            break
        for dr in core.draws_by_date[d]:
            if dr.lottery == "Saturday Lotto":
                draws.append(dr)
    return draws[-count:]

def _build_pair_counts(draws):
    pair_counts = Counter()
    for dr in draws:
        nums = sorted(set(dr.main))
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pair_counts[(nums[i], nums[j])] += 1
    return pair_counts

def _score_numbers(draws):
    freq = Counter()
    recency = Counter()
    # Most recent draw gets highest recency weight
    for idx, dr in enumerate(reversed(draws), 1):
        weight = 1.0 + (NEW_ALGO_RECENCY_DECAY * idx)
        for n in dr.main:
            freq[n] += 1
            recency[n] += weight
    scores = {}
    for n in range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1):
        scores[n] = freq[n] + recency[n]
    return scores

def _weighted_pick(candidates, weights, rng):
    total = sum(weights.get(n, 0.0) for n in candidates)
    if total <= 0:
        return rng.choice(list(candidates))
    r = rng.random() * total
    acc = 0.0
    for n in candidates:
        acc += weights.get(n, 0.0)
        if acc >= r:
            return n
    return list(candidates)[-1]

def _build_tickets_new_algo(target_date, max_tickets):
    draws = _recent_saturday_draws(target_date, NEW_ALGO_RECENT_DRAWS)
    if not draws:
        return []

    scores = _score_numbers(draws)
    pair_counts = _build_pair_counts(draws)

    ranked = [n for n, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    hot = set(ranked[:NEW_ALGO_HOT_SIZE])
    warm = set(ranked[NEW_ALGO_HOT_SIZE:NEW_ALGO_HOT_SIZE + NEW_ALGO_WARM_SIZE])
    cold = set(ranked[NEW_ALGO_HOT_SIZE + NEW_ALGO_WARM_SIZE:NEW_ALGO_POOL_SIZE])
    if not cold:
        cold = set(ranked[NEW_ALGO_HOT_SIZE + NEW_ALGO_WARM_SIZE:])

    rng = random.Random(0)
    used_counts = Counter()
    tickets = []

    for i in range(max_tickets):
        h, w, c = NEW_ALGO_COMPOSITIONS[i % len(NEW_ALGO_COMPOSITIONS)]
        ticket = []
        pools = [
            (hot, h),
            (warm, w),
            (cold, c),
        ]
        for pool, count in pools:
            pool = [n for n in pool if n not in ticket]
            for _ in range(count):
                if not pool:
                    break
                weights = {}
                for n in pool:
                    penalty = 1.0 / (1.0 + used_counts[n])
                    synergy = 0
                    for t in ticket:
                        a, b = sorted((n, t))
                        synergy += pair_counts.get((a, b), 0)
                    weights[n] = scores.get(n, 0.0) * penalty + 0.2 * synergy
                pick = _weighted_pick(pool, weights, rng)
                ticket.append(pick)
                pool.remove(pick)

        # Fill if short
        remaining = [n for n in ranked if n not in ticket]
        while len(ticket) < 6 and remaining:
            pick = _weighted_pick(remaining, scores, rng)
            ticket.append(pick)
            remaining.remove(pick)

        ticket = sorted(ticket)
        tickets.append(ticket)
        for n in ticket:
            used_counts[n] += 1

    return tickets


if __name__ == "__main__":

    weeks_lt3 = 0
    weeks_3p = 0
    weeks_4p = 0
    weeks_5p = 0
    weeks_6p = 0
    max_hit_observed = 0
    total_3_hits = 0
    total_4_hits = 0
    total_5_hits = 0
    total_6_hits = 0

    saturday_dates = last_n_saturdays(today, N)

    start_ts = time.time()
    start_dt = datetime.datetime.now()

    print(f"\n=== RUN START ===")
    print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    saturday_date_set = set(saturday_dates)
    for target_date in reversed(saturday_dates):
        print("\n" + "="*80)
        print(f"PREDICTION RUN FOR Saturday Lotto on {target_date}")
        print("="*80)

        actual_numbers = core.get_actual_main("Saturday Lotto", target_date)
        if actual_numbers is None and target_date == real_draw_date:
            actual_numbers = real_draw_result
        if actual_numbers is None:
            print("[WARN] No actual numbers available - skipping run.")
            continue

        actual_numbers = sorted(actual_numbers)
        print(f"[HITS] Date {target_date} | Actual: {actual_numbers}")

        if USE_NEW_ALGO:
            tickets = _build_tickets_new_algo(target_date, MAX_TICKETS_TO_PRINT)
            if not tickets:
                print("[HITS] No tickets to evaluate.")
                continue
            best_hits = 0
            hit_summary = {"<3": 0, "3": 0, "4": 0, "5": 0, "6": 0}
            for i, nums in enumerate(tickets, 1):
                hits = sorted(set(nums) & set(actual_numbers))
                best_hits = max(best_hits, len(hits))
                hit_count = len(hits)
                if hit_count < 3:
                    hit_summary["<3"] += 1
                elif hit_count == 3:
                    hit_summary["3"] += 1
                elif hit_count == 4:
                    hit_summary["4"] += 1
                elif hit_count == 5:
                    hit_summary["5"] += 1
                elif hit_count == 6:
                    hit_summary["6"] += 1

                if hit_count == 3:
                    total_3_hits += 1
                if hit_count == 4:
                    total_4_hits += 1
                if hit_count == 5:
                    total_5_hits += 1
                if hit_count == 6:
                    total_6_hits += 1
                print(f"[HITS] Ticket #{i}: {nums} | Hits ({len(hits)}): {hits}")
            print(f"[HITS] Best ticket hits: {best_hits} of {len(actual_numbers)}")
            print(
                "[HITS] Summary: "
                f"<3={hit_summary['<3']}, "
                f"3={hit_summary['3']}, "
                f"4={hit_summary['4']}, "
                f"5={hit_summary['5']}, "
                f"6={hit_summary['6']}"
            )

            if best_hits < 3:
                weeks_lt3 += 1
            if best_hits >= 3:
                weeks_3p += 1
            if best_hits >= 4:
                weeks_4p += 1
            if best_hits >= 5:
                weeks_5p += 1
            if best_hits >= 6:
                weeks_6p += 1
            if best_hits > max_hit_observed:
                max_hit_observed = best_hits

    end_ts = time.time()
    end_dt = datetime.datetime.now()
    elapsed_sec = end_ts - start_ts

    print("\n=== BACKTEST SUMMARY (LAST 20 DRAWS) ===")
    print(f"Weeks with <3 hits: {weeks_lt3}")
    print(f"Weeks with 3+ hits: {weeks_3p}")
    print(f"Weeks with 4+ hits: {weeks_4p}")
    print(f"Weeks with 5+ hits: {weeks_5p}")
    print(f"Weeks with 6+ hits: {weeks_6p}")
    print(f"Max hit observed : {max_hit_observed}")
    print(f"Total hits (=3): {total_3_hits}")
    print(f"Total hits (=4): {total_4_hits}")
    print(f"Total hits (=5): {total_5_hits}")
    print(f"Total hits (=6): {total_6_hits}")

    print(f"\n=== RUN END ===")
    print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run time: {elapsed_sec:.2f} seconds "
          f"({elapsed_sec / 60:.2f} minutes)")
