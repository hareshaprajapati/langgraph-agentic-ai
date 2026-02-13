import sys
import os
from datetime import datetime
from collections import Counter

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
    f"siko_power_logs.log"   # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import Siko_Core as core
import time
import datetime

core.global_draws.clear()

core.DECADE_BANDS = [
    (1, 1, 10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 35)
]

for dec_id, start, end in core.DECADE_BANDS:
    print(f"Decade {dec_id}: {start}–{end}")

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)


# --- Number range for Powerball mains (1..35) ---
core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 35
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))
core.WINDOW_LENGTH = (7, 8, 9)


# SFL → Powerball hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Powerball"
core.SFL_MOMENTUM_DAYS = 3
core.SFL_LASTN_DRAWS = 3
core.SFL_LASTN_PM1_K = 0.20
core.SFL_LASTN_PM1_W_MAX = 1.70
core.SFL_PM1_STRICT_COHORT = False
core.SFL_PM1_STRICT_LEADER = False
core.SFL_PM1_COMBO_ENABLED = True
core.SFL_PM1_COMBO_TOP_N = 16
core.SFL_PM1_COMBO_MAX = 20
core.TOP_P_COMBO_ENABLED = False
core.TOP_P_COMBO_N = 14
core.TOP_P_COMBO_MAX = 20
core.PM1_WEIGHTED_COMBO_ENABLED = True
core.PM1_WEIGHTED_COMBO_TOP_N = 18
core.PM1_WEIGHTED_COMBO_MAX = 20
core.PM1_WEIGHTED_SCORE_W = 3.0
core.PM1_GREEDY_ENABLED = True
core.PM1_GREEDY_TOP_N = 12
core.PM1_GREEDY_MAX = 20
core.PM1_ONLY_MODE = True
core.PM1_ONLY_TOP_N = 16

core.addDraws()
core.finalize_data()

# can change below code
# *****************

# Prediction overrides for stronger signal (kept modest for runtime).
core.PREDICTION_CONFIG = {
    "APPLY_PREDICTION_OVERRIDES": True,
    "BASE_TRIALS": 300,
    "MIN_TRIALS": 900,
    "MAX_TRIALS": 1200,
    "CLUSTER_TRIAL_FRAC": 0.25,
}

# core.PREDICTION_CONFIG = {
#     "BASE_TRIALS": 80000,
#     "MIN_TRIALS": 200000,
#     "MAX_TRIALS": 250000,
#     "CLUSTER_TRIAL_FRAC": 0.10,
#     "EXPLORE_FRAC": 0.65,
#     "APPLY_PREDICTION_OVERRIDES": True,
#     "HWC_OVERRIDE": (1, 6, 0),
#     "DECADE_FACTORS_OVERRIDE": {
#         1: 1.05,
#         2: 1.20,
#         3: 1.00,
#         4: 0.70,
#         # 5: 0.10,
#     },
# }

# core.PREDICTION_TARGET = ("Powerball", core.d(18,12), 7)
#
# core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()
#
# print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
# print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
# for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
#     print(" ", lot, dt)

# if __name__ == "__main__":
#     start_ts = time.time()
#     start_dt = datetime.datetime.now()
#
#     print(f"\n=== RUN START ===")
#     print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
#     core.main()
#     end_ts = time.time()
#     end_dt = datetime.datetime.now()
#
#     elapsed_sec = end_ts - start_ts
#
#     print(f"\n=== RUN END ===")
#     print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Total run time: {elapsed_sec:.2f} seconds "
#           f"({elapsed_sec / 60:.2f} minutes)")


def last_n_thursday(today, n):
    dates = []
    d = today
    while len(dates) < n:
        if d.weekday() == 3:  # Tuesday
            dates.append(d)
        d -= datetime.timedelta(days=1)
    return dates

LEADER_POOL_RANK_MAX = 8
LEARNING_DRAW_COUNT = 8
MAX_TICKETS_TO_PRINT = 10
COHORT_USAGE_CAP_FRAC = None      # e.g. 0.40 to cap cohort repeats
COHORT_AUTOPRED_EVAL_LAST_N = 2  # e.g. 2 or 3 for auto predictor window
# OVERRIDE_COHORT_HWC = None          # e.g. (1, 6, 0)
OVERRIDE_COHORT_HWC = None          # e.g. (1, 6, 0)
# OVERRIDE_COHORT_DECADES = None
OVERRIDE_COHORT_DECADES = {1:0, 2:4, 3:2, 4:0}      # e.g. {1:2, 2:2, 3:2, 4:1}
OVERRIDE_RANK_MIN = None            # e.g. 10
OVERRIDE_RANK_MAX = None            # e.g. 30
OVERRIDE_P_MIN = None               # e.g. 0.010
OVERRIDE_P_MAX = None               # e.g. 0.030

RUN_SWEEP = False
EARLY_STOP_ON_HIT5 = True
FAST_TRIALS = {
    "APPLY_PREDICTION_OVERRIDES": True,
    "BASE_TRIALS": 300,
    "MIN_TRIALS": 900,
    "MAX_TRIALS": 1200,
    "CLUSTER_TRIAL_FRAC": 0.25,
}


if __name__ == "__main__":
    REAL_RESULTS = {
        datetime.date(2026, 2, 12): [11, 12, 14, 18, 20, 21, 30],
    }
    # today = datetime.date.today() # - datetime.timedelta(days=1)
    today = datetime.date(2026, 2, 12)  # keep explicit & reproducible
    N = 8
    thursday_dates = last_n_thursday(today, N)
    # print(tuesday_dates)
    tuesday_date_set = set(thursday_dates)
    start_ts = time.time()
    start_dt = datetime.datetime.now()

    print(f"\n=== RUN START ===")
    print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    if RUN_SWEEP:
        configs = [
            {
                "name": "baseline_overrides_1t_lowtrials",
                "leader_pool_rank_max": 5,
                "max_tickets": 1,
                "usage_cap": None,
                "override_hwc": OVERRIDE_COHORT_HWC,
                "override_decades": OVERRIDE_COHORT_DECADES,
                "override_rank_min": OVERRIDE_RANK_MIN,
                "override_rank_max": OVERRIDE_RANK_MAX,
                "override_p_min": OVERRIDE_P_MIN,
                "override_p_max": OVERRIDE_P_MAX,
                "pred_config": None,
            },
            {
                "name": "no_overrides_5t_lowtrials",
                "leader_pool_rank_max": 5,
                "max_tickets": 5,
                "usage_cap": None,
                "override_hwc": None,
                "override_decades": None,
                "override_rank_min": None,
                "override_rank_max": None,
                "override_p_min": None,
                "override_p_max": None,
                "pred_config": None,
            },
            {
                "name": "no_overrides_10t_leader8_lowtrials",
                "leader_pool_rank_max": 8,
                "max_tickets": 10,
                "usage_cap": 0.30,
                "override_hwc": None,
                "override_decades": None,
                "override_rank_min": None,
                "override_rank_max": None,
                "override_p_min": None,
                "override_p_max": None,
                "pred_config": None,
            },
            {
                "name": "no_overrides_10t_leader8_midtrials",
                "leader_pool_rank_max": 8,
                "max_tickets": 10,
                "usage_cap": 0.30,
                "override_hwc": None,
                "override_decades": None,
                "override_rank_min": None,
                "override_rank_max": None,
                "override_p_min": None,
                "override_p_max": None,
                "pred_config": dict(FAST_TRIALS),
            },
            {
                "name": "no_overrides_20t_leader10_midtrials",
                "leader_pool_rank_max": 10,
                "max_tickets": 20,
                "usage_cap": 0.40,
                "override_hwc": None,
                "override_decades": None,
                "override_rank_min": None,
                "override_rank_max": None,
                "override_p_min": None,
                "override_p_max": None,
                "pred_config": dict(FAST_TRIALS),
            },
            {
                "name": "overrides_10t_midtrials",
                "leader_pool_rank_max": 8,
                "max_tickets": 10,
                "usage_cap": 0.30,
                "override_hwc": OVERRIDE_COHORT_HWC,
                "override_decades": OVERRIDE_COHORT_DECADES,
                "override_rank_min": OVERRIDE_RANK_MIN,
                "override_rank_max": OVERRIDE_RANK_MAX,
                "override_p_min": OVERRIDE_P_MIN,
                "override_p_max": OVERRIDE_P_MAX,
                "pred_config": dict(FAST_TRIALS),
            },
        ]

        sweep_results = []
        for cfg in configs:
            print("\n" + "=" * 80)
            print(f"SWEEP CONFIG: {cfg['name']}")
            print("=" * 80)

            core.PREDICTION_CONFIG = cfg["pred_config"] or {}
            core.COHORT_USAGE_CAP_FRAC = cfg["usage_cap"]
            core.COHORT_AUTOPRED_EVAL_LAST_N = COHORT_AUTOPRED_EVAL_LAST_N

            best_exact = -1
            best_date = None
            best_tickets = 0
            hit5_dates = []

            for target_date in reversed(thursday_dates):
                core.PREDICTION_TARGET = ("Powerball", target_date, 7)
                core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()
                core.LOCKED_REGIME_DATES = thursday_dates
                core.LOCKED_REGIME_LOTTERY = "Powerball"

                run_data = core.main()
                if not run_data or run_data.get("prediction_actual") is None:
                    continue

                hit_pack = core.build_locked_tickets(
                    run_data,
                    leader_pool_rank_max=cfg["leader_pool_rank_max"],
                    max_tickets_to_print=cfg["max_tickets"],
                    allowed_dates=tuesday_date_set,
                    allowed_lottery="Powerball",
                    override_cohort_hwc=cfg["override_hwc"],
                    override_cohort_decades=cfg["override_decades"],
                    override_rank_min=cfg["override_rank_min"],
                    override_rank_max=cfg["override_rank_max"],
                    override_p_min=cfg["override_p_min"],
                    override_p_max=cfg["override_p_max"],
                )

                if hit_pack is None:
                    continue

                tickets = hit_pack["tickets"]
                actual_numbers = run_data["prediction_actual"]["actual_numbers"]
                actual_set = set(actual_numbers)
                if not tickets:
                    continue

                local_best = -1
                for t in tickets:
                    nums = list(t["cohort"]) + [t["leader"]]
                    exact = core.exact_hits(nums, actual_set)
                    if exact > local_best:
                        local_best = exact

                if local_best > best_exact:
                    best_exact = local_best
                    best_date = target_date
                    best_tickets = len(tickets)

                if local_best >= 5:
                    hit5_dates.append(target_date)
                    if EARLY_STOP_ON_HIT5:
                        break

            if EARLY_STOP_ON_HIT5 and hit5_dates:
                sweep_results.append({
                    "name": cfg["name"],
                    "best_exact": max(best_exact, 5),
                    "best_date": hit5_dates[0],
                    "best_tickets": best_tickets,
                    "hit5_dates": hit5_dates,
                })
                print(f"\n[SWEEP] Hit 5 exact in config '{cfg['name']}', stopping early.")
                break

            sweep_results.append({
                "name": cfg["name"],
                "best_exact": best_exact,
                "best_date": best_date,
                "best_tickets": best_tickets,
                "hit5_dates": hit5_dates,
            })

        print("\n" + "=" * 80)
        print("SWEEP SUMMARY (BEST EXACT HITS)")
        print("=" * 80)
        for r in sweep_results:
            hit5_note = f" hit5_dates={len(r['hit5_dates'])}" if r["hit5_dates"] else ""
            print(
                f"{r['name']}: best_exact={r['best_exact']} "
                f"best_date={r['best_date']} tickets={r['best_tickets']}{hit5_note}"
            )
        end_ts = time.time()
        end_dt = datetime.datetime.now()
        elapsed_sec = end_ts - start_ts

        print(f"\n=== RUN END ===")
        print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total run time: {elapsed_sec:.2f} seconds "
              f"({elapsed_sec / 60:.2f} minutes)")
        raise SystemExit(0)

    for target_date in reversed(thursday_dates):
        print("\n" + "="*80)
        print(f"PREDICTION RUN FOR Powerball on {target_date}")
        print("="*80)

        core.PREDICTION_TARGET = ("Powerball", target_date, 7)
        core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()
        core.LOCKED_REGIME_DATES = thursday_dates
        core.LOCKED_REGIME_LOTTERY = "Powerball"
        core.COHORT_USAGE_CAP_FRAC = COHORT_USAGE_CAP_FRAC
        core.COHORT_AUTOPRED_EVAL_LAST_N = COHORT_AUTOPRED_EVAL_LAST_N

        # print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
        # print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
        # for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
        #     print(" ", lot, dt)


        run_data = core.main()
        if run_data and run_data.get("prediction_actual") is not None:
            core.LOCKED_REGIME_SNAPSHOT_CACHE[target_date] = run_data["prediction_actual"]
            hit_pack = core.build_locked_tickets(
                run_data,
                leader_pool_rank_max=LEADER_POOL_RANK_MAX,
                max_tickets_to_print=MAX_TICKETS_TO_PRINT,
                allowed_dates=tuesday_date_set,
                allowed_lottery="Powerball",
                override_cohort_hwc=OVERRIDE_COHORT_HWC,
                override_cohort_decades=OVERRIDE_COHORT_DECADES,
                override_rank_min=OVERRIDE_RANK_MIN,
                override_rank_max=OVERRIDE_RANK_MAX,
                override_p_min=OVERRIDE_P_MIN,
                override_p_max=OVERRIDE_P_MAX,
            )
            if hit_pack is None:
                print("\n=== HIT SUMMARY ===")
                print("No locked-regime context available to score.")
            else:
                tickets = hit_pack["tickets"]
                actual_numbers = run_data["prediction_actual"]["actual_numbers"]
                actual_set = set(actual_numbers)
                if not tickets:
                    print("\n=== HIT SUMMARY ===")
                    print("No tickets available to score.")
                else:
                    exact_hist = Counter()
                    near_hist = Counter()
                    best_exact = -1
                    best_near = -1
                    for t in tickets:
                        nums = list(t["cohort"]) + [t["leader"]]
                        exact = core.exact_hits(nums, actual_set)
                        near = core.near_miss_pm1_hits(nums, actual_set)
                        exact_hist[exact] += 1
                        near_hist[near] += 1
                        if exact > best_exact:
                            best_exact = exact
                        if near > best_near:
                            best_near = near

                    exact_counts = " ".join(f"{k}:{exact_hist[k]}" for k in sorted(exact_hist))
                    near_counts = " ".join(f"{k}:{near_hist[k]}" for k in sorted(near_hist))
                    print("\n=== HIT SUMMARY ===")
                    print(f"Tickets evaluated: {len(tickets)}")
                    print(f"Best exact hits: {best_exact} ({exact_hist[best_exact]} ticket(s))")
                    print(f"Exact hit counts: {exact_counts}")
                    print(f"Best +/-1 hits: {best_near} ({near_hist[best_near]} ticket(s))")
                    print(f"+/-1 hit counts: {near_counts}")
        if run_data and run_data.get("prediction_actual") is None:
            if target_date in REAL_RESULTS:
                actual_numbers = REAL_RESULTS[target_date]
                actual_set = set(actual_numbers)
                hit_pack = core.build_locked_tickets(
                    run_data,
                    leader_pool_rank_max=LEADER_POOL_RANK_MAX,
                    max_tickets_to_print=MAX_TICKETS_TO_PRINT,
                    allowed_dates=tuesday_date_set,
                    allowed_lottery="Powerball",
                    override_cohort_hwc=OVERRIDE_COHORT_HWC,
                    override_cohort_decades=OVERRIDE_COHORT_DECADES,
                    override_rank_min=OVERRIDE_RANK_MIN,
                    override_rank_max=OVERRIDE_RANK_MAX,
                    override_p_min=OVERRIDE_P_MIN,
                    override_p_max=OVERRIDE_P_MAX,
                )
                if hit_pack is None:
                    print("\n=== HIT SUMMARY ===")
                    print("No locked-regime context available to score.")
                else:
                    tickets = hit_pack["tickets"]
                    if not tickets:
                        print("\n=== HIT SUMMARY ===")
                        print("No tickets available to score.")
                    else:
                        exact_hist = Counter()
                        near_hist = Counter()
                        best_exact = -1
                        best_near = -1
                        for t in tickets:
                            nums = list(t["cohort"]) + [t["leader"]]
                            exact = core.exact_hits(nums, actual_set)
                            near = core.near_miss_pm1_hits(nums, actual_set)
                            exact_hist[exact] += 1
                            near_hist[near] += 1
                            if exact > best_exact:
                                best_exact = exact
                            if near > best_near:
                                best_near = near

                        exact_counts = " ".join(f"{k}:{exact_hist[k]}" for k in sorted(exact_hist))
                        near_counts = " ".join(f"{k}:{near_hist[k]}" for k in sorted(near_hist))
                        print("\n=== HIT SUMMARY ===")
                        print(f"Tickets evaluated: {len(tickets)}")
                        print(f"Best exact hits: {best_exact} ({exact_hist[best_exact]} ticket(s))")
                        print(f"Exact hit counts: {exact_counts}")
                        print(f"Best +/-1 hits: {best_near} ({near_hist[best_near]} ticket(s))")
                        print(f"+/-1 hit counts: {near_counts}")
            core.print_locked_prediction_steps(
                run_data,
                leader_pool_rank_max=LEADER_POOL_RANK_MAX,
                max_tickets_to_print=MAX_TICKETS_TO_PRINT,
                include_learning_scores=True,
                allowed_dates=tuesday_date_set,
                allowed_lottery="Powerball",
                override_cohort_hwc=OVERRIDE_COHORT_HWC,
                override_cohort_decades=OVERRIDE_COHORT_DECADES,
                override_rank_min=OVERRIDE_RANK_MIN,
                override_rank_max=OVERRIDE_RANK_MAX,
                override_p_min=OVERRIDE_P_MIN,
                override_p_max=OVERRIDE_P_MAX,
            )

    end_ts = time.time()
    end_dt = datetime.datetime.now()
    elapsed_sec = end_ts - start_ts

    print(f"\n=== RUN END ===")
    print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run time: {elapsed_sec:.2f} seconds "
          f"({elapsed_sec / 60:.2f} minutes)")
