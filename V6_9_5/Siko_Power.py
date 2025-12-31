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

core.addDraws()
core.finalize_data()

# can change below code
# *****************

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

LEADER_POOL_RANK_MAX = 5
LEARNING_DRAW_COUNT = 8
MAX_TICKETS_TO_PRINT = 1
COHORT_USAGE_CAP_FRAC = None      # e.g. 0.40 to cap cohort repeats
COHORT_AUTOPRED_EVAL_LAST_N = None  # e.g. 2 or 3 for auto predictor window
# OVERRIDE_COHORT_HWC = None          # e.g. (1, 6, 0)
OVERRIDE_COHORT_HWC = (1, 4, 1) # None          # e.g. (1, 6, 0)
# OVERRIDE_COHORT_DECADES = None
OVERRIDE_COHORT_DECADES = {1:2, 2:2, 3:1, 4:1}      # e.g. {1:2, 2:2, 3:2, 4:1}
OVERRIDE_RANK_MIN = None            # e.g. 10
OVERRIDE_RANK_MAX = None            # e.g. 30
OVERRIDE_P_MIN = None               # e.g. 0.010
OVERRIDE_P_MAX = None               # e.g. 0.030

if __name__ == "__main__":

    # today = datetime.date.today() # - datetime.timedelta(days=1)
    today = datetime.date(2026, 1, 1)  # keep explicit & reproducible
    tuesday_dates = last_n_thursday(today, 8)
    # print(tuesday_dates)
    tuesday_date_set = set(tuesday_dates)
    start_ts = time.time()
    start_dt = datetime.datetime.now()

    print(f"\n=== RUN START ===")
    print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    for target_date in reversed(tuesday_dates):
        print("\n" + "="*80)
        print(f"PREDICTION RUN FOR Powerball on {target_date}")
        print("="*80)

        core.PREDICTION_TARGET = ("Powerball", target_date, 7)
        core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()
        core.LOCKED_REGIME_DATES = tuesday_dates
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
        if run_data and run_data.get("prediction_actual") is None:
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
