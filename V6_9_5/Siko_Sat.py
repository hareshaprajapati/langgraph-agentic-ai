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

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# print("\n" + "="*80)
# print(f"RUN START @ {datetime.now()}")
# print("="*80)

import Siko_Core as core
import time
import datetime

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

# core.PREDICTION_CONFIG = {
#     # "BASE_TRIALS": 80000,
#     # "MIN_TRIALS": 200000,
#     # "MAX_TRIALS": 250000,
#     # "CLUSTER_TRIAL_FRAC": 0.25,
#     # "APPLY_PREDICTION_OVERRIDES": True,
#     "CLUSTER_TRIAL_FRAC": 0.10,
#     "EXPLORE_FRAC": 0.65,
#     "APPLY_PREDICTION_OVERRIDES": True,
#     "HWC_OVERRIDE": (0, 2, 4),
#     "DECADE_FACTORS_OVERRIDE": {1: 1.0, 2: 1.6, 3: 1.0, 4: 1.4, 5: 0.7},
#
# }

# core.PREDICTION_TARGET = ("Saturday Lotto", core.d(20,12), 6)
#
# core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()

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

def last_n_saturdays(today, n):
    dates = []
    d = today
    while len(dates) < n:
        if d.weekday() == 5:  # Saturday
            dates.append(d)
        d -= datetime.timedelta(days=1)
    return dates

LEADER_POOL_RANK_MAX = 5
LEARNING_DRAW_COUNT = 8
MAX_TICKETS_TO_PRINT = 10
COHORT_USAGE_CAP_FRAC = 0.40      # e.g. 0.40 to cap cohort repeats
COHORT_AUTOPRED_EVAL_LAST_N = None  # e.g. 2 or 3 for auto predictor window
# OVERRIDE_COHORT_HWC = None         # e.g. (0, 2, 4)
OVERRIDE_COHORT_HWC = (0, 4, 1)          # e.g. (0, 2, 4)
# OVERRIDE_COHORT_DECADES = None      # e.g. {1:1, 2:2, 3:0, 4:2, 5:1}
OVERRIDE_COHORT_DECADES = {1:2, 2:1, 3:1, 4:1, 5:0}      # e.g. {1:1, 2:2, 3:0, 4:2, 5:1}
# OVERRIDE_RANK_MIN = 5            # e.g. 12
# OVERRIDE_RANK_MAX = 39            # e.g. 40
OVERRIDE_RANK_MIN = None           # e.g. 12
OVERRIDE_RANK_MAX = None            # e.g. 40
OVERRIDE_P_MIN = None               # e.g. 0.010
OVERRIDE_P_MAX = None               # e.g. 0.030


if __name__ == "__main__":

    today = datetime.date.today() # - datetime.timedelta(days=1)
    # today = datetime.date(2025, 12, 27)  # keep explicit & reproducible
    saturday_dates = last_n_saturdays(today, 6)

    start_ts = time.time()
    start_dt = datetime.datetime.now()

    print(f"\n=== RUN START ===")
    print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    saturday_date_set = set(saturday_dates)
    for target_date in reversed(saturday_dates):
        print("\n" + "="*80)
        print(f"PREDICTION RUN FOR Saturday Lotto on {target_date}")
        print("="*80)

        core.PREDICTION_TARGET = ("Saturday Lotto", target_date, 6)
        core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()
        core.LOCKED_REGIME_DATES = saturday_dates
        core.LOCKED_REGIME_LOTTERY = "Saturday Lotto"
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
                allowed_dates=saturday_date_set,
                allowed_lottery="Saturday Lotto",
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
