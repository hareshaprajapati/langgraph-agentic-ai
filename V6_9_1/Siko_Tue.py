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
    f"siko_tue_logs.log"   # single growing log file
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
    (1, 1, 10),
    (2, 11, 20),
    (3, 21, 30),
    (4, 31, 40),
    (5, 41, 47),
]

for dec_id, start, end in core.DECADE_BANDS:
    print(f"Decade {dec_id}: {start}–{end}")

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)

# --- Number range for OZ Lotto mains (1..47) ---

core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 47
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))
core.WINDOW_LENGTH = (7, 8, 9)


# SFL → OZ Lotto hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "OZ Lotto"

# core.EXPLORE_FRAC = 0.40

core.addDraws()

# core.EXPLORE_FRAC = 0.50

# can change below code
# *****************

# core.PREDICTION_CONFIG = {
#     # "BASE_TRIALS": 80000,
#     # "MIN_TRIALS": 200000,
#     # "MAX_TRIALS": 250000,
#     # "CLUSTER_TRIAL_FRAC": 0.0,
#     # "HWC_OVERRIDE": (1, 3, 3),
#     # "APPLY_PREDICTION_OVERRIDES": True,
#     # "EXPLORE_FRAC": 0.65,  # <--- Use 0.65 here to force the variety
#     # "DECADE_FACTORS_OVERRIDE": {
#     #     1: 2.2,
#     #     2: 1.4,
#     #     3: 1.8,
#     #     4: 0.0,
#     #     5: 3.0
#     # },
# }

# core.PREDICTION_TARGET = ("OZ Lotto", core.d(23,12), 7)
#
# core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()
#
# print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
# print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
# for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
#     print(" ", lot, dt)
#
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

def last_n_tuesday(today, n):
    dates = []
    d = today
    while len(dates) < n:
        if d.weekday() == 1:  # Tuesday
            dates.append(d)
        d -= datetime.timedelta(days=1)
    return dates

if __name__ == "__main__":

    today = datetime.date.today() # - datetime.timedelta(days=1)
    # today = datetime.date(2025, 12, 20)  # keep explicit & reproducible
    tuesday_dates = last_n_tuesday(today, 5)
    # print(tuesday_dates)
    start_ts = time.time()
    start_dt = datetime.datetime.now()

    print(f"\n=== RUN START ===")
    print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    for target_date in reversed(tuesday_dates):
        print("\n" + "="*80)
        print(f"PREDICTION RUN FOR OZ Lotto on {target_date}")
        print("="*80)

        core.PREDICTION_TARGET = ("OZ Lotto", target_date, 7)
        core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()

        # print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
        # print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
        # for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
        #     print(" ", lot, dt)


        core.main()

    end_ts = time.time()
    end_dt = datetime.datetime.now()
    elapsed_sec = end_ts - start_ts

    print(f"\n=== RUN END ===")
    print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run time: {elapsed_sec:.2f} seconds "
          f"({elapsed_sec / 60:.2f} minutes)")