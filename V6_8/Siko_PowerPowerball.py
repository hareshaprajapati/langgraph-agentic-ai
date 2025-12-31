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
    f"siko_run.log"  # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print("\n" + "=" * 80)
print(f"RUN START @ {datetime.now()}")
print("=" * 80)

import Siko_Core as core
import time
import datetime

core.global_draws.clear()

core.DECADE_BANDS = [
    (1,  1,  10),
    (2, 11, 20)
]


core.RUNNING_PB_ONLY = True

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)
NUMBER_OF_POWERBALL = 2
core.LOTTERIES["Powerball"]["main_draw_size"] = NUMBER_OF_POWERBALL   # or 2
# --- Number range for Powerball mains (1..35) ---
core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 20
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))


# SFL → Powerball hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Powerball"

core.addDraws()

# can change below code
# *****************

core.PREDICTION_CONFIG = {
    # "BASE_TRIALS": 80000,
    # "MIN_TRIALS": 200000,
    # "MAX_TRIALS": 250000,
    # "CLUSTER_TRIAL_FRAC": 0.10,
    # "EXPLORE_FRAC": 0.65,
    # "APPLY_PREDICTION_OVERRIDES": True,
    # must not be "HWC_OVERRIDE": (1, 6, 0),
    # "DECADE_FACTORS_OVERRIDE": {
    #     1: 1.80,
    #     2: 1.20,
    #     # 3: 1.00,
    #     # 4: 0.70,
    # },
}

core.PREDICTION_TARGET = ("Powerball", core.d(18,12), NUMBER_OF_POWERBALL)

# core.TOP_N_PREDICTIONS = 2      # you can change this to print more/less predictions
core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_two_weeks_back()

print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
    print(" ", lot, dt)

if __name__ == "__main__":
    start_ts = time.time()
    start_dt = datetime.datetime.now()

    print(f"\n=== RUN START ===")
    print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    core.main()
    end_ts = time.time()
    end_dt = datetime.datetime.now()

    elapsed_sec = end_ts - start_ts

    print(f"\n=== RUN END ===")
    print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total run time: {elapsed_sec:.2f} seconds "
          f"({elapsed_sec / 60:.2f} minutes)")