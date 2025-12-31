import Siko_CoreV6_4 as core
import time
import datetime

core.global_draws.clear()

core.DECADE_BANDS = [
    (1,  1,  9),
    (2, 10, 19),
    (3, 20, 29),
    (4, 30, 39),
    (5, 40, 45),
]

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)


# --- Number range for Saturday Lotto mains (1..45) ---
core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 45
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))

# core.RUN_MODE = "predicting"
# SFL → Saturday hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Saturday Lotto"

core.addDraws()

core.PREDICTION_CONFIG = {
    "BASE_TRIALS": 80000,
    "MIN_TRIALS": 200000,
    "MAX_TRIALS": 250000,
    "CLUSTER_TRIAL_FRAC": 0.25,
    "HWC_OVERRIDE": (1, 4, 1),
    "DECADE_FACTORS_OVERRIDE": {
        1: 1.00,
        2: 1.35,
        3: 0.90,
        4: 1.00,
        5: 0.75,
    },
}

# Target draws and prediction
core.TARGET_DRAWS_FOR_LEARNING = [
    # ("Saturday Lotto",   core.d(15,11)),
    # ("Weekday Windfall", core.d(17,11)),   # adjust/remove if you don't have this data
    # ("OZ Lotto",         core.d(18,11)),
    # ("Weekday Windfall", core.d(19,11)),  # NOTE: you must ensure data exists for this date
    # ("Powerball",        core.d(20,11)),
    # ("Weekday Windfall", core.d(21,11)),
    ("Saturday Lotto",   core.d(22,11)),
    ("Weekday Windfall", core.d(24,11)),   # adjust/remove if you don't have this data
    ("OZ Lotto",         core.d(25,11)),
    ("Weekday Windfall", core.d(26,11)),  # NOTE: you must ensure data exists for this date
    ("Powerball",        core.d(27,11)),
    ("Weekday Windfall", core.d(28,11)),
    ("Saturday Lotto",   core.d(29,11)),
    ("Weekday Windfall", core.d(1,12)),   # adjust/remove if you don't have this data
    ("OZ Lotto",         core.d(2,12)),
    ("Weekday Windfall", core.d(3,12)),  # NOTE: you must ensure data exists for this date
    ("Powerball",        core.d(4,12)),
    ("Weekday Windfall", core.d(5,12)),
    ]


core.PREDICTION_TARGET = ("Saturday Lotto", core.d(6,12), 6)

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

