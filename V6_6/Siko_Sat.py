import Siko_CoreV6_6 as core
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


# SFL → Saturday hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Saturday Lotto"

core.addDraws()

# can change below code
# *****************

core.PREDICTION_CONFIG = {
    # "BASE_TRIALS": 80000,
    # "MIN_TRIALS": 200000,
    # "MAX_TRIALS": 250000,
    # "CLUSTER_TRIAL_FRAC": 0.25,
    # "APPLY_PREDICTION_OVERRIDES": True,
    # "HWC_OVERRIDE": (1, 5, 0),
    # "DECADE_FACTORS_OVERRIDE": {
    #     1: 1.00,
    #     2: 1.35,
    #     3: 0.75,
    #     4: 1.00,
    #     5: 1.10,
    # },
}

core.PREDICTION_TARGET = ("Saturday Lotto", core.d(13,12), 6)

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