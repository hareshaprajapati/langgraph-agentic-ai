import Siko_CoreV6_5 as core

core.global_draws.clear()

core.DECADE_BANDS = [
    (1,  1,  9),
    (2, 10, 19),
    (3, 20, 29),
    (4, 30, 35)
]

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)


# --- Number range for Powerball mains (1..35) ---
core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 35
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))


# SFL → Powerball hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Powerball"

core.addDraws()

# core.RUN_MODE = "learning"
# core.RUN_MODE = "predicting"
# Optional: prediction-specific tuning for Saturday
# core.PREDICTION_CONFIG = {
#     "BASE_TRIALS": 80000,
#     "MIN_TRIALS": 200000,
#     "MAX_TRIALS": 250000,
#     "CLUSTER_TRIAL_FRAC": 0.10,
#     "HWC_OVERRIDE": (1, 6, 0),
#     "DECADE_FACTORS_OVERRIDE": {
#         1: 1.05,
#         2: 1.20,
#         3: 1.00,
#         4: 0.70,
#         # 5: 0.10,
#     },
# }


# Target draws and prediction
core.TARGET_DRAWS_FOR_LEARNING = [
    # ("OZ Lotto",         core.d(18,11)),
    # ("Weekday Windfall", core.d(19,11)),  # NOTE: you must ensure data exists for this date
    ("Powerball", core.d(20, 11)),
    ("Weekday Windfall", core.d(21, 11)),
    ("Saturday Lotto", core.d(22, 11)),
    ("Weekday Windfall", core.d(24, 11)),  # adjust/remove if you don't have this data
    ("OZ Lotto", core.d(25, 11)),
    ("Weekday Windfall", core.d(26, 11)),  # NOTE: you must ensure data exists for this date
    ("Powerball", core.d(27, 11)),
    ("Weekday Windfall", core.d(28, 11)),
    ("Saturday Lotto", core.d(29, 11)),
    ("Weekday Windfall", core.d(1, 12)),  # adjust/remove if you don't have this data
    ("OZ Lotto", core.d(2, 12)),
    ("Weekday Windfall", core.d(3, 12)),  # NOTE: you must ensure data exists for this date
    ("Powerball",        core.d(4,12)),
    ("Weekday Windfall", core.d(5,12)),
    ("Saturday Lotto",   core.d(6,12)),
    ("Weekday Windfall", core.d(8,12)),
    ("OZ Lotto", core.d(9, 12)),
    ("Weekday Windfall", core.d(10, 12)),
    ]
core.PREDICTION_TARGET = ("Powerball", core.d(11,12), 7)



if __name__ == "__main__":
    core.main()

