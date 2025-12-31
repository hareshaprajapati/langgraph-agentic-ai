# sat_v6_2.py
import Siko_CoreV6_3 as core   # or whatever the core file is actually called

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


def addDraws():
    # Set for Life
    core.add_draw(core.d(10, 12), "Set for Life", [1, 36, 40, 28, 37, 10, 3], [12, 19])
    core.add_draw(core.d(9, 12), "Set for Life", [12, 20, 16, 38, 26, 13, 39], [22, 40])
    core.add_draw(core.d(8, 12), "Set for Life", [39, 4, 42, 11, 16, 43, 37], [21, 32])
    core.add_draw(core.d(7, 12), "Set for Life", [4, 34, 30, 21, 23, 35, 15], [22, 18])
    core.add_draw(core.d(6, 12), "Set for Life", [42, 15, 24, 31, 5, 40, 39], [19, 1])
    core.add_draw(core.d(5, 12), "Set for Life", [5, 25, 21, 17, 31, 1, 15], [24, 22])
    core.add_draw(core.d(4, 12), "Set for Life", [35, 2, 25, 8, 6, 17, 28], [3, 31])
    core.add_draw(core.d(3, 12), "Set for Life", [22, 29, 44, 31, 10, 25, 30], [8, 14])
    core.add_draw(core.d(2, 12), "Set for Life", [37, 13, 15, 19, 25, 39, 26], [3, 5])
    core.add_draw(core.d(1, 12), "Set for Life", [18, 1, 10, 41, 24, 11, 3], [25, 2])
    core.add_draw(core.d(30, 11), "Set for Life", [7, 44, 18, 27, 32, 22, 11], [38, 9])
    core.add_draw(core.d(29, 11), "Set for Life", [8, 31, 4, 6, 42, 16, 14], [13, 19])
    core.add_draw(core.d(28, 11), "Set for Life", [15, 27, 8, 39, 5, 43, 20], [19, 29])
    core.add_draw(core.d(27, 11), "Set for Life", [12, 36, 6, 7, 37, 41, 29], [8, 43])
    core.add_draw(core.d(26, 11), "Set for Life", [29, 37, 34, 14, 5, 21, 20], [18, 19])
    core.add_draw(core.d(25, 11), "Set for Life", [26, 16, 23, 15, 31, 1, 27], [8, 41])
    core.add_draw(core.d(24, 11), "Set for Life", [41, 1, 17, 29, 14, 40, 22], [35, 31])
    core.add_draw(core.d(23, 11), "Set for Life", [25, 27, 42, 18, 26, 9, 33], [22, 19])
    core.add_draw(core.d(22, 11), "Set for Life", [24, 23, 31, 30, 26, 5, 17], [6, 27])
    core.add_draw(core.d(21, 11), "Set for Life", [27, 32, 10, 42, 38, 33, 17], [19, 39])
    core.add_draw(core.d(20, 11), "Set for Life", [28, 10, 11, 35, 34, 41, 23], [30, 26])
    core.add_draw(core.d(19, 11), "Set for Life", [4, 44, 5, 33, 21, 30, 39], [9, 18])
    core.add_draw(core.d(18, 11), "Set for Life", [33, 35, 44, 32, 20, 29, 39], [5, 41])
    core.add_draw(core.d(17, 11), "Set for Life", [15, 23, 40, 43, 28, 1, 37], [18, 34])
    core.add_draw(core.d(16, 11), "Set for Life", [8, 19, 21, 27, 40, 14, 7], [20, 44])
    core.add_draw(core.d(15, 11), "Set for Life", [13, 4, 27, 14, 2, 5, 42], [33, 39])
    core.add_draw(core.d(14, 11), "Set for Life", [7, 25, 23, 35, 13, 18, 6], [3, 39])
    core.add_draw(core.d(13, 11), "Set for Life", [25, 24, 3, 21, 5, 33, 36], [22, 11])
    core.add_draw(core.d(12, 11), "Set for Life", [15, 20, 29, 21, 5, 10, 6], [32, 17])
    core.add_draw(core.d(11, 11), "Set for Life", [4, 7, 10, 44, 32, 30, 26], [5, 18])
    core.add_draw(core.d(10, 11), "Set for Life", [5, 36, 13, 23, 39, 3, 9], [35, 6])
    core.add_draw(core.d(9, 11), "Set for Life", [11, 4, 44, 26, 6, 31, 40], [21, 33])
    core.add_draw(core.d(8, 11), "Set for Life", [7, 31, 5, 37, 43, 38, 2], [42, 10])
    core.add_draw(core.d(7, 11), "Set for Life", [30, 18, 6, 28, 33, 41, 14], [38, 29])
    core.add_draw(core.d(6, 11), "Set for Life", [12, 20, 35, 42, 41, 10, 18], [33, 32])
    core.add_draw(core.d(5, 11), "Set for Life", [16, 22, 13, 34, 25, 3, 18], [33, 43])
    core.add_draw(core.d(4, 11), "Set for Life", [38, 9, 27, 25, 10, 23, 37], [13, 17])
    core.add_draw(core.d(3, 11), "Set for Life", [8, 15, 25, 26, 13, 24, 23], [4, 2])
    core.add_draw(core.d(2, 11), "Set for Life", [6, 28, 26, 24, 13, 11, 19], [22, 12])
    core.add_draw(core.d(1, 11), "Set for Life", [8, 31, 42, 24, 15, 7, 4], [19, 18])

    # Weekday Windfall
    core.add_draw(core.d(10, 12), "Weekday Windfall", [15, 2, 10, 33, 38, 26], [19, 14])
    core.add_draw(core.d(8, 12), "Weekday Windfall", [26, 40, 6, 39, 37, 12], [24, 7])
    core.add_draw(core.d(5, 12), "Weekday Windfall", [9, 23, 8, 16, 11, 33], [34, 1])
    core.add_draw(core.d(3, 12), "Weekday Windfall", [15, 2, 38, 37, 22, 35], [39, 6])
    core.add_draw(core.d(1, 12), "Weekday Windfall", [8, 6, 30, 38, 36, 1], [43, 5])
    core.add_draw(core.d(28, 11), "Weekday Windfall", [30, 8, 25, 43, 39, 24], [21, 1])
    core.add_draw(core.d(26, 11), "Weekday Windfall", [44, 43, 8, 36, 16, 27], [31, 30])
    core.add_draw(core.d(24, 11), "Weekday Windfall", [44, 15, 20, 17, 4, 18], [7, 11])
    core.add_draw(core.d(21, 11), "Weekday Windfall", [4, 5, 26, 10, 40, 20], [14, 24])
    core.add_draw(core.d(19, 11), "Weekday Windfall", [43, 26, 35, 25, 42, 13], [24, 5])
    core.add_draw(core.d(17, 11), "Weekday Windfall", [37, 11, 4, 2, 5, 7], [30, 22])
    core.add_draw(core.d(14, 11), "Weekday Windfall", [34, 11, 28, 15, 44, 31], [9, 20])
    core.add_draw(core.d(12, 11), "Weekday Windfall", [35, 11, 33, 15, 34, 45], [8, 37])
    core.add_draw(core.d(10, 11), "Weekday Windfall", [38, 3, 31, 22, 28, 5], [26, 14])
    core.add_draw(core.d(7, 11), "Weekday Windfall", [31, 16, 23, 30, 6, 3], [13, 18])
    core.add_draw(core.d(5, 11), "Weekday Windfall", [26, 15, 18, 27, 7, 37], [19, 44])
    core.add_draw(core.d(3, 11), "Weekday Windfall", [25, 14, 29, 23, 45, 13], [31, 8])

    # OZ Lotto
    core.add_draw(core.d(9, 12), "OZ Lotto", [21, 15, 3, 6, 9, 33, 19], [31, 14, 7])
    core.add_draw(core.d(2, 12), "OZ Lotto", [40, 26, 43, 28, 22, 42, 7], [29, 6, 47])
    core.add_draw(core.d(25, 11), "OZ Lotto", [12, 43, 28, 1, 47, 35, 14], [15, 16, 46])
    core.add_draw(core.d(18, 11), "OZ Lotto", [39, 2, 22, 8, 27, 6, 4], [47, 5, 24])
    core.add_draw(core.d(11, 11), "OZ Lotto", [44, 30, 7, 28, 17, 34, 42], [20, 32, 3])
    core.add_draw(core.d(4, 11), "OZ Lotto", [21, 17, 43, 25, 12, 18, 14], [15, 42, 24])

    # Powerball
    core.add_draw(core.d(4, 12), "Powerball", [19, 23, 32, 12, 11, 15, 9], None, [14])
    core.add_draw(core.d(27, 11), "Powerball", [2, 17, 11, 9, 19, 28, 24], None, [1])
    core.add_draw(core.d(20, 11), "Powerball", [19, 11, 12, 4, 29, 13, 27], None, [20])
    core.add_draw(core.d(13, 11), "Powerball", [22, 10, 6, 15, 2, 8, 7], None, [13])
    core.add_draw(core.d(6, 11), "Powerball", [11, 34, 7, 33, 15, 22, 16], None, [13])

    # Saturday Lotto
    core.add_draw(core.d(6, 12), "Saturday Lotto", [17, 42, 5, 10, 33, 45], [31, 44])
    core.add_draw(core.d(29, 11), "Saturday Lotto", [22, 10, 17, 5, 44, 36], [3, 11])
    core.add_draw(core.d(22, 11), "Saturday Lotto", [7, 31, 15, 39, 42, 12], [5, 8])
    core.add_draw(core.d(15, 11), "Saturday Lotto", [36, 19, 33, 41, 39, 1], [25, 20])
    core.add_draw(core.d(8, 11), "Saturday Lotto", [28, 13, 1, 41, 14, 16], [39, 34])
    core.add_draw(core.d(1, 11), "Saturday Lotto", [42, 31, 21, 28, 17, 13], [36, 15])

addDraws()
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

