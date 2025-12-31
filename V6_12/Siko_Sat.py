import sys
import os
from datetime import datetime
import math

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
    f"siko_run.log"   # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print("\n" + "="*80)
print(f"RUN START @ {datetime.now()}")
print("="*80)

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

core.DECADES = [b[0] for b in core.DECADE_BANDS]
core.N_DECADES = len(core.DECADES)


# --- Number range for Saturday Lotto mains (1..45) ---
core.MAIN_NUMBER_MIN = 1
core.MAIN_NUMBER_MAX = 45
core.NUMBER_RANGE = list(range(core.MAIN_NUMBER_MIN, core.MAIN_NUMBER_MAX + 1))
core.WINDOW_SIZE_CANDIDATES = [6, 7, 8, 9]

# SFL → Saturday hop (same as your current config)
core.HOP_SOURCE_LOTTERY = "Set for Life"
core.HOP_DESTINATION_LOTTERY = "Saturday Lotto"

core.addDraws()

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

core.PREDICTION_TARGET = ("Saturday Lotto", core.d(20,12), 6)

core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()

print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
    print(" ", lot, dt)

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


if __name__ == "__main__":

    today = datetime.date.today() - datetime.timedelta(days=1)
    # today = datetime.date(2025, 12, 20)  # keep explicit & reproducible
    saturday_dates = last_n_saturdays(today, 5)
    all_runs = []
    final_payload = None
    for target_date in saturday_dates:
        print("\n" + "="*80)
        print(f"PREDICTION RUN FOR Saturday Lotto on {target_date}")
        print("="*80)

        core.PREDICTION_TARGET = ("Saturday Lotto", target_date, 6)
        core.TARGET_DRAWS_FOR_LEARNING = core.build_targets_for_learning()

        print("\n[SANITY] PREDICTION_TARGET =", core.PREDICTION_TARGET)
        print("[SANITY] TARGET_DRAWS_FOR_LEARNING =")
        for lot, dt in core.TARGET_DRAWS_FOR_LEARNING:
            print(" ", lot, dt)

        start_ts = time.time()
        start_dt = datetime.datetime.now()

        print(f"\n=== RUN START ===")
        print(f"Start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

        payload = core.main()
        all_runs.append(payload)

        end_ts = time.time()
        end_dt = datetime.datetime.now()
        elapsed_sec = end_ts - start_ts

        print(f"\n=== RUN END ===")
        print(f"End time:   {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total run time: {elapsed_sec:.2f} seconds "
              f"({elapsed_sec / 60:.2f} minutes)")

    history_runs = [r for r in all_runs if r["actual_main"] is not None]
    final_run = next(r for r in all_runs if r["target_date"] == saturday_dates[0])
    chosen_window_len, learned_start_rank, learned_hwc, learned_dec = core.derive_learning_from_history(
        history_runs,
        decade_bands=core.DECADE_BANDS,
        window_lens=(6, 7, 8, 9),
    )
    learning_table = []

    for r in history_runs:
        learning_table.append({
            "date": r["target_date"],
            "rank_rows": r["rank_rows"],
            "actual": r["actual_main"],
        })

    # learning_table: list of dict rows from past runs where actual exists
    # Each row should at least have:
    #   row["best_start_rank"], row["actual_hwc"], row["actual_dec"], row["best_window_len"]

    from statistics import median
    from collections import Counter


    def _mode(items):
        if not items:
            return None
        c = Counter(items)
        return c.most_common(1)[0][0]


    # 1) choose window_len
    # lens = [r["best_window_len"] for r in learning_table if r.get("best_window_len") is not None]
    # chosen_window_len = _mode(lens) or 9
    #
    # # 2) learned start rank
    # starts = [r["best_start_rank"] for r in learning_table if r.get("best_start_rank") is not None]
    # learned_start_rank = int(round(median(starts))) if starts else None
    #
    # # 3) learned HWC + Decades
    # hwcs = [tuple(r["actual_hwc"]) for r in learning_table if r.get("actual_hwc")]
    # learned_hwc = _mode(hwcs)
    #
    # # For dicts, convert to stable tuple for mode
    # decs = []
    # for r in learning_table:
    #     d = r.get("actual_dec")
    #     if d:
    #         decs.append(tuple((k, d.get(k, 0)) for k in sorted(d.keys())))
    # learned_dec_t = _mode(decs)
    # learned_dec = dict(learned_dec_t) if learned_dec_t else None

    print("=" * 80)
    print("LEARNED FROM HISTORY (ACTUAL DRAWS)")
    print("=" * 80)
    print(f"chosen_window_len={chosen_window_len}")
    print(f"learned_start_rank(median best start)={learned_start_rank}")
    print(f"learned_hwc(mode)={learned_hwc}")
    print(f"learned_dec(mode)={learned_dec}")
    print("=" * 80)

    result = core.build_tickets_from_run_payload(
        final_run,
        window_len=chosen_window_len,
        n_tickets=10,
        seed=0,
        exclude_hot=False,
        target_hwc=learned_hwc,
        target_dec=learned_dec,
        learned_start_rank=learned_start_rank,
    )

    print("result of build_tickets_from_run_payload:", result)
    print("\n=== FINAL WINDOW PICK ===")
    print(result["best_window"])
    bw = result["best_window"]
    print(
        f"\nChosen window ranks[{bw['start_rank']}..{bw['end_rank']}] sumP={bw['sumP']:.6f} decades={bw['decades']} nums={bw['nums']}")


    print("\n=== FINAL TICKETS ===")
    tickets = result["tickets"]
    if not tickets:
        print("NO TICKETS BUILT (constraints too strict or candidate pool too small)")
    else:
        for i, t in enumerate(tickets, start=1):
            print(
                f"T#{i:02d} {t['nums']}  prodP={t['prodP']:.12f}  sumLog={t['sumLog']:.6f}  hwc={t['hwc']}  dec={t['dec']}")


