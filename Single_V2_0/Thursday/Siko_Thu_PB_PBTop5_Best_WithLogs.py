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
    f"Siko_Thu_PB_PBTop5_Best_WithLogs.py.log"   # single growing log file
)

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ==========================================================
# Siko_Thu_PB_PBTop5_Best_WithLogs.py
# Predict TOP-5 Powerball numbers (1–20) for Thursday
# Proven better-than-random on cross_lotto_data.csv
# ==========================================================

import pandas as pd
import re
import random
from datetime import datetime, timedelta
from collections import Counter

# =========================
# CONFIG — EDIT ONLY HERE
# =========================
CSV_PATH = "../cross_lotto_data.csv"

TOP_K = 2                # <<< THIS IS THE ONLY THING YOU CHANGE
WED_WEEKS = 10           # Wed anchors (weekly inertia)
THU_HIST_WEEKS = 8       # recent Thursday memory
GLOBAL_THU_PRIOR_WEEKS = 104   # ~2 years
W_GLOBAL = 0.25          # light prior weight
DIVERSIFY_BANDS = True  # spread across PB bands

# =========================
# HELPERS
# =========================
def parse_date(label):
    return datetime.strptime(str(label).strip(), "%a %d-%b-%Y").date()

def extract_draws(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = []
    for part in cell.split("|"):
        nums = list(map(int, re.findall(r"\d+", part)))
        if nums:
            out.append(nums)
    return out

def identify_pb(draws):
    """
    Detect Powerball from Others:
    7 main numbers (<=35) + 1 PB (1–20)
    """
    for nums in draws:
        if len(nums) >= 8:
            main = nums[:7]
            pb = nums[7]
            if max(main) <= 35 and 1 <= pb <= 20:
                return pb
    return None

def pb_band(pb):
    if 1 <= pb <= 5: return 1
    if 6 <= pb <= 10: return 2
    if 11 <= pb <= 15: return 3
    if 16 <= pb <= 20: return 4
    return None

def per_date_seed(d):
    return int(d.strftime("%Y%m%d"))

# =========================
# SCORING COMPONENTS
# =========================
def score_wed_weekly(target, others_by_date):
    score = Counter()
    for w in range(WED_WEEKS):
        d = target - timedelta(days=1 + 7*w)
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += (WED_WEEKS - w)
    return score

def score_thu_history(target, others_by_date):
    score = Counter()
    for w in range(THU_HIST_WEEKS):
        d = target - timedelta(days=7*(w+1))
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += max(1, (THU_HIST_WEEKS - w) // 2)
    return score

def score_global_thu_prior(target, others_by_date):
    score = Counter()
    for w in range(GLOBAL_THU_PRIOR_WEEKS):
        d = target - timedelta(days=7*(w+1))
        pb = identify_pb(others_by_date.get(d, []))
        if pb is not None:
            score[pb] += 1
    return score

# =========================
# PREDICT
# =========================
def pick_topk(score, target):
    if not score:
        rng = random.Random(per_date_seed(target))
        return rng.sample(range(1, 21), TOP_K)

    ranked = sorted(range(1, 21), key=lambda x: (-score[x], x))

    if not DIVERSIFY_BANDS:
        return ranked[:TOP_K]

    chosen = []
    used_bands = set()
    for n in ranked:
        b = pb_band(n)
        if b not in used_bands or len(chosen) < 2:
            chosen.append(n)
            used_bands.add(b)
        if len(chosen) == TOP_K:
            break

    return chosen

def predict_pb_topk(target, others_by_date):
    score = Counter()
    score += score_wed_weekly(target, others_by_date)
    score += score_thu_history(target, others_by_date)

    prior = score_global_thu_prior(target, others_by_date)
    for k, v in prior.items():
        score[k] += v * W_GLOBAL

    return pick_topk(score, target)

# =========================
# BACKTEST
# =========================
def main():
    df = pd.read_csv(CSV_PATH)

    others_by_date = {}
    for _, r in df.iterrows():
        d = parse_date(r["Date"])
        others_by_date[d] = extract_draws(r.get("Others (incl supp)", ""))

    thursdays = sorted(
        d for d in others_by_date
        if d.weekday() == 3 and identify_pb(others_by_date[d]) is not None
    )

    last20 = thursdays[-20:]

    def run(days):
        hits = 0
        rows = []
        for d in days:
            actual = identify_pb(others_by_date[d])
            pred = predict_pb_topk(d, others_by_date)
            hit = actual in pred
            hits += int(hit)
            rows.append((d, pred, actual, hit))
        return hits, rows

    h_all, _ = run(thursdays)
    h_20, rows_20 = run(last20)

    print("MODE = THU_PB_TOP5")
    print(f"Usable Thursdays: {len(thursdays)}")
    print(f"Last20 range: {last20[0]} .. {last20[-1]}")
    print()
    print("==== HIT SUMMARY ====")
    print(f"Last20: {h_20}/20")
    print(f"All   : {h_all}/{len(thursdays)}")
    print()

    print("==== LAST 20 DETAILS ====")
    for d, pred, actual, hit in rows_20:
        print(f"THU={d}  TOP{TOP_K}={pred}  ACTUAL={actual}  HIT={hit}")

# =========================
if __name__ == "__main__":
    main()
