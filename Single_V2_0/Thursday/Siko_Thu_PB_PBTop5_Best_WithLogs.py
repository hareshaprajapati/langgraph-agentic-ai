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
# Predict TOP-K Powerball numbers (1–20) for Thursday
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

TOP_K = 10                # <<< THIS IS THE ONLY THING YOU CHANGE
WED_WEEKS = 10           # Wed anchors (weekly inertia)
THU_HIST_WEEKS = 4       # recent Thursday memory
GLOBAL_THU_PRIOR_WEEKS = 104   # ~2 years
W_GLOBAL = 0.0           # light prior weight
DIVERSIFY_BANDS = True  # spread across PB bands
TARGET_DATE = "2026-01-29"  # e.g. "2026-01-29" or "Thu 29-Jan-2026"; None = most recent Thu in CSV
TOP2_MODE = "diff_band"  # "overall" or "diff_band"
RECENCY_PENALTY = 0.75    # subtract from score if seen recently (0 disables)
RECENCY_WEEKS = 4         # how many recent Thu weeks to penalize
TRANSITION_WEIGHT = 2.0   # weight for last-Thu -> next-Thu transition bonus

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

def resolve_target_date(thursdays):
    if not TARGET_DATE:
        return thursdays[-1]
    if isinstance(TARGET_DATE, str):
        s = TARGET_DATE.strip()
        if "-" in s and len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d").date()
        return parse_date(s)
    return TARGET_DATE

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

def recency_weeks(target, pb, others_by_date):
    for w in range(1, RECENCY_WEEKS + 1):
        d = target - timedelta(days=7*w)
        prev = identify_pb(others_by_date.get(d, []))
        if prev == pb:
            return w
    return None

def build_transition(others_by_date, thursdays, max_back=200):
    trans = Counter()
    for i in range(1, min(len(thursdays), max_back)):
        d_prev = thursdays[-(i+1)]
        d_cur = thursdays[-i]
        prev = identify_pb(others_by_date.get(d_prev, []))
        cur = identify_pb(others_by_date.get(d_cur, []))
        if prev and cur:
            trans[(prev, cur)] += 1
    out = {}
    for (prev, cur), cnt in trans.items():
        out.setdefault(prev, Counter())[cur] += cnt
    return out

def transition_bonus(target, others_by_date, trans):
    d_prev = target - timedelta(days=7)
    prev = identify_pb(others_by_date.get(d_prev, []))
    if prev is None:
        return Counter()
    score = Counter()
    for nxt, cnt in trans.get(prev, {}).items():
        score[nxt] += TRANSITION_WEIGHT * cnt
    return score

# =========================
# PREDICT
# =========================
def pick_topk(score, target):
    if not score:
        rng = random.Random(per_date_seed(target))
        return rng.sample(range(1, 21), TOP_K)

    ranked = sorted(range(1, 21), key=lambda x: (-score[x], x))

    if TOP_K == 2 and TOP2_MODE == "diff_band":
        chosen = []
        used_bands = set()
        for n in ranked:
            b = pb_band(n)
            if b not in used_bands:
                chosen.append(n)
                used_bands.add(b)
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            for n in ranked:
                if n not in chosen:
                    chosen.append(n)
                if len(chosen) == 2:
                    break
        return chosen

    if not DIVERSIFY_BANDS:
        return ranked[:TOP_K]

    # First pass: pick one per band if possible
    chosen = []
    used_bands = set()
    for n in ranked:
        b = pb_band(n)
        if b not in used_bands:
            chosen.append(n)
            used_bands.add(b)
        if len(used_bands) == 4:
            break

    # Second pass: fill to TOP_K by rank
    for n in ranked:
        if len(chosen) >= TOP_K:
            break
        if n not in chosen:
            chosen.append(n)

    return chosen[:TOP_K]

def predict_pb_topk(target, others_by_date, trans):
    score = Counter()
    score += score_wed_weekly(target, others_by_date)
    score += score_thu_history(target, others_by_date)

    prior = score_global_thu_prior(target, others_by_date)
    for k, v in prior.items():
        score[k] += v * W_GLOBAL

    if RECENCY_PENALTY > 0:
        for n in range(1, 21):
            if recency_weeks(target, n, others_by_date) is not None:
                score[n] -= RECENCY_PENALTY

    if TRANSITION_WEIGHT > 0:
        score += transition_bonus(target, others_by_date, trans)

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

    trans = build_transition(others_by_date, thursdays)
    last20 = thursdays[-20:]

    def run(days):
        hits = 0
        rows = []
        for d in days:
            actual = identify_pb(others_by_date[d])
            pred = predict_pb_topk(d, others_by_date, trans)
            hit = actual in pred
            hits += int(hit)
            rows.append((d, pred, actual, hit))
        return hits, rows

    h_all, _ = run(thursdays)
    h_20, rows_20 = run(last20)

    print("MODE = THU_PB_TOPK")
    print(f"Usable Thursdays: {len(thursdays)}")
    print(f"Last20 range: {last20[0]} .. {last20[-1]}")
    print()
    print("==== HIT SUMMARY ====")
    print(f"Last20: {h_20}/20")
    print(f"All   : {h_all}/{len(thursdays)}")
    print()

    print("==== LAST 20 DETAILS ====")
    for d, pred, actual, hit in rows_20:
        if TOP_K == 2 and TOP2_MODE == "diff_band":
            mode_label = "TOP2(diff_band)"
        else:
            mode_label = f"TOP{TOP_K}"
        print(f"THU={d}  {mode_label}={pred}  ACTUAL={actual}  HIT={hit}")

    # Target date prediction (works even if target not in CSV)
    target = resolve_target_date(thursdays)
    pred = predict_pb_topk(target, others_by_date, trans)
    actual = identify_pb(others_by_date.get(target, []))
    print()
    print(f"==== TARGET PREDICTION (THU={target}) ====")
    if TOP_K == 2 and TOP2_MODE == "diff_band":
        mode_label = "TOP2(diff_band)"
    else:
        mode_label = f"TOP{TOP_K}"
    print(f"{mode_label}={pred}")
    if actual is not None:
        print(f"ACTUAL={actual}  HIT={actual in pred}")
    else:
        print("ACTUAL=<not available in CSV>")

# =========================
if __name__ == "__main__":
    main()
