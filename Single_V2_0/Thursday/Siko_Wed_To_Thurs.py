import sys
import os
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
from itertools import combinations
import random
from collections import Counter
import pandas as pd

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except OSError:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except OSError:
                pass

log_file_path = os.path.join(".", "Siko_Wed_To_Thurs.py.log")
log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)



import pandas as pd
import re
import random
from datetime import datetime, timedelta
from collections import Counter

# =========================
# CONSTS (EDIT ONLY HERE)
# =========================
CSV_PATH = "../cross_lotto_data.csv"   # keep in same folder as this script (or change path)
RANDOM_SEED_BASE = 0               # base seed; per-date seed = base + YYYYMMDD
TICKETS_PER_DRAW = 20

# BEST CONFIG (LOCKED)
OFFSETS_DAYS = [7, 1]              # use prev Thu (t-7) and Wed (t-1)
EXT_K = 2                          # expand pool by ±2
NEIGHBOR_BONUS = 2                 # add score if neighbor exists in pool
WEIGHT_SCHEME = "flat"             # flat weights work best here

PRINT_DATE_BY_DATE = True
PRINT_TICKETS_FOR_LAST_THU = True  # prints the 20 tickets for the most recent Thu in CSV


# =========================
# PARSE + IDENTIFY
# =========================
def parse_date(label: str):
    # expects "Thu 08-Jan-2026"
    return datetime.strptime(str(label).strip(), "%a %d-%b-%Y").date()

def extract_draws(cell: str):
    # cell example: "[main], [supp] | [main], [supp]"
    if not isinstance(cell, str) or not cell.strip():
        return []
    parts = [p.strip() for p in cell.split("|")]
    out = []
    for p in parts:
        lists = re.findall(r"\[([^\]]*)\]", p)
        main = [int(x) for x in re.findall(r"\d+", lists[0])] if len(lists) >= 1 else []
        supp = [int(x) for x in re.findall(r"\d+", lists[1])] if len(lists) >= 2 else []
        if main or supp:
            out.append({"main": main, "supp": supp})
    return out

def identify_powerball_main(draws):
    # In CSV: Thursday "Others" includes Powerball main (7 nums) + PB (1 num).
    # Choose draw with 7 main numbers and 1 supp number, clamp to 1..35.
    for dr in draws:
        main = dr.get("main", [])
        supp = dr.get("supp", [])
        if len(main) == 7 and len(supp) == 1:
            main = clamp_1_35(main)
            if len(main) == 7:
                return main
    return None

def clamp_1_35(nums):
    return [x for x in nums if 1 <= x <= 35]


# =========================
# BUILD SOURCES
# =========================
def build_others_by_date(df):
    others_by_date = {}
    for _, r in df.iterrows():
        d = parse_date(r["Date"])
        others_by_date[d] = extract_draws(r.get("Others (incl supp)", ""))
    return others_by_date

def build_nums_by_date_all(others_by_date):
    # "all" = take up to first 7 nums of every draw that day, clamp to <=35
    nums_by_date = {}
    for d, draws in others_by_date.items():
        alln = []
        for dr in draws:
            alln.extend(clamp_1_35(dr.get("main", [])[:7]))
        nums_by_date[d] = alln
    return nums_by_date


# =========================
# STRATEGY
# =========================
def per_date_seed(d):
    return RANDOM_SEED_BASE + int(d.strftime("%Y%m%d"))

def make_weights(offsets):
    if WEIGHT_SCHEME == "flat":
        return {off: 1 for off in offsets}
    # keep simple; can extend later
    return {off: 1 for off in offsets}

def build_pool(target, nums_by_date, others_by_date):
    base = set()
    for off in OFFSETS_DAYS:
        prev = target - timedelta(days=off)
        base.update(nums_by_date.get(prev, []))

    # Ensure prev Thu PB main is included explicitly (strong signal)
    prev_thu = target - timedelta(days=7)
    pm = identify_powerball_main(others_by_date.get(prev_thu, []))
    if pm:
        base.update(clamp_1_35(pm))

    pool = set()
    for n in base:
        for k in range(-EXT_K, EXT_K + 1):
            x = n + k
            if 1 <= x <= 35:
                pool.add(x)
    return pool

def score_pool(target, pool, nums_by_date):
    weights = make_weights(OFFSETS_DAYS)
    score = Counter()
    for n in pool:
        s = 0
        for off in OFFSETS_DAYS:
            prev = target - timedelta(days=off)
            if n in nums_by_date.get(prev, []):
                s += weights.get(off, 1)
        if NEIGHBOR_BONUS:
            if (n - 1 in pool) or (n + 1 in pool):
                s += NEIGHBOR_BONUS
        score[n] = s
    return score

def pick_tickets(score, pool, seed):
    random.seed(seed)
    items = list(pool)
    if len(items) < 7:
        return []

    # base weights
    w = [max(0.05, score[i] + 0.5) for i in items]
    sorted_items = sorted(items, key=lambda x: (-score[x], x))
    top_region = set(sorted_items[:min(18, len(sorted_items))])

    tickets = []
    for _ in range(TICKETS_PER_DRAW):
        chosen = set()

        # seed with a run (adjacent numbers) often
        runs = [n for n in top_region if (n + 1 in pool)]
        if runs and random.random() < 0.85:
            a = random.choice(runs)
            chosen.update([a, a + 1])
            if a - 1 in pool and random.random() < 0.35:
                chosen.add(a - 1)
            if a + 2 in pool and random.random() < 0.35:
                chosen.add(a + 2)

        while len(chosen) < 7:
            idx = random.choices(range(len(items)), weights=w, k=1)[0]
            chosen.add(items[idx])

        tickets.append(sorted(chosen))

    # unique tickets
    uniq, seen = [], set()
    for t in tickets:
        tup = tuple(t)
        if tup not in seen:
            seen.add(tup)
            uniq.append(t)
    return uniq


# =========================
# BACKTEST
# =========================
def eval_dates(thursdays, others_by_date, nums_by_date):
    dist = Counter()
    ge3 = ge4 = ge5 = 0
    best = 0
    hit5_dates = []
    rows = []

    for d in thursdays:
        actual = identify_powerball_main(others_by_date[d])
        if not actual:
            continue
        actual_set = set(actual)

        pool = build_pool(d, nums_by_date, others_by_date)
        if len(pool) < 7:
            dist[0] += 1
            rows.append((d, 0, None, actual))
            continue

        score = score_pool(d, pool, nums_by_date)
        tickets = pick_tickets(score, pool, seed=per_date_seed(d))

        mh = 0
        bestt = None
        for t in tickets:
            h = len(set(t) & actual_set)
            if h > mh:
                mh = h
                bestt = t

        dist[mh] += 1
        best = max(best, mh)
        if mh >= 3: ge3 += 1
        if mh >= 4: ge4 += 1
        if mh >= 5:
            ge5 += 1
            hit5_dates.append(str(d))

        rows.append((d, mh, bestt, actual))

    return {
        "draws": len(thursdays),
        "3+": ge3,
        "4+": ge4,
        "5+": ge5,
        "best": best,
        "dist": dict(dist),
        "hit5_dates": hit5_dates,
        "rows": rows,
    }


def main():
    df = pd.read_csv(CSV_PATH)
    others_by_date = build_others_by_date(df)
    nums_by_date = build_nums_by_date_all(others_by_date)

    thursdays = sorted(d for d in others_by_date if d.weekday() == 3 and identify_powerball_main(others_by_date[d]))
    if not thursdays:
        print("No usable Thursdays found (need Powerball main 7 in Others).")
        return

    last20 = thursdays[-20:]

    print("MODE=THU_POWERBALL_MAIN_BEST")
    print(f"CFG: OFFSETS={OFFSETS_DAYS}  EXT_K=±{EXT_K}  NEIGHBOR_BONUS={NEIGHBOR_BONUS}  TICKETS={TICKETS_PER_DRAW}")
    print(f"Usable Thursdays: {len(thursdays)}  | range: {thursdays[0]} .. {thursdays[-1]}")
    print(f"Last20 range: {last20[0]} .. {last20[-1]}")
    print()

    res20 = eval_dates(last20, others_by_date, nums_by_date)
    resall = eval_dates(thursdays, others_by_date, nums_by_date)

    print("==== BACKTEST (LAST 20 usable Thursdays) ====")
    print({k: res20[k] for k in ["draws", "5+", "4+", "3+", "best", "dist", "hit5_dates"]})
    print()

    print("==== BACKTEST (ALL usable Thursdays) ====")
    print({k: resall[k] for k in ["draws", "5+", "4+", "3+", "best", "dist"]})
    print()

    if PRINT_DATE_BY_DATE:
        print("==== DATE-BY-DATE (LAST 20) ====")
        for d, mh, bestt, actual in res20["rows"]:
            hit = sorted(set(bestt or []) & set(actual))
            print(f"THU={d}  MAX_HIT={mh}  BEST={bestt}  HIT={hit}  ACTUAL={actual}")
        print()

    if PRINT_TICKETS_FOR_LAST_THU:
        d = thursdays[-1]
        actual = identify_powerball_main(others_by_date[d])
        pool = build_pool(d, nums_by_date, others_by_date)
        score = score_pool(d, pool, nums_by_date)
        tickets = pick_tickets(score, pool, seed=per_date_seed(d))
        print(f"==== TICKETS (Most recent Thu={d}) ====")
        print(f"ACTUAL(main)={actual}")
        for i, t in enumerate(tickets[:TICKETS_PER_DRAW], 1):
            print(f"Ticket #{i:02d}: {t}")


if __name__ == "__main__":
    main()
