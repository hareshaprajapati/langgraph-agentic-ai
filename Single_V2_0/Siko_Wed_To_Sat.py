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

log_file_path = os.path.join(".", "Siko_Wed_To_Sat.py.py.log")
log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ============================================================
# Siko_Last20_Compare_All.py
# CONSTANTS ONLY (NO ARGS)
# Runs multiple variants on LAST 20 usable Saturdays and compares.
# ============================================================

import csv
import datetime as dt
import re
from collections import defaultdict, Counter

# ============================================================
# 🔒 CONSTANTS
# ============================================================

CSV_PATH = "cross_lotto_data.csv"
TATTS_CSV_PATH = os.path.join("Saturday", "Tattslotto.csv")

# ============================================================
# PRINT LAST BACKTEST TICKETS
# ============================================================
PRINT_LAST_BACKTEST_TICKETS = True
PRINT_LAST_TICKETS_LIMIT = 20   # keep 20 (your N_TICKETS)


LAST_N_USABLE_SATS = 20

N_TICKETS = 20

# Target eval (same format as other scripts)
TARGET_DATE = "2026-1-24"
REAL_DRAW_TARGET = [8, 22, 24, 28, 29, 33]

# Wed->Sat base engine knobs
EXT_K = 4
GAP_K = 2
MIN_GAP_SIZE = 2

# Cross lottery reinforcement base
CROSS_DAYS_DEFAULT = 7

# Sat-only frequency/recency
SAT_LOOKBACK_DRAWS = 12
SAT_POOL_K = 18  # build top-K pool then cover with 20 tickets

# Adaptive window thresholds
ADAPT_TIGHT_RANGE = 14   # if Wed max-min <= 14, use 3-day window else 7-day
ADAPT_DAYS_TIGHT = 3
ADAPT_DAYS_WIDE = 7


# ============================================================
# DATE PARSE (Mon 26-Jan-2026)
# ============================================================

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def parse_date(label: str) -> dt.date:
    label = label.strip()
    _, dmy = label.split()
    dd, mon, yy = dmy.split("-")
    return dt.date(int(yy), MONTHS[mon], int(dd))

def parse_iso_date(s: str) -> dt.date:
    s = s.strip()
    parts = s.split("-")
    if len(parts) != 3:
        raise ValueError(f"Bad TARGET_DATE: {s}")
    yy, mm, dd = (int(parts[0]), int(parts[1]), int(parts[2]))
    return dt.date(yy, mm, dd)

def parse_tatts_date(s: str) -> dt.date:
    s = s.strip()
    dd, mm, yy = s.split("/")
    return dt.date(int(yy), int(mm), int(dd))

def extract_first_list_ints(cell: str):
    if not cell:
        return []
    s = cell.strip()
    m = re.search(r"\[([^\]]+)\]", s)
    if m:
        return [int(x) for x in re.findall(r"\d+", m.group(1))]
    return [int(x) for x in re.findall(r"\d+", s)]

def clamp_1_45(nums):
    return sorted({n for n in nums if 1 <= n <= 45})

def get_first_six(draws):
    for dr in draws:
        if len(dr) == 6:
            return dr
    return None

def hits(a, b):
    return len(set(a) & set(b))


# ============================================================
# LOAD CSV
# ============================================================

sfl_by_date = {}
oth_by_date = defaultdict(list)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        d = parse_date(r["Date"])

        sfl = clamp_1_45(extract_first_list_ints(r.get("Set for Life (incl supp)", "") or ""))
        if sfl:
            sfl_by_date[d] = sfl

        oth_cell = (r.get("Others (incl supp)", "") or "").strip()
        if oth_cell:
            parts = [p.strip() for p in oth_cell.split("|")]
            for p in parts:
                nums = clamp_1_45(extract_first_list_ints(p))
                if nums:
                    oth_by_date[d].append(nums)


# ============================================================
# POOLS
# ============================================================

def neighbours(nums):
    s = set(nums)
    out = []
    for n in nums:
        for d in (-1, 1):
            x = n + d
            if 1 <= x <= 45 and x not in s and x not in out:
                out.append(x)
    return out

class Gap:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.inside = list(range(a + 1, b))
        self.length = len(self.inside)

def gap_pool(wed):
    xs = sorted(set(wed))
    gaps = []
    for a, b in zip(xs, xs[1:]):
        if (b - a - 1) >= MIN_GAP_SIZE:
            gaps.append(Gap(a, b))
    gaps.sort(key=lambda g: (-g.length, g.a, g.b))
    out = []
    for g in gaps[:GAP_K]:
        for n in g.inside:
            if n not in out:
                out.append(n)
    return out

def ext_pool(wed, side):
    xs = sorted(set(wed))
    lo, hi = xs[0], xs[-1]
    if side == "below":
        return list(range(max(1, lo - EXT_K), lo))
    return list(range(hi + 1, min(45, hi + EXT_K) + 1))


# ============================================================
# CROSS LOTTERY SCORING
# ============================================================

def cross_counts(sat_date, cross_days):
    c = Counter()
    for i in range(1, cross_days + 1):
        d = sat_date - dt.timedelta(days=i)
        for n in sfl_by_date.get(d, []):
            c[n] += 1
        for dr in oth_by_date.get(d, []):
            for n in dr:
                c[n] += 1
    return c

def score_num(n, cc):
    k = cc.get(n, 0)
    if k <= 0:
        return 0.0
    return 2.0 + 0.2 * (k - 1)

def ranked(pool, cc, tie=None):
    uniq = list(dict.fromkeys(pool))
    if tie is None:
        return sorted(uniq, key=lambda n: (-score_num(n, cc), n))
    a, b = tie
    return sorted(uniq, key=lambda n: (-score_num(n, cc), min(abs(n-a), abs(n-b)), n))


# ============================================================
# TICKET SHAPES (RECENT_BEST schedule)
# ============================================================

def shapes_recent_best():
    # exactly the schedule you were using:
    # A x8, B x6, C x6  (20 tickets)
    return ["A"] * 8 + ["B"] * 6 + ["C"] * 6


# ============================================================
# WED->SAT ENGINE (baseline + cross variants)
# ============================================================

def build_tickets_wed_to_sat(
    sat_date,
    *,
    use_cross: bool,
    cross_days: int,
    core_primary_only: bool,
):
    wed_date = sat_date - dt.timedelta(days=3)
    wed_main = get_first_six(oth_by_date.get(wed_date, []))
    if not wed_main:
        return None

    core = sorted(wed_main)
    gaps = gap_pool(core)
    neigh = neighbours(core)

    lo, hi = core[0], core[-1]

    cc = cross_counts(sat_date, cross_days) if use_cross else Counter()

    core_r = ranked(core, cc)
    gaps_r = ranked(gaps, cc, tie=(lo, hi))
    neigh_r = ranked(neigh, cc)
    shapes = shapes_recent_best()

    tickets = []
    for i in range(N_TICKETS):
        shape = shapes[i]
        if shape == "A":
            c_ct, g_ct, e_ct = 2, 2, 2
        elif shape == "B":
            c_ct, g_ct, e_ct = 1, 3, 2
        else:
            c_ct, g_ct, e_ct = 2, 3, 1

        side = "below" if (i % 2 == 0) else "above"
        ext_r = ranked(ext_pool(core, side), cc)

        chosen = []

        # CORE picks: rotate for coverage
        for j in range(c_ct):
            chosen.append(core_r[(i + j) % len(core_r)])

        # GAP
        if gaps_r:
            for j in range(g_ct):
                chosen.append(gaps_r[(i * 2 + j) % len(gaps_r)])

        # EXT
        if ext_r:
            for j in range(e_ct):
                chosen.append(ext_r[(i + j) % len(ext_r)])

        # FILL: this is where CORE_PRIMARY_ONLY matters
        if core_primary_only:
            core_primary = core_r[:3]  # top-3 wed
            filler = gaps_r + ext_r + neigh_r + core_primary + core_r
        else:
            filler = gaps_r + ext_r + neigh_r + core_r

        out = []
        for n in chosen + filler:
            if n not in out:
                out.append(n)
            if len(out) == 6:
                break

        # last resort
        if len(out) < 6:
            for n in range(1, 46):
                if n not in out:
                    out.append(n)
                    if len(out) == 6:
                        break

        tickets.append(sorted(out))

    return tickets


# ============================================================
# SAT-ONLY (freq/recency) ENGINE
# ============================================================

def sat_only_pool(sat_date, usable_sats):
    """
    Build a pool from previous SAT_LOOKBACK_DRAWS Saturday winners (actual Saturday lotto 6 nums).
    Score = freq + small recency boost.
    """
    idx = usable_sats.index(sat_date)
    prev = usable_sats[max(0, idx - SAT_LOOKBACK_DRAWS): idx]
    if len(prev) < 3:
        return None

    freq = Counter()
    last_seen = {}
    for j, sd in enumerate(prev):
        nums = get_first_six(oth_by_date.get(sd, []))
        if not nums:
            continue
        for n in nums:
            freq[n] += 1
            last_seen[n] = sd

    def sc(n):
        # freq dominant, recency mild
        f = freq.get(n, 0)
        if f == 0:
            return -999
        gap = (sat_date - last_seen[n]).days if n in last_seen else 999
        rec = 1.0 / (1.0 + gap)
        return f + 0.5 * rec

    cand = sorted(freq.keys(), key=lambda n: (-sc(n), n))
    return cand[:SAT_POOL_K]

def build_tickets_sat_only(sat_date, usable_sats):
    pool = sat_only_pool(sat_date, usable_sats)
    if not pool or len(pool) < 6:
        return None

    # cover pool by sliding windows of 6
    tickets = []
    for i in range(N_TICKETS):
        t = []
        for j in range(6):
            t.append(pool[(i + j) % len(pool)])
        tickets.append(sorted(set(t))[:6])
    return tickets


# ============================================================
# USABLE SATURDAYS (Sat actual + Wed anchor exists)
# ============================================================

all_sats = sorted([d for d in oth_by_date.keys() if d.weekday() == 5])

usable = []
for sat in all_sats:
    sat_actual = get_first_six(oth_by_date.get(sat, []))
    if not sat_actual:
        continue
    wed = sat - dt.timedelta(days=3)
    wed_anchor = get_first_six(oth_by_date.get(wed, []))
    if not wed_anchor:
        continue
    usable.append(sat)

if len(usable) == 0:
    raise SystemExit("No usable Saturdays found (need Sat actual + Wed anchor).")

# Align backtest dates to Tattslotto last-20 (same Saturdays)
tatts_dates = []
try:
    with open(TATTS_CSV_PATH, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d = parse_tatts_date(r["Date"])
            tatts_dates.append(d)
except FileNotFoundError:
    tatts_dates = []

tatts_dates = sorted(set(tatts_dates))
tatts_last20 = tatts_dates[-LAST_N_USABLE_SATS:] if tatts_dates else []

if tatts_last20:
    last20 = tatts_last20
else:
    last20 = usable[-LAST_N_USABLE_SATS:]


# ============================================================
# EVAL HARNESS
# ============================================================

def eval_variant(name, ticket_builder):
    dist = Counter()
    ge3 = 0
    ge4 = 0
    best = 0
    hit4_dates = []

    last_sat = last20[-1] if last20 else None
    last_detail = None

    for sat in last20:
        actual = get_first_six(oth_by_date.get(sat, []))
        tickets = ticket_builder(sat)
        if not tickets or not actual:
            # keep date count aligned with fixed last20
            dist[0] += 1
            continue

        # compute max-hit and best ticket for this sat
        best_ticket = None
        mh = -1
        for t in tickets:
            h = hits(t, actual)
            if h > mh:
                mh = h
                best_ticket = t

        dist[mh] += 1
        best = max(best, mh)
        if mh >= 3:
            ge3 += 1
        if mh >= 4:
            ge4 += 1
            hit4_dates.append(str(sat))

        # capture last backtest saturday tickets
        if last_sat and sat == last_sat:
            last_detail = {
                "sat": sat,
                "actual": actual,
                "tickets": tickets[:PRINT_LAST_TICKETS_LIMIT],
                "max_hit": mh,
                "best_ticket": best_ticket,
            }

    return {
        "name": name,
        "4plus": ge4,
        "3plus": ge3,
        "best": best,
        "dist": dict(dist),
        "hit4_dates": hit4_dates,
        "last_detail": last_detail,
    }


# ============================================================
# RUN ALL VARIANTS (the same rows you showed)
# ============================================================

def adaptive_builder(sat):
    wed = sat - dt.timedelta(days=3)
    wed_main = get_first_six(oth_by_date.get(wed, []))
    if not wed_main:
        return None
    span = max(wed_main) - min(wed_main)
    days = ADAPT_DAYS_TIGHT if span <= ADAPT_TIGHT_RANGE else ADAPT_DAYS_WIDE
    return build_tickets_wed_to_sat(sat, use_cross=True, cross_days=days, core_primary_only=False)

def both_builder(sat):
    wed = sat - dt.timedelta(days=3)
    wed_main = get_first_six(oth_by_date.get(wed, []))
    if not wed_main:
        return None
    span = max(wed_main) - min(wed_main)
    days = ADAPT_DAYS_TIGHT if span <= ADAPT_TIGHT_RANGE else ADAPT_DAYS_WIDE
    return build_tickets_wed_to_sat(sat, use_cross=True, cross_days=days, core_primary_only=True)

variants = [
    ("RECENT_BEST", lambda sat: build_tickets_wed_to_sat(sat, use_cross=False, cross_days=CROSS_DAYS_DEFAULT, core_primary_only=False)),
    ("+ Sat freq/recency", lambda sat: build_tickets_sat_only(sat, usable)),
    ("+ Cross-lottery (7d, full pool)", lambda sat: build_tickets_wed_to_sat(sat, use_cross=True, cross_days=7, core_primary_only=False)),
    ("CORE_PRIMARY only", lambda sat: build_tickets_wed_to_sat(sat, use_cross=True, cross_days=7, core_primary_only=True)),
    ("Adaptive window", adaptive_builder),
    ("BOTH combined", both_builder),
]

results = []
for name, builder in variants:
    res = eval_variant(name, builder)
    res["builder"] = builder
    results.append(res)


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\nFINAL COMPARISON (LAST 20 usable Saturdays)")
print("CSV:", CSV_PATH)
print("usable Saturdays:", len(usable), " | last20 range:", last20[0], "..", last20[-1])
print()

# expected random row (printed text only)
print("Strategy\t4+ hits\t3+ hits\tBest")
print("Random (expected)\t0–1\t4–5\t3")

for r in results:
    print(f"{r['name']}\t{r['4plus']}\t{r['3plus']}\t{r['best']}")

print("\n(Details) hit4_dates per strategy:")
for r in results:
    print(f"- {r['name']}: {r['hit4_dates']}")
if PRINT_LAST_BACKTEST_TICKETS:
    print("\n==== LAST BACKTEST SATURDAY: TICKETS PER STRATEGY ====")
    for r in results:
        ld = r.get("last_detail")
        if not ld:
            print(f"\n--- {r['name']} ---")
            print("No tickets/actual for last Saturday (missing anchor or missing draw).")
            continue

        sat = ld["sat"]
        actual = ld["actual"]
        tickets = ld["tickets"]
        max_hit = ld["max_hit"]
        best_ticket = ld["best_ticket"]

        print(f"\n--- {r['name']} ---")
        print(f"SAT={sat}  ACTUAL={sorted(actual)}  MAX_HIT={max_hit}  BEST={sorted(best_ticket)}")

        for i, t in enumerate(tickets, 1):
            h = hits(t, actual)
            print(f"Ticket #{i:02d}: {sorted(t)}  HIT={h}")

# ============================================================
# TARGET DATE EVAL (using provided REAL_DRAW_TARGET)
# ============================================================

print("\n==== TARGET DATE EVAL ====")
target_date = parse_iso_date(TARGET_DATE)
target_actual = sorted(REAL_DRAW_TARGET)
print(f"TARGET_DATE={TARGET_DATE}  REAL_DRAW_TARGET={target_actual}")

for r in results:
    name = r["name"]
    builder = r.get("builder")
    print(f"\n--- {name} ---")
    if builder is None:
        print("No builder available.")
        continue
    tickets = builder(target_date)
    if not tickets:
        print("No tickets (missing anchor or missing draw).")
        continue
    best_hit = 0
    ge3 = 0
    for i, t in enumerate(tickets, 1):
        h = hits(t, target_actual)
        if h >= 3:
            ge3 += 1
        best_hit = max(best_hit, h)
        print(f"Ticket #{i:02d}: {sorted(t)}  HIT={h}")
    print(f"TARGET summary: ge3={ge3}  best_hit={best_hit}")
