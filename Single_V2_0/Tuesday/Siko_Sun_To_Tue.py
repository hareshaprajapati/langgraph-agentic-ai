import sys
import os
from datetime import datetime

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

log_file_path = os.path.join(
    ".",
    "Siko_Sun_To_Tue.py.log"   # single growing log file
)

log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)


import csv, datetime as dt, re, random
from collections import defaultdict

# =========================
# CONSTANTS (EDIT ONLY HERE)
# =========================
CSV_PATH = "cross_lotto_data.csv"

TARGET_DATE = dt.date(2026, 1, 27)   # <-- change as needed (Tuesday)
N_TICKETS = 20

SUNDAY_ANCHOR_OVERRIDE = [1, 9, 17, 18, 23, 40, 42]  # set [] to force CSV lookup

BALL_MAX = 47
BALL_COUNT = 7

# Pools
EXT_K = 5            # extension range from anchor min/max
GAP_TOP_K = 2        # use biggest K gaps
USE_NEIGH = True
NEIGH_STEPS = (1,)   # only ±1

# Tuesday target pick / Sunday anchor pick (7-number draws)
TUE_TARGET_PICK = "LAST_7"   # "FIRST_7" or "LAST_7"
SUN_ANCHOR_PICK = "FIRST_7"  # "FIRST_7" or "LAST_7"

# Ticket shapes (core, gap, ext, neigh) must sum to 7
SHAPES = (
    (3, 2, 2, 0),
    (2, 3, 2, 0),
    (3, 3, 1, 0),
    (2, 2, 2, 1),
)

# Quality constraints (to avoid ugly sequential tickets)
MAX_RUN_LEN = 3
MAX_ADJ_PAIRS = 2
MIN_SPAN = 20
LOW_CUTOFF = 24
LOW_MIN = 2
LOW_MAX = 5
BAND_CAP = 5  # disallow >=5 in same band
BANDS = ((1,10),(11,20),(21,30),(31,40),(41,47))

MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

DO_BACKTEST = True
BACKTEST_LAST_N = 20   # set None to run all usable Tuesdays


def parse_date(label: str) -> dt.date:
    label = label.strip()
    parts = label.split()
    if len(parts) < 2:
        raise ValueError(f"Bad date label: {label!r}")
    dd, mon, yy = parts[1].split("-")
    return dt.date(int(yy), MONTHS[mon], int(dd))

def clamp(nums):
    return sorted({n for n in nums if 1 <= n <= BALL_MAX})

def extract_first_bracket_list(s: str):
    if not s:
        return []
    m = re.search(r"\[([^\]]+)\]", s)
    if not m:
        return []
    return [int(x) for x in re.findall(r"\d+", m.group(1))]

def parse_others_cell(cell: str):
    if not cell:
        return []
    parts = [p.strip() for p in cell.split("|")]
    draws = []
    for p in parts:
        main = clamp(extract_first_bracket_list(p))
        if main:
            draws.append(main)
    return draws

def pick_7(draws, mode: str):
    sevens = [d for d in draws if len(d) == BALL_COUNT and max(d) <= BALL_MAX]
    if not sevens:
        return None
    return sevens[0] if mode == "FIRST_7" else sevens[-1]

def max_run_length(nums):
    nums = sorted(nums)
    best = cur = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best

def adj_pair_count(nums):
    nums = sorted(nums)
    return sum(1 for i in range(1, len(nums)) if nums[i] == nums[i-1] + 1)

def is_good_ticket(nums):
    nums = sorted(nums)
    if len(nums) != BALL_COUNT:
        return False
    if max_run_length(nums) > MAX_RUN_LEN:
        return False
    if adj_pair_count(nums) > MAX_ADJ_PAIRS:
        return False
    if (nums[-1] - nums[0]) < MIN_SPAN:
        return False
    low = sum(1 for n in nums if n <= LOW_CUTOFF)
    if not (LOW_MIN <= low <= LOW_MAX):
        return False
    for lo, hi in BANDS:
        if sum(1 for n in nums if lo <= n <= hi) >= BAND_CAP:
            return False
    return True

def gap_representatives(anchor):
    xs = sorted(set(anchor))
    gaps = []
    for a, b in zip(xs, xs[1:]):
        inside = b - a - 1
        if inside >= 2:
            gaps.append((inside, a, b))
    gaps.sort(reverse=True)  # biggest first

    reps = []
    for _, a, b in gaps[:GAP_TOP_K]:
        cand = [a + 1, (a + b) // 2, b - 1]  # left/mid/right
        for n in cand:
            if 1 <= n <= BALL_MAX and n not in anchor and n not in reps:
                reps.append(n)
    return sorted(reps)

def neighbour_pool(anchor):
    if not USE_NEIGH:
        return []
    s = set(anchor)
    out = []
    for n in anchor:
        for step in NEIGH_STEPS:
            for d in (-step, step):
                x = n + d
                if 1 <= x <= BALL_MAX and x not in s and x not in out:
                    out.append(x)
    return sorted(out)

def ext_pool(anchor, side):
    xs = sorted(set(anchor))
    lo, hi = xs[0], xs[-1]
    if side == "below":
        return list(range(max(1, lo - EXT_K), lo))
    return list(range(hi + 1, min(BALL_MAX, hi + EXT_K) + 1))

def build_ticket(anchor, rng, shape, side):
    core = list(anchor)
    gaps = gap_representatives(anchor)
    neigh = neighbour_pool(anchor)
    ext = ext_pool(anchor, side)

    c_ct, g_ct, e_ct, n_ct = shape
    chosen = set()

    rng.shuffle(core)
    for n in core:
        if sum(1 for x in chosen if x in anchor) >= c_ct:
            break
        chosen.add(n)

    rng.shuffle(gaps)
    for n in gaps:
        if sum(1 for x in chosen if x in gaps) >= g_ct:
            break
        chosen.add(n)

    rng.shuffle(ext)
    for n in ext:
        if sum(1 for x in chosen if x in ext) >= e_ct:
            break
        chosen.add(n)

    rng.shuffle(neigh)
    for n in neigh:
        if sum(1 for x in chosen if x in neigh) >= n_ct:
            break
        chosen.add(n)

    # fill to 7: gaps -> core -> ext -> neigh
    filler = []
    for lst in (gaps, core, ext, neigh):
        for n in lst:
            if n not in filler:
                filler.append(n)
    rng.shuffle(filler)
    for n in filler:
        if len(chosen) >= BALL_COUNT:
            break
        chosen.add(n)

    # last resort fill
    if len(chosen) < BALL_COUNT:
        for n in range(1, BALL_MAX + 1):
            if n not in chosen:
                chosen.add(n)
                if len(chosen) == BALL_COUNT:
                    break

    return sorted(chosen)

def generate_tickets(anchor, seed):
    rng = random.Random(seed)
    tickets = []
    attempts = 0
    while len(tickets) < N_TICKETS and attempts < 5000:
        attempts += 1
        i = len(tickets)
        shape = SHAPES[i % len(SHAPES)]
        side = "below" if (i % 2 == 0) else "above"
        t = build_ticket(anchor, rng, shape, side)
        if not is_good_ticket(t):
            continue
        if t in tickets:
            continue
        tickets.append(t)
    return tickets, attempts

def hit_count(ticket, actual):
    return len(set(ticket) & set(actual))

def backtest(oth_by_date):
    # Find all usable Tuesdays where:
    # - Tuesday has a 7-number draw (Oz) in Others
    # - Sunday (2 days before) has a 7-number anchor in Others
    tuesdays = sorted([d for d in oth_by_date.keys() if d.weekday() == 1])  # Tue=1

    usable = []
    for tue in tuesdays:
        target = pick_7(oth_by_date.get(tue, []), TUE_TARGET_PICK)
        sun = tue - dt.timedelta(days=2)
        anchor = pick_7(oth_by_date.get(sun, []), SUN_ANCHOR_PICK)
        if target and anchor:
            usable.append(tue)

    if not usable:
        print("No usable Tuesdays found for backtest (need Tue target 7 + Sun anchor 7).")
        return

    if BACKTEST_LAST_N:
        usable = usable[-BACKTEST_LAST_N:]

    dist = defaultdict(int)
    ge3 = ge4 = ge5 = 0
    best_seen = 0
    hit4_dates = []
    hit5_dates = []

    print(f"\n==== BACKTEST ({len(usable)} usable Tuesdays) ====")
    print(f"range: {usable[0]} .. {usable[-1]}")

    for tue in usable:
        target = pick_7(oth_by_date[tue], TUE_TARGET_PICK)
        sun = tue - dt.timedelta(days=2)
        anchor = pick_7(oth_by_date[sun], SUN_ANCHOR_PICK)

        seed = int(tue.strftime("%Y%m%d"))
        tickets, _ = generate_tickets(anchor, seed)
        mh = max(hit_count(t, target) for t in tickets) if tickets else 0

        dist[mh] += 1
        best_seen = max(best_seen, mh)
        if mh >= 3: ge3 += 1
        if mh >= 4:
            ge4 += 1
            hit4_dates.append(str(tue))
        if mh >= 5:
            ge5 += 1
            hit5_dates.append(str(tue))

        # optional per-date line:
        # print(f"TUE={tue}  SUN={sun}  MAX_HIT={mh}  target={target}")

    print({"draws": len(usable), "3plus": ge3, "4plus": ge4, "5plus": ge5, "best": best_seen, "dist": dict(dist)})
    print("hit4_dates:", hit4_dates)
    if hit5_dates:
        print("hit5_dates:", hit5_dates)


def main():
    oth_by_date = defaultdict(list)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d = parse_date(r["Date"])
            draws = parse_others_cell((r.get("Others (incl supp)", "") or "").strip())
            if draws:
                oth_by_date[d] = draws
    if DO_BACKTEST:
        backtest(oth_by_date)

    # Anchor = Sunday (2 days before)
    sun = TARGET_DATE - dt.timedelta(days=2)
    sun_anchor = None
    if SUNDAY_ANCHOR_OVERRIDE:
        sun_anchor = sorted(SUNDAY_ANCHOR_OVERRIDE)
    else:
        sun_anchor = pick_7(oth_by_date.get(sun, []), SUN_ANCHOR_PICK)
    if not sun_anchor:
        raise SystemExit(f"No 7-number Sunday anchor found for {sun} (and no override provided).")

    seed = int(TARGET_DATE.strftime("%Y%m%d"))
    tickets, attempts = generate_tickets(sun_anchor, seed)

    print("OZ LOTTO TUE (Others) — FIXED GENERATOR")
    print(f"CSV={CSV_PATH}")
    print(f"TARGET={TARGET_DATE} | SUNDAY_ANCHOR={sun} {sun_anchor}")
    print(f"Generated {len(tickets)} tickets (attempts={attempts})")
    print(f"Gap reps={gap_representatives(sun_anchor)}")
    print(f"Ext-above={ext_pool(sun_anchor,'above')}")
    print(f"Neigh={neighbour_pool(sun_anchor)}")
    print("\n=== TICKETS ===")
    for i, t in enumerate(tickets, 1):
        print(f"Ticket #{i:02d}: {t}")

if __name__ == "__main__":
    main()
