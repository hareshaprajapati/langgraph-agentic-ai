import csv
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

CSV_FILE = 'cross_lotto_data_backup.csv'
TARGET_DATE_STR = "Thu 10-Sep-2026"

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def parse_date(s):
    """'Thu 10-Sep-2026' -> datetime."""
    return datetime.strptime(s[4:], '%d-%b-%Y')


def extract_all_numbers(cell):
    """Extract every integer in a cell (main + supplementary, all brackets)."""
    if not cell:
        return []
    nums = []
    for part in cell.split(']'):
        part = part.replace('[', '').strip()
        if part:
            for tok in part.split(','):
                tok = tok.strip()
                if tok:
                    try:
                        nums.append(int(tok))
                    except ValueError:
                        pass
    return nums


def extract_main_numbers(cell):
    """Extract only the first bracketed list (main numbers)."""
    if not cell:
        return []
    main_part = cell.split(']')[0].replace('[', '').strip()
    return [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]


def extract_powerball(cell):
    """Extract the single number in the last bracket (Powerball)."""
    if not cell:
        return None
    brackets = re.findall(r'\[([^\]]+)\]', cell)
    if len(brackets) < 2:
        return None
    toks = [int(x.strip()) for x in brackets[-1].split(',') if x.strip().isdigit()]
    return toks[0] if toks else None


# ----------------------------------------------------------------------
# BUILD POOLS — EXACT RULE FROM YOUR SCRIPT
# ----------------------------------------------------------------------
def build_pools(window_rows, max_num):
    """
    Rule (verbatim from import csv.txt):
        EH : freq >= 4
        H  : freq == 3
        W  : 1 <= freq <= 2
        C  : freq == 0
    """
    freq = Counter()
    for _, _, all_nums in window_rows:
        for n in all_nums:
            if 1 <= n <= max_num:
                freq[n] += 1

    pools = {'EH': [], 'H': [], 'W': [], 'C': []}
    for n in range(1, max_num + 1):
        c = freq[n]
        if c >= 4:
            pools['EH'].append(n)
        elif c == 3:
            pools['H'].append(n)
        elif c >= 1:
            pools['W'].append(n)
        else:
            pools['C'].append(n)
    return pools, freq


# ----------------------------------------------------------------------
# READ CSV
# ----------------------------------------------------------------------
def load_rows():
    """Return all_rows = [(date_str, dt, day_abbr, all_nums)] and Thursday draws."""
    all_rows = []
    with open(CSV_FILE, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            date_str = row[0].strip()
            try:
                dt = parse_date(date_str)
            except Exception:
                continue
            sfl    = extract_all_numbers(row[1]) if len(row) > 1 else []
            others = extract_all_numbers(row[2]) if len(row) > 2 and row[2] else []
            all_rows.append((date_str, dt, date_str[:3], sfl + others))
    all_rows.sort(key=lambda x: x[1])

    # Thursday Powerball draws (main numbers + powerball)
    thursday_draws = []
    with open(CSV_FILE, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        others_col = next(
            k for k in reader.fieldnames
            if k.strip().lstrip('\ufeff').lower().startswith('others')
        )
        for row in reader:
            ds = (row.get('Date') or '').strip()
            if not ds.startswith('Thu'):
                continue
            cell = row.get(others_col)
            if not cell:
                continue
            mains = extract_main_numbers(cell)
            pb    = extract_powerball(cell)
            if not mains or pb is None:
                continue
            thursday_draws.append((ds, parse_date(ds), mains, pb))
    thursday_draws.sort(key=lambda x: x[1])
    return all_rows, thursday_draws


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    all_rows, thursday_draws = load_rows()
    MAX_NUM = 35

    # ------------------------------------------------------------------
    # Historical: for each Thursday draw, build the 7-day window ending
    # on that date and classify the actual Powerball.
    # ------------------------------------------------------------------
    print("=" * 90)
    print("HISTORICAL: Thursday Powerball → which pool?")
    print("=" * 90)
    print(f"{'Date':<14} {'PB':<4} {'Pool':<5} {'EH':<4} {'H':<4} {'W':<4} {'C':<4} "
          f"{'Prev PB':<8} {'Same?':<6}")
    print("-" * 90)

    pool_hits   = Counter()
    transition  = defaultdict(Counter)
    successor   = defaultdict(Counter)
    prev_pb, prev_pool = None, None

    for ds, dt, mains, pb in thursday_draws:
        # window = [previous Thursday, this Thursday)
        prior = [d for d in thursday_draws if d[1] < dt]
        if not prior:
            prev_pb = pb
            continue
        prev_thu_dt = prior[-1][1]

        # collect numbers from every lottery in [prev_thu_dt, dt)
        window_rows = [(s, d, nums) for s, d, _, nums in all_rows
                       if prev_thu_dt <= d < dt]
        pools, _ = build_pools(window_rows, MAX_NUM)

        pool = '?'
        for name, nums in pools.items():
            if pb in nums:
                pool = name
                break

        pool_hits[pool] += 1
        if prev_pool:
            transition[prev_pool][pool] += 1
        if prev_pb is not None:
            successor[prev_pb][pb] += 1

        same = "YES" if pool == prev_pool else "no"
        sizes = (len(pools['EH']), len(pools['H']),
                 len(pools['W']), len(pools['C']))
        print(f"{ds:<14} {pb:<4} {pool:<5} {sizes[0]:<4} {sizes[1]:<4} "
              f"{sizes[2]:<4} {sizes[3]:<4} {str(prev_pb):<8} {same:<6}")

        prev_pb, prev_pool = pb, pool

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    total = sum(pool_hits.values()) or 1
    for p in ['EH', 'H', 'W', 'C']:
        print(f"  {p:<3}  {pool_hits[p]:>4}  ({pool_hits[p]/total*100:5.1f}%)")

    print("\nTransition matrix (prev pool → next pool):")
    for p in ['EH', 'H', 'W', 'C']:
        if transition[p]:
            row = ", ".join(f"{k}={v}" for k, v in transition[p].most_common())
            print(f"  {p:<3} -> {row}")

    # ------------------------------------------------------------------
    # Predict for TARGET_DATE_STR
    # ------------------------------------------------------------------
    target_dt = parse_date(TARGET_DATE_STR)

    # previous Thursday before target
    prev_thu = max((d for d in thursday_draws if d[1] < target_dt),
                   key=lambda x: x[1])
    prev_ds, prev_thu_dt, _, prev_pb_num = prev_thu

    # window = [prev Thursday, target date)
    window_rows = [(s, d, nums) for s, d, _, nums in all_rows
                   if prev_thu_dt <= d < target_dt]
    pools, freq = build_pools(window_rows, MAX_NUM)

    print("\n" + "=" * 90)
    print(f"PREDICTION for {TARGET_DATE_STR}")
    print("=" * 90)
    print(f"Window: {prev_ds}  →  {TARGET_DATE_STR} (exclusive)")
    print(f"Numbers contributing to window: {sum(len(n) for _, _, n in window_rows)}")

    print("\nPool details (using rule from import csv.txt):")
    print(f"  EH ({len(pools['EH']):>2}): {sorted(pools['EH'])}")
    print(f"  H  ({len(pools['H']):>2}): {sorted(pools['H'])}")
    print(f"  W  ({len(pools['W']):>2}): {sorted(pools['W'])}")
    print(f"  C  ({len(pools['C']):>2}): {sorted(pools['C'])}")

    # ---- Reasoning ----
    # 1. Dominant pool
    dominant = pool_hits.most_common(1)[0][0]

    # 2. Last PB → its pool → most likely next pool
    last_pb_pool = None
    for name, nums in pools.items():
        if prev_pb_num in nums:
            last_pb_pool = name
            break
    # Find what pool the *previous* Thursday's PB was actually in (from history)
    # We recompute it the same way the loop did.
    prev_window_rows = []
    if len(thursday_draws) >= 2:
        prev_prev_dt = sorted([d[1] for d in thursday_draws if d[1] < prev_thu_dt])[-1]
        prev_window_rows = [(s, d, nums) for s, d, _, nums in all_rows
                            if prev_prev_dt <= d < prev_thu_dt]
    prev_pools, _ = build_pools(prev_window_rows, MAX_NUM)
    prev_pool_name = next((name for name, nums in prev_pools.items()
                           if prev_pb_num in nums), None)

    if prev_pool_name and transition[prev_pool_name]:
        next_pool = transition[prev_pool_name].most_common(1)[0][0]
    else:
        next_pool = dominant

    candidates = pools[next_pool]
    succ = [n for n, _ in successor[prev_pb_num].most_common()] \
        if successor[prev_pb_num] else []
    filtered = [n for n in succ if n in candidates] or candidates

    print("\n--- Reasoning ---")
    print(f"  Last Thursday PB           : {prev_pb_num}  (pool: {prev_pool_name})")
    print(f"  Dominant pool historically : {dominant}")
    print(f"  Transition {prev_pool_name} -> {next_pool}")
    print(f"  Successors of {prev_pb_num}         : {succ}")
    print(f"  Candidates in {next_pool} pool    : {candidates}")

    print("\n--- FINAL PREDICTION ---")
    print(f"  Predicted Pool : {next_pool}")
    print(f"  Ranked PB picks: {filtered}")
    if filtered:
        print(f"  Best single    : {filtered[0]}")


if __name__ == '__main__':
    main()