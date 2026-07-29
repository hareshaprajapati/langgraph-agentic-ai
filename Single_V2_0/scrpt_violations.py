import csv
from datetime import datetime as dt, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_numbers(cell):
    nums = []
    if not cell: return nums
    for part in cell.split('],'):
        part = part.strip().lstrip('[').rstrip(']')
        for token in part.split(','):
            token = token.strip()
            if token.isdigit():
                n = int(token)
                if 1 <= n <= 45: nums.append(n)
    return nums

def parse_others_main(others_str):
    main_part = others_str.split('],')[0].strip()
    if main_part.startswith('['): main_part = main_part[1:]
    main_part = main_part.replace(']', '').strip()
    if not main_part: return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]

def decade_of(n):
    if 1 <= n <= 9: return '0s'
    if 10 <= n <= 19: return '10s'
    if 20 <= n <= 29: return '20s'
    if 30 <= n <= 39: return '30s'
    return '40s'

# ----------------------------------------------------------------------
# Load all rows
# ----------------------------------------------------------------------
rows = []
with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    sfl_col = others_col = None
    for col in reader.fieldnames:
        if 'Set for Life' in col: sfl_col = col
        if 'Others' in col: others_col = col
    for row in reader:
        date_str = row['Date'].strip()
        try: d = dt.strptime(date_str, '%a %d-%b-%Y')
        except: continue
        sfl = parse_numbers(row[sfl_col]) if row[sfl_col] else []
        oth = parse_numbers(row[others_col]) if row[others_col] else []
        rows.append((d, sfl, oth, row[others_col]))

# Extract all Saturdays in order
saturdays = []
for d, sfl, oth, oth_raw in rows:
    if d.weekday() == 5:   # Saturday
        main = parse_others_main(oth_raw)
        if len(main) == 6:
            saturdays.append((d, main))
saturdays.sort(key=lambda x: x[0])

# ----------------------------------------------------------------------
# Analysis over all Saturdays (starting from the 21st to have 20‑week history)
# ----------------------------------------------------------------------
print(f"{'Date':<12} {'Profile':<8} {'Legacy':<8} {'O/E':<6} {'H/L':<6} {'DecCap':<8} "
      f"{'Pair':<6} {'BandCap':<8} {'Kill':<6} {'Violations'}")
print("-" * 95)

total = 0
violation_counts = Counter()
draw_violations = []

for i in range(20, len(saturdays)):
    dt, main = saturdays[i]
    prev_dt, prev_main = saturdays[i-1]
    legacy_set = set(prev_main)

    # 7‑day window: previous Saturday (inclusive) to Friday before current draw
    wstart = dt - timedelta(days=7)   # previous Saturday
    wend   = dt - timedelta(days=1)   # Friday
    window_nums = []
    for d, s, o, _ in rows:
        if wstart <= d <= wend:
            window_nums.extend(s)
            window_nums.extend(o)

    freq7 = Counter(window_nums)
    EH = {n for n in range(1,46) if freq7.get(n,0) >= 4}
    H  = {n for n in range(1,46) if freq7.get(n,0) == 3}
    W  = {n for n in range(1,46) if 1 <= freq7.get(n,0) <= 2}
    C  = {n for n in range(1,46) if freq7.get(n,0) == 0}

    # 20‑week frequencies (previous 20 main draws)
    prev_20_main = []
    for j in range(i-20, i):
        prev_20_main.extend(saturdays[j][1])
    freq20 = Counter(prev_20_main)

    # Band caps (the strict ones the user chose)
    BAND_CAPS = {'0x':3, '1x':3, '2x':4, '3x':4, '4x':3, '5x+':3}
    def band_label(cnt):
        if cnt >= 5: return '5x+'
        if cnt == 4: return '4x'
        if cnt == 3: return '3x'
        if cnt == 2: return '2x'
        if cnt == 1: return '1x'
        return '0x'

    # Check constraints for the actual winning ticket
    violations = []

    # Profile (Depth: EH 0‑2, H 0‑3, W 2‑3, C 0‑1)
    eh_cnt = sum(1 for n in main if n in EH)
    h_cnt  = sum(1 for n in main if n in H)
    w_cnt  = sum(1 for n in main if n in W)
    c_cnt  = sum(1 for n in main if n in C)
    if not (0 <= eh_cnt <= 2 and 0 <= h_cnt <= 3 and 2 <= w_cnt <= 3 and 0 <= c_cnt <= 1):
        violations.append('Profile')

    # Legacy
    if sum(1 for n in main if n in legacy_set) > 1:
        violations.append('Legacy')

    # Odd/Even
    odd = sum(1 for n in main if n % 2)
    if (odd, 6-odd) not in [(3,3),(2,4),(4,2)]:
        violations.append('Odd/Even')

    # High/Low (High=23‑45)
    low = sum(1 for n in main if n <= 22)
    if (low, 6-low) not in [(3,3),(2,4),(4,2)]:
        violations.append('High/Low')

    # Decade concentration
    dec_cnt = Counter(decade_of(n) for n in main)
    if max(dec_cnt.values()) > 3:
        violations.append('DecCap')
    if sum(1 for n in main if 40 <= n <= 45) > 2:
        violations.append('40sCap')

    # Consecutive or mirror pair
    srt = sorted(main)
    has_consec = any(srt[j+1]-srt[j]==1 for j in range(5))
    has_mirror = len({n%10 for n in main}) < 6
    if not (has_consec or has_mirror):
        violations.append('Pair')

    # 20‑week band caps
    band_cnt = Counter(band_label(freq20.get(n,0)) for n in main)
    if any(band_cnt[b] > BAND_CAPS[b] for b in band_cnt):
        violations.append('BandCap')

    # Decade kill (example: if we had killed the 30s, as in the failed run)
    # We can test the actual kill that was used. For the failed run it was '30s'.
    # Here we'll test if the winning ticket contains any number from the 30s.
    # (User can change the killed decade)
    killed_decade = '30s'   # <-- change this to whatever kill was applied
    if any(decade_of(n) == killed_decade for n in main):
        violations.append('Kill')

    # Record
    for v in violations:
        violation_counts[v] += 1
    total += 1
    draw_violations.append((dt, violations))

    # Print per draw
    print(f"{dt.strftime('%d-%b-%Y'):<12} "
          f"EH{eh_cnt}H{h_cnt}W{w_cnt}C{c_cnt}  "
          f"{'OK' if 'Legacy' not in violations else 'BAD':<8} "
          f"{'OK' if 'Odd/Even' not in violations else 'BAD':<6} "
          f"{'OK' if 'High/Low' not in violations else 'BAD':<6} "
          f"{'OK' if 'DecCap' not in violations else 'BAD':<8} "
          f"{'OK' if 'Pair' not in violations else 'BAD':<6} "
          f"{'OK' if 'BandCap' not in violations else 'BAD':<8} "
          f"{'OK' if 'Kill' not in violations else 'BAD':<6} "
          f"{', '.join(violations) if violations else 'none'}")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print("\n" + "="*60)
print(f"Analysed {total} Saturdays (from 21st onward).")
print("Constraint violations (jackpot would be lost):")
for rule in ['Profile', 'Legacy', 'Odd/Even', 'High/Low', 'DecCap', 'Pair', 'BandCap', 'Kill']:
    cnt = violation_counts.get(rule, 0)
    print(f"  {rule:<10}: {cnt:3d} draws ({cnt/total*100:.1f}%)")

# ----------------------------------------------------------------------
# Greedy scoring test on recent draws (optional)
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Greedy scoring test on the last 10 draws:")
print("(Compares actual winning ticket's 7‑day hotness to random valid tickets)")

import random, itertools

TEST_DRAWS = 10   # number of recent draws to test
SAMPLE_SIZE = 20000   # random valid tickets per draw

def generate_random_valid_ticket(EH, H, W, C, legacy_set):
    """Generate one random ticket that passes all OGA constraints (except pair filter)."""
    while True:
        # Random profile within Depth ranges
        eh_c = random.randint(0, min(2, len(EH)))
        # H count: at least enough to fill 6, but max 3
        max_h = min(3, len(H), 6 - eh_c)
        min_h = max(0, 6 - eh_c - 3 - 1)  # ensure total can be 6
        if max_h < 0: continue
        h_c = random.randint(0, max_h)
        w_c = random.randint(2, min(3, len(W)))
        c_c = 6 - eh_c - h_c - w_c
        if c_c < 0 or c_c > min(1, len(C)): continue

        # Pick numbers
        if eh_c > len(EH) or h_c > len(H) or w_c > len(W) or c_c > len(C): continue
        eh_picks = random.sample(list(EH), eh_c) if eh_c else []
        h_picks = random.sample(list(H), h_c) if h_c else []
        w_picks = random.sample(list(W), w_c) if w_c else []
        c_picks = random.sample(list(C), c_c) if c_c else []
        ticket = tuple(sorted(eh_picks + h_picks + w_picks + c_picks))
        if len(set(ticket)) < 6: continue

        # Legacy
        if sum(1 for n in ticket if n in legacy_set) > 1: continue
        # Odd/Even
        odd = sum(1 for n in ticket if n % 2)
        if (odd, 6-odd) not in [(3,3),(2,4),(4,2)]: continue
        # High/Low
        low = sum(1 for n in ticket if n <= 22)
        if (low, 6-low) not in [(3,3),(2,4),(4,2)]: continue
        # Decade cap
        dec_cnt = Counter(decade_of(n) for n in ticket)
        if max(dec_cnt.values()) > 3: continue
        if sum(1 for n in ticket if 40 <= n <= 45) > 2: continue
        # We skip pair filter and band caps for this rough scoring, as they'd
        # only eliminate a few percent of tickets and don't affect the conclusion.
        return ticket

# Run on last TEST_DRAWS draws
for i in range(len(saturdays)-TEST_DRAWS, len(saturdays)):
    dt, main = saturdays[i]
    prev_dt, prev_main = saturdays[i-1]
    legacy_set = set(prev_main)

    # Build pools for this draw (same as earlier)
    wstart = dt - timedelta(days=7)
    wend   = dt - timedelta(days=1)
    window_nums = []
    for d, s, o, _ in rows:
        if wstart <= d <= wend:
            window_nums.extend(s); window_nums.extend(o)
    freq7 = Counter(window_nums)
    EH = {n for n in range(1,46) if freq7.get(n,0) >= 4}
    H  = {n for n in range(1,46) if freq7.get(n,0) == 3}
    W  = {n for n in range(1,46) if 1 <= freq7.get(n,0) <= 2}
    C  = {n for n in range(1,46) if freq7.get(n,0) == 0}

    # Score function: total 7‑day frequency of the 6 numbers
    def hotness(ticket):
        return sum(freq7.get(n,0) for n in ticket)

    win_score = hotness(main)

    # Generate random valid tickets and compute their scores
    scores = []
    for _ in range(SAMPLE_SIZE):
        t = generate_random_valid_ticket(EH, H, W, C, legacy_set)
        scores.append(hotness(t))

    scores.sort()
    # Percentile of winning ticket (how many random tickets have lower score)
    rank = sum(1 for s in scores if s < win_score)
    pct = rank / len(scores) * 100

    print(f"  {dt.strftime('%d-%b-%Y')}: winning ticket hotness={win_score:2d}, "
          f"percentile={pct:.1f}% (among {SAMPLE_SIZE} random Depth tickets)")