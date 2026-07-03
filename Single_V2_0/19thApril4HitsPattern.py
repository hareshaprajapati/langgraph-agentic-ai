import csv
from collections import Counter
from datetime import datetime

CSV_FILE = "cross_lotto_data_backup.csv"
OUTPUT_LAST_N = 60

def parse_date(s):
    return datetime.strptime(s[4:], '%d-%b-%Y')

def extract_numbers(cell):
    nums = []
    for part in cell.split(']'):
        part = part.replace('[', '').strip()
        if part:
            for token in part.split(','):
                token = token.strip()
                if token:
                    n = int(token)
                    if 1 <= n <= 45:
                        nums.append(n)
    return nums

def extract_main6(others_cell):
    """Return the first 6 numbers (main) from the Others column."""
    main_part = others_cell.split(']')[0].replace('[', '').strip()
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

# Read all data
all_rows = []   # (date_str, dt, is_saturday, all_numbers, raw_others_cell)
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 3:
            continue
        date_str = row[0].strip()
        try:
            dt = parse_date(date_str)
        except:
            continue
        sfl_nums = extract_numbers(row[1])
        others_nums = extract_numbers(row[2])
        all_nums = sfl_nums + others_nums
        is_sat = date_str.startswith('Sat ')
        all_rows.append((date_str, dt, is_sat, all_nums, row[2] if is_sat else None))

all_rows.sort(key=lambda x: x[1])

# Saturdays with main6
saturdays = []
for date_str, dt, is_sat, _, others_cell in all_rows:
    if is_sat:
        main6 = extract_main6(others_cell)
        saturdays.append((date_str, dt, main6))

# Analysis
results = []
for i in range(1, len(saturdays)):
    target_date_str, target_dt, target_main = saturdays[i]
    prev_sat_date_str, prev_sat_dt, prev_main = saturdays[i-1]   # legacy set

    # Window: previous Saturday (inclusive) to Friday before target
    window_nums = []
    for date_str, dt, is_sat, nums, _ in all_rows:
        if prev_sat_dt <= dt < target_dt:
            window_nums.extend(nums)

    counter = Counter(window_nums)
    eh = {n for n, cnt in counter.items() if cnt >= 4}
    h  = {n for n, cnt in counter.items() if cnt == 3}
    w  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
    c  = {n for n in range(1,46) if counter[n] == 0}

    w_pool_size = len(w)
    pool_eh_h_count = len(eh) + len(h)   # ← add this line
    c_pool_size = len(c)

    counts = {'EH':0, 'H':0, 'W':0, 'C':0}
    for n in target_main:
        if n in eh:    counts['EH'] += 1
        elif n in h:   counts['H'] += 1
        elif n in w:   counts['W'] += 1
        else:          counts['C'] += 1

    w_count = counts['W']
    profile = "Breadth" if w_count >= 4 else "Depth"

    # Legacy numbers (previous Saturday main numbers that appear in this draw)
    legacy_hits = [n for n in target_main if n in prev_main]

    results.append((target_date_str, profile, counts, w_pool_size, legacy_hits, pool_eh_h_count, c_pool_size))

# Print
n = min(OUTPUT_LAST_N, len(results))
print(f"Last {n} Saturday Lotto draws analysis:\n")
print(f"{'Date':<20} {'Profile':<10} {'EH':<4} {'H':<4} {'W':<4} {'C':<4} {'W-Pool Size':<12} {'Pool EH+H':<10} {'Cold Pool':<10} {'Legacy Hits'}")
print("-" * 90)
for date_str, profile, counts, w_pool, legacy, pool_eh_h, c_pool in results[-n:]:
    legacy_str = str(legacy) if legacy else "None"
    print(
        f"{date_str:<20} {profile:<10} {counts['EH']:<4} {counts['H']:<4} {counts['W']:<4} {counts['C']:<4} {w_pool:<12} {pool_eh_h:<10} {c_pool:<10} {legacy_str}")