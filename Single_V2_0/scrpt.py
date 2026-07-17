import csv
from collections import Counter, defaultdict
from datetime import datetime

CSV_FILE = "cross_lotto_data_backup.csv"

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
    main_part = others_cell.split(']')[0].replace('[', '').strip()
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

# ---------- Read the whole CSV ----------
all_rows = []
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)   # skip header
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

# ---------- Extract Saturdays ----------
saturdays = []
for date_str, dt, is_sat, _, others_cell in all_rows:
    if is_sat:
        main6 = extract_main6(others_cell)
        saturdays.append((date_str, dt, main6))

# ---------- Analyse each Saturday (starting from the second one) ----------
count_both = 0       # Tier Bridge AND Anchor Mirror
count_either = 0     # Tier Bridge OR  Anchor Mirror
count_neither = 0
total_analysed = 0

for i in range(1, len(saturdays)):
    target_date_str, target_dt, target_main = saturdays[i]
    prev_sat_date_str, prev_sat_dt, _ = saturdays[i-1]

    # 7‑day window (Sat–Fri before target)
    window_nums = []
    for date_str, dt, is_sat, nums, _ in all_rows:
        if prev_sat_dt <= dt < target_dt:
            window_nums.extend(nums)

    # Tiering
    counter = Counter(window_nums)
    EH = {n for n, cnt in counter.items() if cnt >= 4}
    H  = {n for n, cnt in counter.items() if cnt == 3}
    W  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}

    # Glue checks
    def has_tier_bridge(t):
        s = sorted(t)
        for j in range(len(s)-1):
            a, b = s[j], s[j+1]
            if b - a == 1:
                if ((a in EH or a in H) and b in W) or ((b in EH or b in H) and a in W):
                    return True
        return False

    def has_anchor_mirror(t):
        for a in t:
            if a in EH or a in H:
                for b in t:
                    if b != a and b % 10 == a % 10:
                        if b in W:
                            return True
        return False

    tb = has_tier_bridge(target_main)
    am = has_anchor_mirror(target_main)

    if tb and am:
        count_both += 1
    if tb or am:
        count_either += 1
    else:
        count_neither += 1

    total_analysed += 1

# ---------- Results ----------
print(f"Total Saturday draws analysed: {total_analysed}")
print(f"Has Tier Bridge AND Anchor Mirror : {count_both} ({count_both/total_analysed*100:.1f}%)")
print(f"Has Tier Bridge OR  Anchor Mirror : {count_either} ({count_either/total_analysed*100:.1f}%)")
print(f"Has neither                       : {count_neither} ({count_neither/total_analysed*100:.1f}%)")