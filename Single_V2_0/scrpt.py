import csv
from datetime import datetime as dt
from collections import Counter

# ---------- CONFIG (must match your generation run) ----------
TARGET_DATE = "2026-07-25"
BAND_CAPS = {
    '0x': 3,
    '1x': 3,
    '2x': 4,
    '3x': 4,
    '4x': 3,
    '5x+': 3
}

# ---------- YOUR 50 TICKETS ----------
tickets = [
    [10, 11, 13, 21, 25, 28],
    [2, 3, 5, 15, 32, 33],
    [4, 17, 18, 31, 37, 39],
    [1, 12, 22, 24, 25, 36],
    [8, 9, 19, 20, 30, 34],
    [2, 14, 16, 31, 32, 39],
    [5, 6, 15, 16, 23, 27],
    [4, 10, 11, 20, 25, 36],
    [9, 12, 14, 31, 37, 39],
    [3, 8, 11, 18, 25, 28],
    [11, 19, 20, 22, 31, 32],
    [2, 9, 10, 16, 37, 39],
    [1, 5, 8, 12, 23, 33],
    [6, 7, 15, 16, 34, 36],
    [4, 13, 14, 22, 27, 31],
    [17, 18, 19, 21, 24, 37],
    [2, 8, 18, 19, 30, 33],
    [4, 5, 12, 20, 25, 32],
    [1, 3, 10, 22, 23, 39],
    [6, 7, 11, 14, 28, 34],
    [8, 9, 13, 17, 27, 36],
    [2, 15, 16, 21, 24, 34],
    [4, 9, 10, 20, 30, 33],
    [5, 14, 15, 22, 25, 31],
    [2, 5, 10, 11, 25, 34],
    [7, 9, 16, 27, 31, 34],
    [4, 9, 15, 20, 25, 31],
    [4, 5, 11, 22, 31, 34],
    [1, 18, 19, 28, 37, 39],
    [2, 8, 19, 20, 28, 37],
    [1, 14, 15, 18, 25, 39],
    [10, 19, 20, 22, 37, 39],
    [4, 5, 16, 18, 28, 37],
    [1, 5, 8, 10, 28, 39],
    [3, 6, 16, 17, 23, 32],
    [1, 3, 8, 14, 23, 32],
    [7, 12, 17, 21, 36, 37],
    [12, 13, 20, 27, 32, 36],
    [11, 12, 23, 24, 33, 36],
    [6, 8, 17, 18, 23, 30],
    [7, 16, 21, 22, 32, 36],
    [9, 12, 15, 24, 25, 33],
    [2, 17, 19, 27, 30, 31],
    [3, 4, 10, 13, 28, 39],
    [5, 6, 12, 21, 23, 32],
    [7, 8, 14, 15, 27, 34],
    [9, 13, 19, 20, 33, 36],
    [3, 17, 18, 22, 30, 37],
    [2, 6, 11, 14, 23, 24],
    [7, 16, 17, 21, 28, 34],
]

# ---------- BUILD 20‑WEEK BAND MAP (same as in generator) ----------
def band_label(count):
    if count >= 5: return '5x+'
    if count == 4: return '4x'
    if count == 3: return '3x'
    if count == 2: return '2x'
    if count == 1: return '1x'
    return '0x'

all_sat_draws = []
with open("cross_lotto_data_backup.csv", newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    others_col = None
    for col in reader.fieldnames:
        if 'Others' in col:
            others_col = col
            break
    if not others_col:
        raise KeyError("Others column not found")
    for row in reader:
        if not row['Date'].strip().startswith('Sat'):
            continue
        date_str = row['Date'].strip()
        try:
            d = dt.strptime(date_str, '%a %d-%b-%Y')
        except:
            continue
        if d >= dt.strptime(TARGET_DATE, '%Y-%m-%d'):
            continue
        main_part = row[others_col].split('],')[0].strip()
        if main_part.startswith('['):
            main_part = main_part[1:]
        main_part = main_part.replace(']', '').strip()
        if main_part:
            nums = [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]
            if len(nums) == 6:
                all_sat_draws.append((d, nums))

all_sat_draws.sort(key=lambda x: x[0])
last_20 = all_sat_draws[-20:] if len(all_sat_draws) >= 20 else all_sat_draws
freq_20w = Counter()
for _, nums in last_20:
    for n in nums:
        freq_20w[n] += 1
band_for_num = {n: band_label(freq_20w.get(n, 0)) for n in range(1, 46)}

# ---------- VALIDATE EVERY TICKET ----------
violations = 0
for i, t in enumerate(tickets, 1):
    band_cnt = Counter(band_for_num[x] for x in t)
    for b, cnt in band_cnt.items():
        if cnt > BAND_CAPS[b]:
            print(f"❌ Ticket {i}: {sorted(t)}  → {b} has {cnt} numbers (cap: {BAND_CAPS[b]})")
            violations += 1
            break  # one violation per ticket is enough

if violations == 0:
    print("✅ All 50 tickets satisfy the 20‑week band caps.")
else:
    print(f"\n{violations} ticket(s) violate the band caps.")