import csv
import sys
from collections import defaultdict, Counter
from datetime import datetime

# ---------- CONFIG ----------
CSV_FILE = "cross_lotto_data_backup.csv"
TARGET_DATE = "Wed 09-Sep-2026"   # change as needed
NUMBER = 15                       # number to find attractions for
WINDOW_SIZE = 30                  # last N draws to consider

# ---------- HELPERS ----------
def parse_date(s):
    return datetime.strptime(s[4:], '%d-%b-%Y')

def parse_main_numbers(cell):
    if not cell:
        return []
    main_part = cell.split(']')[0].replace('[', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

def load_draws_before(csv_file, target_date, window_size):
    """
    Returns a list of draws (each a list of main numbers) from both SFL and Others,
    limited to the last `window_size` draws that occur before `target_date`.
    """
    target_dt = parse_date(target_date)
    all_entries = []
    with open(csv_file, 'r', encoding='utf-8') as f:
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
            if dt >= target_dt:
                continue   # only consider draws before target
            sfl_cell = row[1] if len(row) > 1 else ''
            others_cell = row[2] if len(row) > 2 else ''
            sfl_main = parse_main_numbers(sfl_cell)
            others_main = parse_main_numbers(others_cell)
            if sfl_main:
                all_entries.append((dt, sfl_main))
            if others_main:
                all_entries.append((dt, others_main))
    # Sort by date (most recent last)
    all_entries.sort(key=lambda x: x[0])
    # Take the last `window_size` entries
    recent = all_entries[-window_size:] if len(all_entries) >= window_size else all_entries
    draws = [nums for _, nums in recent]
    return draws

def compute_cooccurrence(draws):
    cooc = defaultdict(Counter)
    for nums in draws:
        nums = sorted(nums)
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                a, b = nums[i], nums[j]
                cooc[a][b] += 1
                cooc[b][a] += 1
    return cooc

# ---------- MAIN ----------
draws = load_draws_before(CSV_FILE, TARGET_DATE, WINDOW_SIZE)
if not draws:
    print(f"No draws found before {TARGET_DATE}.")
    sys.exit(1)

print(f"Using last {len(draws)} draws before {TARGET_DATE}.\n")
cooc = compute_cooccurrence(draws)

if NUMBER not in cooc:
    print(f"{NUMBER} did not appear in any of those draws.")
    sys.exit(0)

top = cooc[NUMBER].most_common(10)
print(f"Top {len(top)} numbers that appear with {NUMBER} in the same draw (across Others and Set for Life):\n")
for num, count in top:
    print(f"  {num:2d} : {count} times")