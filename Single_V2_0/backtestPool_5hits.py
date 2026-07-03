import csv
from collections import Counter
from datetime import datetime, timedelta

def parse_date(date_str):
    return datetime.strptime(date_str.split(' ', 1)[1], '%d-%b-%Y')

def extract_main_numbers(cell):
    if not cell:
        return []
    main_part = cell.split(']')[0].replace('[', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',')]

def extract_all_numbers(cell):
    """Extract all numbers (main + supp) from a cell like '[1,2,3], [4,5]'."""
    if not cell:
        return []
    nums = []
    parts = cell.replace('[', '').replace(']', '').split(',')
    for p in parts:
        p = p.strip()
        if p:
            try:
                nums.append(int(p))
            except:
                pass
    return nums

rows = []
with open('cross_lotto_data_backup.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        date_str, sfl, oth = row[0], row[1], row[2]
        rows.append((date_str, sfl, oth))

all_draws = []
for date_str, sfl, oth in rows:
    dt = parse_date(date_str)
    is_saturday = date_str.startswith('Sat ')
    all_draws.append({
        'date': dt,
        'date_str': date_str,
        'is_saturday': is_saturday,
        'sfl_mains': extract_main_numbers(sfl),
        'oth_mains': extract_main_numbers(oth),
        'sfl_all': extract_all_numbers(sfl),      # added
        'oth_all': extract_all_numbers(oth),      # added
    })

all_draws.sort(key=lambda x: x['date'])
all_saturdays = [d for d in all_draws if d['is_saturday']]

def most_overdue_saturday(prev_sat_draws, n=5):
    last_seen = {}
    for num in range(1, 46):
        for i in range(len(prev_sat_draws) - 1, -1, -1):
            if num in prev_sat_draws[i]['oth_mains']:
                last_seen[num] = i
                break
        else:
            last_seen[num] = -1
    total = len(prev_sat_draws)
    def draws_since(num):
        if last_seen[num] == -1:
            return total + 1
        return total - 1 - last_seen[num]
    return sorted(range(1, 46), key=lambda n: (-draws_since(n), n))[:n]

target_sats = all_saturdays[-20:]
hits = []

for sat in target_sats:
    target_date = sat['date']
    prev_sats = [s for s in all_saturdays if s['date'] < target_date]
    if prev_sats:
        week_start = prev_sats[-1]['date'] + timedelta(days=1)
    else:
        week_start = target_date - timedelta(days=7)
    week_end = target_date - timedelta(days=1)
    week_draws = [d for d in all_draws if week_start <= d['date'] <= week_end and not d['is_saturday']]

    # --- CHANGED: count ALL numbers (main + supp) ---
    all_freq = Counter()
    for d in week_draws:
        all_freq.update(d['sfl_all'])
        all_freq.update(d['oth_all'])

    # Top 35 by frequency, tie‑break lower number
    sorted_weekly = sorted(all_freq.items(), key=lambda x: (-x[1], x[0]))
    top35_weekly = [num for num, cnt in sorted_weekly[:35]]

    # 5 most overdue Saturday numbers
    overdue5 = most_overdue_saturday(prev_sats, n=5)

    # Combine and cap at 35
    pool = set(top35_weekly) | set(overdue5)
    if len(pool) > 35:
        # Remove lowest‑frequency weekly numbers (not in overdue)
        lowest_first = sorted(all_freq.items(), key=lambda x: (x[1], -x[0]))
        for num, cnt in lowest_first:
            if num in pool and num not in overdue5:
                pool.remove(num)
                if len(pool) == 35:
                    break

    winners = set(sat['oth_mains'])
    matched = len(winners & pool)
    hits.append(matched)
    print(f"{sat['date_str']}: {matched}/6 (pool size {len(pool)})")
    # print(pool)
print(f"\nAverage matches: {sum(hits)/len(hits):.2f}")
print(f"Draws with 6/6: {hits.count(6)}")