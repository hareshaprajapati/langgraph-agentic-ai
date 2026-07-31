import csv
from datetime import datetime, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

# ---------- helpers ----------
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
    if 1 <= n <= 9:   return '0s'
    if 10 <= n <= 19: return '10s'
    if 20 <= n <= 29: return '20s'
    if 30 <= n <= 39: return '30s'
    return '40s'

# ---------- load all rows ----------
rows = []
with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    sfl_col = others_col = None
    for col in reader.fieldnames:
        if 'Set for Life' in col: sfl_col = col
        if 'Others' in col: others_col = col
    for row in reader:
        dt_str = row['Date'].strip()
        try: d = datetime.strptime(dt_str, '%a %d-%b-%Y')
        except: continue
        sfl = parse_numbers(row[sfl_col]) if row[sfl_col] else []
        oth = parse_numbers(row[others_col]) if row[others_col] else []
        rows.append((d, sfl, oth, row[others_col]))

# ---------- build Saturday data ----------
saturdays = []
for dt, sfl, oth, oth_raw in rows:
    if dt.weekday() != 5: continue
    main = parse_others_main(oth_raw)
    if len(main) != 6: continue

    wstart = dt - timedelta(days=7)
    wend   = dt - timedelta(days=1)
    window_nums = []
    for d, s, o, _ in rows:
        if wstart <= d <= wend:
            window_nums.extend(s)
            window_nums.extend(o)

    saturdays.append((dt, window_nums, main))

# ---------- analyse ----------
rank_pair_counts = Counter()
total_draws_with_2_missing = 0

for dt, wnums, main in saturdays:
    vols = Counter()
    for n in wnums: vols[decade_of(n)] += 1
    for d in ['0s','10s','20s','30s','40s']:
        if d not in vols: vols[d] = 0

    sorted_decs = sorted(vols.items(), key=lambda x: x[1])
    rank = {d: i+1 for i, (d, v) in enumerate(sorted_decs)}

    present = set(decade_of(n) for n in main)
    missing = {'0s','10s','20s','30s','40s'} - present

    if len(missing) == 2:
        total_draws_with_2_missing += 1
        # get ranks of missing decades, sort them to pair (smaller rank first)
        ranks = sorted(rank[d] for d in missing)
        rank_pair_counts[tuple(ranks)] += 1

# ---------- results ----------
print(f"Total draws with exactly 2 missing decades: {total_draws_with_2_missing}\n")
print("Rank pairs (when 2 decades were missing):")
print("Pair (r1,r2)   Count   Frequency")
print("-------------------------------")
for pair, cnt in rank_pair_counts.most_common():
    print(f"({pair[0]},{pair[1]})          {cnt:3d}    {cnt/total_draws_with_2_missing*100:.1f}%")