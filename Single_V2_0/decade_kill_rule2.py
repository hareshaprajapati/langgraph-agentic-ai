import csv
from datetime import datetime, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

def parse_numbers(cell):
    nums = []
    for part in cell.split('],'):
        part = part.strip().lstrip('[').rstrip(']')
        for token in part.split(','):
            token = token.strip()
            if token.isdigit():
                n = int(token)
                if n <= 45: nums.append(n)
    return nums

def missing_decades(main):
    present = set()
    for n in main:
        if 1 <= n <= 9: present.add('0s')
        elif 10 <= n <= 19: present.add('10s')
        elif 20 <= n <= 29: present.add('20s')
        elif 30 <= n <= 39: present.add('30s')
        elif 40 <= n <= 45: present.add('40s')
    return {'0s','10s','20s','30s','40s'} - present

def main():
    rows = []
    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        sfl_col = others_col = None
        for col in reader.fieldnames:
            if 'Set for Life' in col: sfl_col = col
            if 'Others' in col: others_col = col
        for row in reader:
            dt_str = row['Date'].strip()
            try:
                dt = datetime.strptime(dt_str, '%a %d-%b-%Y')
            except ValueError:
                continue
            sfl = parse_numbers(row[sfl_col]) if row[sfl_col] else []
            oth = parse_numbers(row[others_col]) if row[others_col] else []
            rows.append((dt, sfl + oth))

    sat_info = []
    for dt, _ in rows:
        if dt.weekday() != 5: continue
        wstart = dt - timedelta(days=7)
        wend = dt - timedelta(days=1)
        wnums = []
        for d, nums in rows:
            if wstart <= d <= wend:
                wnums.extend(nums)
        sat_info.append((dt, wnums, None))

    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['Date'].strip().startswith('Sat'): continue
            dt = datetime.strptime(row['Date'].strip(), '%a %d-%b-%Y')
            main_part = row[others_col].split('],')[0].strip()
            if main_part.startswith('['): main_part = main_part[1:]
            main_part = main_part.replace(']', '').strip()
            if main_part:
                main_nums = [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]
                if len(main_nums) == 6:
                    for i, (d, w, _) in enumerate(sat_info):
                        if d == dt:
                            sat_info[i] = (d, w, main_nums)
                            break

    total = 0
    pairs = [('0s', '40s'), ('10s', '40s'), ('20s', '40s'), ('30s', '40s')]
    hits = {pair: 0 for pair in pairs}
    # also track 40s-only for comparison
    only_40s_hits = 0

    for dt, wnums, main in sat_info:
        if main is None: continue
        total += 1
        missing = missing_decades(main)

        if '40s' in missing:
            only_40s_hits += 1

        for d1, d2 in pairs:
            if d1 in missing and d2 in missing:
                hits[(d1, d2)] += 1

    print(f"Total Saturday draws: {total}\n")
    print(f"{'Killed pair':<15} {'Both missing':>12} {'Rate':>8}")
    print("-" * 38)
    print(f"{'40s only':<15} {only_40s_hits:>12} {only_40s_hits/total*100:>7.1f}%")
    for pair in pairs:
        h = hits[pair]
        print(f"{pair[0]}+{pair[1]:<10} {h:>12} {h/total*100:>7.1f}%")

if __name__ == '__main__':
    main()