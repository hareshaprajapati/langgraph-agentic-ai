import csv
from datetime import datetime, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

# ---------- Helpers ----------
def parse_numbers(cell):
    nums = []
    for part in cell.split('],'):
        part = part.strip().lstrip('[').rstrip(']')
        for token in part.split(','):
            token = token.strip()
            if token.isdigit():
                n = int(token)
                if n <= 45:
                    nums.append(n)
    return nums

def decade_counts(numbers):
    cnt = Counter()
    for n in numbers:
        if 1 <= n <= 9:      cnt['0s'] += 1
        elif 10 <= n <= 19:  cnt['10s'] += 1
        elif 20 <= n <= 29:  cnt['20s'] += 1
        elif 30 <= n <= 39:  cnt['30s'] += 1
        elif 40 <= n <= 45:  cnt['40s'] += 1
    # ensure all five keys exist
    for d in ['0s','10s','20s','30s','40s']:
        if d not in cnt:
            cnt[d] = 0
    return cnt

def missing_decades(main_nums):
    present = set()
    for n in main_nums:
        if 1 <= n <= 9: present.add('0s')
        elif 10 <= n <= 19: present.add('10s')
        elif 20 <= n <= 29: present.add('20s')
        elif 30 <= n <= 39: present.add('30s')
        elif 40 <= n <= 45: present.add('40s')
    return {'0s','10s','20s','30s','40s'} - present

# ---------- Kill rules to test ----------
def kill_lowest_always(vols):
    """Always kill the single decade with the smallest volume.
       If tie, pick the first (arbitrary, but we can handle later)."""
    return [min(vols, key=vols.get)]

def kill_lowest_if_below(vols, threshold):
    """Kill the lowest-volume decade only if its count <= threshold."""
    lowest_dec = min(vols, key=vols.get)
    if vols[lowest_dec] <= threshold:
        return [lowest_dec]
    return []   # kill nothing

def kill_40s_always(vols):
    """Always kill the 40s."""
    return ['40s']

def kill_40s_if_low(vols, threshold=3):
    """Kill 40s only if its volume <= threshold."""
    if vols['40s'] <= threshold:
        return ['40s']
    return []

def kill_all_below_threshold(vols, threshold=3):
    """Kill every decade with volume <= threshold."""
    return [d for d in vols if vols[d] <= threshold]

def kill_lowest_and_40s(vols):
    """Kill the lowest-volume decade and also the 40s (if not already lowest)."""
    lowest = min(vols, key=vols.get)
    res = [lowest]
    if lowest != '40s':
        res.append('40s')
    return res

# ---------- Backtest ----------
def main():
    rows = []
    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        sfl_col = None
        others_col = None
        for col in reader.fieldnames:
            if 'Set for Life' in col: sfl_col = col
            if 'Others' in col: others_col = col
        for row in reader:
            date_str = row['Date'].strip()
            try:
                dt = datetime.strptime(date_str, '%a %d-%b-%Y')
            except ValueError:
                continue
            sfl_nums = parse_numbers(row[sfl_col]) if row[sfl_col] else []
            others_nums = parse_numbers(row[others_col]) if row[others_col] else []
            rows.append((dt, sfl_nums + others_nums))

    # Extract Saturdays and build windows
    saturdays = []
    for dt, _ in rows:
        if dt.weekday() == 5:  # Saturday
            # window: previous Saturday (dt-7) to Friday (dt-1)
            window_start = dt - timedelta(days=7)
            window_end = dt - timedelta(days=1)
            window_nums = []
            for d, nums in rows:
                if window_start <= d <= window_end:
                    window_nums.extend(nums)
            # Saturday main numbers (others main) – we need to re-parse from the row
            # For simplicity, we'll reconstruct: loop again to find the exact row for this dt
            # Better: store main numbers separately.
            saturdays.append((dt, window_nums))
    # Second pass to get Saturday main numbers
    sat_main = {}
    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_str = row['Date'].strip()
            if not date_str.startswith('Sat'):
                continue
            dt = datetime.strptime(date_str, '%a %d-%b-%Y')
            others_str = row[others_col]
            # extract first list (main numbers)
            main_part = others_str.split('],')[0].strip()
            if main_part.startswith('['):
                main_part = main_part[1:]
            main_part = main_part.replace(']', '').strip()
            if main_part:
                main_nums = [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]
                if len(main_nums) == 6:
                    sat_main[dt] = main_nums

    # Build list of (dt, window_nums, sat_main_nums)
    test_data = []
    for dt, wnums in saturdays:
        if dt in sat_main:
            test_data.append((dt, wnums, sat_main[dt]))

    # Define rules to test
    rules = {
        'Lowest always': lambda vols: kill_lowest_always(vols),
        'Lowest if ≤3': lambda vols: kill_lowest_if_below(vols, 3),
        'Lowest if ≤5': lambda vols: kill_lowest_if_below(vols, 5),
        '40s always': lambda vols: kill_40s_always(vols),
        '40s if ≤3': lambda vols: kill_40s_if_low(vols, 3),
        'Lowest + 40s': lambda vols: kill_lowest_and_40s(vols),
        'All ≤3': lambda vols: kill_all_below_threshold(vols, 3),
    }

    results = {name: {'hits': 0, 'total': 0} for name in rules}

    for dt, wnums, main in test_data:
        vols = decade_counts(wnums)
        missing = missing_decades(main)
        for name, rule_func in rules.items():
            killed = rule_func(vols)
            # A draw is hittable if at least one killed decade is missing
            if any(d in missing for d in killed):
                results[name]['hits'] += 1
            results[name]['total'] += 1

    print(f"Backtest on {len(test_data)} Saturdays\n")
    print(f"{'Rule':<20} {'Hittable':>8} {'Total':>6} {'Rate':>7}")
    print("-" * 45)
    for name, res in sorted(results.items(), key=lambda x: -x[1]['hits']/x[1]['total']):
        rate = res['hits'] / res['total'] * 100
        print(f"{name:<20} {res['hits']:>8} {res['total']:>6} {rate:>6.1f}%")

if __name__ == '__main__':
    main()