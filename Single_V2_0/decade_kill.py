import csv
from datetime import datetime, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"   # your file

def parse_numbers(combined_str):
    """
    Parse a string like "[1,2,3,4,5,6], [7,8]" and return a flat list of ints.
    Discards numbers > 45.
    """
    parts = combined_str.split('],')
    nums = []
    for part in parts:
        part = part.strip().rstrip(']').lstrip('[')
        if part:
            for token in part.split(','):
                token = token.strip()
                if token.isdigit():
                    n = int(token)
                    if n <= 45:
                        nums.append(n)
    return nums

def decade_counts(numbers):
    """Return Counter of decade strings for a flat list of numbers."""
    cnt = Counter()
    for n in numbers:
        if 1 <= n <= 9:      cnt['0s'] += 1
        elif 10 <= n <= 19:  cnt['10s'] += 1
        elif 20 <= n <= 29:  cnt['20s'] += 1
        elif 30 <= n <= 39:  cnt['30s'] += 1
        elif 40 <= n <= 45:  cnt['40s'] += 1
    return cnt

def missing_decades(main_nums):
    """Return set of decades NOT present in the 6 main numbers."""
    present = set()
    for n in main_nums:
        if 1 <= n <= 9:      present.add('0s')
        elif 10 <= n <= 19:  present.add('10s')
        elif 20 <= n <= 29:  present.add('20s')
        elif 30 <= n <= 39:  present.add('30s')
        elif 40 <= n <= 45:  present.add('40s')
    return {'0s','10s','20s','30s','40s'} - present

def main():
    # ------- 1. Read all rows, parse dates & numbers -------
    rows = []
    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        # Locate columns
        sfl_col = None
        others_col = None
        for col in reader.fieldnames:
            if 'Set for Life' in col:
                sfl_col = col
            if 'Others' in col:
                others_col = col
        if not sfl_col or not others_col:
            raise KeyError("Missing required columns")

        for row in reader:
            date_str = row['Date'].strip()
            try:
                dt = datetime.strptime(date_str, '%a %d-%b-%Y')
            except ValueError:
                continue  # skip malformed dates
            sfl_nums = parse_numbers(row[sfl_col]) if row[sfl_col] else []
            others_nums = parse_numbers(row[others_col]) if row[others_col] else []
            all_nums = sfl_nums + others_nums
            rows.append((dt, all_nums, others_nums))  # others_nums will be used for Saturday's main

    # ------- 2. For each Saturday, build window and validate kill logic -------
    saturdays = [(dt, nums) for dt, _, nums in rows if dt.weekday() == 5]  # Monday=0, Saturday=5
    total = len(saturdays)
    if total == 0:
        print("No Saturdays found.")
        return

    hittable = 0          # missing decade ∈ killable set
    miss_no_missing = 0   # draw covers all 5 decades
    miss_unhittable = 0   # missing only decades NOT in killable set

    decade_absence_freq = Counter()
    killable_set = {'40s'}  # 40s always killed

    for dt, main_nums in saturdays:
        missing = missing_decades(main_nums)
        for d in missing:
            decade_absence_freq[d] += 1

        # Build window: from dt-7 days to dt-1 day (inclusive)
        window_start = dt - timedelta(days=7)   # previous Saturday
        window_end   = dt - timedelta(days=1)   # Friday
        window_nums = []
        for d, all_nums, _ in rows:
            if window_start <= d <= window_end:
                window_nums.extend(all_nums)
        decade_vol = decade_counts(window_nums)

        if not decade_vol:
            # incomplete window data, skip
            continue

        # Determine H, L, middle
        sorted_decades = sorted(decade_vol.items(), key=lambda x: x[1])
        L = sorted_decades[0][0]  # lowest volume
        H = sorted_decades[-1][0] # highest volume
        middle = [d for d in ['0s','10s','20s','30s','40s'] if d != L and d != H]

        # Killable set for this draw = {40s} ∪ middle decades
        killable = {'40s'} | set(middle)

        if not missing:
            miss_no_missing += 1
        elif missing & killable:
            hittable += 1
        else:
            miss_unhittable += 1

    # ------- 3. Results -------
    print(f"Total Saturday draws analysed: {total}\n")
    print(f"✅ Hittable (missing a killable decade):   {hittable:3d} ({hittable/total*100:.1f}%)")
    print(f"❌ Missed – draw covered all 5 decades:    {miss_no_missing:3d} ({miss_no_missing/total*100:.1f}%)")
    print(f"❌ Missed – missing only unkillable decade: {miss_unhittable:3d} ({miss_unhittable/total*100:.1f}%)")

    print(f"\nDecade absence frequency (how often each decade was absent):")
    for d in ['0s','10s','20s','30s','40s']:
        print(f"  {d}: {decade_absence_freq[d]:3d} times ({decade_absence_freq[d]/total*100:.1f}%)")

    print(f"\nInterpretation:")
    print(f"The kill rule (always kill 40s + one random middle decade) covers {hittable/total*100:.1f}% of draws.")
    print(f"The remaining {(miss_no_missing+miss_unhittable)/total*100:.1f}% are structurally out of reach.")

if __name__ == '__main__':
    main()