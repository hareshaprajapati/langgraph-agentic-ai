import csv
import ast
from collections import defaultdict
from pathlib import Path

# ----- Configuration -----
CSV_FILE = "cross_lotto_data_backup.csv"  # <-- point to your CSV
OUTPUT_FILE = "backtest_consec_mirror.csv"


def has_consecutive(nums):
    s = sorted(nums)
    return any(s[i + 1] - s[i] == 1 for i in range(len(s) - 1))


def has_mirror(nums):
    return len({n % 10 for n in nums}) < 6


def parse_others(others_str):
    """
    The 'Others' field looks like:
    "[1, 2, 3, 4, 5, 6], [10, 20]"
    We extract the first list (main numbers) and return as a list of ints.
    """
    # Split on '],' to separate the two lists
    parts = others_str.split('],')
    if not parts:
        return []
    main_part = parts[0].strip()
    # Remove leading '[' if present and trailing whitespace
    if main_part.startswith('['):
        main_part = main_part[1:]
    # Now it's something like "1, 2, 3, 4, 5, 6" or "1, 2, 3, 4, 5, 6]"
    main_part = main_part.replace(']', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',')]


# ----- Main -----
def main():
    saturdays = []
    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:   # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        # find the column that contains 'Others'
        others_col = None
        for col in reader.fieldnames:
            if 'Others' in col:
                others_col = col
                break
        if others_col is None:
            raise KeyError("Could not find a column with 'Others' in its name. Columns: " + str(reader.fieldnames))

        for row in reader:
            date = row['Date'].strip()
            if date.startswith('Sat'):
                others_str = row[others_col]
                main_nums = parse_others(others_str)
                if len(main_nums) == 6:
                    saturdays.append((date, main_nums))
    # … rest is unchanged

    total = len(saturdays)
    if total == 0:
        print("No Saturday draws found. Check CSV content and column name.")
        return

    counts = {
        'both': 0,
        'only_consec': 0,
        'only_mirror': 0,
        'neither': 0
    }
    results = []
    for date, nums in saturdays:
        c = has_consecutive(nums)
        m = has_mirror(nums)
        if c and m:
            counts['both'] += 1
            cat = 'both'
        elif c and not m:
            counts['only_consec'] += 1
            cat = 'consec_only'
        elif m and not c:
            counts['only_mirror'] += 1
            cat = 'mirror_only'
        else:
            counts['neither'] += 1
            cat = 'neither'
        results.append((date, nums, cat))

    print(f"Total Saturday draws analyzed: {total}\n")
    for k, v in counts.items():
        print(f"{k:15s}: {v:3d}  ({v/total*100:.1f}%)")

    # Write detailed CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Main_Numbers', 'Consecutive', 'Mirror', 'Category'])
        for date, nums, cat in results:
            writer.writerow([date, str(nums), has_consecutive(nums), has_mirror(nums), cat])
    print(f"\nDetailed results written to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()