import csv
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

def decade(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

# Read all Saturday main numbers
sat_data = []
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)   # header
    for row in reader:
        if len(row) < 3: continue
        date_str = row[0].strip()
        if not date_str.startswith('Sat '): continue
        others = row[2]
        main_part = others.split(']')[0].replace('[', '').strip()
        if not main_part: continue
        nums = [int(x.strip()) for x in main_part.split(',') if x.strip()]
        if len(nums) == 6:
            sat_data.append((date_str, nums))

# Analyse
max_counts = []   # list of (date, max_in_one_decade)
for date, nums in sat_data:
    cnt = Counter(decade(n) for n in nums)
    max_cnt = max(cnt.values())
    max_counts.append((date, max_cnt))

# Frequency table
freq = Counter()
for _, mc in max_counts:
    freq[mc] += 1

print("Max numbers in a single decade across the 6 winning numbers:")
for k in sorted(freq):
    print(f"  {k} numbers: {freq[k]} draws")

total = len(sat_data)
print(f"\nTotal Saturday draws analysed: {total}")
print("\nPercentage of draws:")
for k in sorted(freq):
    print(f"  {k} numbers: {freq[k]/total*100:.1f}%")