import csv
from datetime import datetime
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_others_main(others_str):
    """Extract the 6 main numbers from the Others column."""
    main_part = others_str.split('],')[0].strip()
    if main_part.startswith('['):
        main_part = main_part[1:]
    main_part = main_part.replace(']', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]

def band_of(count):
    """20‑week frequency band label."""
    if count >= 5: return '5x+'
    if count == 4: return '4x'
    if count == 3: return '3x'
    if count == 2: return '2x'
    if count == 1: return '1x'
    return '0x'

# ----------------------------------------------------------------------
# Load all Saturdays in chronological order
# ----------------------------------------------------------------------
saturdays = []
with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    others_col = None
    for col in reader.fieldnames:
        if 'Others' in col:
            others_col = col
            break
    for row in reader:
        if not row['Date'].strip().startswith('Sat'):
            continue
        dt_str = row['Date'].strip()
        try:
            dt = datetime.strptime(dt_str, '%a %d-%b-%Y')
        except:
            continue
        main = parse_others_main(row[others_col])
        if len(main) == 6:
            saturdays.append(main)   # list of 6‑element lists, oldest first

# ----------------------------------------------------------------------
# Analyse draws from the 21st onward (need 20 draws of history)
# ----------------------------------------------------------------------
violations_max_band = 0
violations_hot_combo = 0
total = 0
band_max_counts = []  # for stats

for i in range(20, len(saturdays)):
    # previous 20 draws
    prev_flat = []
    for j in range(i-20, i):
        prev_flat.extend(saturdays[j])
    freq = Counter(prev_flat)

    curr = saturdays[i]
    bands_in_draw = [band_of(freq.get(n, 0)) for n in curr]
    band_counts = Counter(bands_in_draw)

    total += 1
    max_band_cnt = max(band_counts.values())
    band_max_counts.append(max_band_cnt)

    # Rule 1: max numbers from any single band ≤ 3
    if max_band_cnt > 3:
        violations_max_band += 1
        print(f"⚠️  Draw {i} violates ≤3 per band: {curr}, bands={dict(band_counts)}")

    # Rule 2: max numbers from combined 5x+ and 4x ≤ 2
    hot_count = band_counts.get('5x+', 0) + band_counts.get('4x', 0)
    if hot_count > 2:
        violations_hot_combo += 1
        print(f"🔥 Draw {i} violates ≤2 hot (5x+4x): {curr}, hot={hot_count}")

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print(f"\nAnalysed {total} draws (from the 21st onward)")
print(f"Violations of '≤3 per band': {violations_max_band} ({violations_max_band/total*100:.2f}%)")
print(f"Violations of '≤2 hot (5x+4x)': {violations_hot_combo} ({violations_hot_combo/total*100:.2f}%)")

# Additional stats on max per band
print(f"\nMax per band statistics: min={min(band_max_counts)}, max={max(band_max_counts)}")
dist = Counter(band_max_counts)
for k in sorted(dist):
    print(f"  {k} numbers from a single band in a draw: {dist[k]} draws")