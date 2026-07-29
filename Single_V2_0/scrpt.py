import csv
from datetime import datetime, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_numbers(cell):
    """Return flat list of ints ≤45 from '[1,2,3], [4,5]'."""
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

def extract_main6(others_cell):
    """Extract first 6 numbers from Others column."""
    main_part = others_cell.split('],')[0].strip()
    if main_part.startswith('['): main_part = main_part[1:]
    main_part = main_part.replace(']', '').strip()
    if not main_part: return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]

def odd_even_ratio(nums):
    """Return (odd_count, even_count)."""
    if not nums: return (0,0)
    odd = sum(1 for n in nums if n % 2)
    return (odd, len(nums)-odd)

def high_low_ratio(nums):
    """Return (low_count, high_count). Low = 1-22, High = 23-45."""
    if not nums: return (0,0)
    low = sum(1 for n in nums if n <= 22)
    return (low, len(nums)-low)

def has_consecutive(nums):
    s = sorted(nums)
    return any(s[i+1]-s[i]==1 for i in range(len(s)-1))

def has_mirror(nums):
    return len({n%10 for n in nums}) < len(nums)

# ----------------------------------------------------------------------
# Load all rows
# ----------------------------------------------------------------------
rows = []
with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    sfl_col = others_col = None
    for col in reader.fieldnames:
        if 'Set for Life' in col: sfl_col = col
        if 'Others' in col: others_col = col
    for row in reader:
        date_str = row['Date'].strip()
        try: d = datetime.strptime(date_str, '%a %d-%b-%Y')
        except: continue
        sfl = parse_numbers(row[sfl_col]) if row[sfl_col] else []
        oth = parse_numbers(row[others_col]) if row[others_col] else []
        rows.append((d, sfl, oth, row[others_col]))

# ----------------------------------------------------------------------
# Find all Fridays and the next Saturday
# ----------------------------------------------------------------------
fridays = []   # (friday_date, sfl_main6, others_main6)
saturdays = [] # (saturday_date, main6, prev_friday_data)

for d, sfl, oth, oth_raw in rows:
    if d.weekday() == 4:   # Friday
        # Extract main 6 from both games
        sfl_main = sfl[:7]   # Set for Life main are first 7 numbers? Actually SFL has 7 main + 2 supp. We'll take first 7 as main, but we only need 6 for ratio? We'll use all 7 main. Actually the ratio calculation works on any length. We'll use the first 7 numbers as main.
        # But SFL format: "[main7], [supp2]". We'll take the first part as main.
        # Instead of slicing, we'll re-parse the raw cell for SFL main.
        # Better: store raw strings for Friday as well. But we have only others_raw. We'll re-parse from the raw row. Since we have all rows, we can just use sfl (which is all numbers). We'll use all SFL numbers (main+supp) but we need the main only for the rule "same extreme split appears in Set for Life main numbers". To be precise, we'll use the first list (main) only.
        # Let's extract SFL main from the raw string. We'll add sfl_raw to rows.
        # We didn't store sfl_raw. Let's store it.
        # We'll just re-read the CSV, but we can do a quick fix: use the first 7 numbers of sfl as main (since SFL always has 7 main). That's reliable.
        sfl_main = sfl[:7] if len(sfl) >= 7 else sfl
        oth_main = extract_main6(oth_raw) if oth_raw else []
        fridays.append((d, sfl_main, oth_main))
    elif d.weekday() == 5:   # Saturday
        main6 = extract_main6(oth_raw)
        if len(main6) == 6:
            saturdays.append((d, main6))

# Sort and pair each Saturday with the preceding Friday
saturdays.sort(key=lambda x: x[0])
fridays.sort(key=lambda x: x[0])

# For each Saturday, find the Friday immediately before
results = []
fri_idx = 0
for sat_date, sat_main in saturdays:
    # Find Friday where date = sat_date - 1 day
    target_fri = sat_date - timedelta(days=1)
    fri_data = None
    for f_date, sfl_main, oth_main in fridays:
        if f_date == target_fri:
            fri_data = (sfl_main, oth_main)
            break
    if fri_data is None:
        continue   # no Friday data (shouldn't happen)
    sfl_main, oth_main = fri_data
    results.append((sat_date, sat_main, sfl_main, oth_main))

# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------
print(f"Analysed {len(results)} Saturday draws with preceding Friday data.\n")

# Odd/Even extreme analysis
oe_total_extreme = 0
oe_fri_condition_met = 0
oe_fri_and_pair_met = 0   # winning ticket also has both consecutive and mirror
# High/Low extreme analysis
hl_total_extreme = 0
hl_fri_condition_met = 0
hl_fri_and_pair_met = 0

for sat_date, sat_main, sfl_main, oth_main in results:
    sat_oe = odd_even_ratio(sat_main)
    sat_hl = high_low_ratio(sat_main)

    # Check odd/even extreme in Saturday
    sat_oe_extreme = (sat_oe[0] == 1 and sat_oe[1] == 5) or (sat_oe[0] == 5 and sat_oe[1] == 1)
    if sat_oe_extreme:
        oe_total_extreme += 1
        # Check Friday condition: SFL main or Others main has the same extreme split
        sfl_oe = odd_even_ratio(sfl_main)
        oth_oe = odd_even_ratio(oth_main) if oth_main else (0,0)
        cond_met = (sfl_oe[0] == 1 and sfl_oe[1] == 5) or (sfl_oe[0] == 5 and sfl_oe[1] == 1) \
                   or (oth_oe[0] == 1 and oth_oe[1] == 5) or (oth_oe[0] == 5 and oth_oe[1] == 1)
        if cond_met:
            oe_fri_condition_met += 1
            # Does winning ticket have both consecutive and mirror?
            if has_consecutive(sat_main) and has_mirror(sat_main):
                oe_fri_and_pair_met += 1

    # Check high/low extreme in Saturday
    sat_hl_extreme = (sat_hl[0] == 1 and sat_hl[1] == 5) or (sat_hl[0] == 5 and sat_hl[1] == 1)
    if sat_hl_extreme:
        hl_total_extreme += 1
        sfl_hl = high_low_ratio(sfl_main)
        oth_hl = high_low_ratio(oth_main) if oth_main else (0,0)
        cond_met = (sfl_hl[0] == 1 and sfl_hl[1] == 5) or (sfl_hl[0] == 5 and sfl_hl[1] == 1) \
                   or (oth_hl[0] == 1 and oth_hl[1] == 5) or (oth_hl[0] == 5 and oth_hl[1] == 1)
        if cond_met:
            hl_fri_condition_met += 1
            if has_consecutive(sat_main) and has_mirror(sat_main):
                hl_fri_and_pair_met += 1

# Print results
print("----- Odd/Even Extreme Ratio (1:5 or 5:1) -----")
print(f"Total extreme Saturday draws: {oe_total_extreme}")
print(f"Friday condition met (SFL or Others had extreme split): {oe_fri_condition_met} "
      f"({oe_fri_condition_met/oe_total_extreme*100:.1f}%)" if oe_total_extreme else "N/A")
print(f"Friday condition met AND winning ticket had both Consec+Mirror: {oe_fri_and_pair_met} "
      f"({oe_fri_and_pair_met/oe_total_extreme*100:.1f}%)" if oe_total_extreme else "N/A")

print("\n----- High/Low Extreme Ratio (1:5 or 5:1) -----")
print(f"Total extreme Saturday draws: {hl_total_extreme}")
print(f"Friday condition met: {hl_fri_condition_met} "
      f"({hl_fri_condition_met/hl_total_extreme*100:.1f}%)" if hl_total_extreme else "N/A")
print(f"Friday condition met AND winning ticket had both Consec+Mirror: {hl_fri_and_pair_met} "
      f"({hl_fri_and_pair_met/hl_total_extreme*100:.1f}%)" if hl_total_extreme else "N/A")

# Overall coverage of extreme draws by the conditional rule (if we allow such tickets only when Friday condition met)
print("\n===== Overall conditional rule coverage =====")
print(f"Extreme O/E draws covered: {oe_fri_and_pair_met}/{oe_total_extreme} "
      f"({oe_fri_and_pair_met/oe_total_extreme*100:.1f}%)" if oe_total_extreme else "N/A")
print(f"Extreme H/L draws covered: {hl_fri_and_pair_met}/{hl_total_extreme} "
      f"({hl_fri_and_pair_met/hl_total_extreme*100:.1f}%)" if hl_total_extreme else "N/A")