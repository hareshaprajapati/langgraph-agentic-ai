import pandas as pd
import ast
from collections import defaultdict, Counter
import itertools

def parse_main_numbers(cell):
    """
    Extract the 7 main numbers from the 'Others' column.
    Example: "[2, 11, 15, 22, 29, 39], [28, 41]" -> [2,11,15,22,29,39]
    """
    if pd.isna(cell) or cell == '':
        return []
    start = cell.find('[')
    if start == -1:
        return []
    end = cell.find(']', start)
    if end == -1:
        return []
    try:
        nums = ast.literal_eval(cell[start:end+1])
        if isinstance(nums, list):
            return nums
    except:
        return []
    return []

# Read CSV
df = pd.read_csv('cross_lotto_data_backup.csv', encoding='utf-8')

# Show available columns (for debugging)
print("Columns in CSV:", df.columns.tolist())

# Find the column that contains 'Others' (case-insensitive)
others_col = None
for col in df.columns:
    if 'others' in col.lower():
        others_col = col
        break

if others_col is None:
    raise ValueError("Could not find a column with 'Others' in its name.")

print(f"Using column: '{others_col}'")

# Parse main numbers
df['Others_main'] = df[others_col].apply(parse_main_numbers)

# Keep only rows with exactly 7 numbers
df = df[df['Others_main'].apply(len) == 7].copy()
df = df.sort_values('Date')  # ensure chronological order

if len(df) == 0:
    print("No valid rows found. Check parsing.")
    exit()

# --- 1. Co-occurrence (attraction) ---
pair_count = Counter()
for nums in df['Others_main']:
    for pair in itertools.combinations(sorted(nums), 2):
        pair_count[pair] += 1

print("\nTop 20 most frequent number pairs (attraction):")
for pair, cnt in pair_count.most_common(20):
    print(f"{pair[0]:2d} - {pair[1]:2d} : {cnt} times")

# --- 2. Next-draw pair transitions ---
dates = df['Date'].values
nums_list = df['Others_main'].values

transition_counter = defaultdict(lambda: defaultdict(int))

for i in range(len(nums_list)-1):
    current = nums_list[i]
    next_draw = nums_list[i+1]
    for pair in itertools.combinations(sorted(current), 2):
        a, b = pair
        # Check if same pair appears
        if set(pair).issubset(next_draw):
            transition_counter[pair]['same'] += 1
        # Check for shifts
        if (a, b+1) in itertools.combinations(sorted(next_draw), 2):
            transition_counter[pair]['b+1'] += 1
        if (a+1, b) in itertools.combinations(sorted(next_draw), 2):
            transition_counter[pair]['a+1'] += 1
        if (a, b-1) in itertools.combinations(sorted(next_draw), 2):
            transition_counter[pair]['b-1'] += 1
        if (a-1, b) in itertools.combinations(sorted(next_draw), 2):
            transition_counter[pair]['a-1'] += 1
        if (a+1, b+1) in itertools.combinations(sorted(next_draw), 2):
            transition_counter[pair]['both+1'] += 1
        if (a-1, b-1) in itertools.combinations(sorted(next_draw), 2):
            transition_counter[pair]['both-1'] += 1

print("\nTransition patterns for top 20 pairs (next draw):")
top_pairs = [p for p, _ in pair_count.most_common(20)]
for pair in top_pairs:
    trans = transition_counter.get(pair, {})
    if not trans:
        continue
    total = sum(trans.values())
    if total == 0:
        continue
    print(f"\nPair ({pair[0]}, {pair[1]}) – occurred {pair_count[pair]} times, followed by:")
    for key, val in trans.items():
        print(f"  {key}: {val} times ({val/total*100:.1f}%)")

print("\nDone.")