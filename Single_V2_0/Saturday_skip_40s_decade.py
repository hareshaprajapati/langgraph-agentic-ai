import pandas as pd
import re

# =====================
# CONFIG
# =====================
CSV_PATH = "cross_lotto_data_backup.csv"   # <- change to your CSV filename

# =====================
# LOAD CSV
# =====================
df = pd.read_csv(CSV_PATH)

# Auto-detect the Others column
date_col = "Date"
others_col = [c for c in df.columns if c.lower().startswith("others")][0]

# =====================
# PARSE DATES AND FILTER SATURDAYS
# =====================
df["_date"] = pd.to_datetime(df[date_col], format="%a %d-%b-%Y", errors="coerce")

sat = df[df["_date"].dt.weekday == 5].copy()
sat = sat.dropna(subset=["_date"]).sort_values("_date")

# =====================
# PARSE MAIN NUMBERS FROM OTHERS COLUMN
# =====================
def parse_main_numbers(text):
    """
    Extracts the first bracketed list from the Others column.
    Example:
      "[9, 11, 20, 36, 37, 45], [8, 29]"
    Returns:
      [9, 11, 20, 36, 37, 45]
    """
    if pd.isna(text):
        return []

    text = str(text)
    match = re.search(r"\[([0-9,\s]+)\]", text)

    if not match:
        return []

    numbers = match.group(1).split(",")
    return [int(n.strip()) for n in numbers if n.strip().isdigit()]

sat["main_numbers"] = sat[others_col].apply(parse_main_numbers)

# Drop any Saturday rows where the Others column could not be parsed
sat = sat[sat["main_numbers"].apply(len) > 0]

# =====================
# DEFINE SKIP vs KEEP
# =====================
def skips_40_45(nums):
    """Return True if NO main number is between 40 and 45 inclusive."""
    return not any(40 <= n <= 45 for n in nums)

sat["skips_40_45"] = sat["main_numbers"].apply(skips_40_45).astype(bool)

total = len(sat)
skip_count = int(sat["skips_40_45"].sum())
keep_count = total - skip_count

print("=" * 60)
print("Saturday Lotto main numbers: 1-39 only vs includes 40-45")
print("=" * 60)
print(f"Total Saturdays parsed       : {total}")
print(f"Skip 40-45  (1-39 only)     : {skip_count}  ({skip_count / total:.1%})")
print(f"Keep 40-45  (has 40-45)     : {keep_count}  ({keep_count / total:.1%})")
print()

if skip_count > keep_count:
    print("Result: More Saturdays skipped 40-45.")
else:
    print("Result: More Saturdays included at least one number from 40-45.")
print()

# =====================
# TRANSITION PROBABILITIES
# =====================
sat["prev_skips"] = sat["skips_40_45"].shift(1)
trans = sat.dropna(subset=["prev_skips"]).copy()

# Convert to boolean to avoid object dtype problems with ~
trans["prev_skips"] = trans["prev_skips"].astype(bool)
trans["skips_40_45"] = trans["skips_40_45"].astype(bool)

p_skip_after_skip = trans[trans["prev_skips"]]["skips_40_45"].mean()
p_keep_after_skip = 1 - p_skip_after_skip

p_skip_after_keep = trans[~trans["prev_skips"]]["skips_40_45"].mean()
p_keep_after_keep = 1 - p_skip_after_keep

print("=" * 60)
print("Transition probabilities")
print("=" * 60)
print(f"P(next skip  | previous skip ) : {p_skip_after_skip:.1%}")
print(f"P(next 40-45 | previous skip ) : {p_keep_after_skip:.1%}")
print(f"P(next skip  | previous 40-45) : {p_skip_after_keep:.1%}")
print(f"P(next 40-45 | previous 40-45) : {p_keep_after_keep:.1%}")
print()

# =====================
# CURRENT STREAK AND PREDICTION
# =====================
state_values = sat["skips_40_45"].astype(int).tolist()

run_lengths = []
previous_state = None
current_run = 0

for state in state_values:
    if state == previous_state:
        current_run += 1
    else:
        current_run = 1

    run_lengths.append(current_run)
    previous_state = state

sat["run_len"] = run_lengths

last_row = sat.iloc[-1]
last_date = last_row["_date"].strftime("%a %d-%b-%Y")
current_state = "skip (1-39 only)" if last_row["skips_40_45"] else "keep (has 40-45)"
current_run_len = int(last_row["run_len"])

print("=" * 60)
print("Current state")
print("=" * 60)
print(f"Last Saturday in data        : {last_date}")
print(f"Current state                : {current_state}")
print(f"Current streak length        : {current_run_len}")
print()

# If currently in a skip streak, show probability of continuing
if last_row["skips_40_45"]:
    sat["prev_run_len"] = sat["run_len"].shift(1)
    sat["prev_skips_state"] = sat["skips_40_45"].shift(1)

    # Ensure boolean
    sat["prev_skips_state"] = sat["prev_skips_state"].astype("boolean")

    skip_streak_df = sat[sat["prev_skips_state"] == True].copy()

    if not skip_streak_df.empty:
        streak_table = skip_streak_df.groupby("prev_run_len")["skips_40_45"].agg(
            count="count",
            prob_next_skip="mean"
        )
        streak_table["prob_40_45_returns"] = 1 - streak_table["prob_next_skip"]

        print("=" * 60)
        print("Probability of another skip by current skip streak length")
        print("=" * 60)
        print(streak_table.to_string(float_format=lambda x: f"{x:.1%}"))
        print()

        if current_run_len in streak_table.index:
            p = streak_table.loc[current_run_len, "prob_next_skip"]
            print(f"Prediction for next Saturday using current skip streak length {current_run_len}:")
            print(f"  P(skip again, 1-39 only) = {p:.1%}")
            print(f"  P(40-45 returns)         = {1 - p:.1%}")