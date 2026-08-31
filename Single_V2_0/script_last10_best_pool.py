import pandas as pd
from collections import Counter
import random

# ================= CONFIG =================
SATURDAY_FILE = "Saturday_data.csv"
N_ITERATIONS = 20000
RANDOM_SEED = 42

# ================= HELPERS =================
def dec(n):
    if n <= 9:
        return '0s'
    if n <= 19:
        return '10s'
    if n <= 29:
        return '20s'
    if n <= 39:
        return '30s'
    return '40s'

def parse_main_str(s):
    if pd.isna(s) or s == "":
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip().isdigit()]

# ================= LOAD DATA =================
sat_df = pd.read_csv(SATURDAY_FILE)
sat_df["Date_dt"] = pd.to_datetime(sat_df["Date"], format="%a %d-%b-%Y", errors="coerce")
sat_df["nums"] = sat_df["Main"].apply(parse_main_str)
sat_df = sat_df[["Date_dt", "nums"]].dropna(subset=["nums"]).sort_values("Date_dt")

no40_df = sat_df[sat_df["nums"].apply(lambda nums: all(dec(n) != "40s" for n in nums))].copy().sort_values("Date_dt")

# ================= LAST 10 DRAWS =================
last10 = no40_df.tail(10).reset_index(drop=True)

print(f"Last 10 no-40 Saturday draws:\n")
for i, row in last10.iterrows():
    print(f"{i+1:2d}. {pd.to_datetime(row['Date_dt']).strftime('%d-%b-%Y')} -> {sorted(row['nums'])}")

# ================= FITNESS =================
def count_six(pool, draws_df):
    pool_set = set(pool)
    count = 0
    for nums in draws_df["nums"]:
        if set(nums).issubset(pool_set):
            count += 1
    return count

def count_five_plus(pool, draws_df):
    pool_set = set(pool)
    count = 0
    for nums in draws_df["nums"]:
        if len(set(nums) & pool_set) >= 5:
            count += 1
    return count

# ================= SEARCH =================
random.seed(RANDOM_SEED)

# Initial pool = top 15 frequent numbers in last 10 draws
freq = Counter()
for nums in last10["nums"]:
    freq.update(nums)

initial_pool = [n for n, _ in freq.most_common(15)]
best_pool = initial_pool.copy()
best_six = count_six(best_pool, last10)

print(f"\nInitial pool: {sorted(best_pool)}")
print(f"Initial 6/6 count in last 10: {best_six}")

# Random mutation
for it in range(N_ITERATIONS):
    candidate = best_pool.copy()

    # Swap 1 or 2 numbers
    for _ in range(random.randint(1, 2)):
        if candidate:
            remove_n = random.choice(candidate)
            add_n = random.choice([n for n in range(1, 40) if n not in candidate])
            candidate.remove(remove_n)
            candidate.append(add_n)

    cand_six = count_six(candidate, last10)

    if cand_six > best_six:
        best_six = cand_six
        best_pool = candidate.copy()

        if it % 1000 == 0:
            print(f"Iter {it}: best_6/6 = {best_six}, pool = {sorted(best_pool)}")

best_pool = sorted(best_pool)

# ================= RESULTS =================
print("\n" + "=" * 70)
print("BEST FIXED 15-NUMBER POOL FOR LAST 10 DRAWS")
print("=" * 70)
print(f"Pool: {best_pool}")

six_count = count_six(best_pool, last10)
five_count = count_five_plus(best_pool, last10)

print(f"\n6/6 traps in last 10: {six_count}")
print(f"5+ traps in last 10: {five_count}")

print("\nPer-draw capture:")
pool_set = set(best_pool)
for i, row in last10.iterrows():
    captured = set(row["nums"]) & pool_set
    cov = len(captured)
    print(f"{pd.to_datetime(row['Date_dt']).strftime('%d-%b-%Y')}: {cov}/6 -> {sorted(captured)}")