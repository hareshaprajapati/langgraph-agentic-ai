import pandas as pd
from collections import Counter

# ================= CONFIG =================
MODE = "predict"
PREDICT_DATE = "Sat 22-Aug-2026"

SATURDAY_FILE = "Saturday_data.csv"
CROSS_FILE = "cross_lotto_data_backup.csv"

# ================= HELPERS =================
def dec(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

def parse_main_str(s):
    if pd.isna(s) or s == "":
        return []
    return [int(x.strip()) for x in s.split(',') if x.strip().isdigit()]

def parse_all_ints(s):
    if pd.isna(s) or s == "":
        return []
    return [int(x) for x in str(s).replace('[',' ').replace(']',' ').replace(',',' ').split() if x.isdigit()]

def get_others_col(df):
    for col in df.columns:
        if "Others" in col:
            return col
    raise KeyError("No Others column")

# ================= LOAD DATA =================
sat_df = pd.read_csv(SATURDAY_FILE)
sat_df["Date_dt"] = pd.to_datetime(sat_df["Date"], format="%a %d-%b-%Y", errors="coerce")
sat_df["nums"] = sat_df["Main"].apply(parse_main_str)
sat_df = sat_df[["Date_dt", "nums"]].dropna(subset=["nums"]).sort_values("Date_dt")

cross_df = pd.read_csv(CROSS_FILE)
cross_df["Date_dt"] = pd.to_datetime(cross_df["Date"], format="%a %d-%b-%Y", errors="coerce")
cross_df = cross_df.sort_values("Date_dt")
others_col = get_others_col(cross_df)

no40_df = sat_df[sat_df["nums"].apply(lambda nums: all(dec(n) != "40s" for n in nums))].copy().sort_values("Date_dt")

# ================= TIERS =================
def get_tiers_saturday_to_friday(target_date):
    prev_sat = cross_df[(cross_df["Date_dt"] < target_date) & (cross_df["Date"].str.startswith("Sat"))].tail(1)
    if prev_sat.empty:
        return None

    start_dt = prev_sat.iloc[0]["Date_dt"]
    window = cross_df[(cross_df["Date_dt"] >= start_dt) & (cross_df["Date_dt"] < target_date)]

    pool = []
    for _, row in window.iterrows():
        pool.extend(parse_all_ints(row["Set for Life (incl supp)"]))
        pool.extend(parse_all_ints(row[others_col]))

    counts = Counter([n for n in pool if 1 <= n <= 45])

    EH = set(n for n, c in counts.items() if c >= 4 and dec(n) != "40s")
    H  = set(n for n, c in counts.items() if c == 3 and dec(n) != "40s")
    W  = set(n for n, c in counts.items() if 1 <= c <= 2 and dec(n) != "40s")
    C  = set(n for n in range(1, 46) if dec(n) != "40s" and n not in counts)

    return EH, H, W, C

# ================= FEATURE EXTRACTION =================
def compute_expanded_features(target_date, prior_sat):
    windows = [5, 10, 20, 50, 100]
    freq = {w: Counter() for w in windows}
    for w in windows:
        for nums in prior_sat.tail(w)['nums']:
            freq[w].update(nums)

    gap = {}
    last_seen = {}
    all_prior = list(prior_sat['nums'])
    for idx, nums in enumerate(all_prior):
        for n in nums:
            last_seen[n] = idx
    max_idx = len(all_prior) - 1
    for n in range(1, 46):
        gap[n] = max_idx - last_seen.get(n, -1)

    pos_counts = {pos: Counter() for pos in range(1, 7)}
    prior_no40 = prior_sat[prior_sat["nums"].apply(lambda nums: all(dec(n) != "40s" for n in nums))]
    for _, row in prior_no40.tail(300).iterrows():
        sorted_nums = sorted(row["nums"])
        for pos, n in enumerate(sorted_nums, 1):
            pos_counts[pos][n] += 1
    position_score = Counter()
    for pos in pos_counts:
        for n, c in pos_counts[pos].items():
            position_score[n] += c

    cross_total_1w = Counter()
    cross_sfl_1w = Counter()
    cross_other_1w = Counter()
    prev_sat_cross = cross_df[(cross_df["Date_dt"] < target_date) & (cross_df["Date"].str.startswith("Sat"))].tail(1)
    if not prev_sat_cross.empty:
        start_dt = prev_sat_cross.iloc[0]["Date_dt"]
        window = cross_df[(cross_df["Date_dt"] >= start_dt) & (cross_df["Date_dt"] < target_date)]
        for _, row in window.iterrows():
            sfl_nums = parse_all_ints(row["Set for Life (incl supp)"])
            other_nums = parse_all_ints(row[others_col])
            for n in sfl_nums:
                if 1 <= n <= 45:
                    cross_sfl_1w[n] += 1
                    cross_total_1w[n] += 1
            for n in other_nums:
                if 1 <= n <= 45:
                    cross_other_1w[n] += 1
                    cross_total_1w[n] += 1

    cross_total_14d = Counter()
    cross_total_28d = Counter()
    start_14d = target_date - pd.Timedelta(days=14)
    start_28d = target_date - pd.Timedelta(days=28)
    window_14d = cross_df[(cross_df["Date_dt"] >= start_14d) & (cross_df["Date_dt"] < target_date)]
    window_28d = cross_df[(cross_df["Date_dt"] >= start_28d) & (cross_df["Date_dt"] < target_date)]

    for _, row in window_14d.iterrows():
        for n in parse_all_ints(row["Set for Life (incl supp)"]):
            if 1 <= n <= 45:
                cross_total_14d[n] += 1
        for n in parse_all_ints(row[others_col]):
            if 1 <= n <= 45:
                cross_total_14d[n] += 1

    for _, row in window_28d.iterrows():
        for n in parse_all_ints(row["Set for Life (incl supp)"]):
            if 1 <= n <= 45:
                cross_total_28d[n] += 1
        for n in parse_all_ints(row[others_col]):
            if 1 <= n <= 45:
                cross_total_28d[n] += 1

    no40_prior = no40_df[no40_df["Date_dt"] < target_date]
    freq_no40_10 = Counter()
    freq_no40_20 = Counter()
    freq_no40_50 = Counter()
    for nums in no40_prior.tail(10)['nums']:
        freq_no40_10.update(nums)
    for nums in no40_prior.tail(20)['nums']:
        freq_no40_20.update(nums)
    for nums in no40_prior.tail(50)['nums']:
        freq_no40_50.update(nums)

    gap_no40 = {}
    last_seen_no40 = {}
    no40_nums_list = list(no40_prior['nums'])
    for idx, nums in enumerate(no40_nums_list):
        for n in nums:
            last_seen_no40[n] = idx
    max_idx_no40 = len(no40_nums_list) - 1
    for n in range(1, 40):
        gap_no40[n] = max_idx_no40 - last_seen_no40.get(n, -1)

    last_digit_freq = Counter()
    for nums in prior_sat.tail(50)['nums']:
        for n in nums:
            last_digit_freq[n % 10] += 1

    decade_freq = Counter()
    for nums in prior_sat.tail(20)['nums']:
        for n in nums:
            decade_freq[dec(n)] += 1

    last_draw_nums = prior_sat.iloc[-1]['nums'] if len(prior_sat) > 0 else []

    return {
        'freq5': freq[5], 'freq10': freq[10], 'freq20': freq[20],
        'freq50': freq[50], 'freq100': freq[100],
        'gap': gap, 'pos_score': position_score,
        'cross_total_1w': cross_total_1w,
        'cross_sfl_1w': cross_sfl_1w,
        'cross_other_1w': cross_other_1w,
        'cross_total_14d': cross_total_14d,
        'cross_total_28d': cross_total_28d,
        'freq_no40_10': freq_no40_10,
        'freq_no40_20': freq_no40_20,
        'freq_no40_50': freq_no40_50,
        'gap_no40': gap_no40,
        'last_digit_freq': last_digit_freq,
        'decade_freq': decade_freq,
        'last_draw_nums': last_draw_nums
    }

# ================= POOL BUILDER =================
def build_pool_general_with_tiers(
    base_score, gap, last_draw_nums, eligible,
    caps, hot_count, medium_count, cold_count,
    ld_cap, max_prev, max_run, odd_even_cap,
    tier_of, tier_caps
):
    hot_sorted = sorted(eligible, key=base_score, reverse=True)
    cold_sorted = sorted(eligible, key=lambda n: gap.get(n, 0), reverse=True)

    hot_picks = hot_sorted[:hot_count]
    hot_set = set(hot_picks)

    cold_picks = []
    for n in cold_sorted:
        if n in hot_set:
            continue
        cold_picks.append(n)
        if len(cold_picks) == cold_count:
            break

    selected_set = hot_set | set(cold_picks)

    medium_picks = []
    for n in hot_sorted:
        if n in selected_set:
            continue
        medium_picks.append(n)
        if len(medium_picks) == medium_count:
            break

    priority = hot_picks + medium_picks + cold_picks

    pool = []
    pool_set = set()
    decade_counts = Counter()
    ld_counts = Counter()
    tier_counts = Counter()
    odd_count = 0
    even_count = 0
    prev_total = 0

    def run_len_if_add(n):
        if max_run is None:
            return 0
        s = set(pool_set)
        s.add(n)
        left = n - 1
        right = n + 1
        run = 1
        while left in s:
            run += 1
            left -= 1
        while right in s:
            run += 1
            right += 1
        return run

    def can_add(n):
        nonlocal prev_total

        if n in pool_set:
            return False
        if n in last_draw_nums and prev_total >= max_prev:
            return False
        if n % 2 == 1 and odd_count >= odd_even_cap:
            return False
        if n % 2 == 0 and even_count >= odd_even_cap:
            return False
        if decade_counts[dec(n)] >= caps.get(dec(n), 4):
            return False
        if ld_counts[n % 10] >= ld_cap:
            return False
        if max_run is not None and run_len_if_add(n) > max_run:
            return False

        tier = tier_of(n)
        if tier_counts[tier] >= tier_caps.get(tier, 99):
            return False

        return True

    def add(n):
        nonlocal odd_count, even_count, prev_total

        pool.append(n)
        pool_set.add(n)
        decade_counts[dec(n)] += 1
        ld_counts[n % 10] += 1
        tier_counts[tier_of(n)] += 1

        if n % 2 == 1:
            odd_count += 1
        else:
            even_count += 1

        if n in last_draw_nums:
            prev_total += 1

    for n in priority:
        if len(pool) >= 15:
            break
        if can_add(n):
            add(n)

    for n in hot_sorted:
        if len(pool) >= 15:
            break
        if n in pool_set:
            continue
        if can_add(n):
            add(n)

    for n in cold_sorted:
        if len(pool) >= 15:
            break
        if n in pool_set:
            continue
        if can_add(n):
            add(n)

    if len(pool) < 15:
        for n in hot_sorted:
            if len(pool) >= 15:
                break
            if n not in pool_set:
                add(n)

    return sorted(pool)

# ================= BEST CONFIGURATION =================
BEST_WEIGHTS = {
    'freq5': 1.2532,
    'freq10': -0.2386,
    'freq20': 1.0384,
    'freq50': 2.3986,
    'freq100': -0.1908,
    'gap': 0.1080,
    'pos_score': -0.0095,
    'cross_total_1w': -0.3296,
    'cross_sfl_1w': 0.0729,
    'cross_other_1w': -0.1047,
    'cross_total_14d': -1.0839,
    'cross_total_28d': 0.7758,
    'freq_no40_10': 1.2269,
    'freq_no40_20': 0.4492,
    'freq_no40_50': 0.3940,
    'gap_no40': 0.9103,
    'last_digit_freq': -0.0609,
    'decade_freq': -0.2836,
    'tier_EH': -0.9942,
    'tier_H': 0.5878,
    'tier_W': 1.9972,
    'tier_C': -1.2885,
}

BEST_CAPS = {'0s': 6, '10s': 3, '20s': 3, '30s': 3}
BEST_OE = 8
BEST_PREV = 1
BEST_LD = 3
BEST_RUN = 3
BEST_HOT = 7
BEST_MED = 4
BEST_COLD = 6
BEST_TIER_CAPS = {'EH': 2, 'H': 3, 'W': 7, 'C': 1}

# ================= DIAGNOSTIC =================
def tier_of(n, t):
    if n in t[0]: return 'EH'
    if n in t[1]: return 'H'
    if n in t[2]: return 'W'
    return 'C'

def score_of(n, f, t):
    w = BEST_WEIGHTS
    return (
        w['freq5'] * f['freq5'].get(n, 0)
        + w['freq10'] * f['freq10'].get(n, 0)
        + w['freq20'] * f['freq20'].get(n, 0)
        + w['freq50'] * f['freq50'].get(n, 0)
        + w['freq100'] * f['freq100'].get(n, 0)
        + w['gap'] * f['gap'].get(n, 0)
        + w['pos_score'] * f['pos_score'].get(n, 0)
        + w['cross_total_1w'] * f['cross_total_1w'].get(n, 0)
        + w['cross_sfl_1w'] * f['cross_sfl_1w'].get(n, 0)
        + w['cross_other_1w'] * f['cross_other_1w'].get(n, 0)
        + w['cross_total_14d'] * f['cross_total_14d'].get(n, 0)
        + w['cross_total_28d'] * f['cross_total_28d'].get(n, 0)
        + w['freq_no40_10'] * f['freq_no40_10'].get(n, 0)
        + w['freq_no40_20'] * f['freq_no40_20'].get(n, 0)
        + w['freq_no40_50'] * f['freq_no40_50'].get(n, 0)
        + w['gap_no40'] * f['gap_no40'].get(n, 0)
        + w['last_digit_freq'] * f['last_digit_freq'].get(n % 10, 0)
        + w['decade_freq'] * f['decade_freq'].get(dec(n), 0)
        + (w['tier_EH'] if n in t[0] else 0)
        + (w['tier_H'] if n in t[1] else 0)
        + (w['tier_W'] if n in t[2] else 0)
        + (w['tier_C'] if n in t[3] else 0)
    )

# ================= RUN DIAGNOSTIC =================
target_date = pd.to_datetime(PREDICT_DATE, format="%a %d-%b-%Y")
prior_sat = sat_df[sat_df["Date_dt"] < target_date]
features = compute_expanded_features(target_date, prior_sat)
tiers = get_tiers_saturday_to_friday(target_date)

eligible = [n for n in range(1, 46) if dec(n) != "40s"]

pool = build_pool_general_with_tiers(
    lambda n: score_of(n, features, tiers),
    features['gap'],
    features['last_draw_nums'],
    eligible,
    BEST_CAPS,
    BEST_HOT,
    BEST_MED,
    BEST_COLD,
    BEST_LD,
    BEST_PREV,
    BEST_RUN,
    BEST_OE,
    lambda n: tier_of(n, tiers),
    BEST_TIER_CAPS
)

actual_winning = {1, 2, 9, 11, 13, 20}
captured = set(pool) & actual_winning
missed = actual_winning - set(pool)

print("\n" + "="*100)
print(f"DIAGNOSTIC ANALYSIS FOR {PREDICT_DATE}")
print("="*100)

print(f"\nFinal pool:")
print(pool)

print(f"\nActual winning numbers : {sorted(actual_winning)}")
print(f"Captured winning numbers : {sorted(captured)}")
print(f"Missed winning numbers   : {sorted(missed)}")

# Tier counts in pool
tier_counts = Counter(tier_of(n, tiers) for n in pool)
print(f"\nTier counts in pool: EH={tier_counts['EH']}, H={tier_counts['H']}, W={tier_counts['W']}, C={tier_counts['C']}")

# All numbers table
rows = []
for n in range(1, 40):
    rows.append({
        'n': n,
        'decade': dec(n),
        'score': score_of(n, features, tiers),
        'tier': tier_of(n, tiers),
        'freq_no40_10': features['freq_no40_10'].get(n, 0),
        'freq_no40_20': features['freq_no40_20'].get(n, 0),
        'gap_no40': features['gap_no40'].get(n, 0),
        'cross_14d': features['cross_total_14d'].get(n, 0),
        'in_pool': n in pool,
        'won': n in actual_winning,
    })

df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
df.insert(0, 'rank', df.index + 1)

print("\nTop 25 numbers by legacy score:")
print(df.head(25)[['rank','n','decade','score','tier','freq_no40_10','freq_no40_20','gap_no40','cross_14d','in_pool','won']].to_string(index=False))

print("\nWinning numbers detail:")
win_df = df[df['won']].sort_values('score', ascending=False)
print(win_df[['rank','n','decade','score','tier','freq_no40_10','freq_no40_20','gap_no40','cross_14d','in_pool']].to_string(index=False))

print("\nEH numbers sorted by score:")
eh_df = df[df['tier'] == 'EH'].sort_values('score', ascending=False)
print(eh_df[['rank','n','decade','score','freq_no40_10','freq_no40_20','gap_no40','cross_14d','in_pool','won']].to_string(index=False))

print("\nH numbers sorted by score:")
h_df = df[df['tier'] == 'H'].sort_values('score', ascending=False)
print(h_df[['rank','n','decade','score','in_pool','won']].to_string(index=False))

print("\nW numbers sorted by score:")
w_df = df[df['tier'] == 'W'].sort_values('score', ascending=False)
print(w_df[['rank','n','decade','score','in_pool','won']].to_string(index=False))

print("\nC numbers sorted by score:")
c_df = df[df['tier'] == 'C'].sort_values('score', ascending=False)
print(c_df[['rank','n','decade','score','in_pool','won']].to_string(index=False))