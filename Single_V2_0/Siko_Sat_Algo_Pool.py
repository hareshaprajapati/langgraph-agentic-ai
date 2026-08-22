import pandas as pd
from collections import Counter

# ================= CONFIGURATION =================
MODE = "backtest"          # "backtest" or "predict"
PREDICT_DATE = "Sat 22-Aug-2026"   # used only if MODE = "predict" 22-Aug-2026

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

# ================= EXPANDED FEATURE EXTRACTION =================
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
    last_10 = no40_prior.tail(10)
    last_20 = no40_prior.tail(20)
    last_50 = no40_prior.tail(50)
    for nums in last_10['nums']:
        freq_no40_10.update(nums)
    for nums in last_20['nums']:
        freq_no40_20.update(nums)
    for nums in last_50['nums']:
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

# ================= POOL BUILDER WITH TIER CAPS =================
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
    prev_counts = Counter()
    tier_counts = Counter()
    odd_count = 0
    even_count = 0

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
        if n in pool_set:
            return False
        if n in last_draw_nums and prev_counts[n] >= max_prev:
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
        nonlocal odd_count, even_count
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
            prev_counts[n] += 1

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

# ================= EVALUATE CONFIG =================
def evaluate_config(cache, weights, caps, oe, prev, ld, run, hot, med, cold, tier_caps):
    six = 0
    five = 0
    four = 0
    high_draws = []

    for entry in cache:
        f = entry['features']
        t = entry['tiers']

        def tier_of(n):
            if n in t[0]: return 'EH'
            if n in t[1]: return 'H'
            if n in t[2]: return 'W'
            return 'C'

        def score(n):
            return (
                weights['freq5'] * f['freq5'].get(n, 0)
                + weights['freq10'] * f['freq10'].get(n, 0)
                + weights['freq20'] * f['freq20'].get(n, 0)
                + weights['freq50'] * f['freq50'].get(n, 0)
                + weights['freq100'] * f['freq100'].get(n, 0)
                + weights['gap'] * f['gap'].get(n, 0)
                + weights['pos_score'] * f['pos_score'].get(n, 0)
                + weights['cross_total_1w'] * f['cross_total_1w'].get(n, 0)
                + weights['cross_sfl_1w'] * f['cross_sfl_1w'].get(n, 0)
                + weights['cross_other_1w'] * f['cross_other_1w'].get(n, 0)
                + weights['cross_total_14d'] * f['cross_total_14d'].get(n, 0)
                + weights['cross_total_28d'] * f['cross_total_28d'].get(n, 0)
                + weights['freq_no40_10'] * f['freq_no40_10'].get(n, 0)
                + weights['freq_no40_20'] * f['freq_no40_20'].get(n, 0)
                + weights['freq_no40_50'] * f['freq_no40_50'].get(n, 0)
                + weights['gap_no40'] * f['gap_no40'].get(n, 0)
                + weights['last_digit_freq'] * f['last_digit_freq'].get(n % 10, 0)
                + weights['decade_freq'] * f['decade_freq'].get(dec(n), 0)
                + (weights['tier_EH'] if n in t[0] else 0)
                + (weights['tier_H'] if n in t[1] else 0)
                + (weights['tier_W'] if n in t[2] else 0)
                + (weights['tier_C'] if n in t[3] else 0)
            )

        pool = build_pool_general_with_tiers(
            score, f['gap'], f['last_draw_nums'], entry['eligible'],
            caps, hot, med, cold, ld, prev, run, oe,
            tier_of, tier_caps
        )

        cov = len(set(pool) & entry['real_nums'])
        if cov >= 6:
            six += 1
        if cov >= 5:
            five += 1
            high_draws.append((entry['date'], sorted(set(pool) & entry['real_nums']), pool))
        if cov >= 4:
            four += 1

    return six, five, four, high_draws

# ================= BACKTEST MODE =================
def run_backtest():
    print("Precomputing features for last 20 no-40 draws...")
    test_draws = no40_df.tail(20)
    cache = []
    for _, target_row in test_draws.iterrows():
        target_date = target_row["Date_dt"]
        real_nums = set(target_row["nums"])
        prior_sat = sat_df[sat_df["Date_dt"] < target_date]
        features = compute_expanded_features(target_date, prior_sat)
        tiers = get_tiers_saturday_to_friday(target_date)
        eligible = [n for n in range(1, 46) if dec(n) != "40s"]
        cache.append({
            'date': target_date,
            'real_nums': real_nums,
            'eligible': eligible,
            'features': features,
            'tiers': tiers,
        })
        print(f"  cached {len(cache)}: {pd.to_datetime(target_date).strftime('%d-%b-%Y')}")

    print("\nEvaluating best configuration...")
    six, five, four, high_draws = evaluate_config(
        cache, BEST_WEIGHTS, BEST_CAPS, BEST_OE, BEST_PREV, BEST_LD,
        BEST_RUN, BEST_HOT, BEST_MED, BEST_COLD, BEST_TIER_CAPS
    )

    print("\n" + "=" * 70)
    print("BACKTEST RESULT (LAST 20 NO-40 DRAWS)")
    print("=" * 70)
    print(f"5+ traps : {five}/20")
    print(f"6/6 traps: {six}/20")
    print(f"4+ traps : {four}/20")

    if high_draws:
        print("\nHigh-capture draws (5+):")
        for date, cap, pool in high_draws:
            print(f"  {pd.to_datetime(date).strftime('%d-%b-%Y')}: {len(cap)}/6 -> {cap} | pool={pool}")

# ================= PREDICTION MODE =================
def predict_for_date(date_str):
    target_date = pd.to_datetime(date_str, format="%a %d-%b-%Y")
    print(f"Generating pool for {target_date.strftime('%d-%b-%Y')}...\n")

    prior_sat = sat_df[sat_df["Date_dt"] < target_date]
    if prior_sat.empty:
        print("No prior Saturday data available.")
        return

    features = compute_expanded_features(target_date, prior_sat)
    tiers = get_tiers_saturday_to_friday(target_date)
    if tiers is None:
        print("Could not compute tiers for this date.")
        return

    eligible = [n for n in range(1, 46) if dec(n) != "40s"]

    def tier_of(n):
        if n in tiers[0]: return 'EH'
        if n in tiers[1]: return 'H'
        if n in tiers[2]: return 'W'
        return 'C'

    def score(n):
        w = BEST_WEIGHTS
        f = features
        t = tiers
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

    pool = build_pool_general_with_tiers(
        score, features['gap'], features['last_draw_nums'], eligible,
        BEST_CAPS, BEST_HOT, BEST_MED, BEST_COLD,
        BEST_LD, BEST_PREV, BEST_RUN, BEST_OE,
        tier_of, BEST_TIER_CAPS
    )

    print("Predicted 15-number pool:")
    print(pool)

    # Show tier breakdown
    tier_counts = Counter()
    for n in pool:
        tier_counts[tier_of(n)] += 1
    print(f"\nTier breakdown: EH={tier_counts['EH']}, H={tier_counts['H']}, W={tier_counts['W']}, C={tier_counts['C']}")

# ================= MAIN =================
if __name__ == "__main__":
    if MODE == "backtest":
        run_backtest()
    elif MODE == "predict":
        predict_for_date(PREDICT_DATE)
    else:
        print("Invalid MODE. Use 'backtest' or 'predict'.")