import pandas as pd
from collections import Counter
import itertools

# ================= CONFIGURATION =================
MODE = "tune"                # set to "tune" to run sweep, or "backtest" for single config
NO_OF_BACKTEST_DRAWS = 10
POOL_SIZE = 15
SATURDAY_FILE = "Saturday_data.csv"
CROSS_FILE = "cross_lotto_data_backup.csv"

# ================= HELPERS (same as before, compact) =================
def dec(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

def parse_main_str(s):
    if pd.isna(s) or s == "": return []
    return [int(x.strip()) for x in s.split(',') if x.strip().isdigit()]

def parse_all_ints(s):
    if pd.isna(s) or s == "": return []
    return [int(x) for x in str(s).replace('[',' ').replace(']',' ').replace(',',' ').split() if x.isdigit()]

def get_others_col(df):
    for col in df.columns:
        if "Others" in col: return col
    raise KeyError("No Others column")

sat_df = pd.read_csv(SATURDAY_FILE)
sat_df["Date_dt"] = pd.to_datetime(sat_df["Date"], format="%a %d-%b-%Y", errors="coerce")
sat_df["nums"] = sat_df["Main"].apply(parse_main_str)
sat_df = sat_df[["Date_dt", "nums"]].dropna(subset=["nums"]).sort_values("Date_dt")

cross_df = pd.read_csv(CROSS_FILE)
cross_df["Date_dt"] = pd.to_datetime(cross_df["Date"], format="%a %d-%b-%Y", errors="coerce")
cross_df = cross_df.sort_values("Date_dt")
others_col = get_others_col(cross_df)

no40_df = sat_df[sat_df["nums"].apply(lambda nums: all(dec(n) != "40s" for n in nums))].copy().sort_values("Date_dt")

def get_tiers_saturday_to_friday(target_date):
    prev_sat = cross_df[(cross_df["Date_dt"] < target_date) & (cross_df["Date"].str.startswith("Sat"))].tail(1)
    if prev_sat.empty: return None
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
    pos_score = Counter()
    for pos in pos_counts:
        for n, c in pos_counts[pos].items():
            pos_score[n] += c
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
            if 1 <= n <= 45: cross_total_14d[n] += 1
        for n in parse_all_ints(row[others_col]):
            if 1 <= n <= 45: cross_total_14d[n] += 1
    for _, row in window_28d.iterrows():
        for n in parse_all_ints(row["Set for Life (incl supp)"]):
            if 1 <= n <= 45: cross_total_28d[n] += 1
        for n in parse_all_ints(row[others_col]):
            if 1 <= n <= 45: cross_total_28d[n] += 1
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
        'gap': gap, 'pos_score': pos_score,
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

# ================= DYNAMIC WEIGHTS =================
BASE_WEIGHTS = {
    'freq5': 1.2532, 'freq10': -0.2386, 'freq20': 1.0384,
    'freq50': 2.3986, 'freq100': -0.1908,
    'pos_score': -0.0095, 'cross_total_1w': -0.3296,
    'cross_sfl_1w': 0.0729, 'cross_other_1w': -0.1047,
    'cross_total_14d': -1.0839, 'cross_total_28d': 0.7758,
    'freq_no40_10': 1.2269, 'freq_no40_20': 0.4492,
    'freq_no40_50': 0.3940,
    'last_digit_freq': -0.0609, 'decade_freq': -0.2836,
    'tier_EH': -0.9942, 'tier_H': 0.5878,
    'tier_W': 1.9972, 'tier_C': -1.2885,
}

# We will vary gap and gap_no40 weights, and vote weight scaling.
# All other weights stay as BASE_WEIGHTS.

def make_legacy_score(gap_w, gap_no40_w):
    def score(n, f, t):
        return (BASE_WEIGHTS['freq5'] * f['freq5'].get(n, 0)
                + BASE_WEIGHTS['freq10'] * f['freq10'].get(n, 0)
                + BASE_WEIGHTS['freq20'] * f['freq20'].get(n, 0)
                + BASE_WEIGHTS['freq50'] * f['freq50'].get(n, 0)
                + BASE_WEIGHTS['freq100'] * f['freq100'].get(n, 0)
                + gap_w * f['gap'].get(n, 0)
                + BASE_WEIGHTS['pos_score'] * f['pos_score'].get(n, 0)
                + BASE_WEIGHTS['cross_total_1w'] * f['cross_total_1w'].get(n, 0)
                + BASE_WEIGHTS['cross_sfl_1w'] * f['cross_sfl_1w'].get(n, 0)
                + BASE_WEIGHTS['cross_other_1w'] * f['cross_other_1w'].get(n, 0)
                + BASE_WEIGHTS['cross_total_14d'] * f['cross_total_14d'].get(n, 0)
                + BASE_WEIGHTS['cross_total_28d'] * f['cross_total_28d'].get(n, 0)
                + BASE_WEIGHTS['freq_no40_10'] * f['freq_no40_10'].get(n, 0)
                + BASE_WEIGHTS['freq_no40_20'] * f['freq_no40_20'].get(n, 0)
                + BASE_WEIGHTS['freq_no40_50'] * f['freq_no40_50'].get(n, 0)
                + gap_no40_w * f['gap_no40'].get(n, 0)
                + BASE_WEIGHTS['last_digit_freq'] * f['last_digit_freq'].get(n % 10, 0)
                + BASE_WEIGHTS['decade_freq'] * f['decade_freq'].get(dec(n), 0)
                + (BASE_WEIGHTS['tier_EH'] if n in t[0] else 0)
                + (BASE_WEIGHTS['tier_H'] if n in t[1] else 0)
                + (BASE_WEIGHTS['tier_W'] if n in t[2] else 0)
                + (BASE_WEIGHTS['tier_C'] if n in t[3] else 0))
    return score

# Pool builder (original with fallback)
def build_pool(score_func, f, t, last_draw_nums, caps, max_prev):
    eligible = [n for n in range(1, 46) if dec(n) != "40s"]
    hot_sorted = sorted(eligible, key=score_func, reverse=True)
    cold_sorted = sorted(eligible, key=lambda n: f['gap'].get(n, 0), reverse=True)
    hot_picks = hot_sorted[:7]
    hot_set = set(hot_picks)
    cold_picks = []
    for n in cold_sorted:
        if n in hot_set: continue
        cold_picks.append(n)
        if len(cold_picks) == 6: break
    selected_set = hot_set | set(cold_picks)
    medium_picks = []
    for n in hot_sorted:
        if n in selected_set: continue
        medium_picks.append(n)
        if len(medium_picks) == 4: break
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
        s = set(pool_set)
        s.add(n)
        left = n - 1
        right = n + 1
        r = 1
        while left in s:
            r += 1
            left -= 1
        while right in s:
            r += 1
            right += 1
        return r
    def can_add(n):
        nonlocal prev_total
        if n in pool_set: return False
        if n in last_draw_nums and prev_total >= max_prev: return False
        if n % 2 == 1 and odd_count >= 8: return False
        if n % 2 == 0 and even_count >= 8: return False
        if decade_counts[dec(n)] >= caps.get(dec(n), 4): return False
        if ld_counts[n % 10] >= 3: return False
        if run_len_if_add(n) > 3: return False
        tier = 'EH' if n in t[0] else 'H' if n in t[1] else 'W' if n in t[2] else 'C'
        if tier_counts[tier] >= {'EH':2,'H':3,'W':7,'C':1}.get(tier, 99): return False
        return True
    def add(n):
        nonlocal odd_count, even_count, prev_total
        pool.append(n)
        pool_set.add(n)
        decade_counts[dec(n)] += 1
        ld_counts[n % 10] += 1
        tier = 'EH' if n in t[0] else 'H' if n in t[1] else 'W' if n in t[2] else 'C'
        tier_counts[tier] += 1
        if n % 2 == 1: odd_count += 1
        else: even_count += 1
        if n in last_draw_nums: prev_total += 1
    for n in priority:
        if len(pool) >= POOL_SIZE: break
        if can_add(n): add(n)
    for n in hot_sorted:
        if len(pool) >= POOL_SIZE: break
        if n in pool_set: continue
        if can_add(n): add(n)
    for n in cold_sorted:
        if len(pool) >= POOL_SIZE: break
        if n in pool_set: continue
        if can_add(n): add(n)
    if len(pool) < POOL_SIZE:
        for n in hot_sorted:
            if len(pool) >= POOL_SIZE: break
            if n not in pool_set: add(n)
    return sorted(pool)

# Build 3-legacy vote with given score function
def build_3_legacy_vote(score_func, f, t, last_draw, vote_weight=100.0):
    # Bind f and t to the score function so it only takes n
    bound_score = lambda n: score_func(n, f, t)
    caps_std = {'0s':6, '10s':3, '20s':3, '30s':3}
    pool1 = build_pool(bound_score, f, t, last_draw, caps_std, max_prev=6)
    pool2 = build_pool(bound_score, f, t, last_draw, caps_std, max_prev=2)
    caps_search = {'0s':5, '10s':4, '20s':3, '30s':2}
    pool3 = build_pool(bound_score, f, t, last_draw, caps_search, max_prev=2)
    freq = Counter()
    for p in [pool1, pool2, pool3]:
        freq.update(p)
    def final_score(n):
        return vote_weight * freq.get(n, 0) + bound_score(n)
    final_pool = build_pool(final_score, f, t, last_draw, caps_std, max_prev=2)
    return final_pool

# Precompute cache
def precompute_cache():
    cache = []
    for _, row in no40_df.iterrows():
        target_date = row["Date_dt"]
        real_nums = set(row["nums"])
        prior_sat = sat_df[sat_df["Date_dt"] < target_date]
        if prior_sat.empty: continue
        features = compute_expanded_features(target_date, prior_sat)
        tiers = get_tiers_saturday_to_friday(target_date)
        if tiers is None: continue
        cache.append({'date': target_date, 'real_nums': real_nums,
                      'features': features, 'tiers': tiers})
    return cache

def run_config(gap_w, gap_no40_w, vote_weight, cache, n_test=10):
    score_func = make_legacy_score(gap_w, gap_no40_w)
    target_cache = cache[-n_test:]
    total_five = 0
    total_cov = 0
    for entry in target_cache:
        f = entry['features']
        t = entry['tiers']
        last_draw = f['last_draw_nums']
        pool = build_3_legacy_vote(score_func, f, t, last_draw, vote_weight)
        captured = set(pool) & entry['real_nums']
        cov = len(captured)
        total_cov += cov
        if cov >= 5:
            total_five += 1
    return total_five, total_cov / n_test

# ================= MAIN TUNING =================
if __name__ == "__main__":
    cache = precompute_cache()
    print(f"Cached {len(cache)} draws. Tuning on last 10...")
    # Parameter grid
    gap_ws = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    gap_no40_ws = [0.5, 0.9, 1.2, 1.5, 2.0]
    vote_weights = [50, 80, 100, 120, 150]
    best = (0, None)
    for gw, gnw, vw in itertools.product(gap_ws, gap_no40_ws, vote_weights):
        five, avg = run_config(gw, gnw, vw, cache)
        if five > best[0]:
            best = (five, (gw, gnw, vw, avg))
        print(f"gap_w={gw}, gap_no40_w={gnw}, vote_w={vw} -> 5+ traps={five}, avg_cov={avg:.2f}")
    print("\nBest config:")
    print(f"5+ traps={best[0]}, params={best[1]}")