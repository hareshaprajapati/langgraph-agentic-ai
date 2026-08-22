import pandas as pd
from collections import Counter
import random
import time

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

    # Greedy fill with all constraints
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

    # Final fallback to guarantee 15 numbers (bypass tier caps if necessary)
    if len(pool) < 15:
        for n in hot_sorted:
            if len(pool) >= 15:
                break
            if n not in pool_set:
                add(n)

    return sorted(pool)

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

# ================= CACHE LAST 20 =================
def precompute_cache_last20_expanded():
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
    return cache

# ================= EVALUATE CONFIG WITH TIER CAPS =================
def evaluate_config(cache, weights, caps, oe, prev, ld, run, hot, med, cold, tier_caps):
    six = 0
    five = 0
    four = 0
    high_draws = []
    draw_pools = {}

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

        draw_pools[entry['date']] = pool

    return six, five, four, high_draws, draw_pools

# ================= HILL CLIMB =================
def hill_climb(cache, init_weights, init_constraints, init_tier_caps, steps=300):
    current_weights = init_weights.copy()
    current_constraints = list(init_constraints)
    current_tier_caps = init_tier_caps.copy()
    current_six, current_five, current_four, _, _ = evaluate_config(
        cache, current_weights, *current_constraints, current_tier_caps
    )
    current_fitness = (current_five, current_six, current_four)

    best = {
        'weights': current_weights.copy(),
        'constraints': current_constraints.copy(),
        'tier_caps': current_tier_caps.copy(),
        'six': current_six,
        'five': current_five,
        'four': current_four,
        'fitness': current_fitness,
    }

    cap_options = [
        {'0s': 5, '10s': 4, '20s': 3, '30s': 3},
        {'0s': 5, '10s': 4, '20s': 4, '30s': 2},
        {'0s': 6, '10s': 4, '20s': 3, '30s': 2},
        {'0s': 5, '10s': 5, '20s': 3, '30s': 2},
        {'0s': 4, '10s': 5, '20s': 3, '30s': 3},
        {'0s': 6, '10s': 3, '20s': 3, '30s': 3},
    ]
    tier_cap_options = {
        'EH': [1, 2, 3],
        'H': [2, 3, 4],
        'W': [4, 5, 6, 7],
        'C': [0, 1, 2]
    }

    for step in range(steps):
        new_weights = current_weights.copy()
        for key in new_weights:
            if random.random() < 0.4:
                new_weights[key] += random.uniform(-0.3, 0.3)

        new_constraints = list(current_constraints)
        if random.random() < 0.2:
            new_constraints[0] = random.choice(cap_options)
        if random.random() < 0.2:
            new_constraints[1] = random.choice([7, 8, 9])
        if random.random() < 0.2:
            new_constraints[2] = random.choice([1, 2])
        if random.random() < 0.2:
            new_constraints[3] = random.choice([3, 4])
        if random.random() < 0.2:
            new_constraints[4] = random.choice([2, 3, None])
        if random.random() < 0.2:
            new_constraints[5] = random.choice([5, 6, 7, 8])
        if random.random() < 0.2:
            new_constraints[6] = random.choice([2, 3, 4])
        if random.random() < 0.2:
            new_constraints[7] = random.choice([4, 5, 6])

        new_tier_caps = current_tier_caps.copy()
        if random.random() < 0.3:
            tier_key = random.choice(list(tier_cap_options.keys()))
            new_tier_caps[tier_key] = random.choice(tier_cap_options[tier_key])

        new_six, new_five, new_four, new_high, new_pools = evaluate_config(
            cache, new_weights, *new_constraints, new_tier_caps
        )
        new_fitness = (new_five, new_six, new_four)

        if new_fitness > current_fitness:
            current_weights = new_weights
            current_constraints = new_constraints
            current_tier_caps = new_tier_caps
            current_six, current_five, current_four = new_six, new_five, new_four
            current_fitness = new_fitness

            if new_fitness > best['fitness']:
                best = {
                    'weights': current_weights.copy(),
                    'constraints': current_constraints.copy(),
                    'tier_caps': current_tier_caps.copy(),
                    'six': current_six,
                    'five': current_five,
                    'four': current_four,
                    'fitness': current_fitness,
                }

    final_six, final_five, final_four, final_high, final_pools = evaluate_config(
        cache, best['weights'], *best['constraints'], best['tier_caps']
    )
    best['high_draws'] = final_high
    best['draw_pools'] = final_pools
    return best

# ================= GENETIC ALGORITHM WITH TIER CAPS =================
def run_ga_tier_caps(pop_size=150, generations=150, mutation_rate=0.3, hill_steps=300):
    print("Precomputing expanded features for last 20 no-40 draws...")
    cache = precompute_cache_last20_expanded()
    print(f"Cached {len(cache)} draws.\n")

    weight_ranges = {
        'freq5': (-1.0, 1.5),
        'freq10': (-1.0, 1.5),
        'freq20': (-1.0, 2.0),
        'freq50': (-0.5, 2.5),
        'freq100': (-1.0, 1.0),
        'gap': (-0.5, 0.5),
        'pos_score': (-0.05, 0.05),
        'cross_total_1w': (-0.5, 1.0),
        'cross_sfl_1w': (-0.5, 0.5),
        'cross_other_1w': (-0.5, 0.5),
        'cross_total_14d': (-1.0, 1.5),
        'cross_total_28d': (-1.0, 1.5),
        'freq_no40_10': (-1.0, 2.0),
        'freq_no40_20': (-1.0, 2.0),
        'freq_no40_50': (-0.5, 2.5),
        'gap_no40': (-0.5, 1.0),
        'last_digit_freq': (-0.5, 0.5),
        'decade_freq': (-0.5, 0.5),
        'tier_EH': (-2.0, 0.5),
        'tier_H': (0.0, 2.0),
        'tier_W': (0.0, 2.0),
        'tier_C': (-2.0, 0.5),
    }

    cap_options = [
        {'0s': 5, '10s': 4, '20s': 3, '30s': 3},
        {'0s': 5, '10s': 4, '20s': 4, '30s': 2},
        {'0s': 6, '10s': 4, '20s': 3, '30s': 2},
        {'0s': 5, '10s': 5, '20s': 3, '30s': 2},
        {'0s': 4, '10s': 5, '20s': 3, '30s': 3},
        {'0s': 6, '10s': 3, '20s': 3, '30s': 3},
    ]
    oe_options = [7, 8, 9]
    prev_options = [1, 2]
    ld_options = [3, 4]
    run_options = [2, 3, None]
    hot_options = [5, 6, 7, 8]
    med_options = [2, 3, 4]
    cold_options = [4, 5, 6]
    tier_cap_options = {
        'EH': [1, 2, 3],
        'H': [2, 3, 4],
        'W': [4, 5, 6, 7],
        'C': [0, 1, 2]
    }

    # Seed from latest best 5+ config
    seed_weights = {
        'freq5': -0.3514,
        'freq10': -0.7094,
        'freq20': 1.0384,
        'freq50': 1.6005,
        'freq100': 0.0156,
        'gap': -0.1976,
        'pos_score': 0.0704,
        'cross_total_1w': -0.3296,
        'cross_sfl_1w': -0.1901,
        'cross_other_1w': -0.6520,
        'cross_total_14d': -0.8843,
        'cross_total_28d': 1.4363,
        'freq_no40_10': 1.7386,
        'freq_no40_20': 1.6522,
        'freq_no40_50': 0.2767,
        'gap_no40': 0.5477,
        'last_digit_freq': -0.0609,
        'decade_freq': -0.1674,
        'tier_EH': -1.6344,
        'tier_H': 1.5291,
        'tier_W': 1.9156,
        'tier_C': -1.0881,
    }
    seed_constraints = [
        {'0s': 4, '10s': 5, '20s': 3, '30s': 3},
        8, 1, 4, None, 7, 4, 6
    ]
    seed_tier_caps = {'EH': 2, 'H': 3, 'W': 6, 'C': 1}

    population = [
        {
            'weights': seed_weights.copy(),
            'caps': seed_constraints[0].copy(),
            'oe': seed_constraints[1],
            'prev': seed_constraints[2],
            'ld': seed_constraints[3],
            'run': seed_constraints[4],
            'hot': seed_constraints[5],
            'med': seed_constraints[6],
            'cold': seed_constraints[7],
            'tier_caps': seed_tier_caps.copy(),
        }
    ]
    while len(population) < pop_size:
        population.append({
            'weights': {k: random.uniform(v[0], v[1]) for k, v in weight_ranges.items()},
            'caps': random.choice(cap_options),
            'oe': random.choice(oe_options),
            'prev': random.choice(prev_options),
            'ld': random.choice(ld_options),
            'run': random.choice(run_options),
            'hot': random.choice(hot_options),
            'med': random.choice(med_options),
            'cold': random.choice(cold_options),
            'tier_caps': {k: random.choice(v) for k, v in tier_cap_options.items()},
        })

    best_ind = None
    best_fitness = (-1, -1, -1)  # (five, six, four)
    best_high = []
    best_pools = {}

    start_time = time.time()

    for gen in range(generations):
        fitnesses = []
        for ind in population:
            six, five, four, high, pools = evaluate_config(
                cache,
                ind['weights'],
                ind['caps'],
                ind['oe'],
                ind['prev'],
                ind['ld'],
                ind['run'],
                ind['hot'],
                ind['med'],
                ind['cold'],
                ind['tier_caps']
            )
            fitness = (five, six, four)
            fitnesses.append((fitness, high, pools, ind))

        for fitness, high, pools, ind in fitnesses:
            if fitness > best_fitness:
                best_fitness = fitness
                best_ind = ind.copy()
                best_high = high
                best_pools = pools

        fitnesses.sort(key=lambda x: x[0], reverse=True)

        if gen % 10 == 0 or gen == generations - 1:
            elapsed = time.time() - start_time
            print(f"Gen {gen:3d}/{generations} | best 5+={best_fitness[0]}, 6/6={best_fitness[1]}, 4+={best_fitness[2]} | elapsed={elapsed:.1f}s")

        new_population = []
        elite_count = max(5, pop_size // 20)
        for i in range(elite_count):
            new_population.append(fitnesses[i][3].copy())

        while len(new_population) < pop_size:
            def tournament(k=5):
                selected = random.sample(fitnesses, k)
                selected.sort(key=lambda x: x[0], reverse=True)
                return selected[0][3].copy()

            p1 = tournament()
            p2 = tournament()

            child = {
                'weights': {},
                'caps': p1['caps'] if random.random() < 0.5 else p2['caps'],
                'oe': p1['oe'] if random.random() < 0.5 else p2['oe'],
                'prev': p1['prev'] if random.random() < 0.5 else p2['prev'],
                'ld': p1['ld'] if random.random() < 0.5 else p2['ld'],
                'run': p1['run'] if random.random() < 0.5 else p2['run'],
                'hot': p1['hot'] if random.random() < 0.5 else p2['hot'],
                'med': p1['med'] if random.random() < 0.5 else p2['med'],
                'cold': p1['cold'] if random.random() < 0.5 else p2['cold'],
                'tier_caps': {},
            }
            for key in weight_ranges:
                if random.random() < 0.5:
                    child['weights'][key] = p1['weights'][key]
                else:
                    child['weights'][key] = p2['weights'][key]
            for key in tier_cap_options:
                if random.random() < 0.5:
                    child['tier_caps'][key] = p1['tier_caps'][key]
                else:
                    child['tier_caps'][key] = p2['tier_caps'][key]

            if random.random() < mutation_rate:
                for key in child['weights']:
                    if random.random() < 0.2:
                        child['weights'][key] += random.uniform(-0.3, 0.3)
            if random.random() < 0.1:
                child['caps'] = random.choice(cap_options)
            if random.random() < 0.1:
                child['oe'] = random.choice(oe_options)
            if random.random() < 0.1:
                child['prev'] = random.choice(prev_options)
            if random.random() < 0.1:
                child['ld'] = random.choice(ld_options)
            if random.random() < 0.1:
                child['run'] = random.choice(run_options)
            if random.random() < 0.1:
                child['hot'] = random.choice(hot_options)
            if random.random() < 0.1:
                child['med'] = random.choice(med_options)
            if random.random() < 0.1:
                child['cold'] = random.choice(cold_options)
            if random.random() < 0.2:
                key = random.choice(list(tier_cap_options.keys()))
                child['tier_caps'][key] = random.choice(tier_cap_options[key])

            new_population.append(child)

        population = new_population

    # Final hill climb
    print(f"\nGA complete. Best before hill climb: 5+={best_fitness[0]}, 6/6={best_fitness[1]}, 4+={best_fitness[2]}")
    constraints = [
        best_ind['caps'], best_ind['oe'], best_ind['prev'], best_ind['ld'],
        best_ind['run'], best_ind['hot'], best_ind['med'], best_ind['cold']
    ]
    hill_result = hill_climb(cache, best_ind['weights'], constraints, best_ind['tier_caps'], steps=hill_steps)

    print("\n" + "=" * 90)
    print("FINAL BEST AFTER TIER-CAPPED GA + HILL CLIMB")
    print("=" * 90)
    print(f"5+ traps : {hill_result['five']}/20")
    print(f"6/6 traps: {hill_result['six']}/20")
    print(f"4+ traps : {hill_result['four']}/20")
    print("\nWeights:")
    for k, v in hill_result['weights'].items():
        print(f"  {k}: {v:.4f}")
    print("\nConstraints:")
    c = hill_result['constraints']
    print(f"  caps={c[0]}")
    print(f"  odd_even_cap={c[1]}")
    print(f"  max_prev={c[2]}")
    print(f"  ld_cap={c[3]}")
    print(f"  max_run={c[4]}")
    print(f"  hot_count={c[5]}")
    print(f"  medium_count={c[6]}")
    print(f"  cold_count={c[7]}")
    print(f"  tier_caps={hill_result['tier_caps']}")

    if hill_result['high_draws']:
        print("\nHigh-capture draws (5+):")
        for date, cap, pool in hill_result['high_draws']:
            print(f"  {pd.to_datetime(date).strftime('%d-%b-%Y')}: {len(cap)}/6 -> {cap} | pool={pool}")

if __name__ == "__main__":
    run_ga_tier_caps(pop_size=150, generations=150, mutation_rate=0.3, hill_steps=300)