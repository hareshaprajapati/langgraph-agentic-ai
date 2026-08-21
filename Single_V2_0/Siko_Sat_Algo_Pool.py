import pandas as pd
import numpy as np
from collections import Counter
import random
import math

SATURDAY_FILE = "Saturday_data.csv"
CROSS_FILE = "cross_lotto_data_backup.csv"

# ================= HELPERS =================
def dec(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

def last_digit(n):
    return n % 10

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

# ================= TIER COMPUTATION =================
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

# ================= FEATURE COMPUTATION =================
def compute_features(target_date, prior_sat):
    freq50 = Counter()
    gap = {}
    last_seen = {}

    for _, row in prior_sat.tail(50).iterrows():
        for n in row["nums"]:
            freq50[n] += 1

    all_prior = [(r["Date_dt"], r["nums"]) for _, r in prior_sat.iterrows()]
    for idx, (_, nums) in enumerate(all_prior):
        for n in nums:
            last_seen[n] = idx
    max_idx = len(all_prior) - 1
    for n in range(1, 46):
        gap[n] = max_idx - last_seen.get(n, -1)

    last_draw_nums = []
    if len(prior_sat) > 0:
        last_draw_nums = prior_sat.iloc[-1]["nums"]

    # Position score
    pos_counts = {pos: Counter() for pos in range(1, 7)}
    prior_no40 = prior_sat[prior_sat["nums"].apply(lambda nums: all(dec(n) != "40s" for n in nums))]
    for _, row in prior_no40.tail(300).iterrows():
        sorted_nums = sorted(row["nums"])
        for pos, n in enumerate(sorted_nums, start=1):
            pos_counts[pos][n] += 1

    position_score = Counter()
    for pos in range(1, 7):
        for n, cnt in pos_counts[pos].items():
            position_score[n] += cnt

    return freq50, gap, last_draw_nums, position_score

# ================= CONSTRAINTS =================
def apply_constraints(priority, base_score, gap, last_draw_nums, eligible, caps,
                      ld_cap=3, max_prev=1, max_run=2):
    pool = []
    pool_set = set()
    decade_counts = Counter()
    ld_counts = Counter()
    prev_counts = Counter()
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
        if n % 2 == 1 and odd_count >= 8:
            return False
        if n % 2 == 0 and even_count >= 8:
            return False
        if decade_counts[dec(n)] >= caps.get(dec(n), 4):
            return False
        if ld_counts[n % 10] >= ld_cap:
            return False
        if max_run is not None and run_len_if_add(n) > max_run:
            return False
        return True

    def add(n):
        nonlocal odd_count, even_count
        pool.append(n)
        pool_set.add(n)
        decade_counts[dec(n)] += 1
        ld_counts[n % 10] += 1
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

    hot_sorted = sorted(eligible, key=base_score, reverse=True)
    cold_sorted = sorted(eligible, key=lambda n: gap.get(n, 0), reverse=True)
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

    return sorted(pool)

# ================= POOL BUILDERS =================
def build_baseline_pool(base_score, gap, last_draw_nums, eligible, caps,
                        hot_count=7, medium_count=3, cold_count=5,
                        ld_cap=3, max_prev=1, max_run=2):
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
    return apply_constraints(priority, base_score, gap, last_draw_nums, eligible, caps,
                             ld_cap, max_prev, max_run)

def build_diversity_pool(base_score, gap, last_draw_nums, eligible, caps,
                         diversity_weight=0.01,
                         hot_count=7, medium_count=3, cold_count=5,
                         ld_cap=3, max_prev=1, max_run=2):
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

    def similarity(a, b):
        s = 0.0
        if dec(a) == dec(b):
            s += 1.0
        if last_digit(a) == last_digit(b):
            s += 1.0
        if (a % 2) == (b % 2):
            s += 0.5
        if (a <= 22) == (b <= 22):
            s += 0.5
        return s

    pool = []
    pool_set = set()
    decade_counts = Counter()
    ld_counts = Counter()
    prev_counts = Counter()
    odd_count = 0
    even_count = 0

    def can_add(n):
        if n in pool_set:
            return False
        if n in last_draw_nums and prev_counts[n] >= max_prev:
            return False
        if n % 2 == 1 and odd_count >= 8:
            return False
        if n % 2 == 0 and even_count >= 8:
            return False
        if decade_counts[dec(n)] >= caps.get(dec(n), 4):
            return False
        if ld_counts[n % 10] >= ld_cap:
            return False
        return True

    def add(n):
        nonlocal odd_count, even_count
        pool.append(n)
        pool_set.add(n)
        decade_counts[dec(n)] += 1
        ld_counts[n % 10] += 1
        if n % 2 == 1:
            odd_count += 1
        else:
            even_count += 1
        if n in last_draw_nums:
            prev_counts[n] += 1

    remaining_priority = list(priority)
    while len(pool) < 15 and remaining_priority:
        best_n = None
        best_score = -1e18

        for n in remaining_priority:
            if not can_add(n):
                continue
            diversity_penalty = sum(diversity_weight * similarity(n, m) for m in pool)
            score = base_score(n) - diversity_penalty
            if score > best_score:
                best_score = score
                best_n = n

        if best_n is None:
            for n in hot_sorted:
                if n not in pool_set and can_add(n):
                    best_n = n
                    break
            if best_n is None:
                break

        add(best_n)
        remaining_priority = [x for x in remaining_priority if x != best_n]

    for n in hot_sorted:
        if len(pool) >= 15:
            break
        if n not in pool_set and can_add(n):
            add(n)

    return sorted(pool)

def build_maximin_pool(base_score, gap, last_draw_nums, eligible, caps,
                       feature_matrix,
                       hot_count=7, medium_count=3, cold_count=5,
                       ld_cap=3, max_prev=1, max_run=2):
    """
    Greedy maximin using feature_matrix (DataFrame with eligible as index).
    """
    selected = []
    selected_set = set()
    selected_vectors = []

    first = max(eligible, key=lambda n: base_score(n))
    selected.append(first)
    selected_set.add(first)
    selected_vectors.append(feature_matrix.loc[first].values)

    while len(selected) < 15:
        best_n = None
        best_dist = -1
        for n in eligible:
            if n in selected_set:
                continue
            min_dist = min(np.linalg.norm(feature_matrix.loc[n].values - v) for v in selected_vectors)
            if min_dist > best_dist:
                best_dist = min_dist
                best_n = n

        if best_n is None:
            break
        selected.append(best_n)
        selected_set.add(best_n)
        selected_vectors.append(feature_matrix.loc[best_n].values)

    return apply_constraints(selected, base_score, gap, last_draw_nums, eligible, caps,
                             ld_cap, max_prev, max_run)

def build_genetic_pool(base_score, gap, last_draw_nums, eligible, caps,
                       prior_draws, population_size=30, generations=10,
                       ld_cap=3, max_prev=1, max_run=2):
    """
    Genetic algorithm for 15-number pool.
    """
    def random_pool():
        shuffled = list(eligible)
        random.shuffle(shuffled)
        pool = []
        decade_counts = Counter()
        ld_counts = Counter()
        prev_counts = Counter()
        odd_count = 0
        even_count = 0

        for n in shuffled:
            if len(pool) >= 15:
                break
            if n in last_draw_nums and prev_counts[n] >= max_prev:
                continue
            if n % 2 == 1 and odd_count >= 8:
                continue
            if n % 2 == 0 and even_count >= 8:
                continue
            if decade_counts[dec(n)] >= caps.get(dec(n), 4):
                continue
            if ld_counts[n % 10] >= ld_cap:
                continue
            pool.append(n)
            decade_counts[dec(n)] += 1
            ld_counts[n % 10] += 1
            if n % 2 == 1:
                odd_count += 1
            else:
                even_count += 1
            if n in last_draw_nums:
                prev_counts[n] += 1
        return sorted(pool)

    def fitness(pool):
        score = 0
        for _, row in prior_draws.iterrows():
            hits = len(set(pool) & set(row["nums"]))
            if hits == 6:
                score += 10
            elif hits == 5:
                score += 3
            elif hits == 4:
                score += 1
        return score

    population = [random_pool() for _ in range(population_size)]
    population = [p for p in population if len(p) == 15]
    while len(population) < population_size:
        population.append(random_pool())
    population = [sorted(p) for p in population]

    for _ in range(generations):
        fitnesses = [fitness(p) for p in population]
        sorted_pop = [p for _, p in sorted(zip(fitnesses, population), key=lambda x: -x[0])]
        population = sorted_pop[:population_size//2]

        new_pop = []
        while len(new_pop) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = list(set(parent1) & set(parent2))
            union = list(set(parent1) | set(parent2))
            random.shuffle(union)
            for n in union:
                if len(child) >= 15:
                    break
                if n not in child:
                    child.append(n)
            if len(child) < 15:
                missing = [n for n in eligible if n not in child]
                random.shuffle(missing)
                child.extend(missing[:15-len(child)])
            if random.random() < 0.3 and child:
                idx = random.randrange(len(child))
                child.pop(idx)
                missing = [n for n in eligible if n not in child]
                child.append(random.choice(missing))
            child = sorted(child)
            new_pop.append(child)
        population = new_pop

    best_pool = max(population, key=fitness)
    return sorted(best_pool)

# ================= HYPERGEOMETRIC SCORING =================
def hypergeometric_prob(pool_size, total, k):
    def comb(n, r):
        if r > n or r < 0:
            return 0
        return math.comb(n, r)
    return comb(pool_size, k) * comb(total-pool_size, 6-k) / comb(total, 6)

# ================= EVALUATION =================
def evaluate_pool(pool, real_nums):
    return len(set(pool) & set(real_nums))

def run_full_evaluation():
    no40 = sat_df[sat_df["nums"].apply(lambda nums: all(dec(n) != "40s" for n in nums))].copy()
    test_draws = no40.tail(20)

    caps = {'0s': 5, '10s': 4, '20s': 3, '30s': 3}
    tier_bonuses = {'EH': -0.7, 'H': 0.8, 'W': 1.0, 'C': -0.7}
    position_weight = 0.01

    methods = {
        'baseline': {},
        'diversity_0.01': {'type': 'diversity', 'weight': 0.01},
        'maximin': {'type': 'maximin'},
        'genetic': {'type': 'genetic'},
    }

    results = {name: {'six':0, 'five':0, 'four':0, 'tested':0, 'high_draws':[], 'hyper_score_sum':0.0}
               for name in methods}

    mc_six_count = 0
    mc_five_plus_count = 0
    mc_four_plus_count = 0
    mc_total_pools = 0

    random.seed(42)

    for _, target_row in test_draws.iterrows():
        target_date = target_row["Date_dt"]
        real_nums = set(target_row["nums"])
        prior_sat = sat_df[sat_df["Date_dt"] < target_date]

        freq50, gap, last_draw_nums, position_score = compute_features(target_date, prior_sat)
        tiers = get_tiers_saturday_to_friday(target_date)
        if tiers is None:
            continue

        eligible = [n for n in range(1, 46) if dec(n) != "40s"]

        def tier_of(n):
            if n in tiers[0]:
                return 'EH'
            if n in tiers[1]:
                return 'H'
            if n in tiers[2]:
                return 'W'
            return 'C'

        base_score = lambda n: (
            freq50.get(n, 0)
            + 0.1 * gap.get(n, 0)
            + tier_bonuses.get(tier_of(n), 0)
            + position_weight * position_score.get(n, 0)
        )

        # Prepare feature matrix for maximin
        feature_vectors = {}
        for n in eligible:
            feature_vectors[n] = [
                freq50.get(n, 0),
                gap.get(n, 0),
                position_score.get(n, 0),
                int(dec(n)[0]),
                last_digit(n),
                n % 2,
            ]
        feature_matrix = pd.DataFrame(feature_vectors).T
        feature_matrix = (feature_matrix - feature_matrix.mean()) / (feature_matrix.std() + 1e-6)

        # Build pools
        pools = {}
        for method_name, params in methods.items():
            if method_name == 'baseline':
                pool = build_baseline_pool(base_score, gap, last_draw_nums, eligible, caps)
            elif params.get('type') == 'diversity':
                pool = build_diversity_pool(base_score, gap, last_draw_nums, eligible, caps,
                                            diversity_weight=params['weight'])
            elif params.get('type') == 'maximin':
                pool = build_maximin_pool(base_score, gap, last_draw_nums, eligible, caps,
                                          feature_matrix)
            elif params.get('type') == 'genetic':
                prior_draws = no40[no40["Date_dt"] < target_date].tail(50)
                pool = build_genetic_pool(base_score, gap, last_draw_nums, eligible, caps,
                                          prior_draws, population_size=30, generations=10)
            else:
                continue

            pools[method_name] = pool

        # Evaluate each pool
        for method_name, pool in pools.items():
            cov = evaluate_pool(pool, real_nums)
            res = results[method_name]
            res['tested'] += 1
            if cov >= 6:
                res['six'] += 1
            if cov >= 5:
                res['five'] += 1
                res['high_draws'].append((target_date, sorted(set(pool)&real_nums), pool))
            if cov >= 4:
                res['four'] += 1

            p_ge = 0
            for k in range(cov, 7):
                p_ge += hypergeometric_prob(15, 39, k)
            res['hyper_score_sum'] += -math.log10(p_ge + 1e-12)

        # Monte Carlo for this target
        for _ in range(200):
            random_pool = random.sample(eligible, 15)
            cov = evaluate_pool(random_pool, real_nums)
            if cov >= 6:
                mc_six_count += 1
            if cov >= 5:
                mc_five_plus_count += 1
            if cov >= 4:
                mc_four_plus_count += 1
            mc_total_pools += 1

    # Print results
    print("\n" + "=" * 90)
    print("FULL EVALUATION WITH HYPERGEOMETRIC AND MONTE CARLO")
    print("=" * 90)
    print(f"{'Method':<20} {'6/6':<6} {'5+':<6} {'4+':<6} {'Hypergeometric Score':<20}")
    print("-" * 70)

    for method_name, res in results.items():
        if res['tested'] > 0:
            avg_hyper = res['hyper_score_sum'] / res['tested']
            print(f"{method_name:<20} {res['six']:<6} {res['five']:<6} {res['four']:<6} {avg_hyper:<20.3f}")
        else:
            print(f"{method_name:<20} no data")

    print("\n" + "=" * 90)
    print("MONTE CARLO RANDOM POOL BENCHMARK (200 random pools per draw)")
    print("=" * 90)
    print(f"Total random pools generated: {mc_total_pools}")
    print(f"6/6 rate:   {mc_six_count / mc_total_pools:.4%}")
    print(f"5+ rate:    {mc_five_plus_count / mc_total_pools:.4%}")
    print(f"4+ rate:    {mc_four_plus_count / mc_total_pools:.4%}")
    print("\nOur best method results per 20 draws:")
    best_method = max(results.items(), key=lambda x: (x[1]['five'], x[1]['six'], x[1]['four']))
    name, res = best_method
    print(f"{name}: 6/6={res['six']}/20, 5+={res['five']}/20, 4+={res['four']}/20")
    print("Random expected per 20 draws (approx):")
    print(f"6/6: {20 * mc_six_count / mc_total_pools:.2f}")
    print(f"5+:  {20 * mc_five_plus_count / mc_total_pools:.2f}")
    print(f"4+:  {20 * mc_four_plus_count / mc_total_pools:.2f}")

    if res['high_draws']:
        print(f"\nHigh-capture draws for {name}:")
        for date, cap, pool in res['high_draws']:
            print(f"  {pd.to_datetime(date).strftime('%d-%b-%Y')}: {len(cap)}/6 -> {cap} | pool={pool}")

if __name__ == "__main__":
    run_full_evaluation()