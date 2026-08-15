import csv
import ast
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import random
import math
from itertools import combinations

EPS = 1e-9
file_path = 'cross_lotto_data_backup.csv'

# ============================================================
# 1. Load Data
# ============================================================
wdw_data = []
sfl_data = {}

with open(file_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row or len(row) < 3:
            continue
        date_str = row[0].strip()
        try:
            date = datetime.strptime(date_str, '%a %d-%b-%Y')
        except:
            continue

        if len(row) > 1 and row[1].strip():
            sfl_str = row[1].strip()
            if sfl_str.startswith('"') and sfl_str.endswith('"'):
                sfl_str = sfl_str[1:-1]
            import re
            nums = [int(x) for x in re.findall(r'\d+', sfl_str)]
            sfl_data[date] = nums

        if date.strftime('%A') in ['Monday', 'Wednesday', 'Friday']:
            others_str = row[2].strip()
            if not others_str:
                continue
            if others_str.startswith('"') and others_str.endswith('"'):
                others_str = others_str[1:-1]
            parts = others_str.split('], [')
            if len(parts) < 2:
                continue
            main_str = parts[0] + ']'
            supp_str = '[' + parts[1]
            try:
                main_nums = ast.literal_eval(main_str)
                supp_nums = ast.literal_eval(supp_str)
                if len(main_nums) > 6:
                    main_nums = main_nums[:6]
                wdw_data.append({
                    'date': date,
                    'numbers': main_nums,
                    'supp': supp_nums
                })
            except:
                continue

wdw_data.sort(key=lambda x: x['date'])
print(f"Total WDW draws parsed: {len(wdw_data)}")
if len(wdw_data) < 21:
    exit()

# ============================================================
# 2. Core Functions
# ============================================================
def get_profile(nums):
    return {
        'sum': sum(nums),
        'odd': sum(1 for n in nums if n % 2 == 1),
        'spread': max(nums) - min(nums) if nums else 0,
        'low': sum(1 for n in nums if n <= 15),
        'mid': sum(1 for n in nums if 16 <= n <= 30),
        'high': sum(1 for n in nums if n >= 31),
    }

def profile_from_draws(draws):
    sums, odds, spreads, lows, mids, highs = [], [], [], [], [], []
    for nums in draws:
        p = get_profile(nums)
        sums.append(p['sum']); odds.append(p['odd']); spreads.append(p['spread'])
        lows.append(p['low']); mids.append(p['mid']); highs.append(p['high'])
    avg = {
        'sum': sum(sums)/len(sums),
        'odd': sum(odds)/len(odds),
        'spread': sum(spreads)/len(spreads),
        'low': sum(lows)/len(lows),
        'mid': sum(mids)/len(mids),
        'high': sum(highs)/len(highs),
    }
    std = {
        'sum': math.sqrt(sum((x-avg['sum'])**2 for x in sums)/len(sums)),
        'spread': math.sqrt(sum((x-avg['spread'])**2 for x in spreads)/len(spreads)),
        'odd': math.sqrt(sum((x-avg['odd'])**2 for x in odds)/len(odds)),
        'low': math.sqrt(sum((x-avg['low'])**2 for x in lows)/len(lows)),
        'mid': math.sqrt(sum((x-avg['mid'])**2 for x in mids)/len(mids)),
        'high': math.sqrt(sum((x-avg['high'])**2 for x in highs)/len(highs)),
    }
    return avg, std

def composite_scores(train_data, target_date, sfl_data):
    # Hot numbers (weighted frequency)
    scores_hot = {n: 0.0 for n in range(1, 46)}
    recent = train_data[-20:] if len(train_data) >= 20 else train_data
    for idx, d in enumerate(recent):
        weight = 3 if idx >= len(recent) - 5 else 1
        for n in d['numbers']:
            scores_hot[n] += weight

    # Markov chain
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(train_data)):
        prev = set(train_data[i-1]['numbers'])
        curr = set(train_data[i]['numbers'])
        for p in prev:
            for c in curr:
                if p != c:
                    transitions[p][c] += 1
    scores_markov = {n: 0.0 for n in range(1, 46)}
    if train_data:
        last_draw = train_data[-1]['numbers']
        for n in last_draw:
            for target, count in transitions[n].items():
                scores_markov[target] += count

    # KNN profile matching
    last_5 = train_data[-5:] if len(train_data) >= 5 else train_data
    target_prof, _ = profile_from_draws([d['numbers'] for d in last_5])
    distances = []
    for d in train_data:
        prof = get_profile(d['numbers'])
        dist = math.sqrt(sum((prof[k]-target_prof[k])**2 for k in ['sum','odd','spread','low','mid','high']))
        distances.append((dist, d))
    distances.sort(key=lambda x: x[0])
    similar = [d for _, d in distances[:5]]
    freq_knn = Counter()
    for d in similar:
        for n in d['numbers']:
            freq_knn[n] += 1
    scores_knn = {n: freq_knn.get(n, 0) for n in range(1, 46)}

    # Entropy
    freq = Counter()
    for d in train_data:
        for n in d['numbers']:
            freq[n] += 1
    ranges = [(1,9), (10,18), (19,27), (28,36), (37,45)]
    scores_entropy = {n: freq.get(n, 0) for n in range(1, 46)}
    for n in range(1, 46):
        for lo, hi in ranges:
            if lo <= n <= hi:
                scores_entropy[n] += 1.0 / (1 + abs(n - (lo+hi)/2))
                break

    def normalize(d):
        maxv = max(d.values()) if d else 1
        return {k: v/maxv for k, v in d.items()}

    norm_hot = normalize(scores_hot)
    norm_markov = normalize(scores_markov)
    norm_knn = normalize(scores_knn)
    norm_entropy = normalize(scores_entropy)

    composite = {n: 0.0 for n in range(1, 46)}
    for n in range(1, 46):
        composite[n] = (0.6 * norm_hot[n] +
                        0.2 * norm_markov[n] +
                        0.1 * norm_knn[n] +
                        0.1 * norm_entropy[n])

    # SFL bonus
    prev_day = target_date - timedelta(days=1)
    if prev_day in sfl_data:
        for n in sfl_data[prev_day]:
            if 1 <= n <= 45:
                composite[n] += 2.0

    return composite, target_prof

def run_backtest(top_n, tickets_per_draw=100):
    min_train = 50
    test_size = 40
    start_idx = max(min_train, len(wdw_data) - test_size)
    test_data = wdw_data[start_idx:]
    best_hit_ever = 0
    best_draw = None
    hit_counts = defaultdict(int)
    coverage_log = []  # log whether winning numbers were in top_n

    for target in test_data:
        train_data = wdw_data[:wdw_data.index(target)]
        composite, _ = composite_scores(train_data, target['date'], sfl_data)
        candidates = sorted(composite, key=lambda n: composite[n], reverse=True)[:top_n]
        candidate_set = set(candidates)
        actual = set(target['numbers'])
        covered = actual & candidate_set
        coverage_log.append((target['date'], len(covered), actual - candidate_set))

        co_occ = defaultdict(lambda: defaultdict(int))
        for d in train_data:
            nums = d['numbers']
            for a in nums:
                for b in nums:
                    if a != b:
                        co_occ[a][b] += 1

        # Generate all combinations from candidates
        all_combos = list(combinations(candidates, 6))
        last_5 = train_data[-5:] if len(train_data) >= 5 else train_data
        avg_prof, std_prof = profile_from_draws([d['numbers'] for d in last_5])
        filtered = []
        for combo in all_combos:
            p = get_profile(combo)
            if abs(p['sum'] - avg_prof['sum']) > 2.0 * std_prof['sum']:
                continue
            if abs(p['spread'] - avg_prof['spread']) > 2.0 * std_prof['spread']:
                continue
            if p['odd'] < 2 or p['odd'] > 4:
                continue
            if p['low'] < 1 or p['low'] > 3 or p['mid'] < 1 or p['mid'] > 3 or p['high'] < 1 or p['high'] > 3:
                continue
            score = sum(composite[n] for n in combo)
            pair = 0
            for i in range(6):
                for j in range(i+1, 6):
                    pair += co_occ[combo[i]][combo[j]]
            score += pair * 0.3
            filtered.append((combo, score))
        filtered.sort(key=lambda x: x[1], reverse=True)
        tickets = [combo for combo, _ in filtered[:tickets_per_draw]]
        if len(tickets) < tickets_per_draw:
            while len(tickets) < tickets_per_draw:
                fallback = tuple(sorted(random.sample(candidates, 6)))
                if fallback not in tickets:
                    tickets.append(fallback)

        best_hit = 0
        for t in tickets:
            hits = len(set(t) & actual)
            if hits > best_hit:
                best_hit = hits
                if hits == 6:
                    print(f"JACKPOT! 6/6 on {target['date'].strftime('%a %d-%b-%Y')} with ticket {sorted(t)}")
        hit_counts[best_hit] += 1
        if best_hit > best_hit_ever:
            best_hit_ever = best_hit
            best_draw = target['date'].strftime('%a %d-%b-%Y')

    # Print coverage summary
    print(f"\nCoverage summary (top_{top_n}):")
    for date, covered, missing in coverage_log:
        print(f"  {date.strftime('%a %d-%b-%Y')}: covered {covered}/6, missing: {sorted(missing)}")

    return best_hit_ever, best_draw, hit_counts

# ============================================================
# 3. Test different candidate pool sizes
# ============================================================
for top_n in [20, 25, 30, 35, 40]:
    print(f"\n{'='*60}")
    print(f"Testing top_n = {top_n}, tickets = 100")
    print("="*60)
    best_hit, best_draw, hit_counts = run_backtest(top_n, tickets_per_draw=100)
    print(f"Best hit: {best_hit} on {best_draw}")
    print("Distribution:")
    for h in sorted(hit_counts.keys()):
        print(f"  {h} hits: {hit_counts[h]} draws ({hit_counts[h]/40*100:.1f}%)")
    if best_hit == 6:
        print("*** JACKPOT FOUND! ***")
        break