import csv
from collections import Counter, defaultdict
from datetime import datetime

CSV_FILE = "cross_lotto_data_backup.csv"
OUTPUT_LAST_N = 60*60
FUTURE_DATE_STR = "Sat 05-Sep-2026"

# Backtest settings
RUN_BACKTEST = True
BACKTEST_N = 100        # number of most recent draws to backtest
K_NEIGHBORS = 10        # for conditional mode predictor

def parse_date(s):
    return datetime.strptime(s[4:], '%d-%b-%Y')

def extract_numbers(cell):
    nums = []
    for part in cell.split(']'):
        part = part.replace('[', '').strip()
        if part:
            for token in part.split(','):
                token = token.strip()
                if token:
                    n = int(token)
                    if 1 <= n <= 45:
                        nums.append(n)
    return nums

def extract_main6(others_cell):
    main_part = others_cell.split(']')[0].replace('[', '').strip()
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

def predict_counts_proportional(eh_pool, h_pool, w_pool, c_pool):
    sizes = [eh_pool, h_pool, w_pool, c_pool]
    raw = [size * 6 / 45 for size in sizes]
    rounded = [round(x) for x in raw]
    for i in range(4):
        frac = raw[i] - int(raw[i])
        if abs(frac - 0.5) < 1e-9:
            rounded[i] = int(raw[i]) + 1
    diff = 6 - sum(rounded)
    if diff > 0:
        order = sorted(range(4), key=lambda i: sizes[i], reverse=True)
        for i in order[:diff]:
            rounded[i] += 1
    elif diff < 0:
        order = sorted(range(4), key=lambda i: sizes[i])
        for i in order[:-diff]:
            rounded[i] -= 1
    return tuple(rounded)

def predict_counts_mode(eh_pool, h_pool, w_pool, c_pool, history, k=K_NEIGHBORS):
    """
    Conditional Mode predictor:
    Find the k nearest historical draws (by pool size Euclidean distance),
    return the most frequent EH/H/W/C tuple among them.
    If no history, fallback to proportional.
    """
    if not history:
        return predict_counts_proportional(eh_pool, h_pool, w_pool, c_pool)

    target = (eh_pool, h_pool, w_pool, c_pool)
    distances = []
    for h_eh, h_h, h_w, h_c, h_counts in history:
        dist = ((eh_pool - h_eh)**2 + (h_pool - h_h)**2 +
                (w_pool - h_w)**2 + (c_pool - h_c)**2) ** 0.5
        distances.append((dist, h_counts))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    # Count frequency of each outcome
    freq = Counter(cnt for _, cnt in neighbors)
    # Find most common outcome(s)
    max_freq = max(freq.values())
    modes = [cnt for cnt, f in freq.items() if f == max_freq]
    if len(modes) == 1:
        return modes[0]
    else:
        # Tie: choose the one with smallest average distance to target
        best_mode = None
        best_avg_dist = float('inf')
        for mode in modes:
            dists = [d for d, cnt in neighbors if cnt == mode]
            avg = sum(dists) / len(dists)
            if avg < best_avg_dist:
                best_avg_dist = avg
                best_mode = mode
        return best_mode

# Read all data
all_rows = []
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 3:
            continue
        date_str = row[0].strip()
        try:
            dt = parse_date(date_str)
        except:
            continue
        sfl_nums = extract_numbers(row[1])
        others_nums = extract_numbers(row[2])
        all_nums = sfl_nums + others_nums
        is_sat = date_str.startswith('Sat ')
        all_rows.append((date_str, dt, is_sat, all_nums, row[2] if is_sat else None))

all_rows.sort(key=lambda x: x[1])

# Saturdays with main6
saturdays = []
for date_str, dt, is_sat, _, others_cell in all_rows:
    if is_sat:
        main6 = extract_main6(others_cell)
        saturdays.append((date_str, dt, main6))

# Analysis for all historical Saturdays
results = []
for i in range(1, len(saturdays)):
    target_date_str, target_dt, target_main = saturdays[i]
    prev_sat_date_str, prev_sat_dt, prev_main = saturdays[i-1]

    window_nums = []
    for date_str, dt, is_sat, nums, _ in all_rows:
        if prev_sat_dt <= dt < target_dt:
            window_nums.extend(nums)

    counter = Counter(window_nums)
    eh = {n for n, cnt in counter.items() if cnt >= 4}
    h  = {n for n, cnt in counter.items() if cnt == 3}
    w  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
    c  = {n for n in range(1,46) if counter[n] == 0}

    eh_pool_size = len(eh)
    h_pool_size = len(h)
    w_pool_size = len(w)
    c_pool_size = len(c)
    eh_h_pool_size = eh_pool_size + h_pool_size

    counts = {'EH':0, 'H':0, 'W':0, 'C':0}
    for n in target_main:
        if n in eh:    counts['EH'] += 1
        elif n in h:   counts['H'] += 1
        elif n in w:   counts['W'] += 1
        else:          counts['C'] += 1

    w_count = counts['W']
    profile = "Breadth" if w_count >= 4 else "Depth"

    legacy_hits = [n for n in target_main if n in prev_main]

    results.append({
        'date': target_date_str,
        'profile': profile,
        'counts_tuple': (counts['EH'], counts['H'], counts['W'], counts['C']),
        'pools_tuple': (eh_pool_size, h_pool_size, w_pool_size, c_pool_size),
        'eh_h_pool': eh_h_pool_size,
        'legacy': legacy_hits
    })

# Print historical table (last N)
n = min(OUTPUT_LAST_N, len(results))
print(f"Last {n} Saturday Lotto draws analysis:\n")
print(f"{'Date':<20} {'Profile':<10} {'EH':<4} {'H':<4} {'W':<4} {'C':<4} "
      f"{'EH-Pool':<8} {'H-Pool':<8} {'W-Pool':<8} {'C-Pool':<8} "
      f"{'EH+H-Pool':<10} {'Legacy Hits'}")
print("-" * 90)
for r in results[-n:]:
    eh, h, w, c = r['counts_tuple']
    peh, ph, pw, pc = r['pools_tuple']
    legacy_str = str(r['legacy']) if r['legacy'] else "None"
    print(f"{r['date']:<20} {r['profile']:<10} {eh:<4} {h:<4} {w:<4} {c:<4} "
          f"{peh:<8} {ph:<8} {pw:<8} {pc:<8} "
          f"{r['eh_h_pool']:<10} {legacy_str}")

# ------------------------------------------------------------
# Additional Analysis: Frequency distribution of outcomes
# ------------------------------------------------------------
print("\n" + "="*80)
print("Outcome frequency distribution (all historical draws):")
print("="*80)
outcome_freq = Counter(r['counts_tuple'] for r in results)
total_draws = len(results)
for outcome, freq in outcome_freq.most_common():
    print(f"  {outcome}  ->  {freq} times  ({freq/total_draws:.1%})")

# ------------------------------------------------------------
# Backtest comparison: Proportional vs Conditional Mode
# ------------------------------------------------------------
if RUN_BACKTEST:
    print("\n" + "="*100)
    print(f"Backtest comparison over last {BACKTEST_N} draws:")
    print("="*100)
    print(f"{'Date':<20} {'Pool Sizes':<15} {'Prop Pred':<15} {'Mode Pred':<15} {'Actual':<12} {'Prop Err':<8} {'Mode Err':<8}")
    print("-" * 100)

    history = []
    total_results = len(results)
    start_idx = total_results - BACKTEST_N

    prop_exact = 0
    mode_exact = 0
    prop_abs_err = 0
    mode_abs_err = 0

    for idx in range(total_results):
        r = results[idx]
        if idx >= start_idx:
            # Proportional prediction
            prop_pred = predict_counts_proportional(*r['pools_tuple'])
            # Mode prediction (using only history before this draw)
            mode_pred = predict_counts_mode(*r['pools_tuple'], history, k=K_NEIGHBORS)

            actual = r['counts_tuple']
            prop_err = sum(abs(p - a) for p, a in zip(prop_pred, actual))
            mode_err = sum(abs(p - a) for p, a in zip(mode_pred, actual))

            if prop_pred == actual:
                prop_exact += 1
            if mode_pred == actual:
                mode_exact += 1
            prop_abs_err += prop_err
            mode_abs_err += mode_err

            peh, ph, pw, pc = r['pools_tuple']
            pool_str = f"{peh}/{ph}/{pw}/{pc}"
            prop_str = f"{prop_pred[0]}/{prop_pred[1]}/{prop_pred[2]}/{prop_pred[3]}"
            mode_str = f"{mode_pred[0]}/{mode_pred[1]}/{mode_pred[2]}/{mode_pred[3]}"
            actual_str = f"{actual[0]}/{actual[1]}/{actual[2]}/{actual[3]}"
            print(f"{r['date']:<20} {pool_str:<15} {prop_str:<15} {mode_str:<15} {actual_str:<12} {prop_err:<8} {mode_err:<8}")

        # Always add to history after prediction
        history.append((*r['pools_tuple'], r['counts_tuple']))

    print("-" * 100)
    print(f"Proportional model: Exact matches = {prop_exact}/{BACKTEST_N} ({prop_exact/BACKTEST_N:.0%}), Avg abs error = {prop_abs_err/BACKTEST_N:.2f}")
    print(f"Conditional Mode  : Exact matches = {mode_exact}/{BACKTEST_N} ({mode_exact/BACKTEST_N:.0%}), Avg abs error = {mode_abs_err/BACKTEST_N:.2f}")

# ------------------------------------------------------------
# Future date prediction
# ------------------------------------------------------------
print("\n" + "="*90)
print(f"Prediction for {FUTURE_DATE_STR}")
print("="*90)

future_dt = parse_date(FUTURE_DATE_STR)
prev_sat = None
for date_str, dt, _ in saturdays:
    if dt < future_dt:
        prev_sat = (date_str, dt)
    else:
        break

if prev_sat is None:
    print("No previous Saturday found in data.")
else:
    prev_sat_date_str, prev_sat_dt = prev_sat
    window_nums = []
    for date_str, dt, is_sat, nums, _ in all_rows:
        if prev_sat_dt <= dt < future_dt:
            window_nums.extend(nums)

    counter = Counter(window_nums)
    eh = {n for n, cnt in counter.items() if cnt >= 4}
    h  = {n for n, cnt in counter.items() if cnt == 3}
    w  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
    c  = {n for n in range(1,46) if counter[n] == 0}

    eh_pool = len(eh)
    h_pool = len(h)
    w_pool = len(w)
    c_pool = len(c)

    print(f"Previous Saturday used for window: {prev_sat_date_str}")
    print(f"Window: {prev_sat_date_str} to Friday before {FUTURE_DATE_STR}")
    print("\nPool details:")
    print(f"EH {sorted(eh)}  EH-Pool-Size: {eh_pool}")
    print(f"H  {sorted(h)}  H-Pool-Size: {h_pool}")
    print(f"W  {sorted(w)}  W-Pool-Size: {w_pool}")
    print(f"C  {sorted(c)}  C-Pool-Size: {c_pool}")
    print(f"EH+H Pool Size: {eh_pool + h_pool}")

    # Build full history
    full_history = [(r['pools_tuple'][0], r['pools_tuple'][1], r['pools_tuple'][2], r['pools_tuple'][3], r['counts_tuple'])
                    for r in results]

    prop_pred = predict_counts_proportional(eh_pool, h_pool, w_pool, c_pool)
    mode_pred = predict_counts_mode(eh_pool, h_pool, w_pool, c_pool, full_history, k=K_NEIGHBORS)

    print("\nProportional prediction:", prop_pred)
    print("Conditional mode prediction:", mode_pred)
    print("(Use the one that performed better in backtest)")