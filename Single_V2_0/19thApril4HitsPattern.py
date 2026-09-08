import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math

# ---------- CONFIGURATION ----------
CSV_FILE = "cross_lotto_data_backup.csv"
OUTPUT_LAST_N = 20          # rows shown in historical tables

# Backtest settings (applied per lottery)
RUN_BACKTEST = False
BACKTEST_N = 10             # number of most recent draws to backtest (per lottery)
K_NEIGHBORS = 10            # for conditional mode predictor

# Set this to a specific date to predict only that day's lottery.
# Leave empty to predict the next draw for ALL lotteries.
# FUTURE_DATE_STR = "Sat 05-Sep-2026"   # example: "Fri 04-Sep-2026", "Tue 01-Sep-2026", etc.
FUTURE_DATE_STR = "Tue 08-Sep-2026"   # example: "Fri 04-Sep-2026", "Tue 01-Sep-2026", etc.
# FUTURE_DATE_STR = "Mon 31-Aug-2026"   # example: "Fri 04-Sep-2026", "Tue 01-Sep-2026", etc.
# FUTURE_DATE_STR = ""                # uncomment to process all lotteries

# Lottery definitions: day abbreviation -> (name, max number, main count)
LOTTERY_CONFIG = {
    'Mon': ('Weekday Windfall', 45, 6),
    'Tue': ('Oz Lotto', 47, 7),
    'Wed': ('Weekday Windfall', 45, 6),
    'Thu': ('Powerball', 35, 6),
    'Fri': ('Weekday Windfall', 45, 6),
    'Sat': ('Saturday Lotto', 45, 6),
}

# ---------- HELPER FUNCTIONS ----------
def parse_date(s):
    # input: "Sat 05-Sep-2026" -> datetime
    return datetime.strptime(s[4:], '%d-%b-%Y')

def extract_main_numbers(cell):
    """Extract the main numbers from the first bracket of the 'Others' column."""
    if not cell:
        return []
    main_part = cell.split(']')[0].replace('[', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

def extract_all_numbers(cell):
    """Extract all numbers (main + supplementary) from a cell."""
    nums = []
    for part in cell.split(']'):
        part = part.replace('[', '').strip()
        if part:
            for token in part.split(','):
                token = token.strip()
                if token:
                    nums.append(int(token))
    return nums

def round_to_sum(raw, target_sum):
    """Round a list of non‑negative floats to integers that sum to target_sum."""
    raw = [max(0.0, x) for x in raw]
    total = sum(raw)
    if total == 0:
        base = target_sum // len(raw)
        rem = target_sum % len(raw)
        res = [base] * len(raw)
        for i in range(rem):
            res[i] += 1
        return tuple(res)
    floors = [int(x) for x in raw]
    remainders = [(raw[i] - floors[i], i) for i in range(len(raw))]
    current_sum = sum(floors)
    remainders.sort(reverse=True)
    for i in range(target_sum - current_sum):
        idx = remainders[i % len(raw)][1]
        floors[idx] += 1
    return tuple(floors)

# ---------- PREDICTION FUNCTIONS ----------
def predict_counts_proportional(pools, total_numbers, main_count):
    sizes = pools
    raw = [size * main_count / total_numbers for size in sizes]
    return round_to_sum(raw, main_count)

def predict_counts_mode(pools, history, total_numbers, main_count, k=K_NEIGHBORS):
    if not history:
        return predict_counts_proportional(pools, total_numbers, main_count)
    target = pools
    distances = []
    for h_pools, h_counts in history:
        dist = sum((target[i] - h_pools[i])**2 for i in range(4)) ** 0.5
        distances.append((dist, h_counts))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    freq = Counter(cnt for _, cnt in neighbors)
    max_freq = max(freq.values())
    modes = [cnt for cnt, f in freq.items() if f == max_freq]
    if len(modes) == 1:
        return modes[0]
    else:
        best_mode = None
        best_avg_dist = float('inf')
        for mode in modes:
            dists = [d for d, cnt in neighbors if cnt == mode]
            avg = sum(dists) / len(dists)
            if avg < best_avg_dist:
                best_avg_dist = avg
                best_mode = mode
        return best_mode

def predict_counts_time_weighted_rate(pools, history, total_numbers, main_count, decay=0.95):
    if not history:
        return predict_counts_proportional(pools, total_numbers, main_count)
    total_pool = [0.0] * 4
    total_count = [0.0] * 4
    weight_sum = 0.0
    for i, (h_pools, h_counts) in enumerate(history):
        weight = decay ** (len(history) - 1 - i)
        for j in range(4):
            total_pool[j] += weight * h_pools[j]
            total_count[j] += weight * h_counts[j]
        weight_sum += weight
    if weight_sum == 0:
        return predict_counts_proportional(pools, total_numbers, main_count)
    rates = [total_count[j] / total_pool[j] if total_pool[j] > 0 else 0.0 for j in range(4)]
    raw = [pools[i] * rates[i] for i in range(4)]
    return round_to_sum(raw, main_count)

def predict_counts_ensemble(pools, history, total_numbers, main_count, k=K_NEIGHBORS, decay=0.95):
    raw_prop = [pools[i] * main_count / total_numbers for i in range(4)]
    if history:
        total_pool = [0.0] * 4
        total_count = [0.0] * 4
        for i, (h_pools, h_counts) in enumerate(history):
            weight = decay ** (len(history) - 1 - i)
            for j in range(4):
                total_pool[j] += weight * h_pools[j]
                total_count[j] += weight * h_counts[j]
        rates = [total_count[j] / total_pool[j] if total_pool[j] > 0 else 0.0 for j in range(4)]
        raw_rate = [pools[i] * rates[i] for i in range(4)]
    else:
        raw_rate = raw_prop
    if history:
        target = pools
        distances = []
        for h_pools, h_counts in history:
            dist = sum((target[i] - h_pools[i])**2 for i in range(4)) ** 0.5
            distances.append((dist, h_counts))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        eps = 1e-6
        inv_dists = [1.0 / (d + eps) for d, _ in neighbors]
        total_inv = sum(inv_dists)
        raw_mode = [0.0] * 4
        for (d, cnt), w in zip(neighbors, inv_dists):
            for j in range(4):
                raw_mode[j] += w * cnt[j] / total_inv
    else:
        raw_mode = raw_prop
    raw_ens = [(raw_prop[i] + raw_rate[i] + raw_mode[i]) / 3.0 for i in range(4)]
    return round_to_sum(raw_ens, main_count)

# ---------- PROCESSING FUNCTION FOR A GIVEN LOTTERY ----------
def process_lottery(day_abbr, draws, all_rows, max_num, main_count, lottery_name,
                    future_date=None):
    """
    Process one lottery type.
    If future_date is None, predict the next occurrence.
    If future_date is given, predict for that exact date.
    """
    if len(draws) < 2:
        print(f"\n{lottery_name} ({day_abbr}): Not enough draws (need at least 2). Skipping.")
        return

    print(f"\n{'='*80}")
    print(f"Processing {lottery_name} ({day_abbr}) – numbers 1–{max_num}, {main_count} main numbers")
    print('='*80)

    # Build historical results for this day only
    results = []
    for i in range(1, len(draws)):
        target_date_str, target_dt, target_main = draws[i]
        prev_date_str, prev_dt, prev_main = draws[i-1]

        window_nums = []
        for date_str, dt, day_abbr2, all_nums, _ in all_rows:
            if prev_dt <= dt < target_dt:
                valid_nums = [n for n in all_nums if 1 <= n <= max_num]
                window_nums.extend(valid_nums)

        counter = Counter(window_nums)
        eh = {n for n, cnt in counter.items() if cnt >= 4}
        h  = {n for n, cnt in counter.items() if cnt == 3}
        w  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
        c  = {n for n in range(1, max_num+1) if counter[n] == 0}

        eh_pool = len(eh)
        h_pool = len(h)
        w_pool = len(w)
        c_pool = len(c)

        counts = {'EH':0, 'H':0, 'W':0, 'C':0}
        for n in target_main:
            if n in eh:    counts['EH'] += 1
            elif n in h:   counts['H'] += 1
            elif n in w:   counts['W'] += 1
            else:          counts['C'] += 1

        w_count = counts['W']
        profile = "Breadth" if w_count >= (main_count // 2 + 1) else "Depth"

        legacy_hits = [n for n in target_main if n in prev_main]

        results.append({
            'date': target_date_str,
            'profile': profile,
            'counts_tuple': (counts['EH'], counts['H'], counts['W'], counts['C']),
            'pools_tuple': (eh_pool, h_pool, w_pool, c_pool),
            'eh_h_pool': eh_pool + h_pool,
            'legacy': legacy_hits
        })

    # Print historical table
    n = min(OUTPUT_LAST_N, len(results))
    print(f"\nLast {n} {lottery_name} draws analysis:\n")
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

    # Frequency distribution
    print("\n" + "="*80)
    print(f"Outcome frequency distribution (all {lottery_name} draws):")
    print("="*80)
    outcome_freq = Counter(r['counts_tuple'] for r in results)
    total_draws = len(results)
    for outcome, freq in outcome_freq.most_common():
        print(f"  {outcome}  ->  {freq} times  ({freq/total_draws:.1%})")

    # ---------- BACKTEST ----------
    if RUN_BACKTEST and len(results) >= BACKTEST_N:
        print("\n" + "="*100)
        print(f"Backtest comparison over last {BACKTEST_N} draws ({lottery_name}):")
        print("="*100)
        print(f"{'Date':<20} {'Pool Sizes':<15} {'Prop Pred':<15} {'Mode Pred':<15} {'TW-Rate Pred':<15} {'Ensemble Pred':<15} {'Actual':<12} {'Prop Err':<8} {'Mode Err':<8} {'TW-Rate Err':<8} {'Ens Err':<8}")
        print("-" * 100)

        history = []
        total_results = len(results)
        start_idx = total_results - BACKTEST_N

        prop_exact = mode_exact = twrate_exact = ens_exact = 0
        prop_abs_err = mode_abs_err = twrate_abs_err = ens_abs_err = 0

        for idx in range(total_results):
            r = results[idx]
            if idx >= start_idx:
                pools = r['pools_tuple']
                actual = r['counts_tuple']

                prop_pred = predict_counts_proportional(pools, max_num, main_count)
                mode_pred = predict_counts_mode(pools, history, max_num, main_count, k=K_NEIGHBORS)
                twrate_pred = predict_counts_time_weighted_rate(pools, history, max_num, main_count)
                ens_pred = predict_counts_ensemble(pools, history, max_num, main_count, k=K_NEIGHBORS)

                prop_err = sum(abs(p - a) for p, a in zip(prop_pred, actual))
                mode_err = sum(abs(p - a) for p, a in zip(mode_pred, actual))
                twrate_err = sum(abs(p - a) for p, a in zip(twrate_pred, actual))
                ens_err = sum(abs(p - a) for p, a in zip(ens_pred, actual))

                if prop_pred == actual: prop_exact += 1
                if mode_pred == actual: mode_exact += 1
                if twrate_pred == actual: twrate_exact += 1
                if ens_pred == actual: ens_exact += 1

                prop_abs_err += prop_err
                mode_abs_err += mode_err
                twrate_abs_err += twrate_err
                ens_abs_err += ens_err

                pool_str = f"{pools[0]}/{pools[1]}/{pools[2]}/{pools[3]}"
                prop_str = f"{prop_pred[0]}/{prop_pred[1]}/{prop_pred[2]}/{prop_pred[3]}"
                mode_str = f"{mode_pred[0]}/{mode_pred[1]}/{mode_pred[2]}/{mode_pred[3]}"
                twrate_str = f"{twrate_pred[0]}/{twrate_pred[1]}/{twrate_pred[2]}/{twrate_pred[3]}"
                ens_str = f"{ens_pred[0]}/{ens_pred[1]}/{ens_pred[2]}/{ens_pred[3]}"
                actual_str = f"{actual[0]}/{actual[1]}/{actual[2]}/{actual[3]}"

                print(f"{r['date']:<20} {pool_str:<15} {prop_str:<15} {mode_str:<15} {twrate_str:<15} {ens_str:<15} {actual_str:<12} "
                      f"{prop_err:<8} {mode_err:<8} {twrate_err:<8} {ens_err:<8}")

            history.append((r['pools_tuple'], r['counts_tuple']))

        print("-" * 100)
        print(f"Proportional model: Exact matches = {prop_exact}/{BACKTEST_N} ({prop_exact/BACKTEST_N:.0%}), Avg abs error = {prop_abs_err/BACKTEST_N:.2f}")
        print(f"Conditional Mode  : Exact matches = {mode_exact}/{BACKTEST_N} ({mode_exact/BACKTEST_N:.0%}), Avg abs error = {mode_abs_err/BACKTEST_N:.2f}")
        print(f"Time‑Weighted Rate: Exact matches = {twrate_exact}/{BACKTEST_N} ({twrate_exact/BACKTEST_N:.0%}), Avg abs error = {twrate_abs_err/BACKTEST_N:.2f}")
        print(f"Ensemble Model    : Exact matches = {ens_exact}/{BACKTEST_N} ({ens_exact/BACKTEST_N:.0%}), Avg abs error = {ens_abs_err/BACKTEST_N:.2f}")

    # ---------- PREDICTION ----------
    if future_date is not None:
        # Predict for the given future date
        future_dt = future_date
        # find the previous occurrence of this day before future_dt
        prev_draw = None
        for date_str, dt, main_nums in draws:
            if dt < future_dt:
                prev_draw = (date_str, dt, main_nums)
            else:
                break
        if prev_draw is None:
            print("No previous draw found for this day before the given future date.")
            return
        prev_date_str, prev_dt, _ = prev_draw
        target_date_str = future_dt.strftime('%a %d-%b-%Y')
        print(f"\nPrediction for {lottery_name} on {target_date_str}")
        print("="*90)
        # Window from prev_dt to future_dt (exclusive)
        window_nums = []
        for date_str, dt, day_abbr2, all_nums, _ in all_rows:
            if prev_dt <= dt < future_dt:
                valid_nums = [n for n in all_nums if 1 <= n <= max_num]
                window_nums.extend(valid_nums)
        print(f"Window: {prev_date_str} to {future_dt.strftime('%a %d-%b-%Y')}")
    else:
        # Predict next occurrence
        last_date_str, last_dt, last_main = draws[-1]
        next_date = last_dt + timedelta(days=1)
        while next_date.strftime('%a')[:3] != day_abbr:
            next_date += timedelta(days=1)
        target_date_str = next_date.strftime('%a %d-%b-%Y')
        print(f"\nPrediction for next {lottery_name} draw: {target_date_str}")
        print("="*90)
        # Window from last draw to next_date
        window_nums = []
        for date_str, dt, day_abbr2, all_nums, _ in all_rows:
            if last_dt <= dt < next_date:
                valid_nums = [n for n in all_nums if 1 <= n <= max_num]
                window_nums.extend(valid_nums)
        print(f"Window: {last_date_str} to {target_date_str}")

    counter = Counter(window_nums)
    eh = {n for n, cnt in counter.items() if cnt >= 4}
    h  = {n for n, cnt in counter.items() if cnt == 3}
    w  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
    c  = {n for n in range(1, max_num+1) if counter[n] == 0}

    eh_pool = len(eh)
    h_pool = len(h)
    w_pool = len(w)
    c_pool = len(c)

    print("\nPool details:")
    print(f"EH {sorted(eh)}  EH-Pool-Size: {eh_pool}")
    print(f"H  {sorted(h)}  H-Pool-Size: {h_pool}")
    print(f"W  {sorted(w)}  W-Pool-Size: {w_pool}")
    print(f"C  {sorted(c)}  C-Pool-Size: {c_pool}")
    print(f"EH+H Pool Size: {eh_pool + h_pool}")

    full_history = [(r['pools_tuple'], r['counts_tuple']) for r in results]

    prop_pred = predict_counts_proportional((eh_pool, h_pool, w_pool, c_pool), max_num, main_count)
    mode_pred = predict_counts_mode((eh_pool, h_pool, w_pool, c_pool), full_history, max_num, main_count, k=K_NEIGHBORS)
    twrate_pred = predict_counts_time_weighted_rate((eh_pool, h_pool, w_pool, c_pool), full_history, max_num, main_count)
    ens_pred = predict_counts_ensemble((eh_pool, h_pool, w_pool, c_pool), full_history, max_num, main_count, k=K_NEIGHBORS)

    print("\nProportional prediction:", prop_pred)
    print("Conditional mode prediction:", mode_pred)
    print("Time‑Weighted Rate prediction:", twrate_pred)
    print("Ensemble prediction:", ens_pred)
    print("(Use the one that performed best in backtest)")

# ---------- MAIN ----------
# Read all rows
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
        sfl_nums = extract_all_numbers(row[1])
        others_nums = extract_all_numbers(row[2]) if row[2] else []
        all_nums = sfl_nums + others_nums
        day_abbr = date_str[:3]
        all_rows.append((date_str, dt, day_abbr, all_nums, row[2] if row[2] else None))

all_rows.sort(key=lambda x: x[1])

# Group draws by day
draws_by_day = defaultdict(list)
for date_str, dt, day_abbr, _, others_cell in all_rows:
    if day_abbr in LOTTERY_CONFIG and others_cell:
        main_nums = extract_main_numbers(others_cell)
        if main_nums:
            draws_by_day[day_abbr].append((date_str, dt, main_nums))

# Determine what to process
if FUTURE_DATE_STR:
    # Parse the future date
    future_dt = parse_date(FUTURE_DATE_STR)
    day_abbr = FUTURE_DATE_STR[:3]
    if day_abbr not in LOTTERY_CONFIG:
        print(f"Error: '{day_abbr}' is not a supported lottery day.")
    else:
        lottery_name, max_num, main_count = LOTTERY_CONFIG[day_abbr]
        draws = draws_by_day.get(day_abbr, [])
        if not draws:
            print(f"No draws found for {day_abbr}.")
        else:
            process_lottery(day_abbr, draws, all_rows, max_num, main_count, lottery_name, future_dt)
else:
    # Process all lotteries and predict the next draw for each
    for day_abbr, config in LOTTERY_CONFIG.items():
        lottery_name, max_num, main_count = config
        draws = draws_by_day.get(day_abbr, [])
        if draws:
            process_lottery(day_abbr, draws, all_rows, max_num, main_count, lottery_name, future_date=None)
        else:
            print(f"\n{lottery_name} ({day_abbr}): No draws found. Skipping.")