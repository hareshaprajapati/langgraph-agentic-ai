import csv
from collections import defaultdict, Counter
import sys
from datetime import datetime, timedelta

# ================= CONFIGURATION =================
CSV_FILE = "Saturday_data.csv"
TARGET_NUMBER = 5                   # Default target number
WINDOW_DAYS = 270                   # For the attraction tables (not used in validation)
TOP_COOC = 10                       # top 10 co-occurrences
NEIGHBOUR_RANGE = 2                 # default for display

# Validation settings
N_TEST = 100                        # number of most recent draws to test offset hits
OFFSETS = [0, 1, -1, 2, -2]         # offsets to test (0 = direct)

# ================= DATE PARSING =================
def parse_date(date_str):
    return datetime.strptime(date_str.strip(), "%a %d-%b-%Y")

# ================= LOAD DRAWS =================
def load_draws(filename):
    draws = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                dt = parse_date(row[0])
                numbers = [int(x.strip()) for x in row[1].split(',') if x.strip()]
                if numbers:
                    draws.append((dt, numbers))
            except (ValueError, IndexError):
                continue
    draws.sort(key=lambda x: x[0])
    return draws

# ================= BUILD CO-OCCURRENCE =================
def build_cooccurrence(draws):
    cooc = defaultdict(Counter)
    for _, nums in draws:
        for i, n1 in enumerate(nums):
            for n2 in nums[i+1:]:
                cooc[n1][n2] += 1
                cooc[n2][n1] += 1
    return cooc

# ================= DIRECT ATTRACTION =================
def get_top_attracted(number, cooc, top_n=TOP_COOC):
    if number not in cooc:
        return []
    sorted_pairs = sorted(cooc[number].items(), key=lambda x: (-x[1], x[0]))
    return sorted_pairs[:top_n]

# ================= NEIGHBOUR ATTRACTION (for display) =================
def get_neighbour_attracted(number, cooc, top_n=TOP_COOC, neighbour_range=NEIGHBOUR_RANGE):
    if number not in cooc:
        return []
    top_pairs = sorted(cooc[number].items(), key=lambda x: (-x[1], x[0]))[:top_n]
    neighbour_scores = Counter()
    for cooc_num, cooc_count in top_pairs:
        for d in range(1, neighbour_range+1):
            if cooc_num - d >= 1:
                neighbour_scores[cooc_num - d] += cooc_count
            if cooc_num + d <= 45:
                neighbour_scores[cooc_num + d] += cooc_count
    neighbour_scores.pop(number, None)
    sorted_neighbours = sorted(neighbour_scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted_neighbours[:top_n]

# ================= OFFSET VALIDATION =================
def validate_offsets(all_draws, target_number, offsets, n_test=N_TEST):
    """
    For each of the last n_test draws that contain target_number,
    compute hit rates for each offset using prior draws only.
    Returns a dict: offset -> hit_rate.
    """
    start_test = len(all_draws) - n_test
    if start_test < 0:
        start_test = 0

    hit_counts = Counter()
    trial_counts = Counter()

    for i in range(start_test, len(all_draws)):
        target_date, nums = all_draws[i]
        if target_number not in nums:
            continue
        prior_draws = all_draws[:i]
        if len(prior_draws) < 10:
            continue

        cooc = build_cooccurrence(prior_draws)
        if target_number not in cooc:
            continue

        top_pairs = sorted(cooc[target_number].items(), key=lambda x: (-x[1], x[0]))[:TOP_COOC]
        top_nums = [n for n, _ in top_pairs]

        # Build candidate sets for each offset
        offset_sets = {}
        for off in offsets:
            if off == 0:
                offset_sets[off] = set(top_nums)
            else:
                s = set()
                for n in top_nums:
                    candidate = n + off
                    if 1 <= candidate <= 45:
                        s.add(candidate)
                offset_sets[off] = s

        # Other numbers in the draw
        others = [n for n in nums if n != target_number]

        for off, s in offset_sets.items():
            hits = sum(1 for n in others if n in s)
            hit_counts[off] += hits
            trial_counts[off] += len(others)

    rates = {}
    for off in offsets:
        if trial_counts[off] > 0:
            rates[off] = hit_counts[off] / trial_counts[off]
        else:
            rates[off] = 0.0
    return rates

# ================= MAIN =================
if __name__ == "__main__":
    # Optional command-line args: target_number [window_days]
    if len(sys.argv) > 1:
        try:
            TARGET_NUMBER = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number '{sys.argv[1]}'. Using default {TARGET_NUMBER}.")
    if len(sys.argv) > 2:
        try:
            WINDOW_DAYS = int(sys.argv[2])
        except ValueError:
            print(f"Invalid window '{sys.argv[2]}'. Using default {WINDOW_DAYS}.")

    print(f"Loading draws from '{CSV_FILE}'...")
    all_draws = load_draws(CSV_FILE)
    if not all_draws:
        print("No draws found. Check file path and format.")
        sys.exit(1)

    print(f"Total draws loaded: {len(all_draws)}")

    # ======= Offset validation =======
    print(f"\nValidating offsets for number {TARGET_NUMBER} on last {N_TEST} draws (walk‑forward)...")
    rates = validate_offsets(all_draws, TARGET_NUMBER, OFFSETS, N_TEST)
    print("\nOffset hit rates (proportion of other drawn numbers found):")
    print("-" * 50)
    for off in OFFSETS:
        label = "Direct (0)" if off == 0 else f"{off:+d}"
        print(f"  {label:>10}: {rates[off]:.4f}")

    # ======= Existing attraction tables (unchanged) =======
    # Overall co-occurrence
    overall_cooc = build_cooccurrence(all_draws)
    latest_date = all_draws[-1][0]
    cutoff = latest_date - timedelta(days=WINDOW_DAYS)
    window_draws = [(dt, nums) for dt, nums in all_draws if dt >= cutoff]
    window_cooc = build_cooccurrence(window_draws)

    top_overall_direct = get_top_attracted(TARGET_NUMBER, overall_cooc)
    top_window_direct = get_top_attracted(TARGET_NUMBER, window_cooc)
    top_overall_neighbour = get_neighbour_attracted(TARGET_NUMBER, overall_cooc)
    top_window_neighbour = get_neighbour_attracted(TARGET_NUMBER, window_cooc)

    max_rows_direct = max(len(top_overall_direct), len(top_window_direct))
    print("\n" + "="*100)
    print(f"DIRECT ATTRACTION: Top numbers co-occurring with {TARGET_NUMBER}")
    print("="*100)
    print(f"{'OVERALL (all history)':<50} {'LAST ' + str(WINDOW_DAYS) + ' DAYS':<50}")
    print("-"*100)
    for i in range(max_rows_direct):
        left_str = f"{top_overall_direct[i][0]:2d} (co-occurred {top_overall_direct[i][1]} times)" if i < len(top_overall_direct) else ""
        right_str = f"{top_window_direct[i][0]:2d} (co-occurred {top_window_direct[i][1]} times)" if i < len(top_window_direct) else ""
        print(f"{left_str:<50} {right_str}")

    max_rows_neighbour = max(len(top_overall_neighbour), len(top_window_neighbour))
    print("\n" + "="*100)
    print(f"NEIGHBOUR ATTRACTION (±{NEIGHBOUR_RANGE}) for {TARGET_NUMBER}")
    print("="*100)
    print(f"{'OVERALL (all history)':<50} {'LAST ' + str(WINDOW_DAYS) + ' DAYS':<50}")
    print("-"*100)
    for i in range(max_rows_neighbour):
        left_str = f"{top_overall_neighbour[i][0]:2d} (score {top_overall_neighbour[i][1]})" if i < len(top_overall_neighbour) else ""
        right_str = f"{top_window_neighbour[i][0]:2d} (score {top_window_neighbour[i][1]})" if i < len(top_window_neighbour) else ""
        print(f"{left_str:<50} {right_str}")

    print("="*100)