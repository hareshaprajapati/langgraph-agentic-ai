import csv
from collections import defaultdict, Counter
from datetime import datetime

# ================= CONFIGURATION =================
CSV_FILE = "Saturday_data.csv"
TEST_LAST_N = 100
TOP_N = 10
NEIGHBOUR_RANGE = 1   # ±1, ±2

# ================= DATE PARSING =================
def parse_date(date_str):
    return datetime.strptime(date_str.strip(), "%a %d-%b-%Y")

# ================= LOAD DATA =================
def load_draws(filename):
    draws = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                dt = parse_date(row[0])
                nums = [int(x.strip()) for x in row[1].split(',') if x.strip()]
                if nums:
                    draws.append((dt, nums))
            except (ValueError, IndexError):
                continue
    draws.sort(key=lambda x: x[0])
    return draws

# ================= CO-OCCURRENCE =================
def build_cooccurrence(draws):
    cooc = defaultdict(Counter)
    for _, nums in draws:
        for i, n1 in enumerate(nums):
            for n2 in nums[i+1:]:
                cooc[n1][n2] += 1
                cooc[n2][n1] += 1
    return cooc

# ================= NEIGHBOURS =================
def get_neighbours(numbers, offset_range=NEIGHBOUR_RANGE):
    neighbours = set()
    for n in numbers:
        for d in range(1, offset_range+1):
            if n-d >= 1:
                neighbours.add(n-d)
            if n+d <= 45:
                neighbours.add(n+d)
    return neighbours

# ================= WALK-FORWARD TEST (ALL HISTORY) =================
def evaluate_direct_vs_neighbour_all_history(draws, test_last_n):
    direct_hits = 0
    neighbour_hits = 0
    total_possible = 0
    start_test = len(draws) - test_last_n

    for i in range(start_test, len(draws)):
        target_date, target_nums = draws[i]
        # Use ALL previous draws (no window)
        historical_draws = draws[:i]
        if not historical_draws:
            continue

        cooc = build_cooccurrence(historical_draws)

        for idx, n in enumerate(target_nums):
            others = [m for j, m in enumerate(target_nums) if j != idx]
            total_possible += len(others)

            # Direct: top10 co-occurrences
            if n not in cooc:
                direct_set = set()
            else:
                top_direct = [num for num, _ in sorted(cooc[n].items(), key=lambda x: (-x[1], x[0]))[:TOP_N]]
                direct_set = set(top_direct)

            # Neighbour: neighbours of those top co-occurrences (excluding direct numbers themselves)
            neighbour_set = set()
            if direct_set:
                neighbour_set = get_neighbours(direct_set, NEIGHBOUR_RANGE) - direct_set

            # Count hits
            direct_hits += sum(1 for m in others if m in direct_set)
            neighbour_hits += sum(1 for m in others if m in neighbour_set)

    if total_possible == 0:
        return 0, 0
    return direct_hits / total_possible, neighbour_hits / total_possible

# ================= MAIN =================
if __name__ == "__main__":
    print(f"Loading draws from '{CSV_FILE}'...")
    all_draws = load_draws(CSV_FILE)
    print(f"Total draws loaded: {len(all_draws)}")

    direct_avg, neighbour_avg = evaluate_direct_vs_neighbour_all_history(all_draws, TEST_LAST_N)

    print(f"\nWalk-forward comparison over last {TEST_LAST_N} draws (ALL HISTORY):")
    print(f"Direct attraction hit rate: {direct_avg:.4f}")
    print(f"Neighbour attraction hit rate (±{NEIGHBOUR_RANGE}): {neighbour_avg:.4f}")