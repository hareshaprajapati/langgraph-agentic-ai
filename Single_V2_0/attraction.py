import csv
from collections import defaultdict, Counter
import sys
from datetime import datetime, timedelta

# ================= CONFIGURATION =================
CSV_FILE = "Saturday_data.csv"      # Change to your actual file path
TARGET_NUMBER = 8                   # Default target number
WINDOW_DAYS = 270                   # Last 1 year (365 days)

# ================= DATE PARSING =================
def parse_date(date_str):
    """Parse date like 'Sat 29-Aug-2026' to datetime."""
    return datetime.strptime(date_str.strip(), "%a %d-%b-%Y")

# ================= LOAD DRAWS =================
def load_draws(filename):
    """
    Reads the CSV and returns a list of (date, draw) tuples.
    Expected CSV format: Date,Main
      Sat 29-Aug-2026,"17,23,25,27,33,45"
    """
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
    """
    Given a list of (date, numbers) tuples, return dict: number -> Counter of other numbers.
    """
    cooc = defaultdict(Counter)
    for _, numbers in draws:
        for i, n1 in enumerate(numbers):
            for n2 in numbers[i+1:]:
                cooc[n1][n2] += 1
                cooc[n2][n1] += 1
    return cooc

# ================= GET TOP ATTRACTED =================
def get_top_attracted(number, cooc, top_n=10):
    """Return list of (number, count) sorted by count descending."""
    if number not in cooc:
        return []
    sorted_pairs = sorted(cooc[number].items(), key=lambda x: (-x[1], x[0]))
    return sorted_pairs[:top_n]

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

    # Overall co-occurrence (all data)
    overall_cooc = build_cooccurrence(all_draws)

    # Last window_days co-occurrence
    latest_date = all_draws[-1][0]
    cutoff = latest_date - timedelta(days=WINDOW_DAYS)
    window_draws = [(dt, nums) for dt, nums in all_draws if dt >= cutoff]
    print(f"Draws in last {WINDOW_DAYS} days: {len(window_draws)}")
    window_cooc = build_cooccurrence(window_draws)

    # Results for target number
    top_overall = get_top_attracted(TARGET_NUMBER, overall_cooc, top_n=10)
    top_window = get_top_attracted(TARGET_NUMBER, window_cooc, top_n=10)

    # Determine the maximum number of rows to print
    max_rows = max(len(top_overall), len(top_window))

    # Print side-by-side table
    print("\n" + "="*100)
    print(f"Top 10 numbers most attracted to {TARGET_NUMBER} OVERALL (all history):  Top 10 numbers most attracted to {TARGET_NUMBER} in last {WINDOW_DAYS} days:")
    print("="*100)

    for i in range(max_rows):
        # Left column (overall)
        if i < len(top_overall):
            num, count = top_overall[i]
            left_str = f"{num:2d}  (co-occurred {count} times)"
        else:
            left_str = ""

        # Right column (window)
        if i < len(top_window):
            num, count = top_window[i]
            right_str = f"{num:2d}  (co-occurred {count} times)"
        else:
            right_str = ""

        # Print with fixed width for alignment (adjust width as needed)
        print(f"{left_str:<50s} {right_str}")

    print("="*100)