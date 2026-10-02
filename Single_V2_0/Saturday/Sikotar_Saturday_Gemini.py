import sys
import os
from datetime import datetime

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except OSError:
                pass

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except OSError:
                pass

log_file_path = os.path.join(
    ".",
    "Sikotar_Saturday_Gemini.py.log"   # single growing log file
)

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import pandas as pd
import re
import random
import csv
from pathlib import Path
from datetime import timedelta

# --- APEX SNIPER CONFIGURATION (SEED 2613) ---
# The source header contains unquoted commas, so it is read with csv.reader
# instead of pandas' header-based parser.
CSV_FILE = Path(__file__).resolve().parent.parent / 'cross_lotto_data_backup.csv'

# Choose exactly one: "prediction" or "backtest".
# Prediction mode accepts only a future Saturday that is absent from the data.
# Backtest mode generates tickets from earlier rows, then reads the held-out
# result only to measure performance.
RUN_MODE = "backtest"
PREDICTION_DATE = "2026-10-03"
BACKTEST_DRAWS = 20
BACKTEST_VERBOSE = False

TICKETS_PER_DRAW = 20
RANDOM_SEED = 2613


# ---------------------------------------------

def parse_numbers(value):
    if pd.isna(value) or str(value).strip() == "":
        return [], []
    try:
        # Matches formats like "[1, 2, 3], [4, 5]"
        match = re.search(r'\[(.*?)\]\s*,\s*\[(.*?)\]', str(value))
        if match:
            main = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
            supp = [int(x.strip()) for x in match.group(2).split(',') if x.strip()]
            return main, supp
        match_single = re.search(r'\[(.*?)\]', str(value))
        if match_single:
            main = [int(x.strip()) for x in match_single.group(1).split(',') if x.strip()]
            return main, []
        return [], []
    except (TypeError, ValueError):
        return [], []


def load_lotto_data(csv_file: Path) -> pd.DataFrame:
    """Load the daily "Others" field without trusting its malformed header."""
    rows = []
    with open(csv_file, newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle)
        next(reader, None)  # Header has unquoted commas; data rows remain valid.
        for raw in reader:
            if len(raw) < 3:
                continue
            main, supp = parse_numbers(raw[-1])
            if not main:
                continue
            rows.append({'Date': raw[0].strip(), 'Main': main, 'Supp': supp})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No usable lottery rows found in {csv_file}")
    df['Date'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).copy()
    df['Weekday'] = df['Date'].dt.day_name()
    df['All_Nums'] = df.apply(lambda row: row['Main'] + row['Supp'], axis=1)
    return df.sort_values('Date').reset_index(drop=True)


def generate_tickets_from_history(history: pd.DataFrame, sat_date: pd.Timestamp, rng: random.Random):
    """Build tickets from a history frame that excludes sat_date and all future rows."""
    recent_history = history.tail(100)
    all_hist = [number for numbers in recent_history['Main'] for number in numbers]
    freq = pd.Series(all_hist).value_counts().reindex(range(1, 46), fill_value=0)
    hot_pool = set(freq.sort_values(ascending=False).head(12).index)
    cold_pool = set(freq.sort_values(ascending=True).head(12).index)
    average_pool = set(range(1, 46)) - hot_pool - cold_pool

    week_df = history[history['Date'] >= (sat_date - timedelta(days=7))]
    scores = {}
    last_week_all = set()
    for _, row in week_df.iterrows():
        for number in row['All_Nums']:
            if not 1 <= number <= 45:
                continue
            last_week_all.add(number)
            scores.setdefault(number, 0)
            if row['Weekday'] == 'Thursday':
                scores[number] += 6
            elif row['Weekday'] == 'Tuesday':
                scores[number] += 4
            elif row['Weekday'] == 'Monday':
                scores[number] += 2
            else:
                scores[number] += 1

    fresh_pool = sorted((set(range(1, 46)) - last_week_all).intersection(average_pool))
    elite_pool = sorted(scores, key=lambda number: (-scores[number], number))[:12]
    if len(fresh_pool) < 2 or len(elite_pool) < 4:
        raise RuntimeError(
            f"Insufficient candidate pools for {sat_date:%Y-%m-%d}: "
            f"fresh={len(fresh_pool)}, elite={len(elite_pool)}"
        )

    tickets = []
    for _ in range(TICKETS_PER_DRAW):
        # Fresh and elite pools are disjoint: fresh numbers were absent last week,
        # while elite numbers were scored from that week.
        ticket = sorted(rng.sample(fresh_pool, 2) + rng.sample(elite_pool, 4))
        if len(ticket) != 6:
            raise RuntimeError("Ticket generation produced duplicate numbers")
        tickets.append(ticket)
    return tickets


def run_prediction(df: pd.DataFrame) -> None:
    target_date = pd.Timestamp(PREDICTION_DATE)
    if pd.isna(target_date):
        raise ValueError("PREDICTION_DATE must be parseable as YYYY-MM-DD")
    if target_date.dayofweek != 5:
        raise ValueError("PREDICTION_DATE must be a Saturday")

    latest_saturday = df[(df['Weekday'] == 'Saturday') & (df['Main'].map(len) == 6)]['Date'].max()
    if target_date <= latest_saturday:
        raise ValueError(
            "Prediction mode only accepts a date later than the newest Saturday result "
            f"({latest_saturday:%Y-%m-%d}). Use RUN_MODE = 'backtest' for known draws."
        )

    history = df[df['Date'] < target_date].copy()
    tickets = generate_tickets_from_history(history, target_date, random.Random(RANDOM_SEED))
    print("=== PREDICTION MODE ===")
    print(f"Target date: {target_date:%Y-%m-%d}")
    print(f"Latest known draw used: {history['Date'].max():%Y-%m-%d}")
    print("No result is loaded or compared in prediction mode.")
    for index, ticket in enumerate(tickets, 1):
        print(f"Ticket #{index:02d}: {ticket}")


def run_backtest(df: pd.DataFrame) -> None:
    if BACKTEST_DRAWS < 1:
        raise ValueError("BACKTEST_DRAWS must be at least 1")
    saturday_rows = df[(df['Weekday'] == 'Saturday') & (df['Main'].map(len) == 6)].sort_values('Date')
    if len(saturday_rows) < BACKTEST_DRAWS + 1:
        raise ValueError("Not enough Saturday results for the requested backtest")

    held_out_rows = saturday_rows.tail(BACKTEST_DRAWS)
    rng = random.Random(RANDOM_SEED)
    ticket_hits = {hits: 0 for hits in range(7)}
    best_hits = []

    print(f"=== APEX SNIPER BACKTEST: LAST {BACKTEST_DRAWS} SATURDAYS ===")
    print("No-look-ahead guard: each ticket uses only rows dated before its held-out draw.")
    for _, held_out_row in held_out_rows.iterrows():
        sat_date = held_out_row['Date']
        history = df[df['Date'] < sat_date].copy()

        # Generate first. The actual result is read only after this call returns.
        tickets = generate_tickets_from_history(history, sat_date, rng)
        actual_results = set(held_out_row['Main'])
        hits = [len(set(ticket).intersection(actual_results)) for ticket in tickets]
        best_hit = max(hits, default=0)
        best_hits.append(best_hit)
        for hit in hits:
            ticket_hits[hit] += 1

        high_hits = ", ".join(
            f"#{index + 1:02d} ({hit})" for index, hit in enumerate(hits) if hit >= 3
        ) or "none"
        print(
            f"{sat_date:%Y-%m-%d} | history={len(history)} | actual={sorted(actual_results)} | "
            f"best={best_hit} | 3+ tickets: {high_hits}"
        )
        if BACKTEST_VERBOSE:
            for index, ticket in enumerate(tickets, 1):
                print(f"  Ticket #{index:02d}: {ticket}")

    weeks_ge = {threshold: sum(best >= threshold for best in best_hits) for threshold in range(3, 7)}
    print(f"\n=== BACKTEST SUMMARY (LAST {BACKTEST_DRAWS} SATURDAYS) ===")
    print(f"Tickets generated: {BACKTEST_DRAWS * TICKETS_PER_DRAW}")
    print(f"Weeks with 3+ hits: {weeks_ge[3]}")
    print(f"Weeks with 4+ hits: {weeks_ge[4]}")
    print(f"Weeks with 5+ hits: {weeks_ge[5]}")
    print(f"Weeks with 6+ hits: {weeks_ge[6]}")
    print(f"Best ticket hit count: {max(best_hits, default=0)}")
    print(f"Tickets with exactly 3 hits: {ticket_hits[3]}")
    print(f"Tickets with exactly 4 hits: {ticket_hits[4]}")
    print(f"Tickets with exactly 5 hits: {ticket_hits[5]}")
    print(f"Jackpot tickets (6 hits): {ticket_hits[6]}")
    print("Jackpot result: HIT" if ticket_hits[6] else "Jackpot result: no jackpot hit")


def main() -> None:
    df = load_lotto_data(CSV_FILE)
    mode = RUN_MODE.strip().lower()
    if mode == 'prediction':
        run_prediction(df)
    elif mode == 'backtest':
        run_backtest(df)
    else:
        raise ValueError("RUN_MODE must be either 'prediction' or 'backtest'")

if __name__ == "__main__":
    main()
