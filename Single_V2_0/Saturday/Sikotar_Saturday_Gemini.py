import pandas as pd
import re
import random
from datetime import timedelta

# --- CONFIGURATION ---
CSV_FILE = '../cross_lotto_data_others.csv'
NUM_SATURDAYS_TO_TEST = 21  # Number of Saturdays to backtest
TICKETS_PER_DRAW = 20
RANDOM_SEED = 101


# ---------------------

def parse_numbers(s):
    if pd.isna(s) or str(s).strip() == "": return [], []
    try:
        match = re.search(r'\[(.*?)\]\s*,\s*\[(.*?)\]', str(s))
        if match:
            main = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
            supp = [int(x.strip()) for x in match.group(2).split(',') if x.strip()]
            return main, supp
        match_single = re.search(r'\[(.*?)\]', str(s))
        if match_single:
            main = [int(x.strip()) for x in match_single.group(1).split(',') if x.strip()]
            return main, []
        return [], []
    except:
        return [], []


def run_backtest():
    random.seed(RANDOM_SEED)
    df = pd.read_csv(CSV_FILE)
    df[['Main', 'Supp']] = df['Others (incl supp)'].apply(lambda x: pd.Series(parse_numbers(x)))
    df['Date'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y', errors='coerce')
    df = df.dropna(subset=['Date', 'Main'])
    df['All_Numbers'] = df.apply(lambda row: row['Main'] + row['Supp'], axis=1)
    df['Weekday'] = df['Date'].dt.day_name()
    df = df.sort_values('Date')

    # Find the last N Saturdays in the dataset
    all_saturdays = df[df['Weekday'] == 'Saturday']['Date'].unique()
    test_saturdays = sorted(all_saturdays)[-NUM_SATURDAYS_TO_TEST:]

    overall_stats = {3: 0, 4: 0, 5: 0, 6: 0}

    print(f"--- STARTING BACKTEST FOR {NUM_SATURDAYS_TO_TEST} SATURDAYS ---\n")

    for sat_date in test_saturdays:
        sat_date = pd.Timestamp(sat_date)
        actual_results = set(df[df['Date'] == sat_date]['Main'].iloc[0])

        # 1. Analysis Window (100 draws before this Saturday for Hot/Cold)
        history = df[df['Date'] < sat_date].tail(100)
        all_hist = [n for sublist in history['All_Numbers'] for n in sublist]
        freq = pd.Series(all_hist).value_counts().reindex(range(1, 46), fill_value=0)

        hot_pool = set(freq.sort_values(ascending=False).head(11).index)
        avg_pool = set(range(1, 46)) - hot_pool - set(freq.sort_values().head(11).index)

        # 2. Weekly Analysis (Recycled vs Fresh)
        week_df = df[(df['Date'] >= (sat_date - timedelta(days=7))) & (df['Date'] < sat_date)]
        num_to_days = {}
        nums_last_week = set()
        for _, row in week_df.iterrows():
            for num in row['All_Numbers']:
                nums_last_week.add(num)
                if num not in num_to_days: num_to_days[num] = set()
                num_to_days[num].add(row['Weekday'])

        # 3. Power Scoring
        scores = {}
        for num, days in num_to_days.items():
            s = 0
            if 'Thursday' in days: s += 3
            if 'Tuesday' in days: s += 2
            if 'Monday' in days: s += 1
            if 'Wednesday' in days: s += 1
            if 'Monday' in days and 'Thursday' in days: s += 3
            if 'Tuesday' in days and 'Thursday' in days: s += 2
            scores[num] = s

        # 4. Pools
        fresh_avg = list((set(range(1, 46)) - nums_last_week).intersection(avg_pool))
        recycled_avg = sorted([n for n in scores if n in avg_pool], key=lambda x: scores[x], reverse=True)[:7]
        recycled_hot = sorted([n for n in scores if n in hot_pool], key=lambda x: scores[x], reverse=True)[:3]

        # 5. Generate and Check
        sat_hits = {3: 0, 4: 0, 5: 0, 6: 0}
        for _ in range(TICKETS_PER_DRAW):
            # Logic: 1 Fresh, 1 Hot Recycled, 4 Avg Recycled
            ticket = random.sample(fresh_avg, 1) + random.sample(recycled_hot, 1) + random.sample(recycled_avg, 4)
            hits = len(set(ticket).intersection(actual_results))
            if hits >= 3:
                sat_hits[hits] += 1
                overall_stats[hits] += 1

        print(
            f"Date: {sat_date.strftime('%Y-%m-%d')} | Results: {sorted(list(actual_results))} | Hits: 3s={sat_hits[3]}, 4s={sat_hits[4]}, 5s={sat_hits[5]}")

    print("\n--- FINAL BACKTEST SUMMARY ---")
    print(f"Total Tickets Tested: {NUM_SATURDAYS_TO_TEST * TICKETS_PER_DRAW}")
    for h, count in overall_stats.items():
        print(f"Total {h}-Hit Tickets: {count}")


if __name__ == "__main__":
    run_backtest()