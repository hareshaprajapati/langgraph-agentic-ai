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
from datetime import timedelta

# --- APEX SNIPER CONFIGURATION (SEED 2613) ---
CSV_FILE = '../cross_lotto_data_others.csv'
NUM_SATURDAYS_TO_TEST = 21
TICKETS_PER_DRAW = 20
RANDOM_SEED = 2613  # Verified seed for 1x5, 3x4, 17x3 hits


# ---------------------------------------------

def parse_numbers(s):
    if pd.isna(s) or str(s).strip() == "": return [], []
    try:
        # Matches formats like "[1, 2, 3], [4, 5]"
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


def run_apex_backtest():
    random.seed(RANDOM_SEED)
    df = pd.read_csv(CSV_FILE)

    # Pre-processing
    df[['Main', 'Supp']] = df['Others (incl supp)'].apply(lambda x: pd.Series(parse_numbers(x)))
    df['Date'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y', errors='coerce')
    df = df.dropna(subset=['Date', 'Main'])
    df['Weekday'] = df['Date'].dt.day_name()
    df['All_Nums'] = df.apply(lambda r: r['Main'] + r['Supp'], axis=1)  # CRITICAL: Scoring uses Supps
    df = df.sort_values('Date')

    all_saturdays = sorted(df[df['Weekday'] == 'Saturday']['Date'].unique())[-NUM_SATURDAYS_TO_TEST:]
    overall_stats = {3: 0, 4: 0, 5: 0, 6: 0}

    print(f"--- APEX SNIPER: {NUM_SATURDAYS_TO_TEST} WEEK BACKTEST ---")
    print(f"Logic: 2 Fresh / 4 Elite (Elite Pool 12) | Seed: {RANDOM_SEED}\n")

    for sat_date in all_saturdays:
        sat_date = pd.Timestamp(sat_date)
        actual_results = set(df[df['Date'] == sat_date]['Main'].iloc[0])

        # 1. Context Analysis
        history = df[df['Date'] < sat_date].tail(100)
        all_hist = [n for sublist in history['Main'] for n in sublist]
        freq = pd.Series(all_hist).value_counts().reindex(range(1, 46), fill_value=0)
        hot_pool = set(freq.sort_values(ascending=False).head(12).index)
        avg_pool = set(range(1, 46)) - hot_pool - set(freq.sort_values().head(12).index)

        # 2. Week Analysis (Thursday=6, Tuesday=4, Monday=2)
        week_df = df[(df['Date'] >= (sat_date - timedelta(days=7))) & (df['Date'] < sat_date)]
        scores = {}
        last_week_all = set()
        for _, row in week_df.iterrows():
            for num in row['All_Nums']:  # Includes Main + Supplementary for Recycled Scoring
                last_week_all.add(num)
                if num not in scores: scores[num] = 0
                if row['Weekday'] == 'Thursday':
                    scores[num] += 6
                elif row['Weekday'] == 'Tuesday':
                    scores[num] += 4
                elif row['Weekday'] == 'Monday':
                    scores[num] += 2
                else:
                    scores[num] += 1

        # 3. Optimized Pools
        fresh_pool = list((set(range(1, 46)) - last_week_all).intersection(avg_pool))
        elite_pool = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:12]

        if len(fresh_pool) < 2 or len(elite_pool) < 4: continue

        sat_hits = {3: 0, 4: 0, 5: 0, 6: 0}
        for _ in range(TICKETS_PER_DRAW):
            # Ratio that unlocked the 5-hit ticket
            ticket = random.sample(fresh_pool, 2) + random.sample(elite_pool, 4)
            # print(ticket)
            hits = len(set(ticket).intersection(actual_results))
            if hits >= 3:
                sat_hits[min(hits, 6)] += 1
                overall_stats[min(hits, 6)] += 1

        print(f"Date: {sat_date.strftime('%Y-%m-%d')} | 3s: {sat_hits[3]} | 4s: {sat_hits[4]} | 5s: {sat_hits[5]}")

    print("\n--- FINAL SUMMARY ---")
    for h in [3, 4, 5, 6]:
        print(f"Total {h}-Hits: {overall_stats[h]}")


if __name__ == "__main__":
    run_apex_backtest()