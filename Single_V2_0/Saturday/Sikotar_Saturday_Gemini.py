import pandas as pd
import re
import random
from datetime import timedelta

# --- CONFIGURATION ---
CSV_FILE = '../cross_lotto_data_others.csv'
TOTAL_TICKETS = 20
RANDOM_SEED = 13  # Your "Lucky Seed"
REAL_DRAW_RESULT = [4, 11, 26, 30, 42, 43]


# ---------------------

def parse_numbers(s):
    if pd.isna(s) or str(s).strip() == "":
        return [], []
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


def generate_lotto_tickets():
    random.seed(RANDOM_SEED)

    # 1. Load Data
    df = pd.read_csv(CSV_FILE)
    df[['Main', 'Supp']] = df['Others (incl supp)'].apply(lambda x: pd.Series(parse_numbers(x)))
    df['Date'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df['All_Numbers'] = df.apply(lambda row: row['Main'] + row['Supp'], axis=1)
    df['Weekday'] = df['Date'].dt.day_name()
    df = df.sort_values('Date')

    last_date = df['Date'].max()
    days_to_sat = (5 - last_date.weekday() + 7) % 7
    if days_to_sat == 0: days_to_sat = 7
    target_dt = last_date + timedelta(days=days_to_sat)

    # 2. Historical Frequency
    history = df[df['Date'] < target_dt].tail(100)
    all_history_nums = [n for sublist in history['All_Numbers'] for n in sublist]
    freq_counts = pd.Series(all_history_nums).value_counts().reindex(range(1, 46), fill_value=0)

    sorted_freq = freq_counts.sort_values(ascending=False)
    hot_pool = set(sorted_freq.head(11).index)
    avg_pool = set(range(1, 46)) - hot_pool - set(sorted_freq.tail(11).index)

    # 3. Weekly Analysis
    past_week_df = df[(df['Date'] >= (target_dt - timedelta(days=7))) & (df['Date'] < target_dt)]
    num_to_days = {}
    numbers_last_week = set()
    for _, row in past_week_df.iterrows():
        for num in row['All_Numbers']:
            numbers_last_week.add(num)
            if num not in num_to_days: num_to_days[num] = set()
            num_to_days[num].add(row['Weekday'])

    # 4. Power Scoring
    recycled_scores = {}
    for num, days in num_to_days.items():
        score = 0
        if 'Thursday' in days: score += 3
        if 'Tuesday' in days: score += 2
        if 'Monday' in days: score += 1
        if 'Wednesday' in days: score += 1
        if 'Saturday' in days: score += 1
        if 'Monday' in days and 'Thursday' in days: score += 3
        if 'Tuesday' in days and 'Thursday' in days: score += 2
        recycled_scores[num] = score

    # 5. Narrowed Candidate Pools
    fresh_pool = set(range(1, 46)) - numbers_last_week
    candidate_fresh_avg = list(fresh_pool.intersection(avg_pool))

    # SORT AND NARROW TO TOP 7
    recycled_avg = [n for n in recycled_scores if n in avg_pool]
    recycled_avg.sort(key=lambda x: recycled_scores[x], reverse=True)
    top_recycled_avg = recycled_avg[:7]  # NARROWED FROM 10 TO 7

    recycled_hot = [n for n in recycled_scores if n in hot_pool]
    recycled_hot.sort(key=lambda x: recycled_scores[x], reverse=True)
    top_recycled_hot = recycled_hot[:3]  # NARROWED FOR CONCENTRATION

    # 6. Build
    print(f"Targeting: {target_dt.strftime('%Y-%m-%d')} | Seed: {RANDOM_SEED}")
    all_hits = []

    for i in range(1, TOTAL_TICKETS + 1):
        ticket = random.sample(candidate_fresh_avg, 1)  # 1 Fresh

        if top_recycled_hot:
            ticket += random.sample(top_recycled_hot, 1)  # 1 Focused Hot Recycled
        else:
            ticket += random.sample(recycled_avg[7:12], 1)

        ticket += random.sample(top_recycled_avg, 4)  # 4 Focused Avg Recycled
        ticket.sort()

        hits = set(ticket).intersection(set(REAL_DRAW_RESULT))
        all_hits.append(len(hits))
        print(f"Ticket {i:02}: {ticket} (Hits: {len(hits)}) Matched: {sorted(list(hits)) if hits else ''}")

    print(f"\nSummary: Max Hits {max(all_hits)} | 3+ Hits: {len([h for h in all_hits if h >= 3])}")


if __name__ == "__main__":
    generate_lotto_tickets()