import pandas as pd
import ast
from collections import defaultdict, Counter
import itertools

# ---------- POOLS ----------
EH = [1, 2, 5, 11, 14, 15, 17, 22, 27, 28, 32, 34, 43]
H  = [16, 18, 19, 30, 38, 39, 40, 42, 44]
W  = [3, 4, 7, 8, 9, 10, 12, 13, 20, 21, 23, 24, 25, 26, 29, 35, 36, 37, 41, 45, 46]
C  = [6, 31, 33, 47]
ALL_NUMS = set(range(1, 48))

def parse_main_numbers(cell):
    if pd.isna(cell) or cell == '':
        return []
    start = cell.find('[')
    if start == -1:
        return []
    end = cell.find(']', start)
    if end == -1:
        return []
    try:
        nums = ast.literal_eval(cell[start:end+1])
        if isinstance(nums, list):
            return [int(x) for x in nums if 1 <= x <= 47]
    except:
        return []
    return []

# ---------- READ DATA ----------
df = pd.read_csv('cross_lotto_data_backup.csv', encoding='utf-8')
others_col = [c for c in df.columns if 'others' in c.lower()][0]
df['Others_main'] = df[others_col].apply(parse_main_numbers)
df = df[df['Others_main'].apply(len) == 7].copy()
df['Date'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Filter only Tuesdays
tuesdays = df[df['Date'].dt.day_name() == 'Tuesday'].copy().reset_index(drop=True)
print(f"Total Tuesday draws: {len(tuesdays)}")

# ---------- BACKTEST LOOP ----------
results = []
for i in range(1, len(tuesdays)):
    seed_draw = tuesdays.iloc[i-1]['Others_main']
    actual_draw = tuesdays.iloc[i]['Others_main']
    seed_date = tuesdays.iloc[i-1]['Date']

    # ---- Build Co-occurrence & Transition from PAST data only ----
    past_tuesdays = tuesdays[tuesdays['Date'] < seed_date]
    cooc_counter = Counter()
    trans_counter = defaultdict(Counter)

    # Co-occurrence
    for _, row in past_tuesdays.iterrows():
        nums = row['Others_main']
        for pair in itertools.combinations(sorted(nums), 2):
            cooc_counter[pair] += 1

    # Unigram transitions (Tuesday -> next Tuesday)
    for j in range(len(past_tuesdays)-1):
        curr = past_tuesdays.iloc[j]['Others_main']
        nxt = past_tuesdays.iloc[j+1]['Others_main']
        for num in curr:
            for nxt_num in nxt:
                trans_counter[num][nxt_num] += 1

    # ---- Score candidates ----
    candidate_scores = defaultdict(float)
    for seed_num in seed_draw:
        # Transitions
        for nxt_num, cnt in trans_counter.get(seed_num, {}).items():
            candidate_scores[nxt_num] += cnt * 1.0  # weight 1
        # Co-occurrence: find pairs that contain seed_num
        for (a, b), cnt in cooc_counter.items():
            if a == seed_num:
                candidate_scores[b] += cnt * 0.7  # weight 0.7
            elif b == seed_num:
                candidate_scores[a] += cnt * 0.7

    # ---- Select top candidates per pool ----
    eh_candidates = [(n, s) for n, s in candidate_scores.items() if n in EH]
    h_candidates  = [(n, s) for n, s in candidate_scores.items() if n in H]
    w_candidates  = [(n, s) for n, s in candidate_scores.items() if n in W]
    # (ignore C)

    eh_candidates.sort(key=lambda x: -x[1])
    h_candidates.sort(key=lambda x: -x[1])
    w_candidates.sort(key=lambda x: -x[1])

    # Fallback if not enough candidates
    if len(eh_candidates) < 3:
        for n in EH:
            if n not in [x[0] for x in eh_candidates]:
                eh_candidates.append((n, 0))
    if not h_candidates:
        for n in H:
            h_candidates.append((n, 0))
    if len(w_candidates) < 3:
        for n in W:
            if n not in [x[0] for x in w_candidates]:
                w_candidates.append((n, 0))

    # ---- Generate 6 tickets (varying choices) ----
    top_eh = [x[0] for x in eh_candidates[:5]]
    top_h = [x[0] for x in h_candidates[:3]]
    top_w = [x[0] for x in w_candidates[:6]]

    # Fallback: ensure at least 3 EH, 1 H, 3 W
    if len(top_eh) < 3:
        for n in EH:
            if n not in top_eh:
                top_eh.append(n)
                if len(top_eh) == 3:
                    break
    if not top_h:
        top_h = [x for x in H if x in seed_draw][:1] or [30]  # fallback H
    if len(top_w) < 3:
        for n in W:
            if n not in top_w:
                top_w.append(n)
                if len(top_w) == 3:
                    break

    tickets = []
    for ticket_idx in range(6):
        # Pick EH trio (rotate for variety)
        eh_choice = sorted(top_eh[ticket_idx % len(top_eh):] + top_eh[:ticket_idx % len(top_eh)])[:3]
        # Pick H (cycle)
        h_choice = top_h[ticket_idx % len(top_h)]
        # Pick 3 W (cycle)
        w_choices = top_w[ticket_idx % len(top_w):] + top_w[:ticket_idx % len(top_w)]
        w_choice = w_choices[:3]
        if len(w_choice) < 3:
            w_choice = [x for x in seed_draw if x in W][:3]
        ticket = sorted(set(eh_choice + [h_choice] + w_choice))
        # Ensure 7 numbers (if duplicates or short, pad with seed numbers)
        while len(ticket) < 7:
            for n in seed_draw:
                if n not in ticket:
                    ticket.append(n)
                    break
            else:
                # fallback to top_w[0]
                ticket.append(top_w[0])
        ticket = sorted(ticket[:7])
        tickets.append(ticket)

    # ---- Evaluate ----
    max_hits = 0
    best_ticket = tickets[0] if tickets else sorted(seed_draw)  # fallback
    for ticket in tickets:
        hits = len(set(ticket) & set(actual_draw))
        if hits > max_hits:
            max_hits = hits
            best_ticket = ticket

    results.append({
        'seed_date': seed_date.strftime('%d-%b-%Y'),
        'actual': sorted(actual_draw),
        'best_ticket': sorted(best_ticket),
        'max_hits': max_hits
    })

# ---------- PRINT RESULTS ----------
print("\n" + "="*70)
print("BACKTEST RESULTS: Hybrid Method (Transitions + Co-occurrence)")
print("="*70)

total_hits = 0
jackpots = []
for r in results:
    total_hits += r['max_hits']
    if r['max_hits'] >= 6:
        jackpots.append(r)
    print(f"{r['seed_date']} -> {r['actual']} | Best: {r['best_ticket']} | Hits: {r['max_hits']}")

avg_hits = total_hits / len(results)
print("\n" + "="*70)
print(f"Total transitions tested: {len(results)}")
print(f"Average Max Hits per draw: {avg_hits:.2f} (Random expectation: ~1.09)")
print(f"Jackpots (6 or 7 hits): {len(jackpots)} times")
for r in jackpots:
    print(f"  - {r['seed_date']}: {r['max_hits']} hits -> Actual: {r['actual']}")