import pandas as pd
import re
import random
from collections import Counter


# --- SECTION 1 & 4: CORE FUNCTIONS ---
def parse_nums(s):
    if pd.isna(s) or s == "": return []
    return [int(n) for n in re.findall(r'\d+', s)]


def parse_main_6(s):
    if pd.isna(s) or s == "": return []
    match = re.search(r'\[(.*?)\]', s)
    if match:
        nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]
        return sorted(nums[:6])
    return []


def get_tiers(df_window):
    """Calculates EH, H, W, C tiers for a 7-day window [cite: 3-4]."""
    pool = []
    for _, row in df_window.iterrows():
        pool.extend(parse_nums(row['Set for Life (incl supp)']))
        pool.extend(parse_nums(row['Others (incl supp)']))
    counts = Counter([n for n in pool if 1 <= n <= 45])
    EH = sorted([n for n, c in counts.items() if c >= 4])  # EH: 4+ [cite: 4]
    H = sorted([n for n, c in counts.items() if c == 3])  # H: 3 [cite: 4]
    W = sorted([n for n, c in counts.items() if 1 <= c <= 2])  # W: 1-2 [cite: 4]
    C = sorted([n for n in range(1, 46) if n not in counts])  # C: 0 [cite: 4]
    if len(H) < 4:  # H-Pool Promotion [cite: 4]
        pseudo_h = [n for n, c in counts.items() if c == 2]
        H = sorted(list(set(H) | set(pseudo_h)))
    return EH, H, W, C, counts


def classify_sat(sat_main_6, prev_tiers):
    """Classifies Saturday as Breadth (W>=4) or Depth (W<=3) [cite: 5-6]."""
    w_pool = set(prev_tiers[2])
    w_hits = len([n for n in sat_main_6 if n in w_pool])
    return "Breadth" if w_hits >= 4 else "Depth"


def meets_ratios(nums):
    """Strict 3:3, 2:4, or 4:2 ratios for O/E and H/L[cite: 27]."""
    odd = len([n for n in nums if n % 2 != 0])
    high = len([n for n in nums if n >= 23])
    ratios = ['3:3', '2:4', '4:2']
    return f"{odd}:{6 - odd}" in ratios and f"{high}:{6 - high}" in ratios


def check_collision(ticket, existing):
    """Collision Shield: No two tickets share > 3 numbers[cite: 28]."""
    for t in existing:
        if len(set(ticket) & set(t)) > 3: return True
    return False


# --- SECTION 4, 6, 7 & 8: TICKET GENERATION ---
def generate_ticket(struct, strike, kill_dec, has_ac, has_bm, has_cp, tiers, legacy, pred_breadth, vibrations,
                    existing):
    EH, H, W, C = tiers
    attempts = 0
    while attempts < 1000:
        attempts += 1
        ticket = []

        def filter_dec(pool):
            return [n for n in pool if n // 10 != kill_dec] if kill_dec is not None else pool

        fEH, fH, fW, fC = map(filter_dec, [EH, H, W, C])
        fL = filter_dec(legacy)
        active_L = fL if strike == "Legacy" else []

        # 1. Anchors (Max 1 EH, At least 1 H) [cite: 19, 22]
        num_anchors = 3 if struct == "High" else 2
        anchors = []
        if fEH and random.random() < 0.6: anchors.append(random.choice(fEH))
        while len(anchors) < num_anchors:
            cand = [n for n in fH if n not in anchors]
            if not cand: cand = [n for n in fW if n not in anchors]
            if not cand: break
            anchors.append(random.choice(cand))
        if not any(n in fH for n in anchors): continue

        # Section 4: Anchor Couple (Mirror/Consecutive link) [cite: 20-21]
        if has_ac and len(anchors) >= 2:
            base = anchors[0]
            couples = [n for n in (fEH + fH + fW) if n != base and (n % 10 == base % 10 or abs(n - base) == 1)]
            if couples: anchors[1] = random.choice(couples)
        ticket.extend(anchors)

        # 2. Strike Slot (Legacy/Cold/Pure) [cite: 23, 29-31]
        if strike == "Legacy" and active_L:
            cand = [n for n in active_L if n not in ticket]
            if cand: ticket.append(random.choice(cand))
        elif strike == "Cold" and fC:
            cand = [n for n in fC if n not in ticket]
            if cand: ticket.append(random.choice(cand))

        # 3. Rule C: 40s Breadth-Maker [cite: 8-9, 40]
        if pred_breadth and not any(n >= 40 for n in ticket) and kill_dec != 4:
            p40s = [n for n in (fEH + fH + fW) if n >= 40 and n not in ticket]
            if p40s: ticket.append(random.choice(p40s))

        # 4. Section 8 Step 5: Mirror Bridge (70% Factor) [cite: 25, 49-53]
        if has_bm:
            has_m = any(ticket[i] % 10 == ticket[j] % 10 for i in range(len(ticket)) for j in range(i + 1, len(ticket)))
            if not has_m:
                h_anchors = [n for n in ticket if n in fH]
                base = random.choice(h_anchors) if h_anchors else random.choice(ticket)
                mirrs = [n for n in (fW + active_L + fH) if n % 10 == base % 10 and n != base and n not in ticket]
                v_mirrs = [n for n in mirrs if n in vibrations]
                if v_mirrs:
                    ticket.append(random.choice(v_mirrs))
                elif mirrs:
                    ticket.append(random.choice(mirrs))
                if len(ticket) < 5:  # Step 5.4 [cite: 52]
                    w_mirrs = [n for n in fW if n not in ticket]
                    w_pairs = [(a, b) for a in w_mirrs for b in w_mirrs if a % 10 == b % 10 and a < b]
                    if w_pairs: ticket.extend(list(random.choice(w_pairs)))

        # 5. Section 8 Bridge Method: Consecutive (60% Factor) [cite: 24, 35-48]
        if has_cp:
            has_p = any(abs(ticket[i] - ticket[j]) == 1 for i in range(len(ticket)) for j in range(i + 1, len(ticket)))
            if not has_p:
                h_anchs = [n for n in ticket if n in fH]
                base = random.choice(h_anchs) if h_anchs else random.choice(ticket)
                neibs = [n for n in fW if abs(n - base) == 1 and n not in ticket]
                if neibs:  # Step 1 [cite: 36]
                    ticket.append(random.choice(neibs))
                elif pred_breadth and kill_dec != 4:  # Step 2 [cite: 40]
                    p_opts = [(40, 41), (41, 42), (43, 44), (44, 45)]
                    valid = [p for p in p_opts if p[0] not in ticket and p[1] not in ticket]
                    if valid and len(ticket) <= 4: ticket.extend(list(random.choice(valid)))
                elif active_L:  # Step 4 [cite: 47]
                    l_num = active_L[0]
                    l_neibs = [n for n in (fH + fW) if abs(n - l_num) == 1 and n not in ticket]
                    if l_neibs: ticket.append(random.choice(l_neibs))

        # 6. Final Body Fill (Legacy Sweep Compliance) [cite: 34]
        while len(ticket) < 6:
            cand = [n for n in fW if n not in ticket and n not in (fL if strike != "Legacy" else [])]
            if not cand: cand = [n for n in (fH + fC) if
                                 n not in ticket and n not in (fL if strike != "Legacy" else [])]
            if not cand: break
            ticket.append(random.choice(cand))

        final_t = sorted(ticket[:6])
        l_hits = len(set(final_t) & set(legacy))
        if strike == "Legacy" and l_hits != 1: continue
        if strike != "Legacy" and l_hits != 0: continue
        if len(final_t) == 6 and meets_ratios(final_t) and not check_collision(final_t, existing):
            return final_t
    return None


def main():
    random.seed(109)
    N = 100
    df = pd.read_csv('cross_lotto_data.csv')
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y')
    df = df.sort_values('Date_dt')

    target_date = pd.to_datetime('2026-04-18')
    current_window = df[(df['Date_dt'] >= target_date - pd.Timedelta(days=7)) & (df['Date_dt'] < target_date)]
    midweek_window = df[
        (df['Date_dt'] >= target_date - pd.Timedelta(days=4)) & (df['Date_dt'] < target_date)]  # Tue-Fri

    EH, H, W, C, counts = get_tiers(current_window)
    print(EH)
    mEH, mH, mW, mC, m_counts = get_tiers(midweek_window)
    vibrations = [n for n, c in m_counts.items() if c >= 2]  # Mid-week velocity [cite: 44]

    # --- SECTION 2: MOMENTUM LOGIC [cite: 5-7] ---
    sats = df[(df['Date_dt'] < target_date) & (df['Date'].str.startswith('Sat'))].tail(5).copy()
    history = []
    for _, r in sats.iterrows():
        hist_win = df[(df['Date_dt'] >= r['Date_dt'] - pd.Timedelta(days=7)) & (df['Date_dt'] < r['Date_dt'])]
        if not hist_win.empty:
            history.append(classify_sat(parse_main_6(r['Others (incl supp)']), get_tiers(hist_win)))

    exhaustion = (len(history) >= 3 and all(h == "Breadth" for h in history[-3:]))  # [cite: 6]
    concentration = (len(set(n // 10 for n in (mEH + mH))) <= 2)  # [cite: 7]
    pred_breadth = not (exhaustion or concentration)

    # --- SECTION 8: DECADE ROTATION [cite: 54-59] ---
    dec_vols = Counter([n // 10 for n in m_counts.keys()])
    sorted_decs = sorted(range(5), key=lambda x: dec_vols[x], reverse=True)
    # Action C (50%) + Action A (30%) + Action B (20%)
    kills = ([sorted_decs[2]] * 40 + [sorted_decs[-1]] * 24 + [4] * 16 + [None] * 20)
    random.shuffle(kills)
    legacy = parse_main_6(df[df['Date_dt'] == sats.iloc[-1]['Date_dt']].iloc[0]['Others (incl supp)'])

    # --- SECTION 7: GLOBAL SATURATION AUDIT  ---
    while True:
        tickets = []
        for i in range(N):
            struct = "High" if i < int(N * 0.60) else "Standard"
            strike = "Legacy" if i < int(N * 0.50) else "Pure" if i < int(N * 0.90) else "Cold"
            t = generate_ticket(struct, strike, kills[i], (i < 42), (i < 70), (i < 60), (EH, H, W, C), legacy,
                                pred_breadth, vibrations, tickets)
            if t: tickets.append(t)
        all_nums = [n for t in tickets for n in t]
        if all(c <= N * 0.25 for c in Counter(all_nums).values()): break

    print(f"Global Saturation Audit: PASSED for N={N}.")
    # [Evaluation: Result 3, 8, 18, 39, 40, 41]
    real_res = [3, 8, 18, 39, 40, 41]
    hit_counts = Counter([len(set(t) & set(real_res)) for t in tickets])
    print(f"\n--- Strike Audit (April 18 Result: {real_res}) ---")
    for h in sorted(hit_counts.keys(), reverse=True):
        print(f"{h}-Hits: {hit_counts[h]} tickets")


if __name__ == "__main__":
    main()