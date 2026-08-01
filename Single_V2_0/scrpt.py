import csv, random, itertools
from datetime import datetime as dt, timedelta
from collections import Counter

CSV_FILE = "cross_lotto_data_backup.csv"

# ---------- Helpers (unchanged) ----------
def parse_numbers(cell):
    nums = []
    if not cell: return nums
    for part in cell.split('],'):
        part = part.strip().lstrip('[').rstrip(']')
        for token in part.split(','):
            token = token.strip()
            if token.isdigit():
                n = int(token)
                if 1 <= n <= 45: nums.append(n)
    return nums

def parse_others_main(others_str):
    main_part = others_str.split('],')[0].strip()
    if main_part.startswith('['): main_part = main_part[1:]
    main_part = main_part.replace(']', '').strip()
    if not main_part: return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]

def decade_of(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

def consecutive(t): return any(sorted(t)[i+1]-sorted(t)[i]==1 for i in range(len(t)-1))
def mirror(t): return len({x%10 for x in t}) < len(t)
def score(t): return (2 if consecutive(t) else 0) + (2 if mirror(t) else 0)
def overlap(a,b): return len(set(a)&set(b))

def band_label(c):
    if c>=5: return '5x+'
    if c==4: return '4x'
    if c==3: return '3x'
    if c==2: return '2x'
    if c==1: return '1x'
    return '0x'

# ---------- Load all rows ----------
rows = []
with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    sfl_col = others_col = None
    for col in reader.fieldnames:
        if 'Set for Life' in col: sfl_col = col
        if 'Others' in col: others_col = col
    for row in reader:
        dt_str = row['Date'].strip()
        try: d = dt.strptime(dt_str, '%a %d-%b-%Y')
        except: continue
        sfl = parse_numbers(row[sfl_col]) if row[sfl_col] else []
        oth = parse_numbers(row[others_col]) if row[others_col] else []
        rows.append((d, sfl, oth, row[others_col]))

saturdays = []
for d, sfl, oth, oth_raw in rows:
    if d.weekday() == 5:
        main = parse_others_main(oth_raw)
        if len(main) == 6: saturdays.append((d, main))
saturdays.sort(key=lambda x: x[0])

# ---------- Configuration ----------
TEST_DRAWS = 20
TOTAL_TICKETS = 50
SAMPLE_SIZE = 10000
FREQ_CAP = 25
OVERLAP_CAP = 4
BAND_CAPS = {'0x':3, '1x':3, '2x':4, '3x':4, '4x':3, '5x+':3}

AVG_EH = 1.0
AVG_H  = 1.5
AVG_W  = 2.5
AVG_C  = 0.5

random.seed(42)

overall = {'tested':0, 'jackpot_in_pool':0, 'jackpot_selected':0,
           'structurally_impossible':0, 'total_3plus':0,
           'total_4plus':0, 'total_5plus':0, 'total_6':0}

def weighted_choice(population, weights, k):
    chosen = []
    pop = list(population)
    w = list(weights)
    for _ in range(k):
        if not pop: break
        total_w = sum(w)
        if total_w == 0:
            idx = random.randrange(len(pop))
        else:
            r = random.uniform(0, total_w)
            cum = 0
            for i, weight in enumerate(w):
                cum += weight
                if r <= cum:
                    idx = i
                    break
        chosen.append(pop[idx])
        pop.pop(idx)
        w.pop(idx)
    return chosen

for idx in range(len(saturdays)-TEST_DRAWS, len(saturdays)):
    target_dt, target_main = saturdays[idx]
    REAL = set(target_main)
    jackpot_ticket = tuple(sorted(REAL))

    print(f"\n{'='*80}")
    print(f"📅 {target_dt.strftime('%d‑%b‑%Y')}  |  Winning: {sorted(REAL)}")
    print(f"{'='*80}")

    # ---- 7‑day window ----
    wstart = target_dt - timedelta(days=7)
    wend   = target_dt - timedelta(days=1)
    window_nums = []
    for d, s, o, _ in rows:
        if wstart <= d <= wend:
            window_nums.extend(s); window_nums.extend(o)
    freq7 = Counter(window_nums)

    EH = [n for n in range(1,46) if freq7.get(n,0) >= 4]
    H  = [n for n in range(1,46) if freq7.get(n,0) == 3]
    W  = [n for n in range(1,46) if 1 <= freq7.get(n,0) <= 2]
    C  = [n for n in range(1,46) if freq7.get(n,0) == 0]

    prev_idx = idx - 1
    LEGACY = saturdays[prev_idx][1] if prev_idx >= 0 else []

    print(f"Pools: EH={len(EH)} H={len(H)} W={len(W)} C={len(C)}")

    # ---- Cheat: kill exactly the one missing decade ----
    present_decs = set(decade_of(n) for n in REAL)
    all_decs = {'0s','10s','20s','30s','40s'}
    missing_decs = all_decs - present_decs
    if not missing_decs:
        print("❌ All five decades are present – no single‑decade kill possible. Skipping.")
        overall['structurally_impossible'] += 1
        overall['tested'] += 1
        continue
    kill_dec = '40s' if '40s' in missing_decs else sorted(missing_decs)[0]
    print(f"🔧 Killing missing decade '{kill_dec}' (absent from real draw).")

    # ---- 20‑week bands ----
    prev_20_main = []
    for j in range(max(0, idx-20), idx):
        prev_20_main.extend(saturdays[j][1])
    freq20 = Counter(prev_20_main)
    band_for_num = {n: band_label(freq20.get(n,0)) for n in range(1,46)}

    # ---- Pre‑filter pools ----
    eh_use = [n for n in EH if decade_of(n) != kill_dec]
    h_use  = [n for n in H  if decade_of(n) != kill_dec]
    w_use  = [n for n in W  if decade_of(n) != kill_dec]
    c_use  = [n for n in C  if decade_of(n) != kill_dec]

    total_eh = TOTAL_TICKETS * AVG_EH
    total_h  = TOTAL_TICKETS * AVG_H
    total_w  = TOTAL_TICKETS * AVG_W
    total_c  = TOTAL_TICKETS * AVG_C

    def ideal_band(tp, sz):
        if sz == 0: return (0,0)
        avg = tp / sz
        lo = max(0, int(avg) - 2)
        hi = int(avg) + 2
        return (lo, hi)

    def distance_from_ideal(freq_cnt, pools, ideals):
        total = 0
        for pool, ideal in zip(pools, ideals):
            target = (ideal[0] + ideal[1]) / 2
            for n in pool:
                total += (freq_cnt[n] - target) ** 2
        return total

    eh_ideal = ideal_band(total_eh, len(eh_use))
    h_ideal  = ideal_band(total_h,  len(h_use))
    w_ideal  = ideal_band(total_w,  len(w_use))
    c_ideal  = ideal_band(total_c,  len(c_use))
    pools = [eh_use, h_use, w_use, c_use]
    ideals = [eh_ideal, h_ideal, w_ideal, c_ideal]

    # ---- Ticket validator (decade kill already handled by pools) ----
    def ticket_valid(t):
        if sum(1 for n in t if n in LEGACY) > 2: return False
        o = sum(1 for n in t if n%2)
        extreme_oe = (o,6-o) in [(1,5),(5,1)]
        if not extreme_oe and (o,6-o) not in [(3,3),(2,4),(4,2)]: return False
        lo = sum(1 for n in t if n<=22)
        extreme_hl = (lo,6-lo) in [(1,5),(5,1)]
        if not extreme_hl and (lo,6-lo) not in [(3,3),(2,4),(4,2)]: return False
        if max(Counter(decade_of(n) for n in t).values()) > 3: return False
        if sum(1 for n in t if 40<=n<=45) > 2: return False
        if not (consecutive(t) or mirror(t)): return False
        band_cnt = Counter(band_for_num[n] for n in t)
        if any(band_cnt[b] > BAND_CAPS[b] for b in band_cnt): return False
        return True

    # ---- Generate weighted random tickets ----
    random_tickets = set()
    attempts = 0
    while len(random_tickets) < SAMPLE_SIZE and attempts < SAMPLE_SIZE*20:
        attempts += 1
        eh_c = random.randint(0, min(2, len(eh_use)))
        max_h = min(3, len(h_use), 6 - eh_c)
        h_c = random.randint(0, max_h)
        w_c = random.randint(2, min(3, len(w_use)))
        c_c = 6 - eh_c - h_c - w_c
        if c_c < 0 or c_c > min(1, len(c_use)): continue
        try:
            eh = weighted_choice(eh_use, [freq7.get(n,1) for n in eh_use], eh_c) if eh_c else []
            h  = weighted_choice(h_use,  [freq7.get(n,1) for n in h_use],  h_c)  if h_c  else []
            w  = weighted_choice(w_use,  [freq7.get(n,1) for n in w_use],  w_c)  if w_c  else []
            c  = weighted_choice(c_use,  [freq7.get(n,1) for n in c_use],  c_c)  if c_c  else []
        except ValueError:
            continue
        t = tuple(sorted(eh + h + w + c))
        if len(set(t)) < 6: continue
        if not ticket_valid(t): continue
        random_tickets.add(t)

    print(f"Generated {len(random_tickets)} weighted valid tickets.")
    all_candidates = list(random_tickets)
    if jackpot_ticket not in all_candidates:
        all_candidates.append(jackpot_ticket)
    random.shuffle(all_candidates)

    # ---------- NATURAL SELECTION (NO INJECTION) ----------
    selected = []
    sel_set = set()
    freq_use = Counter()

    for slot in range(TOTAL_TICKETS):
        best_t = None
        best_key = None
        avail = [t for t in all_candidates if t not in sel_set]
        if not avail: break
        for t in avail:
            if any(overlap(t, s) > OVERLAP_CAP for s in selected): continue
            if any(freq_use[n] >= FREQ_CAP for n in t): continue
            hotness = sum(freq7.get(n,0) for n in t)
            sc = score(t)
            max_dec = max(Counter(decade_of(n) for n in t).values())
            temp_freq = freq_use.copy()
            for n in t: temp_freq[n] += 1
            dist = distance_from_ideal(temp_freq, pools, ideals)
            key = (dist, -sc - 0.0001 * hotness, max_dec)
            if best_key is None or key < best_key:
                best_key = key
                best_t = t
        if best_t is None: break
        selected.append(best_t)
        sel_set.add(best_t)
        for n in best_t: freq_use[n] += 1

    total_sel = len(selected)
    jp_selected = jackpot_ticket in sel_set
    hit_counts = Counter()
    for t in selected:
        hits = len(set(t) & REAL)
        hit_counts[hits] += 1

    print(f"Tickets selected: {total_sel}")
    print(f"Hits → 3:{hit_counts.get(3,0)}  4:{hit_counts.get(4,0)}  5:{hit_counts.get(5,0)}  6:{hit_counts.get(6,0)}")

    # ---------- DETAILED DIAGNOSTICS ----------
    # Jackpot ticket profile
    jt_eh = sum(1 for n in jackpot_ticket if n in EH)
    jt_h  = sum(1 for n in jackpot_ticket if n in H)
    jt_w  = sum(1 for n in jackpot_ticket if n in W)
    jt_c  = sum(1 for n in jackpot_ticket if n in C)
    jt_leg = sum(1 for n in jackpot_ticket if n in LEGACY)
    jt_sc = score(jackpot_ticket)
    jt_hot = sum(freq7.get(n,0) for n in jackpot_ticket)
    jt_bands = Counter(band_for_num[n] for n in jackpot_ticket)
    jt_dec_max = max(Counter(decade_of(n) for n in jackpot_ticket).values())
    jt_dist0 = distance_from_ideal(Counter({n:1 for n in jackpot_ticket}), pools, ideals)
    jt_key0 = (jt_dist0, -jt_sc - 0.0001 * jt_hot, jt_dec_max)

    print(f"\n🔎 JACKPOT TICKET PROFILE")
    print(f"   Numbers: {sorted(jackpot_ticket)}")
    print(f"   EH={jt_eh}, H={jt_h}, W={jt_w}, C={jt_c} | Legacy={jt_leg}")
    print(f"   PairScore={jt_sc}, Hotness={jt_hot}")
    print(f"   20‑week bands: {dict(jt_bands)}")
    print(f"   Initial key (empty freq): {jt_key0}")

    if jp_selected:
        print("   ✅ JACKPOT WAS SELECTED!")
        # Show which slot it was picked in
        for i, t in enumerate(selected):
            if t == jackpot_ticket:
                print(f"   Picked at slot {i+1}")
                break
    else:
        print("   ❌ JACKPOT NOT SELECTED")
        # Check why
        blocked_ov = any(overlap(jackpot_ticket, s) > OVERLAP_CAP for s in selected)
        blocked_freq = any(freq_use[n] >= FREQ_CAP for n in jackpot_ticket)
        if blocked_ov:
            print("   ⛔ BLOCKED by collision shield")
            for s in selected:
                if overlap(jackpot_ticket, s) > OVERLAP_CAP:
                    print(f"      Overlaps with: {sorted(s)}")
                    break
        elif blocked_freq:
            print("   ⛔ BLOCKED by frequency cap")
            for n in jackpot_ticket:
                if freq_use[n] >= FREQ_CAP:
                    print(f"      {n} used {freq_use[n]} times (cap={FREQ_CAP})")
        else:
            print("   ⚠️  OUTSCORED – available but not selected")
            # Show keys of first 10 selected tickets vs jackpot
            print("   First 10 selected tickets (key at selection):")
            for i, t in enumerate(selected[:10], 1):
                t_hot = sum(freq7.get(n,0) for n in t)
                t_sc = score(t)
                t_dec_max = max(Counter(decade_of(n) for n in t).values())
                # approximate key with initial distance (not exact, but indicative)
                t_dist0 = distance_from_ideal(Counter({n:1 for n in t}), pools, ideals)
                t_key0 = (t_dist0, -t_sc - 0.0001 * t_hot, t_dec_max)
                mark = " ← JACKPOT" if t == jackpot_ticket else ""
                print(f"   {i:2d}: {sorted(t)} hot={t_hot} sc={t_sc} key0={t_key0}{mark}")

    # Final frequency of jackpot numbers
    print(f"   Final usage of jackpot numbers:")
    for n in sorted(jackpot_ticket):
        print(f"      {n}: {freq_use[n]}")
    print(f"   Total tickets picked: {total_sel}")

    overall['tested'] += 1
    overall['jackpot_in_pool'] += 1
    if jp_selected: overall['jackpot_selected'] += 1
    if hit_counts.get(3,0): overall['total_3plus'] += 1
    if hit_counts.get(4,0): overall['total_4plus'] += 1
    if hit_counts.get(5,0): overall['total_5plus'] += 1
    if hit_counts.get(6,0): overall['total_6'] += 1

# ---------- Final Summary ----------
print("\n" + "="*80)
print("📊 OVERALL SUMMARY (20 draws, kill‑cheat, NO injection)")
print("="*80)
print(f"Draws tested: {overall['tested']}")
print(f"Structurally impossible (all decades present): {overall['structurally_impossible']}")
print(f"Jackpot ticket in pool: {overall['jackpot_in_pool']}/{overall['tested']}")
print(f"Jackpot ticket NATURALLY SELECTED: {overall['jackpot_selected']}/{overall['tested']}")
print(f"≥3 matches: {overall['total_3plus']}")
print(f"≥4 matches: {overall['total_4plus']}")
print(f"≥5 matches: {overall['total_5plus']}")
print(f"6 matches (jackpot): {overall['total_6']}")



================================================================================
📅 14‑Mar‑2026  |  Winning: [14, 21, 27, 34, 36, 40]
================================================================================
Pools: EH=9 H=12 W=22 C=2
🔧 Killing missing decade '0s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:6  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [14, 21, 27, 34, 36, 40]
   EH=1, H=1, W=4, C=0 | Legacy=1
   PairScore=2, Hotness=16
   20‑week bands: {'3x': 2, '1x': 2, '4x': 1, '2x': 1}
   Initial key (empty freq): (1774.0, -2.0016, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [11, 24, 37, 38, 41, 43] hot=13 sc=4 key0=(1762.0, -4.0013, 2)
    2: [13, 22, 28, 30, 31, 40] hot=13 sc=4 key0=(1762.0, -4.0013, 2)
    3: [10, 18, 27, 28, 33, 39] hot=13 sc=4 key0=(1762.0, -4.0013, 2)
    4: [11, 14, 15, 20, 35, 36] hot=15 sc=4 key0=(1764.0, -4.0015, 3)
    5: [16, 28, 29, 34, 44, 45] hot=15 sc=4 key0=(1766.0, -4.0015, 2)
    6: [11, 12, 17, 21, 24, 32] hot=11 sc=4 key0=(1764.0, -4.0011, 3)
    7: [18, 19, 26, 28, 42, 44] hot=14 sc=4 key0=(1766.0, -4.0014, 2)
    8: [11, 15, 20, 25, 36, 37] hot=14 sc=4 key0=(1764.0, -4.0014, 2)
    9: [10, 11, 23, 27, 33, 39] hot=14 sc=4 key0=(1764.0, -4.0014, 2)
   10: [12, 13, 22, 28, 31, 35] hot=13 sc=4 key0=(1762.0, -4.0013, 2)
   Final usage of jackpot numbers:
      14: 8
      21: 8
      27: 8
      34: 8
      36: 8
      40: 8
   Total tickets picked: 50

================================================================================
📅 21‑Mar‑2026  |  Winning: [11, 16, 20, 27, 43, 45]
================================================================================
Pools: EH=14 H=9 W=16 C=6
🔧 Killing missing decade '0s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:1  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [11, 16, 20, 27, 43, 45]
   EH=1, H=1, W=4, C=0 | Legacy=1
   PairScore=0, Hotness=14
   20‑week bands: {'1x': 1, '2x': 4, '3x': 1}
   Initial key (empty freq): (1883.0, -0.0014, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [15, 16, 26, 30, 37, 40] hot=15 sc=4 key0=(1869.0, -4.0015, 2)
    2: [17, 19, 23, 24, 33, 38] hot=15 sc=4 key0=(1869.0, -4.0015, 2)
    3: [11, 15, 19, 20, 36, 39] hot=15 sc=4 key0=(1869.0, -4.0015, 3)
    4: [13, 17, 20, 27, 28, 37] hot=13 sc=4 key0=(1869.0, -4.0013, 3)
    5: [11, 19, 24, 26, 44, 45] hot=13 sc=4 key0=(1869.0, -4.0013, 2)
    6: [16, 17, 23, 24, 36, 37] hot=15 sc=4 key0=(1869.0, -4.0015, 2)
    7: [15, 20, 26, 27, 30, 33] hot=15 sc=4 key0=(1869.0, -4.0015, 3)
    8: [17, 19, 20, 38, 39, 40] hot=15 sc=4 key0=(1869.0, -4.0015, 2)
    9: [13, 15, 24, 27, 28, 37] hot=13 sc=4 key0=(1869.0, -4.0013, 3)
   10: [16, 17, 20, 26, 44, 45] hot=13 sc=4 key0=(1869.0, -4.0013, 2)
   Final usage of jackpot numbers:
      11: 10
      16: 9
      20: 11
      27: 10
      43: 6
      45: 9
   Total tickets picked: 50

================================================================================
📅 28‑Mar‑2026  |  Winning: [1, 2, 3, 25, 29, 30]
================================================================================
Pools: EH=10 H=9 W=23 C=3
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:3  4:0  5:0  6:1

🔎 JACKPOT TICKET PROFILE
   Numbers: [1, 2, 3, 25, 29, 30]
   EH=0, H=1, W=5, C=0 | Legacy=0
   PairScore=2, Hotness=12
   20‑week bands: {'3x': 2, '0x': 1, '1x': 2, '2x': 1}
   Initial key (empty freq): (1763.0, -2.0012, 3)
   ✅ JACKPOT WAS SELECTED!
   Picked at slot 28
   Final usage of jackpot numbers:
      1: 7
      2: 6
      3: 7
      25: 11
      29: 7
      30: 7
   Total tickets picked: 50

================================================================================
📅 04‑Apr‑2026  |  Winning: [2, 4, 5, 13, 14, 37]
================================================================================
Pools: EH=11 H=10 W=21 C=3
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:4  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [2, 4, 5, 13, 14, 37]
   EH=1, H=3, W=2, C=0 | Legacy=1
   PairScore=4, Hotness=17
   20‑week bands: {'1x': 2, '3x': 1, '2x': 2, '0x': 1}
   Initial key (empty freq): (1764.0, -4.0017, 3)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [3, 15, 19, 26, 27, 36] hot=13 sc=4 key0=(1760.0, -4.0013, 2)
    2: [5, 6, 12, 13, 28, 33] hot=13 sc=4 key0=(1760.0, -4.0013, 2)
    3: [6, 10, 11, 14, 22, 32] hot=13 sc=4 key0=(1760.0, -4.0013, 3)
    4: [3, 13, 15, 20, 21, 36] hot=13 sc=4 key0=(1760.0, -4.0013, 2)
    5: [5, 7, 10, 11, 19, 25] hot=12 sc=4 key0=(1760.0, -4.0012, 3)
    6: [6, 13, 14, 23, 33, 39] hot=13 sc=4 key0=(1760.0, -4.0013, 2)
    7: [3, 4, 11, 18, 19, 34] hot=18 sc=4 key0=(1764.0, -4.0018, 3)
    8: [5, 8, 14, 15, 16, 37] hot=12 sc=4 key0=(1764.0, -4.0012, 3)
    9: [2, 5, 6, 15, 24, 38] hot=17 sc=4 key0=(1764.0, -4.0017, 3)
   10: [9, 11, 13, 22, 30, 31] hot=17 sc=4 key0=(1770.0, -4.0017, 2)
   Final usage of jackpot numbers:
      2: 8
      4: 7
      5: 11
      13: 10
      14: 10
      37: 6
   Total tickets picked: 50

================================================================================
📅 11‑Apr‑2026  |  Winning: [8, 11, 15, 32, 33, 44]
================================================================================
Pools: EH=9 H=14 W=18 C=4
🔧 Killing missing decade '20s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:2  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [8, 11, 15, 32, 33, 44]
   EH=1, H=1, W=3, C=1 | Legacy=0
   PairScore=2, Hotness=13
   20‑week bands: {'5x+': 2, '2x': 2, '1x': 1, '3x': 1}
   Initial key (empty freq): (1839.0, -2.0013, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [2, 7, 16, 19, 32, 33] hot=18 sc=4 key0=(1829.0, -4.0018, 2)
    2: [3, 13, 18, 37, 38, 44] hot=18 sc=4 key0=(1829.0, -4.0018, 2)
    3: [6, 14, 16, 31, 34, 35] hot=18 sc=4 key0=(1829.0, -4.0018, 3)
    4: [9, 12, 18, 19, 33, 39] hot=17 sc=4 key0=(1829.0, -4.0017, 3)
    5: [4, 14, 15, 30, 35, 38] hot=15 sc=4 key0=(1829.0, -4.0015, 3)
    6: [5, 10, 17, 18, 36, 37] hot=17 sc=4 key0=(1833.0, -4.0017, 3)
    7: [11, 16, 33, 35, 40, 45] hot=15 sc=2 key0=(1829.0, -2.0015, 2)
    8: [1, 6, 7, 14, 19, 37] hot=14 sc=4 key0=(1829.0, -4.0014, 3)
    9: [3, 4, 13, 18, 32, 38] hot=20 sc=4 key0=(1833.0, -4.002, 2)
   10: [12, 14, 19, 30, 31, 34] hot=20 sc=4 key0=(1833.0, -4.002, 3)
   Final usage of jackpot numbers:
      8: 7
      11: 8
      15: 8
      32: 9
      33: 10
      44: 8
   Total tickets picked: 50

================================================================================
📅 18‑Apr‑2026  |  Winning: [3, 8, 18, 39, 40, 41]
================================================================================
Pools: EH=12 H=12 W=16 C=5
🔧 Killing missing decade '20s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:6  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [3, 8, 18, 39, 40, 41]
   EH=0, H=3, W=3, C=0 | Legacy=1
   PairScore=4, Hotness=14
   20‑week bands: {'4x': 1, '5x+': 1, '0x': 2, '2x': 1, '3x': 1}
   Initial key (empty freq): (2084.0, -4.0014, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [7, 8, 18, 36, 40, 43] hot=15 sc=4 key0=(2084.0, -4.0015, 2)
    2: [9, 11, 16, 31, 32, 41] hot=15 sc=4 key0=(2084.0, -4.0015, 2)
    3: [4, 7, 12, 19, 37, 38] hot=13 sc=4 key0=(2084.0, -4.0013, 2)
    4: [3, 6, 9, 10, 30, 39] hot=14 sc=4 key0=(2086.0, -4.0014, 3)
    5: [12, 13, 31, 36, 41, 44] hot=19 sc=4 key0=(2088.0, -4.0019, 2)
    6: [2, 8, 10, 11, 39, 40] hot=15 sc=4 key0=(2086.0, -4.0015, 2)
    7: [3, 4, 14, 15, 31, 37] hot=15 sc=4 key0=(2088.0, -4.0015, 2)
    8: [7, 8, 9, 33, 35, 38] hot=17 sc=4 key0=(2088.0, -4.0017, 3)
    9: [18, 19, 30, 36, 39, 41] hot=14 sc=4 key0=(2084.0, -4.0014, 3)
   10: [2, 4, 5, 10, 37, 42] hot=12 sc=4 key0=(2090.0, -4.0012, 3)
   Final usage of jackpot numbers:
      3: 7
      8: 14
      18: 7
      39: 13
      40: 7
      41: 14
   Total tickets picked: 50

================================================================================
📅 25‑Apr‑2026  |  Winning: [3, 11, 12, 14, 17, 45]
================================================================================
Pools: EH=11 H=7 W=24 C=3
🔧 Killing missing decade '20s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:0  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [3, 11, 12, 14, 17, 45]
   EH=1, H=0, W=5, C=0 | Legacy=1
   PairScore=2, Hotness=12
   20‑week bands: {'5x+': 1, '3x': 4, '1x': 1}
   Initial key (empty freq): (2425.0, -2.0012, 4)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [1, 4, 14, 18, 42, 43] hot=13 sc=4 key0=(2351.0, -4.0013, 2)
    2: [4, 18, 19, 34, 36, 44] hot=13 sc=4 key0=(2351.0, -4.0013, 2)
    3: [1, 4, 16, 17, 34, 37] hot=13 sc=4 key0=(2351.0, -4.0013, 2)
    4: [1, 2, 8, 18, 31, 34] hot=15 sc=4 key0=(2353.0, -4.0015, 3)
    5: [4, 6, 7, 18, 34, 45] hot=14 sc=4 key0=(2353.0, -4.0014, 3)
    6: [1, 4, 11, 12, 18, 35] hot=13 sc=4 key0=(2353.0, -4.0013, 3)
    7: [1, 4, 5, 10, 34, 43] hot=11 sc=4 key0=(2351.0, -4.0011, 3)
    8: [1, 2, 17, 18, 34, 44] hot=13 sc=4 key0=(2351.0, -4.0013, 2)
    9: [4, 16, 18, 19, 31, 34] hot=13 sc=4 key0=(2351.0, -4.0013, 3)
   10: [1, 4, 7, 34, 36, 37] hot=15 sc=4 key0=(2353.0, -4.0015, 3)
   Final usage of jackpot numbers:
      3: 5
      11: 8
      12: 8
      14: 8
      17: 8
      45: 8
   Total tickets picked: 50

================================================================================
📅 02‑May‑2026  |  Winning: [9, 18, 19, 29, 34, 45]
================================================================================
Pools: EH=10 H=12 W=22 C=1
❌ All five decades are present – no single‑decade kill possible. Skipping.

================================================================================
📅 09‑May‑2026  |  Winning: [2, 11, 12, 17, 33, 37]
================================================================================
Pools: EH=11 H=8 W=22 C=4
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:1  4:1  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [2, 11, 12, 17, 33, 37]
   EH=0, H=1, W=4, C=1 | Legacy=0
   PairScore=4, Hotness=10
   20‑week bands: {'2x': 3, '4x': 1, '5x+': 1, '1x': 1}
   Initial key (empty freq): (1673.0, -4.001, 3)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [4, 10, 16, 26, 27, 33] hot=13 sc=4 key0=(1661.0, -4.0013, 2)
    2: [3, 17, 18, 21, 36, 38] hot=13 sc=4 key0=(1661.0, -4.0013, 2)
    3: [4, 5, 22, 24, 30, 37] hot=13 sc=4 key0=(1661.0, -4.0013, 2)
    4: [6, 16, 17, 22, 32, 38] hot=13 sc=4 key0=(1661.0, -4.0013, 2)
    5: [3, 14, 18, 24, 29, 30] hot=13 sc=4 key0=(1661.0, -4.0013, 2)
    6: [1, 16, 17, 26, 33, 35] hot=13 sc=4 key0=(1661.0, -4.0013, 2)
    7: [2, 4, 11, 18, 30, 31] hot=14 sc=4 key0=(1665.0, -4.0014, 2)
    8: [3, 4, 25, 26, 34, 38] hot=11 sc=4 key0=(1661.0, -4.0011, 2)
    9: [17, 18, 23, 24, 33, 36] hot=12 sc=4 key0=(1661.0, -4.0012, 2)
   10: [12, 16, 21, 22, 26, 30] hot=12 sc=4 key0=(1661.0, -4.0012, 3)
   Final usage of jackpot numbers:
      2: 7
      11: 7
      12: 6
      17: 10
      33: 9
      37: 7
   Total tickets picked: 50

================================================================================
📅 16‑May‑2026  |  Winning: [3, 10, 23, 32, 33, 39]
================================================================================
Pools: EH=12 H=9 W=17 C=7
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:4  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [3, 10, 23, 32, 33, 39]
   EH=2, H=1, W=2, C=1 | Legacy=1
   PairScore=4, Hotness=16
   20‑week bands: {'5x+': 2, '1x': 4}
   Initial key (empty freq): (1748.0, -4.0016, 3)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [7, 9, 10, 19, 20, 32] hot=15 sc=4 key0=(1728.0, -4.0015, 2)
    2: [5, 16, 18, 22, 34, 35] hot=15 sc=4 key0=(1728.0, -4.0015, 2)
    3: [14, 17, 26, 29, 37, 39] hot=13 sc=2 key0=(1728.0, -2.0013, 2)
    4: [5, 12, 15, 16, 24, 32] hot=13 sc=4 key0=(1728.0, -4.0013, 3)
    5: [9, 20, 28, 30, 35, 37] hot=13 sc=2 key0=(1728.0, -2.0013, 3)
    6: [10, 17, 24, 29, 30, 34] hot=13 sc=4 key0=(1728.0, -4.0013, 2)
    7: [5, 7, 18, 19, 20, 37] hot=15 sc=4 key0=(1728.0, -4.0015, 2)
    8: [9, 12, 14, 15, 29, 32] hot=14 sc=4 key0=(1728.0, -4.0014, 3)
    9: [14, 16, 17, 26, 34, 39] hot=13 sc=4 key0=(1728.0, -4.0013, 3)
   10: [5, 11, 12, 13, 22, 28] hot=21 sc=4 key0=(1740.0, -4.0021, 3)
   Final usage of jackpot numbers:
      3: 6
      10: 10
      23: 4
      32: 9
      33: 6
      39: 9
   Total tickets picked: 50

================================================================================
📅 23‑May‑2026  |  Winning: [11, 19, 20, 28, 31, 40]
================================================================================
Pools: EH=13 H=7 W=22 C=3
🔧 Killing missing decade '0s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:1  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [11, 19, 20, 28, 31, 40]
   EH=2, H=2, W=2, C=0 | Legacy=0
   PairScore=4, Hotness=18
   20‑week bands: {'5x+': 1, '3x': 1, '2x': 2, '1x': 2}
   Initial key (empty freq): (1964.0, -4.0018, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [14, 18, 29, 32, 44, 45] hot=13 sc=4 key0=(1936.0, -4.0013, 2)
    2: [11, 13, 26, 38, 40, 41] hot=13 sc=4 key0=(1936.0, -4.0013, 2)
    3: [13, 18, 21, 37, 38, 40] hot=13 sc=4 key0=(1936.0, -4.0013, 2)
    4: [11, 14, 17, 31, 32, 45] hot=13 sc=4 key0=(1936.0, -4.0013, 3)
    5: [13, 18, 27, 32, 33, 45] hot=13 sc=4 key0=(1936.0, -4.0013, 2)
    6: [11, 14, 30, 36, 38, 40] hot=12 sc=2 key0=(1936.0, -2.0012, 3)
    7: [18, 19, 22, 32, 40, 45] hot=12 sc=4 key0=(1936.0, -4.0012, 2)
    8: [10, 11, 13, 35, 38, 40] hot=12 sc=4 key0=(1936.0, -4.0012, 3)
    9: [12, 14, 18, 25, 32, 38] hot=11 sc=2 key0=(1936.0, -2.0011, 3)
   10: [11, 13, 14, 21, 37, 45] hot=13 sc=4 key0=(1936.0, -4.0013, 3)
   Final usage of jackpot numbers:
      11: 14
      19: 7
      20: 7
      28: 7
      31: 7
      40: 13
   Total tickets picked: 50

================================================================================
📅 30‑May‑2026  |  Winning: [8, 10, 12, 19, 28, 36]
================================================================================
Pools: EH=12 H=9 W=21 C=3
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:2  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [8, 10, 12, 19, 28, 36]
   EH=1, H=2, W=2, C=1 | Legacy=2
   PairScore=2, Hotness=14
   20‑week bands: {'5x+': 1, '1x': 1, '3x': 2, '2x': 2}
   Initial key (empty freq): (1787.0, -2.0014, 3)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [7, 8, 11, 19, 27, 29] hot=13 sc=4 key0=(1779.0, -4.0013, 2)
    2: [6, 10, 20, 28, 37, 38] hot=13 sc=4 key0=(1779.0, -4.0013, 2)
    3: [3, 5, 6, 14, 21, 35] hot=13 sc=4 key0=(1779.0, -4.0013, 3)
    4: [8, 10, 11, 23, 29, 33] hot=13 sc=4 key0=(1779.0, -4.0013, 2)
    5: [5, 6, 13, 17, 20, 37] hot=12 sc=4 key0=(1779.0, -4.0012, 2)
    6: [3, 8, 9, 19, 21, 35] hot=12 sc=4 key0=(1779.0, -4.0012, 3)
    7: [3, 6, 18, 21, 22, 36] hot=9 sc=4 key0=(1781.0, -4.0009, 2)
    8: [1, 5, 8, 15, 36, 37] hot=9 sc=4 key0=(1781.0, -4.0009, 3)
    9: [2, 8, 10, 14, 28, 29] hot=11 sc=4 key0=(1781.0, -4.0011, 2)
   10: [6, 7, 11, 13, 20, 33] hot=12 sc=4 key0=(1781.0, -4.0012, 2)
   Final usage of jackpot numbers:
      8: 13
      10: 9
      12: 6
      19: 9
      28: 8
      36: 8
   Total tickets picked: 50

================================================================================
📅 06‑Jun‑2026  |  Winning: [10, 25, 30, 31, 43, 44]
================================================================================
Pools: EH=12 H=10 W=20 C=3
🔧 Killing missing decade '0s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:4  4:2  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [10, 25, 30, 31, 43, 44]
   EH=2, H=1, W=2, C=1 | Legacy=1
   PairScore=4, Hotness=15
   20‑week bands: {'2x': 4, '1x': 1, '3x': 1}
   Initial key (empty freq): (1938.0, -4.0015, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [14, 15, 23, 28, 34, 37] hot=13 sc=4 key0=(1906.0, -4.0013, 2)
    2: [18, 20, 26, 27, 30, 40] hot=13 sc=4 key0=(1906.0, -4.0013, 3)
    3: [15, 25, 29, 30, 34, 42] hot=13 sc=4 key0=(1906.0, -4.0013, 2)
    4: [16, 23, 26, 27, 31, 35] hot=14 sc=4 key0=(1908.0, -4.0014, 3)
    5: [15, 22, 23, 27, 36, 43] hot=14 sc=4 key0=(1908.0, -4.0014, 3)
    6: [20, 21, 26, 30, 34, 45] hot=11 sc=4 key0=(1906.0, -4.0011, 3)
    7: [17, 25, 26, 27, 30, 38] hot=11 sc=4 key0=(1906.0, -4.0011, 3)
    8: [15, 23, 34, 35, 37, 42] hot=13 sc=4 key0=(1906.0, -4.0013, 3)
    9: [14, 15, 27, 28, 31, 34] hot=15 sc=4 key0=(1908.0, -4.0015, 2)
   10: [18, 23, 26, 29, 30, 36] hot=15 sc=4 key0=(1908.0, -4.0015, 3)
   Final usage of jackpot numbers:
      10: 6
      25: 9
      30: 14
      31: 9
      43: 8
      44: 6
   Total tickets picked: 50

================================================================================
📅 13‑Jun‑2026  |  Winning: [12, 16, 30, 31, 40, 43]
================================================================================
Pools: EH=12 H=8 W=21 C=4
🔧 Killing missing decade '0s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:1  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [12, 16, 30, 31, 40, 43]
   EH=2, H=2, W=2, C=0 | Legacy=3
   PairScore=4, Hotness=21
   20‑week bands: {'4x': 1, '2x': 1, '3x': 4}
   Initial key (empty freq): (1994.0, -4.0021, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [11, 19, 26, 27, 33, 37] hot=13 sc=4 key0=(1968.0, -4.0013, 2)
    2: [12, 20, 35, 36, 40, 42] hot=13 sc=4 key0=(1968.0, -4.0013, 2)
    3: [16, 28, 33, 34, 42, 44] hot=13 sc=4 key0=(1968.0, -4.0013, 2)
    4: [11, 17, 27, 39, 40, 44] hot=13 sc=4 key0=(1968.0, -4.0013, 2)
    5: [10, 12, 26, 27, 30, 35] hot=13 sc=4 key0=(1968.0, -4.0013, 2)
    6: [12, 24, 28, 32, 33, 42] hot=11 sc=4 key0=(1968.0, -4.0011, 2)
    7: [11, 23, 27, 35, 44, 45] hot=11 sc=4 key0=(1968.0, -4.0011, 2)
    8: [20, 26, 28, 37, 41, 42] hot=11 sc=2 key0=(1970.0, -2.0011, 3)
    9: [16, 26, 36, 39, 40, 42] hot=12 sc=4 key0=(1970.0, -4.0012, 2)
   10: [12, 17, 27, 30, 33, 34] hot=12 sc=4 key0=(1970.0, -4.0012, 3)
   Final usage of jackpot numbers:
      12: 11
      16: 9
      30: 9
      31: 5
      40: 10
      43: 5
   Total tickets picked: 50

================================================================================
📅 20‑Jun‑2026  |  Winning: [3, 6, 9, 14, 21, 22]
================================================================================
Pools: EH=12 H=8 W=24 C=1
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:2  4:0  5:0  6:1

🔎 JACKPOT TICKET PROFILE
   Numbers: [3, 6, 9, 14, 21, 22]
   EH=0, H=2, W=4, C=0 | Legacy=0
   PairScore=2, Hotness=11
   20‑week bands: {'5x+': 1, '1x': 2, '3x': 1, '4x': 1, '2x': 1}
   Initial key (empty freq): (1426.0, -2.0011, 3)
   ✅ JACKPOT WAS SELECTED!
   Picked at slot 30
   Final usage of jackpot numbers:
      3: 6
      6: 6
      9: 6
      14: 6
      21: 13
      22: 12
   Total tickets picked: 50

================================================================================
📅 27‑Jun‑2026  |  Winning: [15, 17, 24, 28, 36, 37]
================================================================================
Pools: EH=11 H=10 W=23 C=1
🔧 Killing missing decade '40s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:2  4:0  5:1  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [15, 17, 24, 28, 36, 37]
   EH=1, H=2, W=3, C=0 | Legacy=0
   PairScore=4, Hotness=16
   20‑week bands: {'1x': 2, '3x': 2, '2x': 2}
   Initial key (empty freq): (1474.0, -4.0016, 2)
   ❌ JACKPOT NOT SELECTED
   ⛔ BLOCKED by collision shield
      Overlaps with: [15, 17, 26, 28, 36, 37]
   Final usage of jackpot numbers:
      15: 7
      17: 7
      24: 10
      28: 7
      36: 10
      37: 7
   Total tickets picked: 50

================================================================================
📅 04‑Jul‑2026  |  Winning: [4, 8, 15, 32, 43, 44]
================================================================================
Pools: EH=14 H=7 W=19 C=5
🔧 Killing missing decade '20s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:2  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [4, 8, 15, 32, 43, 44]
   EH=1, H=1, W=4, C=0 | Legacy=1
   PairScore=4, Hotness=13
   20‑week bands: {'2x': 3, '3x': 2, '4x': 1}
   Initial key (empty freq): (2373.0, -4.0013, 2)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [5, 7, 13, 37, 41, 42] hot=15 sc=4 key0=(2337.0, -4.0015, 2)
    2: [8, 30, 34, 35, 41, 44] hot=15 sc=4 key0=(2337.0, -4.0015, 3)
    3: [13, 16, 32, 37, 43, 44] hot=13 sc=4 key0=(2337.0, -4.0013, 2)
    4: [3, 4, 13, 30, 33, 37] hot=12 sc=4 key0=(2337.0, -4.0012, 3)
    5: [5, 6, 11, 30, 41, 44] hot=12 sc=4 key0=(2337.0, -4.0012, 2)
    6: [7, 9, 13, 37, 38, 44] hot=13 sc=4 key0=(2337.0, -4.0013, 2)
    7: [8, 13, 30, 35, 40, 41] hot=13 sc=4 key0=(2337.0, -4.0013, 2)
    8: [16, 30, 34, 37, 44, 45] hot=12 sc=4 key0=(2337.0, -4.0012, 3)
    9: [3, 13, 32, 37, 41, 42] hot=14 sc=4 key0=(2337.0, -4.0014, 2)
   10: [4, 6, 30, 38, 41, 44] hot=12 sc=2 key0=(2337.0, -2.0012, 2)
   Final usage of jackpot numbers:
      4: 9
      8: 9
      15: 5
      32: 10
      43: 8
      44: 16
   Total tickets picked: 50

================================================================================
📅 11‑Jul‑2026  |  Winning: [13, 14, 16, 21, 29, 41]
================================================================================
Pools: EH=8 H=16 W=20 C=1
🔧 Killing missing decade '0s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:1  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [13, 14, 16, 21, 29, 41]
   EH=1, H=2, W=3, C=0 | Legacy=0
   PairScore=4, Hotness=14
   20‑week bands: {'2x': 2, '4x': 1, '3x': 2, '1x': 1}
   Initial key (empty freq): (2123.0, -4.0014, 3)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [14, 17, 25, 26, 40, 45] hot=16 sc=4 key0=(2083.0, -4.0016, 2)
    2: [10, 12, 32, 35, 44, 45] hot=14 sc=4 key0=(2083.0, -4.0014, 2)
    3: [18, 27, 28, 36, 37, 45] hot=12 sc=4 key0=(2083.0, -4.0012, 2)
    4: [19, 20, 29, 33, 42, 45] hot=11 sc=4 key0=(2085.0, -4.0011, 2)
    5: [21, 28, 30, 31, 43, 45] hot=11 sc=4 key0=(2085.0, -4.0011, 2)
    6: [14, 16, 26, 39, 44, 45] hot=14 sc=4 key0=(2083.0, -4.0014, 2)
    7: [10, 27, 34, 35, 42, 45] hot=15 sc=4 key0=(2085.0, -4.0015, 2)
    8: [11, 12, 25, 33, 43, 45] hot=15 sc=4 key0=(2085.0, -4.0015, 2)
    9: [15, 22, 23, 32, 36, 45] hot=13 sc=4 key0=(2089.0, -4.0013, 2)
   10: [17, 24, 29, 37, 38, 45] hot=10 sc=4 key0=(2087.0, -4.001, 2)
   Final usage of jackpot numbers:
      13: 7
      14: 9
      16: 8
      21: 8
      29: 8
      41: 7
   Total tickets picked: 50

================================================================================
📅 18‑Jul‑2026  |  Winning: [5, 7, 13, 24, 30, 41]
================================================================================
Pools: EH=13 H=10 W=17 C=5
❌ All five decades are present – no single‑decade kill possible. Skipping.

================================================================================
📅 25‑Jul‑2026  |  Winning: [2, 6, 8, 12, 22, 43]
================================================================================
Pools: EH=11 H=9 W=21 C=4
🔧 Killing missing decade '30s' (absent from real draw).
Generated 10000 weighted valid tickets.
Tickets selected: 50
Hits → 3:3  4:0  5:0  6:0

🔎 JACKPOT TICKET PROFILE
   Numbers: [2, 6, 8, 12, 22, 43]
   EH=1, H=1, W=4, C=0 | Legacy=0
   PairScore=2, Hotness=12
   20‑week bands: {'3x': 1, '1x': 2, '4x': 3}
   Initial key (empty freq): (2030.0, -2.0012, 3)
   ❌ JACKPOT NOT SELECTED
   ⚠️  OUTSCORED – available but not selected
   First 10 selected tickets (key at selection):
    1: [4, 5, 16, 18, 20, 40] hot=13 sc=4 key0=(2004.0, -4.0013, 2)
    2: [8, 11, 15, 25, 26, 42] hot=13 sc=4 key0=(2004.0, -4.0013, 2)
    3: [5, 9, 10, 16, 20, 29] hot=13 sc=4 key0=(2004.0, -4.0013, 2)
    4: [8, 11, 12, 14, 25, 28] hot=13 sc=4 key0=(2006.0, -4.0013, 3)
    5: [5, 8, 22, 25, 43, 44] hot=14 sc=4 key0=(2006.0, -4.0014, 2)
    6: [11, 16, 19, 20, 23, 29] hot=11 sc=4 key0=(2004.0, -4.0011, 3)
    7: [2, 8, 11, 16, 17, 42] hot=11 sc=4 key0=(2004.0, -4.0011, 3)
    8: [5, 10, 20, 25, 40, 45] hot=12 sc=2 key0=(2004.0, -2.0012, 2)
    9: [4, 5, 14, 16, 20, 26] hot=15 sc=4 key0=(2006.0, -4.0015, 2)
   10: [8, 9, 11, 12, 15, 25] hot=14 sc=4 key0=(2006.0, -4.0014, 3)
   Final usage of jackpot numbers:
      2: 7
      6: 6
      8: 14
      12: 7
      22: 8
      43: 8
   Total tickets picked: 50

================================================================================
📊 OVERALL SUMMARY (20 draws, kill‑cheat, NO injection)
================================================================================
Draws tested: 20
Structurally impossible (all decades present): 2
Jackpot ticket in pool: 18/20
Jackpot ticket NATURALLY SELECTED: 2/20
≥3 matches: 17
≥4 matches: 2
≥5 matches: 1
6 matches (jackpot): 2
