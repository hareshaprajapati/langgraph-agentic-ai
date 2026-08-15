import csv
from datetime import datetime as dt
import itertools
from collections import Counter, defaultdict

TARGET_DATE = "2026-05-16"
# ---------- POOLS  ----------
EH = [13, 11, 3, 33]
# 10
H  = [9, 32, 5, 16]
# 7
W  = [10, 14, 7, 12, 18, 19]
# 25
C = [6, 23]
# 3
LEGACY = [2, 11, 12, 17, 33, 37]
REAL = {3, 10, 23, 32, 33, 39}
# REAL = set()
WIN = tuple(sorted(REAL)) if REAL else ()
# ---------- DECADE KILLS & TOTAL ----------
TOTAL = 50
kill_list = ["40s"]
# kill_list = ["none"]
# kill_list = ['30s'] * 50
# kill_list = ['40s+30s'] * 25 + ['40s+10s'] * 25
# ---------- TARGET DATE & 20‑WEEK HISTORY ----------

# ---------- SAFE RANGES ----------
SAFE_DEPTH_RANGES = [
    (0, 2),   # EH
    (0, 3),   # H
    (2, 3),   # W
    (0, 1)    # C
]

SAFE_BREADTH_RANGES = [
    (0, 1),   # EH
    (0, 2),   # H
    (4, 5),   # W
    (0, 0)    # C
]

# ---------- PROFILE SELECTION ----------
USE_DEPTH   = True
USE_BREADTH = False


# Band caps (stricter, 95.5% safe)
BAND_CAPS = {
    '0x': 3,
    '1x': 3,
    '2x': 4,
    '3x': 4,
    '4x': 3,
    '5x+': 3
}

def band_label(count):
    if count >= 5: return '5x+'
    if count == 4: return '4x'
    if count == 3: return '3x'
    if count == 2: return '2x'
    if count == 1: return '1x'
    return '0x'

# Parse all Saturday main draws from the CSV
all_sat_draws = []   # list of (date_str, main_numbers)
with open("cross_lotto_data_backup.csv", newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    others_col = None
    for col in reader.fieldnames:
        if 'Others' in col:
            others_col = col
            break
    if not others_col:
        raise KeyError("Could not find 'Others' column")
    for row in reader:
        if not row['Date'].strip().startswith('Sat'):
            continue
        date_str = row['Date'].strip()
        try:
            d = dt.strptime(date_str, '%a %d-%b-%Y')
        except:
            continue
        if d >= dt.strptime(TARGET_DATE, '%Y-%m-%d'):
            continue
        main_part = row[others_col].split('],')[0].strip()
        if main_part.startswith('['):
            main_part = main_part[1:]
        main_part = main_part.replace(']', '').strip()
        if main_part:
            nums = [int(x.strip()) for x in main_part.split(',') if x.strip().isdigit()]
            if len(nums) == 6:
                all_sat_draws.append((d, nums))

# Sort by date and take the LAST 20 draws
all_sat_draws.sort(key=lambda x: x[0])
last_20_draws = all_sat_draws[-20:] if len(all_sat_draws) >= 20 else all_sat_draws

# 20‑week frequency counter
freq_20w = Counter()
for _, nums in last_20_draws:
    for n in nums:
        freq_20w[n] += 1

# Pre‑compute band for every number 1‑45
band_for_num = {n: band_label(freq_20w.get(n, 0)) for n in range(1, 46)}

print(f"20‑week window: {len(last_20_draws)} draws available")
print(f"Deep‑cold numbers (0x): {sorted(n for n, b in band_for_num.items() if b == '0x')}")


# Auto‑adjust ticket counts
if USE_DEPTH and USE_BREADTH:
    TOTAL_DEPTH, TOTAL_BREADTH = 30, 20
elif USE_DEPTH:
    TOTAL_DEPTH, TOTAL_BREADTH = 50, 0
elif USE_BREADTH:
    TOTAL_DEPTH, TOTAL_BREADTH = 0, 50
else:
    raise ValueError("At least one profile must be active.")

def parse_kill(kill_str):
    """ '40s' -> {'40s'}   '40s+30s' -> {'40s','30s'} """
    if kill_str == 'none':
        return set()
    return set(kill_str.split('+'))

# Estimated average picks per pool using the safe ranges
avg_EH = sum(range(SAFE_DEPTH_RANGES[0][0], SAFE_DEPTH_RANGES[0][1] + 1)) / (
    SAFE_DEPTH_RANGES[0][1] - SAFE_DEPTH_RANGES[0][0] + 1
)
avg_H = sum(range(SAFE_DEPTH_RANGES[1][0], SAFE_DEPTH_RANGES[1][1] + 1)) / (
    SAFE_DEPTH_RANGES[1][1] - SAFE_DEPTH_RANGES[1][0] + 1
)
avg_W = sum(range(SAFE_DEPTH_RANGES[2][0], SAFE_DEPTH_RANGES[2][1] + 1)) / (
    SAFE_DEPTH_RANGES[2][1] - SAFE_DEPTH_RANGES[2][0] + 1
)
avg_C = sum(range(SAFE_DEPTH_RANGES[3][0], SAFE_DEPTH_RANGES[3][1] + 1)) / (
    SAFE_DEPTH_RANGES[3][1] - SAFE_DEPTH_RANGES[3][0] + 1
)

total_picks_EH = int(TOTAL * avg_EH)
total_picks_H  = int(TOTAL * avg_H)
total_picks_W  = int(TOTAL * avg_W)
total_picks_C  = int(TOTAL * avg_C)

def ideal_band(total_picks, pool_size):
    if pool_size == 0:
        return (0, 0)
    avg = total_picks / pool_size
    lo = max(0, int(avg) - 2)     # was max(1, int(avg)-1)
    hi = int(avg) + 2             # was int(avg)+1
    return (lo, hi)

EH_IDEAL = ideal_band(total_picks_EH, len(EH))
H_IDEAL  = ideal_band(total_picks_H,  len(H))
W_IDEAL  = ideal_band(total_picks_W,  len(W))
C_IDEAL  = ideal_band(total_picks_C,  len(C))

print(f"Ideal bands → EH:{EH_IDEAL}  H:{H_IDEAL}  W:{W_IDEAL}  C:{C_IDEAL}")

# ---------- HELPERS ----------
def dec(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

def valid(t, kill):
    killed_set = parse_kill(kill)
    if any(dec(x) in killed_set for x in t):
        return False
    if sum(1 for x in t if x in LEGACY) > 1: return False
    o = sum(1 for x in t if x % 2)
    if (o, 6-o) not in [(3,3),(2,4),(4,2)]: return False
    lo = sum(1 for x in t if x <= 22)
    if (lo, 6-lo) not in [(3,3),(2,4),(4,2)]: return False
    if max(Counter(dec(x) for x in t).values()) > 3: return False
    # 40s cap – no more than 2 numbers from the 40s
    if sum(1 for x in t if 40 <= x <= 45) > 2:
        return False
    return True

def consecutive(t):
    s = sorted(t)
    return any(s[i+1]-s[i]==1 for i in range(5))
def mirror(t):
    return len({x%10 for x in t}) < 6
def score(t):
    return (2 if consecutive(t) else 0) + (2 if mirror(t) else 0)
def overlap(a,b):
    return len(set(a)&set(b))
def matches(t, res):
    return len(set(t)&res)

def in_ideal(n, freq):
    if n in EH: return EH_IDEAL[0] <= freq[n] <= EH_IDEAL[1]
    if n in H:  return H_IDEAL[0] <= freq[n] <= H_IDEAL[1]
    if n in W:  return W_IDEAL[0] <= freq[n] <= W_IDEAL[1]
    if n in C:  return C_IDEAL[0] <= freq[n] <= C_IDEAL[1]
    return True

def distance_from_ideal(freq):
    total = 0
    for pool, ideal in [(EH, EH_IDEAL), (H, H_IDEAL), (W, W_IDEAL), (C, C_IDEAL)]:
        target = (ideal[0] + ideal[1]) / 2
        for n in pool:
            total += (freq[n] - target) ** 2
    return total

# ---------- PRE‑FILTER POOLS FOR THE KILLED DECADES ----------
# All tickets use the same kill in this test; we take the first kill string.
kill_str = kill_list[0]
killed_set = parse_kill(kill_str)
print(f"Killed decades: {killed_set}")

EH_use = [n for n in EH if dec(n) not in killed_set]
H_use  = [n for n in H  if dec(n) not in killed_set]
W_use  = [n for n in W  if dec(n) not in killed_set]
C_use  = [n for n in C  if dec(n) not in killed_set]

# Recalculate ideal bands based on the reduced pools
EH_IDEAL = ideal_band(total_picks_EH, len(EH_use))
H_IDEAL  = ideal_band(total_picks_H,  len(H_use))
W_IDEAL  = ideal_band(total_picks_W,  len(W_use))
C_IDEAL  = ideal_band(total_picks_C,  len(C_use))
print(f"Reduced pools → EH:{len(EH_use)}  H:{len(H_use)}  W:{len(W_use)}  C:{len(C_use)}")
print(f"Updated ideal bands → EH:{EH_IDEAL}  H:{H_IDEAL}  W:{W_IDEAL}  C:{C_IDEAL}")

# ---------- GENERATE DEPTH CANDIDATES ----------
combos_depth = []
if TOTAL_DEPTH > 0:
    print("Generating Depth candidates...")
    for eh_c in range(SAFE_DEPTH_RANGES[0][0], SAFE_DEPTH_RANGES[0][1] + 1):
        if eh_c > len(EH_use): continue
        for h_c in range(SAFE_DEPTH_RANGES[1][0], SAFE_DEPTH_RANGES[1][1] + 1):
            if h_c > len(H_use): continue
            for w_c in range(SAFE_DEPTH_RANGES[2][0], SAFE_DEPTH_RANGES[2][1] + 1):
                if w_c > len(W_use): continue
                for c_c in range(SAFE_DEPTH_RANGES[3][0], SAFE_DEPTH_RANGES[3][1] + 1):
                    if c_c > len(C_use): continue
                    if eh_c + h_c + w_c + c_c != 6:
                        continue
                    for eh in itertools.combinations(EH_use, eh_c):
                        for h in itertools.combinations(H_use, h_c):
                            for w in itertools.combinations(W_use, w_c):
                                for c in itertools.combinations(C_use, c_c):
                                    t = tuple(sorted(eh + h + w + c))
                                    if len(set(t)) < 6: continue
                                    # constraints (same as before, no kill check needed)
                                    if sum(1 for x in t if x in LEGACY) > 1: continue
                                    o = sum(1 for x in t if x % 2)
                                    if (o,6-o) not in [(3,3),(2,4),(4,2)]: continue
                                    lo = sum(1 for x in t if x <= 22)
                                    if (lo,6-lo) not in [(3,3),(2,4),(4,2)]: continue
                                    if max(Counter(dec(x) for x in t).values()) > 3: continue
                                    if sum(1 for x in t if 40 <= x <= 45) > 2: continue
                                    if not (consecutive(t) or mirror(t)): continue
                                    band_cnt = Counter(band_for_num[x] for x in t)
                                    if any(band_cnt[b] > BAND_CAPS[b] for b in band_cnt): continue
                                    combos_depth.append(t)
else: combos_depth = []
# Recalculate ideal bands for Depth pool
depth_total_picks_EH = int(30 * avg_EH)   # 30 Depth tickets
depth_total_picks_H  = int(30 * avg_H)
depth_total_picks_W  = int(30 * avg_W)
depth_total_picks_C  = int(30 * avg_C)
EH_IDEAL_D = ideal_band(depth_total_picks_EH, len(EH_use))
H_IDEAL_D  = ideal_band(depth_total_picks_H,  len(H_use))
W_IDEAL_D  = ideal_band(depth_total_picks_W,  len(W_use))
C_IDEAL_D  = ideal_band(depth_total_picks_C,  len(C_use))
print(f"Depth ideal bands → EH:{EH_IDEAL_D}  H:{H_IDEAL_D}  W:{W_IDEAL_D}  C:{C_IDEAL_D}")

# ---------- GENERATE BREADTH CANDIDATES ----------
combos_breadth = []
if TOTAL_BREADTH > 0:
    print("Generating Breadth candidates...")
    for eh_c in range(SAFE_BREADTH_RANGES[0][0], SAFE_BREADTH_RANGES[0][1] + 1):
        if eh_c > len(EH_use): continue
        for h_c in range(SAFE_BREADTH_RANGES[1][0], SAFE_BREADTH_RANGES[1][1] + 1):
            if h_c > len(H_use): continue
            # enforce EH+H between 1 and 2
            if eh_c + h_c < 1 or eh_c + h_c > 2: continue
            for w_c in range(SAFE_BREADTH_RANGES[2][0], SAFE_BREADTH_RANGES[2][1] + 1):
                if w_c > len(W_use): continue
                for c_c in range(SAFE_BREADTH_RANGES[3][0], SAFE_BREADTH_RANGES[3][1] + 1):
                    if c_c > len(C_use): continue
                    if eh_c + h_c + w_c + c_c != 6: continue
                    for eh in itertools.combinations(EH_use, eh_c):
                        for h in itertools.combinations(H_use, h_c):
                            for w in itertools.combinations(W_use, w_c):
                                for c in itertools.combinations(C_use, c_c):
                                    t = tuple(sorted(eh + h + w + c))
                                    if len(set(t)) < 6: continue
                                    if sum(1 for x in t if x in LEGACY) > 1: continue
                                    o = sum(1 for x in t if x % 2)
                                    if (o,6-o) not in [(3,3),(2,4),(4,2)]: continue
                                    lo = sum(1 for x in t if x <= 22)
                                    if (lo,6-lo) not in [(3,3),(2,4),(4,2)]: continue
                                    if max(Counter(dec(x) for x in t).values()) > 3: continue
                                    if sum(1 for x in t if 40 <= x <= 45) > 2: continue
                                    if not (consecutive(t) or mirror(t)): continue
                                    band_cnt = Counter(band_for_num[x] for x in t)
                                    if any(band_cnt[b] > BAND_CAPS[b] for b in band_cnt): continue
                                    combos_breadth.append(t)
else: combos_breadth = []

breadth_total_picks_EH = int(20 * 0.5)   # avg EH picks per Breadth ticket = 0.5
breadth_total_picks_H  = int(20 * 1.0)   # avg H picks = 1.0
breadth_total_picks_W  = int(20 * 4.5)   # avg W picks = 4.5
breadth_total_picks_C  = 0
EH_IDEAL_B = ideal_band(breadth_total_picks_EH, len(EH_use))
H_IDEAL_B  = ideal_band(breadth_total_picks_H,  len(H_use))
W_IDEAL_B  = ideal_band(breadth_total_picks_W,  len(W_use))
C_IDEAL_B  = (0, 0)
print(f"Breadth ideal bands → EH:{EH_IDEAL_B}  H:{H_IDEAL_B}  W:{W_IDEAL_B}  C:{C_IDEAL_B}")

# ---------- FAIRNESS SELECTION (Depth first, then Breadth) ----------
selected = []
sel_set = set()
freq = Counter()

def ticket_band_bonus(t):
    bonus = 0
    for n in t:
        band = band_for_num.get(n, '0x')
        if band == '0x':
            bonus += 1.5
        elif band == '1x':
            bonus += 2.0
        elif band == '2x':
            bonus += 1.0
        elif band == '3x':
            bonus += 0.5
    return bonus

def pick_as_many_as_possible(combos, max_needed, EH_IDEAL_CUR, H_IDEAL_CUR, W_IDEAL_CUR, C_IDEAL_CUR):
    def static_score(t):
        # historical 20-week frequency + band bonus
        freq_sum = sum(freq_20w.get(n, 0) for n in t)
        band_sum = ticket_band_bonus(t)
        return freq_sum + band_sum

    # Sort all candidates once, best historical score first
    ordered = sorted(combos, key=static_score, reverse=True)

    picked = []
    freq = Counter()

    for t in ordered:
        if len(picked) >= max_needed:
            break

        # collision shield relaxed to 4
        if any(overlap(t, s) > 4 for s in selected + picked):
            continue

        # use a usage cap high enough to allow 50 tickets
        if any(freq[n] >= 30 for n in t):
            continue

        picked.append(t)
        for n in t:
            freq[n] += 1

    return picked

# ---------- DIAGNOSTIC: TOP 10 CANDIDATES BY BAND_SCORE ----------
if REAL:
    print("\nTop 10 candidates by band_score:")
    for t in sorted(combos_depth, key=ticket_band_bonus, reverse=True)[:10]:
        print(f"  {sorted(t)}  band_score={ticket_band_bonus(t):.1f}  hits={matches(t, REAL)}")

# Select Depth tickets
if TOTAL_DEPTH > 0:
    depth_tickets = pick_as_many_as_possible(combos_depth, 50, EH_IDEAL_D, H_IDEAL_D, W_IDEAL_D, C_IDEAL_D)
    selected.extend(depth_tickets)
    TOTAL = len(selected)

# Select Breadth tickets (collision shield against Depth already included)
if TOTAL_BREADTH > 0:
    breadth_tickets = pick_as_many_as_possible(combos_breadth, 20, EH_IDEAL_B, H_IDEAL_B, W_IDEAL_B, C_IDEAL_B)
    selected.extend(breadth_tickets)
    TOTAL = len(selected)

print(f"Generated {TOTAL} Depth tickets (maximum possible from this pool).")

# ---------- Recompute global frequency counter for audit ----------
freq = Counter()
for t in selected:
    for n in t:
        freq[n] += 1

# Combine candidate pools for post‑balance
combos_combined = combos_depth + combos_breadth

if REAL:
    print("\n5-match candidates found in combos_depth:")
    found = False
    for t in combos_depth:
        m = matches(t, REAL)
        if m >= 5:
            print(f"  {sorted(t)} -> {m} hits")
            found = True
    if not found:
        print("  None")

# ---------- POST‑BALANCE (blended ideal bands) ----------
print("Balancing towards ideal frequencies...")

# Compute blended ideal bands: total picks from Depth (30 tix) + Breadth (20 tix)
blended_EH_picks = depth_total_picks_EH + breadth_total_picks_EH
blended_H_picks  = depth_total_picks_H  + breadth_total_picks_H
blended_W_picks  = depth_total_picks_W  + breadth_total_picks_W
blended_C_picks  = depth_total_picks_C  + breadth_total_picks_C

EH_IDEAL_BL = ideal_band(blended_EH_picks, len(EH_use))
H_IDEAL_BL  = ideal_band(blended_H_picks,  len(H_use))
W_IDEAL_BL  = ideal_band(blended_W_picks,  len(W_use))
C_IDEAL_BL  = ideal_band(blended_C_picks,  len(C_use))

for _ in range(0):
    out_of_ideal = []
    for n in EH_use:
        if freq[n] < EH_IDEAL_BL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > EH_IDEAL_BL[1]: out_of_ideal.append((n, 'high'))
    for n in H_use:
        if freq[n] < H_IDEAL_BL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > H_IDEAL_BL[1]: out_of_ideal.append((n, 'high'))
    for n in W_use:
        if freq[n] < W_IDEAL_BL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > W_IDEAL_BL[1]: out_of_ideal.append((n, 'high'))
    for n in C_use:
        if freq[n] < C_IDEAL_BL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > C_IDEAL_BL[1]: out_of_ideal.append((n, 'high'))
    if not out_of_ideal: break

    num, direction = min(out_of_ideal, key=lambda x: freq[x[0]] if x[1]=='low' else -freq[x[0]])
    improved = False
    for idx, t in enumerate(selected):
        if direction == 'low' and num in t: continue
        if direction == 'high' and num not in t: continue
        candidates = []
        if direction == 'low':
            candidates = [x for x in combos_combined if x not in sel_set and num in x]
        else:
            candidates = [x for x in combos_combined if x not in sel_set and num not in x]
        if not candidates: continue
        for cand in candidates:
            if any(overlap(cand, selected[j]) > 3 for j in range(TOTAL) if j != idx): continue
            temp_freq = freq.copy()
            for n in t: temp_freq[n] -= 1
            for n in cand: temp_freq[n] += 1
            if max(temp_freq.values()) > 18: continue
            # compute distance using blended ideal bands
            old_dist = 0
            new_dist = 0
            for pool, ideal in [(EH_use, EH_IDEAL_BL), (H_use, H_IDEAL_BL),
                                (W_use, W_IDEAL_BL), (C_use, C_IDEAL_BL)]:
                target = (ideal[0] + ideal[1]) / 2
                for n in pool:
                    old_dist += (freq[n] - target) ** 2
                    new_dist += (temp_freq[n] - target) ** 2
            if new_dist < old_dist:
                sel_set.remove(t)
                sel_set.add(cand)
                for n in t: freq[n] -= 1
                for n in cand: freq[n] += 1
                selected[idx] = cand
                improved = True
                break
        if improved: break
    if not improved: break

# ---------- AUDIT ----------
consec = sum(consecutive(t) for t in selected)
mirr = sum(mirror(t) for t in selected)
max_freq = max(freq.values())
max_ov = max(overlap(selected[i], selected[j]) for i in range(TOTAL) for j in range(i+1, TOTAL)) if TOTAL>1 else 0

print("\n--- AUDIT ---")
print(f"Consecutive pairs: {consec}/{TOTAL} (need >=40)")
print(f"Mirror pairs: {mirr}/{TOTAL} (need >=35)")
print(f"Max frequency: {max_freq} (need <=18)")
print(f"Max overlap: {max_ov} (need <=3)")

low_high_counts = Counter()
for t in selected:
    lo = sum(1 for n in t if n <= 22)
    high = 6 - lo
    low_high_counts[(lo, high)] += 1
five_one = low_high_counts.get((5,1),0) + low_high_counts.get((1,5),0)
print(f"5:1 or 1:5 tickets: {five_one} (allow <=3)")

# ---------- HIT ANALYSIS (only if REAL is populated) ----------
if REAL:
    print("\n--- HIT ANALYSIS ---")
    hits = [(t, matches(t, REAL)) for t in selected if matches(t, REAL) >= 3]
    print(f"≥3 hits: {len(hits)}")
    for t, m in sorted(hits, key=lambda x: -x[1])[:10]:
        print(f"  {sorted(t)} -> {m}")
    print(f"\n🎯 Jackpot ticket in set: {WIN in sel_set}")
else:
    print("\n(REAL is empty – add the winning numbers after the draw and re‑run)")

print(f"\nNumber frequencies (target EH:{EH_IDEAL}, H:{H_IDEAL}, W:{W_IDEAL}, C:{C_IDEAL}):")
for n in sorted(freq):
    if freq[n] > 0:
        pool = 'EH' if n in EH else ('H' if n in H else ('W' if n in W else 'C'))
        print(f"  {pool} {n:2d}: {freq[n]}")

print("\nFinal Tickets:")
for i, t in enumerate(selected, 1):
    print(f"{i:2d}: {sorted(t)}")