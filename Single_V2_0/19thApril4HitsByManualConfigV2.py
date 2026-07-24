import itertools
from collections import Counter, defaultdict

# ---------- POOLS (18-Jul-2026) ----------
EH = [3, 4, 6, 13, 16, 17, 22, 27, 30, 34, 35, 38, 39]
H  = [5, 10, 14, 15, 20, 21, 23, 25, 28, 29]
W  = [1, 7, 9, 11, 12, 18, 19, 26, 32, 33, 36, 37, 41, 42, 43, 44, 45]
C  = [2, 8, 24, 31, 40]
LEGACY = [13, 14, 16, 21, 29, 41]   # from 27-Jun result
REAL = set()
WIN = tuple(sorted(REAL)) if REAL else ()
# ---------- DECADE KILLS & TOTAL ----------
TOTAL = 50
kill_list = ['40s']*15 + ['0s']*12 + ['10s']*12 + ['30s']*11

# ---------- SAFE DEPTH RANGES & IDEAL BANDS ----------
SAFE_DEPTH_RANGES = [
    (0, 2),   # EH
    (0, 3),   # H
    (2, 3),   # W
    (0, 1)    # C
]

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
    lo = max(1, int(avg) - 1)
    hi = int(avg) + 1
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
    if any(dec(x) == kill for x in t): return False
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

# ---------- GENERATE CANDIDATES (safe‑range, pair‑filtered) ----------
print("Generating candidates...")
all_cand = {}
combos = []
for eh_c in range(SAFE_DEPTH_RANGES[0][0], SAFE_DEPTH_RANGES[0][1] + 1):
    for h_c in range(SAFE_DEPTH_RANGES[1][0], SAFE_DEPTH_RANGES[1][1] + 1):
        for w_c in range(SAFE_DEPTH_RANGES[2][0], SAFE_DEPTH_RANGES[2][1] + 1):
            for c_c in range(SAFE_DEPTH_RANGES[3][0], SAFE_DEPTH_RANGES[3][1] + 1):
                if eh_c + h_c + w_c + c_c != 6:
                    continue
                for eh in itertools.combinations(EH, eh_c):
                    for h in itertools.combinations(H, h_c):
                        for w in itertools.combinations(W, w_c):
                            for c in itertools.combinations(C, c_c):
                                t = tuple(sorted(eh + h + w + c))
                                if len(set(t)) < 6:
                                    continue
                                combos.append(t)

for kill in set(kill_list):
    valid_tix = [t for t in combos if valid(t, kill)]
    valid_tix = [t for t in valid_tix if consecutive(t) or mirror(t)]
    all_cand[(0, kill)] = valid_tix
    print(f"  Kill {kill}: {len(valid_tix)} candidates")

# All tickets belong to the same virtual profile (0)
req = defaultdict(int)
for k in kill_list:
    req[(0, k)] += 1

# ---------- FAIRNESS‑FIRST SELECTION ----------
selected = []
sel_set = set()
freq = Counter()
ticket_specs = []

kill_order = sorted(req.keys(), key=str)

for spec in kill_order:
    need = req[spec]
    pool = all_cand[spec]
    avail = [t for t in pool if t not in sel_set]
    for _ in range(need):
        best_t = None
        best_key = None
        for t in avail:
            if any(overlap(t, s) > 3 for s in selected):
                continue
            if any(freq[n] >= 12 for n in t):
                continue
            temp_freq = freq.copy()
            for n in t: temp_freq[n] += 1
            all_in = all(in_ideal(n, temp_freq) for n in t)
            dist = distance_from_ideal(temp_freq)
            sc = score(t)
            max_dec_cnt = max(Counter(dec(n) for n in t).values())
            key = (0 if all_in else 1, dist, -sc, max_dec_cnt)
            if best_key is None or key < best_key:
                best_key = key
                best_t = t
        # Fallback 1
        if best_t is None:
            for t in avail:
                if any(overlap(t, s) > 3 for s in selected): continue
                if any(freq[n] >= 12 for n in t): continue
                temp_freq = freq.copy()
                for n in t: temp_freq[n] += 1
                dist = distance_from_ideal(temp_freq)
                max_dec_cnt = max(Counter(dec(n) for n in t).values())
                key = (1, dist, -score(t), max_dec_cnt)
                if best_key is None or key < best_key:
                    best_key = key
                    best_t = t
        # Fallback 2 (saturation up to 15)
        if best_t is None:
            for t in avail:
                if any(overlap(t, s) > 3 for s in selected): continue
                if any(freq[n] >= 15 for n in t): continue
                temp_freq = freq.copy()
                for n in t: temp_freq[n] += 1
                dist = distance_from_ideal(temp_freq)
                max_dec_cnt = max(Counter(dec(n) for n in t).values())
                key = (2, dist, -score(t), max_dec_cnt)
                if best_key is None or key < best_key:
                    best_key = key
                    best_t = t
        if best_t is None:
            raise RuntimeError(f"No ticket for {spec}")
        selected.append(best_t)
        sel_set.add(best_t)
        for n in best_t: freq[n] += 1
        ticket_specs.append((best_t, spec))
        avail.remove(best_t)

# ---------- POST‑BALANCE ----------
print("Balancing towards ideal frequencies...")
for _ in range(500):
    out_of_ideal = []
    for n in EH:
        if freq[n] < EH_IDEAL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > EH_IDEAL[1]: out_of_ideal.append((n, 'high'))
    for n in H:
        if freq[n] < H_IDEAL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > H_IDEAL[1]: out_of_ideal.append((n, 'high'))
    for n in W:
        if freq[n] < W_IDEAL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > W_IDEAL[1]: out_of_ideal.append((n, 'high'))
    for n in C:
        if freq[n] < C_IDEAL[0]: out_of_ideal.append((n, 'low'))
        elif freq[n] > C_IDEAL[1]: out_of_ideal.append((n, 'high'))
    if not out_of_ideal: break

    num, direction = min(out_of_ideal, key=lambda x: freq[x[0]] if x[1]=='low' else -freq[x[0]])
    improved = False
    for idx, (t, spec) in enumerate(ticket_specs):
        if direction == 'low' and num in t: continue
        if direction == 'high' and num not in t: continue
        candidates = []
        if direction == 'low':
            candidates = [x for x in all_cand[spec] if x not in sel_set and num in x]
        else:
            candidates = [x for x in all_cand[spec] if x not in sel_set and num not in x]
        if not candidates: continue
        for cand in candidates:
            if any(overlap(cand, selected[j]) > 3 for j in range(TOTAL) if j != idx): continue
            temp_freq = freq.copy()
            for n in t: temp_freq[n] -= 1
            for n in cand: temp_freq[n] += 1
            if max(temp_freq.values()) > 12: continue
            old_dist = distance_from_ideal(freq)
            new_dist = distance_from_ideal(temp_freq)
            if new_dist < old_dist:
                sel_set.remove(t)
                sel_set.add(cand)
                for n in t: freq[n] -= 1
                for n in cand: freq[n] += 1
                selected[idx] = cand
                ticket_specs[idx] = (cand, spec)
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
print(f"Max frequency: {max_freq} (need <=12)")
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