import itertools
from collections import Counter, defaultdict

# ---------- POOLS (27-Jun-2026) ----------
EH = [6, 9, 10, 11, 16, 21, 26, 28, 29, 33, 44]
H  = [8, 12, 13, 19, 22, 24, 25, 36, 38, 40]
W  = [1, 2, 3, 4, 5, 7, 14, 15, 17, 18, 20, 23, 27, 30, 31, 32, 34, 35, 37, 39, 41, 42, 43]
C  = [45]
LEGACY = [3, 6, 9, 14, 21, 22]
REAL = {15, 17, 24, 28, 36, 37}
WIN = tuple(sorted(REAL))

# ---------- PROFILES & KILLS ----------
PROFILES = [((1, 2, 3, 0), 50)]          # (EH, H, W, C) per ticket, number of tickets
kill_list = ['40s']*15 + ['0s']*12 + ['10s']*12 + ['30s']*11
TOTAL = 50

# ---------- DYNAMIC IDEAL BANDS (fairness by pool) ----------
# Calculate total picks from each pool across all tickets
total_picks_EH = sum(eh * count for (eh, h, w, c), count in PROFILES)
total_picks_H  = sum(h  * count for (eh, h, w, c), count in PROFILES)
total_picks_W  = sum(w  * count for (eh, h, w, c), count in PROFILES)
total_picks_C  = sum(c  * count for (eh, h, w, c), count in PROFILES)

pool_size_EH = len(EH)
pool_size_H  = len(H)
pool_size_W  = len(W)
pool_size_C  = len(C)

def ideal_band(total_picks, pool_size):
    if pool_size == 0:
        return (0, 0)
    avg = total_picks / pool_size
    lo = max(1, int(avg) - 1)
    hi = int(avg) + 1
    return (lo, hi)

EH_IDEAL = ideal_band(total_picks_EH, pool_size_EH)
H_IDEAL  = ideal_band(total_picks_H,  pool_size_H)
W_IDEAL  = ideal_band(total_picks_W,  pool_size_W)
C_IDEAL  = ideal_band(total_picks_C,  pool_size_C)

print(f"Ideal frequency bands:\n  EH: {EH_IDEAL}\n  H:  {H_IDEAL}\n  W:  {W_IDEAL}\n  C:  {C_IDEAL}")

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
    if n in H:  return H_IDEAL[0]  <= freq[n] <= H_IDEAL[1]
    if n in W:  return W_IDEAL[0]  <= freq[n] <= W_IDEAL[1]
    if n in C:  return C_IDEAL[0]  <= freq[n] <= C_IDEAL[1]
    return True

def distance_from_ideal(freq):
    total = 0
    for pool, ideal in [(EH, EH_IDEAL), (H, H_IDEAL), (W, W_IDEAL), (C, C_IDEAL)]:
        target = (ideal[0] + ideal[1]) / 2
        for n in pool:
            total += (freq[n] - target) ** 2
    return total

# ---------- GENERATE & PAIR‑FILTER ----------
print("Generating candidates (pair‑filtered)...")
all_cand = {}
for (eh_c, h_c, w_c, c_c), _ in PROFILES:
    combos = []
    for eh in itertools.combinations(EH, eh_c):
        for h in itertools.combinations(H, h_c):
            for w in itertools.combinations(W, w_c):
                for c in itertools.combinations(C, c_c):
                    t = tuple(sorted(eh+h+w+c))
                    if len(set(t))<6: continue
                    combos.append(t)
    for kill in set(kill_list):
        valid_tix = [t for t in combos if valid(t, kill)]
        # keep only tickets with BOTH consecutive & mirror → faster selection
        valid_tix = [t for t in valid_tix if consecutive(t) and mirror(t)]
        all_cand[(0, kill)] = valid_tix
        print(f"  Kill {kill}: {len(valid_tix)} candidates")

req = defaultdict(int)
for k in kill_list: req[(0, k)] += 1

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
            all_in_ideal = all(in_ideal(n, temp_freq) for n in t)
            dist = distance_from_ideal(temp_freq)
            sc = score(t)
            key = (0 if all_in_ideal else 1, dist, -sc)
            if best_key is None or key < best_key:
                best_key = key
                best_t = t
        # Fallback: allow temporary deviation from ideal
        if best_t is None:
            for t in avail:
                if any(overlap(t, s) > 3 for s in selected): continue
                if any(freq[n] >= 12 for n in t): continue
                temp_freq = freq.copy()
                for n in t: temp_freq[n] += 1
                dist = distance_from_ideal(temp_freq)
                sc = score(t)
                key = (1, dist, -sc)
                if best_key is None or key < best_key:
                    best_key = key
                    best_t = t
        # Last resort: allow saturation up to 15
        if best_t is None:
            for t in avail:
                if any(overlap(t, s) > 3 for s in selected): continue
                if any(freq[n] >= 15 for n in t): continue
                temp_freq = freq.copy()
                for n in t: temp_freq[n] += 1
                dist = distance_from_ideal(temp_freq)
                sc = score(t)
                key = (2, dist, -sc)
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
for _ in range(300):
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
    if not out_of_ideal:
        break

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
            if any(overlap(cand, selected[j]) > 3 for j in range(TOTAL) if j != idx):
                continue
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

print("\n--- HIT ANALYSIS ---")
hits = [(t, matches(t, REAL)) for t in selected if matches(t, REAL) >= 3]
print(f"≥3 hits: {len(hits)}")
for t, m in sorted(hits, key=lambda x: -x[1])[:10]:
    print(f"  {sorted(t)} -> {m}")

print(f"\n🎯 Jackpot ticket in set: {WIN in sel_set}")

print(f"\nNumber frequencies (target EH:{EH_IDEAL}, H:{H_IDEAL}, W:{W_IDEAL}, C:{C_IDEAL}):")
for n in sorted(freq):
    if freq[n] > 0:
        pool = 'EH' if n in EH else ('H' if n in H else ('W' if n in W else 'C'))
        print(f"  {pool} {n:2d}: {freq[n]}")

print("\nFinal Tickets:")
for i, t in enumerate(selected, 1):
    print(f"{i:2d}: {sorted(t)}")