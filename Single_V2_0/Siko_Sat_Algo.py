import csv
import itertools
import re
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ================= CONFIGURATION =================
CSV_FILE = "cross_lotto_data_backup.csv"

RUN_BACKTEST = False          # True = run backtest, False = future prediction

TARGET_DATE = "2026-08-15"   # Used only if RUN_BACKTEST = False

# Option A: provide EH/H/W/C pools directly (total should be 15)
EH = []   # e.g., [1, 4, 6, 7, 8, 11]
H  = []   # e.g., [2, 5, 9, 12, 15]
W  = []   # e.g., [3, 10, 13, 17, 20]
C  = []   # e.g., [14]

# Option B: provide a single 15-number pool list
POOL = [1, 2, 7, 12, 13, 17, 20, 23, 25, 26, 27, 30, 31, 33, 34]

TOTAL = 50
kill_list = ["40s"]

SAFE_DEPTH_RANGES = [(0, 2), (0, 3), (2, 5), (0, 1)]
BAND_CAPS = {'0x': 3, '1x': 4, '2x': 4, '3x': 4, '4x': 3, '5x+': 3}

# ================= HELPERS =================
def band_label(c):
    if c >= 5: return '5x+'
    if c == 4: return '4x'
    if c == 3: return '3x'
    if c == 2: return '2x'
    if c == 1: return '1x'
    return '0x'

def parse_kill(s):
    return set(s.split('+')) if s != 'none' else set()

def dec(n):
    if n <= 9: return '0s'
    if n <= 19: return '10s'
    if n <= 29: return '20s'
    if n <= 39: return '30s'
    return '40s'

def consecutive(t):
    s = sorted(t)
    return any(s[i+1]-s[i] == 1 for i in range(5))

def mirror(t):
    return len({x % 10 for x in t}) < 6

def parse_nums(s):
    if pd.isna(s) or s == "":
        return []
    return [int(n) for n in re.findall(r"\d+", s)]

def parse_all_from_others(s):
    """
    Parse all main + supplement numbers from Others string.
    Supports formats like:
      [1,2,3,4,5,6], [7,8]
      [1,2,3,4,5,6] [7,8]
      1 2 3 4 5 6  7 8
    """
    if pd.isna(s) or s == "":
        return []
    s = str(s)

    # Try to find numbers inside square brackets (all occurrences)
    bracket_groups = re.findall(r"\[(.*?)\]", s)
    if bracket_groups:
        all_nums = []
        for group in bracket_groups:
            nums = [int(x.strip()) for x in group.split(',') if x.strip().isdigit()]
            all_nums.extend(nums)
        return all_nums

    # If no brackets, check for parentheses after main
    paren_groups = re.findall(r"\((.*?)\)", s)
    if paren_groups:
        main_nums = [int(n) for n in re.findall(r"\d+", s.split('(')[0])]
        supp_nums = []
        for group in paren_groups:
            supp_nums.extend([int(x.strip()) for x in group.split(',') if x.strip().isdigit()])
        return main_nums + supp_nums

    # Fallback: extract all numbers in order; first 6 are main, rest are supp
    return [int(n) for n in re.findall(r"\d+", s)]

def parse_main_from_others(s):
    if pd.isna(s) or s == "":
        return []
    m = re.search(r"\[(.*?)\]", s)
    if m:
        nums = [int(x.strip()) for x in m.group(1).split(',') if x.strip().isdigit()][:6]
        return sorted(nums)
    return []

def get_others_col(df):
    for col in df.columns:
        if "Others" in col:
            return col
    raise KeyError("No Others column")

def get_tiers(df_window):
    pool = []
    for _, r in df_window.iterrows():
        pool.extend(parse_nums(r["Set for Life (incl supp)"]))
        pool.extend(parse_nums(r[others_col]))
    counts = Counter([n for n in pool if 1 <= n <= 45])

    EH = sorted([n for n, c in counts.items() if c >= 4])
    H = sorted([n for n, c in counts.items() if c == 3])
    W = sorted([n for n, c in counts.items() if 1 <= c <= 2])
    C = sorted([n for n in range(1, 46) if n not in counts])

    if len(H) < 4:
        pseudo = [n for n, c in counts.items() if c == 2]
        H = sorted(set(H) | set(pseudo))
        W = [n for n in W if n not in pseudo]

    return EH, H, W, C, counts

def ticket_band_bonus(t, band_for_num):
    bonus = 0
    for n in t:
        b = band_for_num.get(n, '0x')
        if b == '0x': bonus += 1.5
        elif b == '1x': bonus += 2.0
        elif b == '2x': bonus += 1.0
        elif b == '3x': bonus += 0.5
        elif b == '4x': bonus += 0.2
        elif b == '5x+': bonus += 1.0
    return bonus

# ================= LOAD DATA =================
df = pd.read_csv(CSV_FILE)
df["Date_dt"] = pd.to_datetime(df["Date"], format="%a %d-%b-%Y", errors="coerce")
df = df.sort_values("Date_dt")
others_col = get_others_col(df)

# Keep raw Others column for supplement parsing
sat_rows = df[df["Date"].str.startswith("Sat")].copy()
sat_rows["nums"] = sat_rows[others_col].apply(parse_main_from_others)
sat_draws = sat_rows[["Date_dt", "nums", others_col]].dropna(subset=["nums"]).sort_values("Date_dt")

# ================= FEATURE CONTEXT =================
def compute_feature_context(window7, prior_sat, all_prior_sat):
    prev_sat = prior_sat.tail(1)
    LEGACY = prev_sat.iloc[0]["nums"] if not prev_sat.empty else []

    freq_20w = Counter()
    for _, r in prior_sat.iterrows():
        for n in r["nums"]:
            freq_20w[n] += 1

    total_sat_freq = Counter()
    for _, r in all_prior_sat.iterrows():
        for n in r["nums"]:
            total_sat_freq[n] += 1

    all_prior_list = [(r["Date_dt"], r["nums"]) for _, r in all_prior_sat.iterrows()]
    last_seen = {}
    for idx, (_, nums) in enumerate(all_prior_list):
        for n in nums:
            last_seen[n] = idx

    max_idx = len(all_prior_list) - 1
    gap = {n: max_idx - last_seen.get(n, -1) for n in range(1, 46)}

    band_for_num = {n: band_label(freq_20w.get(n, 0)) for n in range(1, 46)}

    _, _, _, _, counts = get_tiers(window7)

    return {
        'LEGACY': LEGACY,
        'freq20': freq_20w,
        'total_freq': total_sat_freq,
        'gap': gap,
        'band_for_num': band_for_num,
        'counts': counts,
    }

# ================= FEATURES =================
def make_features(t, ctx):
    counts = ctx['counts']
    freq20 = ctx['freq20']
    total_freq = ctx['total_freq']
    LEGACY = ctx['LEGACY']
    band_for_num = ctx['band_for_num']
    gap = ctx['gap']

    return {
        'f7': sum(counts.get(n, 0) for n in t),
        'f20': sum(freq20.get(n, 0) for n in t),
        'ftot': sum(total_freq.get(n, 0) for n in t),
        'legacy': sum(1 for n in t if n in LEGACY),
        'band': ticket_band_bonus(t, band_for_num),
        'gap_sum': sum(gap.get(n, 0) for n in t),
        'gap_min': min(gap.get(n, 0) for n in t),
        'gap_max': max(gap.get(n, 0) for n in t),
        'decade_max': max(Counter(dec(x) for x in t).values()),
        'oe_ratio': sum(1 for x in t if x % 2) / 6,
        'hl_ratio': sum(1 for x in t if x <= 22) / 6,
    }

# ================= COMBO GENERATION =================
def generate_combos(eh_use, h_use, w_use, c_use, ctx):
    LEGACY = ctx['LEGACY']
    band_for_num = ctx['band_for_num']
    combos = []

    for eh_c in range(SAFE_DEPTH_RANGES[0][0], SAFE_DEPTH_RANGES[0][1] + 1):
        if eh_c > len(eh_use): continue
        for h_c in range(SAFE_DEPTH_RANGES[1][0], SAFE_DEPTH_RANGES[1][1] + 1):
            if h_c > len(h_use): continue
            for w_c in range(SAFE_DEPTH_RANGES[2][0], SAFE_DEPTH_RANGES[2][1] + 1):
                if w_c > len(w_use): continue
                for c_c in range(SAFE_DEPTH_RANGES[3][0], SAFE_DEPTH_RANGES[3][1] + 1):
                    if c_c > len(c_use): continue
                    if eh_c + h_c + w_c + c_c != 6: continue

                    for eh in itertools.combinations(eh_use, eh_c):
                        for h in itertools.combinations(h_use, h_c):
                            for w in itertools.combinations(w_use, w_c):
                                for c in itertools.combinations(c_use, c_c):
                                    t = tuple(sorted(eh + h + w + c))

                                    if len(set(t)) < 6: continue
                                    if sum(1 for x in t if x in LEGACY) > 1: continue

                                    o = sum(1 for x in t if x % 2)
                                    if (o, 6 - o) not in [(3, 3), (2, 4), (4, 2), (1, 5), (5, 1)]: continue

                                    lo = sum(1 for x in t if x <= 22)
                                    if (lo, 6 - lo) not in [(3, 3), (2, 4), (4, 2), (1, 5), (5, 1)]: continue

                                    if max(Counter(dec(x) for x in t).values()) > 3: continue
                                    if sum(1 for x in t if 40 <= x <= 45) > 2: continue
                                    if not (consecutive(t) or mirror(t)): continue

                                    band_cnt = Counter(band_for_num[x] for x in t)
                                    if any(band_cnt[b] > BAND_CAPS[b] for b in band_cnt): continue

                                    combos.append(t)

    return combos

# ================= FORCED-WINNER POOL (BACKTEST) =================
def build_forced_pool_and_combos(target_date, real_nums, window7, prior_sat, all_prior_sat):
    REAL = set(real_nums)

    if window7.empty:
        return None, None, None

    ctx = compute_feature_context(window7, prior_sat, all_prior_sat)
    EH, H, W, C, _ = get_tiers(window7)

    killed_set = parse_kill(kill_list[0])

    EH_use_orig = [n for n in EH if dec(n) not in killed_set]
    H_use_orig  = [n for n in H if dec(n) not in killed_set]
    W_use_orig  = [n for n in W if dec(n) not in killed_set]
    C_use_orig  = [n for n in C if dec(n) not in killed_set]

    def band_bonus(n):
        b = ctx['band_for_num'].get(n, '0x')
        return {'0x': 1.5, '1x': 2.0, '2x': 1.0, '3x': 0.5, '4x': 0.2, '5x+': 1.0}.get(b, 0)

    def fill_score(n):
        return 3.0 * band_bonus(n) + ctx['counts'].get(n, 0) + 0.1 * ctx['freq20'].get(n, 0)

    quota = {"EH": 4, "H": 4, "W": 5, "C": 2}

    eh_use = sorted(EH_use_orig, key=fill_score, reverse=True)[:quota["EH"]]
    h_use  = sorted(H_use_orig,  key=fill_score, reverse=True)[:quota["H"]]
    w_use  = sorted(W_use_orig,  key=fill_score, reverse=True)[:quota["W"]]
    c_use  = sorted(C_use_orig,  key=fill_score, reverse=True)[:quota["C"]]

    pool_set = set(eh_use) | set(h_use) | set(w_use) | set(c_use)
    missing = REAL - pool_set

    if missing:
        for n in missing:
            if n in EH_use_orig: eh_use.append(n)
            elif n in H_use_orig: h_use.append(n)
            elif n in W_use_orig: w_use.append(n)
            else: c_use.append(n)

        all_pool = eh_use + h_use + w_use + c_use
        all_pool = sorted(set(all_pool), key=lambda n: (-int(n in REAL), -fill_score(n)))[:15]

        eh_use = [n for n in all_pool if n in EH_use_orig]
        h_use  = [n for n in all_pool if n in H_use_orig]
        w_use  = [n for n in all_pool if n in W_use_orig]
        c_use  = [n for n in all_pool if n in C_use_orig]

    combos = generate_combos(eh_use, h_use, w_use, c_use, ctx)
    features = [make_features(t, ctx) for t in combos]

    full_pool = sorted(set(eh_use) | set(h_use) | set(w_use) | set(c_use))

    return combos, features, full_pool

# ================= TRAINING =================
def build_training_set(limit_date):
    train_rows = []
    training_draws = sat_draws[sat_draws["Date_dt"] < limit_date].tail(200)

    for _, hist_row in training_draws.iterrows():
        hist_date = hist_row["Date_dt"]
        hist_nums = hist_row["nums"]

        if any(dec(n) == "40s" for n in hist_nums):
            continue

        hist_window7 = df[(df["Date_dt"] >= hist_date - timedelta(days=7)) & (df["Date_dt"] < hist_date)]
        if hist_window7.empty:
            continue

        hist_prior_sat = sat_draws[sat_draws["Date_dt"] < hist_date].tail(20)
        hist_all_prior = sat_draws[sat_draws["Date_dt"] < hist_date]

        combos, feats, _ = build_forced_pool_and_combos(
            hist_date,
            hist_nums,
            hist_window7,
            hist_prior_sat,
            hist_all_prior
        )

        if combos is None or len(combos) == 0:
            continue

        win_ticket = tuple(sorted(hist_nums))

        for i, t in enumerate(combos):
            label = len(set(t) & set(win_ticket))  # 0-6 match count
            train_rows.append((feats[i], label))

    if len(train_rows) < 10:
        return None

    train_df = pd.DataFrame([f | {'label': lbl} for f, lbl in train_rows])
    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values

    return X_train, y_train

def train_model(limit_date):
    data = build_training_set(limit_date)
    if data is None:
        return None

    X_train, y_train = data
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)
    )
    model.fit(X_train, y_train)
    return model

# ================= BACKTEST =================
def run_backtest():
    test_draws = []

    for _, row in sat_draws.iloc[::-1].iterrows():
        if any(dec(n) == "40s" for n in row["nums"]):
            continue
        test_draws.append(row)
        if len(test_draws) == 20:
            break

    test_draws.reverse()

    hits = 0
    tested = 0

    for target_row in test_draws:
        target_date = target_row["Date_dt"]
        raw_others = target_row[others_col]
        all_nums = parse_all_from_others(raw_others)
        real_nums = sorted(all_nums[:6]) if len(all_nums) >= 6 else target_row["nums"]
        supp_nums = sorted(all_nums[6:]) if len(all_nums) > 6 else []

        prior_sat = sat_draws[sat_draws["Date_dt"] < target_date].tail(20)
        all_prior_sat = sat_draws[sat_draws["Date_dt"] < target_date]
        window7 = df[(df["Date_dt"] >= target_date - timedelta(days=7)) & (df["Date_dt"] < target_date)]

        if window7.empty:
            continue

        combos, features, pool_used = build_forced_pool_and_combos(
            target_date, real_nums, window7, prior_sat, all_prior_sat
        )

        if combos is None or len(combos) == 0:
            continue

        model = train_model(target_date)
        if model is None:
            continue

        X_test = pd.DataFrame(features).values
        preds = model.predict(X_test)
        ranked_idx = np.argsort(-preds)
        top_tickets = [combos[i] for i in ranked_idx[:TOTAL]]

        jackpot = tuple(sorted(real_nums))
        selected = jackpot in top_tickets

        if selected:
            hits += 1
        tested += 1

        # Compute best rank / match
        best_match = 0
        best_rank = None
        for rank_idx, ticket in enumerate(top_tickets):
            match_count = len(set(ticket) & set(real_nums))
            if match_count > best_match:
                best_match = match_count
                best_rank = rank_idx + 1

        # Check for 5 main + supplement
        five_supp_rank = None
        if supp_nums:
            for rank_idx, ticket in enumerate(top_tickets):
                main_matches = len(set(ticket) & set(real_nums))
                if main_matches == 5:
                    supp_matches = len(set(ticket) & set(supp_nums))
                    if supp_matches >= 1:
                        five_supp_rank = rank_idx + 1
                        break

        supp_str = f"5+supp: #{five_supp_rank}" if five_supp_rank else "5+supp: None"

        print(f"Target {target_date.strftime('%d-%b-%Y')}: jackpot selected? {selected} | Best: #{best_rank} ({best_match} matches) | {supp_str} | Pool: {pool_used}")

    print(f"\nBacktest result: {hits}/{tested} jackpot selected.")

# ================= FUTURE PREDICTION =================
def predict_future(target_date_str, pool_list=None, eh_list=None, h_list=None, w_list=None, c_list=None):
    target_date = pd.to_datetime(target_date_str)

    target_window7 = df[(df["Date_dt"] >= target_date - timedelta(days=7)) & (df["Date_dt"] < target_date)]
    target_prior_sat = sat_draws[sat_draws["Date_dt"] < target_date].tail(20)
    target_all_prior = sat_draws[sat_draws["Date_dt"] < target_date]

    if target_window7.empty:
        print("No 7-day window data before target.")
        return

    if eh_list and h_list and w_list and c_list:
        eh_input = eh_list
        h_input = h_list
        w_input = w_list
        c_input = c_list

    elif pool_list:
        EH_tiers, H_tiers, W_tiers, C_tiers, _ = get_tiers(target_window7)

        eh_input = [n for n in pool_list if n in EH_tiers]
        h_input  = [n for n in pool_list if n in H_tiers]
        w_input  = [n for n in pool_list if n in W_tiers]
        c_input  = [n for n in pool_list if n in C_tiers]

        all_tiered = set(eh_input) | set(h_input) | set(w_input) | set(c_input)
        leftovers = [n for n in pool_list if n not in all_tiered]
        w_input.extend(leftovers)

    else:
        print("Please provide EH/H/W/C or POOL.")
        return

    total_pool = set(eh_input) | set(h_input) | set(w_input) | set(c_input)
    if len(total_pool) != 15:
        print(f"Warning: pool size is {len(total_pool)}, expected 15.")

    ctx = compute_feature_context(target_window7, target_prior_sat, target_all_prior)

    killed_set = parse_kill(kill_list[0])
    eh_use = [n for n in eh_input if dec(n) not in killed_set]
    h_use  = [n for n in h_input if dec(n) not in killed_set]
    w_use  = [n for n in w_input if dec(n) not in killed_set]
    c_use  = [n for n in c_input if dec(n) not in killed_set]

    combos = generate_combos(eh_use, h_use, w_use, c_use, ctx)
    features = [make_features(t, ctx) for t in combos]

    if combos is None or len(combos) == 0:
        print("No valid combos generated.")
        return

    model = train_model(target_date)
    if model is None:
        print("Not enough training data.")
        return

    X_target = pd.DataFrame(features).values
    preds = model.predict(X_target)
    ranked_idx = np.argsort(-preds)
    top_tickets = [combos[i] for i in ranked_idx[:TOTAL]]

    print(f"\nTop {TOTAL} predicted tickets for {target_date_str}:\n")
    for i, t in enumerate(top_tickets, 1):
        print(f"{i:2d}: {sorted(t)}")

# ================= MAIN =================
if __name__ == "__main__":
    if RUN_BACKTEST:
        run_backtest()
    else:
        if EH and H and W and C:
            predict_future(TARGET_DATE, eh_list=EH, h_list=H, w_list=W, c_list=C)
        elif POOL:
            predict_future(TARGET_DATE, pool_list=POOL)
        else:
            print("Please provide EH/H/W/C or POOL.")