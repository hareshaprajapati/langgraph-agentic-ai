import pandas as pd
from datetime import timedelta
from collections import Counter, defaultdict
import math

# =============================
# CONFIG
# =============================
CSV_PATH = "lotto_last_3_months.csv"  # <-- your file
TARGET_DATE_STR = "2026-01-10" # <-- change target date (Saturday)
TARGET_DECADE = {1: 2, 2: 0, 3: 3, 4:0 , 5: 1}
SKIP_FRIDAY = False                   # <-- your idea
WINDOWS = [1, 2, 3, 4, 5, 6]
TOP_K_NEIGHBORS = 12
BIN_SIZE = 0.10                      # decade share bins (0.0..1.0)
SMOOTHING = 1.0                      # Laplace smoothing
TOP_PATTERNS_TO_PRINT = 10

# =============================
# RECENCY PENALTY (CONFIGURABLE)
# =============================

USE_RECENCY_PENALTY = True

# Two modes:
# - "step": uses RECENCY_STEP_TABLE
# - "exp" : uses exponential decay with half-life
RECENCY_PENALTY_MODE = "step"   # "step" or "exp"

# STEP mode:
# if years_since_last_seen < threshold_years => multiply by factor
# evaluated in order (first match wins)
RECENCY_STEP_TABLE = [
    (1.0, 0.50),   # seen within last 1 year => strong penalty
    (2.0, 0.70),   # seen within 1-2 years
    (3.0, 0.85),   # seen within 2-3 years
]
RECENCY_STEP_DEFAULT = 1.00     # >= last threshold => no penalty

# EXP mode:
# factor = max(RECENCY_EXP_MIN_FACTOR, 0.5 ** (years / RECENCY_EXP_HALF_LIFE_YEARS))
RECENCY_EXP_HALF_LIFE_YEARS = 1.5
RECENCY_EXP_MIN_FACTOR = 0.35

# If a pattern has NEVER been seen in training history (before target date),
# treat it as "old enough" => no penalty.
RECENCY_NEVER_SEEN_YEARS = 99.0

# Decades (Saturday Lotto 1-45)
DECADES = {
    "D1": (1, 10),
    "D2": (11, 20),
    "D3": (21, 30),
    "D4": (31, 40),
    "D5": (41, 45),
}
DECADE_KEYS = ["D1", "D2", "D3", "D4", "D5"]


# =============================
# HELPERS
# =============================
def parse_main_numbers(cell):
    """Extract main numbers from '[..], [supp]' format. Return [] if blank."""
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    s = str(cell)
    if "]" not in s:
        # if it's already plain, try parse anyway
        s = s.strip()
    main_part = s.split("]")[0].strip().lstrip("[")
    nums = []
    for tok in main_part.split(","):
        tok = tok.strip()
        if tok.isdigit():
            nums.append(int(tok))
    return nums


def decade_of(n: int):
    for dk, (a, b) in DECADES.items():
        if a <= n <= b:
            return dk
    return None


def decade_counts(nums):
    c = Counter()
    for n in nums:
        if 1 <= n <= 45:
            dk = decade_of(n)
            if dk:
                c[dk] += 1
    # ensure all keys
    for dk in DECADE_KEYS:
        c[dk] += 0
    return c


def decade_shares(nums):
    c = decade_counts(nums)
    total = sum(c.values())
    if total == 0:
        return {dk: 0.0 for dk in DECADE_KEYS}
    return {dk: c[dk] / total for dk in DECADE_KEYS}


def bin_share(x: float):
    # clamp [0,1], bin to BIN_SIZE steps
    x = max(0.0, min(1.0, float(x)))
    b = int(x / BIN_SIZE)
    if b >= int(1.0 / BIN_SIZE):
        b = int(1.0 / BIN_SIZE) - 1
    return b


def feature_key_from_shares(shares: dict):
    # tuple of binned shares in D1..D5 order
    return tuple(bin_share(shares[dk]) for dk in DECADE_KEYS)


def cosine_sim(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def softmax(xs):
    m = max(xs) if xs else 0.0
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) if exps else 1.0
    return [e / s for e in exps]

def build_last_seen_map(train_sats_df):
    """
    Returns {pattern_tuple: last_seen_timestamp} using ONLY Saturdays before target_date.
    """
    last_seen = {}
    for _, sat in train_sats_df.iterrows():
        pat = saturday_pattern(sat)
        dt = sat["Date"]
        prev = last_seen.get(pat)
        if prev is None or dt > prev:
            last_seen[pat] = dt
    return last_seen


def years_since(dt_last, dt_target):
    if dt_last is None:
        return RECENCY_NEVER_SEEN_YEARS
    return (dt_target - dt_last).days / 365.25


def recency_penalty_factor(years_ago: float) -> float:
    """
    Returns a multiplier in (0..1].
    """
    if not USE_RECENCY_PENALTY:
        return 1.0

    if RECENCY_PENALTY_MODE == "step":
        for thresh, factor in RECENCY_STEP_TABLE:
            if years_ago < thresh:
                return float(factor)
        return float(RECENCY_STEP_DEFAULT)

    if RECENCY_PENALTY_MODE == "exp":
        # 0.5^(years/half_life)
        f = 0.5 ** (years_ago / float(RECENCY_EXP_HALF_LIFE_YEARS))
        return float(max(RECENCY_EXP_MIN_FACTOR, f))

    # Unknown mode => disable penalty safely
    return 1.0

# =============================
# LOAD + NORMALIZE
# =============================
df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="%a %d-%b-%Y", errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

target_date = pd.to_datetime(TARGET_DATE_STR)

# Identify Saturdays (weekday=5)
df["weekday"] = df["Date"].dt.weekday
saturdays = df[df["weekday"] == 5].copy()

# Saturday Lotto numbers come from Others column on Saturday
def saturday_pattern(row):
    sat_nums = parse_main_numbers(row.get("Others (incl supp)", ""))
    c = decade_counts(sat_nums)
    # pattern as tuple counts (sum should be 6 for Sat Lotto main numbers)
    return tuple(c[dk] for dk in DECADE_KEYS)

# Build a day->numbers cache (learning pool uses Set mains + Others mains)
def day_learning_nums(row):
    nums = []
    nums += parse_main_numbers(row.get("Set for Life (incl supp)", ""))
    nums += parse_main_numbers(row.get("Others (incl supp)", ""))
    # keep only 1..45 for decade learning
    return [n for n in nums if 1 <= n <= 45]

df["learn_nums"] = df.apply(day_learning_nums, axis=1)

# =============================
# WINDOW DEFINITIONS
# =============================
def window_dates_for_sat(sat_date, Y):
    """
    Return list of dates to include for a Saturday, considering:
    - prior Y calendar days
    - optionally skip Friday
    """
    days = []
    for d in range(1, Y+1):
        days.append(sat_date - timedelta(days=d))
    if SKIP_FRIDAY:
        days = [d for d in days if d.weekday() != 4]  # Friday = 4
    return sorted(days)

def window_feature_vector(sat_date, Y):
    """
    Compute decade share vector over the window days:
    sum all learn_nums from those days, then decade shares.
    """
    days = window_dates_for_sat(sat_date, Y)
    w = df[df["Date"].isin(days)]
    nums = []
    for _, r in w.iterrows():
        nums.extend(r["learn_nums"])
    shares = decade_shares(nums)
    # return shares in fixed order
    return [shares[dk] for dk in DECADE_KEYS], shares, days

# =============================
# TRAINING DATA (only Saturdays before target_date)
# =============================
train_sats = saturdays[saturdays["Date"] < target_date].copy()
if len(train_sats) < 6:
    raise RuntimeError(f"Not enough Saturday history before {TARGET_DATE_STR}. Found {len(train_sats)}")

# Per-window training stores:
# - binned mapping: key -> Counter(pattern)
# - raw feature vectors: list[(vec, pattern)]
model_bins = {Y: defaultdict(Counter) for Y in WINDOWS}
model_vecs = {Y: [] for Y in WINDOWS}

for _, sat in train_sats.iterrows():
    sd = sat["Date"]
    pat = saturday_pattern(sat)

    for Y in WINDOWS:
        vec, shares, days = window_feature_vector(sd, Y)
        k = feature_key_from_shares(shares)
        model_bins[Y][k][pat] += 1
        model_vecs[Y].append((vec, pat))

# =============================
# WINDOW QUALITY (learn weights from history)
# =============================
# We estimate each window's usefulness by leave-one-out log probability
# using binned model (with smoothing). Higher is better.
def loo_logprob_for_window(Y):
    data = model_vecs[Y]
    if len(data) < 8:
        return -999.0

    total_lp = 0.0
    for i, (vec_i, pat_i) in enumerate(data):
        # Build binned key for this point
        shares_i = {dk: vec_i[j] for j, dk in enumerate(DECADE_KEYS)}
        k_i = feature_key_from_shares(shares_i)

        # counts excluding i: approximate by subtracting one from that pattern if present
        cnt = model_bins[Y][k_i].copy()
        if cnt.get(pat_i, 0) > 0:
            cnt[pat_i] -= 1
            if cnt[pat_i] <= 0:
                del cnt[pat_i]

        # if empty after removal, fallback small penalty
        if not cnt:
            total_lp += -6.0
            continue

        # Laplace smoothing over observed patterns in this bin
        patterns = list(cnt.keys())
        denom = sum(cnt.values()) + SMOOTHING * len(patterns)
        num = cnt.get(pat_i, 0) + SMOOTHING
        p = num / denom
        total_lp += math.log(p + 1e-12)
    return total_lp / len(data)

window_scores = [loo_logprob_for_window(Y) for Y in WINDOWS]
window_weights = softmax(window_scores)  # higher logprob -> higher weight
window_weight_map = {Y: w for Y, w in zip(WINDOWS, window_weights)}

# =============================
# PREDICT FOR TARGET_DATE (scan all windows, ensemble)
# =============================
def predict_dist_for_window(Y, target_date):
    vec_t, shares_t, days_t = window_feature_vector(target_date, Y)
    key_t = feature_key_from_shares(shares_t)

    # 1) if bin seen, use binned empirical dist with smoothing
    cnt = model_bins[Y].get(key_t)
    if cnt and sum(cnt.values()) > 0:
        patterns = list(cnt.keys())
        denom = sum(cnt.values()) + SMOOTHING * len(patterns)
        dist = {p: (cnt[p] + SMOOTHING) / denom for p in patterns}
        return dist, vec_t, shares_t, days_t, "bin"

    # 2) fallback: kNN over raw vectors
    sims = []
    for vec_h, pat_h in model_vecs[Y]:
        s = cosine_sim(vec_t, vec_h)
        sims.append((s, pat_h))
    sims.sort(reverse=True, key=lambda x: x[0])
    sims = sims[:TOP_K_NEIGHBORS]

    if not sims or sims[0][0] <= 0:
        # ultimate fallback: global pattern frequency for that window
        global_cnt = Counter(p for _, p in model_vecs[Y])
        denom = sum(global_cnt.values()) + SMOOTHING * len(global_cnt)
        dist = {p: (global_cnt[p] + SMOOTHING) / denom for p in global_cnt}
        return dist, vec_t, shares_t, days_t, "global"

    # weighted by similarity (shift to positive)
    dist_cnt = Counter()
    wsum = 0.0
    for s, p in sims:
        w = max(0.0, s)
        dist_cnt[p] += w
        wsum += w
    if wsum == 0:
        wsum = 1.0
    dist = {p: dist_cnt[p] / wsum for p in dist_cnt}
    return dist, vec_t, shares_t, days_t, "knn"

# Collect per-window distributions
per_window = {}
for Y in WINDOWS:
    dist, vec_t, shares_t, days_t, mode = predict_dist_for_window(Y, target_date)
    per_window[Y] = {
        "dist": dist,
        "vec": vec_t,
        "shares": shares_t,
        "days": days_t,
        "mode": mode,
        "w": window_weight_map[Y],
    }

# Ensemble distribution across windows
# Ensemble distribution across windows
ensemble = Counter()
for Y in WINDOWS:
    wY = per_window[Y]["w"]
    dist = per_window[Y]["dist"]
    for pat, p in dist.items():
        ensemble[pat] += wY * p

# -----------------------------
# RECENCY PENALTY (soft)
# -----------------------------
last_seen_map = build_last_seen_map(train_sats)  # train_sats already = Saturdays < target_date

if USE_RECENCY_PENALTY and ensemble:
    for pat in list(ensemble.keys()):
        last_dt = last_seen_map.get(pat)
        yrs = years_since(last_dt, target_date)
        ensemble[pat] *= recency_penalty_factor(yrs)

# Normalize (after penalty)
total_p = sum(ensemble.values()) if ensemble else 1.0
if total_p <= 0:
    total_p = 1.0
ensemble = {pat: ensemble[pat] / total_p for pat in ensemble}


# Expected counts per decade
exp = {dk: 0.0 for dk in DECADE_KEYS}
for pat, p in ensemble.items():
    for i, dk in enumerate(DECADE_KEYS):
        exp[dk] += pat[i] * p

# Print results
print("\n==============================")
print("DECADE PREDICTOR RESULT")
print("==============================")
print(f"Target date: {target_date.date()}  (SKIP_FRIDAY={SKIP_FRIDAY})")

print("\n--- Learned window weights (higher = better from history) ---")
for Y in sorted(WINDOWS):
    print(f"Y={Y}: weight={window_weight_map[Y]:.3f}  (score={window_scores[WINDOWS.index(Y)]:.3f})")

bestY = max(window_weight_map.items(), key=lambda x: x[1])[0]
print(f"\nBest learned window (by weight): Y={bestY}")

print("\n--- Window scan details (target date) ---")
for Y in sorted(WINDOWS):
    info = per_window[Y]
    days = ", ".join(d.strftime("%Y-%m-%d") for d in info["days"])
    shares = info["shares"]
    print(f"Y={Y} mode={info['mode']} weight={info['w']:.3f} days=[{days}] shares="
          f"{{D1:{shares['D1']:.2f}, D2:{shares['D2']:.2f}, D3:{shares['D3']:.2f}, D4:{shares['D4']:.2f}, D5:{shares['D5']:.2f}}}")

print("\n--- Ensemble expected decade counts (sum≈6) ---")
print(", ".join([f"{dk}={exp[dk]:.2f}" for dk in DECADE_KEYS]))

print("\n--- Top predicted decade patterns (counts across D1..D5 summing to 6) ---")
top = sorted(ensemble.items(), key=lambda x: x[1], reverse=True)[:TOP_PATTERNS_TO_PRINT]
for i, (pat, p) in enumerate(top, 1):
    pat_str = ", ".join(f"{dk}={pat[j]}" for j, dk in enumerate(DECADE_KEYS))
    print(f"{i:02d}) P={p:.3f}  [{pat_str}]")

print("\n(Each pattern is a Saturday Lotto decade quota for 6 numbers.)")
