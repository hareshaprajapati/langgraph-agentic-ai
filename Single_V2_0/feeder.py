import csv
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# ---------- CONFIG ----------
CSV_FILE = "cross_lotto_data_backup.csv"
TARGET_DAY = "Wed"              # day we want to predict
NUM_HISTORICAL = 10             # number of past target-day draws to use for feeder weights
FUTURE_DATE = "Wed 09-Sep-2026" # date we want to predict
PROFILE = (3, 1, 2, 0)          # (EH, H, W, C) – leave None to auto-predict

# ---------- HELPERS ----------
def parse_date(s):
    return datetime.strptime(s[4:], '%d-%b-%Y')

def extract_main_numbers(cell):
    if not cell:
        return []
    main_part = cell.split(']')[0].replace('[', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

def extract_all_numbers(cell):
    nums = []
    for part in cell.split(']'):
        part = part.replace('[', '').strip()
        if part:
            for token in part.split(','):
                token = token.strip()
                if token:
                    nums.append(int(token))
    return nums

def load_draws(csv_file):
    draws = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            date_str = row[0].strip()
            try:
                dt = parse_date(date_str)
            except:
                continue
            day_abbr = dt.strftime('%a')[:3]
            sfl_cell = row[1] if len(row) > 1 else ''
            others_cell = row[2] if len(row) > 2 else ''
            draws.append({
                'date': dt,
                'day': day_abbr,
                'sfl_main': extract_main_numbers(sfl_cell),
                'sfl_all': extract_all_numbers(sfl_cell),
                'others_main': extract_main_numbers(others_cell),
                'others_all': extract_all_numbers(others_cell)
            })
    draws.sort(key=lambda x: x['date'])
    return draws

# ---------- MAIN ----------
draws = load_draws(CSV_FILE)
target_dt = parse_date(FUTURE_DATE)
target_day_abbr = TARGET_DAY[:3]

# Get all target-day draws (e.g., all Wednesdays)
target_draws = [d for d in draws if d['day'] == target_day_abbr]

# Take the last NUM_HISTORICAL target draws
historical_targets = target_draws[-NUM_HISTORICAL:] if len(target_draws) >= NUM_HISTORICAL else target_draws

if not historical_targets:
    print(f"No {TARGET_DAY} draws found.")
    exit()

print(f"Using last {len(historical_targets)} {TARGET_DAY} draws to compute feeder weights.\n")

# ---------- COMPUTE FEEDER WEIGHTS ----------
feeder_overlaps = {
    'sfl': defaultdict(list),
    'others': defaultdict(list)
}

for tdraw in historical_targets:
    tdate = tdraw['date']
    target_main = set(tdraw['others_main'])  # We are predicting Others, so target is Others main
    for offset in range(1, 8):
        check_date = tdate - timedelta(days=offset)
        feeder = next((d for d in draws if d['date'] == check_date), None)
        if feeder is None:
            continue
        day = feeder['day']
        overlap_sfl = len(set(feeder['sfl_all']) & target_main)
        feeder_overlaps['sfl'][day].append(overlap_sfl)
        overlap_others = len(set(feeder['others_all']) & target_main)
        feeder_overlaps['others'][day].append(overlap_others)

# Compute average overlap per (column, day)
feeder_weights = {}
for col, day_dict in feeder_overlaps.items():
    for day, overlaps in day_dict.items():
        if overlaps:
            avg = sum(overlaps) / len(overlaps)
            feeder_weights[(col, day)] = avg

# ---------- PRINT FEEDER WEIGHTS ----------
print("Computed feeder weights (average overlap with Wednesday's Others main numbers):")
print(f"{'Column':<8} {'Day':<6} {'Weight':<8}")
print("-" * 25)
for (col, day), weight in sorted(feeder_weights.items(), key=lambda x: -x[1]):
    print(f"{col:<8} {day:<6} {weight:.2f}")
print()

# ---------- SCORE NUMBERS FOR THE FUTURE DATE ----------
week_draws = []
for offset in range(1, 8):
    check_date = target_dt - timedelta(days=offset)
    feeder = next((d for d in draws if d['date'] == check_date), None)
    if feeder:
        week_draws.append(feeder)

scores = defaultdict(float)
for feeder in week_draws:
    day = feeder['day']
    weight_sfl = feeder_weights.get(('sfl', day), 0)
    if weight_sfl > 0:
        for num in feeder['sfl_all']:
            scores[num] += weight_sfl
    weight_others = feeder_weights.get(('others', day), 0)
    if weight_others > 0:
        for num in feeder['others_all']:
            scores[num] += weight_others

# ---------- POOL DEFINITIONS ----------
prev_target = None
for d in target_draws:
    if d['date'] < target_dt:
        prev_target = d
    else:
        break
if prev_target is None:
    print("No previous target draw found.")
    exit()

window_nums = []
for d in draws:
    if prev_target['date'] <= d['date'] < target_dt:
        window_nums.extend(d['sfl_all'])
        window_nums.extend(d['others_all'])
window_nums = [n for n in window_nums if 1 <= n <= 45]
counter = Counter(window_nums)
EH = {n for n, cnt in counter.items() if cnt >= 4}
H  = {n for n, cnt in counter.items() if cnt == 3}
W  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
C  = {n for n in range(1, 46) if counter[n] == 0}

# ---------- SELECT TOP SCORES PER POOL ----------
def top_n(pool, n):
    candidates = [(num, scores[num]) for num in pool if num in scores]
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]

eh_top = top_n(EH, 10)
h_top  = top_n(H, 10)
w_top  = top_n(W, 10)

print("Top EH numbers:")
for num, sc in eh_top:
    print(f"  {num:2d} : {sc:.2f}")
print("\nTop H numbers:")
for num, sc in h_top:
    print(f"  {num:2d} : {sc:.2f}")
print("\nTop W numbers:")
for num, sc in w_top:
    print(f"  {num:2d} : {sc:.2f}")

# ---------- GENERATE TICKETS ----------
need_eh, need_h, need_w, need_c = PROFILE
tickets = []
for i in range(6):
    eh_choice = [eh_top[i % len(eh_top)][0],
                 eh_top[(i+1) % len(eh_top)][0],
                 eh_top[(i+2) % len(eh_top)][0]]
    h_choice = h_top[i % len(h_top)][0]
    w1 = w_top[i % len(w_top)][0]
    w2 = w_top[(i+1) % len(w_top)][0]
    ticket = sorted(eh_choice + [h_choice] + [w1, w2])
    if len(ticket) == 6:
        tickets.append(ticket)

print("\nGenerated Tickets (Profile: EH={}, H={}, W={}, C={}):".format(need_eh, need_h, need_w, need_c))
for idx, ticket in enumerate(tickets, 1):
    print(f"Ticket {idx}: {ticket}")