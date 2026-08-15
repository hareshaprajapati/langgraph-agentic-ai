import pandas as pd
import re
from collections import Counter
from datetime import timedelta

SET_FOR_LIFE_COL = 'Set for Life (incl supp)'

def get_others_col(df):
    for col in df.columns:
        if 'Others' in col and ('Weekday windfall' in col or 'OZ Lotto' in col or 'Powerball' in col or 'Saturday Lotto' in col):
            return col
    for col in df.columns:
        if 'Others' in col:
            return col
    raise KeyError("Could not find 'Others' column")

def parse_nums(s):
    if pd.isna(s) or s == "":
        return []
    return [int(n) for n in re.findall(r'\d+', s)]

def parse_main_6(s):
    if pd.isna(s) or s == "":
        return []
    match = re.search(r'\[(.*?)\]', s)
    if match:
        nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip().isdigit()]
        return sorted(nums[:6])
    return []

def decade(n):
    return n // 10

# ---------- Load CSV ----------
df = pd.read_csv('cross_lotto_data_backup.csv')
df['Date_dt'] = pd.to_datetime(df['Date'], format='%a %d-%b-%Y')
df = df.sort_values('Date_dt')

others_col = get_others_col(df)

sat_rows = df[df['Date'].str.startswith('Sat')].copy()
sat_rows['nums'] = sat_rows[others_col].apply(parse_main_6)
sat_draws = sat_rows[['Date_dt', 'nums']].dropna(subset=['nums']).sort_values('Date_dt')

# Store all draw features
records = []

print("Extracting features for each draw...")

for _, row in sat_draws.iterrows():
    target_date = row['Date_dt']
    winning = row['nums']

    # Sun-Fri window
    window = df[(df['Date_dt'] >= target_date - timedelta(days=6)) &
                (df['Date_dt'] < target_date)]

    if window.empty:
        continue

    features = {d: {
        'distinct_sunfri': 0,
        'total_sunfri': 0,
        'distinct_fri': 0,
        'total_fri': 0,
        'distinct_wedthu': 0,
        'total_wedthu': 0,
        'distinct_montue': 0,
        'total_montue': 0,
    } for d in range(5)}

    for _, wr in window.iterrows():
        day = wr['Date_dt'].weekday()   # Monday=0, Friday=4
        all_nums = parse_nums(wr[SET_FOR_LIFE_COL]) + parse_nums(wr[others_col])

        for n in all_nums:
            if 1 <= n <= 45:
                d = decade(n)
                features[d]['total_sunfri'] += 1
                if day == 4:
                    features[d]['total_fri'] += 1
                elif day in (2, 3):   # Wed, Thu
                    features[d]['total_wedthu'] += 1
                elif day in (0, 1):   # Mon, Tue
                    features[d]['total_montue'] += 1

        for n in set(all_nums):
            if 1 <= n <= 45:
                d = decade(n)
                features[d]['distinct_sunfri'] += 1
                if day == 4:
                    features[d]['distinct_fri'] += 1
                elif day in (2, 3):
                    features[d]['distinct_wedthu'] += 1
                elif day in (0, 1):
                    features[d]['distinct_montue'] += 1

    present_decs = set(decade(n) for n in winning)
    missing_decs = set(range(5)) - present_decs

    # Compute rank by distinct_sunfri
    sorted_decs = sorted(range(5), key=lambda d: features[d]['distinct_sunfri'], reverse=True)
    rank_of_dec = {d: sorted_decs.index(d)+1 for d in range(5)}

    records.append({
        'date': target_date,
        'features': features,
        'missing': missing_decs,
        'rank': rank_of_dec,
    })

print(f"Total draws analysed: {len(records)}\n")

# ---------- Candidate rules ----------
def rule_lowest_sunfri_distinct(f):
    return min(range(5), key=lambda d: f[d]['distinct_sunfri'])

def rule_lowest_fri_distinct(f):
    return min(range(5), key=lambda d: f[d]['distinct_fri'])

def rule_lowest_wedthu_distinct(f):
    return min(range(5), key=lambda d: f[d]['distinct_wedthu'])

def rule_lowest_montue_distinct(f):
    return min(range(5), key=lambda d: f[d]['distinct_montue'])

def rule_lowest_sunfri_total(f):
    return min(range(5), key=lambda d: f[d]['total_sunfri'])

def rule_lowest_fri_total(f):
    return min(range(5), key=lambda d: f[d]['total_fri'])

def rule_lowest_wedthu_total(f):
    return min(range(5), key=lambda d: f[d]['total_wedthu'])

def rule_lowest_montue_total(f):
    return min(range(5), key=lambda d: f[d]['total_montue'])

def rule_always_40s(f):
    return 4

rules = {
    'Lowest Sun-Fri distinct (Rank 5)': rule_lowest_sunfri_distinct,
    'Lowest Friday distinct': rule_lowest_fri_distinct,
    'Lowest Wed-Thu distinct': rule_lowest_wedthu_distinct,
    'Lowest Mon-Tue distinct': rule_lowest_montue_distinct,
    'Lowest Sun-Fri total freq': rule_lowest_sunfri_total,
    'Lowest Friday total freq': rule_lowest_fri_total,
    'Lowest Wed-Thu total freq': rule_lowest_wedthu_total,
    'Lowest Mon-Tue total freq': rule_lowest_montue_total,
    'Always 40s': rule_always_40s,
}

# ---------- Evaluate rules ----------
print(f"{'Rule':<30} {'All draws accuracy':<20} {'Non-40s missing draws accuracy':<30}")
print("=" * 80)

for name, func in rules.items():
    hits_all = 0
    hits_non40 = 0
    total = len(records)
    non40_draws = 0

    for rec in records:
        killed = func(rec['features'])
        missing = rec['missing']
        if missing:
            if killed in missing:
                hits_all += 1
            # Non-40s: exclude draws where only 40s missing or all missing is 40s
            missing_non40 = missing - {4}
            if missing_non40:
                non40_draws += 1
                if killed in missing_non40:
                    hits_non40 += 1

    all_acc = hits_all / total * 100 if total else 0
    non40_acc = hits_non40 / non40_draws * 100 if non40_draws else 0
    print(f"{name:<30} {all_acc:>6.1f}%             {non40_acc:>6.1f}%  (n={non40_draws})")

# ---------- Show example: for latest draw, why a rule kills X ----------
print("\n" + "=" * 80)
print("Example: Last analysed draw feature breakdown")
print("=" * 80)

if records:
    rec = records[-1]
    print(f"Date: {rec['date'].strftime('%d-%b-%Y')}")
    print(f"Missing decades: {sorted(rec['missing'])}")
    print(f"Rank by Sun-Fri distinct: {rec['rank']}")
    print()

    print(f"{'Dec':<4} {'Rank':<5} {'SunFriD':<8} {'SunFriT':<8} {'FriD':<5} {'FriT':<5} {'WedThuD':<8} {'WedThuT':<8} {'MonTueD':<8} {'MonTueT':<8}")
    for d in range(5):
        f = rec['features'][d]
        print(f"{d*10:<4} {rec['rank'][d]:<5} {f['distinct_sunfri']:<8} {f['total_sunfri']:<8} {f['distinct_fri']:<5} {f['total_fri']:<5} {f['distinct_wedthu']:<8} {f['total_wedthu']:<8} {f['distinct_montue']:<8} {f['total_montue']:<8}")

    print()
    for name, func in rules.items():
        killed = func(rec['features'])
        print(f"Rule '{name}' would kill decade: {killed*10}s")