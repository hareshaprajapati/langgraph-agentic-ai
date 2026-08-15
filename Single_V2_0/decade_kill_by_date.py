import pandas as pd
import re
from collections import Counter
from datetime import timedelta

CSV = "cross_lotto_data_backup.csv"

def get_others_col(df):
    for col in df.columns:
        if "Others" in col and (
            "Weekday windfall" in col or "OZ Lotto" in col or
            "Powerball" in col or "Saturday Lotto" in col
        ):
            return col
    for col in df.columns:
        if "Others" in col:
            return col
    raise KeyError("Others column not found")

def parse_nums(s):
    if pd.isna(s) or s == "":
        return []
    return [int(n) for n in re.findall(r"\d+", s)]

def predict_kills(target_date_str):
    df = pd.read_csv(CSV)
    df["Date_dt"] = pd.to_datetime(df["Date"], format="%a %d-%b-%Y")
    df = df.sort_values("Date_dt")

    target = pd.to_datetime(target_date_str)
    others_col = get_others_col(df)

    # Sun-Fri window: 6 days before target Saturday
    window = df[
        (df["Date_dt"] >= target - timedelta(days=6)) &
        (df["Date_dt"] < target)
    ]

    if window.empty:
        print("No Sun-Fri data available.")
        return None, None

    m_counts = Counter()
    for _, row in window.iterrows():
        m_counts.update(parse_nums(row["Set for Life (incl supp)"]))
        m_counts.update(parse_nums(row[others_col]))

    # Keep only 1..45
    m_counts = Counter({n: c for n, c in m_counts.items() if 1 <= n <= 45})

    # Count distinct numbers per decade
    dec_vols = Counter({d: 0 for d in range(5)})
    for n in m_counts.keys():
        dec_vols[n // 10] += 1

    sorted_decs = sorted(range(5), key=lambda d: dec_vols[d], reverse=True)

    rank3 = sorted_decs[2]      # 3rd highest
    lowest = sorted_decs[-1]    # lowest, usually 40s

    print(f"Sun-Fri distinct numbers per decade: {dict(dec_vols)}")
    print(f"Sorted decades (highest to lowest distinct): {sorted_decs}")

    print(f"Kill Rank 3 which is {rank3 * 10}s")
    print(f"Kill Rank 5 which is {lowest * 10}s")
    print(f"Kill Fixed 40s")

    print("\nSuggested kill allocation for 50 tickets:")
    print(f"  Rank 3 kill: 20 tickets")
    print(f"  Lowest kill: 12 tickets")
    print(f"  Fixed 40s kill: 8 tickets")
    print(f"  No kill: 10 tickets")

    return rank3, lowest

if __name__ == "__main__":
    predict_kills("2026-08-15")