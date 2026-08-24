import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import csv
import os
import time

# ---------- CONFIG ----------
START_YEAR = 1986
TIMEOUT = 30
OUTPUT_CSV = "Saturday_data.csv"

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

DATE_LINE_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\s+\w+\s+\d{4}$"
)
DRAW_RE = re.compile(r"^Draw\s+[\d,]+$")

# ---------- Helpers ----------
def parse_date_line(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%A %d %B %Y")

def parse_csv_date(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%a %d-%b-%Y")

def fetch_page(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def parse_year_archive(year: int, latest_date=None):
    url = f"https://au.lottonumbers.com/saturday-lotto/results/{year}-archive"
    try:
        html = fetch_page(url)
    except Exception as e:
        print(f"  Skipping {year}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    lines = [t.strip() for t in soup.stripped_strings if t.strip()]

    results = []
    i = 0
    while i < len(lines):
        if DRAW_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and not DATE_LINE_RE.match(lines[j]):
                j += 1
            if j >= len(lines):
                i += 1
                continue

            dt = parse_date_line(lines[j])
            # Skip draws already in CSV
            if latest_date is not None and dt <= latest_date:
                i = j + 1
                continue

            nums = []
            k = j + 1
            while k < len(lines) and len(nums) < 8:
                if lines[k].isdigit():
                    nums.append(int(lines[k]))
                k += 1

            if len(nums) >= 6:
                main = nums[:6]
                results.append((dt, main))

            i = k
        else:
            i += 1

    return results

# ---------- Read existing CSV ----------
existing_draws = {}
latest_date = None

if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                date_str = row[0]
                main_str = row[1]
                try:
                    dt = parse_csv_date(date_str)
                    main = [int(x) for x in main_str.split(",") if x]
                    existing_draws[dt] = main
                    if latest_date is None or dt > latest_date:
                        latest_date = dt
                except Exception:
                    pass

print(f"Loaded {len(existing_draws)} existing draws. Latest date: {latest_date.strftime('%d-%b-%Y') if latest_date else 'None'}")

# ---------- Fetch only newer draws ----------
current_year = datetime.now().year
start_year = START_YEAR if latest_date is None else latest_date.year

print(f"Fetching from year {start_year} to {current_year} (only new draws)...\n")

new_draws = []
for year in range(current_year, start_year - 1, -1):
    print(f"Fetching {year}...")
    year_draws = parse_year_archive(year, latest_date)
    if year_draws:
        print(f"  Found {len(year_draws)} new draws")
        new_draws.extend(year_draws)
    else:
        print(f"  No new draws found for {year}")
    time.sleep(0.5)

# ---------- Merge, deduplicate, sort descending ----------
all_draws = dict(existing_draws)
for dt, main in new_draws:
    if dt not in all_draws:
        all_draws[dt] = main

all_draws_sorted = sorted(all_draws.items(), key=lambda x: x[0], reverse=True)

print(f"\nTotal unique draws after update: {len(all_draws_sorted)}")
if not all_draws_sorted:
    print("No draws found. Exiting.")
    exit(0)

print(f"Date range: {all_draws_sorted[-1][0].strftime('%d-%b-%Y')} (oldest) to "
      f"{all_draws_sorted[0][0].strftime('%d-%b-%Y')} (newest)")

# ---------- Write CSV (newest first) ----------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "Main"])
    for dt, main in all_draws_sorted:
        date_str = dt.strftime("%a %d-%b-%Y")
        main_str = ",".join(str(n) for n in main)
        writer.writerow([date_str, main_str])

print(f"\n✅ Saved {len(all_draws_sorted)} draws to: {os.path.abspath(OUTPUT_CSV)}")