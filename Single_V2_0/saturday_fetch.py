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

def fetch_page(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def parse_year_archive(year: int):
    """
    Fetch and parse a Saturday Lotto archive page.
    Returns list of tuples: (datetime, main_numbers_list)
    """
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

            # Saturday Lotto: 6 main numbers + 2 supplementary = 8 numbers
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

# ---------- Fetch all years ----------
current_year = datetime.now().year
all_draws = []

print(f"Fetching Saturday Lotto history from {START_YEAR} to {current_year} using archive pages...\n")

for year in range(current_year, START_YEAR - 1, -1):
    print(f"Fetching {year}...")
    year_draws = parse_year_archive(year)

    if year_draws:
        print(f"  Found {len(year_draws)} draws")
        all_draws.extend(year_draws)
    else:
        print(f"  No draws found for {year}")

    time.sleep(0.5)

# ---------- Sort and deduplicate by date ----------
all_draws.sort(key=lambda x: x[0])

deduped = {}
for dt, main in all_draws:
    if dt not in deduped:
        deduped[dt] = main

all_draws = sorted(deduped.items(), key=lambda x: x[0])

print(f"\nTotal unique Saturday Lotto draws fetched: {len(all_draws)}")

if not all_draws:
    print("No draws found. Exiting.")
    exit(0)

print(f"Date range: {all_draws[0][0].strftime('%d-%b-%Y')} to "
      f"{all_draws[-1][0].strftime('%d-%b-%Y')}")

# ---------- Save to CSV ----------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "Main"])

    for dt, main in all_draws:
        date_str = dt.strftime("%a %d-%b-%Y")
        main_str = ",".join(str(n) for n in main)
        writer.writerow([date_str, main_str])

print(f"\n✅ Saved {len(all_draws)} draws to: {os.path.abspath(OUTPUT_CSV)}")