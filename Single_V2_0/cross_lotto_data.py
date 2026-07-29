import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
DAYS_BACK = 365 * 5          # fallback if no CSV exists
TIMEOUT = 30

PAGES = {
    "set_for_life": "https://au.lottonumbers.com/set-for-life/past-results",
    "weekday_windfall": "https://au.lottonumbers.com/weekday-windfall/past-results",
    "oz_lotto": "https://au.lottonumbers.com/oz-lotto/past-results",
    "powerball": "https://au.lottonumbers.com/powerball/past-results",
    "saturday_lotto": "https://au.lottonumbers.com/saturday-lotto/past-results",
}

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# ---------- HELPERS (unchanged) ----------
DATE_LINE_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\s+\w+\s+\d{4}$")
DRAW_RE = re.compile(r"^Draw\s+[\d,]+$")

def parse_date_line(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%A %d %B %Y")

def fmt_date_day(d: datetime) -> str:
    return d.strftime("%a %d-%b-%Y")

def normalize_nums(nums):
    return ", ".join(str(n) for n in nums)

def scrape_past_results(url: str):
    """
    Returns list of tuples: (draw_date_dt, main_numbers_list, supp_numbers_list)
    """
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

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

            if "set-for-life" in url:
                main_ct, supp_ct = 7, 2
            elif "weekday-windfall" in url:
                main_ct, supp_ct = 6, 2
            elif "oz-lotto" in url:
                main_ct, supp_ct = 7, 3
            elif "powerball" in url:
                main_ct, supp_ct = 7, 1
            elif "saturday-lotto" in url:
                main_ct, supp_ct = 6, 2
            else:
                main_ct, supp_ct = 0, 0

            nums = []
            k = j + 1
            while k < len(lines) and len(nums) < (main_ct + supp_ct):
                if lines[k].isdigit():
                    nums.append(int(lines[k]))
                k += 1

            if len(nums) == (main_ct + supp_ct):
                main = nums[:main_ct]
                supp = nums[main_ct:]
                results.append((dt, main, supp))

            i = k
        else:
            i += 1

    return results

def within_range(dt: datetime, start: datetime, end: datetime) -> bool:
    return start <= dt <= end

def read_existing_csv(path: str) -> tuple[str | None, list[str], datetime | None]:
    if not os.path.exists(path):
        return None, [], None

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        return None, [], None

    header = lines[0]
    data_lines = [ln for ln in lines[1:] if ln.strip()]

    latest_date = None
    for ln in data_lines:
        date_str = ln.split(",", 1)[0].strip()
        try:
            latest_date = datetime.strptime(date_str, "%a %d-%b-%Y")
            break
        except Exception:
            continue

    return header, data_lines, latest_date

def write_csv(path: str, header: str, new_lines: list[str], existing_lines: list[str]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for ln in new_lines:
            f.write(ln + "\n")
        for ln in existing_lines:
            f.write(ln + "\n")

# ---------- MAIN (OPTIMIZED) ----------
def main():
    end = datetime.now()
    default_start = end - timedelta(days=DAYS_BACK)

    backup_path = os.path.join(".", "cross_lotto_data_backup.csv")
    backup_header_default = "Date,Set for Life (incl supp),Others(e.g. Weekday windfall, OZ Lotto, Powerball, Saturday Lotto) (incl supp)"

    backup_header, backup_existing, backup_latest = read_existing_csv(backup_path)
    backup_header = backup_header or backup_header_default

    if backup_latest:
        start_date = backup_latest + timedelta(days=1)
    else:
        start_date = default_start

    if start_date.date() > end.date():
        print("No new dates to fetch.")
        return

    # Fetch only the main past-results page for each game – they contain plenty of history
    urls = {name: PAGES[name] for name in PAGES}

    # Fetch all pages in parallel
    results_by_game = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_game = {executor.submit(scrape_past_results, url): name for name, url in urls.items()}
        for future in as_completed(future_to_game):
            name = future_to_game[future]
            try:
                results_by_game[name] = future.result()
            except Exception as e:
                print(f"Error fetching {name}: {e}", file=sys.stderr)
                results_by_game[name] = []

    # Process results
    set_for_life_by_date = {}
    others_by_date = defaultdict(list)

    # Set for Life
    for dt, main_nums, supp_nums in results_by_game.get("set_for_life", []):
        if within_range(dt, start_date, end):
            sfl_val = f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]" if supp_nums else normalize_nums(main_nums)
            set_for_life_by_date[dt.date()] = sfl_val

    # Other games
    for game in ["weekday_windfall", "oz_lotto", "powerball", "saturday_lotto"]:
        for dt, main_nums, supp_nums in results_by_game.get(game, []):
            if within_range(dt, start_date, end):
                if supp_nums:
                    entry = f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]"
                else:
                    entry = normalize_nums(main_nums)
                others_by_date[dt.date()].append(entry)

    # Build new CSV lines
    def _q_csv(s: str) -> str:
        if "," in s or "|" in s or "+" in s:
            return f"\"{s}\""
        return s

    new_lines = []
    cur = end.date()
    while cur >= start_date.date():
        dt = datetime.combine(cur, datetime.min.time())
        date_label = fmt_date_day(dt)
        sfl_str = set_for_life_by_date.get(cur, "")
        oth_list = others_by_date.get(cur, [])
        oth_str = " | ".join(oth_list)
        parts = [date_label, _q_csv(sfl_str), _q_csv(oth_str)]
        new_lines.append(",".join(parts))
        cur -= timedelta(days=1)

    write_csv(backup_path, backup_header, new_lines, backup_existing)
    print(f"Added {len(new_lines)} new rows to {backup_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)