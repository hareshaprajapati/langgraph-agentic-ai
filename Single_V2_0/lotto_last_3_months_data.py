import sys
import os
from datetime import datetime

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

# LOG_DIR = ""
# os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(
    ".",
    f"lotto_last_3_months.log"   # single growing log file
)

log_file = open(log_file_path, "w", buffering=1, encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from bs4 import BeautifulSoup

# ---------- CONFIG ----------
DAYS_BACK = 150  # ~ last 3 months
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

# ---------- HELPERS ----------
DATE_LINE_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+\d{1,2}\s+\w+\s+\d{4}$")
DRAW_RE = re.compile(r"^Draw\s+[\d,]+$")

def parse_date_line(s: str) -> datetime:
    # Example: "Tuesday 13 January 2026"
    return datetime.strptime(s.strip(), "%A %d %B %Y")

def fmt_date_day(d: datetime) -> str:
    # "Tue 13-Jan-2026"
    return d.strftime("%a %d-%b-%Y")

def normalize_nums(nums):
    # join as "1,2,3,4,5,6 + 7,8" when we know main+supp split
    return ", ".join(str(n) for n in nums)

def scrape_past_results(url: str):
    """
    Returns list of tuples:
      (draw_date_dt, main_numbers_list, supp_numbers_list)
    This scraper is tolerant: it looks for "Draw X" then the date line then bullet numbers.
    """
    r = requests.get(url, headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # The site renders a text-like sequence; easiest is to walk visible text lines.
    # We'll take all stripped strings (in document order).
    lines = [t.strip() for t in soup.stripped_strings if t.strip()]

    results = []
    i = 0
    while i < len(lines):
        if DRAW_RE.match(lines[i]):
            # move forward to find date line
            j = i + 1
            while j < len(lines) and not DATE_LINE_RE.match(lines[j]):
                j += 1
            if j >= len(lines):
                i += 1
                continue

            dt = parse_date_line(lines[j])

            # After date line, numbers appear as individual tokens (e.g., "3", "5", ...)
            # For these games:
            # - Set for Life: 7 main + 2 supp
            # - Weekday Windfall: 6 main + 2 supp
            # - Oz Lotto: 7 main + 3 supp
            # - Powerball: 7 main + 1 powerball (treat as "supp" in output)
            # - Saturday Lotto: 6 main + 2 supp
            #
            # We'll infer counts based on URL.
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

            # Collect next numeric tokens
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
    # inclusive
    return start <= dt <= end

# ---------- MAIN ----------
def main():
    end = datetime.now()
    start = end - timedelta(days=DAYS_BACK)

    # Map date -> set_for_life string
    set_for_life_by_date = {}

    # Map date -> list of other draw strings (we'll join with " | ")
    others_by_date = defaultdict(list)

    # 1) Set for Life (daily)
    sfl = scrape_past_results(PAGES["set_for_life"])
    for dt, main_nums, supp_nums in sfl:
        if within_range(dt, start, end):
            set_for_life_by_date[dt.date()] = f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]"

    # 2) Others
    for key in ["weekday_windfall", "oz_lotto", "powerball", "saturday_lotto"]:
        rows = scrape_past_results(PAGES[key])
        for dt, main_nums, supp_nums in rows:
            if within_range(dt, start, end):
                # "just numbers" — no draw name
                # but we keep separate draws on same day via " | "
                if supp_nums:
                    others_by_date[dt.date()].append(f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]")
                else:
                    others_by_date[dt.date()].append(f"{normalize_nums(main_nums)}")

    # 3) Build full date series (daily)
    cur = end.date()
    start_date = start.date()

    # CSV header
    print("Date,Set for Life (incl supp),Others (incl supp)")
    while cur >= start_date:
        dt = datetime.combine(cur, datetime.min.time())
        date_label = fmt_date_day(dt)

        sfl_str = set_for_life_by_date.get(cur, "")
        oth_list = others_by_date.get(cur, [])
        oth_str = " | ".join(oth_list)

        def q(s):
            if "," in s or "|" in s or "+" in s:
                return f"\"{s}\""
            return s

        print(f"{date_label},{q(sfl_str)},{q(oth_str)}")
        cur -= timedelta(days=1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
