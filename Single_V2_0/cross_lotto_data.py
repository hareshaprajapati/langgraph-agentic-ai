import sys
import os
from datetime import datetime

import re
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from bs4 import BeautifulSoup

# ---------- CONFIG ----------
DAYS_BACK = 365 * 5
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

def _results_base_from_past_url(past_url: str) -> str:
    # "https://au.lottonumbers.com/saturday-lotto/past-results"
    # -> "https://au.lottonumbers.com/saturday-lotto/results"
    return past_url.replace("/past-results", "/results")

def archive_urls_for_range(past_url: str, start: datetime, end: datetime) -> list[str]:
    base = _results_base_from_past_url(past_url)
    years = range(start.year, end.year + 1)
    return [f"{base}/{y}-archive" for y in years]

def parse_date_label(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%a %d-%b-%Y")

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
            latest_date = parse_date_label(date_str)
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


# ---------- MAIN ----------
def main():
    end = datetime.now()
    default_start = end - timedelta(days=DAYS_BACK)

    main_path = os.path.join(".", "cross_lotto_data.csv")
    sfl_path = os.path.join(".", "cross_lotto_data_set_for_life.csv")
    others_path = os.path.join(".", "cross_lotto_data_others.csv")

    main_header_default = "Date,Set for Life (incl supp),Others (incl supp)"
    sfl_header_default = "Date,Set for Life (incl supp)"
    others_header_default = "Date,Others (incl supp)"

    main_header, main_existing, main_latest = read_existing_csv(main_path)
    sfl_header, sfl_existing, sfl_latest = read_existing_csv(sfl_path)
    others_header, others_existing, others_latest = read_existing_csv(others_path)

    main_header = main_header or main_header_default
    sfl_header = sfl_header or sfl_header_default
    others_header = others_header or others_header_default

    main_start = (main_latest + timedelta(days=1)) if main_latest else default_start
    sfl_start = (sfl_latest + timedelta(days=1)) if sfl_latest else default_start
    others_start = (others_latest + timedelta(days=1)) if others_latest else default_start

    global_start = min(main_start, sfl_start, others_start)

    if global_start.date() > end.date():
        print("No new dates to fetch.")
        return

    # Map date -> set_for_life string
    set_for_life_by_date = {}
    set_for_life_only_by_date = {}

    # Map date -> list of other draw strings (we'll join with " | ")
    others_by_date = defaultdict(list)
    others_only_by_date = defaultdict(list)

    # 1) Set for Life (daily)
    # sfl = scrape_past_results(PAGES["set_for_life"])
    # for dt, main_nums, supp_nums in sfl:
    #     if within_range(dt, start, end):
    #         set_for_life_by_date[dt.date()] = f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]"
    #
    urls = [PAGES["set_for_life"]] + archive_urls_for_range(PAGES["set_for_life"], global_start, end)

    seen = set()  # (date, main_tuple, supp_tuple)
    for url in urls:
        rows = scrape_past_results(url)
        for dt, main_nums, supp_nums in rows:
            k = (dt.date(), tuple(main_nums), tuple(supp_nums))
            if k in seen:
                continue
            seen.add(k)

            if within_range(dt, global_start, end):
                sfl_val = f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]" if supp_nums else f"{normalize_nums(main_nums)}"
                set_for_life_by_date[dt.date()] = sfl_val
                set_for_life_only_by_date[dt.date()] = sfl_val



    # 2) Others
    for key in ["weekday_windfall", "oz_lotto", "powerball", "saturday_lotto"]:
        urls = [PAGES[key]] + archive_urls_for_range(PAGES[key], global_start, end)

        seen = set()  # (date, main_tuple, supp_tuple)
        for url in urls:
            rows = scrape_past_results(url)
            for dt, main_nums, supp_nums in rows:
                k = (dt.date(), tuple(main_nums), tuple(supp_nums))
                if k in seen:
                    continue
                seen.add(k)

                if within_range(dt, global_start, end):
                    if supp_nums:
                        others_only_by_date[dt.date()].append(
                            f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]")
                        others_by_date[dt.date()].append(
                            f"[{normalize_nums(main_nums)}], [{normalize_nums(supp_nums)}]")
                    else:
                        others_only_by_date[dt.date()].append(f"{normalize_nums(main_nums)}")
                        others_by_date[dt.date()].append(f"{normalize_nums(main_nums)}")

    def _q_csv(s: str) -> str:
        if "," in s or "|" in s or "+" in s:
            return f"\"{s}\""
        return s

    def _build_new_lines(start_dt: datetime, include_sfl: bool, include_others: bool) -> list[str]:
        lines = []
        cur = end.date()
        start_date = start_dt.date()
        if start_date > cur:
            return lines

        while cur >= start_date:
            dt = datetime.combine(cur, datetime.min.time())
            date_label = fmt_date_day(dt)

            sfl_str = set_for_life_by_date.get(cur, "")
            oth_list = others_by_date.get(cur, [])
            oth_str = " | ".join(oth_list)

            parts = [date_label]
            if include_sfl:
                parts.append(_q_csv(sfl_str))
            if include_others:
                parts.append(_q_csv(oth_str))

            lines.append(",".join(parts))
            cur -= timedelta(days=1)

        return lines

    def _build_new_lines_only(start_dt: datetime, which: str) -> list[str]:
        lines = []
        cur = end.date()
        start_date = start_dt.date()
        if start_date > cur:
            return lines

        while cur >= start_date:
            dt = datetime.combine(cur, datetime.min.time())
            date_label = fmt_date_day(dt)

            if which == "sfl":
                val = set_for_life_only_by_date.get(cur, "")
            else:
                val = " | ".join(others_only_by_date.get(cur, []))

            lines.append(f"{date_label},{_q_csv(val)}")
            cur -= timedelta(days=1)

        return lines

    main_new = _build_new_lines(main_start, include_sfl=True, include_others=True)
    sfl_new = _build_new_lines_only(sfl_start, "sfl")
    others_new = _build_new_lines_only(others_start, "others")

    write_csv(main_path, main_header, main_new, main_existing)
    write_csv(sfl_path, sfl_header, sfl_new, sfl_existing)
    write_csv(others_path, others_header, others_new, others_existing)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
