"""Rank next Others MAIN numbers after a number appears or repeats in source rows.

Run directly after editing NUMBER_TO_CHECK below, or use:
    python number_repeat_next_others.py --number 30
    python number_repeat_next_others.py --single-number 1

SINGLE_OCCURRENCE_NUMBER controls an independent analysis of every row where
that number appears at least once, in either SFL or Others. Rows containing it
more than once also qualify. Set this variable to None to disable that analysis.

Rows are chronological. A trigger is either two or more occurrences across an
entire row (both columns, all brackets), or occurrences on each of two adjacent
rows, regardless of column. The outcome follows the final row of the trigger.
Blank Others rows are skipped. Each outcome date is counted once per analysis.
Only the FIRST bracket of the following Others result is ranked; supplementary
numbers and Powerball are excluded from all outcome tables and exports.
No files are generated unless --output-dir is supplied.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# Allow direct execution beside the CSV, from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lottery_data import CANONICAL_CSV, groups


# Change this number and run the file, or supply --number on the command line.
NUMBER_TO_CHECK = 7
# Independent one-row presence analysis; change this number, or use --single-number.
SINGLE_OCCURRENCE_NUMBER = 12  # Any appearance in SFL or Others, including extras.
TOP_N = 15
EXACTLY_TWO = False  # True excludes windows containing three or more occurrences.
CSV_PATH = CANONICAL_CSV


@dataclass(frozen=True)
class DrawRow:
    date: date
    sfl: str
    others: str

    def all_numbers(self):
        return [n for text in (self.sfl, self.others) for group in groups(text) for n in group]

    def others_main_numbers(self):
        parsed = groups(self.others)
        return parsed[0] if parsed else []


def load_rows(path=CSV_PATH):
    """Read three data fields without relying on the CSV's malformed header."""
    result = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        for line, raw in enumerate(reader, 2):
            if not raw or not any(value.strip() for value in raw):
                continue
            if len(raw) < 2:
                raise ValueError(f"CSV row {line}: expected date and draw columns")
            try:
                dt = datetime.strptime(raw[0].strip(), "%a %d-%b-%Y").date()
            except ValueError as exc:
                raise ValueError(f"CSV row {line}: invalid date {raw[0]!r}") from exc
            result.append(DrawRow(dt, raw[1].strip(), raw[2].strip() if len(raw) > 2 else ""))
    result.sort(key=lambda row: row.date)
    if len({row.date for row in result}) != len(result):
        raise ValueError("CSV contains duplicate dates; resolve these before analyzing adjacent rows")
    return result


def rank_outcomes(data, target_indices):
    """Count only the first Others bracket, once per number per following draw."""
    main = Counter()
    for index in sorted(set(target_indices)):
        main.update(set(data[index].others_main_numbers()))
    total = len(set(target_indices))
    ranked = [{"number": n, "draws": count, "percent": round(100 * count / total, 4)}
              for n, count in sorted(main.items(), key=lambda item: (-item[1], item[0]))]
    return {"following_draws": total, "main": ranked}


def validate_input(data, number):
    if not 1 <= number <= 47:
        raise ValueError("Choose a number from 1 to 47")
    if any(a.date >= b.date for a, b in zip(data, data[1:])):
        raise ValueError("Rows must have distinct dates in chronological order")


def next_others_indices(data):
    # Precompute the first non-empty Others result strictly AFTER every row.
    next_others = [None] * len(data)
    next_index = None
    for index in range(len(data) - 1, -1, -1):
        next_others[index] = next_index
        if data[index].others_main_numbers():
            next_index = index
    return next_others


def summarize_events(data, events):
    targets = {event["next_index"] for event in events if event["next_index"] is not None}
    return {"trigger_windows": len(events),
            "pending_windows": sum(event["next_index"] is None for event in events),
            "pending_end_dates": sorted({data[event["end_index"]].date.isoformat()
                                         for event in events if event["next_index"] is None}),
            **rank_outcomes(data, targets)}


def analyze_single_occurrence(data, number):
    """Find every row containing the number in either column, at least once."""
    validate_input(data, number)
    following = next_others_indices(data)
    events = []
    for index, row in enumerate(data):
        count = row.all_numbers().count(number)
        if count:
            events.append({"type": "any_occurrence", "start_index": index,
                           "end_index": index, "occurrences": count,
                           "next_index": following[index]})
    return {"number": number, "events": events,
            "summaries": {"any_occurrence": summarize_events(data, events)}}


def analyze(data, number, exactly_two=False):
    validate_input(data, number)
    occurrences = [row.all_numbers().count(number) for row in data]
    next_others = next_others_indices(data)
    events = []
    for end, count in enumerate(occurrences):
        if count == 2 or (count > 2 and not exactly_two):
            events.append({"type": "single_row", "start_index": end,
                           "end_index": end, "occurrences": count,
                           "next_index": next_others[end]})
        if end and occurrences[end - 1] and count:
            window_count = occurrences[end - 1] + count
            if exactly_two and window_count != 2:
                continue
            events.append({"type": "consecutive_rows", "start_index": end - 1,
                           "end_index": end, "occurrences": window_count,
                           "next_index": next_others[end]})
    summaries = {}
    for kind in ("single_row", "consecutive_rows", "combined"):
        selected = events if kind == "combined" else [event for event in events if event["type"] == kind]
        summaries[kind] = summarize_events(data, selected)
    return {"number": number, "exactly_two": exactly_two,
            "events": events, "summaries": summaries}


def print_table(title, ranked, denominator, top):
    print(f"\n{title} ({denominator} distinct following draws)")
    print(f"{'Rank':>4} {'Number':>7} {'Draws':>7} {'Percent':>9}")
    if not ranked:
        print("No following results available.")
    previous = None
    rank = 0
    for position, row in enumerate(ranked[:top], 1):
        if row["draws"] != previous:
            rank = position
        previous = row["draws"]
        print(f"{rank:>4} {row['number']:>7} {row['draws']:>7} {row['percent']:>8.2f}%")


def save_csv(directory, data, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "rankings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pattern", "outcome", "following_draws", "number", "draws", "percent"])
        writer.writeheader()
        for kind, summary in result["summaries"].items():
            for row in summary["main"]:
                writer.writerow({"pattern": kind, "outcome": "main",
                                 "following_draws": summary["following_draws"], **row})
    with (directory / "matching_windows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["type", "start_date", "end_date", "occurrences", "start_sfl", "start_others", "end_sfl", "end_others", "next_draw", "next_others_main", "counted_unique_outcome"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        seen = set()
        for event in result["events"]:
            first = data[event["start_index"]]
            last = data[event["end_index"]]
            target = event["next_index"]
            counted = target is not None and target not in seen
            if target is not None:
                seen.add(target)
            writer.writerow({"type": event["type"], "start_date": first.date, "end_date": last.date,
                             "occurrences": event["occurrences"], "start_sfl": first.sfl, "start_others": first.others,
                             "end_sfl": last.sfl, "end_others": last.others,
                             "next_draw": data[target].date if target is not None else "PENDING",
                             "next_others_main": str(data[target].others_main_numbers()) if target is not None else "",
                             "counted_unique_outcome": counted})


def print_recent_windows(data, events, title="Most recent qualifying windows"):
    print(f"\n{title}:")
    if not events:
        print("  No matching rows.")
    for event in events[-10:]:
        first = data[event["start_index"]].date
        last = data[event["end_index"]].date
        target = event["next_index"]
        dates = str(first) if first == last else f"{first} + {last}"
        following = f"{data[target].date}: {data[target].others_main_numbers()}" if target is not None else "PENDING -- no later Others result in CSV"
        print(f"  {dates} | {event['occurrences']} occurrences | next: {following}")


def print_single_analysis(data, result, top):
    print("\n" + "=" * 85)
    print(f"SINGLE OCCURRENCE / ANY APPEARANCE ANALYSIS: NUMBER {result['number']}")
    print("A row qualifies whenever this number appears at least once in SFL OR Others.")
    print("Main and supplementary/Powerball occurrences qualify, including rows with multiple occurrences.")
    summary = result["summaries"]["any_occurrence"]
    print(f"Matching rows: {summary['trigger_windows']} | "
          f"Distinct following Others draws: {summary['following_draws']} | "
          f"Pending rows: {summary['pending_windows']}")
    print("Each following draw is counted once, even when several source rows lead to it.")
    print_table("ANY APPEARANCE: MAIN NUMBERS ONLY", summary["main"], summary["following_draws"], top)
    print_recent_windows(data, result["events"], "Most recent rows containing the number")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--number", type=int, default=NUMBER_TO_CHECK, help="number to look for (default: NUMBER_TO_CHECK in this file)")
    parser.add_argument("--single-number", type=int, default=SINGLE_OCCURRENCE_NUMBER,
                        help="independent number whose appearance on any row triggers a next-Others analysis")
    parser.add_argument("--top", type=int, default=TOP_N, help="how many ranked numbers to print")
    parser.add_argument("--exactly-two", action="store_true", default=EXACTLY_TWO,
                        help="require exactly two occurrences, excluding three or more")
    parser.add_argument("--output-dir", type=Path, help="optionally save rankings and matching windows as CSV files")
    args = parser.parse_args(argv)
    if not 1 <= args.number <= 47:
        parser.error("--number must be from 1 to 47")
    if args.single_number is not None and not 1 <= args.single_number <= 47:
        parser.error("--single-number must be from 1 to 47")
    if args.top < 1:
        parser.error("--top must be positive")
    data = load_rows()
    result = analyze(data, args.number, args.exactly_two)
    print(f"Number: {args.number} | {'Exactly' if args.exactly_two else 'At least'} two occurrences")
    print(f"Source: {CSV_PATH}")
    if data:
        print(f"Rows: {len(data)} | {data[0].date} to {data[-1].date}")
    print("Trigger checks use the entire SFL and Others row, including supplementary/Powerball brackets.")
    print("Two-row matches require the number on EACH adjacent row; columns may be the same or different.")
    print("Next draw = first later non-empty Others result, after the whole trigger window.")
    print("Only its MAIN numbers (the first Others bracket) are counted and displayed as outcomes.")
    print("Overlapping matches never count the same following draw twice in a ranking.")
    for kind, title in (("single_row", "Single-row repeats"), ("consecutive_rows", "Consecutive-row repeats"), ("combined", "Combined")):
        summary = result["summaries"][kind]
        print(f"\n{title}: {summary['trigger_windows']} windows; "
              f"{summary['following_draws']} distinct following draws; {summary['pending_windows']} pending windows")
        leaders = summary["main"]
        if leaders:
            best = leaders[0]["draws"]
            tied = ", ".join(str(row["number"]) for row in leaders if row["draws"] == best)
            print(f"  Main numbers only: {tied} -- {best}/{summary['following_draws']} draws")
    combined = result["summaries"]["combined"]
    print_table("COMBINED: MAIN NUMBERS ONLY", combined["main"], combined["following_draws"], args.top)
    print("Rankings and percentages count a main number once per distinct following draw; games are pooled.")
    print_recent_windows(data, result["events"])
    if args.single_number is not None:
        single_result = analyze_single_occurrence(data, args.single_number)
        result["single_occurrence_analysis"] = single_result
        print_single_analysis(data, single_result, args.top)
    if args.output_dir:
        save_csv(args.output_dir, data, result)
        if args.single_number is not None:
            save_csv(args.output_dir / "single_occurrence", data, single_result)
        print(f"\nSaved: {args.output_dir.resolve()}")
    return result


if __name__ == "__main__":
    main()
