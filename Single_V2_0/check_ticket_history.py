"""Report tickets sharing at least MIN_MATCHES numbers with a recent Saturday draw.

Run with defaults:
    python Single_V2_0/check_ticket_history.py
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_TICKETS = DATA_DIR / "SikoSat_manual_tickets.csv"
DEFAULT_DRAWS = DATA_DIR / "Saturday_data.csv"

# Change to 4, 5, etc. to require that many matching numbers in a single draw.
MIN_MATCHES = 4


def read_rows(path, required_columns):
    with Path(path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not set(required_columns).issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {', '.join(required_columns)}.")
        return list(reader)


def parse_numbers(value):
    try:
        numbers = [int(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid number list: {value!r}.") from exc
    if any(number < 1 for number in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError(f"Numbers must be positive and unique: {value!r}.")
    return set(numbers)


def check_tickets(tickets_path=DEFAULT_TICKETS, draws_path=DEFAULT_DRAWS,
                  draw_limit=52, min_matches=MIN_MATCHES):
    """Return each ticket's matches and the draws checked, newest first.

    Compare each ticket with each draw separately; number order is ignored.
    The draw window is relative to the latest date in the draws CSV.
    """
    if draw_limit < 1 or min_matches < 1:
        raise ValueError("Draw count and minimum matches must be positive.")
    tickets = read_rows(tickets_path, ["Ticket", "Numbers"])
    draws = read_rows(draws_path, ["Date", "Main"])
    if not draws:
        raise ValueError("The draws CSV contains no draws.")
    try:
        draws.sort(key=lambda row: datetime.strptime(row["Date"], "%a %d-%b-%Y"),
                   reverse=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Draw dates must use the format 'Sat 31-Oct-2026'.") from exc
    recent_draws = draws[:draw_limit]
    parsed_draws = [(draw, parse_numbers(draw["Main"])) for draw in recent_draws]

    results = []
    for ticket in tickets:
        ticket_numbers = parse_numbers(ticket["Numbers"])
        matches = []
        for draw, draw_numbers in parsed_draws:
            common = sorted(ticket_numbers & draw_numbers)
            if len(common) >= min_matches:
                matches.append({"date": draw["Date"], "draw_numbers": sorted(draw_numbers),
                                "matching_numbers": common, "match_count": len(common)})
        results.append({"ticket": ticket, "matches": matches})
    return results, recent_draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS,
                        help="Ticket CSV (default: SikoSat_manual_tickets.csv beside this script).")
    parser.add_argument("--draws-csv", type=Path, default=DEFAULT_DRAWS,
                        help="Draw CSV (default: Saturday_data.csv beside this script).")
    parser.add_argument("--draws", type=int, default=52, help="Latest draws to check (default: 52).")
    parser.add_argument("--min-matches", type=int, default=MIN_MATCHES,
                        help="Minimum numbers shared with one draw (default: %(default)s).")
    args = parser.parse_args()
    try:
        results, draws = check_tickets(args.tickets, args.draws_csv, args.draws, args.min_matches)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    matching_tickets = [result for result in results if result["matches"]]
    total_matches = sum(len(result["matches"]) for result in matching_tickets)
    print(f"Tickets checked: {len(results)}")
    print(f"Latest draws checked: {len(draws)} (requested: {args.draws})")
    print(f"Draw date range: {draws[-1]['Date']} to {draws[0]['Date']}")
    print(f"Tickets with {args.min_matches}+ numbers in a single draw: {len(matching_tickets)}")
    print(f"Total ticket/draw matches: {total_matches}")

    if not matching_tickets:
        print("\nNo matching tickets found.")
    for result in matching_tickets:
        ticket = result["ticket"]
        print(f"\nTicket {ticket['Ticket']}: {ticket['Numbers']}")
        for match in result["matches"]:
            common = ", ".join(map(str, match["matching_numbers"]))
            draw_numbers = ", ".join(map(str, match["draw_numbers"]))
            print(f"  {match['date']} | {match['match_count']} matches: {common}"
                  f" | Draw: {draw_numbers}")


if __name__ == "__main__":
    main()
