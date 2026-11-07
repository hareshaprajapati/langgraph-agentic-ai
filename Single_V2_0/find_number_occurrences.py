"""Find Saturday draws containing all numbers in a supplied list.

Edit NUMBERS below, or run:
    python Single_V2_0/find_number_occurrences.py --numbers 3 23 39
"""

import argparse
import csv
from pathlib import Path


NUMBERS = [7,17,22]
DEFAULT_CSV = Path(__file__).resolve().with_name("Saturday_data.csv")


def find_occurrences(numbers, csv_path=DEFAULT_CSV):
    """Return matching Date/Main rows; all supplied numbers must occur together."""
    numbers = list(numbers)
    if not numbers or any(type(number) is not int or number < 1 for number in numbers):
        raise ValueError("Provide at least one positive integer.")
    if len(set(numbers)) != len(numbers):
        raise ValueError("Provide each number only once.")

    requested = set(numbers)
    matches = []
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not {"Date", "Main"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain Date and Main columns.")
        for row in reader:
            try:
                draw_numbers = {int(value.strip()) for value in row["Main"].split(",")}
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Invalid Main numbers at CSV line {reader.line_num}.") from exc
            if requested.issubset(draw_numbers):
                matches.append({"Date": row["Date"], "Main": row["Main"]})
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numbers", nargs="+", type=int, default=NUMBERS,
        help="Numbers to find together (default: %(default)s).",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV file to search.")
    args = parser.parse_args()
    try:
        matches = find_occurrences(args.numbers, args.csv)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))

    print(f"Numbers: {args.numbers}")
    print(f"Appeared together: {len(matches)} time(s)")
    if matches:
        print("\nDate                 Main numbers")
        for match in matches:
            print(f"{match['Date']:<20} {match['Main']}")
    else:
        print("No matching dates found.")


if __name__ == "__main__":
    main()
