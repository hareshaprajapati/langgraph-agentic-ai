# Config runner for FINAL LOTTERY STRATEGY v1.0 (NO REPEAT LOGIC)

from Siko_Core_Single import generate_tickets, print_report


def main():
    game_name = "Oz Lotto"
    target_date = "2025-12-30"
    csv_path = "oz_lotto.csv"
    ticket_count = 5

    meta, tickets = generate_tickets(csv_path, target_date, ticket_count=ticket_count, game_name=game_name)
    print_report(meta, tickets)


if __name__ == "__main__":
    main()
