"""Walk-forward Oz Lotto tickets from Monday number-following frequencies.

Run beside the canonical CSV:
    python oz_lotto_monday_followups_backtest.py
    python oz_lotto_monday_followups_backtest.py --last 10

Every target Tuesday is hidden during ticket construction. Source numbers come
from the preceding Monday's entire row; only future Others MAIN numbers are
predicted. Both native all-days follow-up tables and Monday->Tuesday tables are
tested with fixed, outcome-independent ticket rules. Reports are written to the
dated lottery_audit directory; the draw CSV and existing scripts are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lottery_data import CANONICAL_CSV, today
from Single_V2_0.number_repeat_next_others import load_rows


CURRENT_FORMAT_START = date(2022, 5, 17)
MIN_HISTORY_DRAWS = 52
TOP_FOLLOWERS_PER_SOURCE = 5
TICKETS_PER_DRAW = 20
PORTFOLIO_POOL_SIZE = 20
RANDOM_SEED = 20261020
NUMBER_RANGE = tuple(range(1, 48))
RULE_REFERENCE = "https://www.lotterywest.wa.gov.au/lotterywest/media-centre/bigger-prizes-and-more-winners-with-oz-lotto-game-change"


@dataclass(frozen=True)
class PreparedRow:
    date: date
    counts: Counter
    main: tuple


def prepare(data):
    return [PreparedRow(row.date, Counter(row.all_numbers()), tuple(row.others_main_numbers())) for row in data]


def valid_tuesday(row):
    return (row.date.weekday() == 1 and row.date >= CURRENT_FORMAT_START
            and len(row.main) == 7 and len(set(row.main)) == 7
            and all(1 <= n <= 47 for n in row.main))


def build_tables(history, method):
    """This function receives strictly pre-target rows, never the target result."""
    present_targets = {n: set() for n in NUMBER_RANGE}
    repeated_targets = {n: set() for n in NUMBER_RANGE}
    if method == "all_days":
        # Same event definitions and next-Others deduplication as the existing script.
        next_result = [None] * len(history)
        following = None
        for i in range(len(history) - 1, -1, -1):
            next_result[i] = following
            if history[i].main:
                following = i
        for i, row in enumerate(history):
            target = next_result[i]
            if target is None:
                continue
            for n, count in row.counts.items():
                if n not in present_targets:
                    continue
                present_targets[n].add(target)
                if count >= 2 or (i and history[i - 1].counts[n]):
                    repeated_targets[n].add(target)
    elif method == "monday_tuesday":
        by_date = {row.date: row for row in history}
        for i, row in enumerate(history):
            if not valid_tuesday(row):
                continue
            monday = by_date.get(row.date - timedelta(days=1))
            if monday is None or monday.date.weekday() != 0:
                continue
            for n, count in monday.counts.items():
                if n not in present_targets:
                    continue
                present_targets[n].add(i)
                if count >= 2:
                    repeated_targets[n].add(i)
    else:
        raise ValueError(f"Unknown training method: {method}")
    tables = {}
    for name, source_targets in (("single", present_targets), ("repeated", repeated_targets)):
        tables[name] = {}
        for n, targets in source_targets.items():
            counts = Counter()
            for i in sorted(targets):
                counts.update(x for x in set(history[i].main) if 1 <= x <= 47)
            tables[name][n] = {"support": len(targets), "counts": counts,
                              "latest_outcome": max((history[i].date for i in targets), default=None)}
    return tables


def group_votes(sources, table):
    votes = Counter()
    details = []
    used = 0
    for source in sources:
        record = table[source]
        ranking = sorted(record["counts"], key=lambda n: (-record["counts"][n], n))
        followers = ranking[:TOP_FOLLOWERS_PER_SOURCE]
        details.append({"source_number": source, "historical_following_draws": record["support"],
                        "latest_training_outcome": record["latest_outcome"].isoformat() if record["latest_outcome"] else None,
                        "top_followers": [{"number": n, "draws": record["counts"][n]} for n in followers]})
        if not followers:
            continue
        used += 1
        points = list(range(len(followers), 0, -1))
        mass = sum(points)
        for n, point in zip(followers, points):
            votes[n] += point / mass
    if used:
        votes = Counter({n: score / used for n, score in votes.items()})
    return votes, details


def seed_for(target, strategy):
    token = f"{RANDOM_SEED}|{target.isoformat()}|{strategy}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def ticket_portfolio(ranking, target, strategy, count=TICKETS_PER_DRAW):
    """First ticket is top seven; others are fixed-seed weighted rank samples."""
    if not ranking:
        return []
    pool = list(ranking[:PORTFOLIO_POOL_SIZE])
    if len(set(pool)) != len(pool) or len(pool) < 7:
        raise ValueError("A candidate ranking needs at least seven distinct numbers")
    if any(n not in NUMBER_RANGE for n in pool):
        raise ValueError("Candidate outside Oz Lotto's 1..47 universe")
    target_count = min(count, math.comb(len(pool), 7))
    tickets = [sorted(pool[:7])]
    seen = {tuple(tickets[0])}
    rng = random.Random(seed_for(target, strategy))
    weights = {n: 1 / (i + 1) for i, n in enumerate(pool)}
    attempts = 0
    while len(tickets) < target_count and attempts < 100000:
        attempts += 1
        remaining = pool.copy()
        picked = []
        for _ in range(7):
            n = rng.choices(remaining, weights=[weights[x] for x in remaining], k=1)[0]
            picked.append(n)
            remaining.remove(n)
        ticket = tuple(sorted(picked))
        if ticket not in seen:
            seen.add(ticket)
            tickets.append(list(ticket))
    if len(tickets) != target_count:
        raise RuntimeError("Could not complete the declared fixed ticket budget")
    return tickets


def predict_for_date(data, target, min_history=MIN_HISTORY_DRAWS):
    """Cut off all current/future outcomes BEFORE deriving any training features."""
    if target.weekday() != 1:
        raise ValueError("Target must be Tuesday")
    history = [row for row in data if row.date < target]
    by_date = {row.date: row for row in history}
    monday = by_date.get(target - timedelta(days=1))
    if monday is None:
        raise ValueError("No preceding Monday row")
    historical_tuesdays = [row for row in history if valid_tuesday(row)]
    if len(historical_tuesdays) < min_history:
        raise ValueError(f"Need {min_history} prior current-format Tuesdays")
    prior_frequency = Counter(n for row in historical_tuesdays for n in row.main)
    base_ranking = sorted(NUMBER_RANGE, key=lambda n: (-prior_frequency[n], n))
    repeated = sorted(n for n, count in monday.counts.items() if count >= 2 and n in NUMBER_RANGE)
    singles = sorted(n for n, count in monday.counts.items() if count == 1 and n in NUMBER_RANGE)
    output = []
    for method in ("all_days", "monday_tuesday"):
        tables = build_tables(history, method)
        repeat_votes, repeat_details = group_votes(repeated, tables["repeated"])
        single_votes, single_details = group_votes(singles, tables["single"])
        for mode in ("repeated_only", "single_only", "combined"):
            active = ([repeat_votes] if mode == "repeated_only" else [single_votes] if mode == "single_only"
                      else [votes for votes in (repeat_votes, single_votes) if votes])
            votes = Counter()
            for group in active:
                for n, score in group.items():
                    votes[n] += score / len(active)
            strategy = f"{method}/{mode}"
            ranking = sorted(NUMBER_RANGE, key=lambda n: (-votes[n], -prior_frequency[n], n)) if votes else []
            tickets = ticket_portfolio(ranking, target, strategy)
            output.append({"strategy": strategy, "target": target.isoformat(),
                           "monday": monday.date.isoformat(), "latest_input_date": history[-1].date.isoformat(),
                           "historical_tuesdays": len(historical_tuesdays),
                           "repeated_sources": repeated, "single_sources": singles,
                           "repeated_followups": repeat_details, "single_followups": single_details,
                           "ranking": ranking, "positive_vote_candidates": sorted(votes),
                           "votes": dict(votes), "tickets": tickets,
                           "abstention": None if tickets else "No supported source numbers for this strategy"})
    # Equal-budget controls, fixed before seeing any test outcome.
    strategy = "baseline/prior_tuesday_frequency"
    output.append({"strategy": strategy, "target": target.isoformat(), "monday": monday.date.isoformat(),
                   "latest_input_date": history[-1].date.isoformat(), "historical_tuesdays": len(historical_tuesdays),
                   "ranking": base_ranking, "tickets": ticket_portfolio(base_ranking, target, strategy), "abstention": None})
    strategy = "baseline/uniform_random"
    rng = random.Random(seed_for(target, strategy))
    tickets = []
    while len(tickets) < TICKETS_PER_DRAW:
        ticket = sorted(rng.sample(NUMBER_RANGE, 7))
        if ticket not in tickets:
            tickets.append(ticket)
    output.append({"strategy": strategy, "target": target.isoformat(), "monday": monday.date.isoformat(),
                   "latest_input_date": history[-1].date.isoformat(), "historical_tuesdays": len(historical_tuesdays),
                   "tickets": tickets, "abstention": None})
    for prediction in output:
        prediction["frozen_ticket_sha256"] = hashlib.sha256(json.dumps(prediction["tickets"]).encode()).hexdigest()
    return output


def score_frozen(prediction, actual):
    digest = hashlib.sha256(json.dumps(prediction["tickets"]).encode()).hexdigest()
    if digest != prediction["frozen_ticket_sha256"]:
        raise ValueError("Ticket set changed before scoring")
    hits = [len(set(ticket) & set(actual)) for ticket in prediction["tickets"]]
    return {**prediction, "actual_main": sorted(actual), "ticket_hits": hits,
            "first_ticket_hits": hits[0] if hits else None, "best_hits": max(hits, default=None),
            "jackpot_tickets": hits.count(7)}


def statistics(records):
    strategies = sorted({r["strategy"] for r in records})
    result = []
    for strategy in strategies:
        selected = [r for r in records if r["strategy"] == strategy]
        hits = Counter(h for r in selected for h in r["ticket_hits"])
        best = [r["best_hits"] or 0 for r in selected]
        row = {"strategy": strategy, "draws": len(selected), "tickets": sum(hits.values()),
               "abstentions": sum(not r["tickets"] for r in selected), "best_single_ticket": max(hits, default=0),
               "best_first_ticket": max((r["first_ticket_hits"] or 0 for r in selected), default=0),
               "mean_first_ticket_hits": sum(r["first_ticket_hits"] or 0 for r in selected) / len(selected),
               "mean_hits_per_ticket": sum(h*c for h,c in hits.items()) / sum(hits.values()) if hits else 0,
               "jackpot_tickets": hits[7]}
        for h in range(3, 8):
            row[f"draws_best_{h}_plus"] = sum(value >= h for value in best)
        for h in range(8):
            row[f"tickets_{h}_hits"] = hits[h]
        result.append(row)
    return result


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = list(dict.fromkeys(k for row in records for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def print_statistics(title, stats):
    print("\n" + title)
    print(f"{'Strategy':<38} {'Draws':>5} {'Tickets':>7} {'First max':>9} {'Best':>4} {'4+':>4} {'5+':>4} {'6+':>4} {'7':>3}")
    for r in stats:
        print(f"{r['strategy']:<38} {r['draws']:>5} {r['tickets']:>7} {r['best_first_ticket']:>9} "
              f"{r['best_single_ticket']:>4} {r['draws_best_4_plus']:>4} {r['draws_best_5_plus']:>4} "
              f"{r['draws_best_6_plus']:>4} {r['jackpot_tickets']:>3}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--last", type=int, default=0, help="number of latest target draws; 0 tests all eligible history")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.last < 0:
        parser.error("--last cannot be negative")
    source_hash = hashlib.sha256(CANONICAL_CSV.read_bytes()).hexdigest()
    data = prepare(load_rows())
    eligible = [row for i, row in enumerate(data) if valid_tuesday(row)
                and sum(valid_tuesday(prior) for prior in data[:i]) >= MIN_HISTORY_DRAWS
                and any(prior.date == row.date - timedelta(days=1) for prior in data[:i])]
    targets = eligible[-args.last:] if args.last else eligible
    if not targets:
        parser.error("No targets with the required pre-draw history")
    out = args.output_dir or PROJECT_ROOT / "lottery_audit" / today().isoformat() / "oz_monday_followups"
    out.mkdir(parents=True, exist_ok=True)
    print("Fixed rules: top 5 followers per source; equal repeated/single group weights; top-seven first ticket.")
    print("Portfolio: 20 tickets/strategy/draw; 19 fixed-seed samples from top 20. No held-out tuning.")
    print(f"Testing {len(targets)} Tuesdays: {targets[0].date} to {targets[-1].date}", flush=True)
    records = []
    for number, target_row in enumerate(targets, 1):
        frozen = predict_for_date(data, target_row.date)
        # The actual result is accessed for scoring only after all tickets are frozen.
        for prediction in frozen:
            records.append(score_frozen(prediction, target_row.main))
        if number % 20 == 0 or number == len(targets):
            print(f"  Completed {number}/{len(targets)}", flush=True)
    all_stats = statistics(records)
    recent_dates = {row.date.isoformat() for row in targets[-10:]}
    last10 = [r for r in records if r["target"] in recent_dates]
    last10_stats = statistics(last10)
    write_csv(out / "statistics_all.csv", all_stats)
    write_csv(out / "statistics_last10.csv", last10_stats)
    per_draw = []
    ticket_rows = []
    for r in records:
        best_index = max(range(len(r["tickets"])), key=lambda i: r["ticket_hits"][i], default=None)
        per_draw.append({"target": r["target"], "monday": r["monday"], "strategy": r["strategy"],
                         "input_through": r["latest_input_date"], "history_tuesdays": r["historical_tuesdays"],
                         "actual_main": " ".join(map(str, r["actual_main"])), "tickets": len(r["tickets"]),
                         "first_ticket": " ".join(map(str, r["tickets"][0])) if r["tickets"] else "",
                         "first_hits": r["first_ticket_hits"], "best_hits": r["best_hits"],
                         "best_ticket": " ".join(map(str, r["tickets"][best_index])) if best_index is not None else "",
                         "jackpot_tickets": r["jackpot_tickets"], "abstention": r["abstention"],
                         "frozen_ticket_sha256": r["frozen_ticket_sha256"]})
        for i, (ticket, hits) in enumerate(zip(r["tickets"], r["ticket_hits"]), 1):
            ticket_rows.append({"target": r["target"], "strategy": r["strategy"], "ticket_number": i,
                                "ticket": " ".join(map(str, ticket)), "main_hits": hits})
    write_csv(out / "per_draw.csv", per_draw)
    write_csv(out / "tickets.csv", ticket_rows)
    (out / "evidence_and_predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    latest_monday = max((row for row in data if row.date.weekday() == 0), key=lambda row: row.date)
    future_target = latest_monday.date + timedelta(days=1)
    future = predict_for_date(data, future_target) if future_target > data[-1].date else []
    (out / "unscored_next_tuesday.json").write_text(json.dumps(future, indent=2), encoding="utf-8")
    assert hashlib.sha256(CANONICAL_CSV.read_bytes()).hexdigest() == source_hash, "Source CSV was changed"
    manifest = {"source": str(CANONICAL_CSV), "source_sha256": source_hash, "source_unchanged": True,
                "source_latest_date": data[-1].date.isoformat(), "target_first": targets[0].date.isoformat(),
                "target_last": targets[-1].date.isoformat(), "target_draws": len(targets),
                "last10_targets": sorted(recent_dates), "format_start": CURRENT_FORMAT_START.isoformat(),
                "format_reference": RULE_REFERENCE, "minimum_prior_current_format_tuesdays": MIN_HISTORY_DRAWS,
                "top_followers_per_source": TOP_FOLLOWERS_PER_SOURCE, "tickets_per_strategy_per_draw": TICKETS_PER_DRAW,
                "pool_size": PORTFOLIO_POOL_SIZE, "seed": RANDOM_SEED,
                "cutoff": "Only rows strictly before the target Tuesday are passed into historical table builders.",
                "fixed_rules": "Each Monday number occurring >=2 times is a repeated source; exactly once is a single source. Historical single tables match any presence. Top five per source vote 5/4/3/2/1, normalized per source and averaged within each group. Combined gives equal weight to supported groups. Ties use prior Tuesday main frequency then number. Top seven form ticket one; remaining tickets are unique inverse-rank weighted samples from top twenty. Zero-vote candidates fill remaining ranked pool slots using prior Tuesday frequency. No supported source group means abstention.",
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_module_sha256": hashlib.sha256((Path(__file__).parent / "number_repeat_next_others.py").read_bytes()).hexdigest(),
                "next_unscored_target": future_target.isoformat() if future else None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print_statistics("ALL ELIGIBLE DRAWS", all_stats)
    print_statistics("LAST 10 DRAWS", last10_stats)
    print(f"\nSaved: {out}")
    return records


if __name__ == "__main__":
    main()
