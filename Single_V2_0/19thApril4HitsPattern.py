import csv
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math

# ---------- CONFIGURATION ----------
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_lotto_data_backup.csv")
OUTPUT_LAST_N = 30          # rows shown in historical tables
FUTURE_DATE_STR = "Sat 26-Sep-2026"   # example: "Fri 04-Sep-2026", "Tue 01-Sep-2026", etc.
# Locked EH/H/W/C profile for conditional winning-trajectory analysis.
# Example: EH=2, H=1, W=3, C=0. (2, 1, 3, 0)
# LOCKED_PROFILE = (1, 0, 5, 0)
LOCKED_PROFILE = ""
LOCKED_TRAJECTORY_TOP_N = 30

# All-history locked-profile trajectory analysis.
# Scope options:
#   "same_weekday"   -> only the target weekday (e.g. Wednesdays)
#   "same_game"      -> all weekdays for the same lottery/game
#                       (Weekday Windfall => Mon/Wed/Fri)
#   "same_main_count"-> any Others draw with the same number of main balls
#                       (for 6-number targets this includes Mon/Wed/Fri/Sat)
#   "all_others"     -> every Others draw (use cautiously when games have
#                       different main-ball counts/universes)
LOCKED_EXACT_SCOPE = "same_main_count"
LOCKED_INDEPENDENT_SCOPE = "same_main_count"
PRINT_ALL_HISTORY_EXACT_PROFILE = True
PRINT_ALL_HISTORY_INDEPENDENT_COUNTS = True
PRINT_LOCKED_MATCHING_DRAW_DETAILS = True
LOCKED_CONDITIONAL_MIN_CANDIDATES = 1

WEEK_TABLE_DAYS = 30
POOL_LOOKBACK_DAYS = 30

# Backtest settings (applied per lottery)
RUN_BACKTEST = False
BACKTEST_N = 10             # number of most recent draws to backtest (per lottery)
K_NEIGHBORS = 10            # for conditional mode predictor

# Trajectory analysis settings
# These do NOT replace the existing EH/H/W/C model. They add a second layer that
# tracks how each number moves between C -> W -> H -> EH (and back) day by day.
PRINT_DAILY_SNAPSHOT_POOLS = True
PRINT_TRAJECTORY_TABLE = True
RUN_TRAJECTORY_HISTORY = True
PRINT_RECENT_WINNER_TRAJECTORIES = True

TRAJECTORY_TOP_N = 30
TRAJECTORY_MIN_SAMPLES = 5
TRAJECTORY_RECENT_DRAWS = 30

# Powerball-ball (1-20) analysis.
# This module runs ONLY when the prediction target is Thursday / Powerball.
POWERBALL_BALL_MAX = 20
POWERBALL_BALL_TOP_N = 5
POWERBALL_BALL_BACKTEST_N = 80
POWERBALL_BALL_DECAY = 0.92
POWERBALL_BALL_PRIOR_STRENGTH = 20.0




# Set this to a specific date to predict only that day's lottery.
# Leave empty to predict the next draw for ALL lotteries.
# FUTURE_DATE_STR = ""                # uncomment to process all lotteries

# Lottery definitions: day abbreviation -> (name, max number, main count)
LOTTERY_CONFIG = {
    'Mon': ('Weekday Windfall', 45, 6),
    'Tue': ('Oz Lotto', 47, 7),
    'Wed': ('Weekday Windfall', 45, 6),
    'Thu': ('Powerball', 35, 7),
    'Fri': ('Weekday Windfall', 45, 6),
    'Sat': ('Saturday Lotto', 45, 6),
}

# ---------- HELPER FUNCTIONS ----------
def parse_date(s):
    # input: "Sat 05-Sep-2026" -> datetime
    return datetime.strptime(s[4:], '%d-%b-%Y')

def extract_main_numbers(cell):
    """Extract the main numbers from the first bracket of the 'Others' column."""
    if not cell:
        return []
    main_part = cell.split(']')[0].replace('[', '').strip()
    if not main_part:
        return []
    return [int(x.strip()) for x in main_part.split(',') if x.strip()]

def extract_all_numbers(cell):
    """Extract all numbers (main + supplementary) from a cell."""
    nums = []
    for part in cell.split(']'):
        part = part.replace('[', '').strip()
        if part:
            for token in part.split(','):
                token = token.strip()
                if token:
                    nums.append(int(token))
    return nums


def extract_bracket_groups(cell):
    """
    Return every [...] group in a lottery result cell as a list of integer lists.

    Example Powerball Others cell:
        "[2, 11, 24, 28, 29, 32, 34], [13]"
    becomes:
        [[2, 11, 24, 28, 29, 32, 34], [13]]
    """
    if not cell:
        return []

    groups = []
    pos = 0
    while True:
        left = cell.find('[', pos)
        if left < 0:
            break
        right = cell.find(']', left + 1)
        if right < 0:
            break

        content = cell[left + 1:right].strip()
        nums = []
        if content:
            for token in content.split(','):
                token = token.strip()
                if not token:
                    continue
                try:
                    nums.append(int(token))
                except ValueError:
                    pass

        groups.append(nums)
        pos = right + 1

    return groups


def extract_powerball_ball(cell, pb_max=20):
    """
    Extract the single Powerball from the SECOND bracket of a Thursday result.

    Main numbers remain in the first bracket and are intentionally ignored here.
    Returns None if a valid 1..pb_max Powerball cannot be found.
    """
    groups = extract_bracket_groups(cell)
    if len(groups) < 2 or not groups[1]:
        return None

    pb = groups[1][0]
    return pb if 1 <= pb <= pb_max else None


def round_to_sum(raw, target_sum):
    """Round a list of non‑negative floats to integers that sum to target_sum."""
    raw = [max(0.0, x) for x in raw]
    total = sum(raw)
    if total == 0:
        base = target_sum // len(raw)
        rem = target_sum % len(raw)
        res = [base] * len(raw)
        for i in range(rem):
            res[i] += 1
        return tuple(res)
    floors = [int(x) for x in raw]
    remainders = [(raw[i] - floors[i], i) for i in range(len(raw))]
    current_sum = sum(floors)
    remainders.sort(reverse=True)
    for i in range(target_sum - current_sum):
        idx = remainders[i % len(raw)][1]
        floors[idx] += 1
    return tuple(floors)


# ---------- POOL / TRAJECTORY HELPERS ----------
POOL_NAMES = ("EH", "H", "W", "C")


def build_pools(counter, max_num):
    """
    Convert a frequency Counter into EH/H/W/C pools.

    EH = frequency >= 4
    H  = frequency == 3
    W  = frequency 1..2
    C  = frequency == 0
    """
    eh = {n for n in range(1, max_num + 1) if counter[n] >= 4}
    h = {n for n in range(1, max_num + 1) if counter[n] == 3}
    w = {n for n in range(1, max_num + 1) if 1 <= counter[n] <= 2}
    c = {n for n in range(1, max_num + 1) if counter[n] == 0}

    return {
        "EH": eh,
        "H": h,
        "W": w,
        "C": c,
    }


def pool_state(number, pools):
    """Return EH/H/W/C for a number in a pool dictionary."""
    for state in POOL_NAMES:
        if number in pools[state]:
            return state
    raise ValueError(f"Number {number} is not present in any pool.")


def trajectory_text(states):
    """Return the COMPLETE daily EH/H/W/C state sequence as display text."""
    return "→".join(states)


def build_rolling_snapshot(snapshot_dt, all_rows, max_num, lookback_days=7):
    """
    Build an EH/H/W/C snapshot for ONE calendar date.

    IMPORTANT:
    - The number universe is fixed to the TARGET lottery's max_num.
    - The same lookback length is used for every calendar-day snapshot.
    - The snapshot excludes draws occurring on snapshot_dt itself.
    - Therefore Sunday can be represented even though there is no 'Others'
      lottery on Sunday.
    """
    window_start = snapshot_dt - timedelta(days=lookback_days)
    window_nums = []

    for _, dt, _, all_nums, _ in all_rows:
        if window_start <= dt < snapshot_dt:
            window_nums.extend(n for n in all_nums if 1 <= n <= max_num)

    counter = Counter(window_nums)
    pools = build_pools(counter, max_num)

    return {
        "date": snapshot_dt,
        "window_start": window_start,
        "counter": counter,
        "pools": pools,
        "sizes": tuple(len(pools[state]) for state in POOL_NAMES),
    }


def build_daily_trajectory_snapshots(prev_target_dt, target_dt, all_rows, max_num):
    """
    Build one rolling snapshot for every calendar day from the previous target
    draw through the target date, inclusive.

    Example for Tuesday Oz Lotto:
        Tue 08, Wed 09, Thu 10, Fri 11, Sat 12, Sun 13, Mon 14, Tue 15(final)

    The lookback length is the actual gap between the two target draws
    (normally 7 days).
    """
    lookback_days = max(1, (target_dt - prev_target_dt).days)

    snapshots = []
    dt = prev_target_dt
    while dt <= target_dt:
        snapshots.append(
            build_rolling_snapshot(
                snapshot_dt=dt,
                all_rows=all_rows,
                max_num=max_num,
                lookback_days=lookback_days,
            )
        )
        dt += timedelta(days=1)

    return snapshots


def number_trajectory(number, snapshots):
    """Return the full daily state sequence for one number."""
    return [pool_state(number, snapshot["pools"]) for snapshot in snapshots]


def print_daily_snapshot_pools(prev_target_dt, target_dt, all_rows, max_num, lottery_name):
    """
    Print the actual pool members for every daily rolling snapshot.
    Sunday is included.
    """
    snapshots = build_daily_trajectory_snapshots(
        prev_target_dt, target_dt, all_rows, max_num
    )

    lookback_days = max(1, (target_dt - prev_target_dt).days)

    print("\n" + "=" * 120)
    print(
        f"Daily Rolling Pool Snapshots for {lottery_name} "
        f"(fixed target universe 1-{max_num}, {lookback_days}-day rolling window)"
    )
    print("Sunday is included. Each snapshot excludes that calendar day's draws.")
    print("=" * 120)

    for snapshot in snapshots:
        dt = snapshot["date"]
        pools = snapshot["pools"]
        sizes = snapshot["sizes"]

        suffix = "  <-- FINAL TARGET POOL" if dt == target_dt else ""
        print(
            f"\n{dt.strftime('%a %d-%b-%Y')}  "
            f"Window: {snapshot['window_start'].strftime('%a %d-%b-%Y')} "
            f"to {dt.strftime('%a %d-%b-%Y')} (exclusive)"
            f"{suffix}"
        )
        print(
            f"  Pool sizes: EH={sizes[0]}  H={sizes[1]}  "
            f"W={sizes[2]}  C={sizes[3]}"
        )
        print(f"  EH: {sorted(pools['EH'])}")
        print(f"  H : {sorted(pools['H'])}")
        print(f"  W : {sorted(pools['W'])}")
        print(f"  C : {sorted(pools['C'])}")


def print_number_trajectory_table(prev_target_dt, target_dt, all_rows, max_num, lottery_name):
    """
    Print one row per number showing the COMPLETE EH/H/W/C state on every day.
    No full daily trajectory is calculated or displayed.
    """
    snapshots = build_daily_trajectory_snapshots(
        prev_target_dt, target_dt, all_rows, max_num
    )

    labels = [snap["date"].strftime("%a%d") for snap in snapshots]
    if labels:
        labels[-1] = "FINAL"

    print("\n" + "=" * 110)
    print(
        f"Number-by-Number FULL EH/H/W/C Daily States for {lottery_name} "
        f"(target universe 1-{max_num})"
    )
    print(
        f"Previous target draw: {prev_target_dt.strftime('%a %d-%b-%Y')}  |  "
        f"Target: {target_dt.strftime('%a %d-%b-%Y')}"
    )
    print("=" * 110)

    header = f"{'No':<4}"
    for label in labels:
        header += f"{label:<8}"
    print(header)
    print("-" * max(80, len(header)))

    for number in range(1, max_num + 1):
        states = number_trajectory(number, snapshots)
        row = f"{number:<4}"
        for state in states:
            row += f"{state:<8}"
        print(row)

def collect_trajectory_pattern_stats(draws, all_rows, max_num, cutoff_dt=None):
    """
    Historical non-cheating trajectory statistics.

    For every historical target draw BEFORE cutoff_dt:
      1. Rebuild the daily rolling snapshots using only data available before
         each snapshot.
      2. Keep every number's COMPLETE daily state sequence.
      3. Count candidate-number instances for each (final pool, trajectory).
      4. Count how many of those candidate instances became main-number winners.

    cutoff_dt:
      If supplied, a target draw on cutoff_dt is deliberately excluded. This is
      important when backtesting a date whose result already exists in the CSV.

    Returns:
      stats          dict keyed by (final_state, full_daily_state_tuple)
      winner_records one record per historical winning main number
      analyzed_draws number of target draws included
    """
    stats = defaultdict(lambda: {"candidates": 0, "winners": 0})
    winner_records = []
    analyzed_draws = 0

    if not all_rows:
        return stats, winner_records, analyzed_draws

    earliest_dt = min(row[1] for row in all_rows)

    for i in range(1, len(draws)):
        target_date_str, target_dt, target_main = draws[i]
        _, prev_dt, _ = draws[i - 1]

        if cutoff_dt is not None and target_dt >= cutoff_dt:
            continue

        lookback_days = max(1, (target_dt - prev_dt).days)

        # We need one complete lookback window BEFORE the previous target draw
        # to construct the first state in the trajectory without partial data.
        if prev_dt - timedelta(days=lookback_days) < earliest_dt:
            continue

        snapshots = build_daily_trajectory_snapshots(
            prev_dt, target_dt, all_rows, max_num
        )

        target_main_set = {
            n for n in target_main
            if 1 <= n <= max_num
        }

        for number in range(1, max_num + 1):
            states = number_trajectory(number, snapshots)
            final_state = states[-1]
            trajectory = tuple(states)
            key = (final_state, trajectory)

            stats[key]["candidates"] += 1

            if number in target_main_set:
                stats[key]["winners"] += 1
                winner_records.append({
                    "date": target_date_str,
                    "dt": target_dt,
                    "number": number,
                    "final_state": final_state,
                    "trajectory": trajectory,
                    "full_states": states,
                })

        analyzed_draws += 1

    return stats, winner_records, analyzed_draws


def print_trajectory_pattern_history(
    draws,
    all_rows,
    max_num,
    main_count,
    lottery_name,
    cutoff_dt=None,
):
    """
    Print historical trajectory-pattern performance and return the stats so the
    same non-cheating history can be used to annotate the current candidates.
    """
    stats, winner_records, analyzed_draws = collect_trajectory_pattern_stats(
        draws=draws,
        all_rows=all_rows,
        max_num=max_num,
        cutoff_dt=cutoff_dt,
    )

    print("\n" + "=" * 120)
    if cutoff_dt is None:
        print(f"Historical Full Daily Trajectory Analysis - {lottery_name}")
    else:
        print(
            f"Historical Full Daily Trajectory Analysis - {lottery_name} "
            f"(STRICTLY BEFORE {cutoff_dt.strftime('%a %d-%b-%Y')})"
        )
    print("=" * 120)

    if analyzed_draws == 0:
        print("Not enough complete historical windows to analyse trajectories.")
        return stats, winner_records

    baseline = main_count / max_num
    print(f"Historical target draws analysed: {analyzed_draws}")
    print(
        f"Unconditional per-number main-draw baseline: "
        f"{main_count}/{max_num} = {baseline:.2%}"
    )
    print(
        "Hit Rate below = historical winning main-number instances / "
        "candidate-number instances with that same final pool + trajectory."
    )

    for final_state in POOL_NAMES:
        rows = []
        for (state, trajectory), values in stats.items():
            if state != final_state:
                continue

            candidates = values["candidates"]
            winners = values["winners"]

            if candidates < TRAJECTORY_MIN_SAMPLES:
                continue

            hit_rate = winners / candidates if candidates else 0.0
            rows.append((trajectory, candidates, winners, hit_rate))

        rows.sort(
            key=lambda x: (
                -x[2],       # winner count
                -x[3],       # hit rate
                -x[1],       # sample size
                x[0],
            )
        )

        print(f"\nFinal pool = {final_state}")
        if not rows:
            print(
                f"  No patterns have at least "
                f"{TRAJECTORY_MIN_SAMPLES} candidate instances."
            )
            continue

        print(
            f"  {'Full daily trajectory':<50} "
            f"{'Candidates':>10} {'Winners':>9} {'Hit Rate':>10}"
        )
        print("  " + "-" * 68)

        for trajectory, candidates, winners, hit_rate in rows[:TRAJECTORY_TOP_N]:
            print(
                f"  {trajectory_text(trajectory):<50} "
                f"{candidates:>10} {winners:>9} {hit_rate:>9.2%}"
            )

    if PRINT_RECENT_WINNER_TRAJECTORIES and winner_records:
        distinct_dates = sorted(
            {r["dt"] for r in winner_records}
        )
        keep_dates = set(distinct_dates[-TRAJECTORY_RECENT_DRAWS:])

        print("\n" + "-" * 120)
        print(
            f"Winning-number trajectories from the most recent "
            f"{min(TRAJECTORY_RECENT_DRAWS, len(distinct_dates))} "
            f"historical {lottery_name} draws used above"
        )
        print("-" * 120)
        print(
            f"{'Date':<20} {'No':<4} {'Final':<6} {'Full daily states'}"
        )

        for record in winner_records:
            if record["dt"] not in keep_dates:
                continue
            full_path = "→".join(record["full_states"])
            print(
                f"{record['date']:<20} "
                f"{record['number']:<4} "
                f"{record['final_state']:<6} "
                f"{full_path}"
            )

    return stats, winner_records


def print_current_trajectory_groups(
    prev_target_dt,
    target_dt,
    all_rows,
    max_num,
    stats,
    lottery_name,
):
    """
    Group the current target pool by full daily trajectory and annotate each
    group with its historical non-cheating candidate/winner counts.
    """
    snapshots = build_daily_trajectory_snapshots(
        prev_target_dt, target_dt, all_rows, max_num
    )

    groups = defaultdict(list)

    for number in range(1, max_num + 1):
        states = number_trajectory(number, snapshots)
        final_state = states[-1]
        trajectory = tuple(states)
        groups[(final_state, trajectory)].append(number)

    print("\n" + "=" * 120)
    print(f"Current {lottery_name} Pool Grouped by Full Daily Trajectory")
    print(
        "Historical figures are based only on target draws before the current "
        "target date, so the current result cannot leak into the score."
    )
    print("=" * 120)

    for final_state in POOL_NAMES:
        rows = []

        for (state, trajectory), numbers in groups.items():
            if state != final_state:
                continue

            hist = stats.get(
                (state, trajectory),
                {"candidates": 0, "winners": 0},
            )
            candidates = hist["candidates"]
            winners = hist["winners"]
            hit_rate = winners / candidates if candidates else 0.0

            rows.append(
                (
                    trajectory,
                    sorted(numbers),
                    candidates,
                    winners,
                    hit_rate,
                )
            )

        rows.sort(
            key=lambda x: (
                -x[4],   # historical hit rate
                -x[3],   # historical winners
                -x[2],   # sample size
                x[0],
            )
        )

        print(f"\nFinal pool = {final_state}")
        print(
            f"  {'Full daily trajectory':<50} {'Current numbers':<38} "
            f"{'Hist N':>7} {'Wins':>6} {'Rate':>9}"
        )
        print("  " + "-" * 98)

        for trajectory, numbers, candidates, winners, hit_rate in rows:
            number_str = str(numbers)
            print(
                f"  {trajectory_text(trajectory):<50} {number_str:<38} "
                f"{candidates:>7} {winners:>6} {hit_rate:>8.2%}"
            )


# ---------- WEEK TABLE FUNCTION ----------
def print_week_table(
    future_dt,
    all_rows,
    draws_by_day,
    target_max_num,
    target_lottery_name,
    table_days=WEEK_TABLE_DAYS,
    pool_lookback_days=7,
):
    """
    Print the preceding 7 CALENDAR DAYS in two formats:

    1. Detailed:
       row + EH/H/W/C pool members + actual hits where an Others draw exists.

    2. Compact:
       same information as the old Week Table, one row per calendar day.

    Sunday is included.

    IMPORTANT:
    - Pool snapshots use the TARGET lottery number universe consistently.
      Example:
          Weekday Windfall -> 1..45
          Oz Lotto         -> 1..47
    - Each day's snapshot uses [day - lookback_days, day), therefore it does
      not use that day's result to build its pool.
    - Sunday has no Others draw, so its actual Profile / EH/H/W/C hit counts
      are shown as "-".  Its pool sizes and members are still valid.
    """

    start_dt = future_dt - timedelta(days=table_days)

    # ------------------------------------------------------------
    # Helper: locate the CSV row for one calendar date
    # ------------------------------------------------------------
    rows_by_date = {
        dt.date(): (date_str, dt, day_abbr, all_nums, others_cell)
        for date_str, dt, day_abbr, all_nums, others_cell in all_rows
    }

    week_rows = []

    dt = start_dt
    while dt < future_dt:

        # Build pool BEFORE this calendar day's results.
        snapshot = build_rolling_snapshot(
            snapshot_dt=dt,
            all_rows=all_rows,
            max_num=target_max_num,
            lookback_days=pool_lookback_days,
        )

        pools = snapshot["pools"]

        eh = pools["EH"]
        h = pools["H"]
        w = pools["W"]
        c = pools["C"]

        eh_pool = len(eh)
        h_pool = len(h)
        w_pool = len(w)
        c_pool = len(c)
        eh_h_pool = eh_pool + h_pool

        day_abbr = dt.strftime("%a")[:3]
        date_str = dt.strftime("%a %d-%b-%Y")

        # --------------------------------------------------------
        # Actual "Others" result for this date, if one exists.
        #
        # Sunday normally has no Others result, therefore main_nums
        # stays empty and the row becomes a snapshot-only row.
        # --------------------------------------------------------
        main_nums = []
        others_cell = None

        source_row = rows_by_date.get(dt.date())

        if source_row is not None:
            _, _, source_day, _, others_cell = source_row

            if others_cell:
                main_nums = [
                    n for n in extract_main_numbers(others_cell)
                    if 1 <= n <= target_max_num
                ]

        if main_nums:

            eh_hits = sorted(n for n in main_nums if n in eh)
            h_hits = sorted(n for n in main_nums if n in h)
            w_hits = sorted(n for n in main_nums if n in w)
            c_hits = sorted(n for n in main_nums if n in c)

            eh_count = len(eh_hits)
            h_count = len(h_hits)
            w_count = len(w_hits)
            c_count = len(c_hits)

            # Use actual number of main numbers on that date.
            actual_main_count = len(main_nums)

            profile = (
                "Breadth"
                if w_count >= (actual_main_count // 2 + 1)
                else "Depth"
            )

            # Previous same-weekday Others draw, for Legacy Hits.
            prev_main = None

            for prev_date, prev_dt, prev_nums in draws_by_day.get(day_abbr, []):
                if prev_dt < dt:
                    prev_main = prev_nums
                else:
                    break

            if prev_main is not None:
                legacy_hits = sorted(
                    n for n in main_nums
                    if n in prev_main
                )
                legacy_str = str(legacy_hits) if legacy_hits else "None"
            else:
                legacy_str = "None"

        else:
            # Sunday / no Others result.
            profile = "Snapshot"

            eh_count = None
            h_count = None
            w_count = None
            c_count = None

            eh_hits = []
            h_hits = []
            w_hits = []
            c_hits = []

            legacy_str = "-"

        week_rows.append({
            "date": date_str,
            "dt": dt,
            "profile": profile,

            "eh_count": eh_count,
            "h_count": h_count,
            "w_count": w_count,
            "c_count": c_count,

            "eh_pool": eh_pool,
            "h_pool": h_pool,
            "w_pool": w_pool,
            "c_pool": c_pool,
            "eh_h_pool": eh_h_pool,

            "legacy": legacy_str,

            "eh": sorted(eh),
            "h": sorted(h),
            "w": sorted(w),
            "c": sorted(c),

            "eh_hits": eh_hits,
            "h_hits": h_hits,
            "w_hits": w_hits,
            "c_hits": c_hits,

            "has_result": bool(main_nums),
        })

        dt += timedelta(days=1)

    # ============================================================
    # FORMAT 1 - DETAILED
    # ============================================================

    print("\n" + "=" * 100)
    print(
        f"Week Table - Detailed Pool Composition "
        f"({target_lottery_name}, universe 1-{target_max_num})"
    )
    print("=" * 100)

    print(
        f"{'Date':<20} "
        f"{'Profile':<10} "
        f"{'EH':<4} {'H':<4} {'W':<4} {'C':<4} "
        f"{'EH-Pool':<8} {'H-Pool':<8} "
        f"{'W-Pool':<8} {'C-Pool':<8} "
        f"{'EH+H-Pool':<10} "
        f"{'Legacy Hits'}"
    )

    print("-" * 100)

    for r in week_rows:

        eh_count = "-" if r["eh_count"] is None else r["eh_count"]
        h_count = "-" if r["h_count"] is None else r["h_count"]
        w_count = "-" if r["w_count"] is None else r["w_count"]
        c_count = "-" if r["c_count"] is None else r["c_count"]

        print(
            f"{r['date']:<20} "
            f"{r['profile']:<10} "
            f"{str(eh_count):<4} "
            f"{str(h_count):<4} "
            f"{str(w_count):<4} "
            f"{str(c_count):<4} "
            f"{r['eh_pool']:<8} "
            f"{r['h_pool']:<8} "
            f"{r['w_pool']:<8} "
            f"{r['c_pool']:<8} "
            f"{r['eh_h_pool']:<10} "
            f"{r['legacy']}"
        )

        if r["has_result"]:

            print(
                f"    EH Pool Numbers: {r['eh']}   "
                f"-> Hits: {r['eh_hits']}"
            )
            print(
                f"    H  Pool Numbers: {r['h']}   "
                f"-> Hits: {r['h_hits']}"
            )
            print(
                f"    W  Pool Numbers: {r['w']}   "
                f"-> Hits: {r['w_hits']}"
            )
            print(
                f"    C  Pool Numbers: {r['c']}   "
                f"-> Hits: {r['c_hits']}"
            )

        else:
            # Sunday
            print(
                f"    EH Pool Numbers: {r['eh']}   "
                f"-> Hits: N/A (no Others draw)"
            )
            print(
                f"    H  Pool Numbers: {r['h']}   "
                f"-> Hits: N/A (no Others draw)"
            )
            print(
                f"    W  Pool Numbers: {r['w']}   "
                f"-> Hits: N/A (no Others draw)"
            )
            print(
                f"    C  Pool Numbers: {r['c']}   "
                f"-> Hits: N/A (no Others draw)"
            )

        print("-" * 100)

    # ============================================================
    # FORMAT 2 - COMPACT / COUNTS ONLY
    # ============================================================

    print("\n" + "=" * 100)
    print("Week Table: Pool composition for each day in the preceding week")
    print("=" * 100)

    print(
        f"{'Date':<20} "
        f"{'Profile':<10} "
        f"{'EH':<4} {'H':<4} {'W':<4} {'C':<4} "
        f"{'EH-Pool':<8} {'H-Pool':<8} "
        f"{'W-Pool':<8} {'C-Pool':<8} "
        f"{'EH+H-Pool':<10} "
        f"{'Legacy Hits'}"
    )

    print("-" * 100)

    for r in week_rows:

        eh_count = "-" if r["eh_count"] is None else r["eh_count"]
        h_count = "-" if r["h_count"] is None else r["h_count"]
        w_count = "-" if r["w_count"] is None else r["w_count"]
        c_count = "-" if r["c_count"] is None else r["c_count"]

        print(
            f"{r['date']:<20} "
            f"{r['profile']:<10} "
            f"{str(eh_count):<4} "
            f"{str(h_count):<4} "
            f"{str(w_count):<4} "
            f"{str(c_count):<4} "
            f"{r['eh_pool']:<8} "
            f"{r['h_pool']:<8} "
            f"{r['w_pool']:<8} "
            f"{r['c_pool']:<8} "
            f"{r['eh_h_pool']:<10} "
            f"{r['legacy']}"
        )

    return week_rows


# ---------- POOL-SIZE -> MOST-COMMON HIT ANALYSIS ----------
def print_pool_size_common_hit_table(
    current_pools,
    historical_results,
    week_rows,
    cutoff_dt,
    recent_n=OUTPUT_LAST_N,
):
    """
    For each CURRENT pool size (EH/H/W/C):

      1. Look at the same historical rows shown in the "Last N ... draws analysis"
         table, strictly before cutoff_dt.
      2. Also look at result-bearing rows from the printed Week Table.
      3. If ANY historical pool (EH/H/W/C) has the same pool size, collect the
         ACTUAL hit count from that SAME historical pool.
      4. Report the most common hit count (mode).

    Example:
        Current EH pool size = 12.

        Historical matches may include:
          - EH-Pool = 12 -> collect that row's EH hit count
          - H-Pool  = 12 -> collect that row's H hit count
          - W-Pool  = 12 -> collect that row's W hit count
          - C-Pool  = 12 -> collect that row's C hit count

    Duplicate protection:
        A Saturday can appear in both the Last-N table and Week Table.
        The same (date, pool-name) observation is counted only once.

    IMPORTANT:
        Each current pool is analysed independently. Therefore the four modal
        values do NOT necessarily sum to the lottery's main-count.
    """
    current_sizes = {
        "EH": len(current_pools["EH"]),
        "H": len(current_pools["H"]),
        "W": len(current_pools["W"]),
        "C": len(current_pools["C"]),
    }

    # size -> list of observations with that historical pool size
    by_size = defaultdict(list)
    seen = set()

    def add_observation(date_str, dt, pool_name, pool_size, hit_count, source):
        if dt is not None and cutoff_dt is not None and dt >= cutoff_dt:
            return

        # Prevent double-counting the same date/pool when it appears
        # in both the Last-N table and Week Table.
        dedupe_key = (date_str, pool_name)
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)

        by_size[pool_size].append({
            "date": date_str,
            "dt": dt,
            "pool": pool_name,
            "hits": hit_count,
            "source": source,
        })

    # ------------------------------------------------------------
    # SOURCE 1: exactly the historical rows corresponding to the
    # displayed "Last N ... draws analysis" table, before target.
    # ------------------------------------------------------------
    eligible_results = [
        r for r in historical_results
        if cutoff_dt is None or r["dt"] < cutoff_dt
    ]
    recent_results = eligible_results[-recent_n:]

    for r in recent_results:
        for idx, pool_name in enumerate(POOL_NAMES):
            add_observation(
                date_str=r["date"],
                dt=r["dt"],
                pool_name=pool_name,
                pool_size=r["pools_tuple"][idx],
                hit_count=r["counts_tuple"][idx],
                source="Last-N",
            )

    # ------------------------------------------------------------
    # SOURCE 2: result-bearing rows in the printed Week Table.
    # Sunday/snapshot-only rows are intentionally ignored.
    # ------------------------------------------------------------
    for r in week_rows or []:
        if not r.get("has_result"):
            continue

        sizes = {
            "EH": r["eh_pool"],
            "H": r["h_pool"],
            "W": r["w_pool"],
            "C": r["c_pool"],
        }
        hits = {
            "EH": r["eh_count"],
            "H": r["h_count"],
            "W": r["w_count"],
            "C": r["c_count"],
        }

        for pool_name in POOL_NAMES:
            if hits[pool_name] is None:
                continue

            add_observation(
                date_str=r["date"],
                dt=r["dt"],
                pool_name=pool_name,
                pool_size=sizes[pool_name],
                hit_count=hits[pool_name],
                source="Week",
            )

    print("\n" + "=" * 125)
    print("Current Pool Size -> Historical Most-Common Actual Hit Count")
    print(
        "Matches ANY historical EH/H/W/C pool having the same pool size; "
        "duplicate date+pool observations are counted once."
    )
    print("=" * 125)

    print(
        f"{'Current':<8}"
        f"{'Pool Size':<11}"
        f"{'Matches':<9}"
        f"{'Hit-count distribution':<35}"
        f"{'Most common':<16}"
        f"{'Mode %':<10}"
        f"{'Matched historical pools'}"
    )
    print("-" * 125)

    predictions = {}

    for current_pool_name in POOL_NAMES:
        size = current_sizes[current_pool_name]
        observations = by_size.get(size, [])

        hit_freq = Counter(obs["hits"] for obs in observations)
        source_pool_freq = Counter(obs["pool"] for obs in observations)

        if not observations:
            predictions[current_pool_name] = None
            print(
                f"{current_pool_name:<8}"
                f"{size:<11}"
                f"{0:<9}"
                f"{'No matches':<35}"
                f"{'-':<16}"
                f"{'-':<10}"
                f"-"
            )
            continue

        max_freq = max(hit_freq.values())
        modes = sorted(
            hit_count
            for hit_count, freq in hit_freq.items()
            if freq == max_freq
        )

        # Do not invent a winner if there is a tie.
        if len(modes) == 1:
            mode_text = str(modes[0])
            predictions[current_pool_name] = modes[0]
        else:
            mode_text = "TIE:" + "/".join(str(x) for x in modes)
            predictions[current_pool_name] = tuple(modes)

        distribution_text = ", ".join(
            f"{hits}:{freq}"
            for hits, freq in sorted(hit_freq.items())
        )

        pool_source_text = ", ".join(
            f"{pool}:{source_pool_freq.get(pool, 0)}"
            for pool in POOL_NAMES
            if source_pool_freq.get(pool, 0) > 0
        )

        mode_pct = max_freq / len(observations)

        print(
            f"{current_pool_name:<8}"
            f"{size:<11}"
            f"{len(observations):<9}"
            f"{distribution_text:<35}"
            f"{mode_text:<16}"
            f"{mode_pct:<10.1%}"
            f"{pool_source_text}"
        )

    numeric_predictions = [
        v for v in predictions.values()
        if isinstance(v, int)
    ]

    if len(numeric_predictions) == len(POOL_NAMES):
        raw_sum = sum(numeric_predictions)
        print(
            f"\nIndependent modal profile: "
            f"{predictions['EH']}/{predictions['H']}/"
            f"{predictions['W']}/{predictions['C']} "
            f"(sum={raw_sum})"
        )
        if raw_sum != 6:
            print(
                "NOTE: These are independent pool-size modes, so the raw profile "
                "is diagnostic only and is NOT forced to sum to 6."
            )
    else:
        print(
            "\nIndependent modal profile contains at least one tie/no-match; "
            "no single combined profile is asserted."
        )

    return predictions, by_size


def print_locked_profile_winning_trajectories_from_tables(
    locked_profile,
    historical_results,
    week_rows,
    all_rows,
    max_num,
    cutoff_dt=None,
    recent_n=OUTPUT_LAST_N,
    top_n=LOCKED_TRAJECTORY_TOP_N,
    lookback_days=7,
):
    """
    Locked-profile trajectory analysis using ONLY the TWO tables printed above:

      1) "Last N <lottery> draws analysis" for the target weekday
      2) "Week Table: Pool composition for each day in the preceding week"

    The two sources are UNIONED and duplicate calendar dates are counted once.
    A target-day row that appears in both tables therefore cannot double-weight the result.

    Example LOCKED_PROFILE = (2, 1, 3, 0):
      - EH=2: among the selected table rows, use every draw whose ACTUAL EH hit
              count is exactly 2. H/W/C in that same draw do not matter.
              Count the trajectories of the winning EH numbers only.
      - H=1 : same idea independently.
      - W=3 : same idea independently.
      - C=0 : count matching draws; there are no winning C trajectories by definition.

    Pool/trajectory construction matches the Week Table:
      * fixed target universe 1..max_num
      * 7-day rolling pool before each selected calendar date
      * the draw on that date is excluded from its own pool
      * numbers outside the target universe are ignored

    A trajectory for draw date D uses daily snapshots D-7 through D inclusive.
    """
    if len(locked_profile) != 4:
        # raise ValueError("LOCKED_PROFILE must contain exactly four values: EH/H/W/C.")
        return;

    target_hits = dict(zip(POOL_NAMES, locked_profile))
    earliest_dt = min((row[1] for row in all_rows), default=None)

    # ------------------------------------------------------------------
    # Select rows from ONLY the two displayed sources.
    # Key by calendar date so overlap (especially Saturdays) is deduped.
    # ------------------------------------------------------------------
    selected_dates = {}

    visible_history = [
        r for r in historical_results
        if cutoff_dt is None or r['dt'] < cutoff_dt
    ][-recent_n:]

    for r in visible_history:
        key = r['dt'].date()
        entry = selected_dates.setdefault(key, {
            'dt': r['dt'],
            'date': r['date'],
            'sources': set(),
        })
        entry['sources'].add('LastNTargetDay')

    for r in week_rows:
        if not r.get('has_result'):
            continue
        if cutoff_dt is not None and r['dt'] >= cutoff_dt:
            continue
        key = r['dt'].date()
        entry = selected_dates.setdefault(key, {
            'dt': r['dt'],
            'date': r['date'],
            'sources': set(),
        })
        entry['sources'].add('WeekTable')

    rows_by_date = {
        dt.date(): (date_str, dt, day_abbr, all_nums, others_cell)
        for date_str, dt, day_abbr, all_nums, others_cell in all_rows
    }

    draw_records = []

    for key in sorted(selected_dates):
        selected = selected_dates[key]
        source_row = rows_by_date.get(key)
        if source_row is None:
            continue

        date_str, dt, day_abbr, _all_nums, others_cell = source_row
        if not others_cell:
            continue

        main_nums = [
            n for n in extract_main_numbers(others_cell)
            if 1 <= n <= max_num
        ]
        if not main_nums:
            continue

        # First trajectory snapshot is at D-7, and that snapshot itself needs
        # the prior 7 days. Skip only if the source history is incomplete.
        if earliest_dt is not None and dt - timedelta(days=2 * lookback_days) < earliest_dt:
            continue

        prev_dt = dt - timedelta(days=lookback_days)
        snapshots = build_daily_trajectory_snapshots(
            prev_target_dt=prev_dt,
            target_dt=dt,
            all_rows=all_rows,
            max_num=max_num,
        )

        final_pools = snapshots[-1]['pools']
        counts = {state: 0 for state in POOL_NAMES}
        winning_records = []

        for number in main_nums:
            state = pool_state(number, final_pools)
            counts[state] += 1

            states = number_trajectory(number, snapshots)
            winning_records.append({
                'number': number,
                'final_state': state,
                'trajectory': tuple(states),
                'full_states': states,
            })

        draw_records.append({
            'date': date_str,
            'dt': dt,
            'day': day_abbr,
            'counts': counts,
            'winners': winning_records,
            'sources': set(selected['sources']),
        })

    lastn_only = sum(1 for r in draw_records if r['sources'] == {'LastNTargetDay'})
    week_only = sum(1 for r in draw_records if r['sources'] == {'WeekTable'})
    overlap = sum(1 for r in draw_records if len(r['sources']) > 1)

    print("\n" + "=" * 165)
    print(
        "Locked EH/H/W/C Hit Count -> Most-Frequent WINNING Trajectory "
        "(Last-N target-day analysis + Week Table only)"
    )
    print(
        f"Locked profile: EH/H/W/C = "
        f"{locked_profile[0]}/{locked_profile[1]}/{locked_profile[2]}/{locked_profile[3]}"
    )
    print(
        f"Unique source rows analysed: {len(draw_records)}  "
        f"Last-N-target-day only={lastn_only}, Week-Table only={week_only}, overlap/deduped={overlap}"
    )
    print(
        "For each pool, ONLY that pool's actual hit count must match; "
        "the other three pool counts in the same draw are ignored."
    )
    print("=" * 165)

    print(
        f"{'Pool':<6} {'Target':<8} {'Matching draws':<15} {'Winner inst.':<14} "
        f"{'Most frequent full trajectory':<50} {'Count':<8} {'Share':<10} "
        f"{'Draw presence':<18} {'Days':<28} {'Sources'}"
    )
    print("-" * 165)

    details = {}

    for state in POOL_NAMES:
        wanted = target_hits[state]

        matching_draws = [
            rec for rec in draw_records
            if rec['counts'][state] == wanted
        ]

        matching_winners = [
            winner
            for rec in matching_draws
            for winner in rec['winners']
            if winner['final_state'] == state
        ]

        traj_freq = Counter(w['trajectory'] for w in matching_winners)
        traj_draws = defaultdict(set)
        day_freq = Counter(rec['day'] for rec in matching_draws)
        source_freq = Counter()

        for rec in matching_draws:
            for src_name in rec['sources']:
                source_freq[src_name] += 1

            seen_here = set()
            for winner in rec['winners']:
                if winner['final_state'] != state:
                    continue
                traj = winner['trajectory']
                if traj not in seen_here:
                    traj_draws[traj].add(rec['date'])
                    seen_here.add(traj)

        actual_instances = len(matching_winners)
        expected_instances = len(matching_draws) * wanted
        warning = None
        if actual_instances != expected_instances:
            warning = (
                f"Expected {expected_instances} winner instances from "
                f"{len(matching_draws)} draws x {wanted}, found {actual_instances}."
            )

        days_str = ",".join(
            f"{d}:{day_freq[d]}"
            for d in ('Mon','Tue','Wed','Thu','Fri','Sat')
            if day_freq[d]
        )
        sources_str = ",".join(
            f"{name}:{source_freq[name]}"
            for name in ('LastNTargetDay', 'WeekTable')
            if source_freq[name]
        )

        if wanted == 0:
            print(
                f"{state:<6} {wanted:<8} {len(matching_draws):<15} {0:<14} "
                f"{'N/A (zero winners by definition)':<38} {'-':<8} {'-':<10} "
                f"{'-':<18} {days_str:<28} {sources_str}"
            )
            details[state] = {
                'target_hits': wanted,
                'matching_draws': len(matching_draws),
                'winner_instances': 0,
                'trajectory_freq': Counter(),
                'day_freq': day_freq,
                'source_freq': source_freq,
                'warning': warning,
            }
            continue

        if not traj_freq:
            print(
                f"{state:<6} {wanted:<8} {len(matching_draws):<15} {actual_instances:<14} "
                f"{'No trajectory records':<38} {'-':<8} {'-':<10} "
                f"{'-':<18} {days_str:<28} {sources_str}"
            )
            details[state] = {
                'target_hits': wanted,
                'matching_draws': len(matching_draws),
                'winner_instances': actual_instances,
                'trajectory_freq': traj_freq,
                'day_freq': day_freq,
                'source_freq': source_freq,
                'warning': warning,
            }
            continue

        ranked = sorted(
            traj_freq.items(),
            key=lambda kv: (-kv[1], -len(traj_draws[kv[0]]), kv[0]),
        )
        best_sig, best_count = ranked[0]
        best_share = best_count / actual_instances if actual_instances else 0.0
        best_draw_count = len(traj_draws[best_sig])
        best_draw_share = best_draw_count / len(matching_draws) if matching_draws else 0.0

        print(
            f"{state:<6} {wanted:<8} {len(matching_draws):<15} {actual_instances:<14} "
            f"{trajectory_text(best_sig):<50} {best_count:<8} {best_share:<10.2%} "
            f"{best_draw_count}/{len(matching_draws)} ({best_draw_share:.1%})".ljust(123)
            + f" {days_str:<28} {sources_str}"
        )

        print(f"    Top winning trajectories for {state}={wanted} from the TWO tables:")
        print(
            f"    {'Full daily trajectory':<50} {'Instances':>10} {'Share':>10} "
            f"{'Draws':>9} {'Draw %':>10}"
        )
        print("    " + "-" * 82)

        for sig, count in ranked[:top_n]:
            draw_count = len(traj_draws[sig])
            share = count / actual_instances if actual_instances else 0.0
            draw_share = draw_count / len(matching_draws) if matching_draws else 0.0
            print(
                f"    {trajectory_text(sig):<50} {count:>10} {share:>9.2%} "
                f"{draw_count:>9} {draw_share:>9.2%}"
            )

        # --------------------------------------------------------------
        # SAME-DRAW TRAJECTORY COMBINATIONS / PARTNERS
        # --------------------------------------------------------------
        # This answers questions such as:
        #   "If EH=2 and W→H→EH is the most-common winning EH trajectory,
        #    what trajectory did the OTHER EH winner have in the same draw?"
        #
        # We preserve multiplicity. Therefore:
        #   (W→H→EH, W→H→EH) means BOTH winners had that trajectory.
        #   (EH, W→H→EH) means one winner had each trajectory.
        combo_freq = Counter()
        combo_dates = defaultdict(list)

        if wanted >= 2:
            for rec in matching_draws:
                sigs = sorted(
                    winner['trajectory']
                    for winner in rec['winners']
                    if winner['final_state'] == state
                )
                if len(sigs) != wanted:
                    continue

                combo = tuple(sigs)
                combo_freq[combo] += 1
                combo_dates[combo].append(rec['date'])

            if combo_freq:
                ranked_combos = sorted(
                    combo_freq.items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )

                print(
                    f"    Same-draw winning trajectory combinations for "
                    f"{state}={wanted}:"
                )
                print(
                    f"    {'Full-trajectory combination':<110} "
                    f"{'Draws':>7} {'Share':>9}  Dates"
                )
                print("    " + "-" * 120)

                for combo, combo_count in ranked_combos[:top_n]:
                    combo_text = " + ".join(trajectory_text(t) for t in combo)
                    combo_share = (
                        combo_count / len(matching_draws)
                        if matching_draws else 0.0
                    )
                    dates_text = ", ".join(combo_dates[combo])
                    print(
                        f"    {combo_text:<110} "
                        f"{combo_count:>7} {combo_share:>8.2%}  {dates_text}"
                    )

                # Anchor-partner analysis: use the most-frequent individual
                # winning trajectory as the first-slot/anchor trajectory.
                # Remove ONE occurrence of the anchor from each matching draw;
                # the remaining trajectory/trajectories are its co-winner(s).
                anchor = best_sig
                partner_freq = Counter()
                partner_draws = defaultdict(set)
                anchor_draw_dates = []
                anchor_draw_count = 0
                both_anchor_draws = 0

                for rec in matching_draws:
                    sigs = [
                        winner['trajectory']
                        for winner in rec['winners']
                        if winner['final_state'] == state
                    ]
                    if len(sigs) != wanted or anchor not in sigs:
                        continue

                    anchor_draw_count += 1
                    anchor_draw_dates.append(rec['date'])

                    remaining = list(sigs)
                    remaining.remove(anchor)  # remove ONE anchor occurrence

                    if anchor in remaining:
                        both_anchor_draws += 1

                    for partner in remaining:
                        partner_freq[partner] += 1
                        partner_draws[partner].add(rec['date'])

                if anchor_draw_count:
                    print(
                        f"    Partner analysis when anchor trajectory "
                        f"'{trajectory_text(anchor)}' is present:"
                    )
                    print(
                        f"      Anchor appeared in {anchor_draw_count}/"
                        f"{len(matching_draws)} matching draws "
                        f"({anchor_draw_count/len(matching_draws):.1%})."
                    )
                    print(
                        f"      Draws where another winner ALSO had "
                        f"'{trajectory_text(anchor)}': {both_anchor_draws}"
                    )
                    print(
                        f"      {'Partner full trajectory':<60} "
                        f"{'Instances':>10} {'Draws':>8} {'Draw %':>9}  Dates"
                    )
                    print("      " + "-" * 110)

                    ranked_partners = sorted(
                        partner_freq.items(),
                        key=lambda kv: (
                            -kv[1],
                            -len(partner_draws[kv[0]]),
                            kv[0],
                        ),
                    )

                    for partner, partner_count in ranked_partners[:top_n]:
                        draw_count = len(partner_draws[partner])
                        draw_share = draw_count / anchor_draw_count
                        dates_text = ", ".join(sorted(partner_draws[partner]))
                        print(
                            f"      {trajectory_text(partner):<60} "
                            f"{partner_count:>10} {draw_count:>8} "
                            f"{draw_share:>8.2%}  {dates_text}"
                        )

        if warning:
            print(f"    WARNING: {warning}")

        details[state] = {
            'target_hits': wanted,
            'matching_draws': len(matching_draws),
            'winner_instances': actual_instances,
            'trajectory_freq': traj_freq,
            'trajectory_draws': {k: len(v) for k, v in traj_draws.items()},
            'day_freq': day_freq,
            'source_freq': source_freq,
            'warning': warning,
        }

    return details



# ---------- ALL-HISTORY LOCKED-PROFILE TRAJECTORY ANALYSIS ----------

def _locked_scope_accepts(
    day_abbr,
    raw_main_nums,
    scope,
    target_day_abbr,
    target_lottery_name,
    target_max_num,
    target_main_count,
):
    """
    Decide whether one historical Others draw belongs to the requested scope.

    Notes:
    - "same_main_count" compares the RAW number of main balls before universe
      filtering. This prevents a 7-ball draw from looking like a 6-ball draw
      merely because one ball lies outside the target universe.
    - "same_game" uses LOTTERY_CONFIG, so Weekday Windfall includes Mon/Wed/Fri.
    """
    if scope == "same_weekday":
        return day_abbr == target_day_abbr

    if scope == "same_game":
        cfg = LOTTERY_CONFIG.get(day_abbr)
        if not cfg:
            return False
        name, native_max, native_main_count = cfg
        return (
            name == target_lottery_name
            and native_max == target_max_num
            and native_main_count == target_main_count
        )

    if scope == "same_main_count":
        return len(raw_main_nums) == target_main_count

    if scope == "all_others":
        return True

    raise ValueError(
        "Unknown locked-analysis scope. Use one of: "
        "'same_weekday', 'same_game', 'same_main_count', 'all_others'."
    )


def collect_all_history_locked_records(
    all_rows,
    max_num,
    cutoff_dt,
    target_day_abbr,
    target_lottery_name,
    target_main_count,
    scope="same_main_count",
    lookback_days=7,
    require_target_main_count=False,
):
    """
    Build non-cheating historical draw records for locked-profile analysis.

    Every record uses:
      * fixed target universe 1..max_num
      * a rolling [D-lookback_days, D) pool for final EH/H/W/C classification
      * trajectory snapshots D-lookback_days through D inclusive
      * only rows strictly before cutoff_dt

    require_target_main_count=True is used for EXACT profile analysis so that
    a 6-ball locked profile can never accidentally match a 7-ball draw after
    universe filtering.
    """
    if not all_rows:
        return []

    earliest_dt = min(row[1] for row in all_rows)
    records = []

    for date_str, dt, day_abbr, _all_nums, others_cell in all_rows:
        if not others_cell:
            continue
        if cutoff_dt is not None and dt >= cutoff_dt:
            continue

        raw_main_nums = extract_main_numbers(others_cell)
        if not raw_main_nums:
            continue

        if not _locked_scope_accepts(
            day_abbr=day_abbr,
            raw_main_nums=raw_main_nums,
            scope=scope,
            target_day_abbr=target_day_abbr,
            target_lottery_name=target_lottery_name,
            target_max_num=max_num,
            target_main_count=target_main_count,
        ):
            continue

        if require_target_main_count and len(raw_main_nums) != target_main_count:
            continue

        main_nums = [n for n in raw_main_nums if 1 <= n <= max_num]

        # For exact-profile analysis, every target main ball must live inside the
        # target universe. Otherwise the EH/H/W/C tuple would not sum correctly.
        if require_target_main_count and len(main_nums) != target_main_count:
            continue

        # The first trajectory snapshot is D-lookback_days and itself needs a
        # complete lookback window. Skip partial-history cases.
        if dt - timedelta(days=2 * lookback_days) < earliest_dt:
            continue

        prev_dt = dt - timedelta(days=lookback_days)
        snapshots = build_daily_trajectory_snapshots(
            prev_target_dt=prev_dt,
            target_dt=dt,
            all_rows=all_rows,
            max_num=max_num,
        )
        final_pools = snapshots[-1]["pools"]

        counts = {state: 0 for state in POOL_NAMES}
        winners = []

        for number in main_nums:
            state = pool_state(number, final_pools)
            counts[state] += 1
            states = number_trajectory(number, snapshots)
            winners.append({
                "number": number,
                "final_state": state,
                "trajectory": tuple(states),
                "full_states": states,
            })

        # Candidate exposure is required to distinguish "common among winners"
        # from "actually higher hit-rate conditional on this profile/count".
        candidates = []
        for number in range(1, max_num + 1):
            states = number_trajectory(number, snapshots)
            candidates.append({
                "number": number,
                "final_state": states[-1],
                "trajectory": tuple(states),
                "full_states": states,
            })

        records.append({
            "date": date_str,
            "dt": dt,
            "day": day_abbr,
            "raw_main_count": len(raw_main_nums),
            "main_nums": main_nums,
            "counts": counts,
            "counts_tuple": tuple(counts[state] for state in POOL_NAMES),
            "winners": winners,
            "candidates": candidates,
        })

    return records


def _print_conditional_state_analysis(
    title,
    state,
    wanted,
    matching_draws,
    current_groups,
    top_n=LOCKED_TRAJECTORY_TOP_N,
    min_candidates=LOCKED_CONDITIONAL_MIN_CANDIDATES,
):
    """
    Print one pool's trajectory behaviour inside an already-conditioned set of
    historical draws, including candidate-normalized rates and same-draw combos.
    """
    print("\n" + "-" * 150)
    print(title)
    print("-" * 150)

    if not matching_draws:
        print("No matching historical draws.")
        return

    if wanted == 0:
        print(
            f"{state} target hit count is 0. "
            f"Matching draws: {len(matching_draws)}. "
            f"No winning {state} trajectories exist by definition."
        )
        return

    winner_freq = Counter()
    winner_draws = defaultdict(set)
    candidate_freq = Counter()
    combo_freq = Counter()
    combo_dates = defaultdict(list)

    total_candidates = 0
    total_winners = 0

    for rec in matching_draws:
        state_candidates = [
            c for c in rec["candidates"]
            if c["final_state"] == state
        ]
        state_winners = [
            w for w in rec["winners"]
            if w["final_state"] == state
        ]

        total_candidates += len(state_candidates)
        total_winners += len(state_winners)

        for c in state_candidates:
            candidate_freq[c["trajectory"]] += 1

        seen_here = set()
        for w in state_winners:
            traj = w["trajectory"]
            winner_freq[traj] += 1
            if traj not in seen_here:
                winner_draws[traj].add(rec["date"])
                seen_here.add(traj)

        if wanted >= 2 and len(state_winners) == wanted:
            combo = tuple(sorted(w["trajectory"] for w in state_winners))
            combo_freq[combo] += 1
            combo_dates[combo].append(rec["date"])

    baseline = (
        total_winners / total_candidates
        if total_candidates else 0.0
    )

    print(
        f"Matching draws={len(matching_draws)} | "
        f"winner instances={total_winners} | "
        f"candidate instances={total_candidates} | "
        f"conditional pool baseline={baseline:.2%}"
    )

    rows = []
    all_sigs = set(candidate_freq) | set(winner_freq)

    for sig in all_sigs:
        cand_n = candidate_freq[sig]
        wins = winner_freq[sig]
        if cand_n < min_candidates:
            continue
        rate = wins / cand_n if cand_n else 0.0
        lift = rate / baseline if baseline else 0.0
        draws_n = len(winner_draws.get(sig, set()))
        win_share = wins / total_winners if total_winners else 0.0
        draw_presence = draws_n / len(matching_draws) if matching_draws else 0.0
        rows.append((
            sig, cand_n, wins, rate, lift,
            win_share, draws_n, draw_presence
        ))

    # Winner count first so the table answers "what actually wins", then use
    # candidate-normalized rate/lift to distinguish prevalence from efficiency.
    rows.sort(
        key=lambda x: (
            -x[2],      # winners
            -x[3],      # conditional hit rate
            -x[1],      # candidate sample
            x[0],
        )
    )

    print(
        f"  {'Full daily trajectory':<50} {'Cand N':>8} {'Wins':>7} "
        f"{'Cond Rate':>10} {'Lift':>8} {'Win Share':>10} "
        f"{'Draws':>7} {'Draw %':>9}"
    )
    print("  " + "-" * 107)

    for sig, cand_n, wins, rate, lift, win_share, draws_n, draw_presence in rows[:top_n]:
        print(
            f"  {trajectory_text(sig):<50} {cand_n:>8} {wins:>7} "
            f"{rate:>9.2%} {lift:>7.2f}x {win_share:>9.2%} "
            f"{draws_n:>7} {draw_presence:>8.2%}"
        )

    if wanted >= 2 and combo_freq:
        print(
            f"\n  Same-draw winning trajectory combinations for {state}={wanted}:"
        )
        print(
            f"  {'Full-trajectory combination':<125} "
            f"{'Draws':>7} {'Share':>9}  Dates"
        )
        print("  " + "-" * 145)

        ranked_combos = sorted(
            combo_freq.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )

        for combo, count in ranked_combos[:top_n]:
            share = count / len(matching_draws)
            dates = ", ".join(combo_dates[combo])
            print(
                f"  {' + '.join(trajectory_text(t) for t in combo):<125} "
                f"{count:>7} {share:>8.2%}  {dates}"
            )

    # Current target mapping using ONLY the conditional evidence above.
    current_state_groups = {
        sig: nums
        for (final_state, sig), nums in current_groups.items()
        if final_state == state
    }

    if current_state_groups:
        print(
            f"\n  Current {state} candidates mapped to this conditional history:"
        )
        print(
            f"  {'Full daily trajectory':<50} {'Current numbers':<34} "
            f"{'Cond N':>8} {'Wins':>7} {'Rate':>9} {'Lift':>8}"
        )
        print("  " + "-" * 108)

        current_rows = []
        for sig, nums in current_state_groups.items():
            cand_n = candidate_freq.get(sig, 0)
            wins = winner_freq.get(sig, 0)
            rate = wins / cand_n if cand_n else 0.0
            lift = rate / baseline if baseline else 0.0
            current_rows.append((
                sig, sorted(nums), cand_n, wins, rate, lift
            ))

        current_rows.sort(
            key=lambda x: (
                -x[4],  # conditional rate
                -x[3],  # wins
                -x[2],  # sample
                x[0],
            )
        )

        for sig, nums, cand_n, wins, rate, lift in current_rows:
            print(
                f"  {trajectory_text(sig):<50} {str(nums):<34} "
                f"{cand_n:>8} {wins:>7} {rate:>8.2%} {lift:>7.2f}x"
            )


def _print_matching_draw_details(
    matching_draws,
    locked_profile,
    limit=None,
):
    """Print exact-profile draw-by-draw winners and their trajectories."""
    if not matching_draws:
        return

    rows = matching_draws if limit is None else matching_draws[-limit:]

    print("\nMatching exact-profile draw details:")
    print("-" * 150)

    for rec in rows:
        state_parts = []
        for state in POOL_NAMES:
            winners = [
                w for w in rec["winners"]
                if w["final_state"] == state
            ]
            if not winners:
                continue
            text = ", ".join(
                f"{w['number']}({trajectory_text(w['trajectory'])})"
                for w in winners
            )
            state_parts.append(f"{state}: {text}")

        print(
            f"{rec['date']:<20} "
            f"profile={'/'.join(str(x) for x in locked_profile):<8}  "
            + " | ".join(state_parts)
        )


def print_all_history_locked_profile_analysis(
    locked_profile,
    all_rows,
    max_num,
    main_count,
    lottery_name,
    target_day_abbr,
    target_dt,
    current_prev_dt,
    exact_scope=LOCKED_EXACT_SCOPE,
    independent_scope=LOCKED_INDEPENDENT_SCOPE,
    top_n=LOCKED_TRAJECTORY_TOP_N,
    lookback_days=7,
):
    """
    Full-history, no-leakage locked-profile trajectory analysis.

    TWO DISTINCT QUESTIONS are answered:

    A) EXACT PROFILE
       Use only historical draws whose complete EH/H/W/C tuple equals the
       locked profile. This is the correct test for:
         "If I knew 1/0/5/0 exactly, what trajectories tended to win?"

    B) INDEPENDENT HIT COUNTS
       For EH=1, ignore H/W/C; for W=5, ignore EH/H/C; etc.
       This preserves the user's original independent conditioning rule.

    Both sections use candidate exposure as the denominator, so a trajectory is
    not called strong merely because it is common among winners.
    """
    if not isinstance(locked_profile, (tuple, list)) or len(locked_profile) != 4:
        print(
            "\nAll-history locked-profile analysis skipped: "
            "LOCKED_PROFILE must be a 4-item tuple/list."
        )
        return {}

    locked_profile = tuple(int(x) for x in locked_profile)

    if sum(locked_profile) != main_count:
        print(
            "\nWARNING: LOCKED_PROFILE sums to "
            f"{sum(locked_profile)}, but target main_count={main_count}. "
            "Exact-profile section is skipped because those cannot describe "
            "the same target draw."
        )
        run_exact = False
    else:
        run_exact = PRINT_ALL_HISTORY_EXACT_PROFILE

    # Current target groups are computed once, strictly from pre-target data.
    current_snapshots = build_daily_trajectory_snapshots(
        prev_target_dt=current_prev_dt,
        target_dt=target_dt,
        all_rows=all_rows,
        max_num=max_num,
    )
    current_groups = defaultdict(list)

    for number in range(1, max_num + 1):
        states = number_trajectory(number, current_snapshots)
        current_groups[(states[-1], tuple(states))].append(number)

    results = {}

    # ================================================================
    # A) EXACT PROFILE ACROSS ALL ELIGIBLE HISTORY
    # ================================================================
    if run_exact:
        exact_records = collect_all_history_locked_records(
            all_rows=all_rows,
            max_num=max_num,
            cutoff_dt=target_dt,
            target_day_abbr=target_day_abbr,
            target_lottery_name=lottery_name,
            target_main_count=main_count,
            scope=exact_scope,
            lookback_days=lookback_days,
            require_target_main_count=True,
        )

        exact_matches = [
            rec for rec in exact_records
            if rec["counts_tuple"] == locked_profile
        ]

        day_freq = Counter(rec["day"] for rec in exact_matches)

        print("\n" + "=" * 170)
        print("ALL-HISTORY EXACT LOCKED-PROFILE TRAJECTORY ANALYSIS")
        print(
            f"Locked profile = "
            f"{locked_profile[0]}/{locked_profile[1]}/"
            f"{locked_profile[2]}/{locked_profile[3]}  |  "
            f"scope={exact_scope}  |  "
            f"STRICTLY BEFORE {target_dt.strftime('%a %d-%b-%Y')}"
        )
        print(
            "This section requires the COMPLETE historical profile to equal the "
            "locked profile. It is different from independent EH=1 / W=5 analysis."
        )
        print("=" * 170)

        days_text = ", ".join(
            f"{d}:{day_freq[d]}"
            for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
            if day_freq[d]
        )

        print(
            f"Eligible historical draws={len(exact_records)} | "
            f"Exact-profile matches={len(exact_matches)} | "
            f"Days={days_text or '-'}"
        )

        if exact_matches:
            for idx, state in enumerate(POOL_NAMES):
                _print_conditional_state_analysis(
                    title=(
                        f"EXACT PROFILE "
                        f"{'/'.join(str(x) for x in locked_profile)} "
                        f"-> {state}={locked_profile[idx]}"
                    ),
                    state=state,
                    wanted=locked_profile[idx],
                    matching_draws=exact_matches,
                    current_groups=current_groups,
                    top_n=top_n,
                )

            if PRINT_LOCKED_MATCHING_DRAW_DETAILS:
                _print_matching_draw_details(
                    matching_draws=exact_matches,
                    locked_profile=locked_profile,
                )

        results["exact"] = {
            "scope": exact_scope,
            "eligible_draws": len(exact_records),
            "matching_draws": exact_matches,
        }

    # ================================================================
    # B) INDEPENDENT POOL HIT COUNTS ACROSS ALL ELIGIBLE HISTORY
    # ================================================================
    if PRINT_ALL_HISTORY_INDEPENDENT_COUNTS:
        independent_records = collect_all_history_locked_records(
            all_rows=all_rows,
            max_num=max_num,
            cutoff_dt=target_dt,
            target_day_abbr=target_day_abbr,
            target_lottery_name=lottery_name,
            target_main_count=main_count,
            scope=independent_scope,
            lookback_days=lookback_days,
            require_target_main_count=(
                independent_scope == "same_main_count"
            ),
        )

        print("\n" + "=" * 170)
        print("ALL-HISTORY INDEPENDENT LOCKED-COUNT TRAJECTORY ANALYSIS")
        print(
            f"Locked counts = "
            f"EH={locked_profile[0]}, H={locked_profile[1]}, "
            f"W={locked_profile[2]}, C={locked_profile[3]}  |  "
            f"scope={independent_scope}  |  "
            f"STRICTLY BEFORE {target_dt.strftime('%a %d-%b-%Y')}"
        )
        print(
            "For each pool, ONLY that pool's actual hit count must match; "
            "the other three pool counts are ignored."
        )
        print("=" * 170)

        independent_details = {}

        for idx, state in enumerate(POOL_NAMES):
            wanted = locked_profile[idx]
            matching = [
                rec for rec in independent_records
                if rec["counts"][state] == wanted
            ]

            day_freq = Counter(rec["day"] for rec in matching)
            days_text = ", ".join(
                f"{d}:{day_freq[d]}"
                for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
                if day_freq[d]
            )

            print(
                f"\n{state}={wanted}: matching draws={len(matching)} "
                f"| days={days_text or '-'}"
            )

            _print_conditional_state_analysis(
                title=f"INDEPENDENT {state}={wanted}",
                state=state,
                wanted=wanted,
                matching_draws=matching,
                current_groups=current_groups,
                top_n=top_n,
            )

            independent_details[state] = matching

        results["independent"] = {
            "scope": independent_scope,
            "eligible_draws": len(independent_records),
            "matching_draws": independent_details,
        }

    return results



# ---------- THURSDAY POWERBALL-BALL (1-20) ANALYSIS ----------
def collect_powerball_ball_history(all_rows, cutoff_dt, pb_max=POWERBALL_BALL_MAX):
    """
    Collect Thursday Powerball-ball results strictly BEFORE cutoff_dt.

    The Powerball is read from the second bracket of the Thursday 'Others' cell.
    The target date is excluded, so this remains safe for historical backtests.
    """
    history = []

    for date_str, dt, day_abbr, _all_nums, others_cell in all_rows:
        if day_abbr != "Thu" or not others_cell:
            continue
        if cutoff_dt is not None and dt >= cutoff_dt:
            continue

        pb = extract_powerball_ball(others_cell, pb_max=pb_max)
        if pb is None:
            continue

        history.append({
            "date": date_str,
            "dt": dt,
            "pb": pb,
        })

    history.sort(key=lambda r: r["dt"])
    return history


def _powerball_ball_candidate_metrics(
    history,
    pb_max=POWERBALL_BALL_MAX,
    decay=POWERBALL_BALL_DECAY,
    prior_strength=POWERBALL_BALL_PRIOR_STRENGTH,
):
    """
    Score all 1..pb_max Powerball candidates.

    IMPORTANT:
    - Score is a ranking score, NOT a calibrated probability.
    - Every individual Powerball has the same theoretical draw probability 1/pb_max.
    - Bayesian shrinkage toward that baseline stops tiny recent samples from
      producing extreme scores.
    - Gap is reported descriptively and is NOT rewarded merely for being "due".
    """
    if not history:
        return []

    values = [r["pb"] for r in history]
    baseline = 1.0 / pb_max

    windows = (5, 10, 20, 40)
    window_weights = {
        5: 0.20,
        10: 0.30,
        20: 0.25,
        40: 0.10,
    }
    decay_weight = 0.15

    # Most-recent observation gets weight 1.0.
    decay_weights = [
        decay ** (len(values) - 1 - i)
        for i in range(len(values))
    ]
    total_decay_weight = sum(decay_weights)

    rows = []

    for number in range(1, pb_max + 1):
        counts = {}
        smoothed = {}

        for window in windows:
            sample = values[-window:]
            count = sample.count(number)
            counts[window] = count

            n = len(sample)
            # Beta/binomial-style shrinkage around the 1/pb_max baseline.
            smoothed[window] = (
                count + prior_strength * baseline
            ) / (
                n + prior_strength
            )

        weighted_hits = sum(
            weight
            for pb, weight in zip(values, decay_weights)
            if pb == number
        )
        decayed_rate = (
            weighted_hits / total_decay_weight
            if total_decay_weight else baseline
        )
        # Shrink the decayed rate as well. Effective recent sample is approximated
        # by the total exponential weight.
        decayed_smoothed = (
            weighted_hits + prior_strength * baseline
        ) / (
            total_decay_weight + prior_strength
        )

        # Draw gap: 0 means it appeared in the most recent Thursday draw.
        gap = None
        for age, pb in enumerate(reversed(values)):
            if pb == number:
                gap = age
                break
        if gap is None:
            gap = len(values)

        score = (
            window_weights[5] * smoothed[5]
            + window_weights[10] * smoothed[10]
            + window_weights[20] * smoothed[20]
            + window_weights[40] * smoothed[40]
            + decay_weight * decayed_smoothed
        )

        rows.append({
            "number": number,
            "c5": counts[5],
            "c10": counts[10],
            "c20": counts[20],
            "c40": counts[40],
            "gap": gap,
            "decayed_rate": decayed_rate,
            "score": score,
        })

    # Do not use "long overdue" as a positive tie-breaker. For deterministic ties,
    # prefer stronger 20-draw evidence, then 40-draw evidence, then lower number.
    rows.sort(
        key=lambda r: (
            -r["score"],
            -r["c20"],
            -r["c40"],
            r["number"],
        )
    )

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def backtest_powerball_ball_ranking(
    history,
    pb_max=POWERBALL_BALL_MAX,
    backtest_n=POWERBALL_BALL_BACKTEST_N,
):
    """
    Walk-forward test the exact ranking rule.

    For each historical Thursday tested, only EARLIER Powerball-ball values are
    used to rank 1..20. Reports top-1 / top-3 / top-5 capture rates.
    """
    min_history = 20
    if len(history) <= min_history:
        return None

    first_idx = max(min_history, len(history) - backtest_n)

    tested = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    for idx in range(first_idx, len(history)):
        prior = history[:idx]
        actual = history[idx]["pb"]

        ranked = _powerball_ball_candidate_metrics(
            prior,
            pb_max=pb_max,
        )
        ranked_nums = [r["number"] for r in ranked]

        tested += 1
        if actual in ranked_nums[:1]:
            top1_hits += 1
        if actual in ranked_nums[:3]:
            top3_hits += 1
        if actual in ranked_nums[:5]:
            top5_hits += 1

    if tested == 0:
        return None

    return {
        "tested": tested,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "top5_hits": top5_hits,
        "top1_rate": top1_hits / tested,
        "top3_rate": top3_hits / tested,
        "top5_rate": top5_hits / tested,
    }


def print_powerball_ball_prediction(
    target_dt,
    all_rows,
    pb_max=POWERBALL_BALL_MAX,
    top_n=POWERBALL_BALL_TOP_N,
):
    """
    Print a dedicated 1..20 Powerball-ball analysis.

    This function is intended to be called ONLY for a Thursday Powerball target.
    """
    history = collect_powerball_ball_history(
        all_rows=all_rows,
        cutoff_dt=target_dt,
        pb_max=pb_max,
    )

    print("\n" + "=" * 120)
    print(
        f"THURSDAY POWERBALL-BALL PREDICTION (1-{pb_max}) - "
        f"{target_dt.strftime('%a %d-%b-%Y')}"
    )
    print("=" * 120)
    print(
        "This section uses ONLY historical Thursday Powerball-ball values from the "
        "second bracket and excludes the target date."
    )
    print(
        f"Theoretical probability for every individual Powerball = "
        f"1/{pb_max} = {1/pb_max:.2%}."
    )

    if len(history) < 20:
        print(
            f"Only {len(history)} valid historical Powerball-ball results were found. "
            "At least 20 are required for this ranking."
        )
        return []

    recent = history[-20:]
    recent_text = ", ".join(
        f"{r['dt'].strftime('%d-%b')}:{r['pb']}"
        for r in recent
    )
    print(f"Historical Powerball-ball results available: {len(history)}")
    print(f"Most recent 20: {recent_text}")

    ranked = _powerball_ball_candidate_metrics(
        history,
        pb_max=pb_max,
    )

    print("\nCandidate ranking:")
    print(
        f"{'Rank':<6} {'PB':<4} {'Last5':>7} {'Last10':>8} "
        f"{'Last20':>8} {'Last40':>8} {'Gap':>6} "
        f"{'Decay raw':>11} {'Rank score':>12}"
    )
    print("-" * 90)

    for row in ranked:
        print(
            f"{row['rank']:<6} {row['number']:<4} "
            f"{row['c5']:>7} {row['c10']:>8} "
            f"{row['c20']:>8} {row['c40']:>8} "
            f"{row['gap']:>6} "
            f"{row['decayed_rate']:>10.2%} "
            f"{row['score']:>11.4%}"
        )

    shortlist = [r["number"] for r in ranked[:top_n]]
    print(
        f"\nModel shortlist (top {top_n}, ranking signal only): {shortlist}"
    )
    print(
        f"Top-ranked Powerball by this historical ranking model: {ranked[0]['number']}"
    )
    print(
        "NOTE: 'Rank score' is NOT the true probability of that Powerball. "
        "A fair 1-20 Powerball remains 5% per number."
    )

    bt = backtest_powerball_ball_ranking(
        history=history,
        pb_max=pb_max,
        backtest_n=POWERBALL_BALL_BACKTEST_N,
    )

    if bt:
        print("\nWalk-forward sanity check (no future result used in each prediction):")
        print(
            f"  Draws tested: {bt['tested']}"
        )
        print(
            f"  Top 1: {bt['top1_hits']}/{bt['tested']} = {bt['top1_rate']:.1%} "
            f"(random coverage baseline 5.0%)"
        )
        print(
            f"  Top 3: {bt['top3_hits']}/{bt['tested']} = {bt['top3_rate']:.1%} "
            f"(random coverage baseline 15.0%)"
        )
        print(
            f"  Top 5: {bt['top5_hits']}/{bt['tested']} = {bt['top5_rate']:.1%} "
            f"(random coverage baseline 25.0%)"
        )

    return ranked


# ---------- PREDICTION FUNCTIONS ----------
def predict_counts_proportional(pools, total_numbers, main_count):
    sizes = pools
    raw = [size * main_count / total_numbers for size in sizes]
    return round_to_sum(raw, main_count)

def predict_counts_mode(pools, history, total_numbers, main_count, k=K_NEIGHBORS):
    if not history:
        return predict_counts_proportional(pools, total_numbers, main_count)
    target = pools
    distances = []
    for h_pools, h_counts in history:
        dist = sum((target[i] - h_pools[i])**2 for i in range(4)) ** 0.5
        distances.append((dist, h_counts))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    freq = Counter(cnt for _, cnt in neighbors)
    max_freq = max(freq.values())
    modes = [cnt for cnt, f in freq.items() if f == max_freq]
    if len(modes) == 1:
        return modes[0]
    else:
        best_mode = None
        best_avg_dist = float('inf')
        for mode in modes:
            dists = [d for d, cnt in neighbors if cnt == mode]
            avg = sum(dists) / len(dists)
            if avg < best_avg_dist:
                best_avg_dist = avg
                best_mode = mode
        return best_mode

def predict_counts_time_weighted_rate(pools, history, total_numbers, main_count, decay=0.95):
    if not history:
        return predict_counts_proportional(pools, total_numbers, main_count)
    total_pool = [0.0] * 4
    total_count = [0.0] * 4
    weight_sum = 0.0
    for i, (h_pools, h_counts) in enumerate(history):
        weight = decay ** (len(history) - 1 - i)
        for j in range(4):
            total_pool[j] += weight * h_pools[j]
            total_count[j] += weight * h_counts[j]
        weight_sum += weight
    if weight_sum == 0:
        return predict_counts_proportional(pools, total_numbers, main_count)
    rates = [total_count[j] / total_pool[j] if total_pool[j] > 0 else 0.0 for j in range(4)]
    raw = [pools[i] * rates[i] for i in range(4)]
    return round_to_sum(raw, main_count)

def predict_counts_ensemble(pools, history, total_numbers, main_count, k=K_NEIGHBORS, decay=0.95):
    raw_prop = [pools[i] * main_count / total_numbers for i in range(4)]
    if history:
        total_pool = [0.0] * 4
        total_count = [0.0] * 4
        for i, (h_pools, h_counts) in enumerate(history):
            weight = decay ** (len(history) - 1 - i)
            for j in range(4):
                total_pool[j] += weight * h_pools[j]
                total_count[j] += weight * h_counts[j]
        rates = [total_count[j] / total_pool[j] if total_pool[j] > 0 else 0.0 for j in range(4)]
        raw_rate = [pools[i] * rates[i] for i in range(4)]
    else:
        raw_rate = raw_prop
    if history:
        target = pools
        distances = []
        for h_pools, h_counts in history:
            dist = sum((target[i] - h_pools[i])**2 for i in range(4)) ** 0.5
            distances.append((dist, h_counts))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        eps = 1e-6
        inv_dists = [1.0 / (d + eps) for d, _ in neighbors]
        total_inv = sum(inv_dists)
        raw_mode = [0.0] * 4
        for (d, cnt), w in zip(neighbors, inv_dists):
            for j in range(4):
                raw_mode[j] += w * cnt[j] / total_inv
    else:
        raw_mode = raw_prop
    raw_ens = [(raw_prop[i] + raw_rate[i] + raw_mode[i]) / 3.0 for i in range(4)]
    return round_to_sum(raw_ens, main_count)

# ---------- PROCESSING FUNCTION FOR A GIVEN LOTTERY ----------
def process_lottery(day_abbr, draws, all_rows, draws_by_day, max_num, main_count, lottery_name,
                    future_date=None):
    """
    Process one lottery type.
    If future_date is None, predict the next occurrence.
    If future_date is given, predict for that exact date.
    """
    if len(draws) < 2:
        print(f"\n{lottery_name} ({day_abbr}): Not enough draws (need at least 2). Skipping.")
        return

    print(f"\n{'='*80}")
    print(f"Processing {lottery_name} ({day_abbr}) – numbers 1–{max_num}, {main_count} main numbers")
    print('='*80)

    # Build historical results for this day only
    results = []
    for i in range(1, len(draws)):
        target_date_str, target_dt, target_main = draws[i]
        prev_date_str, prev_dt, prev_main = draws[i-1]

        window_nums = []
        for date_str, dt, day_abbr2, all_nums, _ in all_rows:
            if prev_dt <= dt < target_dt:
                valid_nums = [n for n in all_nums if 1 <= n <= max_num]
                window_nums.extend(valid_nums)

        counter = Counter(window_nums)
        eh = {n for n, cnt in counter.items() if cnt >= 4}
        h  = {n for n, cnt in counter.items() if cnt == 3}
        w  = {n for n, cnt in counter.items() if 1 <= cnt <= 2}
        c  = {n for n in range(1, max_num+1) if counter[n] == 0}

        eh_pool = len(eh)
        h_pool = len(h)
        w_pool = len(w)
        c_pool = len(c)

        counts = {'EH':0, 'H':0, 'W':0, 'C':0}
        for n in target_main:
            if n in eh:    counts['EH'] += 1
            elif n in h:   counts['H'] += 1
            elif n in w:   counts['W'] += 1
            else:          counts['C'] += 1

        w_count = counts['W']
        profile = "Breadth" if w_count >= (main_count // 2 + 1) else "Depth"

        legacy_hits = [n for n in target_main if n in prev_main]

        results.append({
            'date': target_date_str,
            'dt': target_dt,
            'profile': profile,
            'counts_tuple': (counts['EH'], counts['H'], counts['W'], counts['C']),
            'pools_tuple': (eh_pool, h_pool, w_pool, c_pool),
            'eh_h_pool': eh_pool + h_pool,
            'legacy': legacy_hits
        })

    # Print historical table
    n = min(OUTPUT_LAST_N, len(results))
    print(f"\nLast {n} {lottery_name} draws analysis:\n")
    print(f"{'Date':<20} {'Profile':<10} {'EH':<4} {'H':<4} {'W':<4} {'C':<4} "
          f"{'EH-Pool':<8} {'H-Pool':<8} {'W-Pool':<8} {'C-Pool':<8} "
          f"{'EH+H-Pool':<10} {'Legacy Hits'}")
    print("-" * 90)
    for r in results[-n:]:
        eh, h, w, c = r['counts_tuple']
        peh, ph, pw, pc = r['pools_tuple']
        legacy_str = str(r['legacy']) if r['legacy'] else "None"
        print(f"{r['date']:<20} {r['profile']:<10} {eh:<4} {h:<4} {w:<4} {c:<4} "
              f"{peh:<8} {ph:<8} {pw:<8} {pc:<8} "
              f"{r['eh_h_pool']:<10} {legacy_str}")

    # Frequency distribution
    print("\n" + "="*80)
    print(f"Outcome frequency distribution (all {lottery_name} draws):")
    print("="*80)
    outcome_freq = Counter(r['counts_tuple'] for r in results)
    total_draws = len(results)
    for outcome, freq in outcome_freq.most_common():
        print(f"  {outcome}  ->  {freq} times  ({freq/total_draws:.1%})")

    # ---------- BACKTEST ----------
    if RUN_BACKTEST and len(results) >= BACKTEST_N:
        print("\n" + "="*100)
        print(f"Backtest comparison over last {BACKTEST_N} draws ({lottery_name}):")
        print("="*100)
        print(f"{'Date':<20} {'Pool Sizes':<15} {'Prop Pred':<15} {'Mode Pred':<15} {'TW-Rate Pred':<15} {'Ensemble Pred':<15} {'Actual':<12} {'Prop Err':<8} {'Mode Err':<8} {'TW-Rate Err':<8} {'Ens Err':<8}")
        print("-" * 100)

        history = []
        total_results = len(results)
        start_idx = total_results - BACKTEST_N

        prop_exact = mode_exact = twrate_exact = ens_exact = 0
        prop_abs_err = mode_abs_err = twrate_abs_err = ens_abs_err = 0

        for idx in range(total_results):
            r = results[idx]
            if idx >= start_idx:
                pools = r['pools_tuple']
                actual = r['counts_tuple']

                prop_pred = predict_counts_proportional(pools, max_num, main_count)
                mode_pred = predict_counts_mode(pools, history, max_num, main_count, k=K_NEIGHBORS)
                twrate_pred = predict_counts_time_weighted_rate(pools, history, max_num, main_count)
                ens_pred = predict_counts_ensemble(pools, history, max_num, main_count, k=K_NEIGHBORS)

                prop_err = sum(abs(p - a) for p, a in zip(prop_pred, actual))
                mode_err = sum(abs(p - a) for p, a in zip(mode_pred, actual))
                twrate_err = sum(abs(p - a) for p, a in zip(twrate_pred, actual))
                ens_err = sum(abs(p - a) for p, a in zip(ens_pred, actual))

                if prop_pred == actual: prop_exact += 1
                if mode_pred == actual: mode_exact += 1
                if twrate_pred == actual: twrate_exact += 1
                if ens_pred == actual: ens_exact += 1

                prop_abs_err += prop_err
                mode_abs_err += mode_err
                twrate_abs_err += twrate_err
                ens_abs_err += ens_err

                pool_str = f"{pools[0]}/{pools[1]}/{pools[2]}/{pools[3]}"
                prop_str = f"{prop_pred[0]}/{prop_pred[1]}/{prop_pred[2]}/{prop_pred[3]}"
                mode_str = f"{mode_pred[0]}/{mode_pred[1]}/{mode_pred[2]}/{mode_pred[3]}"
                twrate_str = f"{twrate_pred[0]}/{twrate_pred[1]}/{twrate_pred[2]}/{twrate_pred[3]}"
                ens_str = f"{ens_pred[0]}/{ens_pred[1]}/{ens_pred[2]}/{ens_pred[3]}"
                actual_str = f"{actual[0]}/{actual[1]}/{actual[2]}/{actual[3]}"

                print(f"{r['date']:<20} {pool_str:<15} {prop_str:<15} {mode_str:<15} {twrate_str:<15} {ens_str:<15} {actual_str:<12} "
                      f"{prop_err:<8} {mode_err:<8} {twrate_err:<8} {ens_err:<8}")

            history.append((r['pools_tuple'], r['counts_tuple']))

        print("-" * 100)
        print(f"Proportional model: Exact matches = {prop_exact}/{BACKTEST_N} ({prop_exact/BACKTEST_N:.0%}), Avg abs error = {prop_abs_err/BACKTEST_N:.2f}")
        print(f"Conditional Mode  : Exact matches = {mode_exact}/{BACKTEST_N} ({mode_exact/BACKTEST_N:.0%}), Avg abs error = {mode_abs_err/BACKTEST_N:.2f}")
        print(f"Time‑Weighted Rate: Exact matches = {twrate_exact}/{BACKTEST_N} ({twrate_exact/BACKTEST_N:.0%}), Avg abs error = {twrate_abs_err/BACKTEST_N:.2f}")
        print(f"Ensemble Model    : Exact matches = {ens_exact}/{BACKTEST_N} ({ens_exact/BACKTEST_N:.0%}), Avg abs error = {ens_abs_err/BACKTEST_N:.2f}")

    # ---------- PREDICTION ----------
    prediction_prev_dt = None
    prediction_target_dt = None

    if future_date is not None:
        # Predict for the given future date.
        # If that result already exists in the CSV, the target draw itself is
        # still EXCLUDED from every prediction/history calculation below.
        future_dt = future_date

        # Find the previous occurrence of this same target day before future_dt.
        prev_draw = None
        for date_str, dt, main_nums in draws:
            if dt < future_dt:
                prev_draw = (date_str, dt, main_nums)
            else:
                break

        if prev_draw is None:
            print("No previous draw found for this day before the given future date.")
            return

        prev_date_str, prev_dt, _ = prev_draw
        prediction_prev_dt = prev_dt
        prediction_target_dt = future_dt

        target_date_str = future_dt.strftime('%a %d-%b-%Y')
        print(f"\nPrediction for {lottery_name} on {target_date_str}")
        print("=" * 90)

        # Target pool window: previous same-day target draw -> future target draw.
        window_nums = []
        for date_str, dt, day_abbr2, all_nums, _ in all_rows:
            if prev_dt <= dt < future_dt:
                valid_nums = [n for n in all_nums if 1 <= n <= max_num]
                window_nums.extend(valid_nums)

        print(
            f"Window: {prev_date_str} to "
            f"{future_dt.strftime('%a %d-%b-%Y')}"
        )

    else:
        # Predict the next occurrence.
        last_date_str, last_dt, last_main = draws[-1]
        next_date = last_dt + timedelta(days=1)

        while next_date.strftime('%a')[:3] != day_abbr:
            next_date += timedelta(days=1)

        prediction_prev_dt = last_dt
        prediction_target_dt = next_date

        target_date_str = next_date.strftime('%a %d-%b-%Y')
        print(f"\nPrediction for next {lottery_name} draw: {target_date_str}")
        print("=" * 90)

        window_nums = []
        for date_str, dt, day_abbr2, all_nums, _ in all_rows:
            if last_dt <= dt < next_date:
                valid_nums = [n for n in all_nums if 1 <= n <= max_num]
                window_nums.extend(valid_nums)

        print(f"Window: {last_date_str} to {target_date_str}")

    # Build the final target pool.
    counter = Counter(window_nums)
    target_pools = build_pools(counter, max_num)

    eh = target_pools["EH"]
    h = target_pools["H"]
    w = target_pools["W"]
    c = target_pools["C"]

    eh_pool = len(eh)
    h_pool = len(h)
    w_pool = len(w)
    c_pool = len(c)

    print("\nPool details:")
    print(f"EH {sorted(eh)}  EH-Pool-Size: {eh_pool}")
    print(f"H  {sorted(h)}  H-Pool-Size: {h_pool}")
    print(f"W  {sorted(w)}  W-Pool-Size: {w_pool}")
    print(f"C  {sorted(c)}  C-Pool-Size: {c_pool}")
    print(f"EH+H Pool Size: {eh_pool + h_pool}")

    # ---------- THURSDAY-ONLY POWERBALL BALL (1-20) ----------
    # The Powerball is a separate draw from the seven 1-35 main numbers.
    # Run this module only when the prediction target itself is Thursday.
    if day_abbr == "Thu":
        print_powerball_ball_prediction(
            target_dt=prediction_target_dt,
            all_rows=all_rows,
            pb_max=POWERBALL_BALL_MAX,
            top_n=POWERBALL_BALL_TOP_N,
        )

    # ---------- EXISTING NATIVE-GAME WEEK TABLE ----------
    # This keeps your original table and its actual historical hits.
    # Sunday is intentionally handled by the NEW daily trajectory section,
    # because Sunday has no "Others" target lottery draw to score.
    week_rows = []
    if future_date is not None:
        week_rows = print_week_table(
            future_dt=future_date,
            all_rows=all_rows,
            draws_by_day=draws_by_day,
            target_max_num=max_num,
            target_lottery_name=lottery_name,
            table_days=WEEK_TABLE_DAYS,
            pool_lookback_days=7,
        )

    # ---------- CURRENT POOL-SIZE -> HISTORICAL COMMON HIT COUNTS ----------
    # Example:
    #   current EH size = 12
    #   search BOTH displayed history sources for ANY pool-size = 12,
    #   collect the corresponding actual hit count, then print its mode.
    print_pool_size_common_hit_table(
        current_pools=target_pools,
        historical_results=results,
        week_rows=week_rows,
        cutoff_dt=prediction_target_dt,
        recent_n=OUTPUT_LAST_N,
    )

    # ---------- NEW DAILY SNAPSHOTS / TRAJECTORIES ----------
    if PRINT_DAILY_SNAPSHOT_POOLS:
        print_daily_snapshot_pools(
            prev_target_dt=prediction_prev_dt,
            target_dt=prediction_target_dt,
            all_rows=all_rows,
            max_num=max_num,
            lottery_name=lottery_name,
        )

    if PRINT_TRAJECTORY_TABLE:
        print_number_trajectory_table(
            prev_target_dt=prediction_prev_dt,
            target_dt=prediction_target_dt,
            all_rows=all_rows,
            max_num=max_num,
            lottery_name=lottery_name,
        )

    # Historical full daily trajectory analysis.
    # cutoff_dt makes this non-cheating even when FUTURE_DATE_STR points to a
    # date whose winning result is already present in the CSV.
    trajectory_stats = {}
    trajectory_winner_records = []
    if RUN_TRAJECTORY_HISTORY:
        trajectory_stats, trajectory_winner_records = print_trajectory_pattern_history(
            draws=draws,
            all_rows=all_rows,
            max_num=max_num,
            main_count=main_count,
            lottery_name=lottery_name,
            cutoff_dt=prediction_target_dt,
        )

        print_current_trajectory_groups(
            prev_target_dt=prediction_prev_dt,
            target_dt=prediction_target_dt,
            all_rows=all_rows,
            max_num=max_num,
            stats=trajectory_stats,
            lottery_name=lottery_name,
        )

        # ---------- LOCKED PROFILE -> WINNING TRAJECTORY MODES ----------
        # Each component is conditioned independently using ONLY the two
        # printed sources above: Last-N Saturday analysis + Week Table.
        # Overlapping target-day rows are deduplicated by calendar date.
        print_locked_profile_winning_trajectories_from_tables(
            locked_profile=LOCKED_PROFILE,
            historical_results=results,
            week_rows=week_rows,
            all_rows=all_rows,
            max_num=max_num,
            cutoff_dt=prediction_target_dt,
            recent_n=OUTPUT_LAST_N,
            top_n=LOCKED_TRAJECTORY_TOP_N,
            lookback_days=7,
        )

        # ---------- ALL-HISTORY LOCKED PROFILE / TRAJECTORY ANALYSIS ----------
        # Unlike the two-table section above, this scans the full CSV history
        # strictly before the target date. It prints:
        #   1) exact-profile conditional trajectories
        #   2) independent EH/H/W/C hit-count conditional trajectories
        #   3) candidate-normalized conditional hit rates
        #   4) same-draw trajectory combinations
        #   5) current target candidates mapped to those conditional statistics
        print_all_history_locked_profile_analysis(
            locked_profile=LOCKED_PROFILE,
            all_rows=all_rows,
            max_num=max_num,
            main_count=main_count,
            lottery_name=lottery_name,
            target_day_abbr=day_abbr,
            target_dt=prediction_target_dt,
            current_prev_dt=prediction_prev_dt,
            exact_scope=LOCKED_EXACT_SCOPE,
            independent_scope=LOCKED_INDEPENDENT_SCOPE,
            top_n=LOCKED_TRAJECTORY_TOP_N,
            lookback_days=7,
        )

    # ---------- ORIGINAL EH/H/W/C COUNT PREDICTIONS ----------
    # Keep the original four prediction models in the output.
    # IMPORTANT: only use outcomes strictly BEFORE the target date so the
    # target result cannot leak into the prediction when backtesting a known date.
    model_results = [
        r for r in results
        if r['dt'] < prediction_target_dt
    ]

    full_history = [
        (r['pools_tuple'], r['counts_tuple'])
        for r in model_results
    ]

    current_pool_sizes = (eh_pool, h_pool, w_pool, c_pool)

    prop_pred = predict_counts_proportional(
        current_pool_sizes,
        max_num,
        main_count,
    )
    mode_pred = predict_counts_mode(
        current_pool_sizes,
        full_history,
        max_num,
        main_count,
        k=K_NEIGHBORS,
    )
    twrate_pred = predict_counts_time_weighted_rate(
        current_pool_sizes,
        full_history,
        max_num,
        main_count,
    )
    ens_pred = predict_counts_ensemble(
        current_pool_sizes,
        full_history,
        max_num,
        main_count,
        k=K_NEIGHBORS,
    )

    print("\nProportional prediction:", prop_pred)
    print("Conditional mode prediction:", mode_pred)
    print("Time-Weighted Rate prediction:", twrate_pred)
    print("Ensemble prediction:", ens_pred)
    print("(Use the one that performed best in backtest)")

# ---------- MAIN ----------
# Read all rows
all_rows = []
with open(CSV_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 3:
            continue
        date_str = row[0].strip()
        try:
            dt = parse_date(date_str)
        except:
            continue
        sfl_nums = extract_all_numbers(row[1])
        others_nums = extract_all_numbers(row[2]) if row[2] else []
        all_nums = sfl_nums + others_nums
        day_abbr = date_str[:3]
        all_rows.append((date_str, dt, day_abbr, all_nums, row[2] if row[2] else None))

all_rows.sort(key=lambda x: x[1])

# Group draws by day
draws_by_day = defaultdict(list)
for date_str, dt, day_abbr, _, others_cell in all_rows:
    if day_abbr in LOTTERY_CONFIG and others_cell:
        main_nums = extract_main_numbers(others_cell)
        if main_nums:
            draws_by_day[day_abbr].append((date_str, dt, main_nums))

# Determine what to process
if FUTURE_DATE_STR:
    # Parse the future date
    future_dt = parse_date(FUTURE_DATE_STR)
    day_abbr = FUTURE_DATE_STR[:3]
    if day_abbr not in LOTTERY_CONFIG:
        print(f"Error: '{day_abbr}' is not a supported lottery day.")
    else:
        lottery_name, max_num, main_count = LOTTERY_CONFIG[day_abbr]
        draws = draws_by_day.get(day_abbr, [])
        if not draws:
            print(f"No draws found for {day_abbr}.")
        else:
            process_lottery(day_abbr, draws, all_rows, draws_by_day, max_num, main_count, lottery_name, future_dt)
else:
    # Process all lotteries and predict the next draw for each
    for day_abbr, config in LOTTERY_CONFIG.items():
        lottery_name, max_num, main_count = config
        draws = draws_by_day.get(day_abbr, [])
        if draws:
            process_lottery(day_abbr, draws, all_rows, draws_by_day, max_num, main_count, lottery_name, future_date=None)
        else:
            print(f"\n{lottery_name} ({day_abbr}): No draws found. Skipping.")
