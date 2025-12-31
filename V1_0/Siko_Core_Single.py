# Core engine for FINAL LOTTERY STRATEGY v1.0 (NO REPEAT LOGIC)
# Applies to Oz Lotto-style K-out-of-N games.

import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations


DATE_FMT = "%Y-%m-%d"


@dataclass
class Ticket:
    numbers: list
    decade_vector: list
    freq_counts: dict
    recency_counts: dict
    total_sum: int
    odds: int
    evens: int
    consec_pairs: int


def _parse_date(date_str):
    return datetime.strptime(date_str, DATE_FMT).date()


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _load_draws(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "Winning Number 1" not in headers:
            raise ValueError("Missing 'Winning Number 1' column in CSV")
        start_idx = headers.index("Winning Number 1")
        number_cols = ["Winning Number 1"]
        for i in range(start_idx + 1, len(headers)):
            name = headers[i]
            if name.isdigit():
                number_cols.append(name)
            else:
                break
        draws = []
        for row in reader:
            date_str = row.get("Date", "").strip()
            if not date_str:
                continue
            draw_date = _parse_date(date_str)
            numbers = []
            for col in number_cols:
                val = row.get(col, "").strip()
                if val == "":
                    continue
                numbers.append(int(val))
            if numbers:
                draws.append((draw_date, numbers))
        return number_cols, draws


def _make_frequency_bands(freq, top_pct=0.20, upper_pct=0.20, mid_pct=0.45, cold_pct=0.10):
    nums = sorted(freq.keys())
    nums_sorted = sorted(nums, key=lambda x: (-freq[x], x))
    n_total = len(nums_sorted)
    cut_top = int(round(n_total * top_pct))
    cut_up = int(round(n_total * upper_pct))
    cut_mid = int(round(n_total * mid_pct))
    cut_cold = int(round(n_total * cold_pct))
    cut_very = n_total - (cut_top + cut_up + cut_mid + cut_cold)
    if cut_very < 0:
        cut_mid += cut_very
        cut_very = 0

    bands = {}
    idx = 0
    for n in nums_sorted[idx:idx + cut_top]:
        bands[n] = "top_high"
    idx += cut_top
    for n in nums_sorted[idx:idx + cut_up]:
        bands[n] = "upper_mid"
    idx += cut_up
    for n in nums_sorted[idx:idx + cut_mid]:
        bands[n] = "mid"
    idx += cut_mid
    for n in nums_sorted[idx:idx + cut_cold]:
        bands[n] = "cold"
    idx += cut_cold
    for n in nums_sorted[idx:]:
        bands[n] = "very_cold"
    return bands


def _recency_bands(recency_gap):
    bands = {}
    for n, g in recency_gap.items():
        if g <= 2:
            bands[n] = "very_recent"
        elif g <= 6:
            bands[n] = "short"
        elif g <= 15:
            bands[n] = "medium"
        else:
            bands[n] = "long"
    return bands


def _bucket_index(n, decade_size):
    return (n - 1) // decade_size


def _ticket_stats(nums_sel, freq_class, rec_class, num_buckets, decade_size):
    freq_counts = {"top_high": 0, "mid": 0, "cold": 0, "very_cold": 0}
    rec_counts = {"very_recent": 0, "short": 0, "medium": 0, "long": 0}
    for n in nums_sel:
        freq_counts[freq_class(n)] += 1
        rec_counts[rec_class(n)] += 1
    odds = sum(1 for n in nums_sel if n % 2 == 1)
    evens = len(nums_sel) - odds
    total_sum = sum(nums_sel)
    nums_sorted = sorted(nums_sel)
    consec_pairs = sum(1 for i in range(len(nums_sorted) - 1) if nums_sorted[i + 1] == nums_sorted[i] + 1)
    decade_vector = [0] * num_buckets
    for n in nums_sel:
        decade_vector[_bucket_index(n, decade_size)] += 1
    return freq_counts, rec_counts, odds, evens, total_sum, consec_pairs, decade_vector


def _passes_constraints(nums_sel, freq_class, rec_class, num_buckets, decade_size, sum_low, sum_high, regime):
    freq_counts, rec_counts, odds, evens, total_sum, consec_pairs, decade_vector = _ticket_stats(
        nums_sel, freq_class, rec_class, num_buckets, decade_size
    )
    if not (4 <= freq_counts["mid"] <= 5):
        return False
    if not (1 <= freq_counts["top_high"] <= 2):
        return False
    if not (0 <= freq_counts["cold"] <= 1):
        return False
    if freq_counts["very_cold"] != 0:
        return False

    if not (3 <= rec_counts["medium"] <= 4):
        return False
    if not (2 <= rec_counts["short"] <= 3):
        return False
    if rec_counts["very_recent"] > 1:
        return False
    if rec_counts["long"] > 1:
        return False

    if not (2 <= odds <= 5):
        return False
    if consec_pairs > 1:
        return False
    if not (sum_low <= total_sum <= sum_high):
        return False

    if regime == "A":
        if any(c >= 3 for c in decade_vector):
            return False
    if regime == "B":
        if max(decade_vector) < 4:
            return False
    if regime == "C":
        ok = False
        for a, b in [(i, i + 1) for i in range(num_buckets - 1)]:
            if decade_vector[a] == 3 and decade_vector[b] == 3:
                if max(decade_vector) <= 3:
                    ok = True
                    break
        if not ok:
            return False
    return True


@dataclass
class _Combo:
    nums: tuple
    total_sum: int
    odds: int
    freq_counts: dict
    rec_counts: dict


def _build_combos(bucket_nums, count, freq_class, rec_class):
    if count == 0:
        return [_Combo((), 0, 0, {"top_high": 0, "mid": 0, "cold": 0, "very_cold": 0},
                       {"very_recent": 0, "short": 0, "medium": 0, "long": 0})]
    combos = []
    for c in combinations(bucket_nums, count):
        total_sum = sum(c)
        odds = sum(1 for n in c if n % 2 == 1)
        freq_counts = {"top_high": 0, "mid": 0, "cold": 0, "very_cold": 0}
        rec_counts = {"very_recent": 0, "short": 0, "medium": 0, "long": 0}
        for n in c:
            freq_counts[freq_class(n)] += 1
            rec_counts[rec_class(n)] += 1
        combos.append(_Combo(c, total_sum, odds, freq_counts, rec_counts))
    combos.sort(key=lambda x: (-(x.freq_counts["mid"] + x.rec_counts["medium"]), -x.freq_counts["top_high"],
                               -x.rec_counts["short"], x.total_sum))
    return combos


def _find_ticket(vector, candidates_by_bucket, freq_class, rec_class, num_buckets, decade_size,
                 sum_low, sum_high, regime, existing):
    combos_per_bucket = []
    for i in range(num_buckets):
        combos = _build_combos(candidates_by_bucket.get(i, []), vector[i], freq_class, rec_class)
        combos_per_bucket.append(combos)

    min_sum_rem = [0] * (num_buckets + 1)
    max_sum_rem = [0] * (num_buckets + 1)
    for i in range(num_buckets - 1, -1, -1):
        sums = [c.total_sum for c in combos_per_bucket[i]]
        min_sum_rem[i] = (min(sums) if sums else 0) + min_sum_rem[i + 1]
        max_sum_rem[i] = (max(sums) if sums else 0) + max_sum_rem[i + 1]

    best = None

    def recurse(i, nums_acc, freq_acc, rec_acc, sum_acc, odds_acc):
        nonlocal best
        if best is not None:
            return
        if i == num_buckets:
            nums_sel = list(nums_acc)
            for t in existing:
                if len(set(nums_sel).intersection(t)) > 2:
                    return
            if _passes_constraints(nums_sel, freq_class, rec_class, num_buckets, decade_size, sum_low, sum_high, regime):
                best = sorted(nums_sel)
            return

        if sum_acc + min_sum_rem[i] > sum_high:
            return
        if sum_acc + max_sum_rem[i] < sum_low:
            return

        for combo in combos_per_bucket[i]:
            freq_new = freq_acc.copy()
            rec_new = rec_acc.copy()
            for k, v in combo.freq_counts.items():
                freq_new[k] += v
            for k, v in combo.rec_counts.items():
                rec_new[k] += v
            if freq_new["mid"] > 5 or freq_new["top_high"] > 2 or freq_new["cold"] > 1 or freq_new["very_cold"] > 0:
                continue
            if rec_new["medium"] > 4 or rec_new["short"] > 3 or rec_new["very_recent"] > 1 or rec_new["long"] > 1:
                continue
            odds_new = odds_acc + combo.odds
            if odds_new > 5:
                continue
            recurse(i + 1, nums_acc + combo.nums, freq_new, rec_new, sum_acc + combo.total_sum, odds_new)
            if best is not None:
                return

    freq_acc = {"top_high": 0, "mid": 0, "cold": 0, "very_cold": 0}
    rec_acc = {"very_recent": 0, "short": 0, "medium": 0, "long": 0}
    recurse(0, (), freq_acc, rec_acc, 0, 0)
    return best


def generate_tickets(csv_path, target_date_str, ticket_count=5, game_name="Oz Lotto"):
    number_cols, draws = _load_draws(csv_path)
    target_date = _parse_date(target_date_str)
    hist = [(d, nums) for d, nums in draws if d < target_date]
    if not hist:
        raise ValueError("No draws before target date")

    hist_sorted = sorted(hist, key=lambda x: x[0])
    N = max(n for _, nums in hist_sorted for n in nums)
    K = len(number_cols)

    cur_year = target_date.year
    hist_year = [nums for d, nums in hist_sorted if d.year == cur_year]

    freq = {n: 0 for n in range(1, N + 1)}
    for nums in hist_year:
        for n in nums:
            freq[n] += 1

    last_idx = {n: None for n in range(1, N + 1)}
    for idx, (_, nums) in enumerate(hist_sorted):
        for n in nums:
            last_idx[n] = idx
    total_draws = len(hist_sorted)
    recency_gap = {}
    for n in range(1, N + 1):
        if last_idx[n] is None:
            recency_gap[n] = total_draws
        else:
            recency_gap[n] = total_draws - last_idx[n]

    freq_bands = _make_frequency_bands(freq)
    recency_bands = _recency_bands(recency_gap)

    decade_size = 10
    num_buckets = (N - 1) // decade_size + 1

    def freq_class(n):
        band = freq_bands[n]
        if band == "top_high":
            return "top_high"
        if band in ("upper_mid", "mid"):
            return "mid"
        if band == "cold":
            return "cold"
        return "very_cold"

    def rec_class(n):
        return recency_bands[n]

    last3 = hist_sorted[-3:] if len(hist_sorted) >= 3 else hist_sorted
    bucket_counts_last3 = [0] * num_buckets
    for _, nums in last3:
        for n in nums:
            bucket_counts_last3[_bucket_index(n, decade_size)] += 1

    clustered = max(bucket_counts_last3) >= 6
    min_count = min(bucket_counts_last3)
    under_decades = [i for i, c in enumerate(bucket_counts_last3) if c == min_count]
    if under_decades:
        cluster_decade = under_decades[0]
    else:
        cluster_decade = bucket_counts_last3.index(max(bucket_counts_last3))

    sums = sorted(sum(nums) for nums in hist_year)
    sum_low = _percentile(sums, 20)
    sum_high = _percentile(sums, 80)

    valid_nums = [n for n in range(1, N + 1) if freq_bands[n] != "very_cold"]
    candidates_by_bucket = {i: [] for i in range(num_buckets)}
    for n in valid_nums:
        candidates_by_bucket[_bucket_index(n, decade_size)].append(n)

    regimeA_vectors = []
    if num_buckets == 5:
        regimeA_vectors = [(1, 2, 2, 1, 1), (2, 2, 2, 1, 0), (1, 1, 2, 2, 1), (1, 2, 1, 2, 1)]
    else:
        # Fallback: spread 7 numbers roughly evenly across buckets
        base = [0] * num_buckets
        for i in range(7):
            base[i % num_buckets] += 1
        regimeA_vectors = [tuple(base)]

    vecB = [0] * num_buckets
    vecB[cluster_decade] = 4
    other_indices = [i for i in range(num_buckets) if i != cluster_decade]
    other_indices.sort(key=lambda i: len(candidates_by_bucket[i]), reverse=True)
    for i in other_indices[:3]:
        vecB[i] += 1
    while sum(vecB) > 7:
        for i in other_indices:
            if vecB[i] > 0:
                vecB[i] -= 1
                break
    while sum(vecB) < 7:
        for i in other_indices:
            vecB[i] += 1
            if sum(vecB) == 7:
                break

    adj_pairs = [(i, i + 1) for i in range(num_buckets - 1)]

    tickets = []
    meta = {
        "game_name": game_name,
        "target_date": target_date_str,
        "csv_path": csv_path,
        "draws_used": len(hist_sorted),
        "lookback_year": cur_year,
        "N": N,
        "K": K,
        "sum_low": sum_low,
        "sum_high": sum_high,
        "last3_bucket_counts": bucket_counts_last3,
        "clustered_last3": clustered,
        "cluster_decade": cluster_decade,
    }

    # Regime A tickets
    for vec in regimeA_vectors:
        if len([t for t in tickets if t[0] == "A"]) >= (3 if ticket_count >= 5 else 1):
            break
        t = _find_ticket(vec, candidates_by_bucket, freq_class, rec_class, num_buckets, decade_size,
                         sum_low, sum_high, "A", [x[2] for x in tickets])
        if t:
            tickets.append(("A", vec, t))

    # Regime B ticket
    if ticket_count >= 3:
        t = _find_ticket(tuple(vecB), candidates_by_bucket, freq_class, rec_class, num_buckets, decade_size,
                         sum_low, sum_high, "B", [x[2] for x in tickets])
        if t:
            tickets.append(("B", tuple(vecB), t))

    # Regime C ticket
    if ticket_count >= 3:
        pairs_sorted = sorted(adj_pairs, key=lambda p: len(candidates_by_bucket[p[0]]) + len(candidates_by_bucket[p[1]]),
                              reverse=True)
        for a, b in pairs_sorted:
            vec = [0] * num_buckets
            vec[a] = 3
            vec[b] = 3
            other = [i for i in range(num_buckets) if i not in (a, b)]
            other.sort(key=lambda i: len(candidates_by_bucket[i]), reverse=True)
            if other:
                vec[other[0]] = 1
            t = _find_ticket(tuple(vec), candidates_by_bucket, freq_class, rec_class, num_buckets, decade_size,
                             sum_low, sum_high, "C", [x[2] for x in tickets])
            if t:
                tickets.append(("C", tuple(vec), t))
                break

    # Limit to requested count
    tickets = tickets[:ticket_count]

    final_tickets = []
    for regime, vec, nums in tickets:
        freq_counts, rec_counts, odds, evens, total_sum, consec_pairs, decade_vector = _ticket_stats(
            nums, freq_class, rec_class, num_buckets, decade_size
        )
        final_tickets.append((regime, vec, Ticket(nums, decade_vector, freq_counts, rec_counts,
                                                 total_sum, odds, evens, consec_pairs)))

    return meta, final_tickets


def print_report(meta, final_tickets):
    print("Game:", meta["game_name"])
    print("Target date:", meta["target_date"])
    print("CSV source:", meta["csv_path"])
    print("Draws used:", meta["draws_used"])
    print("Lookback year:", meta["lookback_year"])
    print("N,K:", meta["N"], meta["K"])
    print("Sum band 20-80%:", f"{meta['sum_low']:.1f}", f"{meta['sum_high']:.1f}")
    print("Last 3 draws decade counts:", meta["last3_bucket_counts"])
    print("Clustered last 3:", meta["clustered_last3"])
    print("Cluster decade index (0-based):", meta["cluster_decade"])
    print()

    for idx, (regime, vec, ticket) in enumerate(final_tickets, 1):
        print(f"Ticket {idx} Regime {regime}")
        print("Numbers:", ticket.numbers)
        print("Decade vector:", ticket.decade_vector)
        print("Freq counts:", ticket.freq_counts)
        print("Recency counts:", ticket.recency_counts)
        print("Sum:", ticket.total_sum, "Odd/Even:", ticket.odds, ticket.evens, "Consec pairs:", ticket.consec_pairs)
        print()

    print("Overlap matrix:")
    for i in range(len(final_tickets)):
        row = []
        for j in range(len(final_tickets)):
            if i == j:
                row.append("-")
            else:
                a = set(final_tickets[i][2].numbers)
                b = set(final_tickets[j][2].numbers)
                row.append(str(len(a.intersection(b))))
        print(" ".join(row))

