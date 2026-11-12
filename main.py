#!/usr/bin/env python3
"""Forecast the Roblox Up-and-Coming top 10 for each scored window slot.

The model learns rank movements, entry pressure, and tenure-dependent drop-out
hazards from prior chart snapshots, then simulates forward from the latest
snapshot. It reads task.json (window bounds, slot count, and output placement
count) and history.ndjson (timestamped chart reads) from MAKEFUN_INPUT_DIR,
defaulting to /input, and writes forecast.json to MAKEFUN_OUTPUT_DIR, defaulting
to /output. It uses only the standard library and a fixed seed for repeatable
results.
"""

import heapq
import json
import math
import os
from datetime import datetime

DAY_SECONDS = 86_400.0
HOUR_SECONDS = 3_600.0
SEED = 0x8F3A2D17
MIN_PROBABILITY = 1e-4
MAX_ROWS_PER_PLACE = 100
TOP_K = 4
NAMED_MASS = 0.85
DRAWS = 500
# Validated on pre-window one-read and multi-read holdouts.
STEP_SCALE = 4.0
HAZARD_SCALE = 0.5
# Window slots are weighted relative to their first four anchor horizons.
# The bar is graded over the whole window, so long horizons get real weight.
MIXTURE = (32.0, 32.0, 12.0, 2.0)


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(quantile * (len(ordered) - 1))))]


def rank_band(rank):
    if rank <= 3:
        return 0
    if rank <= 6:
        return 1
    if rank <= 10:
        return 2
    if rank <= 20:
        return 3
    return 4


def top_band(rank):
    return min(2, rank_band(rank))


def tenure_band(days):
    if days < 1:
        return 0
    if days < 4:
        return 1
    return 2


class FixedRandom:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def __call__(self):
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        value = self.state
        value = ((value ^ (value >> 15)) * (value | 1)) & 0xFFFFFFFF
        value ^= (value + ((value ^ (value >> 7)) * (value | 61) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((value ^ (value >> 14)) & 0xFFFFFFFF) / 4_294_967_296.0


def poisson(lam, random):
    if lam <= 0:
        return 0
    if lam < 30:
        limit = math.exp(-lam)
        product = 1.0
        count = 0
        while True:
            count += 1
            product *= random()
            if product <= limit:
                return count - 1
    u1 = max(2.220446049250313e-16, random())
    u2 = random()
    return max(0, round(lam + math.sqrt(lam) * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)))


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as handle:
        task = json.load(handle)
    cutoff = parse_time(task["history_until"])
    snapshots = []
    history_path = os.path.join(input_dir, "history.ndjson")
    if not os.path.isfile(history_path):
        raise SystemExit(f"missing input: {history_path}")
    found_chart_read = False
    with open(history_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            at = parse_time(raw["at"])
            if at >= cutoff:
                continue
            chart_rows = raw.get("up_and_coming")
            if isinstance(chart_rows, list):
                found_chart_read = True
            chart = sorted(chart_rows or [], key=lambda row: row["rank"])
            if not chart:
                continue
            snapshots.append({
                "at": at,
                "chart": chart,
                "details": raw.get("details") or {},
            })
    if not found_chart_read:
        raise SystemExit(f"invalid input: {history_path} has no read with an up_and_coming list")
    snapshots.sort(key=lambda snap: snap["at"])
    if len(snapshots) < 2:
        raise SystemExit("history.ndjson needs at least two snapshots before the window")
    return task, snapshots


class TransitionModel:
    pass


def learn_model(history):
    model = TransitionModel()
    first_seen = {}
    created_at = {}
    for snapshot in history:
        for row in snapshot["chart"]:
            game = row["universe_id"]
            first_seen.setdefault(game, snapshot["at"])
            detail = snapshot["details"].get(str(game)) or {}
            if detail.get("created") and game not in created_at:
                created_at[game] = parse_time(detail["created"])

    censored = {row["universe_id"] for row in history[0]["chart"]}
    gaps = [(history[i]["at"] - history[i - 1]["at"]) / HOUR_SECONDS for i in range(1, len(history))]
    model.short_gap = percentile(gaps, 0.33)
    model.long_gap = percentile(gaps, 0.67)

    raw = [[[[] for _ in range(3)] for _ in range(5)] for _ in range(2)]
    rank_raw = [[[] for _ in range(5)] for _ in range(2)]
    global_raw = [[], []]
    leave_counts = [[0] * 3 for _ in range(3)]
    leave_exposure = [[0.0] * 3 for _ in range(3)]
    entry_ranks = []
    observed_days = 0.0
    new_games = 0

    def tenure_bands(snapshot_index):
        out = {}
        snapshot = history[snapshot_index]
        for row in snapshot["chart"]:
            game = row["universe_id"]
            at = snapshot["at"]
            if game in created_at:
                out[game] = tenure_band((at - created_at[game]) / DAY_SECONDS)
            elif game in censored and first_seen.get(game) == history[0]["at"]:
                out[game] = 2
            else:
                out[game] = tenure_band((at - first_seen.get(game, at)) / DAY_SECONDS)
        return out

    tenures = [tenure_bands(index) for index in range(len(history))]

    for index in range(1, len(history)):
        previous = history[index - 1]
        current = history[index]
        dt = (current["at"] - previous["at"]) / HOUR_SECONDS
        group = 0 if dt <= model.short_gap else 1
        observed_days += dt / 24
        before = {row["universe_id"]: row for row in previous["chart"]}
        after = {row["universe_id"]: row for row in current["chart"]}

        for game, row in after.items():
            prior = before.get(game)
            if prior is None:
                if first_seen.get(game) == current["at"]:
                    entry_ranks.append(row["rank"])
                    new_games += 1
                continue

            delta = row["rank"] - prior["rank"]
            band = rank_band(prior["rank"])
            tenure = tenures[index - 1].get(game, 2)
            raw[group][band][tenure].append(delta)
            rank_raw[group][band].append(delta)
            global_raw[group].append(delta)

            if prior["rank"] <= 10:
                leave_exposure[band][tenure] += dt / 24
                if row["rank"] > 10:
                    leave_counts[band][tenure] += 1

        for row in previous["chart"]:
            if row["rank"] <= 10 and row["universe_id"] not in after:
                band = rank_band(row["rank"])
                tenure = tenures[index - 1].get(row["universe_id"], 2)
                leave_exposure[band][tenure] += dt / 24
                leave_counts[band][tenure] += 1

    model.by_rank = [[sorted(values) for values in group] for group in rank_raw]
    model.global_deltas = [sorted(values) for values in global_raw]
    model.by_rank_tenure = [
        [[sorted(raw[group][band][tenure]) for tenure in range(3)] for band in range(5)]
        for group in range(2)
    ]

    model.leave_hazard = [[0.0] * 3 for _ in range(3)]
    for band in range(3):
        for tenure in range(3):
            prior = 0.25 if band == 0 else 0.35 if band == 1 else 0.45
            pseudo_days = 8.0
            model.leave_hazard[band][tenure] = (
                leave_counts[band][tenure] + prior * pseudo_days
            ) / max(1.0, leave_exposure[band][tenure] + pseudo_days)

    if not entry_ranks:
        entry_ranks = [20, 40, 60, 80, 90, 95, 98]
    model.entry_ranks = sorted(entry_ranks)
    model.entry_rate = max(0.05, new_games / max(0.1, observed_days))
    model.first_seen = first_seen
    model.created_at = created_at
    model.censored = censored
    return model


def delta_pool(model, group, band, tenure):
    tenure_cell = model.by_rank_tenure[group][band][tenure]
    rank_cell = model.by_rank[group][band]
    if len(tenure_cell) >= 24:
        return tenure_cell
    if len(rank_cell) >= 24:
        return rank_cell
    return model.global_deltas[group]


def origin_time(model, game, origin):
    if game in model.created_at:
        return model.created_at[game]
    if game in model.censored and model.first_seen.get(game) == origin:
        return origin - 5 * DAY_SECONDS
    return model.first_seen.get(game, origin)


def window_targets(task):
    slots = max(1, int(task.get("window", {}).get("slots", 72)))
    start = parse_time(task["window"]["from"])
    end = parse_time(task["window"]["until"])
    step_hours = max(1, round((end - start) / max(1, slots) / HOUR_SECONDS))
    available = {(index + 1) * step_hours for index in range(slots)}
    anchors = (1, 8, 24, 72)
    targets = []
    for hour, weight in zip(anchors, MIXTURE):
        chosen = hour if hour in available else max(available)
        targets.append((chosen, weight))
    return targets


def simulate(model, history, targets, places, draws):
    candidates = history[-1]["chart"][:120]
    ids = [row["universe_id"] for row in candidates]
    counts = [[0.0] * places for _ in ids]
    rank_sum = [0.0] * len(ids)
    ghost_mass = [0.0] * places

    total_weight = sum(weight for _, weight in targets)
    max_hour = max(hour for hour, _ in targets)
    weight_at = [0.0] * (max_hour + 1)
    for hour, weight in targets:
        weight_at[max(1, min(max_hour, hour))] += weight

    short_pools = [[delta_pool(model, 0, band, tenure) for tenure in range(3)] for band in range(5)]
    long_pools = [[delta_pool(model, 1, band, tenure) for tenure in range(3)] for band in range(5)]
    entry_pool = model.entry_ranks
    leave_probability = [
        [1 - (1 - min(0.999999, max(0.0, value * HAZARD_SCALE))) ** (1 / 24) for value in row]
        for row in model.leave_hazard
    ]
    entries_per_hour = max(0.0, model.entry_rate) / 24
    random = FixedRandom(SEED)
    origin = history[-1]["at"]
    initial_created = [origin_time(model, row["universe_id"], origin) for row in candidates]
    capacity = len(candidates) + int(math.ceil(entries_per_hour * max_hour * 3)) + 64

    for _ in range(draws):
        active = len(candidates)
        active_ranks = [row["rank"] for row in candidates]
        active_created = list(initial_created)
        active_slots = list(range(len(candidates)))

        for hour in range(1, max_hour + 1):
            now = origin + hour * HOUR_SECONDS
            gap_mix = min(1.0, max(0.0, (hour - model.short_gap) / max(0.25, model.long_gap - model.short_gap)))
            effective_gap = model.short_gap + (model.long_gap - model.short_gap) * gap_mix
            jump_probability = min(1.0, STEP_SCALE / max(0.25, effective_gap))

            for index in range(active):
                rank = active_ranks[index]
                band = rank_band(rank)
                tenure = tenure_band((now - active_created[index]) / DAY_SECONDS)
                pool = long_pools[band][tenure] if random() < gap_mix else short_pools[band][tenure]
                if not pool:
                    pool = short_pools[band][tenure] or model.global_deltas[0]
                step = pool[min(len(pool) - 1, int(random() * len(pool)))] if random() < jump_probability else 0
                rank += step
                rank = 1 if rank < 1 else 110 if rank > 110 else rank
                if rank <= 10.5 and random() < leave_probability[top_band(band)][tenure]:
                    rank = 11 + random() * 12
                active_ranks[index] = rank

            for _ in range(poisson(entries_per_hour, random)):
                rank = entry_pool[min(len(entry_pool) - 1, int(random() * len(entry_pool)))]
                if rank > 110:
                    continue
                if active >= capacity:
                    active = len(candidates)
                active_ranks.append(rank + random())
                active_created.append(now)
                active_slots.append(-1)
                active += 1

            weight = weight_at[hour]
            if weight == 0.0:
                continue

            best = [index for _, index in heapq.nsmallest(
                places, ((active_ranks[index], index) for index in range(active)))]
            for index in range(len(candidates)):
                if active_slots[index] >= 0:
                    rank_sum[index] += weight * active_ranks[index]
            for place, index in enumerate(best):
                slot = active_slots[index]
                if slot >= 0:
                    counts[slot][place] += weight
                else:
                    ghost_mass[place] += weight

    return ids, counts, rank_sum, ghost_mass, total_weight, draws


def make_forecast(ids, counts, rank_sum, ghost_mass, total_weight, draws, places=10):
    del ghost_mass
    denominator = max(1e-9, draws * total_weight)
    rows = []
    by_place = {}

    for place in range(places):
        ranked = sorted(
            ((ids[index], counts[index][place] / denominator) for index in range(len(ids))),
            key=lambda item: (-item[1], item[0]),
        )
        kept = [item for index, item in enumerate(ranked) if index < TOP_K or item[1] > 0.001]
        kept = kept[:MAX_ROWS_PER_PLACE]
        kept_mass = sum(item[1] for item in kept)
        current = {}
        for game, probability in kept:
            final = probability / kept_mass * NAMED_MASS if kept_mass > 0.0 else 0.0
            if final >= MIN_PROBABILITY:
                rows.append({"universe_id": game, "place": place + 1, "p": final})
                current[game] = final
        by_place[place + 1] = current

    top10 = []
    for place in range(1, places + 1):
        choices = [(game, probability) for game, probability in by_place.get(place, {}).items()
                   if game not in top10]
        if choices:
            top10.append(max(choices, key=lambda item: (item[1], -item[0]))[0])
    mean_rank = sorted(
        ((ids[index], rank_sum[index] / denominator) for index in range(len(ids))),
        key=lambda item: (item[1], item[0]),
    )
    for game, _ in mean_rank:
        if len(top10) >= places:
            break
        if game not in top10:
            top10.append(game)

    return {"probabilities": rows, "top10": top10[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, history = load_input(input_dir)
    model = learn_model(history)
    result = simulate(model, history, window_targets(task), int(task.get("places", 10)), DRAWS)
    forecast = make_forecast(*result[:6], places=int(task.get("places", 10)))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w", encoding="utf-8") as handle:
        json.dump(forecast, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
