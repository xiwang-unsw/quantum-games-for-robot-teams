"""Exact stage-duration experiments for the revised Section 5.

Reuse only the original game's local measurements, Born distributions and task
predicates. Stage timing is prescribed here and is independent of the archived
trajectory simulator's physical clock. Run: python scripts/delivery_stage_experiments.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from drone_delivery import (
    born_probabilities, classical_answers, geometry_success,
    success_probability as archived_probability,
)

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "delivery_results"
STAGES = 6
BASE = 10
FAILURE_DURATION = {"A": 50, "B": 80}
PRIOR = {"A": 0.5, "B": 0.3, "idle": 0.2}
COST = {"table": 0, "chsh": 1, "six": 1, "square": 2}
MENU = {"A": ("table", "chsh"), "B": ("table", "six", "square")}
MAX_USABLE_PAIRS = STAGES * max(COST.values())
INVENTORY_LEVELS = tuple(range(MAX_USABLE_PAIRS + 3))
V0 = 0.95
SEED = 20261014
METRICS = ["completion_time_s", "successful_coordination_tasks", "pairs_consumed"]
MEMORY_TIMES = (30, 50, 75, 100, 150, 200, 300, 450, 600, 900,
                1200, 1800, 2400, 3600, 6000, 12000, 30000)


def probability(kind, strategy, v):
    if strategy == "table":
        return 3 / 4 if kind == "A" else 8 / 9
    if strategy == "chsh":
        return (1 + v / math.sqrt(2)) / 2
    if strategy == "six":
        return 2 / 3 + v / (2 * math.sqrt(3))
    if strategy == "square":
        return (9 + 4 * v + 5 * v ** 2) / 18
    raise ValueError(strategy)


class Planner:
    """Minimise total stage duration over the stated finite strategy menu."""

    def __init__(self, inventory=3, tm=2400, policy="age_aware", refill=False):
        self.inventory, self.tm = inventory, tm
        self.policy, self.refill = policy, refill
        self.choices = {}
        # Per-instance caches avoid retaining planners from earlier sweep points.
        self.value = lru_cache(None)(self._value)
        self.evaluate = lru_cache(None)(self._evaluate)

    def visibility(self, age, planning=False):
        if self.tm is None or (planning and self.policy == "ignores_ageing"):
            return V0
        return V0 * math.exp(-2 * age / self.tm)

    def state(self, stage, inventory, age):
        if self.refill and stage == 3:
            return 3, 0
        return inventory, age

    def _value(self, stage, inventory, age):
        if stage == STAGES:
            return 0.0
        inventory, age = self.state(stage, inventory, age)
        v = self.visibility(age, planning=True)
        total = PRIOR["idle"] * (BASE + self.value(stage + 1, inventory, age + BASE))
        for kind in ("A", "B"):
            candidates = []
            for strategy in MENU[kind]:
                cost = COST[strategy]
                if cost > inventory or (self.policy == "classical" and cost):
                    continue
                p = probability(kind, strategy, v)
                failure = FAILURE_DURATION[kind]
                value = p * (BASE + self.value(stage + 1, inventory - cost, age + BASE))
                value += (1 - p) * (failure + self.value(stage + 1, inventory - cost, age + failure))
                candidates.append((value, cost, strategy))
            best = min(candidates)
            self.choices[(stage, inventory, age, kind)] = best[2]
            total += PRIOR[kind] * best[0]
        return total

    def choose(self, stage, inventory, age, kind):
        self.value(stage, inventory, age)
        inventory, age = self.state(stage, inventory, age)
        return self.choices[(stage, inventory, age, kind)]

    def _evaluate(self, stage, inventory, age):
        if stage == STAGES:
            return np.zeros(3)
        inventory, age = self.state(stage, inventory, age)
        total = PRIOR["idle"] * (np.array([BASE, 0, 0]) + self.evaluate(stage + 1, inventory, age + BASE))
        for kind in ("A", "B"):
            strategy = self.choose(stage, inventory, age, kind)
            cost = COST[strategy]
            p = probability(kind, strategy, self.visibility(age))
            failure = FAILURE_DURATION[kind]
            total += PRIOR[kind] * (
                p * (np.array([BASE, 1, cost]) + self.evaluate(stage + 1, inventory - cost, age + BASE))
                + (1 - p) * (np.array([failure, 0, cost]) + self.evaluate(stage + 1, inventory - cost, age + failure))
            )
        return total

    def result(self):
        answer = self.evaluate(0, self.inventory, 0)
        if self.policy != "ignores_ageing":
            assert abs(answer[0] - self.value(0, self.inventory, 0)) < 1e-10
        return answer

    def clear(self):
        self.value.cache_clear()
        self.evaluate.cache_clear()
        self.choices.clear()


@lru_cache(maxsize=100000)
def cdf(kind, strategy, x, y, v):
    answers, probs = born_probabilities(kind, strategy, x, y, v)
    return answers, np.cumsum(probs)


def simulate(planner, missions):
    # Common random numbers across policies; independent task/input/outcome streams.
    streams = [np.random.default_rng(np.random.SeedSequence([SEED, j])) for j in range(3)]
    tasks = streams[0].random((missions, STAGES))
    inputs = streams[1].random((missions, STAGES, 2))
    outcomes = streams[2].random((missions, STAGES))
    rows = np.zeros((missions, 3))
    for m in range(missions):
        inventory, age = planner.inventory, 0
        for stage in range(STAGES):
            inventory, age = planner.state(stage, inventory, age)
            u = tasks[m, stage]
            kind = "A" if u < 0.5 else "B" if u < 0.8 else "idle"
            if kind == "idle":
                rows[m, 0] += BASE
                age += BASE
                continue
            strategy = planner.choose(stage, inventory, age, kind)
            cost = COST[strategy]
            n = 2 if kind == "A" else 3
            x, y = (inputs[m, stage] * n).astype(int)
            if strategy == "table":
                a, b = classical_answers(kind, x, y)
            else:
                answers, cumulative = cdf(kind, strategy, int(x), int(y), planner.visibility(age))
                index = min(int(np.searchsorted(cumulative, outcomes[m, stage], side="right")), len(answers) - 1)
                a, b = answers[index]
            success = geometry_success(kind, x, y, a, b)
            duration = BASE if success else FAILURE_DURATION[kind]
            rows[m] += (duration, success, cost)
            inventory -= cost
            age += duration
    mean = rows.mean(axis=0)
    se = rows.std(axis=0, ddof=1) / math.sqrt(missions)
    exact = planner.result()
    z = np.divide(mean - exact, se, out=np.zeros(3), where=se > 0)
    assert np.all(np.abs(mean[se == 0] - exact[se == 0]) < 1e-10)
    assert np.max(np.abs(z)) < 5, (planner.policy, planner.tm, z)
    return {"mean": mean.tolist(), "standard_error": se.tolist(), "z_score": z.tolist()}


def validate_probabilities():
    checks = []
    # Exhaustive task predicates, independent of timing or quantum distributions.
    for x, y, a, b in itertools.product(range(2), repeat=4):
        assert geometry_success("A", x, y, a, b) == ((a ^ b) == x * y)
    assignments = list(itertools.product((-1, 1), repeat=3))
    for x, y in itertools.product(range(3), repeat=2):
        for a, b in itertools.product(assignments, repeat=2):
            assert geometry_success("B", x, y, a, b) == (math.prod(a) == 1 and math.prod(b) == -1 and a[y] == b[x])
    for kind in ("A", "B"):
        n = 2 if kind == "A" else 3
        for strategy in MENU[kind]:
            for v in (0, 0.3, 0.6, 0.95, 1):
                enumerated = 0
                for x, y in itertools.product(range(n), repeat=2):
                    if strategy == "table":
                        a, b = classical_answers(kind, x, y)
                        enumerated += geometry_success(kind, x, y, a, b) / n ** 2
                    else:
                        answers, probs = born_probabilities(kind, strategy, x, y, v)
                        enumerated += sum(p * geometry_success(kind, x, y, a, b)
                                          for (a, b), p in zip(answers, probs)) / n ** 2
                analytic = probability(kind, strategy, v)
                assert abs(enumerated - analytic) < 1e-10
                assert abs(analytic - archived_probability(kind, strategy, v, 0)) < 1e-10
                checks.append({"task": kind, "strategy": strategy, "visibility": v,
                               "analytic": analytic, "born_enumeration": enumerated})
    return checks


def run(missions=10000):
    validation = {"probabilities": validate_probabilities(), "landing_action_cases": 16,
                  "route_assignment_cases": 576}
    classical = Planner(policy="classical")
    baseline = classical.result()
    assert abs(baseline[0] - 104) < 1e-10
    assert abs(baseline[1] - 3.85) < 1e-10
    validation["classical_exact"] = baseline.tolist()
    inventory_rows = []
    for tm in (None, 2400, 600):
        previous = math.inf
        saturated_time = None
        for inventory in INVENTORY_LEVELS:
            planner = Planner(inventory=inventory, tm=tm)
            result = planner.result()
            assert result[0] <= previous + 1e-10
            if inventory == 0:
                assert np.allclose(result, baseline, rtol=0, atol=1e-10)
            if inventory == MAX_USABLE_PAIRS:
                saturated_time = result[0]
            elif inventory > MAX_USABLE_PAIRS:
                assert abs(result[0] - saturated_time) < 1e-10
            previous = result[0]
            inventory_rows.append({"tm": tm, "inventory": inventory, **dict(zip(METRICS, result.tolist()))})
            planner.clear()
    memory_rows = []
    for tm in MEMORY_TIMES:
        record = {"tm": tm}
        for name, inventory, policy, refill in (("age_aware", 3, "age_aware", False),
                                                ("ignores_ageing", 3, "ignores_ageing", False),
                                                ("refill", 3, "age_aware", True),
                                                ("initial_six", 6, "age_aware", False)):
            planner = Planner(inventory=inventory, tm=tm, policy=policy, refill=refill)
            result = planner.result()
            record[name + "_s"] = float(result[0])
            record[name + "_savings_s"] = float(baseline[0] - result[0])
            record[name + "_successful_tasks"] = float(result[1])
            record[name + "_pairs_consumed"] = float(result[2])
            planner.clear()
        assert record["age_aware_s"] <= record["ignores_ageing_s"] + 1e-10
        assert record["refill_s"] <= record["age_aware_s"] + 1e-10
        memory_rows.append(record)
    payload = {
        "parameters": {"stages": STAGES, "successful_stage_s": BASE, "idle_stage_s": BASE,
                       "unsuccessful_stage_s": FAILURE_DURATION, "task_priors": PRIOR,
                       "initial_visibility": V0, "initial_age_s": 0,
                       "initial_inventory_values": list(INVENTORY_LEVELS),
                       "maximum_consumable_pairs": MAX_USABLE_PAIRS,
                       "memory_sweep_initial_inventory": {"age_aware": 3, "ignores_ageing": 3,
                                                           "refill": 3, "initial_six": 6},
                       "visibility_law": "v(age) = v0 * exp(-2*age/tm); tm=null means no decay",
                       "timely_radio_probability": 0, "strategy_pair_cost": COST,
                       "strategy_menu": MENU, "within_task_inputs": "independent uniform",
                       "replacement": {"before_stage_one_based": 4, "fresh_pairs": 3,
                                       "discard_remaining_stock": True, "age_s": 0, "duration_s": 0},
                       "samples_per_policy": missions, "seed": SEED,
                       "sample_metric_order": METRICS,
                       "successful_task_count_excludes_idle_stages": True,
                       "timing_model": "Prescribed stage durations; no physical trajectory timing"},
        "inventory": inventory_rows, "memory_sweep": memory_rows,
        "validation": validation, "samples": [],
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "stage_experiments.json"
    # Publish exact curves before the optional sampling checks finish.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Exact curves saved:", path, flush=True)
    for row in inventory_rows:
        if row["inventory"] == 3:
            print("inventory=3", row, flush=True)
    scenarios = [(None, 3, "classical", False)]
    scenarios += [(tm, 3, policy, refill) for tm in (30, 600, 2400)
                  for policy, refill in (("age_aware", False), ("ignores_ageing", False), ("age_aware", True))]
    scenarios += [(tm, 6, "age_aware", False) for tm in (30, 600, 2400)]
    for tm, inventory, policy, refill in scenarios:
        planner = Planner(inventory=inventory, tm=tm, policy=policy, refill=refill)
        result = planner.result()
        sample = {"tm": tm, "inventory": inventory, "policy": policy, "refill": refill,
                  "exact": result.tolist(), "monte_carlo": simulate(planner, missions)}
        payload["samples"].append(sample)
        print("sample", tm, inventory, policy, refill, "exact", np.round(result, 6),
              "z", np.round(sample["monte_carlo"]["z_score"], 3), flush=True)
        planner.clear()
    validation["inventory_nonincreasing"] = True
    validation["inventory_saturates_by_consumption_bound"] = True
    validation["zero_inventory_equals_classical"] = True
    validation["age_aware_not_worse_than_ignores_ageing"] = True
    validation["replacement_not_worse_than_three_initial_pairs"] = True
    validation["policy_value_equals_exact_evaluation"] = True
    validation["maximum_absolute_mc_z_score"] = max(abs(z) for s in payload["samples"] for z in s["monte_carlo"]["z_score"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions", type=int, default=10000)
    run(parser.parse_args().missions)
