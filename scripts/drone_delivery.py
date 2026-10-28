"""Six-leg delivery mission using the manuscript's actual Task A and Task B.

An idealised reservation/trajectory simulator, not a flight dynamics simulator.
Samples complete local outputs from Born distributions and scores landing
geometry, endpoint layers and shared cell reservations. Exact dynamic programs
include failure-dependent elapsed time and common-age Werner pairs.

Run from anywhere: python drone_delivery.py --missions 10000
All outputs are written beside the manuscript, in delivery_results/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "delivery_results"
T, LEG, RELEASE = 6, 60, 20
EVENT_PRIOR = {"A": 0.5, "B": 0.3, "plain": 0.2}
RECOVERY = {"A": 45, "B": 12, "plain": 0}
PAIR_COST = {"table": 0, "chsh": 1, "six": 1, "embedded": 1, "square": 2}
OPTIONS = {"A": ("table", "chsh"), "B": ("table", "six", "embedded", "square")}
RADIUS, CELL, SLOT, WINDOW, PHASE_CHANGE = 10.0, 30.0, 5.0, 12.0, 4.0
LOWER, UPPER, SAFE_SEPARATION = 30.0, 42.0, 5.0
SIGNS = (-1, 1)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PHI = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
BELL = np.outer(PHI, PHI.conj())
MIX = np.eye(4) / 4
MP = ((np.kron(I2, Z), np.kron(Z, I2), np.kron(Z, Z)),
      (np.kron(X, I2), np.kron(I2, X), np.kron(X, X)),
      (-np.kron(X, Z), -np.kron(Z, X), np.kron(Y, Y)))
TABLE_A = np.array([[1, 1, 1], [1, 1, 1], [-1, -1, 1]])
TABLE_B = np.array([[1, 1, 1], [1, 1, 1], [-1, -1, -1]])


def observable(angle):
    return math.cos(angle) * Z + math.sin(angle) * X


def qubit_answers(kind, strategy, index, side):
    if kind == "A":
        op = (X, Z)[index] if side == "A" else ((X + Z) / math.sqrt(2), (X - Z) / math.sqrt(2))[index]
        return [(int(m == -1), (I2 + m * op) / 2) for m in SIGNS]
    if strategy == "six":
        angle = ((0, math.pi / 3, 2 * math.pi / 3) if side == "A"
                 else (5 * math.pi / 6, math.pi / 6, math.pi / 2))[index]
        answer = (lambda m: ((m, m, 1), (1, -m, -m), (m, 1, m))[index]) if side == "A" else (
            lambda m: ((-m, 1, m), (m, -m, 1), (1, -m, m))[index])
    else:
        if index == 2:
            return [((1, 1, 1) if side == "A" else (1, -1, 1), I2)]
        angle = ((0, math.pi / 2) if side == "A" else (-math.pi / 4, math.pi / 4))[index]
        answer = (lambda m: ((m, m, 1), (m, -m, -1))[index]) if side == "A" else (lambda m: (m, -m, 1))
    op = observable(angle)
    return [(answer(m), (I2 + m * op) / 2) for m in SIGNS]


def square_answers(index, side):
    operators = MP[index] if side == "A" else tuple(MP[r][index] for r in range(3))
    parity = 1 if side == "A" else -1
    result = []
    for a, b in itertools.product(SIGNS, repeat=2):
        result.append(((a, b, parity * a * b),
                       (np.eye(4) + a * operators[0]) @ (np.eye(4) + b * operators[1]) / 4))
    return result


def reorder_pairs(rho):
    # A1,B1,A2,B2 -> A1,A2,B1,B2 on both bra and ket axes.
    return rho.reshape((2,) * 8).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(16, 16)


@lru_cache(None)
def born_components(kind, strategy, x, y):
    if strategy == "square":
        answers_a, answers_b = square_answers(x, "A"), square_answers(y, "B")
        states = [reorder_pairs(np.kron(a, b)) for a, b in ((BELL, BELL), (BELL, MIX), (MIX, BELL), (MIX, MIX))]
    else:
        answers_a, answers_b = qubit_answers(kind, strategy, x, "A"), qubit_answers(kind, strategy, y, "B")
        states = [BELL, MIX]
    answers, components = [], []
    for a, ea in answers_a:
        for b, eb in answers_b:
            effect = np.kron(ea, eb)
            answers.append((a, b))
            components.append([float(np.trace(rho @ effect).real) for rho in states])
    return answers, np.array(components)


def born_probabilities(kind, strategy, x, y, visibility):
    answers, components = born_components(kind, strategy, x, y)
    v = visibility
    weights = np.array([v * v, v * (1 - v), v * (1 - v), (1 - v) ** 2]) if strategy == "square" else np.array([v, 1 - v])
    probabilities = components @ weights
    assert probabilities.min() > -1e-10 and abs(probabilities.sum() - 1) < 1e-10
    probabilities = np.maximum(probabilities, 0)
    return answers, probabilities / probabilities.sum()


def pad_position(side, private_input, action):
    # Exactly the site labels in Figure 1: A=(E,W)/(N,S), B=(SW,NE)/(NW,SE).
    degrees = (((0, 180), (90, 270)) if side == "A" else ((225, 45), (135, 315)))[private_input][action]
    return RADIUS * np.array([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])


def geometry_success(kind, x, y, a, b):
    if kind == "A":
        return bool(np.linalg.norm(pad_position("A", x, a) - pad_position("B", y, b)) >= RADIUS)
    altitude_a = UPPER
    altitude_b = UPPER
    for mode in a:
        altitude_a = LOWER + UPPER - altitude_a if mode == -1 else altitude_a
    for mode in b:
        altitude_b = LOWER + UPPER - altitude_b if mode == -1 else altitude_b
    # Both local endpoint requirements AND the common cell's phase must hold.
    return bool(altitude_a == UPPER and altitude_b == LOWER and a[y] == b[x])


def classical_answers(kind, x, y):
    return (0, 0) if kind == "A" else (tuple(TABLE_A[x]), tuple(TABLE_B[:, y]))


def timely_response(kind, x, y, a):
    if kind == "A":
        return int(a) ^ (x * y)
    b = [1, 1, 1]
    b[x] = a[y]
    b[(x + 1) % 3] = -a[y]
    return tuple(b)


def success_probability(kind, strategy, v, q):
    if strategy == "table":
        p = 0.75 if kind == "A" else 8 / 9
    elif strategy == "chsh":
        p = 0.5 + v / (2 * math.sqrt(2))
    elif strategy == "six":
        p = 2 / 3 + math.sqrt(3) * v / 6
    elif strategy == "embedded":
        p = (7 + math.sqrt(2) * v) / 9
    else:
        p = 0.5 + (4 * v + 5 * v * v) / 18
    return q + (1 - q) * p


def block_trajectory(side, index, modes, dt=0.05):
    times = np.arange(0, 6 * SLOT + dt / 2, dt)
    start = index * SLOT if side == "A" else (index + 1) * SLOT
    progress = np.clip((times - start) / SLOT, 0, 3)
    distance = -CELL / 2 + CELL * progress
    altitude = np.full(times.shape, UPPER)
    current = UPPER
    for j, mode in enumerate(modes):
        target = LOWER + UPPER - current if mode == -1 else current
        in_cell = (progress >= j) & (progress <= j + 1)
        altitude[in_cell] = current + (target - current) * (progress[in_cell] - j)
        altitude[progress > j + 1] = target
        current = target
    fixed = np.full(times.shape, index * CELL)
    position = np.column_stack((distance, fixed, altitude)) if side == "A" else np.column_stack((fixed, distance, altitude))
    return times, position


def validate():
    checks = {"landing_action_cases": 0, "block_action_cases": 0, "safe_block_cases": 0}
    for x, y, a, b in itertools.product(range(2), repeat=4):
        assert geometry_success("A", x, y, a, b) == ((a ^ b) == x * y)
        checks["landing_action_cases"] += 1
    minimum = float("inf")
    for x, y in itertools.product(range(3), repeat=2):
        for a, b in itertools.product(list(itertools.product(SIGNS, repeat=3)), repeat=2):
            expected = math.prod(a) == 1 and math.prod(b) == -1 and a[y] == b[x]
            assert geometry_success("B", x, y, a, b) == expected
            checks["block_action_cases"] += 1
            if expected:
                _, pa = block_trajectory("A", x, a)
                _, pb = block_trajectory("B", y, b)
                sep = float(np.linalg.norm(pa - pb, axis=1).min())
                assert sep >= SAFE_SEPARATION
                minimum = min(minimum, sep)
                assert abs(pa[-1, 2] - UPPER) < 1e-10 and abs(pb[-1, 2] - LOWER) < 1e-10
                checks["safe_block_cases"] += 1
    checks["sampled_minimum_separation_m"] = minimum
    assert 2 * SLOT <= WINDOW < 2 * SLOT + PHASE_CHANGE
    assert RELEASE + 8 + 6 * SLOT <= LEG
    checks["same_phase_transit_s"] = 2 * SLOT
    checks["opposite_phase_transit_s"] = 2 * SLOT + PHASE_CHANGE
    checks["reservation_window_s"] = WINDOW
    checks["horizontal_speed_mps"] = CELL / SLOT
    checks["transfer_vertical_speed_mps"] = (UPPER - LOWER) / SLOT
    checks["analytic_value_checks"] = []
    for kind in ("A", "B"):
        n = 2 if kind == "A" else 3
        for strategy in OPTIONS[kind]:
            for v in (0.0, 0.5, 0.75, 0.95, 1.0):
                p = 0.0
                for x, y in itertools.product(range(n), repeat=2):
                    if strategy == "table":
                        p += geometry_success(kind, x, y, *classical_answers(kind, x, y)) / n ** 2
                    else:
                        answers, probs = born_probabilities(kind, strategy, x, y, v)
                        p += sum(pr * geometry_success(kind, x, y, a, b) for (a, b), pr in zip(answers, probs)) / n ** 2
                expected = success_probability(kind, strategy, v, 0)
                assert abs(p - expected) < 1e-10, (kind, strategy, v, p, expected)
                checks["analytic_value_checks"].append(dict(kind=kind, strategy=strategy, v=v, enumerated=p, expected=expected))
    return checks


@dataclass(frozen=True)
class Config:
    v0: float = 0.95
    tm: float | None = 2400.0
    inventory: int = 3
    q: float = 0.0
    refill: bool = False


class Planner:
    def __init__(self, config, objective="success", policy="optimal"):
        self.config, self.objective, self.policy = config, objective, policy
        self.choices = {}

    def visibility(self, age, planning=False):
        tm = None if planning and self.policy == "fresh" else self.config.tm
        return self.config.v0 * math.exp(-2 * age / tm) if tm else self.config.v0

    def state(self, t, e, age):
        if self.config.refill and t == 3:
            return self.config.inventory, RELEASE
        return e, age

    @lru_cache(None)
    def value(self, t, e, age):
        if t == T:
            return 0.0
        e, age = self.state(t, e, age)
        v = self.visibility(age, planning=True)
        total = EVENT_PRIOR["plain"] * self.value(t + 1, e, age + LEG)
        for kind in ("A", "B"):
            candidates = []
            for strategy in OPTIONS[kind]:
                k = PAIR_COST[strategy]
                if k > e or (self.policy == "table" and k):
                    continue
                p = success_probability(kind, strategy, v, self.config.q)
                instant = p if self.objective == "success" else -(1 - p) * RECOVERY[kind]
                continuation = (p * self.value(t + 1, e - k, age + LEG)
                                + (1 - p) * self.value(t + 1, e - k, age + LEG + RECOVERY[kind]))
                criterion = instant if self.policy == "myopic" else instant + continuation
                candidates.append((criterion, -k, strategy, instant + continuation))
            best = max(candidates)
            self.choices[(t, e, age, kind)] = best[2]
            total += EVENT_PRIOR[kind] * best[3]
        return total

    def choose(self, t, e, age, kind):
        self.value(t, e, age)
        e, age = self.state(t, e, age)
        return self.choices[(t, e, age, kind)]

    @lru_cache(None)
    def evaluate(self, t=0, e=None, age=RELEASE):
        e = self.config.inventory if e is None else e
        if t == T:
            return np.zeros(3)
        e, age = self.state(t, e, age)
        answer = EVENT_PRIOR["plain"] * self.evaluate(t + 1, e, age + LEG)
        for kind in ("A", "B"):
            strategy = self.choose(t, e, age, kind)
            k = PAIR_COST[strategy]
            p = success_probability(kind, strategy, self.visibility(age), self.config.q)
            instant = np.array([p, (1 - p) * RECOVERY[kind], k])
            answer += EVENT_PRIOR[kind] * (instant + p * self.evaluate(t + 1, e - k, age + LEG)
                                          + (1 - p) * self.evaluate(t + 1, e - k, age + LEG + RECOVERY[kind]))
        return answer


@lru_cache(100000)
def sampling_cdf(kind, strategy, x, y, v):
    answers, probabilities = born_probabilities(kind, strategy, x, y, v)
    return answers, np.cumsum(probabilities)


def simulate(planner, missions, seed=20261014):
    # Separate, policy-independent streams for task types, inputs, latency and outcomes.
    # The same mission index gets the same draws under every policy/configuration.
    streams = [np.random.default_rng(np.random.SeedSequence([seed, j])) for j in range(4)]
    kinds_u = streams[0].random((missions, T))
    inputs_u = streams[1].random((missions, T, 2))
    latency_u = streams[2].random((missions, T))
    outcomes_u = streams[3].random((missions, T))
    rows = np.zeros((missions, 3))
    for mission in range(missions):
        e, age = planner.config.inventory, RELEASE
        for t in range(T):
            e, age = planner.state(t, e, age)
            kind = "A" if kinds_u[mission, t] < 0.5 else "B" if kinds_u[mission, t] < 0.8 else "plain"
            if kind == "plain":
                age += LEG
                continue
            strategy = planner.choose(t, e, age, kind)
            k = PAIR_COST[strategy]
            n = 2 if kind == "A" else 3
            x, y = (inputs_u[mission, t] * n).astype(int)
            if strategy == "table":
                a, b = classical_answers(kind, x, y)
            else:
                answers, cdf = sampling_cdf(kind, strategy, int(x), int(y), planner.visibility(age))
                index = min(int(np.searchsorted(cdf, outcomes_u[mission, t], side="right")), len(answers) - 1)
                a, b = answers[index]
            if latency_u[mission, t] < planner.config.q:
                b = timely_response(kind, x, y, a)
            success = geometry_success(kind, x, y, a, b)
            delay = 0 if success else RECOVERY[kind]
            rows[mission] += (success, delay, k)
            e -= k
            age += LEG + delay
    return dict(mean=rows.mean(axis=0).tolist(), se=(rows.std(axis=0, ddof=1) / math.sqrt(missions)).tolist())


def campaign(missions):
    OUT.mkdir(exist_ok=True)
    validation = validate()
    scenarios = [("fixed", Config(tm=None, inventory=6)),
                 ("scarce", Config(tm=None, inventory=3)),
                 ("aging", Config(tm=2400, inventory=3)),
                 ("short", Config(tm=600, inventory=3)),
                 ("radio", Config(tm=2400, inventory=3, q=0.5)),
                 ("refill", Config(tm=2400, inventory=3, refill=True))]
    results = []
    for name, config in scenarios:
        for policy, objective in (("table", "success"), ("optimal", "success"), ("optimal", "delay"), ("fresh", "success")):
            planner = Planner(config, objective, policy)
            exact = planner.evaluate().tolist()
            mc = simulate(planner, missions)
            result = dict(scenario=name, config=asdict(config), policy=policy, objective=objective, exact=exact, mc=mc)
            results.append(result)
            print(name, policy, objective, "exact", np.round(exact, 4), "MC", np.round(mc["mean"], 4), flush=True)
    # Reproduce the existing Section 4 inventory values with static visibility.
    inventory_values = {}
    for v in (1.0, 0.95, 0.9, 0.85):
        inventory_values[str(v)] = [float(Planner(Config(v0=v, tm=None, inventory=e)).evaluate()[0]) for e in range(7)]
    sweeps = []
    for tm in (300, 450, 600, 900, 1200, 1800, 2400, 3600, 6000, 12000, 30000):
        cfg = Config(tm=tm, inventory=3)
        sweeps.append(dict(tm=tm, optimal=Planner(cfg).evaluate().tolist(),
                           time=Planner(cfg, "delay").evaluate().tolist(),
                           fresh=Planner(cfg, policy="fresh").evaluate().tolist(),
                           refill=Planner(Config(tm=tm, inventory=3, refill=True)).evaluate().tolist()))
    # Theorem 2(b) only applies to the fixed-quality case, where actions do not change later stage values.
    radio_validation = []
    for q in (0.0, 0.5, 0.9):
        cfg = Config(tm=None, inventory=3, q=q)
        quantum = Planner(cfg).evaluate()[0]
        classical = Planner(cfg, policy="table").evaluate()[0]
        blackout_gain = inventory_values["0.95"][3] - 3.85
        assert abs(quantum - classical - (1 - q) * blackout_gain) < 1e-10
        radio_validation.append(dict(q=q, quantum=float(quantum), classical=float(classical), gain=float(quantum - classical)))
    payload = dict(missions=missions, seed=20261014, validation=validation, results=results,
                   inventory_values=inventory_values, sweeps=sweeps, radio_validation=radio_validation,
                   parameters=dict(stages=T, nominal_leg_s=LEG, input_release_s=RELEASE,
                                   landing_recovery_s=RECOVERY["A"], block_recovery_s=RECOVERY["B"],
                                   radius_m=RADIUS, cell_m=CELL, cell_slot_s=SLOT, cell_window_s=WINDOW, phase_change_s=PHASE_CHANGE,
                                   lower_m=LOWER, upper_m=UPPER, minimum_separation_m=SAFE_SEPARATION))
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions", type=int, default=10000)
    arguments = parser.parse_args()
    campaign(arguments.missions)
