from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def retrieval_metrics(true_set: Iterable[int], retrieved_set: Iterable[int], lambda_fn: float = 5.0, lambda_fp: float = 1.0) -> dict:
    true_set = set(map(int, true_set))
    retrieved_set = set(map(int, retrieved_set))
    tp = len(true_set & retrieved_set)
    fp = len(retrieved_set - true_set)
    fn = len(true_set - retrieved_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "false_negative_rate": fn / len(true_set) if true_set else 0.0,
        "clinical_loss": lambda_fn * fn + lambda_fp * fp,
    }


def grover_success_probability(N: int, M: int, iterations: int) -> float:
    if M <= 0:
        return 0.0
    if M >= N:
        return 1.0
    theta = math.asin(math.sqrt(M / N))
    return float(math.sin((2 * iterations + 1) * theta) ** 2)


def optimal_iterations(N: int, M: int) -> int:
    if M <= 0:
        return 0
    return max(0, int(math.floor(math.pi / 4.0 * math.sqrt(N / M))))


def adaptive_iterations(N: int, M_estimate: int | None = None) -> int:
    # Unknown-M schedule: a conservative BBHT-like single-stage proxy that
    # avoids using the true M in the main CARE adaptive method.
    if N <= 2:
        return 1
    return max(1, int(round(math.sqrt(N) / 2.0)))


def bbht_schedule(N: int, seed: int, growth: float = 1.2, max_rounds: int = 8) -> list[int]:
    """Unknown-M Grover schedule inspired by Boyer-Brassard-Hoyer-Tapp.

    The schedule does not use the true marked count. Each round samples a
    Grover iteration count below a growing bound, capped by sqrt(N).
    """
    rng = np.random.default_rng(seed)
    max_m = max(1, int(math.ceil(math.sqrt(N))))
    m = 1
    schedule: list[int] = []
    for _ in range(max_rounds):
        upper = max(1, min(max_m, int(math.ceil(m))))
        schedule.append(int(rng.integers(0, upper)))
        if upper >= max_m:
            break
        m = min(max_m, math.ceil(growth * upper))
    return schedule


def bbht_success_probability(N: int, M: int, schedule: list[int]) -> tuple[float, int]:
    fail = 1.0
    calls = 0
    for r in schedule:
        p = grover_success_probability(N, M, r)
        fail *= 1.0 - p
        calls += 2 * r + 1
    return float(1.0 - fail), int(calls)


def ci95(values: Iterable[float]) -> tuple[float, float]:
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    mean = float(vals.mean())
    half = 1.96 * float(vals.std(ddof=1)) / math.sqrt(len(vals))
    return mean - half, mean + half
