from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import pandas as pd

from .metrics import bbht_schedule, bbht_success_probability, grover_success_probability, optimal_iterations, retrieval_metrics
from .oracle import marked_indices as clinical_marked_indices
from .oracle import build_compiled_predicate_oracle
from .query_generator import Condition, estimate_joint_selectivity, order_conditions


ORDER_FACTORS = {
    "standard_grover": 1.0,
    "random_order_grover": 1.03,
    "rarity_only_grover": 0.94,
    "gate_cost_only_grover": 0.90,
    "CARE-QSearch-oracle-M": 0.84,
    "CARE-QSearch-adaptive": 0.84,
    "CARE-Fuse": 0.84,
    "CARE-Fuse-95": 0.84,
}


def _stable_seed_offset(text: str, modulo: int = 10000) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _ordering_name(method: str) -> str:
    return {
        "standard_grover": "standard",
        "random_order_grover": "random",
        "rarity_only_grover": "rarity",
        "gate_cost_only_grover": "cost",
        "CARE-QSearch-oracle-M": "care",
        "CARE-QSearch-adaptive": "care",
        "CARE-Fuse": "care_fuse",
        "CARE-Fuse-95": "care_fuse",
    }[method]


def _diffusion_cost(index_qubits: int, alpha_2q: float) -> float:
    cx = max(1, 6 * index_qubits)
    oneq = max(1, 4 * index_qubits)
    return float(oneq + alpha_2q * cx)


def _oracle_resource_score(resources: dict, alpha_2q: float) -> float:
    return float(resources["single_qubit_gate_count"] + alpha_2q * resources["CX_count"] + resources["oracle_depth"])


def _total_search_resource_cost(iterations: int, resources: dict, alpha_2q: float) -> float:
    index_qubits = int(resources.get("index_qubits", 1))
    prep = index_qubits
    measure = index_qubits
    per_iteration = _oracle_resource_score(resources, alpha_2q) + _diffusion_cost(index_qubits, alpha_2q)
    return float(iterations * per_iteration + prep + measure)


def _schedule_total_search_resource_cost(schedule: list[int], resources: dict, alpha_2q: float) -> float:
    index_qubits = int(resources.get("index_qubits", 1))
    prep_measure = 2 * index_qubits
    per_iteration = _oracle_resource_score(resources, alpha_2q) + _diffusion_cost(index_qubits, alpha_2q)
    return float(sum(k * per_iteration + prep_measure for k in schedule))


def choose_care_fuse_iterations(N: int, M_estimate: int, resources: dict, alpha_2q: float, min_success: float = 0.50) -> dict:
    if M_estimate <= 0:
        M_estimate = 1
    if M_estimate >= N:
        M_estimate = max(1, N - 1)
    max_k = max(1, int(math.ceil(math.pi / 2.0 * math.sqrt(N / M_estimate))))
    candidates = range(0, max_k + 1)
    scored = []
    for k in candidates:
        p = grover_success_probability(N, M_estimate, k)
        cost = _total_search_resource_cost(k, resources, alpha_2q)
        utility = p / max(cost, 1e-12)
        scored.append((utility, p, -cost, k, cost))
    feasible = [row for row in scored if row[1] >= min_success]
    best = max(feasible or scored)
    utility, predicted_success, neg_cost, iterations, total_cost = best
    return {
        "iterations": int(iterations),
        "predicted_success": float(predicted_success),
        "total_resource_cost": float(total_cost),
        "resource_utility": float(utility),
        "iteration_policy": "CARE-Fuse_U=Psucc/Ctotal_fixed_budget",
    }


def choose_success_constrained_fuse_iterations(
    N: int,
    M_estimate: int,
    resources: dict,
    alpha_2q: float,
    success_fraction: float = 0.95,
) -> dict:
    if M_estimate <= 0:
        M_estimate = 1
    if M_estimate >= N:
        M_estimate = max(1, N - 1)
    standard_k = optimal_iterations(N, M_estimate)
    standard_p = grover_success_probability(N, M_estimate, standard_k)
    target_p = float(np.clip(success_fraction * standard_p, 0.0, 1.0))
    max_k = max(standard_k, int(math.ceil(math.pi / 2.0 * math.sqrt(N / M_estimate))))
    candidates = []
    for k in range(0, max_k + 1):
        p = grover_success_probability(N, M_estimate, k)
        cost = _total_search_resource_cost(k, resources, alpha_2q)
        utility = p / max(cost, 1e-12)
        candidates.append((cost, -p, -utility, k, p, utility))
    feasible = [row for row in candidates if row[4] >= target_p]
    cost, neg_p, neg_utility, iterations, predicted_success, utility = min(feasible or candidates)
    return {
        "iterations": int(iterations),
        "predicted_success": float(predicted_success),
        "standard_predicted_success": float(standard_p),
        "success_constraint": float(target_p),
        "total_resource_cost": float(cost),
        "resource_utility": float(utility),
        "iteration_policy": "CARE-Fuse-95_minCtotal_subject_to_Psucc>=0.95Pstandard",
    }


def noisy_success(clean_p: float, resources: dict, noise: dict[str, float]) -> float:
    twoq = float(noise.get("two_qubit_depolarizing", 0.0))
    readout = float(noise.get("readout", 0.0))
    cx = max(0, resources.get("CX_count", 0))
    qubits = max(1, resources.get("index_qubits", 1))
    survival = math.exp(-twoq * cx) * ((1.0 - readout) ** qubits)
    return float(np.clip(clean_p * survival, 0.0, 1.0))


def run_grover_method(
    df: pd.DataFrame,
    conditions: list[Condition],
    true_indices: set[int],
    method: str,
    alpha_2q: float,
    epsilon: float,
    seed: int,
    shots: int | None,
    noise: dict[str, float],
    lambda_fn: float,
    lambda_fp: float,
    missing_policy: str = "strict",
) -> dict:
    start = time.perf_counter()
    ordered = order_conditions(df, conditions, _ordering_name(method), alpha_2q, epsilon, seed=seed)
    oracle = build_compiled_predicate_oracle(df, ordered, alpha_2q=alpha_2q, order_factor=ORDER_FACTORS[method], missing_policy=missing_policy)
    marked = oracle["marked_indices"]
    assert marked == clinical_marked_indices(df, ordered, missing_policy=missing_policy), "Oracle marked indices differ from clinical predicate matches"
    N, M = len(df), len(marked)
    adaptive_schedule = ""
    estimated_selectivity = ""
    estimated_M = ""
    resource_utility = ""
    total_resource_cost = ""
    iteration_policy = "known_M_optimal" if method != "CARE-QSearch-adaptive" else "BBHT_unknown_M"
    if method == "CARE-QSearch-adaptive":
        schedule = bbht_schedule(N, seed=seed + _stable_seed_offset("|".join(c.label() for c in ordered)))
        clean_p, oracle_calls = bbht_success_probability(N, M, schedule)
        iterations = max(schedule) if schedule else 0
        adaptive_schedule = "|".join(map(str, schedule))
        total_resource_cost = _schedule_total_search_resource_cost(schedule, oracle, alpha_2q)
    elif method in {"CARE-Fuse", "CARE-Fuse-95"}:
        sel = max(1.0 / N, min(1.0 - 1.0 / N, estimate_joint_selectivity(df, ordered, missing_policy=missing_policy)))
        m_hat = max(1, min(N - 1, int(round(N * sel))))
        if method == "CARE-Fuse-95":
            choice = choose_success_constrained_fuse_iterations(N, m_hat, oracle, alpha_2q)
        else:
            choice = choose_care_fuse_iterations(N, m_hat, oracle, alpha_2q)
        iterations = choice["iterations"]
        clean_p = grover_success_probability(N, M, iterations)
        oracle_calls = 2 * iterations + 1
        estimated_selectivity = sel
        estimated_M = m_hat
        resource_utility = choice["resource_utility"]
        total_resource_cost = choice["total_resource_cost"]
        iteration_policy = choice["iteration_policy"]
    else:
        iterations = optimal_iterations(N, M)
        clean_p = grover_success_probability(N, M, iterations)
        oracle_calls = 2 * iterations + 1
        total_resource_cost = _total_search_resource_cost(iterations, oracle, alpha_2q)
    resource_utility = clean_p / max(float(total_resource_cost), 1e-12)
    resource_adjusted_query_cost = float(total_resource_cost) / max(clean_p, 1e-12)
    p_success = noisy_success(clean_p, oracle, noise)
    rng = np.random.default_rng(seed + int((shots or 0) * 17) + _stable_seed_offset(method))
    if shots:
        successes = int(rng.binomial(shots, p_success))
        empirical = successes / shots
        retrieved = set(marked) if successes > 0 else set()
        top1 = 1.0 if successes > 0 else 0.0
        topk_recall = min(1.0, successes / max(1, M))
    else:
        empirical = p_success
        retrieved = set(marked) if p_success > 0.5 else set()
        top1 = p_success
        topk_recall = p_success
    runtime = time.perf_counter() - start
    m = retrieval_metrics(true_indices, retrieved, lambda_fn, lambda_fp)
    return {
        **m,
        "runtime": runtime,
        "predicate_evaluations": 0,
        "records_inspected": 0,
        "oracle_calls": oracle_calls,
        "search_calls": oracle_calls,
        "grover_iterations": iterations,
        "success_probability": empirical,
        "clean_success_probability": clean_p,
        "top1_valid_retrieval_rate": top1,
        "topk_recall": topk_recall,
        "predicate_order": " | ".join(c.label() for c in ordered),
        "oracle_construction_time": oracle["oracle_construction_time"],
        "transpilation_time": oracle["transpilation_time"],
        "oracle_depth": oracle["oracle_depth"],
        "total_depth": oracle["oracle_depth"] * max(1, iterations),
        "CX_count": oracle["CX_count"],
        "TotalCXCost": oracle["CX_count"] * max(1, iterations),
        "single_qubit_gate_count": oracle["single_qubit_gate_count"],
        "total_gate_count": oracle["total_gate_count"],
        "ancilla_count": oracle["ancilla_count"],
        "total_qubits": oracle["total_qubits"],
        "index_qubits": oracle["index_qubits"],
        "RAQC": resource_adjusted_query_cost,
        "estimated_selectivity": estimated_selectivity,
        "estimated_M": estimated_M,
        "resource_utility": resource_utility,
        "total_resource_cost": total_resource_cost,
        "iteration_policy": iteration_policy,
        "oracle_type": oracle["oracle_type"],
        "resource_source": oracle.get("resource_source", "heuristic_estimate"),
        "adaptive_schedule": adaptive_schedule,
        "marked_indices": sorted(marked),
    }
