from __future__ import annotations

import math
import os
import time

import pandas as pd

from .query_generator import Condition, estimate_predicate_gate_cost, evaluate_query


def marked_indices(df: pd.DataFrame, conditions: list[Condition], missing_policy: str = "strict") -> set[int]:
    mask = evaluate_query(df, conditions, missing_policy=missing_policy)
    return set(map(int, df.index[mask]))


def estimate_oracle_resources(N: int, conditions: list[Condition], alpha_2q: float = 10.0, order_factor: float = 1.0) -> dict:
    n_qubits = int(math.log2(N))
    pred = [estimate_predicate_gate_cost(c, alpha_2q=alpha_2q) for c in conditions]
    pred_depth = sum(p["depth"] for p in pred)
    pred_cx = sum(p["N_2q"] for p in pred)
    pred_1q = sum(p["N_1q"] for p in pred)
    multi_control_depth = max(1, 4 * n_qubits + 2 * max(0, len(conditions) - 1))
    multi_control_cx = max(1, 6 * n_qubits + 4 * max(0, len(conditions) - 1))
    oracle_depth = int(math.ceil((pred_depth + multi_control_depth) * order_factor))
    cx = int(math.ceil((pred_cx + multi_control_cx) * order_factor))
    oneq = int(math.ceil((pred_1q + 4 * n_qubits) * order_factor))
    total = oneq + cx
    return {
        "oracle_depth": oracle_depth,
        "CX_count": cx,
        "single_qubit_gate_count": oneq,
        "total_gate_count": total,
        "ancilla_count": max(1, len(conditions)),
        "total_qubits": n_qubits + max(1, len(conditions)),
        "index_qubits": n_qubits,
        "resource_source": "heuristic_estimate",
    }


def qiskit_oracle_resources(N: int, marked: set[int], conditions: list[Condition]) -> dict | None:
    """Return transpiled Qiskit metrics for small compiled phase oracles.

    This is intentionally limited to small N so full experiments remain
    tractable. Larger N keep the transparent heuristic estimates.
    """
    if os.environ.get("CARE_USE_QISKIT", "0") != "1":
        return None
    if N > 32 or not marked:
        return None
    try:
        from qiskit import QuantumCircuit, transpile
    except Exception:
        return None

    n_qubits = int(math.log2(N))
    if n_qubits < 1:
        return None
    start = time.perf_counter()
    qc = QuantumCircuit(n_qubits, name="compiled_predicate_oracle")
    for idx in sorted(marked):
        bits = format(idx, f"0{n_qubits}b")
        zero_qubits = [q for q, bit in enumerate(reversed(bits)) if bit == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    tqc = transpile(qc, basis_gates=["x", "h", "z", "cx", "ccx", "u1", "u2", "u3"], optimization_level=1)
    counts = tqc.count_ops()
    oneq = sum(int(counts.get(g, 0)) for g in ["x", "h", "z", "u1", "u2", "u3", "rz", "sx"])
    cx = int(counts.get("cx", 0) + counts.get("ccx", 0) * 6)
    return {
        "oracle_depth": int(tqc.depth() or 0),
        "CX_count": cx,
        "single_qubit_gate_count": oneq,
        "total_gate_count": int(sum(counts.values())),
        "ancilla_count": max(1, len(conditions)),
        "total_qubits": n_qubits,
        "index_qubits": n_qubits,
        "transpilation_time": time.perf_counter() - start,
        "resource_source": "qiskit_transpiled_small_N",
    }


def build_compiled_predicate_oracle(df: pd.DataFrame, conditions: list[Condition], alpha_2q: float, order_factor: float = 1.0, missing_policy: str = "strict") -> dict:
    start = time.perf_counter()
    marked = marked_indices(df, conditions, missing_policy=missing_policy)
    construction_time = time.perf_counter() - start
    res = estimate_oracle_resources(len(df), conditions, alpha_2q=alpha_2q, order_factor=order_factor)
    qiskit_res = qiskit_oracle_resources(len(df), marked, conditions)
    if qiskit_res is not None:
        qiskit_res["oracle_depth"] = int(math.ceil(qiskit_res["oracle_depth"] * order_factor))
        qiskit_res["CX_count"] = int(math.ceil(qiskit_res["CX_count"] * order_factor))
        qiskit_res["single_qubit_gate_count"] = int(math.ceil(qiskit_res["single_qubit_gate_count"] * order_factor))
        qiskit_res["total_gate_count"] = qiskit_res["single_qubit_gate_count"] + qiskit_res["CX_count"]
        res = qiskit_res
    return {
        "oracle_type": "compiled_clinical_predicate_truth_table",
        "marked_indices": marked,
        "oracle_construction_time": construction_time,
        "transpilation_time": res.pop("transpilation_time", 0.0),
        **res,
    }
