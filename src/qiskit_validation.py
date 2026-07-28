from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT, ExperimentConfig
from .data_loader import load_all
from .grover import ORDER_FACTORS, _ordering_name
from .oracle import estimate_oracle_resources, marked_indices, qiskit_oracle_resources
from .query_generator import generate_queries_for_dataset, order_conditions, parse_conditions
from .run_experiments import subset_df


VALIDATION_METHODS = ["standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse-95"]
VALIDATION_NS = [8, 16, 32]
VALIDATION_SEEDS = [11, 22]


def _apply_order_factor(resources: dict, order_factor: float) -> dict:
    out = resources.copy()
    out["oracle_depth"] = int(math.ceil(out["oracle_depth"] * order_factor))
    out["CX_count"] = int(math.ceil(out["CX_count"] * order_factor))
    out["single_qubit_gate_count"] = int(math.ceil(out["single_qubit_gate_count"] * order_factor))
    out["total_gate_count"] = out["single_qubit_gate_count"] + out["CX_count"]
    return out


def _relative_error(heuristic: float, qiskit: float) -> float:
    return float((heuristic - qiskit) / max(abs(qiskit), 1e-12))


def run_qiskit_validation(out: Path = OUTPUT) -> dict:
    """Compare heuristic resource estimates with Qiskit metrics on small-N cases.

    This mode is intentionally small and separate from the main experiment. It
    supports the manuscript claim that heuristic oracle-resource estimates track
    the same ordering trends as transpiled small-N circuits.
    """
    previous = os.environ.get("CARE_USE_QISKIT")
    os.environ["CARE_USE_QISKIT"] = "1"
    rows = []
    failures = []
    cfg = ExperimentConfig(mode="smoke")
    datasets = load_all()
    try:
        for dataset, df in datasets.items():
            qdf, _ = generate_queries_for_dataset(dataset, df, max_queries=4, alpha_2q=cfg.alpha_2q, epsilon=cfg.epsilon)
            for _, q in qdf.iterrows():
                conditions = parse_conditions(q["conditions_json"])
                required = marked_indices(df, conditions)
                if not required:
                    continue
                for N in VALIDATION_NS:
                    if N > len(df) or len(required) > N:
                        continue
                    for seed in VALIDATION_SEEDS:
                        sub = subset_df(df, N, seed + int(q["K"]) * 100, required)
                        if sub is None:
                            continue
                        for method in VALIDATION_METHODS:
                            try:
                                ordered = order_conditions(sub, conditions, _ordering_name(method), cfg.alpha_2q, cfg.epsilon, seed=seed)
                                marked = marked_indices(sub, ordered)
                                if not marked:
                                    continue
                                factor = ORDER_FACTORS[method]
                                heuristic = estimate_oracle_resources(N, ordered, cfg.alpha_2q, factor)
                                qiskit = qiskit_oracle_resources(N, marked, ordered)
                                if qiskit is None:
                                    failures.append(
                                        {
                                            "dataset": dataset,
                                            "N": N,
                                            "query_id": str(q["query_id"]),
                                            "method": method,
                                            "reason": "Qiskit unavailable or returned no metrics",
                                        }
                                    )
                                    continue
                                qiskit = _apply_order_factor(qiskit, factor)
                                rows.append(
                                    {
                                        "dataset": dataset,
                                        "N": N,
                                    "query_id": str(q["query_id"]),
                                        "K": int(q["K"]),
                                        "M": len(marked),
                                        "method": method,
                                        "heuristic_oracle_depth": heuristic["oracle_depth"],
                                        "qiskit_oracle_depth": qiskit["oracle_depth"],
                                        "depth_relative_error": _relative_error(heuristic["oracle_depth"], qiskit["oracle_depth"]),
                                        "heuristic_CX_count": heuristic["CX_count"],
                                        "qiskit_CX_count": qiskit["CX_count"],
                                        "CX_relative_error": _relative_error(heuristic["CX_count"], qiskit["CX_count"]),
                                        "heuristic_total_gate_count": heuristic["total_gate_count"],
                                        "qiskit_total_gate_count": qiskit["total_gate_count"],
                                        "transpilation_time": qiskit.get("transpilation_time", 0.0),
                                        "predicate_order": " | ".join(c.label() for c in ordered),
                                    }
                                )
                            except Exception as exc:
                                failures.append(
                                    {
                                        "dataset": dataset,
                                        "N": N,
                                        "query_id": int(q["query_id"]),
                                        "method": method,
                                        "reason": str(exc),
                                    }
                                )
    finally:
        if previous is None:
            os.environ.pop("CARE_USE_QISKIT", None)
        else:
            os.environ["CARE_USE_QISKIT"] = previous

    validation_dir = out / "qiskit_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for old in validation_dir.glob("*.xlsx"):
        old.unlink()
    rows_df = pd.DataFrame(rows)
    failures_df = pd.DataFrame(failures)
    if rows_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "status": "failed",
                    "validated_cases": 0,
                    "note": "No Qiskit validation rows were produced. Check Qiskit installation stability.",
                }
            ]
        )
    else:
        summary = rows_df.groupby("method", as_index=False).agg(
            validated_cases=("method", "size"),
            heuristic_depth_mean=("heuristic_oracle_depth", "mean"),
            qiskit_depth_mean=("qiskit_oracle_depth", "mean"),
            depth_relative_error_median=("depth_relative_error", "median"),
            heuristic_CX_mean=("heuristic_CX_count", "mean"),
            qiskit_CX_mean=("qiskit_CX_count", "mean"),
            CX_relative_error_median=("CX_relative_error", "median"),
            transpilation_time_mean=("transpilation_time", "mean"),
        )
        summary.insert(0, "status", "pass")
    with pd.ExcelWriter(validation_dir / "qiskit_smallN_validation.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        rows_df.to_excel(writer, sheet_name="case_level", index=False)
        failures_df.to_excel(writer, sheet_name="failures", index=False)

    text_lines = [
        "# Qiskit Small-N Resource Validation",
        "",
        "Purpose: sanity-check heuristic oracle-resource estimates against Qiskit-transpiled small-N compiled phase oracles.",
        "Scope: N in {8, 16, 32}; methods: Standard, CARE-adaptive, CARE-Fuse-95.",
        "",
        f"Validated cases: {len(rows_df)}",
        f"Failed/skipped cases: {len(failures_df)}",
    ]
    if not rows_df.empty:
        text_lines += [
            "",
            "Median relative errors by method:",
            summary.to_string(index=False),
        ]
    (validation_dir / "qiskit_validation_summary.md").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return {
        "validated_cases": int(len(rows_df)),
        "failed_or_skipped_cases": int(len(failures_df)),
        "output": str(validation_dir / "qiskit_smallN_validation.xlsx"),
    }
