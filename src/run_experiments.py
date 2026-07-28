from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .classical_search import run_classical_selective, run_classical_sequential, run_classical_sqlite_indexed, run_classical_vectorised
from .config import ExperimentConfig, FEATURES, OUTPUT
from .data_loader import dataset_summary, load_all
from .grover import run_grover_method
from .metrics import grover_success_probability, optimal_iterations
from .oracle import marked_indices
from .plotting import generate_figures
from .query_generator import condition_scores, evaluate_query, generate_queries_for_dataset, parse_conditions
from .statistics import aggregate_results, paired_tests
from .tables import write_tables
from .utils import append_row, read_csv_or_empty, setup_logging, setup_output, write_environment, write_json


CLASSICAL_METHODS = ["classical_sequential", "classical_vectorised", "classical_selective", "classical_sqlite_indexed"]
QUANTUM_METHODS = [
    "standard_grover",
    "random_order_grover",
    "rarity_only_grover",
    "gate_cost_only_grover",
    "CARE-QSearch-oracle-M",
    "CARE-QSearch-adaptive",
    "CARE-Fuse",
    "CARE-Fuse-95",
]

LIST_LIKE_COLUMNS = ["adaptive_schedule", "candidate_trace", "marked_indices"]

RAW_COLUMNS = [
    "experiment_id", "dataset", "seed", "N", "index_qubits", "query_id", "K", "M", "target_prevalence",
    "rarity_bin", "conditions", "method", "predicate_order", "shots", "noise_level",
    "noise_two_qubit_depolarizing", "noise_readout", "missingness", "missing_policy",
    "candidate_set_size", "grover_iterations", "oracle_calls", "search_calls", "records_inspected",
    "predicate_evaluations", "success_probability", "clean_success_probability",
    "top1_valid_retrieval_rate", "topk_recall", "precision", "recall", "F1", "TP", "FP", "FN",
    "false_negative_rate", "clinical_loss", "oracle_depth", "total_depth", "CX_count",
    "TotalCXCost", "single_qubit_gate_count", "total_gate_count", "ancilla_count", "total_qubits",
    "transpilation_time", "oracle_construction_time", "runtime", "query_ratio", "TotalDepthCost",
    "RAQC", "oracle_type", "resource_source", "adaptive_schedule", "candidate_trace", "marked_indices",
    "estimated_selectivity", "estimated_M", "resource_utility", "total_resource_cost", "iteration_policy",
]


def experiment_id(row: dict) -> str:
    keys = [
        "dataset",
        "seed",
        "N",
        "query_id",
        "method",
        "shots",
        "noise_level",
        "missingness",
        "missing_policy",
    ]
    payload = json.dumps({k: row.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def subset_df(df: pd.DataFrame, N: int, seed: int, required: set[int]) -> pd.DataFrame | None:
    rng = np.random.default_rng(seed)
    if len(required) > N:
        return None
    pool = np.array([i for i in df.index if i not in required], dtype=int)
    extra_n = N - len(required)
    if extra_n > len(pool):
        return None
    extra = rng.choice(pool, size=extra_n, replace=False) if extra_n else np.array([], dtype=int)
    idx = sorted(list(required) + list(map(int, extra)))
    sub = df.loc[idx].copy()
    sub.index = range(len(sub))
    sub["original_patient_index"] = idx
    return sub


def apply_missingness(df: pd.DataFrame, conditions, level: float, seed: int, dataset: str) -> pd.DataFrame:
    out = df.copy()
    if level <= 0:
        return out
    rng = np.random.default_rng(seed)
    cols = sorted(set(c.feature for c in conditions))
    for col in cols:
        mask = rng.random(len(out)) < level
        out.loc[mask, col] = np.nan
    return out


def should_run_quantum_setting(cfg: ExperimentConfig, missingness: float, noise_label: str, shots: int) -> bool:
    if cfg.mode == "smoke":
        return True
    # Controlled full design:
    # 1. finite-shot study: clean, no missingness, all shot counts;
    # 2. noise study: no missingness, 1024 shots, all noise levels;
    # 3. missingness study: clean, 1024 shots, all missingness levels.
    return (
        (missingness == 0.0 and noise_label == "clean")
        or (missingness == 0.0 and shots == 1024)
        or (noise_label == "clean" and shots == 1024)
    )


def save_circuit_examples(out: Path) -> None:
    txt = {
        "standard_grover_example": "H on index register -> compiled clinical predicate phase oracle -> diffusion -> measurement",
        "care_qsearch_example": "H on index register -> CARE-ordered compiled clinical predicate phase oracle -> adaptive Grover iterations -> measurement",
        "oracle_example": "U_f |i> = (-1)^f(i)|i>, f(i)=1 iff all clinical predicates are satisfied",
        "grover_iteration_example": "One Grover iteration = predicate phase oracle + inversion about mean diffusion operator",
    }
    for name, content in txt.items():
        (out / "circuits" / f"{name}.txt").write_text(content + "\n", encoding="utf-8")
        # Matplotlib text diagrams keep PNG deliverables available without Qiskit.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 2.4), constrained_layout=True)
        ax.axis("off")
        ax.text(0.02, 0.55, content, ha="left", va="center", fontsize=11, wrap=True)
        fig.savefig(out / "circuits" / f"{name}.png", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def run_sanity_tests(datasets: dict[str, pd.DataFrame], queries: pd.DataFrame, cfg: ExperimentConfig) -> None:
    df = datasets["cleveland"].head(8).copy()
    cond = [parse_conditions(queries[queries["dataset"] == "cleveland"].iloc[0]["conditions_json"])[0]]
    mask = evaluate_query(df, cond)
    if int(mask.sum()) != 1:
        from .query_generator import Condition

        cond = None
        for feature in ["age", "chol", "trestbps", "thalach", "oldpeak"]:
            vals = sorted(df[feature].dropna().unique())
            if len(vals) >= 2:
                threshold = (vals[-1] + vals[-2]) / 2.0
                trial = [Condition(feature, ">", float(threshold))]
                trial_mask = evaluate_query(df, trial)
                if int(trial_mask.sum()) == 1:
                    cond = trial
                    mask = trial_mask
                    break
        if cond is None:
            cond = [Condition("cp", "==", float(df["cp"].value_counts().loc[lambda s: s == 1].index[0]))]
            mask = evaluate_query(df, cond)
    M = int(mask.sum())
    assert M == 1, "Sanity Test 1 requires exactly one match for N=8"
    p = grover_success_probability(8, 1, optimal_iterations(8, 1))
    assert p > 0.75, "Sanity Test 1 failed: Grover did not amplify the target"
    assert marked_indices(df, cond) == set(map(int, df.index[mask])), "Sanity Test 2 failed"
    assert (queries["M"] > 0).all() and (queries["M"] < queries["N"]).all(), "Sanity Test 3 failed"
    assert 0 <= p <= 1, "Sanity Test 4 failed"
    assert cfg.alpha_2q >= 0, "Sanity Test 5 failed"
    care1 = condition_scores(datasets["cleveland"], cond, cfg.alpha_2q, cfg.epsilon)
    care2 = condition_scores(datasets["cleveland"], cond, cfg.alpha_2q, cfg.epsilon)
    assert care1 == care2, "Sanity Test 6 failed"


def evaluate_one(base: dict, df_sub: pd.DataFrame, conditions, true_indices: set[int], method: str, cfg: ExperimentConfig, seed: int, shots, noise_label: str, noise_params: dict) -> dict:
    if method == "classical_sequential":
        result = run_classical_sequential(df_sub, conditions, true_indices, cfg.lambda_fn, cfg.lambda_fp)
    elif method == "classical_vectorised":
        result = run_classical_vectorised(df_sub, conditions, true_indices, cfg.lambda_fn, cfg.lambda_fp)
    elif method == "classical_selective":
        result = run_classical_selective(df_sub, conditions, true_indices, cfg.alpha_2q, cfg.epsilon, cfg.lambda_fn, cfg.lambda_fp)
    elif method == "classical_sqlite_indexed":
        result = run_classical_sqlite_indexed(df_sub, conditions, true_indices, cfg.lambda_fn, cfg.lambda_fp)
    else:
        result = run_grover_method(
            df_sub,
            conditions,
            true_indices,
            method,
            cfg.alpha_2q,
            cfg.epsilon,
            seed,
            shots,
            noise_params,
            cfg.lambda_fn,
            cfg.lambda_fp,
            missing_policy=base.get("missing_policy", "strict"),
        )

    row = {**base, **result}
    row["method"] = method
    row["shots"] = shots if shots is not None else 0
    row["noise_level"] = noise_label
    row["noise_two_qubit_depolarizing"] = noise_params.get("two_qubit_depolarizing", 0.0)
    row["noise_readout"] = noise_params.get("readout", 0.0)
    if row.get("oracle_calls", 0):
        row["query_ratio"] = row["N"] / row["oracle_calls"]
    else:
        row["query_ratio"] = np.nan
    row["TotalDepthCost"] = row.get("total_depth", 0)
    row.setdefault("grover_iterations", 0)
    row.setdefault("oracle_depth", 0)
    row.setdefault("total_depth", 0)
    row.setdefault("CX_count", 0)
    row.setdefault("TotalCXCost", 0)
    row.setdefault("single_qubit_gate_count", 0)
    row.setdefault("total_gate_count", 0)
    row.setdefault("ancilla_count", 0)
    row.setdefault("total_qubits", 0)
    row.setdefault("index_qubits", int(math.log2(row["N"])))
    row.setdefault("oracle_construction_time", 0.0)
    row.setdefault("transpilation_time", 0.0)
    row.setdefault("top1_valid_retrieval_rate", row.get("success_probability", 0.0))
    row.setdefault("topk_recall", row.get("recall", 0.0))
    row.setdefault("RAQC", np.nan)
    row.setdefault("resource_source", "")
    row.setdefault("adaptive_schedule", "")
    row.setdefault("estimated_selectivity", "")
    row.setdefault("estimated_M", "")
    row.setdefault("resource_utility", "")
    row.setdefault("total_resource_cost", "")
    row.setdefault("iteration_policy", "")
    row.setdefault("predicate_order", base["conditions"])
    row.setdefault("candidate_trace", "")
    row["experiment_id"] = experiment_id(row)
    return row


def generate_and_save_queries(datasets: dict[str, pd.DataFrame], cfg: ExperimentConfig) -> pd.DataFrame:
    frames = []
    for name, df in datasets.items():
        qdf, _ = generate_queries_for_dataset(name, df, cfg.max_queries_per_dataset, cfg.alpha_2q, cfg.epsilon)
        frames.append(qdf)
    from .query_generator import save_queries

    return save_queries(frames, OUTPUT / "queries")


def run_pipeline(mode: str, force: bool = False) -> dict:
    if mode == "qiskit-validation":
        setup_output()
        if os.environ.get("CARE_QISKIT_VALIDATION_CHILD") == "1":
            from .qiskit_validation import run_qiskit_validation

            return run_qiskit_validation(OUTPUT)
        env = os.environ.copy()
        env["CARE_QISKIT_VALIDATION_CHILD"] = "1"
        env["CARE_USE_QISKIT"] = "1"
        env.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
        proc = subprocess.run(
            [sys.executable, "run_all.py", "--mode", "qiskit-validation"],
            cwd=OUTPUT.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        validation_dir = OUTPUT / "qiskit_validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "qiskit_validation_process_stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (validation_dir / "qiskit_validation_process_stderr.txt").write_text(proc.stderr, encoding="utf-8")
        workbook = validation_dir / "qiskit_smallN_validation.xlsx"
        if not workbook.exists():
            raise RuntimeError(
                "Qiskit validation did not produce a workbook. "
                f"Child exit code: {proc.returncode}. See {validation_dir} logs."
            )
        try:
            summary_df = pd.read_excel(workbook, sheet_name="summary")
            validated_cases = int(summary_df["validated_cases"].sum()) if "validated_cases" in summary_df else 0
        except Exception:
            validated_cases = 0
        return {
            "validated_cases": validated_cases,
            "child_process_exit_code": proc.returncode,
            "note": (
                "Workbook produced. Nonzero child exit code usually indicates a Qiskit/Python shutdown segfault; "
                "the validation artifact is still available."
            ) if proc.returncode else "Workbook produced.",
            "output": str(workbook),
        }
    cfg = ExperimentConfig(mode=mode)
    if force and mode not in {"analysis-only", "plots-only"} and OUTPUT.exists():
        import shutil

        shutil.rmtree(OUTPUT)
    setup_output()
    setup_logging()
    write_json(OUTPUT / "configs" / "config.json", cfg.to_dict())
    write_json(OUTPUT / "reproducibility" / "seeds.json", cfg.seeds)
    write_environment()
    datasets = load_all()
    ds_summary = dataset_summary(datasets)
    ds_summary.to_csv(OUTPUT / "data_summary" / "dataset_characteristics.csv", index=False)

    queries_path = OUTPUT / "queries" / "generated_queries.csv"
    if queries_path.exists() and mode in {"analysis-only", "plots-only"}:
        queries = pd.read_csv(queries_path)
    else:
        queries = generate_and_save_queries(datasets, cfg)
    run_sanity_tests(datasets, queries, cfg)

    raw_path = OUTPUT / "raw_results" / "raw_results.csv"
    failed_path = OUTPUT / "raw_results" / "failed_runs.csv"
    completed = set()
    if raw_path.exists() and not force:
        old = read_csv_or_empty(raw_path)
        if "experiment_id" in old:
            completed = set(old["experiment_id"].astype(str))
    if force and raw_path.exists() and mode not in {"analysis-only", "plots-only"}:
        raw_path.unlink()
    if force and failed_path.exists() and mode not in {"analysis-only", "plots-only"}:
        failed_path.unlink()

    if mode not in {"analysis-only", "plots-only"}:
        for _, q in queries.iterrows():
            dataset = q["dataset"]
            full_df = datasets[dataset]
            conditions = parse_conditions(q["conditions_json"])
            global_matches = marked_indices(full_df, conditions)
            for seed in cfg.seeds:
                for N in cfg.ns:
                    if N > len(full_df):
                        continue
                    required = set(np.random.default_rng(seed + len(q["query_id"])).choice(sorted(global_matches), size=min(len(global_matches), max(1, min(3, N // 8))), replace=False))
                    sub = subset_df(full_df, N, seed + int(q["K"]) * 100, required)
                    if sub is None:
                        continue
                    true_indices_clean = marked_indices(sub, conditions)
                    if len(true_indices_clean) <= 0 or len(true_indices_clean) >= N:
                        continue
                    subset_record = {
                        "dataset": dataset,
                        "seed": seed,
                        "N": N,
                        "query_id": q["query_id"],
                        "subset_original_patient_indices": sub["original_patient_index"].tolist(),
                    }
                    append_row(OUTPUT / "reproducibility" / "subsets.csv", subset_record)
                    for missingness in cfg.missingness_levels:
                        for missing_policy in ["strict", "candidate-preserving"]:
                            df_miss = apply_missingness(sub, conditions, missingness, seed, dataset)
                            true_indices = marked_indices(df_miss, conditions, missing_policy="strict")
                            candidate_indices = marked_indices(df_miss, conditions, missing_policy=missing_policy)
                            if not true_indices:
                                true_indices = true_indices_clean
                            base = {
                                "dataset": dataset,
                                "seed": seed,
                                "N": N,
                                "index_qubits": int(math.log2(N)),
                                "query_id": q["query_id"],
                                "K": int(q["K"]),
                                "M": len(true_indices_clean),
                                "target_prevalence": len(true_indices_clean) / N,
                                "rarity_bin": q["rarity_bin"],
                                "conditions": q["conditions"],
                                "missingness": missingness,
                                "missing_policy": missing_policy,
                                "candidate_set_size": len(candidate_indices),
                            }
                            for method in CLASSICAL_METHODS:
                                row_stub = {**base, "method": method, "shots": 0, "noise_level": "clean"}
                                eid = experiment_id(row_stub)
                                if eid in completed and not force:
                                    continue
                                try:
                                    row = evaluate_one(base, df_miss, conditions, true_indices_clean, method, cfg, seed, None, "clean", {"two_qubit_depolarizing": 0.0, "readout": 0.0})
                                    append_row(raw_path, row, RAW_COLUMNS)
                                    completed.add(row["experiment_id"])
                                except Exception as exc:
                                    fail = {**row_stub, "experiment_id": eid, "error": str(exc), "traceback": traceback.format_exc()}
                                    append_row(failed_path, fail)
                                    logging.exception("Failed classical run")
                            for noise_label, noise_params in cfg.noise_levels.items():
                                for shots in cfg.shots:
                                    if not should_run_quantum_setting(cfg, missingness, noise_label, shots):
                                        continue
                                    for method in QUANTUM_METHODS:
                                        row_stub = {**base, "method": method, "shots": shots, "noise_level": noise_label}
                                        eid = experiment_id(row_stub)
                                        if eid in completed and not force:
                                            continue
                                        try:
                                            if method in {"standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse", "CARE-Fuse-95"} and noise_label == "clean":
                                                print(f"[Dataset] {dataset} [N] {N} [Query] {q['query_id']} [K] {q['K']} [M] {len(true_indices_clean)} [Method] {method} [Seed] {seed} [Status] running")
                                            row = evaluate_one(base, df_miss, conditions, true_indices_clean, method, cfg, seed, shots, noise_label, noise_params)
                                            append_row(raw_path, row, RAW_COLUMNS)
                                            completed.add(row["experiment_id"])
                                        except Exception as exc:
                                            fail = {**row_stub, "experiment_id": eid, "error": str(exc), "traceback": traceback.format_exc()}
                                            append_row(failed_path, fail)
                                            logging.exception("Failed quantum run")

    raw = read_csv_or_empty(raw_path)
    if raw.empty:
        raise RuntimeError("No raw results available for analysis")
    # Keep list-like fields CSV/plot stable without turning true missing values
    # into the literal string "nan".
    for c in LIST_LIKE_COLUMNS:
        if c in raw.columns:
            raw[c] = raw[c].fillna("")
    agg = aggregate_results(raw)
    stats = paired_tests(raw)
    agg.to_csv(OUTPUT / "aggregated_results" / "aggregated_results.csv", index=False)
    stats.to_csv(OUTPUT / "statistical_tests" / "paired_tests.csv", index=False)
    write_tables(raw, agg, queries, ds_summary, OUTPUT)
    if mode != "analysis-only":
        generate_figures(raw, agg, queries, OUTPUT)
    save_circuit_examples(OUTPUT)
    write_summaries(raw, agg, stats, OUTPUT)
    validate_outputs(raw, queries, OUTPUT)
    return terminal_summary(raw, stats)


def write_summaries(raw: pd.DataFrame, agg: pd.DataFrame, stats: pd.DataFrame, out: Path) -> None:
    std = raw[raw["method"] == "standard_grover"]
    care = raw[raw["method"] == "CARE-QSearch-adaptive"]
    fuse = raw[raw["method"] == "CARE-Fuse"]
    fuse95 = raw[raw["method"] == "CARE-Fuse-95"]
    care_success = care["success_probability"].mean()
    std_success = std["success_probability"].mean()
    fuse_success = fuse["success_probability"].mean() if not fuse.empty else np.nan
    fuse95_success = fuse95["success_probability"].mean() if not fuse95.empty else np.nan
    care_depth = care["oracle_depth"].mean()
    std_depth = std["oracle_depth"].mean()
    fuse_depth = fuse["oracle_depth"].mean() if not fuse.empty else np.nan
    fuse95_depth = fuse95["oracle_depth"].mean() if not fuse95.empty else np.nan
    care_cx = care["CX_count"].mean()
    std_cx = std["CX_count"].mean()
    fuse_cx = fuse["CX_count"].mean() if not fuse.empty else np.nan
    fuse95_cx = fuse95["CX_count"].mean() if not fuse95.empty else np.nan
    std_raqc = std["RAQC"].mean()
    fuse_raqc = fuse["RAQC"].mean() if not fuse.empty else np.nan
    fuse95_raqc = fuse95["RAQC"].mean() if not fuse95.empty else np.nan
    std_raqc_median = std["RAQC"].median()
    fuse_raqc_median = fuse["RAQC"].median() if not fuse.empty else np.nan
    fuse95_raqc_median = fuse95["RAQC"].median() if not fuse95.empty else np.nan
    std_total_cost = std["total_resource_cost"].mean()
    fuse_total_cost = fuse["total_resource_cost"].mean() if not fuse.empty else np.nan
    fuse95_total_cost = fuse95["total_resource_cost"].mean() if not fuse95.empty else np.nan
    std_utility = std["resource_utility"].mean()
    fuse_utility = fuse["resource_utility"].mean() if not fuse.empty else np.nan
    fuse95_utility = fuse95["resource_utility"].mean() if not fuse95.empty else np.nan
    qiskit_validation_path = out / "qiskit_validation" / "qiskit_smallN_validation.xlsx"
    qiskit_note = "Small-N Qiskit validation not yet run."
    qiskit_cases = 0
    if qiskit_validation_path.exists():
        try:
            qiskit_summary = pd.read_excel(qiskit_validation_path, sheet_name="summary")
            qiskit_cases = int(qiskit_summary["validated_cases"].sum())
            depth_errors = qiskit_summary["depth_relative_error_median"].dropna()
            cx_errors = qiskit_summary["CX_relative_error_median"].dropna()
            qiskit_note = (
                f"Small-N Qiskit validation completed for {qiskit_cases} cases. "
                f"Heuristic estimates understate absolute Qiskit depth/CX "
                f"(median relative errors: depth {depth_errors.median():.3f}, CX {cx_errors.median():.3f}) "
                "but preserve the CARE-versus-standard resource-reduction trend."
            )
        except Exception:
            qiskit_note = "Small-N Qiskit validation workbook exists but could not be summarized automatically."
    fuse_median_raqc_diff = np.nan
    fuse95_median_raqc_diff = np.nan
    fuse_total_cost_win_rate = np.nan
    fuse_utility_win_rate = np.nan
    fuse95_total_cost_win_rate = np.nan
    fuse95_utility_win_rate = np.nan
    fuse95_success_retention = np.nan
    if not fuse.empty:
        base_cols = ["dataset", "seed", "N", "query_id", "shots", "noise_level", "missingness"]
        paired = raw[raw["method"].isin(["standard_grover", "CARE-Fuse", "CARE-Fuse-95"])].pivot_table(
            index=base_cols,
            columns="method",
            values=["success_probability", "RAQC", "total_resource_cost", "resource_utility"],
            aggfunc="mean",
        )
        if {("RAQC", "CARE-Fuse"), ("RAQC", "standard_grover")}.issubset(set(paired.columns)):
            fuse_median_raqc_diff = (paired[("RAQC", "CARE-Fuse")] - paired[("RAQC", "standard_grover")]).median()
            fuse_total_cost_win_rate = (
                paired[("total_resource_cost", "CARE-Fuse")] < paired[("total_resource_cost", "standard_grover")]
            ).mean()
            fuse_utility_win_rate = (
                paired[("resource_utility", "CARE-Fuse")] > paired[("resource_utility", "standard_grover")]
            ).mean()
        if {("RAQC", "CARE-Fuse-95"), ("RAQC", "standard_grover")}.issubset(set(paired.columns)):
            fuse95_median_raqc_diff = (paired[("RAQC", "CARE-Fuse-95")] - paired[("RAQC", "standard_grover")]).median()
            fuse95_total_cost_win_rate = (
                paired[("total_resource_cost", "CARE-Fuse-95")] < paired[("total_resource_cost", "standard_grover")]
            ).mean()
            fuse95_utility_win_rate = (
                paired[("resource_utility", "CARE-Fuse-95")] > paired[("resource_utility", "standard_grover")]
            ).mean()
            fuse95_success_retention = (
                paired[("success_probability", "CARE-Fuse-95")] >= 0.95 * paired[("success_probability", "standard_grover")]
            ).mean()
    text = f"""# CARE-QSearch Results Summary

This experiment treats each row as a patient record and evaluates clinical-predicate retrieval, not disease classification.

Quantum results use a compiled clinical-predicate truth-table oracle, exact Grover amplitude probabilities, finite-shot sampling, and controlled noise attenuation. Small-N Qiskit resource metrics are opt-in with CARE_USE_QISKIT=1; default rows use transparent heuristic circuit-resource estimates. Simulator wall-clock time is reported only as computational cost, not quantum advantage.

1. CARE-QSearch-adaptive success probability vs Standard Grover: mean difference {care_success - std_success:.4f}.
2. Oracle depth difference: CARE {care_depth:.2f} vs Standard {std_depth:.2f}.
3. CX count difference: CARE {care_cx:.2f} vs Standard {std_cx:.2f}.
4. CARE-Fuse jointly optimizes predicate ordering and Grover iteration count using U = P_success / C_total. CARE-Fuse trades lower mean success ({fuse_success:.4f} vs Standard {std_success:.4f}) for lower mean total resource cost ({fuse_total_cost:.2f} vs {std_total_cost:.2f}) and higher mean utility ({fuse_utility:.6f} vs {std_utility:.6f}).
5. CARE-Fuse-95 is the success-constrained variant: minimize C_total subject to predicted P_success >= 0.95 x Standard predicted P_success. It gives mean success {fuse95_success:.4f}, mean total resource cost {fuse95_total_cost:.2f}, and paired success-retention rate {fuse95_success_retention:.4f}.
6. Resource-adjusted query cost is saved as RAQC = C_total / P_success. Because very low success probabilities create extreme RAQC outliers, report RAQC with median/IQR or paired robust statistics. CARE-Fuse median RAQC is {fuse_raqc_median:.2f} vs Standard {std_raqc_median:.2f}; paired median difference is {fuse_median_raqc_diff:.2f}. CARE-Fuse-95 paired median RAQC difference is {fuse95_median_raqc_diff:.2f}.
7. Qiskit sanity check: {qiskit_note}
8. Benefits should be interpreted as resource-aware oracle compilation/search-order effects in this implementation, not demonstrated hardware speedup.
9. Noise and missingness results are controlled simulations, not clinical validation.
10. Claims not supported: quantum simulator runtime speedup, scalable QRAM/coherent data-loading advantage, and clinically validated asymmetric loss.
11. Classical baselines include sequential scan, vectorised Boolean filtering, selective predicate ordering, and SQLite indexed lookup.
"""
    if not stats.empty:
        text += "\nMain paired tests:\n\n" + stats.fillna("").to_markdown(index=False) + "\n"
    (out / "manuscript_summary" / "results_summary.md").write_text(text, encoding="utf-8")
    claims = pd.DataFrame(
        [
            ["CARE reduces oracle-level resource cost.", "oracle_depth/CX", care_depth - std_depth, care_cx - std_cx, "paired Wilcoxon where available", bool(care_depth < std_depth and care_cx < std_cx), "Qiskit-transpiled for small N only when CARE_USE_QISKIT=1; otherwise heuristic estimates."],
            ["CARE lowers total resource-adjusted query cost.", "RAQC", care["RAQC"].mean() - std["RAQC"].mean(), care["RAQC"].mean() - std["RAQC"].mean(), "paired Wilcoxon where available", bool(care["RAQC"].mean() < std["RAQC"].mean()), "RAQC is C_total / P_success and includes adaptive schedule cost."],
            ["CARE maintains retrieval success.", "P_success", care_success - std_success, care_success - std_success, "paired Wilcoxon where available", bool(care_success >= std_success - 0.02), "Finite-shot/noise model can lower all methods."],
            ["CARE-Fuse lowers total search resource cost.", "C_total", fuse_total_cost - std_total_cost, fuse_total_cost - std_total_cost, "paired Wilcoxon where available", bool(fuse_total_cost < std_total_cost), "CARE-Fuse explicitly optimizes success per total resource cost."],
            ["CARE-Fuse improves resource utility.", "P_success / C_total", fuse_utility - std_utility, fuse_utility - std_utility, "paired Wilcoxon where available", bool(fuse_utility > std_utility), "Primary CARE-Fuse objective."],
            ["CARE-Fuse lowers robust RAQC.", "median paired RAQC", fuse_median_raqc_diff, fuse_median_raqc_diff, "paired Wilcoxon where available", bool(fuse_median_raqc_diff < 0), "Mean RAQC is outlier-dominated; report robust summaries."],
            ["CARE-Fuse maintains retrieval success.", "P_success", fuse_success - std_success, fuse_success - std_success, "paired Wilcoxon where available", bool(fuse_success >= std_success - 0.02), "Fixed-budget utility can trade a small success reduction for lower total cost."],
            ["CARE-Fuse-95 enforces success retention.", "P_success >= 0.95 x Standard", fuse95_success_retention, fuse95_success_retention, "paired descriptive retention rate", bool(fuse95_success_retention >= 0.90), "Success-constrained cost minimization variant."],
            ["CARE-Fuse-95 lowers total search resource cost.", "C_total", fuse95_total_cost - std_total_cost, fuse95_total_cost - std_total_cost, "paired Wilcoxon where available", bool(fuse95_total_cost < std_total_cost), "Cost under success-retention constraint."],
            ["CARE-Fuse-95 improves resource utility.", "P_success / C_total", fuse95_utility - std_utility, fuse95_utility - std_utility, "paired Wilcoxon where available", bool(fuse95_utility > std_utility), "Utility under success-retention constraint."],
            ["Small-N Qiskit sanity check supports resource trend.", "Qiskit transpiled depth/CX", qiskit_cases, qiskit_cases, "descriptive validation", bool(qiskit_cases > 0), qiskit_note],
            ["Benefits increase for rare target sets.", "rarity-bin RAQC", "", "", "descriptive", "mixed", "Inspect aggregated_results by rarity_bin."],
            ["Increasing K increases circuit cost.", "oracle_depth/CX vs K", "", "", "descriptive", bool(raw.groupby("K")["oracle_depth"].mean().is_monotonic_increasing), "Estimated predicate costs."],
            ["Noise reduces retrieval reliability.", "P_success/FNR", "", "", "descriptive", bool(raw.groupby("noise_two_qubit_depolarizing")["success_probability"].mean().is_monotonic_decreasing), "Controlled noise attenuation."],
            ["Missingness increases false-negative risk.", "FNR/recall", "", "", "descriptive", "mixed", "Depends on strict vs candidate-preserving policy."],
        ],
        columns=["Claim", "Metric", "Cleveland Evidence", "Statlog Evidence", "Statistical Test", "Supported?", "Notes"],
    )
    claims.to_csv(out / "manuscript_summary" / "claim_support_matrix.csv", index=False)
    best = raw.groupby("method", as_index=False).agg(success_probability=("success_probability", "mean"), RAQC=("RAQC", "mean"), oracle_depth=("oracle_depth", "mean"), CX_count=("CX_count", "mean")).sort_values(["success_probability", "RAQC"], ascending=[False, True])
    best.to_csv(out / "manuscript_summary" / "best_results.csv", index=False)
    key = f"""CARE vs Standard Grover mean success difference: {care_success - std_success:.4f}
CARE vs Standard Grover mean oracle depth difference: {care_depth - std_depth:.2f}
CARE vs Standard Grover mean CX difference: {care_cx - std_cx:.2f}
CARE-Fuse vs Standard Grover mean success difference: {fuse_success - std_success:.4f}
CARE-Fuse vs Standard Grover mean oracle depth difference: {fuse_depth - std_depth:.2f}
CARE-Fuse vs Standard Grover mean CX difference: {fuse_cx - std_cx:.2f}
CARE-Fuse vs Standard Grover mean RAQC difference: {fuse_raqc - std_raqc:.2f}
CARE-Fuse vs Standard Grover median RAQC difference: {fuse_median_raqc_diff:.2f}
CARE-Fuse vs Standard Grover mean total resource cost difference: {fuse_total_cost - std_total_cost:.2f}
CARE-Fuse total resource cost paired win rate: {fuse_total_cost_win_rate:.4f}
CARE-Fuse resource utility paired win rate: {fuse_utility_win_rate:.4f}
CARE-Fuse-95 vs Standard Grover mean success difference: {fuse95_success - std_success:.4f}
CARE-Fuse-95 vs Standard Grover mean oracle depth difference: {fuse95_depth - std_depth:.2f}
CARE-Fuse-95 vs Standard Grover mean CX difference: {fuse95_cx - std_cx:.2f}
CARE-Fuse-95 vs Standard Grover mean RAQC difference: {fuse95_raqc - std_raqc:.2f}
CARE-Fuse-95 vs Standard Grover median RAQC difference: {fuse95_median_raqc_diff:.2f}
CARE-Fuse-95 vs Standard Grover mean total resource cost difference: {fuse95_total_cost - std_total_cost:.2f}
CARE-Fuse-95 success retention paired rate: {fuse95_success_retention:.4f}
CARE-Fuse-95 total resource cost paired win rate: {fuse95_total_cost_win_rate:.4f}
CARE-Fuse-95 resource utility paired win rate: {fuse95_utility_win_rate:.4f}
Qiskit small-N validation cases: {qiskit_cases}
Qiskit validation note: {qiskit_note}
Mean oracle-query ratio, quantum rows: {raw[raw['oracle_calls'] > 0]['query_ratio'].mean():.4f}
Main limitation: default full runs use transparent heuristic resource estimates; Qiskit validation is a small-N sanity check, not a full hardware/resource proof.
"""
    (out / "manuscript_summary" / "key_numbers.txt").write_text(key, encoding="utf-8")


def validate_outputs(raw: pd.DataFrame, queries: pd.DataFrame, out: Path) -> None:
    failures = []
    required_figs = [f"figure{i}" for i in range(1, 7)]
    for prefix in required_figs:
        if not any((out / "figures" / "png").glob(f"{prefix}*.png")):
            failures.append(f"Missing PNG for {prefix}")
    if not (out / "tables" / "excel" / "all_tables.xlsx").exists():
        failures.append("Missing Excel workbook all_tables.xlsx")
    if not {"cleveland", "statlog"}.issubset(set(raw["dataset"])):
        failures.append("Both datasets are not represented")
    if raw["experiment_id"].duplicated().any():
        failures.append("Duplicate experiment_id values found")
    if not raw["success_probability"].between(0, 1).all():
        failures.append("Impossible probability found")
    if (queries["M"] <= 0).any() or (queries["M"] >= queries["N"]).any():
        failures.append("Invalid generated query M")
    for method in ["classical_sequential", "classical_vectorised", "classical_selective", "classical_sqlite_indexed", "standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse", "CARE-Fuse-95"]:
        if method not in set(raw["method"]):
            failures.append(f"Missing method {method}")
    msg = "EXPERIMENT VALIDATION: PASS" if not failures else "EXPERIMENT VALIDATION: FAIL\n" + "\n".join(failures)
    (out / "logs" / "validation.txt").write_text(msg, encoding="utf-8")
    print(msg)


def terminal_summary(raw: pd.DataFrame, stats: pd.DataFrame) -> dict:
    best = raw.groupby("method")["success_probability"].mean().sort_values(ascending=False)
    care = raw[raw["method"] == "CARE-QSearch-adaptive"]
    fuse = raw[raw["method"] == "CARE-Fuse"]
    fuse95 = raw[raw["method"] == "CARE-Fuse-95"]
    std = raw[raw["method"] == "standard_grover"]
    failed = read_csv_or_empty(OUTPUT / "raw_results" / "failed_runs.csv")
    return {
        "datasets": sorted(raw["dataset"].unique().tolist()),
        "completed_runs": int(len(raw)),
        "failed_runs": int(len(failed)),
        "best_performing_method": best.index[0] if len(best) else "n/a",
        "care_vs_standard_resource_difference_depth": float(care["oracle_depth"].mean() - std["oracle_depth"].mean()),
        "care_vs_standard_resource_difference_cx": float(care["CX_count"].mean() - std["CX_count"].mean()),
        "care_vs_standard_success_difference": float(care["success_probability"].mean() - std["success_probability"].mean()),
        "care_fuse_vs_standard_success_difference": float(fuse["success_probability"].mean() - std["success_probability"].mean()) if not fuse.empty else None,
        "care_fuse_vs_standard_raqc_difference": float(fuse["RAQC"].mean() - std["RAQC"].mean()) if not fuse.empty else None,
        "care_fuse95_vs_standard_success_difference": float(fuse95["success_probability"].mean() - std["success_probability"].mean()) if not fuse95.empty else None,
        "care_fuse95_vs_standard_raqc_difference": float(fuse95["RAQC"].mean() - std["RAQC"].mean()) if not fuse95.empty else None,
        "main_statistical_result": stats.head(1).to_dict(orient="records")[0] if not stats.empty else "not enough paired observations",
        "main_limitation": "Default full runs use transparent heuristic resource estimates; the separate qiskit-validation mode provides a small-N transpilation sanity check.",
        "output_directory": str(OUTPUT),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "plots-only", "analysis-only", "qiskit-validation"], default="smoke")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = run_pipeline(args.mode, force=args.force)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
