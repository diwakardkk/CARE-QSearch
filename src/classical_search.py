from __future__ import annotations

import sqlite3
import time

import pandas as pd

from .metrics import retrieval_metrics
from .query_generator import Condition, evaluate_condition, evaluate_query, order_conditions


def run_classical_sequential(df: pd.DataFrame, conditions: list[Condition], true_indices: set[int], lambda_fn: float, lambda_fp: float) -> dict:
    start = time.perf_counter()
    retrieved = []
    predicate_evals = 0
    for idx, row in df.iterrows():
        ok = True
        for cond in conditions:
            predicate_evals += 1
            val = row[cond.feature]
            if pd.isna(val):
                ok = False
            elif cond.op == ">" and not val > cond.value:
                ok = False
            elif cond.op == "<" and not val < cond.value:
                ok = False
            elif cond.op == "==" and not val == cond.value:
                ok = False
            if not ok:
                break
        if ok:
            retrieved.append(int(idx))
    runtime = time.perf_counter() - start
    m = retrieval_metrics(true_indices, retrieved, lambda_fn, lambda_fp)
    return {
        **m,
        "records_inspected": len(df),
        "predicate_evaluations": predicate_evals,
        "runtime": runtime,
        "oracle_calls": 0,
        "search_calls": len(df),
        "success_probability": m["recall"],
    }


def run_classical_vectorised(df: pd.DataFrame, conditions: list[Condition], true_indices: set[int], lambda_fn: float, lambda_fp: float) -> dict:
    start = time.perf_counter()
    mask = evaluate_query(df, conditions)
    retrieved = set(map(int, df.index[mask]))
    runtime = time.perf_counter() - start
    m = retrieval_metrics(true_indices, retrieved, lambda_fn, lambda_fp)
    return {
        **m,
        "records_inspected": len(df),
        "predicate_evaluations": len(df) * len(conditions),
        "runtime": runtime,
        "oracle_calls": 0,
        "search_calls": len(df),
        "success_probability": m["recall"],
    }


def run_classical_selective(df: pd.DataFrame, conditions: list[Condition], true_indices: set[int], alpha_2q: float, epsilon: float, lambda_fn: float, lambda_fp: float) -> dict:
    start = time.perf_counter()
    ordered = order_conditions(df, conditions, "rarity", alpha_2q, epsilon)
    candidates = pd.Series(True, index=df.index)
    predicate_evals = 0
    candidate_trace = []
    for cond in ordered:
        active = candidates[candidates].index
        predicate_evals += len(active)
        cond_mask = evaluate_condition(df.loc[active], cond)
        candidates.loc[active] = cond_mask
        candidate_trace.append(float(candidates.mean()))
    retrieved = set(map(int, df.index[candidates]))
    runtime = time.perf_counter() - start
    m = retrieval_metrics(true_indices, retrieved, lambda_fn, lambda_fp)
    return {
        **m,
        "records_inspected": len(df),
        "predicate_evaluations": predicate_evals,
        "runtime": runtime,
        "oracle_calls": 0,
        "search_calls": predicate_evals,
        "success_probability": m["recall"],
        "predicate_order": " | ".join(c.label() for c in ordered),
        "candidate_trace": candidate_trace,
    }


def run_classical_sqlite_indexed(df: pd.DataFrame, conditions: list[Condition], true_indices: set[int], lambda_fn: float, lambda_fp: float) -> dict:
    start = time.perf_counter()
    conn = sqlite3.connect(":memory:")
    indexed_cols = sorted({c.feature for c in conditions})
    table_df = df.copy()
    table_df["row_id"] = table_df.index.astype(int)
    table_df.to_sql("patients", conn, index=False, if_exists="replace")
    cur = conn.cursor()
    for col in indexed_cols:
        cur.execute(f'CREATE INDEX idx_{col} ON patients("{col}")')

    clauses = []
    params = []
    for cond in conditions:
        clauses.append(f'"{cond.feature}" IS NOT NULL AND "{cond.feature}" {cond.op} ?')
        params.append(float(cond.value))
    sql = "SELECT row_id FROM patients WHERE " + " AND ".join(clauses)
    retrieved = {int(r[0]) for r in cur.execute(sql, params).fetchall()}
    conn.close()
    runtime = time.perf_counter() - start
    m = retrieval_metrics(true_indices, retrieved, lambda_fn, lambda_fp)
    return {
        **m,
        "records_inspected": len(df),
        "predicate_evaluations": 0,
        "runtime": runtime,
        "oracle_calls": 0,
        "search_calls": len(retrieved),
        "success_probability": m["recall"],
        "predicate_order": "SQLITE_INDEXED",
    }
