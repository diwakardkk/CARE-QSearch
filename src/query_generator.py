from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURES


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float

    def label(self) -> str:
        val = int(self.value) if float(self.value).is_integer() else round(float(self.value), 3)
        return f"{self.feature} {self.op} {val}"


def evaluate_condition(df: pd.DataFrame, cond: Condition, missing_policy: str = "strict") -> pd.Series:
    s = df[cond.feature]
    if cond.op == ">":
        mask = s > cond.value
    elif cond.op == "<":
        mask = s < cond.value
    elif cond.op == "==":
        mask = s == cond.value
    else:
        raise ValueError(f"Unsupported operator: {cond.op}")
    if missing_policy == "strict":
        return mask.fillna(False)
    if missing_policy == "candidate-preserving":
        return mask | s.isna()
    raise ValueError(f"Unknown missing policy: {missing_policy}")


def evaluate_query(df: pd.DataFrame, conditions: list[Condition], missing_policy: str = "strict") -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for cond in conditions:
        mask &= evaluate_condition(df, cond, missing_policy=missing_policy)
    return mask.fillna(False)


def predicate_prevalence(df: pd.DataFrame, cond: Condition) -> float:
    return float(evaluate_condition(df, cond).mean())


def estimate_predicate_gate_cost(cond: Condition, alpha_2q: float = 10.0) -> dict:
    continuous = cond.op in {">", "<"}
    categorical = cond.op == "=="
    if continuous:
        n_1q, n_2q, depth = 12, 6, 20
    elif categorical:
        n_1q, n_2q, depth = 6, 3, 10
    else:
        n_1q, n_2q, depth = 8, 4, 14
    if cond.feature in {"ca", "thal", "cp", "restecg", "slope"}:
        n_1q += 2
        n_2q += 1
    return {"N_1q": n_1q, "N_2q": n_2q, "G": n_1q + alpha_2q * n_2q, "depth": depth}


def condition_scores(df: pd.DataFrame, conditions: list[Condition], alpha_2q: float, epsilon: float) -> list[dict]:
    rows = []
    for c in conditions:
        p = predicate_prevalence(df, c)
        rarity = -math.log(p + epsilon)
        cost = estimate_predicate_gate_cost(c, alpha_2q=alpha_2q)
        care = rarity / (cost["G"] + epsilon)
        scr = rarity / ((cost["G"] + epsilon) ** 1.0)
        rows.append({"condition": c.label(), "prevalence": p, "rarity_score": rarity, "gate_cost": cost["G"], "CARE_score": care, "SCR_score": scr, **cost})
    return rows


def estimate_joint_selectivity(df: pd.DataFrame, conditions: list[Condition], missing_policy: str = "strict") -> float:
    if not conditions:
        return 1.0
    return float(evaluate_query(df, conditions, missing_policy=missing_policy).mean())


def expected_order_cost(df: pd.DataFrame, ordered: list[Condition], alpha_2q: float, missing_policy: str = "strict") -> float:
    expected = 0.0
    prefix: list[Condition] = []
    for cond in ordered:
        active_fraction = estimate_joint_selectivity(df, prefix, missing_policy=missing_policy)
        expected += active_fraction * estimate_predicate_gate_cost(cond, alpha_2q=alpha_2q)["G"]
        prefix.append(cond)
    return float(expected)


def order_conditions_correlation_aware(df: pd.DataFrame, conditions: list[Condition], alpha_2q: float, missing_policy: str = "strict") -> list[Condition]:
    remaining = list(conditions)
    ordered: list[Condition] = []
    while remaining:
        best = min(
            remaining,
            key=lambda c: (
                expected_order_cost(df, ordered + [c], alpha_2q, missing_policy=missing_policy),
                estimate_predicate_gate_cost(c, alpha_2q=alpha_2q)["G"],
                c.label(),
            ),
        )
        ordered.append(best)
        remaining.remove(best)
    return ordered


def order_conditions(df: pd.DataFrame, conditions: list[Condition], method: str, alpha_2q: float, epsilon: float, seed: int | None = None) -> list[Condition]:
    if method == "standard":
        return list(conditions)
    if method == "random":
        rng = np.random.default_rng(seed)
        ordered = list(conditions)
        rng.shuffle(ordered)
        return ordered
    scores = condition_scores(df, conditions, alpha_2q, epsilon)
    score_by_label = {r["condition"]: r for r in scores}
    if method == "rarity":
        return sorted(conditions, key=lambda c: score_by_label[c.label()]["rarity_score"], reverse=True)
    if method == "cost":
        return sorted(conditions, key=lambda c: score_by_label[c.label()]["gate_cost"])
    if method == "care":
        return sorted(conditions, key=lambda c: (score_by_label[c.label()]["CARE_score"], c.label()), reverse=True)
    if method == "scr":
        return sorted(conditions, key=lambda c: (score_by_label[c.label()]["SCR_score"], c.label()), reverse=True)
    if method == "care_fuse":
        return order_conditions_correlation_aware(df, conditions, alpha_2q)
    raise ValueError(f"Unknown ordering method: {method}")


def _candidate_conditions(df: pd.DataFrame) -> list[Condition]:
    specs = []
    for f, qs, op in [
        ("age", [0.55, 0.65, 0.75, 0.85], ">"),
        ("chol", [0.60, 0.70, 0.80, 0.90], ">"),
        ("trestbps", [0.60, 0.70, 0.80], ">"),
        ("thalach", [0.20, 0.30, 0.40], "<"),
        ("oldpeak", [0.60, 0.70, 0.80], ">"),
    ]:
        values = df[f].dropna()
        for q in qs:
            if len(values):
                specs.append(Condition(f, op, float(values.quantile(q))))
    for f in ["fbs", "exang", "cp", "restecg", "slope", "ca", "thal"]:
        for v in sorted(df[f].dropna().unique()):
            p = float((df[f] == v).mean())
            if 0.01 <= p <= 0.75:
                specs.append(Condition(f, "==", float(v)))
    unique = {}
    for c in specs:
        unique[c.label()] = c
    return list(unique.values())


def _rarity_bin(frac: float) -> str:
    if frac > 0.20:
        return "common"
    if frac > 0.05:
        return "moderate"
    if frac > 0.01:
        return "rare"
    return "very_rare"


def generate_queries_for_dataset(dataset: str, df: pd.DataFrame, max_queries: int, alpha_2q: float, epsilon: float, seed: int = 11) -> tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    candidates = _candidate_conditions(df)
    rng.shuffle(candidates)
    rows: list[dict] = []
    seen: set[str] = set()

    manual = [
        [Condition("age", ">", 60), Condition("chol", ">", 240)],
        [Condition("age", ">", 60), Condition("trestbps", ">", 140), Condition("chol", ">", 240)],
        [Condition("age", ">", 55), Condition("fbs", "==", 1), Condition("thalach", "<", 130)],
        [Condition("age", ">", 60), Condition("chol", ">", 240), Condition("trestbps", ">", 140), Condition("thalach", "<", 130)],
        [Condition("age", ">", 60), Condition("chol", ">", 240), Condition("trestbps", ">", 140), Condition("thalach", "<", 130), Condition("oldpeak", ">", 1)],
    ]

    def add_query(conditions: list[Condition], manual_flag: bool = False) -> None:
        key = " AND ".join(sorted(c.label() for c in conditions))
        if key in seen:
            return
        mask = evaluate_query(df, conditions)
        M = int(mask.sum())
        N = len(df)
        if M <= 0 or M >= N:
            return
        seen.add(key)
        scores = condition_scores(df, conditions, alpha_2q, epsilon)
        care_order = [c.label() for c in order_conditions(df, conditions, "care", alpha_2q, epsilon)]
        fuse_order = [c.label() for c in order_conditions(df, conditions, "care_fuse", alpha_2q, epsilon)]
        rows.append(
            {
                "query_id": f"{dataset}_Q{len(rows) + 1:03d}",
                "dataset": dataset,
                "conditions": " AND ".join(c.label() for c in conditions),
                "conditions_json": json.dumps([asdict(c) for c in conditions]),
                "K": len(conditions),
                "M": M,
                "N": N,
                "target_prevalence": M / N,
                "rarity_bin": _rarity_bin(M / N),
                "predicate_prevalence": json.dumps({r["condition"]: r["prevalence"] for r in scores}),
                "predicate_order": json.dumps([c.label() for c in conditions]),
                "CARE_scores": json.dumps({r["condition"]: r["CARE_score"] for r in scores}),
                "SCR_scores": json.dumps({r["condition"]: r["SCR_score"] for r in scores}),
                "CARE_order": json.dumps(care_order),
                "CARE_Fuse_order": json.dumps(fuse_order),
                "joint_selectivity": estimate_joint_selectivity(df, conditions),
                "manual_example": manual_flag,
            }
        )

    for q in manual:
        add_query(q, manual_flag=True)

    for K in range(1, 6):
        for combo in combinations(candidates[:24], K):
            add_query(list(combo))
            bins = set(r["rarity_bin"] for r in rows if r["K"] == K)
            if len([r for r in rows if r["K"] == K]) >= max(3, max_queries // 5) and {"common", "moderate"}.issubset(bins):
                break
        if len(rows) >= max_queries:
            break

    query_df = pd.DataFrame(rows[:max_queries])
    query_records = query_df.to_dict(orient="records")
    return query_df, query_records


def parse_conditions(conditions_json: str) -> list[Condition]:
    return [Condition(**d) for d in json.loads(conditions_json)]


def save_queries(query_frames: list[pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_queries = pd.concat(query_frames, ignore_index=True)
    all_queries.to_csv(out_dir / "generated_queries.csv", index=False)
    (out_dir / "generated_queries.json").write_text(json.dumps(all_queries.to_dict(orient="records"), indent=2), encoding="utf-8")
    manual = all_queries[all_queries["manual_example"] == True]
    manual.to_csv(out_dir / "manual_interpretable_queries.csv", index=False)
    return all_queries
