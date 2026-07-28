from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import ci95


def _p_value_display(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    if p == 0:
        return "p < 1e-300"
    return f"{p:.3e}"


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "oracle_calls",
        "search_calls",
        "success_probability",
        "precision",
        "recall",
        "F1",
        "false_negative_rate",
        "clinical_loss",
        "oracle_depth",
        "total_depth",
        "CX_count",
        "TotalCXCost",
        "total_gate_count",
        "RAQC",
        "total_resource_cost",
        "resource_utility",
        "runtime",
        "oracle_construction_time",
    ]
    rows = []
    group_cols = ["dataset", "N", "K", "method"]
    for keys, g in raw.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys, strict=False))
        row["runs"] = len(g)
        for m in metrics:
            if m in g:
                vals = pd.to_numeric(g[m], errors="coerce").dropna()
                row[f"{m}_mean"] = vals.mean()
                row[f"{m}_std"] = vals.std(ddof=1)
                row[f"{m}_median"] = vals.median()
                row[f"{m}_iqr"] = vals.quantile(0.75) - vals.quantile(0.25)
                lo, hi = ci95(vals)
                row[f"{m}_ci95_low"] = lo
                row[f"{m}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    base_cols = ["dataset", "seed", "N", "query_id", "shots", "noise_level", "missingness"]
    rows = []
    comparisons = [
        ("CARE-QSearch-adaptive", "standard_grover"),
        ("CARE-Fuse", "standard_grover"),
        ("CARE-Fuse-95", "standard_grover"),
        ("CARE-Fuse", "CARE-QSearch-adaptive"),
        ("CARE-Fuse-95", "CARE-Fuse"),
        ("CARE-Fuse-95", "CARE-QSearch-adaptive"),
    ]
    for metric in ["success_probability", "oracle_depth", "CX_count", "RAQC", "total_resource_cost", "resource_utility"]:
        pivot = raw.pivot_table(index=base_cols, columns="method", values=metric, aggfunc="mean")
        for treatment, baseline in comparisons:
            if {baseline, treatment}.issubset(pivot.columns):
                paired = pivot[[baseline, treatment]].dropna()
            else:
                continue
            if len(paired) >= 3:
                diff = paired[treatment] - paired[baseline]
                if np.allclose(diff, 0.0, equal_nan=True):
                    stat, p = 0.0, np.nan
                    test_note = "identical paired values; Wilcoxon not applicable"
                else:
                    try:
                        stat, p = stats.wilcoxon(paired[treatment], paired[baseline], zero_method="wilcox")
                        test_note = ""
                    except ValueError:
                        stat, p = np.nan, np.nan
                        test_note = "Wilcoxon not applicable"
                rows.append(
                    {
                        "comparison": f"{treatment} vs {baseline}",
                        "metric": metric,
                        "n_pairs": len(paired),
                        "treatment_mean": paired[treatment].mean(),
                        "baseline_mean": paired[baseline].mean(),
                        "mean_difference": diff.mean(),
                        "median_difference": diff.median(),
                        "wilcoxon_statistic": stat,
                        "p_value": p,
                        "p_value_display": _p_value_display(p),
                        "effect_size_rank_biserial_proxy": float(np.sign(diff).mean()),
                        "test_note": test_note,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        finite_p = out["p_value"].notna()
        order = out.loc[finite_p, "p_value"].rank(method="first")
        m = len(out)
        out["holm_alpha_0.05"] = np.nan
        out.loc[finite_p, "holm_alpha_0.05"] = 0.05 / (m - order + 1)
        out["holm_significant_0.05"] = False
        out.loc[finite_p, "holm_significant_0.05"] = (
            out.loc[finite_p, "p_value"] <= out.loc[finite_p, "holm_alpha_0.05"]
        )
    return out
