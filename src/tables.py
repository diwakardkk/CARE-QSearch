from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_tables(raw: pd.DataFrame, agg: pd.DataFrame, queries: pd.DataFrame, dataset_summary: pd.DataFrame, out: Path) -> None:
    excel_dir = out / "tables" / "excel"
    excel_dir.mkdir(parents=True, exist_ok=True)
    for old in excel_dir.glob("*.xlsx"):
        old.unlink()
    main_methods = ["standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse", "CARE-Fuse-95"]
    ablation_methods = [
        "standard_grover",
        "CARE-QSearch-oracle-M",
        "random_order_grover",
        "rarity_only_grover",
        "gate_cost_only_grover",
        "CARE-QSearch-adaptive",
        "CARE-Fuse",
        "CARE-Fuse-95",
    ]
    clean_main = raw[
        raw["method"].isin(main_methods)
        & (raw["noise_level"] == "clean")
        & (raw["missingness"] == 0)
        & (raw["shots"] == 1024)
    ]
    main_summary = clean_main.groupby(["dataset", "method"], as_index=False).agg(
        runs=("success_probability", "size"),
        success_probability_mean=("success_probability", "mean"),
        success_probability_median=("success_probability", "median"),
        recall_mean=("recall", "mean"),
        F1_mean=("F1", "mean"),
        oracle_calls_median=("oracle_calls", "median"),
        oracle_depth_mean=("oracle_depth", "mean"),
        CX_count_mean=("CX_count", "mean"),
        RAQC_median=("RAQC", "median"),
        total_resource_cost_mean=("total_resource_cost", "mean"),
        resource_utility_mean=("resource_utility", "mean"),
    )
    tradeoff_summary = raw[raw["method"].isin(main_methods)].groupby("method", as_index=False).agg(
        success_probability_mean=("success_probability", "mean"),
        success_probability_median=("success_probability", "median"),
        RAQC_mean=("RAQC", "mean"),
        RAQC_median=("RAQC", "median"),
        total_resource_cost_mean=("total_resource_cost", "mean"),
        total_resource_cost_median=("total_resource_cost", "median"),
        resource_utility_mean=("resource_utility", "mean"),
        oracle_depth_mean=("oracle_depth", "mean"),
        CX_count_mean=("CX_count", "mean"),
    )

    tables = {
        "table1_dataset_characteristics": dataset_summary,
        "table2_query_definitions": queries[[c for c in ["query_id", "dataset", "K", "conditions", "M", "target_prevalence", "joint_selectivity", "SCR_scores", "CARE_Fuse_order"] if c in queries.columns]],
        "table3_main_clean_performance": main_summary,
        "table4_resource_success_tradeoff": tradeoff_summary,
        "table5_scaling_results": agg.groupby(["N", "method"], as_index=False).mean(numeric_only=True),
        "table6_query_complexity": agg.groupby(["K", "method"], as_index=False).mean(numeric_only=True),
        "table7_predicate_ordering_ablation": agg[agg["method"].isin(ablation_methods)],
        "table8_noise_robustness": raw.groupby(["noise_level", "method"], as_index=False).mean(numeric_only=True),
        "table9_missing_data_robustness": raw.groupby(["missingness", "missing_policy", "method"], as_index=False).mean(numeric_only=True),
        "table10_full_ablation": agg[agg["method"].isin(ablation_methods)],
        "table11_cross_dataset_summary": agg.groupby(["dataset", "method"], as_index=False).mean(numeric_only=True),
    }
    workbook_path = excel_dir / "all_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df = df.copy()
            sheet_name = name.replace("table", "t", 1)[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    for name, df in tables.items():
        df = df.copy()
        df.to_excel(excel_dir / f"{name}.xlsx", index=False)
