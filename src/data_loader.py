from __future__ import annotations

from .pandas_compat import configure_pandas

configure_pandas()

import pandas as pd

from .config import DATASETS, FEATURES, LABELS


def load_cleveland() -> pd.DataFrame:
    cols = FEATURES + [LABELS["cleveland"]]
    df = pd.read_csv(DATASETS["cleveland"], names=cols, na_values="?", dtype_backend="numpy_nullable")
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dataset"] = "cleveland"
    df["patient_index"] = range(len(df))
    return df


def load_statlog() -> pd.DataFrame:
    cols = FEATURES + [LABELS["statlog"]]
    df = pd.read_csv(DATASETS["statlog"], sep=r"\s+", names=cols, dtype_backend="numpy_nullable")
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dataset"] = "statlog"
    df["patient_index"] = range(len(df))
    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {"cleveland": load_cleveland(), "statlog": load_statlog()}


def dataset_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        feature_df = df[FEATURES]
        rows.append(
            {
                "Dataset": name,
                "Records": len(df),
                "Features": len(FEATURES),
                "Missing values": int(feature_df.isna().sum().sum()),
                "Clinical variables used": ", ".join(FEATURES),
            }
        )
    return pd.DataFrame(rows)
