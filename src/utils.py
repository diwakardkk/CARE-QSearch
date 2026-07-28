from __future__ import annotations

import csv
import importlib.metadata
import json
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .pandas_compat import configure_pandas

configure_pandas()

import numpy as np
import pandas as pd

from .config import OUTPUT, OUTPUT_DIRS


def clean_csv_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def setup_output() -> None:
    for legacy in ["figures/pdf", "tables/latex", "tables/csv"]:
        path = OUTPUT / legacy
        if path.exists():
            shutil.rmtree(path)
    circuits = OUTPUT / "circuits"
    if circuits.exists():
        for pdf in circuits.glob("*.pdf"):
            pdf.unlink()
    for d in OUTPUT_DIRS:
        (OUTPUT / d).mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    setup_output()
    logging.basicConfig(
        filename=OUTPUT / "logs" / "experiment.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def append_row(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: clean_csv_value(v) for k, v in row.items()}
    exists = path.exists()
    if exists and fieldnames is None:
        with path.open("r", newline="", encoding="utf-8") as existing:
            first = existing.readline().strip()
            fieldnames = next(csv.reader([first])) if first else list(row.keys())
    if fieldnames is None:
        fieldnames = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def write_environment(out: Path = OUTPUT) -> None:
    import matplotlib
    import pandas
    import scipy

    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "not installed"
    try:
        qiskit_version = importlib.metadata.version("qiskit")
    except importlib.metadata.PackageNotFoundError:
        qiskit_version = "not installed"
    lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"Processor: {platform.processor()}",
        f"NumPy: {np.__version__}",
        f"Pandas: {pandas.__version__}",
        f"SciPy: {scipy.__version__}",
        f"Matplotlib: {matplotlib.__version__}",
        f"scikit-learn: {sklearn_version}",
        f"Qiskit: {qiskit_version}",
        "",
        "Quantum implementation note: Qiskit metrics are opt-in with CARE_USE_QISKIT=1 for small N. Default runs use a compiled clinical-predicate truth-table oracle with exact Grover amplitude-amplification probabilities and transparent heuristic circuit-resource estimates.",
    ]
    (out / "reproducibility" / "environment.txt").write_text("\n".join(lines), encoding="utf-8")
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        (out / "reproducibility" / "requirements_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    except Exception as exc:
        (out / "reproducibility" / "requirements_freeze.txt").write_text(f"pip freeze failed: {exc}", encoding="utf-8")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
