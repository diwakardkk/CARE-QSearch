from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
DATASETS = {
    "cleveland": ROOT / "heart+disease" / "processed.cleveland.data",
    "statlog": ROOT / "statlog+heart" / "heart.dat",
}

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

LABELS = {"cleveland": "num", "statlog": "target"}
SEEDS_FULL = [11, 22, 33, 44, 55]
SEEDS_SMOKE = [11, 22]
N_FULL = [8, 16, 32, 64, 128, 256]
N_SMOKE = [8, 16]
SHOTS_FULL = [128, 256, 512, 1024, 2048, 4096]
SHOTS_SMOKE = [256]
NOISE_LEVELS_FULL = {
    "clean": {"two_qubit_depolarizing": 0.0, "readout": 0.0},
    "low": {"two_qubit_depolarizing": 0.001, "readout": 0.01},
    "medium": {"two_qubit_depolarizing": 0.005, "readout": 0.03},
    "high": {"two_qubit_depolarizing": 0.02, "readout": 0.05},
}
NOISE_LEVELS_SMOKE = {"clean": NOISE_LEVELS_FULL["clean"], "medium": NOISE_LEVELS_FULL["medium"]}
MISSINGNESS_FULL = [0.0, 0.05, 0.10, 0.20, 0.30]
MISSINGNESS_SMOKE = [0.0, 0.10]


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    alpha_2q: float = 10.0
    lambda_fn: float = 5.0
    lambda_fp: float = 1.0
    epsilon: float = 1e-9
    max_queries_per_dataset_full: int = 8
    max_queries_per_dataset_smoke: int = 6
    random_order_repeats: int = 3

    @property
    def seeds(self) -> list[int]:
        return SEEDS_SMOKE if self.mode == "smoke" else SEEDS_FULL

    @property
    def ns(self) -> list[int]:
        return N_SMOKE if self.mode == "smoke" else N_FULL

    @property
    def shots(self) -> list[int]:
        return SHOTS_SMOKE if self.mode == "smoke" else SHOTS_FULL

    @property
    def noise_levels(self) -> dict[str, dict[str, float]]:
        return NOISE_LEVELS_SMOKE if self.mode == "smoke" else NOISE_LEVELS_FULL

    @property
    def missingness_levels(self) -> list[float]:
        return MISSINGNESS_SMOKE if self.mode == "smoke" else MISSINGNESS_FULL

    @property
    def max_queries_per_dataset(self) -> int:
        return self.max_queries_per_dataset_smoke if self.mode == "smoke" else self.max_queries_per_dataset_full

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seeds"] = self.seeds
        data["N"] = self.ns
        data["shots"] = self.shots
        data["noise_levels"] = self.noise_levels
        data["missingness_levels"] = self.missingness_levels
        data["full_mode_controlled_design"] = (
            "Full mode avoids an excessive Cartesian product: finite-shot runs use clean/no-missingness across all shots; "
            "noise runs use no-missingness at 1024 shots across noise levels; missing-data runs use clean noise at 1024 shots across missingness levels."
            " It uses all configured N values and full seeds with a controlled generated-query panel."
        )
        return data


OUTPUT_DIRS = [
    "configs",
    "data_summary",
    "queries",
    "raw_results",
    "aggregated_results",
    "statistical_tests",
    "tables/excel",
    "figures/png",
    "circuits",
    "logs",
    "predictions",
    "reproducibility",
    "manuscript_summary",
    "qiskit_validation",
]
