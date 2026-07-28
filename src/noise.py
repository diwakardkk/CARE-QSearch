from __future__ import annotations


def noise_label_params(levels: dict[str, dict[str, float]]) -> list[tuple[str, dict[str, float]]]:
    return list(levels.items())

