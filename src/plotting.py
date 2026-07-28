from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import OUTPUT


METHOD_ORDER = [
    "standard_grover",
    "CARE-QSearch-adaptive",
    "CARE-Fuse",
    "CARE-Fuse-95",
    "random_order_grover",
    "rarity_only_grover",
    "gate_cost_only_grover",
    "CARE-QSearch-oracle-M",
    "classical_sequential",
    "classical_vectorised",
    "classical_selective",
    "classical_sqlite_indexed",
]

METHOD_LABELS = {
    "standard_grover": "Standard",
    "CARE-QSearch-adaptive": "CARE-adaptive",
    "CARE-Fuse": "CARE-Fuse",
    "CARE-Fuse-95": "CARE-Fuse-95",
    "random_order_grover": "Random",
    "rarity_only_grover": "Rarity",
    "gate_cost_only_grover": "Gate-cost",
    "CARE-QSearch-oracle-M": "CARE-oracle-M",
    "classical_sequential": "Seq. scan",
    "classical_vectorised": "Vectorised",
    "classical_selective": "Selective",
    "classical_sqlite_indexed": "SQLite index",
}

# Okabe-Ito/Wong-inspired accessible palette, with black reserved for CARE-adaptive.
COLORS = {
    "standard_grover": "#D55E00",
    "CARE-QSearch-adaptive": "#000000",
    "CARE-Fuse": "#009E73",
    "CARE-Fuse-95": "#0072B2",
    "random_order_grover": "#CC79A7",
    "rarity_only_grover": "#E69F00",
    "gate_cost_only_grover": "#999999",
    "CARE-QSearch-oracle-M": "#0072B2",
    "classical_sequential": "#56B4E9",
    "classical_vectorised": "#117733",
    "classical_selective": "#332288",
    "classical_sqlite_indexed": "#88CCEE",
}

MARKERS = {
    "standard_grover": "o",
    "CARE-QSearch-adaptive": "s",
    "CARE-Fuse": "^",
    "CARE-Fuse-95": "v",
    "random_order_grover": "D",
    "rarity_only_grover": "v",
    "gate_cost_only_grover": "P",
    "CARE-QSearch-oracle-M": "X",
    "classical_sequential": "o",
    "classical_vectorised": "s",
    "classical_selective": "^",
    "classical_sqlite_indexed": "D",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _json_or_empty(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        if not value.strip() or value.strip().lower() == "nan":
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


def _fig(width_mm: float = 89, height_mm: float = 62):
    return plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4), constrained_layout=True)


def _save(fig: plt.Figure, name: str, out: Path = OUTPUT) -> None:
    path = out / "figures" / "png" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style(ax, *, ygrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(direction="out", length=2.5, width=0.55, pad=1.5)
    if ygrid:
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    else:
        ax.grid(False)


def _ordered_methods(methods) -> list[str]:
    present = set(methods)
    return [m for m in METHOD_ORDER if m in present] + sorted(present.difference(METHOD_ORDER))


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _summary(raw: pd.DataFrame, x: str, y: str, methods: list[str] | None = None, query: str | None = None) -> pd.DataFrame:
    dat = raw.copy()
    if methods is not None:
        dat = dat[dat["method"].isin(methods)]
    if query:
        dat = dat.query(query)
    grouped = dat.groupby([x, "method"], as_index=False)[y]
    return grouped.agg(median="median", q1=lambda s: s.quantile(0.25), q3=lambda s: s.quantile(0.75), mean="mean")


def _line_summary(ax, dat: pd.DataFrame, x: str, y_label: str, *, logy: bool = False) -> None:
    for method in _ordered_methods(dat["method"].unique()):
        g = dat[dat["method"] == method].sort_values(x)
        if g.empty:
            continue
        ax.plot(
            g[x],
            g["median"],
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "#333333"),
            label=_label(method),
        )
        if np.issubdtype(g[x].dtype, np.number):
            ax.fill_between(g[x].to_numpy(), g["q1"].to_numpy(), g["q3"].to_numpy(), color=COLORS.get(method, "#333333"), alpha=0.10, linewidth=0)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(y_label)
    _style(ax)


def _barh(ax, labels: list[str], values: list[float], colors: list[str], xlabel: str, *, logx: bool = False) -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    if logx:
        ax.set_xscale("log")
    _style(ax, ygrid=False)


def _panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def _bar_values(ax, values: list[float], *, fmt: str = "{:.2f}") -> None:
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for patch, value in zip(ax.patches, values, strict=False):
        if np.isfinite(value):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.025 * span,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=0,
            )


def _method_bar(ax, data: pd.DataFrame, metric: str, ylabel: str, methods: list[str], *, ylim=None, value_fmt="{:.2f}") -> None:
    vals = [float(data.loc[m, metric]) if m in data.index else np.nan for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, vals, color=[COLORS[m] for m in methods], width=0.68)
    ax.set_xticks(x, [_label(m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax, ygrid=True)
    _bar_values(ax, vals, fmt=value_fmt)


def _plot_metric_line(ax, raw: pd.DataFrame, x: str, y: str, methods: list[str], xlabel: str, ylabel: str, *, query: str | None = None, logy: bool = False) -> None:
    dat = _summary(raw, x, y, methods, query=query)
    _line_summary(ax, dat, x, ylabel, logy=logy)
    ax.set_xlabel(xlabel)


def _plot_metric_mean_line(ax, raw: pd.DataFrame, x: str, y: str, methods: list[str], xlabel: str, ylabel: str, *, query: str | None = None) -> None:
    dat = raw.copy()
    if query:
        dat = dat.query(query)
    dat = dat[dat["method"].isin(methods)].groupby([x, "method"], as_index=False)[y].mean()
    for method in _ordered_methods(dat["method"].unique()):
        g = dat[dat["method"] == method].sort_values(x)
        ax.plot(
            g[x],
            g[y],
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "#333333"),
            label=_label(method),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style(ax)


def _clear_old_figures(out: Path) -> None:
    fig_dir = out / "figures" / "png"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("figure*.png"):
        old.unlink()


def generate_figures(raw: pd.DataFrame, agg: pd.DataFrame, queries: pd.DataFrame, out: Path = OUTPUT) -> None:
    _configure_style()
    _clear_old_figures(out)

    main_methods = ["standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse", "CARE-Fuse-95"]
    ablation_methods = [
        "standard_grover",
        "random_order_grover",
        "rarity_only_grover",
        "gate_cost_only_grover",
        "CARE-QSearch-adaptive",
        "CARE-Fuse",
        "CARE-Fuse-95",
    ]
    clean = raw[
        raw["method"].isin(main_methods)
        & (raw["noise_level"] == "clean")
        & (raw["missingness"] == 0)
        & (raw["shots"] == 1024)
    ]
    clean_summary = clean.groupby("method").agg(
        success=("success_probability", "mean"),
        oracle_depth=("oracle_depth", "mean"),
        cx=("CX_count", "mean"),
        total_cost=("total_resource_cost", "mean"),
        median_raqc=("RAQC", "median"),
        utility=("resource_utility", "mean"),
    )

    # Figure 1: dataset and query design.
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 74 / 25.4), constrained_layout=True)
    ax = axes[0]
    for ds, g in queries.groupby("dataset"):
        ax.scatter(
            g["K"],
            g["target_prevalence"],
            label=ds.title(),
            alpha=0.88,
            s=34,
            linewidths=0.45,
            edgecolors="white",
        )
    ax.set_xlabel("Predicate count, K")
    ax.set_ylabel("Target prevalence, M/N")
    ax.set_title("Query difficulty")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.0)
    _style(ax)
    _panel_label(ax, "a")

    ax = axes[1]
    design_counts = pd.Series(
        {
            "Datasets": raw["dataset"].nunique(),
            "N levels": raw["N"].nunique(),
            "Seeds": raw["seed"].nunique(),
            "Query IDs": raw["query_id"].nunique(),
            "Methods": raw["method"].nunique(),
            "Total runs": len(raw),
        }
    )
    y = np.arange(len(design_counts))
    colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#999999", "#444444"]
    ax.barh(y, design_counts.values, color=colors[: len(design_counts)], height=0.62)
    ax.set_yticks(y, design_counts.index)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Count, log scale")
    ax.set_title("Controlled experiment grid")
    _style(ax)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.45)
    for yi, val in zip(y, design_counts.values, strict=False):
        ax.text(val * 1.10, yi, f"{int(val):,}", va="center", ha="left", fontsize=7)
    ax.set_xlim(0.8, max(design_counts.values) * 2.4)
    _panel_label(ax, "b")
    _save(fig, "figure1_study_design", out)

    # Figure 2: main clean-result comparison.
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 118 / 25.4), constrained_layout=True)
    panels = [
        ("a", "success", "Mean success probability", (0.70, 1.02), "{:.3f}"),
        ("b", "total_cost", "Mean total resource cost", None, "{:.0f}"),
        ("c", "median_raqc", "Median RAQC", None, "{:.0f}"),
        ("d", "utility", "Mean resource utility", None, "{:.4f}"),
    ]
    for ax, (label, metric, ylabel, ylim, fmt) in zip(axes.ravel(), panels, strict=True):
        _method_bar(ax, clean_summary, metric, ylabel, main_methods, ylim=ylim, value_fmt=fmt)
        ax.set_title(ylabel)
        _panel_label(ax, label)
    _save(fig, "figure2_main_clean_performance", out)

    # Figure 3: resource scaling and oracle-cost growth.
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 70 / 25.4), constrained_layout=True)
    scaling_methods = ["classical_sequential", "standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse", "CARE-Fuse-95"]
    oracle_methods = ["standard_grover", "CARE-QSearch-adaptive", "CARE-Fuse-95"]
    _plot_metric_line(axes[0], raw, "N", "search_calls", scaling_methods, "Search space size, N", "Median search/oracle calls", logy=True)
    axes[0].set_title("Search scaling")
    _panel_label(axes[0], "a")
    _plot_metric_line(axes[1], raw, "K", "oracle_depth", oracle_methods, "Predicate count, K", "Median oracle depth")
    axes[1].set_title("Oracle depth")
    _panel_label(axes[1], "b")
    _plot_metric_line(axes[2], raw, "K", "CX_count", oracle_methods, "Predicate count, K", "Median CX gates")
    axes[2].set_title("CX count")
    _panel_label(axes[2], "c")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.52, 1.06), fontsize=6.2)
    _save(fig, "figure3_scaling_and_oracle_cost", out)

    # Figure 4: trade-off and ablations.
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 65 / 25.4), constrained_layout=True)
    ax = axes[0]
    trade = raw[raw["method"].isin(main_methods)].groupby("method").agg(
        success=("success_probability", "median"),
        cx=("CX_count", "median"),
        cost=("total_resource_cost", "median"),
    ).reindex(main_methods)
    x = np.arange(len(main_methods))
    bars = ax.bar(x, trade["success"], color=[COLORS[m] for m in main_methods], width=0.62)
    ax.set_xticks(x, [_label(m) for m in main_methods], rotation=20, ha="right")
    ax.set_ylabel("Median success probability")
    ax.set_ylim(0.65, 1.02)
    ax.set_title("Success-CX tradeoff")
    _style(ax, ygrid=True)
    for bar, val in zip(bars, trade["success"], strict=False):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5)
    ax2 = ax.twinx()
    ax2.plot(x, trade["cx"], color="#444444", marker="o", linewidth=1.1, label="Median CX")
    ax2.set_ylabel("Median CX gates")
    ax2.set_ylim(0, max(trade["cx"]) * 1.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.tick_params(direction="out", length=2.5, width=0.55, pad=1.5, labelsize=6)
    ax2.legend(loc="upper right", fontsize=5.5, frameon=False)
    _panel_label(ax, "a")

    ax = axes[1]
    vals = clean_summary.loc[main_methods, "total_cost"]
    rel = (vals / vals.loc["standard_grover"]).to_numpy()
    ax.bar(np.arange(len(main_methods)), rel, color=[COLORS[m] for m in main_methods], width=0.68)
    ax.axhline(1, color="#333333", linewidth=0.65)
    ax.set_xticks(np.arange(len(main_methods)), [_label(m) for m in main_methods], rotation=25, ha="right")
    ax.set_ylabel("Relative cost")
    ax.set_title("Clean cost ratio")
    _style(ax, ygrid=True)
    _bar_values(ax, list(rel), fmt="{:.2f}")
    _panel_label(ax, "b")

    ax = axes[2]
    ablation = raw[raw["method"].isin(ablation_methods)].groupby("method")["RAQC"].median().reindex(_ordered_methods(ablation_methods)).dropna()
    _barh(ax, [_label(m) for m in ablation.index], ablation.tolist(), [COLORS.get(m, "#333333") for m in ablation.index], "Median RAQC", logx=True)
    ax.set_title("Ordering ablation")
    _panel_label(ax, "c")
    _save(fig, "figure4_resource_tradeoff", out)

    # Figure 5: robustness.
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4), constrained_layout=True)
    _plot_metric_mean_line(axes[0], raw, "noise_two_qubit_depolarizing", "success_probability", main_methods, "Two-qubit depolarizing rate", "Mean success probability", query="missingness == 0 and shots == 1024")
    axes[0].set_title("Noise: success")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(loc="upper right", fontsize=5.5)
    _panel_label(axes[0], "a")
    noise = raw[
        raw["method"].isin(main_methods)
        & (raw["missingness"] == 0)
        & (raw["shots"] == 1024)
    ].groupby(["noise_two_qubit_depolarizing", "method"], as_index=False)["success_probability"].mean()
    clean_ref = noise[noise["noise_two_qubit_depolarizing"] == 0].set_index("method")["success_probability"]
    noise["success_retained"] = noise.apply(
        lambda r: r["success_probability"] / max(clean_ref.get(r["method"], np.nan), 1e-12),
        axis=1,
    )
    for method in _ordered_methods(main_methods):
        g = noise[noise["method"] == method].sort_values("noise_two_qubit_depolarizing")
        axes[1].plot(
            g["noise_two_qubit_depolarizing"],
            g["success_retained"],
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "#333333"),
            label=_label(method),
        )
    axes[1].set_xlabel("Two-qubit depolarizing rate")
    axes[1].set_ylabel("Success retained")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Noise: retention")
    axes[1].legend(loc="lower left", fontsize=5.5)
    _style(axes[1])
    _panel_label(axes[1], "b")
    miss = raw[
        raw["method"].isin(main_methods)
        & (raw["noise_level"] == "clean")
        & (raw["shots"] == 1024)
    ].groupby(["missingness", "missing_policy"], as_index=False)["recall"].mean()
    policy_colors = {"candidate-preserving": "#0072B2", "strict": "#D55E00"}
    policy_labels = {"candidate-preserving": "Candidate-preserving", "strict": "Strict"}
    for policy, g in miss.groupby("missing_policy"):
        g = g.sort_values("missingness")
        axes[2].plot(
            g["missingness"],
            g["recall"],
            marker="o" if policy == "strict" else "s",
            color=policy_colors.get(policy, "#333333"),
            label=policy_labels.get(policy, policy),
        )
    axes[2].set_xlabel("Missingness rate")
    axes[2].set_ylabel("Mean recall")
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("Missingness policy")
    axes[2].legend(loc="lower left", fontsize=5.5)
    _style(axes[2])
    _panel_label(axes[2], "c")
    _save(fig, "figure5_robustness", out)

    # Figure 6: Qiskit small-N sanity check.
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 74 / 25.4), constrained_layout=True)
    qiskit_path = out / "qiskit_validation" / "qiskit_smallN_validation.xlsx"
    if qiskit_path.exists():
        qv = pd.read_excel(qiskit_path, sheet_name="summary").set_index("method")
        methods = [m for m in main_methods if m in qv.index and m != "CARE-Fuse"]
        for ax, metric, ylabel, label in [
            (axes[0], "depth", "Mean oracle depth", "a"),
            (axes[1], "CX", "Mean CX gates", "b"),
        ]:
            y = np.arange(len(methods))
            heuristic = [qv.loc[m, f"heuristic_{metric}_mean"] for m in methods]
            qiskit = [qv.loc[m, f"qiskit_{metric}_mean"] for m in methods]
            height = 0.34
            ax.barh(y + height / 2, heuristic, height=height, color="#56B4E9", label="Heuristic", edgecolor="white", linewidth=0.35)
            ax.barh(y - height / 2, qiskit, height=height, color="#D55E00", label="Qiskit transpiled", edgecolor="white", linewidth=0.35)
            ax.set_yticks(y, [_label(m) for m in methods])
            ax.invert_yaxis()
            ax.set_xlabel(ylabel, fontsize=8.5)
            ax.set_title(f"Small-N {metric} validation", fontsize=9.5)
            xmax = max(max(heuristic), max(qiskit)) * 1.20
            ax.set_xlim(0, xmax)
            for yi, h_val, q_val in zip(y, heuristic, qiskit, strict=False):
                ax.text(h_val + 0.015 * xmax, yi + height / 2, f"{h_val:.0f}", va="center", ha="left", fontsize=7.5)
                ax.text(q_val + 0.015 * xmax, yi - height / 2, f"{q_val:.0f}", va="center", ha="left", fontsize=7.5)
            ax.legend(loc="lower right", frameon=False, fontsize=8)
            _style(ax, ygrid=False)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(axis="x", color="#E6E6E6", linewidth=0.45)
            _panel_label(ax, label)
    else:
        for ax, label in zip(axes, ["a", "b"], strict=True):
            ax.text(0.5, 0.5, "Run qiskit-validation mode", ha="center", va="center")
            ax.set_axis_off()
            _panel_label(ax, label)
    _save(fig, "figure6_qiskit_validation", out)
