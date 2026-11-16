"""Plots for analysing adaptive NN--BK weighting experiments."""

from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.getLogger("matplotlib").setLevel(logging.WARNING)


DLS_COLUMN = "Damerau-Levenshtein Acts"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNAL_COLUMNS = [
    "NN Entropy",
    "BK Entropy",
    "Prefix Similarity",
    "Prefix Support",
    "Prefix Alignment",
    "Contradiction",
    "BK discrimination factor",
    "Weight",
]
WEIGHT_RANGE_EDGES = np.linspace(0.0, 1.0, 6)
WEIGHT_RANGE_ORDER = [
    "[0.0, 0.2]",
    "(0.2, 0.4]",
    "(0.4, 0.6]",
    "(0.6, 0.8]",
    "(0.8, 1.0]",
]

DATASET_COLORS = {
    "BPIC13-CP": "#1F77B4",
    "BPIC13-I": "#FF7F0E",
    "Synthetic-P": "#2CA02C",
    "Synthetic-SP": "#D62728",
    "Helpdesk": "#E6B800" ,
    "Prepaid": "#8C564B",
    "Request": "#E377C2",
    "International": "#4C78A8",
    "Dreyers": "#BCBD22",
    "Domestic": "#17BECF",
    "BPIC12": "#003F5C",
    "Permit": "#6A3D9A",
    "Road Traffic": "#A05195",
    "Sepsis": "#00876C",
}

DATASET_NAMES = {
    "BPI2013_CP": "BPIC13-CP",
    "BPI2013_In": "BPIC13-I",
    "Synthetic": "Synthetic-P",
    "Synthetic1": "Synthetic-SP",
    "helpdesk": "Helpdesk",
    "PrepaidTravelCost": "Prepaid",
    "RequestForPayment": "Request",
    "InternationalDeclarations": "International",
    "log": "Dreyers",
    "DomesticDeclarations": "Domestic",
    "BPIC2012": "BPIC12",
    "PermitLog": "Permit",
    "Road_Traffic": "Road Traffic",
    "Sepsis_cases": "Sepsis",
}

def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths independently of the process working directory."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    working_directory_path = Path.cwd() / path
    if working_directory_path.exists():
        return working_directory_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def parse_history(value) -> list[float]:
    """Parse a scalar or list-valued result cell into a list of floats."""
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, (list, tuple)):
        return [float(item) for item in parsed]
    try:
        return [float(parsed)]
    except (TypeError, ValueError):
        return []




def _history_summary(signal: str, values: list[float]) -> float:
    if not values:
        return np.nan
    # Contradiction is an event: retain whether it occurred at any suffix step.
    return float(np.max(values) if signal == "Contradiction" else np.mean(values))


def load_adaptive_results(
    output_root: str | Path,
    logs: Sequence[str],
    methods: Sequence[str],
    results_folder: str = "results_vf",
    folds: Iterable[int] = range(3),
) -> pd.DataFrame:
    """Load result files and summarize per-step histories for each suffix."""
    output_root = resolve_project_path(output_root)
    frames = []
    for fold in folds:
        result_dir = output_root / str(fold) / results_folder / "CF"
        for log_name in logs:
            for method in methods:
                result_file = result_dir / f"{log_name}_{method}_results.csv"
                if not result_file.exists():
                    continue
                frame = pd.read_csv(result_file, quotechar="|")
                if DLS_COLUMN not in frame:
                    raise ValueError(f"Missing '{DLS_COLUMN}' in {result_file}")
                frame["Fold"] = fold
                frame["Log"] = log_name
                frame["Method"] = method
                frame["Source file"] = str(result_file)
                for signal in SIGNAL_COLUMNS:
                    if signal not in frame:
                        continue
                    history_column = f"{signal} history"
                    frame[history_column] = frame[signal].apply(parse_history)
                    frame[signal] = frame[history_column].apply(
                        lambda values, name=signal: _history_summary(name, values)
                    )
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No result files found below {output_root} for logs={list(logs)} "
            f"and methods={list(methods)}"
        )
    return pd.concat(frames, ignore_index=True)


def _save_figure(figure, output_dir: Path, name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}.pdf", output_dir / f"{name}.png"]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight", dpi=300)
    plt.close(figure)
    return paths

from matplotlib.patches import Patch


def _place_legend_below(figure, axes) -> None:
    """Shared legend with exact colors from DATASET_COLORS."""
    axes = np.atleast_1d(axes).ravel()

    datasets = []

    # Collect labels first
    for axis in axes:
        _, labels = axis.get_legend_handles_labels()

        for label in labels:
            if label in DATASET_COLORS and label not in datasets:
                datasets.append(label)

    # Then remove seaborn legends
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()

    handles = [
        Patch(
            facecolor=DATASET_COLORS[dataset],
            edgecolor=DATASET_COLORS[dataset],
            label=dataset,
        )
        for dataset in datasets
    ]

    if handles:
        figure.legend(
            handles=handles,
            title="Datasets:",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(6, len(handles)),
            frameon=True,
            fontsize=12,
            title_fontsize=15,
            alignment ="left",
        )

        figure.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        figure.tight_layout()

def _weight_ranges(weights: pd.Series) -> pd.Series:
    """Bin adaptive BK weights into ordered fifths of the unit interval."""
    return pd.cut(
        pd.to_numeric(weights, errors="coerce"),
        bins=WEIGHT_RANGE_EDGES,
        labels=WEIGHT_RANGE_ORDER,
        include_lowest=True,
    )


def plot_signal_weight_ranges_suffix(
    results: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Compare suffix-level reliability signals across adaptive weight ranges."""
    signals = [
        signal for signal in SIGNAL_COLUMNS[:-1]
        if signal in results and signal != "Contradiction"
    ]
    if not signals or "Weight" not in results:
        return []
    data = results.copy()
    data["Weight range"] = _weight_ranges(data["Weight"])
    long_data = data.melt(
        id_vars=["Log", "Method", "Weight range"], value_vars=signals,
        var_name="Signal", value_name="Value",
    ).dropna(subset=["Value", "Weight range"])
    rows = int(np.ceil(len(signals) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(14, 3.8 * rows), squeeze=False)
    for signal, axis in zip(signals, axes.flat):
        subset = long_data[long_data["Signal"] == signal]
        sns.boxplot(
            data=subset,
            x="Weight range",
            y="Value",
            hue="Log",
            order=WEIGHT_RANGE_ORDER,
            palette=DATASET_COLORS,
            ax=axis,
        )
        axis.set_ylabel(signal)
        axis.set_xlabel("BK weight range")
        axis.tick_params(axis="x", labelrotation=30)
    for axis in axes.flat[len(signals):]:
        axis.set_visible(False)
    _place_legend_below(figure, axes.flat[:len(signals)])
    return _save_figure(figure, output_dir, "signals_by_weight_range_suffix")

def _position_signal_data(results: pd.DataFrame, signals: Sequence[str]) -> pd.DataFrame:
    """Align list-valued signals with adaptive weights at decoding positions."""
    position_rows = []
    for _, row in results.iterrows():
        prediction = str(row["Predicted Acts"]).split(">>") + ["<END>"]
        weights = row.get("Weight history", [])
        for signal in signals:
            values = row.get(f"{signal} history", [])
            for position, (value, weight) in enumerate(zip(values, weights)):
                predicted_token = (
                    prediction[position] if position < len(prediction) else "<MISSING>"
                )
                position_rows.append(
                    {
                        "Log": row["Log"],
                        "Method": row["Method"],
                        "Signal": signal,
                        "Value": value,
                        "Weight": weight,
                        "Prediction target": (
                            "End of trace" if predicted_token == "<END>" else "Activity"
                        ),
                    }
                )
    data = pd.DataFrame(position_rows)
    if data.empty:
        return data
    data["Weight range"] = _weight_ranges(data["Weight"])
    return data.dropna(subset=["Value", "Weight range"])


def plot_signals_by_weight_range(
    results: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Compare signals across adaptive BK-weight ranges."""
    signals = [
        signal for signal in SIGNAL_COLUMNS[:-1]
        if signal in results
    ]
    if not signals:
        return []

    long_data = _position_signal_data(results, signals)
    if long_data.empty:
        return []
    long_data.to_csv(output_dir / "signal_position_weight_ranges.csv", index=False)
    rows = int(np.ceil(len(signals) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(14, 3.8 * rows), squeeze=False)
    for signal, axis in zip(signals, axes.flat):
        subset = long_data[long_data["Signal"] == signal]
        if signal == "Contradiction":
            sns.barplot(
                data=subset,
                x="Weight range",
                y="Value",
                hue="Log",
                saturation=1,
                order=WEIGHT_RANGE_ORDER,
                errorbar=None,
                palette=DATASET_COLORS,
                ax=axis,
            )
            axis.set_ylabel("Contradiction rate")
            axis.set_ylim(0, 1)
        else:
            sns.boxplot(
                data=subset,
                x="Weight range",
                y="Value",
                hue="Log",
                palette=DATASET_COLORS,
                order=WEIGHT_RANGE_ORDER,
                saturation=1,
                boxprops={"alpha": 1},
                ax=axis,
            )
            axis.set_ylabel(signal)
        axis.set_xlabel("BK weight range")
        axis.tick_params(axis="x", labelrotation=30)
    for axis in axes.flat[len(signals):]:
        axis.set_visible(False)
    _place_legend_below(figure, axes.flat[:len(signals)])
    return _save_figure(figure, output_dir, "signals_by_weight_range")


def plot_contradiction_rate(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot the contradiction proportion instead of a binary boxplot."""
    if "Contradiction" not in results:
        return []
    data = _position_signal_data(results, ["Contradiction"])
    if data.empty:
        return []

    summary = (
        data.groupby(
            ["Log", "Prediction target", "Weight range"],
            as_index=False,
            observed=True,
        )
        .agg(
            Contradiction_rate=("Value", "mean"),
            Contradictions=("Value", "sum"),
            Positions=("Value", "size"),
        )
    )
    summary.to_csv(output_dir / "contradiction_rate_by_weight_range.csv", index=False)

    targets = [
        target for target in ("Activity", "End of trace")
        if target in set(summary["Prediction target"])
    ]
    figure, axes = plt.subplots(
        1, len(targets), figsize=(7 * len(targets), 5), squeeze=False
    )
    for target, axis in zip(targets, axes.flat):
        subset = summary[summary["Prediction target"] == target]
        sns.barplot(
            data=subset,
            x="Weight range",
            y="Contradiction_rate",
            hue="Log",
            order=WEIGHT_RANGE_ORDER,
            errorbar=None,
            ax=axis,
        )
        axis.set_title(target)
        axis.set_xlabel("BK weight range")
        axis.set_ylabel("Contradiction rate")
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", labelrotation=30)
    _place_legend_below(figure, axes.flat)
    return _save_figure(figure, output_dir, "contradiction_rate_by_weight_range")

def plot_entropy_heatmaps(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    if not {"NN Entropy", "BK Entropy"}.issubset(results.columns):
        return []
    data = results.dropna(subset=["NN Entropy", "BK Entropy"]).copy()
    bins = np.linspace(0.0, 1.0, 6)
    data["NN entropy bin"] = pd.cut(data["NN Entropy"], bins, include_lowest=True)
    data["BK entropy bin"] = pd.cut(data["BK Entropy"], bins, include_lowest=True)
    mean_dls = data.pivot_table(
        index="NN entropy bin", columns="BK entropy bin", values=DLS_COLUMN,
        aggfunc="mean", observed=False,
    )
    counts = data.pivot_table(
        index="NN entropy bin", columns="BK entropy bin", values=DLS_COLUMN,
        aggfunc="size", fill_value=0, observed=False,
    )
    mean_dls.to_csv(output_dir / "entropy_heatmap_mean_dls.csv")
    counts.to_csv(output_dir / "entropy_heatmap_counts.csv")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(mean_dls, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=axes[0])
    sns.heatmap(counts, annot=True, fmt="g", cmap="Blues", ax=axes[1])
    axes[0].set_title("Mean DLS")
    axes[1].set_title("Number of suffixes")
    for axis in axes:
        axis.set_xlabel("BK entropy")
        axis.set_ylabel("NN entropy")
    figure.tight_layout()
    return _save_figure(figure, output_dir, "nn_bk_entropy_heatmaps")

def generate_adaptive_plots(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate all available plots and return their output paths."""
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    generated = []
    for plot_function in (
        plot_signals_by_weight_range,
        plot_entropy_heatmaps,
    ):
        generated.extend(plot_function(results, output_dir))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",default="docs/source/data/CAiSe27/output", type=Path)
    parser.add_argument("--logs", nargs="+",default= ['BPI2013_CP',"BPI2013_In", "Synthetic", "Synthetic1","helpdesk", 'PrepaidTravelCost','RequestForPayment',
                                                      'InternationalDeclarations',"log", 'DomesticDeclarations',"BPIC2012", "PermitLog", "Road_Traffic", "Sepsis_cases"
                                                      ])
    parser.add_argument("--methods", default=["adaptive_w_both_entropy_sim_supp_align_fallback"]#, "adaptive_w_BK_entropy", "adaptive_w_both_entropy","adaptive_w_both_entropy_supp"]
                        , nargs="+")
    parser.add_argument("--results-folder", default="results_vf")
    parser.add_argument("--plot-dir",default="docs/source/data/CAiSE27/plots", type=Path)
    args = parser.parse_args()
    results = load_adaptive_results(
        args.output_root, args.logs, args.methods, args.results_folder
    )
    results["Log"] = results["Log"].replace(DATASET_NAMES)
    paths = generate_adaptive_plots(results, args.plot_dir)
    print(f"Generated {len(paths)} plot files in {args.plot_dir}")


if __name__ == "__main__":
    main()
