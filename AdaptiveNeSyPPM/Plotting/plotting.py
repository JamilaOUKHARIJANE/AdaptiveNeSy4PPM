"""Plots for analysing adaptive NN--Weighting experiments."""

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

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
ENTROPY_SIGNALS = {"NN Entropy", "BK Entropy"}
SIGNAL_LABELS = {
    "NN Entropy": "1-NN Entropy",
    "BK Entropy": "1-BK Entropy",
}
SIGNAL_COLORS = {
    "1-NN Entropy": "#1F77B4",
    "1-BK Entropy": "#FF7F0E",
    "Prefix Similarity": "#2CA02C",
    "Prefix Support": "#D62728",
    "Prefix Alignment": "#9467BD",
    "Contradiction": "#8C564B",
    "BK discrimination factor": "#E377C2",
}
WEIGHT_BINS = np.linspace(0.0, 1.0, 11)
WEIGHT_BIN_LABELS = [f"{left:.1f}-{right:.1f}" for left, right in zip(
    WEIGHT_BINS[:-1], WEIGHT_BINS[1:]
)]

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
                    if signal in ENTROPY_SIGNALS:
                        frame[history_column] = frame[history_column].apply(
                            lambda values: [1.0 - value for value in values]
                        )
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

def plot_signal_weights_suffix(
    results: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Plot suffix-level reliability signals against adaptive Weights."""
    signals = [
        signal for signal in SIGNAL_COLUMNS[:-1]
        if signal in results and signal != "Contradiction"
    ]
    if not signals or "Weight" not in results:
        return []
    data = results.copy()
    data["Weight"] = pd.to_numeric(data["Weight"], errors="coerce")
    long_data = data.melt(
        id_vars=["Log", "Method", "Weight"], value_vars=signals,
        var_name="Signal", value_name="Value",
    ).dropna(subset=["Value", "Weight"])
    rows = int(np.ceil(len(signals) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(14, 3.8 * rows), squeeze=False)
    for signal, axis in zip(signals, axes.flat):
        subset = long_data[long_data["Signal"] == signal]
        sns.lineplot(
            data=subset,
            x="Weight",
            y="Value",
            hue="Log",
            palette=DATASET_COLORS,
            estimator="mean",
            errorbar=None,
            ax=axis,
        )
        axis.set_ylabel(signal)
        axis.set_xlabel("Weight")
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(np.linspace(0, 1, 6))
    for axis in axes.flat[len(signals):]:
        axis.set_visible(False)
    _place_legend_below(figure, axes.flat[:len(signals)])
    return _save_figure(figure, output_dir, "signals_by_weight_suffix")

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
                        "Signal": SIGNAL_LABELS.get(signal, signal),
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
    data["Weight"] = pd.to_numeric(data["Weight"], errors="coerce")
    return data.dropna(subset=["Value", "Weight"])


def plot_signals_by_weight(
    results: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Plot all signal values over Weight in dataset panels."""
    signals = [
        signal for signal in SIGNAL_COLUMNS[:-1]
        if signal in results
    ]
    if not signals:
        return []

    long_data = _position_signal_data(results, signals)
    if long_data.empty:
        return []
    long_data.to_csv(output_dir / "signal_position_weights.csv", index=False)
    display_signals = [SIGNAL_LABELS.get(signal, signal) for signal in signals]
    datasets = sorted(long_data["Log"].dropna().unique())
    long_data = long_data.dropna(subset=["Weight", "Value"])
    pdf_path = output_dir / "signals_by_weight_and_prefix_length.pdf"
    with PdfPages(pdf_path) as pdf:
        columns = 3
        rows = int(np.ceil(len(datasets) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(15, 4.3 * rows), squeeze=False,
            constrained_layout=True,
        )
        markers = ["o", "s", "^", "D", "P", "X", "v"]
        signal_colors = {
            signal: SIGNAL_COLORS.get(signal, "#7F7F7F")
            for signal in display_signals
        }
        for dataset, axis in zip(datasets, axes.flat):
            dataset_data = long_data[long_data["Log"] == dataset]
            for signal, display_signal, marker in zip(
                signals, display_signals, markers
            ):
                subset = dataset_data[dataset_data["Signal"] == display_signal]
                sns.lineplot(
                    data=subset, x="Weight", y="Value",
                    color=signal_colors[display_signal],
                    marker=marker, markersize=3, linewidth=1.2,
                    estimator="mean", errorbar=None, ax=axis,
                )
            axis.set_title(str(dataset))
            axis.set_xlabel("Weight")
            axis.set_ylabel("Signal value")
            axis.set_ylim(-0.03, 1.03)
            axis.set_xlim(-0.03, 1.03)
            axis.set_xticks(np.linspace(0, 1, 6))
        for axis in axes.flat[len(datasets):]:
            axis.set_visible(False)
        handles = [
            Line2D([0], [0], marker=marker, linestyle="", color=signal_colors[display_signal],
                   markersize=7, label=display_signal)
            for display_signal, marker in zip(display_signals, markers)
        ]
        figure.legend(handles=handles, title="Signal", loc="lower center",
                      bbox_to_anchor=(0.5, 0.0), ncol=min(4, len(handles)))
        #figure.suptitle("Signal values by Weight", fontsize=16)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    boxplot_path = output_dir / "signals_boxplots_by_dataset.pdf"
    with PdfPages(boxplot_path) as pdf:
        columns = 3
        rows = int(np.ceil(len(datasets) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(15, 4.3 * rows), squeeze=False,
            constrained_layout=True,
        )
        boxplot_data = long_data.copy()
        boxplot_data["Weight range"] = pd.cut(
            boxplot_data["Weight"], bins=WEIGHT_BINS,
            labels=WEIGHT_BIN_LABELS, include_lowest=True,
        )
        for dataset, axis in zip(datasets, axes.flat):
            subset = boxplot_data[boxplot_data["Log"] == dataset]
            sns.boxplot(
                data=subset,
                x="Weight range",
                y="Value",
                hue="Signal",
                palette=signal_colors,
                order=WEIGHT_BIN_LABELS,
                legend=False,
                showfliers=False,
                medianprops={"color": "none"},
                ax=axis,
            )
            hatches = ["", "///", "\\\\", "...", "xx", "++", "oo"]
            for patch_index, patch in enumerate(axis.patches):
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)
                patch.set_hatch(hatches[patch_index % len(display_signals)])
            axis.set_title(str(dataset))
            axis.set_xlabel("weight")
            #axis.set_ylabel("Signal value")
            axis.set_ylim(-0.03, 1.03)
            axis.tick_params(axis="x", labelrotation=35)
        for axis in axes.flat[len(datasets):]:
            axis.set_visible(False)
        reference_patches = next(
            (axis.patches for axis in axes.flat[:len(datasets)] if axis.patches), []
        )
        boxplot_handles = []
        for index, signal in enumerate(display_signals):
            patch = reference_patches[index % len(reference_patches)]
            boxplot_handles.append(
                Patch(
                    facecolor=patch.get_facecolor(),
                    edgecolor=patch.get_edgecolor(),
                    hatch=patch.get_hatch(),
                    label=signal,
                )
            )
        figure.legend(
            handles=boxplot_handles,
            #title="Signal",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(3, len(boxplot_handles)),
            frameon=True,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return [pdf_path, boxplot_path]


def plot_contradiction_rate(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot the contradiction proportion instead of a binary boxplot."""
    if "Contradiction" not in results:
        return []
    data = _position_signal_data(results, ["Contradiction"])
    if data.empty:
        return []

    summary = (
        data.groupby(
            ["Log", "Prediction target", "Weight"],
            as_index=False,
            observed=True,
        )
        .agg(
            Contradiction_rate=("Value", "mean"),
            Contradictions=("Value", "sum"),
            Positions=("Value", "size"),
        )
    )
    summary.to_csv(output_dir / "contradiction_rate_by_weight.csv", index=False)

    targets = [
        target for target in ("Activity", "End of trace")
        if target in set(summary["Prediction target"])
    ]
    figure, axes = plt.subplots(
        1, len(targets), figsize=(7 * len(targets), 5), squeeze=False
    )
    for target, axis in zip(targets, axes.flat):
        subset = summary[summary["Prediction target"] == target]
        sns.lineplot(
            data=subset,
            x="Weight",
            y="Contradiction_rate",
            hue="Log",
            palette=DATASET_COLORS,
            estimator=None,
            ax=axis,
        )
        axis.set_title(target)
        axis.set_xlabel("Weight")
        axis.set_ylabel("Contradiction rate")
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_ylim(-0.03, 1.03)
        axis.tick_params(axis="x", labelrotation=30)
    _place_legend_below(figure, axes.flat)
    return _save_figure(figure, output_dir, "contradiction_rate_by_weight_range")

def plot_entropy_heatmaps(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    if not {"NN Entropy", "BK Entropy"}.issubset(results.columns):
        return []
    data = results.dropna(subset=["NN Entropy", "BK Entropy"]).copy()
    data["NN entropy bin"] = pd.cut(
        data["NN Entropy"], WEIGHT_BINS, labels=WEIGHT_BIN_LABELS,
        include_lowest=True,
    )
    data["BK entropy bin"] = pd.cut(
        data["BK Entropy"], WEIGHT_BINS, labels=WEIGHT_BIN_LABELS,
        include_lowest=True,
    )
    mean_dls = data.pivot_table(
        index="NN entropy bin", columns="BK entropy bin", values=DLS_COLUMN,
        aggfunc="mean", observed=False,
    )
    counts = data.pivot_table(
        index="NN entropy bin", columns="BK entropy bin", values=DLS_COLUMN,
        aggfunc="size", fill_value=0, observed=False,
    )
    mean_dls = mean_dls.reindex(
        index=WEIGHT_BIN_LABELS, columns=WEIGHT_BIN_LABELS
    )
    counts = counts.reindex(
        index=WEIGHT_BIN_LABELS, columns=WEIGHT_BIN_LABELS, fill_value=0
    )
    mean_dls.to_csv(output_dir / "entropy_heatmap_mean_dls.csv")
    counts.to_csv(output_dir / "entropy_heatmap_counts.csv")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(mean_dls, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=axes[0])
    sns.heatmap(counts, annot=True, fmt="g", cmap="Blues", ax=axes[1])
    axes[0].set_title("Mean DLS")
    axes[1].set_title("Number of suffixes")
    for axis in axes:
        axis.set_xlabel("1-BK entropy")
        axis.set_ylabel("1-NN entropy")
    figure.tight_layout()
    return _save_figure(figure, output_dir, "nn_bk_entropy_heatmaps")

def generate_adaptive_plots(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate all available plots and return their output paths."""
    output_dir = resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    generated = []
    for plot_function in (
        plot_signals_by_weight,
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
