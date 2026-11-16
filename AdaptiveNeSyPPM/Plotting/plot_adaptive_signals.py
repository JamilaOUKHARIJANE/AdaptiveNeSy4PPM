#!/usr/bin/env python3
"""
Adaptive-weight signal analysis for suffix prediction.

Generates:
1. Full Spearman correlation heatmap
2. Signal -> adaptive-weight correlation bar chart
3. Signal -> DLS correlation bar chart
4. Adaptive weight vs DLS scatter plot
5. Adaptive-weight bins vs mean DLS
6. Individual signal vs DLS plots
7. Contradiction vs DLS boxplot
8. Weight vs DLS split by agreement/contradiction
9. Optional prefix-length stratified correlations
10. CSV summaries of correlations and weight bins

Example:
    python plot_adaptive_signals.py results.csv --output-dir plots

The script expects columns similar to:
    NN Entropy
    BK Entropy
    Prefix Similarity
    Prefix Support
    Prefix Alignment
    Contradiction
    BK discrimination factor
    Weight
    Damerau-Levenshtein Acts

List-valued cells such as "[0.4, 0.7, 1.0]" are aggregated by mean
at prediction/suffix level before correlation with the row-level DLS.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


SIGNAL_COLUMNS = [
    "NN Entropy",
    "BK Entropy",
    "Prefix Similarity",
    "Prefix Support",
    "Prefix Alignment",
    "Contradiction",
    "BK discrimination factor",
]
DISPLAY_SIGNAL_NAMES = {
    "NN Entropy": "1-NN Entropy",
    "BK Entropy": "1-BK Entropy",
}

WEIGHT_COLUMN = "Weight"

DLS_CANDIDATES = [
    "Damerau-Levenshtein Acts",
    "DLS",
    "Damerau Levenshtein Acts",
    "Damerau-Levenshtein",
]

PREFIX_LENGTH_CANDIDATES = [
    "Prefix Length",
    "prefix_length",
    "prefix length",
    "Prefix length",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot correlations between adaptive signals, weight, and DLS."
    )
    parser.add_argument(
        "csv_file",
        type=Path, nargs="*", default=None,
        help="Input CSV containing adaptive signals and DLS."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../docs/source/data/CAiSe27/adaptive_signal_plots"),
        help="Directory where plots and summary CSV files are saved."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("docs/source/data/CAiSe27/output"),
        help="Root searched recursively when no CSV files are supplied.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median", "first", "last", "max", "min"],
        default="mean",
        help="How to aggregate list-valued signals per suffix prediction."
    )
    return parser.parse_args()


def parse_sequence(value):
    """Convert a CSV cell into a numeric list when possible."""
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    else:
        text = str(value).strip()

        # Some exported lists may include separators such as "|".
        text = text.replace("|", "")

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                return [float(text)]
            except ValueError:
                return []

        if isinstance(parsed, (list, tuple, np.ndarray)):
            values = parsed
        else:
            values = [parsed]

    result = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result


def aggregate_sequence(value, method="mean"):
    values = parse_sequence(value)

    if len(values) == 0:
        return np.nan

    arr = np.asarray(values, dtype=float)

    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))
    if method == "first":
        return float(arr[0])
    if method == "last":
        return float(arr[-1])
    if method == "max":
        return float(np.max(arr))
    if method == "min":
        return float(np.min(arr))

    raise ValueError(f"Unknown aggregation method: {method}")


def find_first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def safe_spearman(x, y):
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(data) < 3:
        return np.nan, np.nan, len(data)

    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan, len(data)

    rho, p = spearmanr(data["x"], data["y"])
    return float(rho), float(p), len(data)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare_dataframe(df, aggregation):
    df = df.copy()

    available_signals = []

    for col in SIGNAL_COLUMNS:
        if col in df.columns:
            new_col = f"{col}__agg"
            df[new_col] = df[col].apply(
                lambda x: aggregate_sequence(x, aggregation)
            )
            available_signals.append(col)

    if WEIGHT_COLUMN not in df.columns:
        raise KeyError(
            f"Required weight column '{WEIGHT_COLUMN}' not found.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df["Adaptive Weight__agg"] = df[WEIGHT_COLUMN].apply(
        lambda x: aggregate_sequence(x, aggregation)
    )

    dls_col = find_first_existing(df.columns, DLS_CANDIDATES)

    if dls_col is None:
        raise KeyError(
            "Could not find a DLS column. Expected one of:\n"
            f"{DLS_CANDIDATES}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    df["DLS"] = pd.to_numeric(df[dls_col], errors="coerce")

    prefix_col = find_first_existing(df.columns, PREFIX_LENGTH_CANDIDATES)

    return df, available_signals, dls_col, prefix_col


def make_full_correlation_heatmap(df, available_signals, out_dir):
    feature_map = {
        DISPLAY_SIGNAL_NAMES.get(signal, signal): f"{signal}__agg"
        for signal in available_signals
    }

    feature_map["Adaptive Weight"] = "Adaptive Weight__agg"
    usable = {
        label: col
        for label, col in feature_map.items()
        if df[col].notna().sum() > 0 and df[col].nunique(dropna=True) > 1
    }

    if len(usable) < 2:
        return

    corr_data = pd.DataFrame({
        label: df[col]
        for label, col in usable.items()
    })

    corr = corr_data.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Adaptive signals, adaptive weight, and DLS")

    save_figure(fig, out_dir / "01_full_spearman_heatmap.png")
    corr.to_csv(out_dir / "full_spearman_correlation_matrix.csv")


def compute_signal_correlations(df, available_signals):
    rows = []

    for signal in available_signals:
        signal_col = f"{signal}__agg"

        rho_weight, p_weight, n_weight = safe_spearman(
            df[signal_col],
            df["Adaptive Weight__agg"],
        )

        rho_dls, p_dls, n_dls = safe_spearman(
            df[signal_col],
            df["DLS"],
        )

        rows.append({
            "Signal": DISPLAY_SIGNAL_NAMES.get(signal, signal),
            "rho_signal_weight": rho_weight,
            "p_signal_weight": p_weight,
            "n_signal_weight": n_weight,
            "rho_signal_DLS": rho_dls,
            "p_signal_DLS": p_dls,
            "n_signal_DLS": n_dls,
        })

    return pd.DataFrame(rows)


def plot_horizontal_correlations(summary, value_col, title, xlabel, path):
    data = summary[["Signal", value_col]].dropna().copy()

    if data.empty:
        return

    data = data.sort_values(value_col)

    fig_height = max(4, 0.55 * len(data) + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    y = np.arange(len(data))

    ax.barh(y, data[value_col].values)
    ax.set_yticks(y)
    ax.set_yticklabels(data["Signal"].values)
    ax.axvline(0, linewidth=1)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for idx, value in enumerate(data[value_col].values):
        offset = 0.03 if value >= 0 else -0.03
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            idx,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    save_figure(fig, path)


def plot_weight_vs_dls(df, out_dir):
    data = df[["Adaptive Weight__agg", "DLS"]].dropna()

    if len(data) < 3:
        return

    rho, p, _ = safe_spearman(
        data["Adaptive Weight__agg"],
        data["DLS"],
    )

    rng = np.random.default_rng(42)

    # Small jitter only for visualization.
    y_range = data["DLS"].max() - data["DLS"].min()
    jitter_sd = 0.01 if y_range <= 1 else 0.01 * y_range
    jittered_y = data["DLS"].values + rng.normal(
        0,
        jitter_sd,
        size=len(data),
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.scatter(
        data["Adaptive Weight__agg"],
        jittered_y,
        alpha=0.30,
        s=24,
    )

    # Linear trend is only a visualization aid.
    if data["Adaptive Weight__agg"].nunique() > 1:
        coef = np.polyfit(
            data["Adaptive Weight__agg"],
            data["DLS"],
            deg=1,
        )
        xx = np.linspace(
            data["Adaptive Weight__agg"].min(),
            data["Adaptive Weight__agg"].max(),
            200,
        )
        ax.plot(xx, np.polyval(coef, xx), linewidth=2)

    ax.set_xlabel("Mean adaptive BK weight")
    ax.set_ylabel("DLS")
    ax.set_title(
        f"Adaptive weight vs DLS\n"
        f"Spearman ρ={rho:.2f}, p={p:.3g}"
    )

    save_figure(fig, out_dir / "04_adaptive_weight_vs_DLS.png")


def plot_weight_bins(df, out_dir):
    data = df[["Adaptive Weight__agg", "DLS"]].dropna().copy()

    if data.empty:
        return None

    bins = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
    labels = [
        "[0, 0.2]",
        "(0.2, 0.4]",
        "(0.4, 0.6]",
        "(0.6, 0.8]",
        "(0.8, 1.0]",
    ]

    data["Weight bin"] = pd.cut(
        data["Adaptive Weight__agg"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    summary = (
        data.groupby("Weight bin", observed=True)["DLS"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    if summary.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(
        summary["Weight bin"].astype(str),
        summary["mean"],
    )

    ax.set_xlabel("Mean adaptive BK weight")
    ax.set_ylabel("Mean DLS")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction quality across adaptive-weight regimes")

    for i, row in summary.iterrows():
        ax.text(
            i,
            row["mean"] + 0.02,
            f"n={int(row['count'])}",
            ha="center",
            fontsize=9,
        )

    save_figure(fig, out_dir / "05_weight_bins_vs_mean_DLS.png")
    summary.to_csv(out_dir / "weight_bin_DLS_summary.csv", index=False)

    return summary


def plot_each_signal_vs_dls(df, available_signals, out_dir):
    signal_dir = out_dir / "signal_vs_DLS"
    signal_dir.mkdir(exist_ok=True)

    for signal in available_signals:
        col = f"{signal}__agg"
        display_signal = DISPLAY_SIGNAL_NAMES.get(signal, signal)

        data = df[[col, "DLS"]].dropna()

        if len(data) < 3 or data[col].nunique() < 2:
            continue

        rho, p, _ = safe_spearman(data[col], data["DLS"])

        # Binary/discrete signals are more useful as grouped boxplots.
        unique_count = data[col].nunique()

        if unique_count <= 4:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))

            groups = sorted(data[col].unique())
            group_values = [
                data.loc[data[col] == g, "DLS"].values
                for g in groups
            ]

            ax.boxplot(
                group_values,
                tick_labels=[str(round(g, 3)) for g in groups],
            )

            ax.set_xlabel(display_signal)
            ax.set_ylabel("DLS")
            ax.set_title(
                f"{display_signal} vs DLS\n"
                f"Spearman ρ={rho:.2f}, p={p:.3g}"
            )

        else:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))

            ax.scatter(
                data[col],
                data["DLS"],
                alpha=0.30,
                s=22,
            )

            if data[col].nunique() > 1:
                coef = np.polyfit(data[col], data["DLS"], deg=1)
                xx = np.linspace(
                    data[col].min(),
                    data[col].max(),
                    200,
                )
                ax.plot(
                    xx,
                    np.polyval(coef, xx),
                    linewidth=2,
                )

            ax.set_xlabel(display_signal)
            ax.set_ylabel("DLS")
            ax.set_title(
                f"{display_signal} vs DLS\n"
                f"Spearman ρ={rho:.2f}, p={p:.3g}"
            )

        safe_name = (
            display_signal.lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        save_figure(
            fig,
            signal_dir / f"{safe_name}_vs_DLS.png",
        )


def plot_contradiction_vs_dls(df, out_dir):
    col = "Contradiction__agg"

    if col not in df.columns:
        return

    data = df[[col, "DLS"]].dropna()

    if data.empty:
        return

    # If contradiction is averaged over a sequence, it may not be binary.
    # We therefore create two intuitive groups:
    # no contradiction anywhere/average=0 vs contradiction present/average>0.
    grouped = data.copy()
    grouped["Contradiction group"] = np.where(
        grouped[col] > 0,
        "Contradiction present",
        "No contradiction",
    )

    groups = [
        "No contradiction",
        "Contradiction present",
    ]

    values = [
        grouped.loc[
            grouped["Contradiction group"] == group,
            "DLS"
        ].values
        for group in groups
    ]

    nonempty = [
        (group, vals)
        for group, vals in zip(groups, values)
        if len(vals) > 0
    ]

    if len(nonempty) < 1:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.boxplot(
        [vals for _, vals in nonempty],
        tick_labels=[group for group, _ in nonempty],
    )

    ax.set_ylabel("DLS")
    ax.set_title("DLS under NN-BK agreement and contradiction")

    save_figure(fig, out_dir / "07_contradiction_vs_DLS.png")


def plot_weight_vs_dls_by_contradiction(df, out_dir):
    contradiction_col = "Contradiction__agg"

    if contradiction_col not in df.columns:
        return

    data = df[
        [
            "Adaptive Weight__agg",
            contradiction_col,
            "DLS",
        ]
    ].dropna().copy()

    if len(data) < 3:
        return

    data["Group"] = np.where(
        data[contradiction_col] > 0,
        "Contradiction",
        "Agreement",
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    for group in ["Agreement", "Contradiction"]:
        subset = data[data["Group"] == group]

        if subset.empty:
            continue

        ax.scatter(
            subset["Adaptive Weight__agg"],
            subset["DLS"],
            alpha=0.32,
            s=24,
            label=group,
        )

        if (
            len(subset) >= 3
            and subset["Adaptive Weight__agg"].nunique() > 1
        ):
            coef = np.polyfit(
                subset["Adaptive Weight__agg"],
                subset["DLS"],
                deg=1,
            )
            xx = np.linspace(
                subset["Adaptive Weight__agg"].min(),
                subset["Adaptive Weight__agg"].max(),
                100,
            )
            ax.plot(
                xx,
                np.polyval(coef, xx),
                linewidth=2,
            )

    ax.set_xlabel("Mean adaptive BK weight")
    ax.set_ylabel("DLS")
    ax.set_title("Adaptive weight vs DLS by NN-BK agreement")
    ax.legend()

    save_figure(
        fig,
        out_dir / "08_weight_vs_DLS_by_contradiction.png",
    )


def compute_contradiction_correlations(df):
    contradiction_col = "Contradiction__agg"

    if contradiction_col not in df.columns:
        return pd.DataFrame()

    temp = df[
        [
            "Adaptive Weight__agg",
            contradiction_col,
            "DLS",
        ]
    ].dropna().copy()

    temp["Group"] = np.where(
        temp[contradiction_col] > 0,
        "Contradiction",
        "Agreement",
    )

    rows = []

    for group, subset in temp.groupby("Group"):
        rho, p, n = safe_spearman(
            subset["Adaptive Weight__agg"],
            subset["DLS"],
        )

        rows.append({
            "Group": group,
            "n": n,
            "rho_weight_DLS": rho,
            "p_weight_DLS": p,
            "mean_DLS": subset["DLS"].mean(),
            "mean_weight": subset["Adaptive Weight__agg"].mean(),
        })

    return pd.DataFrame(rows)


def prefix_length_analysis(df, prefix_col, out_dir):
    if prefix_col is None:
        return

    temp = df[
        [
            prefix_col,
            "Adaptive Weight__agg",
            "DLS",
        ]
    ].dropna()

    if temp.empty:
        return

    rows = []

    for prefix_length, subset in temp.groupby(prefix_col):
        rho, p, n = safe_spearman(
            subset["Adaptive Weight__agg"],
            subset["DLS"],
        )

        rows.append({
            "Prefix Length": prefix_length,
            "n": n,
            "rho_weight_DLS": rho,
            "p_weight_DLS": p,
            "mean_DLS": subset["DLS"].mean(),
            "mean_weight": subset["Adaptive Weight__agg"].mean(),
        })

    summary = pd.DataFrame(rows).sort_values("Prefix Length")
    summary.to_csv(
        out_dir / "prefix_length_weight_DLS_correlations.csv",
        index=False,
    )

    if len(summary) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        summary["Prefix Length"],
        summary["rho_weight_DLS"],
        marker="o",
    )

    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Spearman ρ: adaptive weight vs DLS")
    ax.set_ylim(-1, 1)
    ax.set_title("Weight-DLS relationship across prefix lengths")

    save_figure(
        fig,
        out_dir / "09_weight_DLS_correlation_by_prefix_length.png",
    )


def analyze_dataset(csv_file, output_dir, aggregation):
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = csv_file if isinstance(csv_file, (list, tuple)) else [csv_file]
    df = pd.concat(
        [pd.read_csv(path, quotechar="|") for path in input_files],
        ignore_index=True,
    )

    df, available_signals, dls_col, prefix_col = prepare_dataframe(
        df,
        aggregation,
    )

    print(f"Input file(s): {', '.join(str(path) for path in input_files)}")
    print(f"Rows: {len(df)}")
    print(f"DLS column: {dls_col}")
    print(f"Aggregation: {aggregation}")
    print(f"Signals found: {available_signals}")

    missing = [
        signal
        for signal in SIGNAL_COLUMNS
        if signal not in df.columns
    ]

    if missing:
        print(f"Signals not present and skipped: {missing}")

    if prefix_col:
        print(f"Prefix-length column: {prefix_col}")
    else:
        print("No prefix-length column found; prefix analysis skipped.")

    make_full_correlation_heatmap(
        df,
        available_signals,
        output_dir,
    )

    corr_summary = compute_signal_correlations(
        df,
        available_signals,
    )

    corr_summary.to_csv(
        output_dir / "signal_correlation_summary.csv",
        index=False,
    )

    plot_horizontal_correlations(
        corr_summary,
        value_col="rho_signal_weight",
        title="Which reliability signals drive the adaptive weight?",
        xlabel="Spearman ρ with adaptive weight",
        path=output_dir / "02_signals_vs_adaptive_weight.png",
    )

    plot_horizontal_correlations(
        corr_summary,
        value_col="rho_signal_DLS",
        title="Which reliability signals are associated with DLS?",
        xlabel="Spearman ρ with DLS",
        path=output_dir / "03_signals_vs_DLS.png",
    )

    plot_weight_vs_dls(
        df,
        output_dir,
    )

    plot_weight_bins(
        df,
        output_dir,
    )

    plot_each_signal_vs_dls(
        df,
        available_signals,
        output_dir,
    )

    plot_contradiction_vs_dls(
        df,
        output_dir,
    )

    plot_weight_vs_dls_by_contradiction(
        df,
        output_dir,
    )

    contradiction_summary = compute_contradiction_correlations(df)

    if not contradiction_summary.empty:
        contradiction_summary.to_csv(
            output_dir /
            "agreement_contradiction_weight_DLS_summary.csv",
            index=False,
        )

    prefix_length_analysis(
        df,
        prefix_col,
        output_dir,
    )

    print()
    print("Saved outputs to:")
    print(output_dir.resolve())
    print()
    print("Main paper-oriented figures:")
    print("  02_signals_vs_adaptive_weight.png")
    print("  03_signals_vs_DLS.png")
    print("  04_adaptive_weight_vs_DLS.png")
    print("  05_weight_bins_vs_mean_DLS.png")
    print("  08_weight_vs_DLS_by_contradiction.png")


def expand_fold_files(path):
    """Return matching fold 0/1/2 files when a path belongs to fold 0."""
    path = Path(path)
    parts = list(path.parts)
    if "0" not in parts:
        return [path]
    fold_index = parts.index("0")
    candidates = []
    for fold in range(3):
        fold_parts = parts.copy()
        fold_parts[fold_index] = str(fold)
        candidate = Path(*fold_parts)
        if candidate.exists():
            candidates.append(candidate)
    return candidates or [path]


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    input_root = args.input_root
    if not input_root.is_absolute():
        input_root = (project_root / input_root).resolve()
    csv_files = args.csv_file or sorted(
        path for path in input_root.glob("**/*.csv")
        if "adaptive_w_dls" in path.name and "results" in path.name
    )
    if not csv_files:
        raise FileNotFoundError(f"No adaptive-signal CSV files found under {input_root}")

    grouped_files = {}
    expanded_files = []
    for csv_file in csv_files:
        expanded_files.extend(expand_fold_files(csv_file))
    for csv_file in expanded_files:
        dataset_name = csv_file.name.split("_adaptive", 1)[0]
        grouped_files.setdefault(dataset_name, []).append(csv_file)

    for dataset_name, dataset_files in grouped_files.items():
        analyze_dataset(
            dataset_files,
            args.output_dir / dataset_name,
            args.aggregation,
        )


if __name__ == "__main__":
    main()
