"""Generate step-level adaptive-signal correlation plots."""

import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

from AdaptiveNeSyPPM.Data_preprocessing.log_utils import LogData


AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 13
PLOT_FILENAME = "05_signals_vs_step_adaptive_weight.pdf"

SIGNAL_COLUMNS = [
    "NN Entropy",
    "Prefix Support",
    "Prefix Similarity",
   # "NN reliability",
    "BK Entropy",
    "Prefix Alignment",
   # "BK reliability",
]
DISPLAY_SIGNAL_NAMES = {
    "NN Entropy": r"$1-U_{NN}$",
    "Prefix Support": r"$Support$",
    "Prefix Similarity": r"$Similarity$",
    "NN reliability": r"$R_{NN}$",
    "BK Entropy": r"$1-U_{BK}$",
    "Prefix Alignment": r"$C_{BK}$",
    "BK reliability": r"$R_{BK}$",
}

# Keep the same order as plot_adapted_weight.py for the combined figure.
SELECTED_DATASETS = (
    "BPI2013_CP",
    "BPI2013_In",
    "DomesticDeclarations",
    "log",
    "helpdesk",
    "InternationalDeclarations",
)
DATASET_NAMES = {
    "BPI2013_CP": "BPIC2013-CP",
    "BPI2013_In": "BPIC2013-I",
    "DomesticDeclarations": "Domestic",
    "log": "Dreyers",
    "helpdesk": "Helpdesk",
    "InternationalDeclarations": "International",
}
DATASET_ORDER = tuple(DATASET_NAMES[name] for name in SELECTED_DATASETS)

WEIGHT_COLUMN = "Weight"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "docs/source/CAiSE_data/output/one_hot"
OUTPUT_ROOT = PROJECT_ROOT / "docs/source/CAiSE_data/adaptive_signal_plots"
LOG_ROOT = PROJECT_ROOT / "docs/source/CAiSE_data/input/logs"
CORRECTED_RESULTS_FOLDER = "adaptive_w_results_final"
FOLDS_BY_DATASET = {
    "BPI2013_CP": (5, 0, 3, 1),
    "BPI2013_In": (5, 0, 3, 2),
    "InternationalDeclarations": (5, 0, 2, 3),
    "DomesticDeclarations": (5, 4, 1, 3),
    "helpdesk": (0, 1, 2, 3),
    "log": (1, 3, 4, 5),
}


def parse_sequence(value):
    """Convert a CSV cell containing a scalar or list into floats."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, np.ndarray)):
        values = value
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    else:
        if pd.isna(value):
            return []

        text = str(value).strip().replace("|", "")
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                return [float(text)]
            except ValueError:
                return []

        values = parsed if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]

    result = []
    for item in values:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result


def parse_activity_sequence(value):
    """Split a human-readable activity sequence from a result row."""
    if value is None or pd.isna(value):
        return []
    return [activity for activity in str(value).split(">>") if activity]


def make_log_data(log_name):
    """Load the training split used to recompute prefix support."""
    return LogData(
        log_path=LOG_ROOT,
        log_name=log_name,
        train_log=f"{log_name}_train.xes",
        feedback_log=f"{log_name}_feedback.xes",
        test_log=f"{log_name}_test.xes",
    )


def recompute_support_history(prediction, log_data, activity_to_code):
    """Compute support before each predicted suffix activity."""
    trace_prefix = parse_activity_sequence(prediction["Trace Prefix Act"])
    predicted_acts = parse_activity_sequence(prediction["Predicted Acts"])
    weights = parse_sequence(prediction[WEIGHT_COLUMN])
    support_history = []

    for suffix_position in range(1, len(weights) + 1):
        prefix_activities = trace_prefix + predicted_acts[:suffix_position - 1]
        try:
            encoded_prefix = "".join(
                activity_to_code[activity] for activity in prefix_activities
            )
        except KeyError:
            encoded_prefix = None

        support_history.append(
            0.0
            if encoded_prefix is None
            else float(log_data.training_prefix_support(encoded_prefix))
        )

    return support_history


def safe_spearman(x, y):
    data = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(data) < 3:
        return np.nan
    if data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan

    rho, _ = spearmanr(data["x"], data["y"])
    return float(rho)


def compute_step_signal_correlations(df, available_signals):
    """Compute signal/weight Spearman correlations at aligned steps."""
    rows = []
    for signal in available_signals:
        signal_values = []
        weight_values = []

        for _, record in df.iterrows():
            weights = parse_sequence(record[WEIGHT_COLUMN])
            values = parse_sequence(record[signal])
            if signal in {"NN Entropy", "BK Entropy"}:
                values = [1.0 - value for value in values]

            length = min(len(weights), len(values))
            signal_values.extend(values[:length])
            weight_values.extend(weights[:length])

        rows.append(
            {
                "Signal": DISPLAY_SIGNAL_NAMES.get(signal, signal),
                "rho_signal_weight": safe_spearman(signal_values, weight_values),
            }
        )

    return pd.DataFrame(rows)


def draw_horizontal_correlations(summary, value_col, xlabel, axis):
    data = summary[["Signal", value_col]].dropna().copy()
    if data.empty:
        return False

    signal_order = [DISPLAY_SIGNAL_NAMES.get(signal, signal) for signal in SIGNAL_COLUMNS]
    data["Signal"] = pd.Categorical(
        data["Signal"], categories=signal_order, ordered=True
    )
    data = data.sort_values("Signal")

    y = np.arange(len(data))
    axis.barh(y, data[value_col].values)
    axis.set_yticks(y)
    axis.set_yticklabels(data["Signal"].values, fontsize=TICK_LABEL_FONTSIZE)
    axis.axvline(0, linewidth=1)
    axis.set_xlim(-1.15, 1.15)
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    axis.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)

    for index, value in enumerate(data[value_col].values):
        offset = 0.03 if value >= 0 else -0.03
        axis.text(
            value + offset,
            index,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    return True


def plot_horizontal_correlations(summary, value_col, xlabel, axis):
    """Draw one dataset's correlations on an existing subplot axis."""
    return draw_horizontal_correlations(summary, value_col, xlabel, axis)


def prediction_paths(log_name):
    """Return the fold-specific paths used by plot_adapted_weight.py."""
    return [
        INPUT_ROOT
        / str(fold)
        / "results"
        / "CF"
        / f"{log_name}_adaptive_weight_linear_results.csv"
        for fold in FOLDS_BY_DATASET[log_name]
    ]


def corrected_prediction_path(prediction_path):
    """Place a corrected CSV in a sibling folder of the fold directory."""
    fold_directory = prediction_path.parents[2]
    return fold_directory / CORRECTED_RESULTS_FOLDER / prediction_path.name


def correct_prediction_file(prediction_path, log_data, activity_to_code):
    """Recompute Support and save one corrected prediction CSV."""
    frame = pd.read_csv(prediction_path, quotechar="|")
    required_columns = {WEIGHT_COLUMN, "Trace Prefix Act", "Predicted Acts"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise KeyError(
            f"Required column(s) missing in {prediction_path}: {missing_columns}"
        )

    frame["Prefix Support"] = frame.apply(
        lambda prediction: recompute_support_history(
            prediction,
            log_data,
            activity_to_code,
        ),
        axis=1,
    )

    output_path = corrected_prediction_path(prediction_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output_path,
        index=False,
        quotechar="|",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"Saved corrected CSV: {output_path}")
    return frame


def analyze_dataset(log_name):
    input_files = prediction_paths(log_name)
    missing_files = [path for path in input_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing result file(s) for {log_name}: {missing_files}"
        )

    '''log_data = make_log_data(log_name)
    activity_to_code = {
        activity: code
        for code, activity in log_data.act_enc_mapping.items()
    }'''
    frames = [pd.read_csv(corrected_prediction_path(prediction_path), quotechar="|")
              for prediction_path in input_files
    ]
    '''frames = [
        correct_prediction_file(prediction_path, log_data, activity_to_code)
        for prediction_path in input_files
    ]'''
    df = pd.concat(frames, ignore_index=True)

    available_signals = [
        signal for signal in SIGNAL_COLUMNS if signal in df.columns
    ]
    if not available_signals:
        raise KeyError(
            "None of the expected adaptive-signal columns were found. "
            f"Expected one of: {SIGNAL_COLUMNS}"
        )

    print(
        f"{log_name}: {len(df)} rows from {len(input_files)} files; "
        f"signals={available_signals}"
    )
    return compute_step_signal_correlations(df, available_signals)


def plot_all_logs(summaries, output_dir):
    """Draw all dataset correlation plots on one 2x3 PDF figure."""
    dataset_names = [
        name for name in DATASET_ORDER if name in summaries
    ]

    columns = 3
    rows = int(np.ceil(len(dataset_names) / columns))
    figure, axes = plt.subplots(
        rows,
        ncols=columns,
        figsize=(18, 10),
        sharey=True,
        layout="constrained",
        gridspec_kw={"hspace": 0.05}
    )

    for dataset_name, axis in zip(dataset_names, axes.flat):
        summary = summaries[dataset_name]
        plot_horizontal_correlations(
            summary,
            value_col="rho_signal_weight",
            xlabel="Spearman ρ with adaptive weight",
            axis=axis,
        )
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            dataset_name,
            fontweight="bold",
            loc="center",
        )

    for axis in axes.flat[len(dataset_names):]:
        axis.set_visible(False)

    plt.show()
    '''figure.subplots_adjust(hspace=0, wspace=0.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / PLOT_FILENAME
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)'''


def main():
    summaries = {
        DATASET_NAMES[log_name]: analyze_dataset(log_name)
        for log_name in SELECTED_DATASETS
    }
    plot_all_logs(summaries, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
