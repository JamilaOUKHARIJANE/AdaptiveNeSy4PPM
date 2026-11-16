import argparse
import os
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pm4py
from AdaptiveNeSyPPM.Data_preprocessing.data_preprocessing import end_to_end_process
from AdaptiveNeSyPPM.Data_preprocessing.utils import NN_model
from AdaptiveNeSyPPM.Evaluation import discover_petri_net
from AdaptiveNeSyPPM.Prediction import predict_suffix
from AdaptiveNeSyPPM.Training.train_model import train
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


def _trace_variants(path: Path) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    frame = pm4py.read_xes(str(path))
    if not isinstance(frame, pd.DataFrame):
        frame = pm4py.convert_to_dataframe(frame)
    case_key = "case:concept:name"
    activity_key = "concept:name"
    if case_key not in frame.columns or activity_key not in frame.columns:
        raise ValueError(f"{path} must contain {case_key!r} and {activity_key!r}")
    sort_columns = [case_key]
    if "time:timestamp" in frame.columns:
        sort_columns.append("time:timestamp")
    frame = frame.sort_values(sort_columns, kind="mergesort")
    variants = {
        str(case_id): tuple(str(activity) for activity in group[activity_key])
        for case_id, group in frame.groupby(case_key, sort=False)
    }
    return frame, variants

def _create_novel_feedback_log(
    train_path: Path,
    feedback_path: Path,
    output_path: Path,
    min_train_support: float=0.1,
) -> Path:
    """Keep learning_w variants unseen/rare in train for drift-focused BK.
    """
    _, train_variants = _trace_variants(train_path)
    feedback_frame, feedback_variants = _trace_variants(feedback_path)
    feedback_support = Counter(feedback_variants.values())
    train_support = Counter(train_variants.values())
    threshold = min_train_support * max(len(train_variants), 1)
    selected_cases = {
        case_id
        for case_id, variant in feedback_variants.items()
        if (variant not in train_support or train_support[variant] < threshold) and feedback_support[variant] >= 3
    }

    filtered = feedback_frame[
        feedback_frame["case:concept:name"].astype(str).isin(selected_cases)
    ].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xes_exporter.apply(filtered, str(output_path))
    print(
        f"Novel learning_w BK source: {len(selected_cases)}/"
        f"{len(feedback_variants)} cases, "
        f"{len(set(feedback_variants[case] for case in selected_cases))}/"
        f"{len(set(feedback_variants.values()))} variants"
    )
    return output_path

model_arch = NN_model.Transformer


parser = argparse.ArgumentParser()
parser.add_argument('--log', default="helpdesk", help='input log')
parser.add_argument('--method', default="adaptive", help='input log')
args = parser.parse_args()
loglist = [args.log.strip()]

bsize = 3
log_path = Path.cwd()/'data'/'input'/'logs'

fixed = False
neural = False
symbolic = False
adaptive = False
learning_w =False
learned = False
method = args.method.strip()
if str(method).startswith("adaptive"):
    adaptive = True
elif method == "neural":
    neural = True
elif method == "symbolic":
    symbolic = True
elif method == "fixed_w":
    fixed = True
elif method == "learning_w":
    learning_w = True
elif method == "learned_w":
    learned = True
else:
    fixed = False
    neural = False
    symbolic = False
    adaptive = False
    learning_w =False
    learned = False

for log_name in loglist:
    log_data, x, y_a, y_g = end_to_end_process(log_path=log_path, log_name=log_name,
                                                   train_log=f"{log_name}_train.xes",
                                                   feedback_log=f"{log_name}_feedback.xes",
                                                   test_log=f"{log_name}_test.xes")
    ############## Max length in both neural and symbolic partitions######
    neural_log = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.training_trace_ids)]
    neural_traces_sizes = list(neural_log.value_counts(subset=[log_data.case_name_key], sort=False))
    symbolic_log = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.feedback_trace_ids)]
    symbolic_traces_sizes = list(symbolic_log.value_counts(subset=[log_data.case_name_key], sort=False))
    max_trace_len = max(neural_traces_sizes + symbolic_traces_sizes)
    ##### test log prefixes lengths
    test_log = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    test_trace_sizes = list(test_log.value_counts(subset=[log_data.case_name_key], sort=False))
    max_prefix_len = max(test_trace_sizes)

    #### Petri net discovery
    if not neural:
        SDFA_path = log_path.parent / "petrinets"
        os.makedirs(SDFA_path, exist_ok=True)
        feedback_path = Path(log_path) / f"{log_name}_feedback.xes"
        train_path = Path(log_path) / f"{log_name}_train.xes"
        novel_path = (Path(log_path) / f"{log_name}_feedback_filtered.xes")
        novel_path = _create_novel_feedback_log(train_path, feedback_path, novel_path)
        pm_model = discover_petri_net(novel_path)
        pnml_path = SDFA_path / f"{log_name}.pnml"
        pm4py.write_pnml(
            pm_model["net"],
            pm_model["initial_marking"],
            pm_model["final_marking"],
            str(pnml_path),
        )

    for fold in range(4):
        output_folder = Path.cwd()/'data'/ 'output'/str(fold)
        train(log_data, model_arch=model_arch, output_folder=output_folder, X=x, y_a=y_a)
        ## learning weight
        if learning_w:
            case_ids = log_data.feedback_trace_ids
            prediction_filename = f"{log_name}_feedback_results.csv"
            for w in np.arange(0, 1.1, 0.1):
                for prefix in range(1, max_prefix_len):
                    predict_suffix.predict_evaluate(log_data,  max_trace_len, models_folder=output_folder, 
                                                    output_filename=prediction_filename, pm_model=pm_model,
                                                    evaluation_trace_ids=case_ids,
                                                    prefix_size=prefix,
                                                    beam_size=bsize, weight=w
                                                    )

        ### prediction
        if learned:
            w = 0.6 # here best learned weight
            prediction_filename = f"{log_name}_learned_w_results.csv"
            for prefix in range(1, max_prefix_len):
                predict_suffix.predict_evaluate(log_data,  max_trace_len, models_folder=output_folder, 
                                                output_filename=prediction_filename, pm_model=pm_model,
                                                prefix_size=prefix,
                                                beam_size=bsize, weight=w
                                                )
        if fixed:
            prediction_filename = f"{log_name}_fixed_w_results.csv"
            for prefix in range(1, max_prefix_len):
                predict_suffix.predict_evaluate(log_data,  max_trace_len, models_folder=output_folder, 
                                                output_filename=prediction_filename, pm_model=pm_model,
                                                prefix_size=prefix,
                                                beam_size=bsize, weight=0.5
                                                )
        if symbolic:
            prediction_filename = f"{log_name}_symbolic_results.csv"
            predict_suffix.predict_evaluate(log_data,  max_trace_len, models_folder=output_folder, 
                                            output_filename=prediction_filename, pm_model=pm_model,
                                            beam_size=bsize, weight=1
                                            )
        if adaptive:
            prediction_filename = f"{log_name}_{method}_results.csv"
            nn_entropy = True
            pm_entropy = True
            prefix_support = True
            prefix_sim = True
            pm_conformance = True
            if method == "adaptive_w_dls-nnEntropy":
                nn_entropy = False
            if method == "adaptive_w_dls-prefix_supp":
                prefix_support = False
            if method == "adaptive_w_dls-prefix_sim":
                prefix_sim = False
            if method == "adaptive_w_dls-pm_conformance":
                pm_conformance = False
            if method == "adaptive_w_dls-bkEntropy":
                pm_entropy = False
            for prefix in range(1, max_prefix_len):
                predict_suffix.predict_evaluate(log_data,  max_trace_len, models_folder=output_folder, 
                                                output_filename=prediction_filename, pm_model=pm_model,
                                                prefix_size=prefix,
                                                beam_size=bsize, adaptive_weight=True, nn_entropy=nn_entropy,
                                                pm_entropy=pm_entropy,
                                                prefix_support=prefix_support, prefix_sim=prefix_sim,
                                                pm_conformance=pm_conformance
                                                )
        if neural:
            prediction_filename = f"{log_name}_neural_results.csv"
            for prefix in range(1, max_prefix_len):
                predict_suffix.predict_evaluate(log_data,  max_trace_len, models_folder=output_folder, 
                                                output_filename=prediction_filename, pm_model=None, prefix_size=prefix,
                                                beam_size=bsize, weight=0
                                                )

