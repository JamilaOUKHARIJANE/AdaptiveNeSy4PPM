from pathlib import Path

from AdaptiveNeSyPPM.Data_preprocessing.log_utils import LogData
from AdaptiveNeSyPPM.Data_preprocessing.utils import extract_last_model_checkpoint, Encodings, BK_type
from AdaptiveNeSyPPM.Prediction import inference_algorithm
from AdaptiveNeSyPPM.ProbDeclmonitor.probDeclPredictor import AggregationMethod


def predict_evaluate(log_data: LogData,max_trace_len, models_folder:Path,  output_filename:str,encoder=Encodings.One_hot,
                     evaluation_trace_ids=None, pm_model=None,
                     beam_size:int=3, ProbAgg_method: AggregationMethod=None,binary_compliance=False,
                     adaptive_weight=False, weight: float=0.0, resource: bool=False, bk_end:bool=False, prefix_size:int=0,
                     termination_pruning=False, feasibility_pruning=False,
                     nn_entropy=False, pm_entropy=False,prefix_support=False, prefix_sim=False, pm_conformance=False
                     ):
    if pm_model is not None and bk_end and pm_model["type"] == BK_type.Declare: pm_model["type"] = BK_type.Declare_END
    elif pm_model is not None and bk_end and pm_model["type"] == BK_type.Procedural: pm_model["type"] = BK_type.Procedural_END
    elif pm_model is not None and bk_end and pm_model["type"] == BK_type.SDFA:
        pm_model["type"] = BK_type.SDFA_END
    if adaptive_weight:
        weight = None

    evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(log_data.evaluation_trace_ids)]
    if evaluation_trace_ids is not None:
        evaluation_traces = log_data.log[log_data.log[log_data.case_name_key].isin(evaluation_trace_ids)]
    prediction_type = 'CF' + 'R' * resource

    folder_path = models_folder /'SDFA_results' / prediction_type
    if not Path.exists(folder_path):
        Path.mkdir(folder_path, parents=True)
    print(f"{'Activity' + ' & Resource'*resource} Prediction ...")
    output_filename = folder_path / output_filename
    model_filename = extract_last_model_checkpoint(
        log_data.log_name, 'CF' + 'R' * resource, models_folder
    )
    inference_algorithm.run_experiments(log_data, max_trace_len, evaluation_traces, encoder,model_filename, output_filename, pm_model,
                                   ProbAgg_method, resource,adaptive_weight, weight, bk_end, beam_size, prefix_size,termination_pruning, feasibility_pruning,
                                        binary_compliance, nn_entropy, pm_entropy,prefix_support, prefix_sim, pm_conformance)
