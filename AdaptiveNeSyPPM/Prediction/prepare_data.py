"""
This script prepares data in the format for the testing
algorithms to run
The script is expanded to the resource attribute
"""

from __future__ import division
import itertools
from functools import lru_cache
from typing import Dict
import operator
import numpy as np
import pm4py
import pandas as pd
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessMiningTasks.ConformanceChecking.MPDeclareResultsBrowser import MPDeclareResultsBrowser
from Declare4Py.Utils.Declare.TraceStates import TraceState
from rapidfuzz import process
from rapidfuzz.distance import DamerauLevenshtein

from AdaptiveNeSyPPM.Data_preprocessing.utils import Encodings, BK_type
from AdaptiveNeSyPPM.Prediction.Checkers import TraceDeclareAnalyzer
from AdaptiveNeSyPPM.Prediction.create_event_log import convert_to_log
from enum import Enum

class ConstraintChecker(Enum):
    SATISFIED = 1
    POSSIBLY_SATISFIED = 0.66
    POSSIBLY_VIOLATED = 0.33
    VIOLATED = 0

ADAPTIVE_SIGNAL_NAMES = ("NN Entropy", "BK Entropy", "Prefix Similarity", "Prefix Support","Prefix Alignment","Contradiction","NN reliability","BK reliability")

def get_binary_pn_fitness(bk_model, trace: pd.DataFrame, log_data, completed_trace:bool=True) -> int:
    # Decode traces for feeding them to the Petri net
    dec_log = trace.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    dec_log[log_data.timestamp_key] = pd.to_datetime(log_data.log[log_data.timestamp_key], unit='s')
    net = bk_model["net"]
    initial_marking = bk_model["initial_marking"]
    final_marking = bk_model["final_marking"] if completed_trace else type(bk_model["final_marking"])() #empty final marking if it is not completed trace
    trace_acts = list(dec_log[log_data.act_name_key]) # list of prefix activities

    replay_result = pm4py.conformance_diagnostics_token_based_replay(dec_log, net, initial_marking, final_marking,
                                                                     activity_key=log_data.act_name_key,
                                                                     case_id_key=log_data.case_name_key,
                                                                     timestamp_key=log_data.timestamp_key)[0]
    if completed_trace:
        return 1.0 if replay_result["trace_is_fit"] else 0.0
    else:
        return (1.0 if (not replay_result["transitions_with_problems"]  # no missing transitions
                             and trace_acts == [t.label for t in replay_result["activated_transitions"] if t.label is not None]) #all activities in the prefix are activated
                     else 0.0)


# === Helper functions ===
def encode(crop_trace: pd.DataFrame, log_data, encoder:Encodings, maxlen: int, char_indices: Dict[str, int],
                      char_indices_group: Dict[str, int], resource: bool) -> np.ndarray:
    """
    encoding of an ongoing trace (control-flow + resource)
    """
    chars = list(char_indices.keys())
    if resource:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        sentence_group = ''.join(crop_trace[log_data.res_name_key].tolist())
        chars_group = list(char_indices_group.keys())
        if encoder== Encodings.One_hot:
            num_features = len(chars) + len(chars_group)
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if char in chars:
                    x[0, t + leftpad, char_indices[char] - 1] = 1
                if t < len(sentence_group):
                    if sentence_group[t] in chars_group:
                        x[0, t + leftpad, len(char_indices) + char_indices_group[sentence_group[t]] - 1] = 1
        elif encoder== Encodings.Multi_encoders:
            num_features = maxlen
            x_a = np.zeros((1, num_features), dtype=np.float32)
            x_g = np.zeros((1, num_features), dtype=np.float32)
            for t, char in enumerate(sentence):
                x_a[0, t] = char_indices[char]
                x_g[0, t]= char_indices_group[sentence_group[t]]
            x = {
                'x_act': x_a,
                'x_group' : x_g
            }
        else:
            if encoder == Encodings.Shrunk_Index_based:
                result_list = [x + y for x, y in itertools.product(chars, chars_group)]
                target_to_int = dict((c, i + 1) for i, c in enumerate(result_list))
                num_features = maxlen
                x = np.zeros((1, num_features), dtype=np.float32)
                for t, char in enumerate(sentence):
                    x[0, t] = target_to_int[char + sentence_group[t]]
            else:
                num_features = maxlen * 2
                counter_act = 0
                counter_res = 1
                x = np.zeros((1, num_features), dtype=np.float32)
                for t, char in enumerate(sentence):
                    x[0, counter_act] = char_indices[char]
                    if t < len(sentence_group):
                        x[0, counter_res] = char_indices_group[sentence_group[t]]
                    counter_act += 2
                    counter_res += 2
    else:
        sentence = ''.join(crop_trace[log_data.act_name_key].tolist())
        if encoder == Encodings.One_hot:
            num_features = len(chars)
            x = np.zeros((1, maxlen, num_features), dtype=np.float32)
            leftpad = maxlen - len(sentence)
            for t, char in enumerate(sentence):
                if char in chars:
                    x[0, t + leftpad, char_indices[char]-1] = 1
        else:
            num_features = maxlen
            x = np.zeros((1, num_features), dtype=np.float32)
            for t, char in enumerate(sentence):
                x[0, t] = char_indices[char]
    return x

@lru_cache(maxsize=None)
def prefix_train_similarity(training_prefixes:frozenset[str],prefix):
    if not training_prefixes:
        raise ValueError("Training prefixes cannot be empty.")

    prefix = prefix.rstrip("!")
    if prefix in training_prefixes:
        return 1.0
    match = process.extractOne(prefix,training_prefixes, scorer=DamerauLevenshtein.normalized_similarity,)
    return float(match[1])


def normalized_entropy(probabilities) -> float:
    """Return normalized entropy in ``[0, 1]``."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    num_candidates = probabilities.size

    positive_probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(positive_probabilities * np.log(positive_probabilities))
    return float(np.clip(entropy / np.log(num_candidates), 0.0, 1.0))


def aggregate_reliability(factors, neutral=0.5):
    if not factors:
        return neutral
    return float(np.prod(factors))


def prefix_tbr_score(prefix_trace, bk_model, log_data, completed=False):
    """Return a continuous token-replay score for a trace/prefix."""
    decoded_prefix = prefix_trace.replace(to_replace={
        log_data.act_name_key: log_data.act_enc_mapping,
    })

    net = bk_model["net"]
    initial_marking = bk_model["initial_marking"]
    prefix_final_marking = bk_model["final_marking"] if completed else type(bk_model["final_marking"])()
    replay_result = pm4py.conformance_diagnostics_token_based_replay(decoded_prefix, net, initial_marking, prefix_final_marking,
                                                     activity_key=log_data.act_name_key,
                                                     case_id_key=log_data.case_name_key,
                                                     timestamp_key=log_data.timestamp_key)[0]

    if completed:
        return replay_result['trace_fitness']
    else:
        missing = replay_result["missing_tokens"]
        consumed = replay_result["consumed_tokens"]
        token_fitness = 1.0 - (missing / consumed) if consumed > 0 else float(missing == 0)
        # % prefix activities successfully replayed
        replayed_acts = [t.label for t in replay_result["activated_transitions"] if t.label is not None]
        r_replay = min(len(replayed_acts) / len(decoded_prefix), 1.0) if len(decoded_prefix) else 1.0
        return token_fitness  * r_replay


def compute_score(prefix_trace, child_node, log_data, bk_model, adaptive_weight, weight, resource=False, feasibility_pruning=False,
                  termination_pruning=False, ProbAgg_method=None, cache_fitness=None, binary_compliance=False,
                  nn_entropy=False, bk_entropy=False, prefix_support=False, prefix_sim=False, bk_conformance=False):
    prediction = None
    BK_res = None
    adaptive_signals = None
    prefix_align = None
    # Compute NN score.
    if weight != 1.0:
        prediction= child_node.nn_probability

    # Compute BK score.
    if  weight is None or weight > 0.0:
        if resource:
            if bk_model and bk_model["type"] == BK_type.Declare:
                BK_res = np.zeros((len(log_data.target_int_to_act), len(log_data.target_int_to_res)), dtype=np.float32)
                for act_pred_idx, res_pred_idx in np.ndindex(BK_res.shape):
                    temp_next_act = log_data.target_int_to_act[act_pred_idx + 1]
                    temp_next_res = log_data.target_int_to_res[res_pred_idx + 1]
                    completed = True if temp_next_act == "!" or temp_next_res == "!" else False
                    BK_res[act_pred_idx, res_pred_idx] = declare_compliance_checking(log_data, bk_model["model"],
                                                                                     prefix_trace, completed, temp_next_act,
                                                                                     temp_next_res, resource)
        else:
            if bk_model and bk_model["type"] == BK_type.Declare:
                BK_res = np.zeros(len(log_data.target_int_to_act), dtype=np.float32)
                prefix_align = declare_compliance_checking(log_data, bk_model["model"], prefix_trace)
                for idx, temp_next_act in log_data.target_int_to_act.items():
                    completed = True if temp_next_act == "!" else False
                    BK_res[idx - 1] = declare_compliance_checking(log_data, bk_model["model"], prefix_trace, completed, temp_next_act)
                std = BK_res.std()
                if std == 0:
                    BK_res = np.full(BK_res.shape, 1.0 / BK_res.size)
                else:
                    logits = (BK_res - BK_res.mean()) / std
                    logits -= logits.max()
                    BK_res = np.exp(logits)
                    BK_res = BK_res / BK_res.sum()
            if bk_model and bk_model["type"] == BK_type.ProbDeclare:
                act_prefix = child_node.cropped_line
                prefix_act = [log_data.act_enc_mapping[index] for index in act_prefix]
                results = bk_model["model"].processPrefix(prefix_act, ProbAgg_method)
                BK_res = np.zeros(len(log_data.target_int_to_act), dtype=np.float32)
                target_acts = [log_data.act_enc_mapping[target_act] if target_act != "!" else "False"
                    for target_act in log_data.target_int_to_act.values()]
                for event, event_score in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
                    if event is False:  # End symbol
                        BK_res[target_acts.index("False")] = round(event_score, 3)
                    elif event is not True:  # Activities present in the declare model
                        BK_res[target_acts.index(event)] = round(event_score, 3)
                    else:  # if event is True: Activities not present in the declare model
                        act_indices = [i for i, act in enumerate(target_acts) if act not in results.keys() and act != "False"]
                        for idx in act_indices:
                            BK_res[idx] = round(event_score, 3)
            if bk_model and bk_model["type"] == BK_type.Procedural:
                prefix_align = prefix_tbr_score(prefix_trace, bk_model, log_data)
                BK_res = np.asarray(fitness_checking(log_data, log_data.target_int_to_act, cache_fitness, bk_model, prefix_trace, binary_compliance=binary_compliance), dtype=np.float32)
    # Final score
    if prediction is None and BK_res is None:
        raise ValueError("Cannot compute score without NN prediction or BK result.")
    if prediction is None:
        score = np.log(BK_res)
    elif BK_res is None:
        score = np.log(prediction)
    else:
        if adaptive_weight:
            nn_uncertainty = normalized_entropy(prediction)
            BK_uncertainty = normalized_entropy(BK_res)
            prefix_supp = log_data.training_prefix_support(child_node.cropped_line)
            prefix_similarity = prefix_train_similarity(log_data.training_act_prefixes, child_node.cropped_line)
            nn_factors = []
            if nn_entropy:
                nn_factors.append(1.0 - nn_uncertainty)
            familiarity = []
            if prefix_support:
                familiarity.append(prefix_supp)
            if prefix_sim :
                familiarity.append(prefix_similarity)
            if familiarity:
                nn_familiarity = sum(familiarity) / len(familiarity)
                nn_factors.append(nn_familiarity)

            bk_factors = []
            if bk_entropy:
                bk_factors.append(1.0 - BK_uncertainty)
            if bk_conformance and prefix_align is not None:
                bk_factors.append(prefix_align)

            a = 0
            b = 0
            if not bk_conformance and not bk_entropy:
                # No BK reliability signals: rely on neural reliability only.
                a = aggregate_reliability(nn_factors)
                weight = 1 - a
            elif not nn_entropy and not familiarity:
                # No neural reliability signals: rely on BK reliability only.
                b = aggregate_reliability(bk_factors)
                weight = b
            else:
                # Full adaptive weighting or partial ablation.
                a = aggregate_reliability(nn_factors)
                b = aggregate_reliability(bk_factors)
                weight = b / (a + b ) if a + b > 0 else 0.5
            adaptive_signals = {
                "NN Entropy": nn_uncertainty,
                "BK Entropy": BK_uncertainty,
                "Prefix Similarity": prefix_similarity,
                "Prefix Support": prefix_supp,
                "Prefix Alignment": prefix_align,
                "Contradiction": float(BK_res[np.argmax(prediction)] <= min(BK_res)),
                "NN reliability": a,
                "BK reliability": b
            }

        score = (np.log(prediction) * (1 - weight)) + (np.log(BK_res) * weight)
    return score, weight, adaptive_signals

def get_beam_size(self, NodePrediction, current_node, bk_model, adaptive_weight, weight, log_data, resource, beam_size, cache_fitness,
                  ProbAgg_method,binary_compliance=False, termination_pruning=False, feasibility_pruning=False,nn_entropy=False, bk_entropy=False,prefix_support=False, prefix_sim=False, bk_conformance=False):
    prefix_trace = current_node.cropped_trace
    score, step_weight, step_signals = compute_score(prefix_trace, current_node, log_data, bk_model, adaptive_weight, weight, resource=resource,
                                                        ProbAgg_method=ProbAgg_method, feasibility_pruning=feasibility_pruning,
                                                        termination_pruning=termination_pruning, cache_fitness=cache_fitness,binary_compliance=binary_compliance,
                                                        nn_entropy=nn_entropy, bk_entropy=bk_entropy, prefix_support=prefix_support,
                                                        prefix_sim=prefix_sim,bk_conformance=bk_conformance)

    if feasibility_pruning or termination_pruning:
        valid_indices = np.flatnonzero(np.isfinite(score))
        valid_indices = valid_indices[np.argsort(score.flat[valid_indices])[::-1]]
    else:
        valid_indices = np.argsort(score, axis=None)[::-1]

    for candidate_idx in valid_indices[:beam_size or len(valid_indices)]:
        if resource:
            act_pred_idx, res_pred_idx = np.unravel_index(candidate_idx, score.shape)
            temp_next_act = log_data.target_int_to_act[act_pred_idx + 1]
            temp_next_res = log_data.target_int_to_res[res_pred_idx + 1]
            probability_this = score[act_pred_idx, res_pred_idx]
        else:
            pred_idx = candidate_idx
            temp_next_act = log_data.target_int_to_act[pred_idx + 1]
            temp_next_res = None
            probability_this = score[pred_idx]

        predicted_row = prefix_trace.tail(1).copy()
        predicted_row.loc[:, log_data.act_name_key] = temp_next_act
        if resource: predicted_row.loc[:, log_data.res_name_key] = temp_next_res
        temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)

        probability_of = (current_node.probability_of + probability_this)

        weight_history = list(current_node.adaptive_weight)
        if adaptive_weight:
            weight_history.append(round(step_weight,3))
        signal_history = {
            name: list(current_node.adaptive_signals[name])
            for name in ADAPTIVE_SIGNAL_NAMES
        }
        if adaptive_weight:
            for name in ADAPTIVE_SIGNAL_NAMES:
                signal_history[name].append(round(step_signals[name], 6) )
        temp = NodePrediction(
            temp_cropped_trace_next, probability_of,
            current_node.prefix_deviation_rate, weight_history, signal_history
        )
        self.put(temp)
    return self

def declare_compliance_checking(log_data, bk_model, prefix_trace, completed=False,temp_next_act= None, temp_next_res= None, resource=False, hard_constraints=False):
    if completed :
        temp_cropped_trace_next = prefix_trace.copy()
    else:
        if temp_next_act is None:
            temp_cropped_trace_next = prefix_trace.copy()
        else:
            predicted_row = prefix_trace.tail(1).copy()
            predicted_row.loc[:, log_data.act_name_key] = temp_next_act
            if resource:
                predicted_row.loc[:, log_data.res_name_key] = temp_next_res
            temp_cropped_trace_next = pd.concat([prefix_trace, predicted_row], axis=0)
    temp_cropped_trace_next[log_data.act_name_key] = temp_cropped_trace_next[log_data.act_name_key].apply(
        lambda x: x.replace(x, log_data.act_enc_mapping[x]))
    if resource:
        temp_cropped_trace_next[log_data.res_name_key] = temp_cropped_trace_next[log_data.res_name_key].apply(
        lambda x: x.replace(str(x), log_data.res_enc_mapping[x] if x != "!" else ""))
    log = convert_to_log(temp_cropped_trace_next, log_data.case_name_key, log_data.act_name_key)
    d_log = D4PyEventLog()
    d_log.log = log
    d_log.log_length = len(d_log.log)
    d_log.timestamp_key = log_data.timestamp_key
    d_log.activity_key = log_data.act_name_key
    basic_checker = TraceDeclareAnalyzer(log=d_log, declare_model=bk_model,
                                         consider_vacuity=False, completed=completed)
    conf_check_res: MPDeclareResultsBrowser = basic_checker.run()
    state = conf_check_res.get_metric(metric="state", trace_id=0)
    if hard_constraints:
        if 0 in state:
            return 1e-20  # violated constraint found
        else:
            results = []
            for result in conf_check_res.model_check_res[0]:
                if result.state == TraceState.POSSIBLY_SATISFIED.value:
                    results.append(ConstraintChecker.POSSIBLY_SATISFIED.value)
                elif result.state == TraceState.SATISFIED.value:
                    results.append(ConstraintChecker.SATISFIED.value)
                elif result.state == TraceState.POSSIBLY_VIOLATED.value:
                    results.append(ConstraintChecker.POSSIBLY_VIOLATED.value)
                else:
                    results.append(ConstraintChecker.VIOLATED.value)
            return np.mean(results)
    else:
        return np.mean(state)


def fitness_checking(log_data, target_indices_char, cache_fitness, bk_model, prefix_trace, binary_compliance=False):
    fitness_scores = []
    for candidate_index in range(1, len(target_indices_char) + 1):
        temp_next_act = target_indices_char[candidate_index]
        temp_cropped_line_next = ''.join(prefix_trace[log_data.act_name_key].tolist()+[temp_next_act])

        check_cache = cache_fitness.get(temp_cropped_line_next)
        if check_cache is None:
            if temp_next_act == "!":
                extended_trace = prefix_trace.copy()
                if binary_compliance:
                    candidate_fitness_score = get_binary_pn_fitness(bk_model, extended_trace, log_data)
                else:
                    candidate_fitness_score = prefix_tbr_score(extended_trace, bk_model, log_data, completed=True)
            else:
                predicted_row = prefix_trace.tail(1).copy()
                predicted_row.loc[:, log_data.act_name_key] = temp_next_act
                extended_trace = pd.concat([prefix_trace, predicted_row])
                if binary_compliance:
                    candidate_fitness_score = get_binary_pn_fitness(bk_model, extended_trace, log_data, completed_trace=False)
                else:
                    candidate_fitness_score = prefix_tbr_score(extended_trace, bk_model, log_data)
            cache_fitness.add(temp_cropped_line_next, candidate_fitness_score)
        else:
            candidate_fitness_score = check_cache
        fitness_scores.append(candidate_fitness_score)
    fitness = np.asarray(fitness_scores, dtype=float)
    std = fitness.std()
    if std == 0:
        return np.full(fitness.shape, 1.0 / fitness.size)
    else:
        logits = (fitness - fitness.mean()) / std
        logits -= logits.max()
        fitness = np.exp(logits)
        fitness = fitness / fitness.sum()
    return  fitness
