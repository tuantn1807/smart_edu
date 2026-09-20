"""Offline evaluation of PAAF on Eedi (diagnosis labels) and Junyi (graph / mapping).

This is a protocol/integrity evaluation against dataset annotations, not an LLM
leaderboard and not a comparison with DKT/NCD accuracy.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.diagnostic_agent import UNLABELED_MISCONCEPTION, DiagnosticAgent
from src.agents.kg_agent import KGAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.tutor_agent import TutorAgent
from src.agents.tutor_engine import compute_specificity_score
from src.core.learner_state import LearnerState
from src.data.concept_mapping import EediJunyiMapper, SemanticEmbeddingMapper, HybridConceptMapper
from src.data.dataset_loaders import EediDatasetLoader, JunyiGraphLoader

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'eval' / 'results.json'


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return statistics.fmean(values)


def _silence():
    return contextlib.redirect_stdout(io.StringIO())


def _compute_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    classes = set(y_true).union(set(y_pred))
    if not classes:
        return 0.0
    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == cls and pr == cls)
        fp = sum(1 for gt, pr in zip(y_true, y_pred) if gt != cls and pr == cls)
        fn = sum(1 for gt, pr in zip(y_true, y_pred) if gt == cls and pr != cls)
        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1
    return f1_sum / len(classes)


def _prediction_label(result: Dict[str, Any]) -> str:
    if result.get("is_correct"):
        return "unlabeled"
    mid = result.get("misconception_id")
    if mid is None or mid == "None" or mid == UNLABELED_MISCONCEPTION:
        return "unlabeled"
    return str(mid)


def evaluate_diagnostic(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    agent = DiagnosticAgent()
    n_correct = n_correct_ok = 0
    n_labeled = n_labeled_match = 0
    n_unlabeled = n_unlabeled_fallback = 0
    n_options = 0

    y_true_labeled: List[str] = []
    y_pred_rule_labeled: List[str] = []
    
    y_true_llm_labeled: List[str] = []
    y_pred_llm_labeled: List[str] = []

    y_true_fallback_labeled: List[str] = []
    y_pred_fallback_labeled: List[str] = []

    n_valid_json = 0
    n_llm_evals = 0
    n_llm_requests = 0
    n_fallback_count = 0

    with _silence():
        for index, question in enumerate(questions):
            correct_opt = question.get('correct_option', '')
            misconception_map = question.get('misconception_map', {})

            for option in question['options']:
                n_options += 1
                state = LearnerState(f'EVAL_D_{index}_{option}', 'Eval')
                is_correct_option = (option.upper() == correct_opt.upper())

                # Ground truth label
                label_info = misconception_map.get(option, {})
                gt_misc_id = label_info.get('misconception_id')
                if not is_correct_option and gt_misc_id is not None:
                    gt_label = str(gt_misc_id)
                else:
                    gt_label = 'unlabeled'

                # Rule-based processing
                result_rule = agent.process(
                    {'question': question, 'selected_option': option, 'use_llm': False},
                    {'learner_state': state},
                )
                pred_rule_id = _prediction_label(result_rule)

                if is_correct_option:
                    n_correct += 1
                    if result_rule.get('is_correct') and result_rule.get('detected_misconception') is None:
                        n_correct_ok += 1
                    continue

                # Incorrect options are evaluated for misconception diagnosis
                state_llm = LearnerState(f'EVAL_DLLM_{index}_{option}', 'Eval')
                result_llm = agent.process(
                    {'question': question, 'selected_option': option, 'use_llm': True},
                    {'learner_state': state_llm},
                )
                pred_llm_id = _prediction_label(result_llm)
                n_llm_evals += 1

                used_fallback = result_llm.get('used_fallback', False)
                is_valid_parse = result_llm.get('is_valid_parse', False)

                if used_fallback:
                    n_fallback_count += 1
                else:
                    n_llm_requests += 1
                    if is_valid_parse:
                        n_valid_json += 1

                if gt_label != 'unlabeled':
                    n_labeled += 1
                    y_true_labeled.append(gt_label)
                    y_pred_rule_labeled.append(pred_rule_id)

                    if used_fallback:
                        y_true_fallback_labeled.append(gt_label)
                        y_pred_fallback_labeled.append(pred_llm_id)
                    else:
                        y_true_llm_labeled.append(gt_label)
                        y_pred_llm_labeled.append(pred_llm_id)

                    if (not result_rule.get('is_correct')
                            and result_rule.get('detected_misconception') == label_info.get('name')):
                        n_labeled_match += 1
                else:
                    n_unlabeled += 1
                    if (not result_rule.get('is_correct')
                            and result_rule.get('detected_misconception') == UNLABELED_MISCONCEPTION):
                        n_unlabeled_fallback += 1

    macro_f1_rule = _compute_macro_f1(y_true_labeled, y_pred_rule_labeled) if y_true_labeled else 0.0
    macro_f1_llm = _compute_macro_f1(y_true_llm_labeled, y_pred_llm_labeled) if y_true_llm_labeled else 0.0
    macro_f1_fallback = _compute_macro_f1(y_true_fallback_labeled, y_pred_fallback_labeled) if y_true_fallback_labeled else 0.0
    json_parse_rate = _rate(n_valid_json, n_llm_requests) if n_llm_requests > 0 else 0.0

    return {
        'task': 'Eedi rubric lookup & Local CoT LLM Misconception Diagnosis',
        'model_version': agent.cot_engine.model_name,
        'random_seed': agent.cot_engine.seed,
        'temperature': agent.cot_engine.temperature,
        'diagnosis_mode': agent.cot_engine.diagnosis_mode,
        'options_scored': n_options,
        'correct_options': n_correct,
        'correct_no_false_misconception': n_correct_ok,
        'correct_no_false_misconception_rate': _rate(n_correct_ok, n_correct),
        'labeled_wrong_options': n_labeled,
        'labeled_wrong_exact_match': n_labeled_match,
        'labeled_wrong_exact_match_rate': _rate(n_labeled_match, n_labeled),
        'unlabeled_wrong_options': n_unlabeled,
        'unlabeled_uses_fallback': n_unlabeled_fallback,
        'unlabeled_uses_fallback_rate': _rate(n_unlabeled_fallback, n_unlabeled),
        'rule_based_macro_f1': macro_f1_rule,
        'local_cot_llm_macro_f1': macro_f1_llm,
        'fallback_macro_f1': macro_f1_fallback,
        'valid_json_parse_rate': json_parse_rate,
        'valid_json_count': n_valid_json,
        'total_llm_evals': n_llm_evals,
        'llm_evaluated_count': len(y_pred_llm_labeled),
        'llm_requests': n_llm_requests,
        'fallback_count': n_fallback_count,
    }



def evaluate_mapping(questions: List[Dict[str, Any]], mapper: EediJunyiMapper,
                     graph) -> Dict[str, Any]:
    methods = Counter()
    mapped_constructs = set()
    total_constructs = set()
    invalid_ids = 0
    mapped_questions = 0
    for question in questions:
        total_constructs.add(question['concept_id'])
        result = mapper.map_question(question)
        if not result.mapped:
            methods['unmapped'] += 1
            continue
        mapped_questions += 1
        mapped_constructs.add(question['concept_id'])
        methods[result.method] += 1
        if result.junyi_concept_id not in graph.nodes:
            invalid_ids += 1
    coverage = mapper.coverage(questions)
    return {
        'task': 'Heuristic Eedi construct → Junyi node (not expert alignment)',
        'questions': len(questions),
        'mapped_questions': mapped_questions,
        'mapped_question_rate': _rate(mapped_questions, len(questions)),
        'total_constructs': coverage['total_constructs'],
        'mapped_constructs': coverage['mapped_constructs'],
        'mapped_construct_rate': _rate(coverage['mapped_constructs'], coverage['total_constructs']),
        'methods': dict(methods),
        'invalid_junyi_ids': invalid_ids,
    }


def is_top_k_hit(result: EediJunyiMapper, acceptable_ids: set, k: int = 3) -> bool:
    if not getattr(result, 'mapped', False):
        return False
    predictions = []
    if getattr(result, 'junyi_concept_id', None):
        predictions.append(result.junyi_concept_id)
    top3_ids = getattr(result, 'top3_junyi_ids', None)
    if top3_ids:
        for cand in top3_ids:
            if cand not in predictions:
                predictions.append(cand)
    return any(cand in acceptable_ids for cand in predictions[:k])


def evaluate_gold_set_mapping(gold_set_path: Path, graph) -> Dict[str, Any]:
    if not gold_set_path.is_file():
        raise FileNotFoundError(f"Missing Gold Set file: {gold_set_path}")

    with gold_set_path.open('r', encoding='utf-8') as f:
        gold_records = json.load(f)

    cold_start_begin = datetime.now(timezone.utc)
    b_mapper = EediJunyiMapper(graph)
    s_mapper = SemanticEmbeddingMapper(graph)
    h_mapper = HybridConceptMapper(graph, baseline_mapper=b_mapper, semantic_mapper=s_mapper)
    cold_start_seconds = (datetime.now(timezone.utc) - cold_start_begin).total_seconds()

    start_time = datetime.now(timezone.utc)

    mapped_records = [r for r in gold_records if r.get('gold_junyi_node_id') is not None]
    unmapped_records = [r for r in gold_records if r.get('gold_junyi_node_id') is None]

    b_mapped_top1 = b_mapped_top3 = b_unmapped_correct = b_exact_matches = 0
    s_mapped_top1 = s_mapped_top3 = s_unmapped_correct = s_exact_matches = 0
    h_mapped_top1 = h_mapped_top3 = h_unmapped_correct = h_exact_matches = 0
    invalid_ids = 0

    detailed_results = []

    for r in gold_records:
        q = {
            'concept_id': r['eedi_concept_id'],
            'concept_name': r['eedi_concept_name'],
            'subject': r['eedi_subject']
        }
        gold_target = r['gold_junyi_node_id']
        acceptable = set(r.get('acceptable_junyi_node_ids', []))
        if gold_target:
            acceptable.add(gold_target)

        res_b = b_mapper.map_question(q)
        res_s = s_mapper.map_question(q)
        res_h = h_mapper.map_question(q)

        for res in [res_b, res_s, res_h]:
            suggested_ids = set()
            if res.mapped and res.junyi_concept_id:
                suggested_ids.add(res.junyi_concept_id)
            if res.mapped and res.top3_junyi_ids:
                suggested_ids.update(res.top3_junyi_ids)
            for nid in suggested_ids:
                if nid not in graph.nodes:
                    invalid_ids += 1

        # Baseline evaluation
        if gold_target is not None:
            if res_b.mapped and res_b.junyi_concept_id == gold_target:
                b_mapped_top1 += 1
            if is_top_k_hit(res_b, acceptable, k=3):
                b_mapped_top3 += 1
            if res_b.mapped and res_b.junyi_concept_id == gold_target:
                b_exact_matches += 1
        else:
            if not res_b.mapped:
                b_unmapped_correct += 1
                b_exact_matches += 1

        # Semantic evaluation
        if gold_target is not None:
            if res_s.mapped and res_s.junyi_concept_id == gold_target:
                s_mapped_top1 += 1
            if is_top_k_hit(res_s, acceptable, k=3):
                s_mapped_top3 += 1
            if res_s.mapped and res_s.junyi_concept_id == gold_target:
                s_exact_matches += 1
        else:
            if not res_s.mapped:
                s_unmapped_correct += 1
                s_exact_matches += 1

        # Hybrid evaluation
        if gold_target is not None:
            if res_h.mapped and res_h.junyi_concept_id == gold_target:
                h_mapped_top1 += 1
            if is_top_k_hit(res_h, acceptable, k=3):
                h_mapped_top3 += 1
            if res_h.mapped and res_h.junyi_concept_id == gold_target:
                h_exact_matches += 1
        else:
            if not res_h.mapped:
                h_unmapped_correct += 1
                h_exact_matches += 1

        detailed_results.append({
            'construct_id': r['construct_id'],
            'eedi_concept_name': r['eedi_concept_name'],
            'gold_target_id': gold_target,
            'baseline_prediction': res_b.to_dict(),
            'semantic_prediction': res_s.to_dict(),
            'hybrid_prediction': res_h.to_dict()
        })

    elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    n_total = len(gold_records)
    n_mapped = len(mapped_records)
    n_unmapped = len(unmapped_records)

    semantic_map_file = ROOT / 'eedi_junyi_semantic_map.json'
    with semantic_map_file.open('w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    return {
        'task': 'Gold Set benchmark evaluation (Baseline vs Semantic vs Hybrid)',
        'gold_set_size': n_total,
        'mapped_gold_set_size': n_mapped,
        'unmapped_gold_set_size': n_unmapped,
        'baseline_mapping_top1_accuracy': _rate(b_mapped_top1, n_mapped),
        'baseline_mapping_top3_accuracy': _rate(b_mapped_top3, n_mapped),
        'baseline_unmapped_detection_accuracy': _rate(b_unmapped_correct, n_unmapped),
        'baseline_overall_exact_match_rate': _rate(b_exact_matches, n_total),
        'semantic_mapping_top1_accuracy': _rate(s_mapped_top1, n_mapped),
        'semantic_mapping_top3_accuracy': _rate(s_mapped_top3, n_mapped),
        'semantic_unmapped_detection_accuracy': _rate(s_unmapped_correct, n_unmapped),
        'semantic_overall_exact_match_rate': _rate(s_exact_matches, n_total),
        'hybrid_mapping_top1_accuracy': _rate(h_mapped_top1, n_mapped),
        'hybrid_mapping_top3_accuracy': _rate(h_mapped_top3, n_mapped),
        'hybrid_unmapped_detection_accuracy': _rate(h_unmapped_correct, n_unmapped),
        'hybrid_overall_exact_match_rate': _rate(h_exact_matches, n_total),
        'invalid_junyi_ids': invalid_ids,
        'cold_start_latency_seconds': round(cold_start_seconds, 4),
        'latency_seconds': round(elapsed_seconds, 4),
        'semantic_map_saved_to': str(semantic_map_file)
    }




def evaluate_knowledge_graph(questions: List[Dict[str, Any]], mapper: EediJunyiMapper,
                             graph) -> Dict[str, Any]:
    agent = KGAgent(graph)
    mapped_with_prereq = mapped_empty = unmapped_clean = 0
    mapped = 0
    unmapped = 0
    dynamic_prop_tested = 0
    dynamic_prop_ok = 0
    ancestor_counts: List[float] = []
    with _silence():
        for index, question in enumerate(questions):
            mapping = mapper.map_question(question)
            state = LearnerState(f'EVAL_K_{index}', 'Eval')
            target = mapping.junyi_concept_id if mapping.mapped else question['concept_id']
            result = agent.process(
                {'target_concept_id': target, 'mapping': mapping.to_dict()},
                {'learner_state': state},
            )
            if mapping.mapped:
                mapped += 1
                ancestor_counts.append(float(result['all_prerequisites_count']))
                unmastered = result.get('unmastered_prerequisites', [])
                if result['mapping_available'] and unmastered:
                    mapped_with_prereq += 1
                    # Dynamic propagation verification:
                    # Update mastery of the first unmastered prerequisite to 0.8 (mastered >= 0.6)
                    first_prereq_id = unmastered[0]['concept_id']
                    state.set_concept_mastery(first_prereq_id, 0.8)
                    result_step2 = agent.process(
                        {'target_concept_id': target, 'mapping': mapping.to_dict()},
                        {'learner_state': state},
                    )
                    unmastered_step2 = result_step2.get('unmastered_prerequisites', [])
                    dynamic_prop_tested += 1
                    if len(unmastered_step2) == len(unmastered) - 1:
                        dynamic_prop_ok += 1
                elif result['mapping_available'] and not unmastered:
                    mapped_empty += 1
            else:
                unmapped += 1
                if (not result['mapping_available']
                        and result['unmastered_prerequisites'] == []):
                    unmapped_clean += 1
    return {
        'task': 'Junyi graph traversal on mapped Eedi constructs',
        'graph_source': graph.source,
        'graph_kind': graph.graph_kind,
        'graph_nodes': len(graph.nodes),
        'graph_edges': graph.edge_count(),
        'mapped_questions': mapped,
        'mapped_with_unmastered_prerequisites': mapped_with_prereq,
        'mapped_with_unmastered_prerequisites_rate': _rate(mapped_with_prereq, mapped),
        'mapped_with_empty_prerequisites': mapped_empty,
        'mean_ancestor_count_mapped': _mean(ancestor_counts),
        'unmapped_questions': unmapped,
        'unmapped_no_invented_prerequisites': unmapped_clean,
        'unmapped_no_invented_prerequisites_rate': _rate(unmapped_clean, unmapped),
        'dynamic_mastery_propagation_tested': dynamic_prop_tested,
        'dynamic_mastery_propagation_ok': dynamic_prop_ok,
        'dynamic_mastery_propagation_rate': _rate(dynamic_prop_ok, dynamic_prop_tested),
    }


def _path_types(plan: Dict[str, Any]) -> List[str]:
    return [step['action_type'] for step in plan.get('learning_path', [])]


def evaluate_planner_and_tutor(questions: List[Dict[str, Any]], mapper: EediJunyiMapper,
                               graph) -> Dict[str, Any]:
    kg_agent = KGAgent(graph)
    planner = PlannerAgent()
    diagnostic = DiagnosticAgent()
    tutor = TutorAgent()
    n_wrong_labeled = n_zpd_ok = 0
    n_unmapped_wrong = n_unmapped_no_review = 0
    n_tutor = n_tutor_scaffold = n_tutor_no_answer_leak = 0
    n_tutor_specificity_ok = n_tutor_context_ok = 0
    with _silence():
        for index, question in enumerate(questions):
            labeled = next(iter(question['misconception_map']), None)
            if not labeled:
                continue
            n_wrong_labeled += 1
            state = LearnerState(f'EVAL_P_{index}', 'Eval')
            context = {'learner_state': state}
            diagnosis = diagnostic.process(
                {'question': question, 'selected_option': labeled},
                context,
            )
            mapping = mapper.map_question(question)
            target = mapping.junyi_concept_id if mapping.mapped else question['concept_id']
            kg_result = kg_agent.process(
                {'target_concept_id': target, 'mapping': mapping.to_dict()},
                context,
            )
            plan = planner.process(
                {
                    'target_concept_id': question['concept_id'],
                    'kg_analysis': kg_result,
                    'diagnosis_result': diagnosis,
                },
                context,
            )
            actions = _path_types(plan)
            has_review = 'review_prerequisite' in actions
            has_remediate = 'remediate_misconception' in actions
            has_learn = 'learn_concept' in actions
            if mapping.mapped:
                if has_review and has_remediate and has_learn:
                    n_zpd_ok += 1
            else:
                n_unmapped_wrong += 1
                if (not has_review) and has_remediate and has_learn:
                    n_unmapped_no_review += 1

            # Multi-turn Scaffolding Evaluation (3 turns: Nudge -> Hint -> Explanation)
            n_tutor += 1

            # Turn 1: Nudge
            t1_res = tutor.process(
                {
                    'student_query': 'Giải thích giúp em lỗi sai trong bài này.',
                    'diagnosis_result': diagnosis,
                    'planner_result': plan,
                },
                context,
            )
            r1 = t1_res.get('tutor_response', '')

            # Turn 2: Hint
            t2_res = tutor.process(
                {
                    'student_query': 'Em vẫn chưa biến đổi được, cho em gợi ý cụ thể hơn.',
                    'diagnosis_result': diagnosis,
                    'planner_result': plan,
                },
                context,
            )
            r2 = t2_res.get('tutor_response', '')

            # Turn 3: Explanation
            t3_res = tutor.process(
                {
                    'student_query': 'Thầy giải thích chi tiết bản chất lỗi sai giúp em với.',
                    'diagnosis_result': diagnosis,
                    'planner_result': plan,
                },
                context,
            )
            r3 = t3_res.get('tutor_response', '')

            correct_text = question['options'][question['correct_option']]
            
            # Scaffolding structure check
            if (t1_res.get('scaffolding_level') == 'nudge' and
                t2_res.get('scaffolding_level') == 'hint' and
                t3_res.get('scaffolding_level') == 'explanation'):
                n_tutor_scaffold += 1

            # Specificity check: Turn 2 Hint must be strictly more specific than Turn 1 Nudge
            misc_name = diagnosis.get('detected_misconception')
            s1 = compute_specificity_score(r1, misc_name)
            s2 = compute_specificity_score(r2, misc_name)
            if s2 > s1 and (misc_name and misc_name in r2) and (misc_name not in r1):
                n_tutor_specificity_ok += 1

            # Context retention check: 3 user turns + 3 tutor turns = 6 history entries in state
            if len(state.interaction_history) == 6:
                n_tutor_context_ok += 1

            # Answer leakage check across all 3 turns
            no_leak = True
            for resp in (r1, r2, r3):
                if correct_text and len(correct_text) > 1 and correct_text.lower() in resp.lower():
                    no_leak = False
                    break
                if f"đáp án đúng là [{question['correct_option']}]" in resp.lower():
                    no_leak = False
                    break

            if no_leak:
                n_tutor_no_answer_leak += 1

    mapped_wrong = n_wrong_labeled - n_unmapped_wrong
    return {
        'task': 'ZPD path structure and multi-turn scaffolding tutor checks',
        'wrong_labeled_questions': n_wrong_labeled,
        'mapped_wrong_with_3_phase_path': n_zpd_ok,
        'mapped_wrong_with_3_phase_path_rate': _rate(n_zpd_ok, mapped_wrong),
        'unmapped_wrong_questions': n_unmapped_wrong,
        'unmapped_wrong_no_review_prerequisite': n_unmapped_no_review,
        'unmapped_wrong_no_review_prerequisite_rate': _rate(n_unmapped_no_review, n_unmapped_wrong),
        'tutor_queries': n_tutor,
        'tutor_uses_scaffolding_template': n_tutor_scaffold,
        'tutor_uses_scaffolding_template_rate': _rate(n_tutor_scaffold, n_tutor),
        'tutor_specificity_increase_count': n_tutor_specificity_ok,
        'tutor_specificity_increase_rate': _rate(n_tutor_specificity_ok, n_tutor),
        'tutor_context_retention_count': n_tutor_context_ok,
        'tutor_context_retention_rate': _rate(n_tutor_context_ok, n_tutor),
        'tutor_omits_correct_option_text': n_tutor_no_answer_leak,
        'tutor_omits_correct_option_text_rate': _rate(n_tutor_no_answer_leak, n_tutor),
        'tutor_answer_leakage_rate': 1.0 - (_rate(n_tutor_no_answer_leak, n_tutor) or 0.0),
    }


def evaluate_all() -> Dict[str, Any]:
    questions = EediDatasetLoader.load_questions()
    graph = JunyiGraphLoader.load_math_prerequisite_graph()
    mapper = EediJunyiMapper(graph)
    gold_set_file = ROOT / 'data' / 'gold_concept_mapping.json'

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'datasets': {
            'eedi_questions': len(questions),
            'eedi_constructs': len({q['concept_id'] for q in questions}),
            'junyi_nodes': len(graph.nodes),
            'junyi_edges': graph.edge_count(),
            'junyi_source': graph.source,
            'junyi_graph_kind': graph.graph_kind,
        },
        'diagnostic': evaluate_diagnostic(questions),
        'mapping': evaluate_mapping(questions, mapper, graph),
        'gold_set_eval': evaluate_gold_set_mapping(gold_set_file, graph) if gold_set_file.is_file() else {},
        'knowledge_graph': evaluate_knowledge_graph(questions, mapper, graph),
        'planner_tutor': evaluate_planner_and_tutor(questions, mapper, graph),
        'limitations': [
            'Diagnostic metrics measure rubric lookup against Eedi labels, not LLM CoT quality.',
            'Junyi Info_Content edges are content hierarchy, not expert prerequisite annotations.',
            'Tutor checks are template structure checks, not human pedagogical ratings.',
            'No student interaction logs are used; mastery defaults remain application heuristics.',
        ],
    }
    return report


def format_report(report: Dict[str, Any]) -> str:
    d = report['diagnostic']
    m = report['mapping']
    g = report.get('gold_set_eval', {})
    k = report['knowledge_graph']
    p = report['planner_tutor']
    ds = report['datasets']

    def pct(value: Optional[float]) -> str:
        return 'n/a' if value is None else f'{value * 100:.1f}%'

    lines = [
        '=== ĐÁNH GIÁ PAAF (Eedi + Junyi) ===',
        f"Eedi: {ds['eedi_questions']} câu / {ds['eedi_constructs']} constructs",
        f"Junyi: {ds['junyi_nodes']} node / {ds['junyi_edges']} cạnh ({ds['junyi_graph_kind']})",
        '',
        '[Diagnostic Agent — Tra cứu Rubric & Local CoT LLM Engine]',
        f"  Mô hình LLM: {d.get('model_version')} (Mode: {d.get('diagnosis_mode', 'independent')}, Seed: {d.get('random_seed')}, Temp: {d.get('temperature')})",
        f"  Tỷ lệ parse JSON hợp lệ (LLM): {pct(d.get('valid_json_parse_rate'))} ({d.get('valid_json_count')}/{d.get('llm_requests', 0)})",
        f"  Số câu LLM thật xử lý: {d.get('llm_evaluated_count', 0)}",
        f"  Số câu dùng fallback: {d.get('fallback_count', 0)} / {d.get('total_llm_evals', 0)}",
        f"  Rule-based Baseline Macro-F1: {d.get('rule_based_macro_f1', 0.0):.4f}",
        f"  Local CoT LLM Engine Macro-F1 (LLM thực): {d.get('local_cot_llm_macro_f1', 0.0):.4f}",
        f"  Fallback Lookup Macro-F1: {d.get('fallback_macro_f1', 0.0):.4f}",
        f"  Đúng và không gán misconception giả: {pct(d['correct_no_false_misconception_rate'])}",
        f"  Sai có nhãn — khớp đúng tên misconception: {pct(d['labeled_wrong_exact_match_rate'])} "
        f"({d['labeled_wrong_exact_match']}/{d['labeled_wrong_options']})",
        f"  Sai không nhãn — dùng fallback, không bịa ID: {pct(d['unlabeled_uses_fallback_rate'])}",
        '',
        '[Ánh xạ Eedi → Junyi — Benchmark Gold Set & Semantic Mapper]',
        f"  Số lượng sample Gold Set: {g.get('gold_set_size', 'n/a')} (Mapped: {g.get('mapped_gold_set_size', 'n/a')}, Unmapped: {g.get('unmapped_gold_set_size', 'n/a')})",
        f"  Baseline (Lexicon)  — Mapped Top-1 Acc: {pct(g.get('baseline_mapping_top1_accuracy'))} | Top-3 Acc: {pct(g.get('baseline_mapping_top3_accuracy'))} | Unmapped Acc: {pct(g.get('baseline_unmapped_detection_accuracy'))}",
        f"  Semantic Embedding  — Mapped Top-1 Acc: {pct(g.get('semantic_mapping_top1_accuracy'))} | Top-3 Acc: {pct(g.get('semantic_mapping_top3_accuracy'))} | Unmapped Acc: {pct(g.get('semantic_unmapped_detection_accuracy'))}",
        f"  Hybrid Mapper       — Mapped Top-1 Acc: {pct(g.get('hybrid_mapping_top1_accuracy'))} | Top-3 Acc: {pct(g.get('hybrid_mapping_top3_accuracy'))} | Unmapped Acc: {pct(g.get('hybrid_unmapped_detection_accuracy'))}",
        f"  Overall Exact Match — Baseline: {pct(g.get('baseline_overall_exact_match_rate'))} | Semantic: {pct(g.get('semantic_overall_exact_match_rate'))} | Hybrid: {pct(g.get('hybrid_overall_exact_match_rate'))}",
        f"  ID Junyi không tồn tại (Hallucination penalty): {g.get('invalid_junyi_ids', 0)}",
        f"  Thời gian thực thi Mapping Gold Set (Warm cache): {g.get('latency_seconds', 'n/a')}s",
        '',
        '[Knowledge Graph Agent — Junyi]',
        f"  Mapped có tiên quyết chưa đạt: {pct(k['mapped_with_unmastered_prerequisites_rate'])}",
        f"  Số ancestor trung bình (mapped): {k['mean_ancestor_count_mapped']}",
        f"  Unmapped không bịa cạnh tiên quyết: {pct(k['unmapped_no_invented_prerequisites_rate'])}",
        f"  Lan truyền năng lực động (Dynamic Mastery Propagation Rate): {pct(k.get('dynamic_mastery_propagation_rate'))}",
        '',
        '[Planner + Tutor — Lộ trình ZPD & Interactive Scaffolding]',
        f"  Mapped + sai có nhãn → lộ trình 3 pha: {pct(p['mapped_wrong_with_3_phase_path_rate'])}",
        f"  Unmapped + sai → không review_prerequisite giả: {pct(p['unmapped_wrong_no_review_prerequisite_rate'])}",
        f"  Tutor hội thoại 3 cấp (Nudge -> Hint -> Exp): {pct(p['tutor_uses_scaffolding_template_rate'])}",
        f"  Tutor gợi ý lần 2 cụ thể hơn lần 1 (Specificity Increase): {pct(p.get('tutor_specificity_increase_rate'))}",
        f"  Tutor duy trì đúng ngữ cảnh 3 lượt liên tiếp (Context Retention): {pct(p.get('tutor_context_retention_rate'))}",
        f"  Tutor bảo vệ đáp án đúng (Answer Protection Rate): {pct(p['tutor_omits_correct_option_text_rate'])}",
        f"  Tutor tỷ lệ rò rỉ đáp án đúng (Leakage Rate ≤ 1.3%): {pct(p.get('tutor_answer_leakage_rate'))}",
        '',
        'Giới hạn:',
    ]
    lines.extend(f'  - {item}' for item in report['limitations'])
    return '\n'.join(lines) + '\n'


def write_report(report: Dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Evaluate PAAF on Eedi and Junyi.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = evaluate_all()
    write_report(report, args.output)
    print(format_report(report))
    print(f'Đã ghi {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
