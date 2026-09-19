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
from src.core.learner_state import LearnerState
from src.data.concept_mapping import EediJunyiMapper
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


def evaluate_knowledge_graph(questions: List[Dict[str, Any]], mapper: EediJunyiMapper,
                             graph) -> Dict[str, Any]:
    agent = KGAgent(graph)
    mapped_with_prereq = mapped_empty = unmapped_clean = 0
    mapped = 0
    unmapped = 0
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
                if result['mapping_available'] and result['unmastered_prerequisites']:
                    mapped_with_prereq += 1
                elif result['mapping_available'] and not result['unmastered_prerequisites']:
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

            n_tutor += 1
            tutor_result = tutor.process(
                {
                    'student_query': 'Giải thích giúp em lỗi sai trong bài này.',
                    'diagnosis_result': diagnosis,
                    'planner_result': plan,
                },
                context,
            )
            response = tutor_result.get('tutor_response', '')
            correct_text = question['options'][question['correct_option']]
            if 'Gợi ý' in response or 'Scaffolding' in response:
                n_tutor_scaffold += 1
            if correct_text and correct_text not in response:
                n_tutor_no_answer_leak += 1
    mapped_wrong = n_wrong_labeled - n_unmapped_wrong
    return {
        'task': 'ZPD path structure and tutor template checks',
        'wrong_labeled_questions': n_wrong_labeled,
        'mapped_wrong_with_3_phase_path': n_zpd_ok,
        'mapped_wrong_with_3_phase_path_rate': _rate(n_zpd_ok, mapped_wrong),
        'unmapped_wrong_questions': n_unmapped_wrong,
        'unmapped_wrong_no_review_prerequisite': n_unmapped_no_review,
        'unmapped_wrong_no_review_prerequisite_rate': _rate(n_unmapped_no_review, n_unmapped_wrong),
        'tutor_queries': n_tutor,
        'tutor_uses_scaffolding_template': n_tutor_scaffold,
        'tutor_uses_scaffolding_template_rate': _rate(n_tutor_scaffold, n_tutor),
        'tutor_omits_correct_option_text': n_tutor_no_answer_leak,
        'tutor_omits_correct_option_text_rate': _rate(n_tutor_no_answer_leak, n_tutor),
    }


def evaluate_all() -> Dict[str, Any]:
    questions = EediDatasetLoader.load_questions()
    graph = JunyiGraphLoader.load_math_prerequisite_graph()
    mapper = EediJunyiMapper(graph)
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
        'knowledge_graph': evaluate_knowledge_graph(questions, mapper, graph),
        'planner_tutor': evaluate_planner_and_tutor(questions, mapper, graph),
        'limitations': [
            'Diagnostic metrics measure rubric lookup against Eedi labels, not LLM CoT quality.',
            'Eedi→Junyi mapping is heuristic; there is no expert alignment gold set.',
            'Junyi Info_Content edges are content hierarchy, not expert prerequisite annotations.',
            'Tutor checks are template structure checks, not human pedagogical ratings.',
            'No student interaction logs are used; mastery defaults remain application heuristics.',
        ],
    }
    return report


def format_report(report: Dict[str, Any]) -> str:
    d = report['diagnostic']
    m = report['mapping']
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
        '[Ánh xạ Eedi → Junyi — heuristic]',
        f"  Câu hỏi ánh xạ được: {pct(m['mapped_question_rate'])} ({m['mapped_questions']}/{m['questions']})",
        f"  Construct ánh xạ được: {pct(m['mapped_construct_rate'])} "
        f"({m['mapped_constructs']}/{m['total_constructs']})",
        f"  ID Junyi không tồn tại trong graph: {m['invalid_junyi_ids']}",
        f"  Phương pháp: {m['methods']}",
        '',
        '[Knowledge Graph Agent — Junyi]',
        f"  Mapped có tiên quyết chưa đạt: {pct(k['mapped_with_unmastered_prerequisites_rate'])}",
        f"  Số ancestor trung bình (mapped): {k['mean_ancestor_count_mapped']}",
        f"  Unmapped không bịa cạnh tiên quyết: {pct(k['unmapped_no_invented_prerequisites_rate'])}",
        '',
        '[Planner + Tutor — kiểm tra cấu trúc]',
        f"  Mapped + sai có nhãn → lộ trình 3 pha: {pct(p['mapped_wrong_with_3_phase_path_rate'])}",
        f"  Unmapped + sai → không review_prerequisite giả: {pct(p['unmapped_wrong_no_review_prerequisite_rate'])}",
        f"  Tutor dùng template scaffolding: {pct(p['tutor_uses_scaffolding_template_rate'])}",
        f"  Tutor không chép nguyên đáp án đúng: {pct(p['tutor_omits_correct_option_text_rate'])}",
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
