"""
Diagnostic Agent: Performs Chain-of-Thought (CoT) Misconception Analysis & Knowledge Gap Detection.
Analyzes student MCQ responses or free-form text answers against diagnostic question rubrics.
"""

from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent
from src.agents.cot_diagnostic_engine import LocalCoTDiagnosticEngine
from src.core.learner_state import LearnerState

UNLABELED_MISCONCEPTION = "Chưa có nhãn hiểu lầm cho lựa chọn này"
UNLABELED_DESCRIPTION = "Học sinh chọn đáp án sai nhưng chưa gán nhãn cụ thể trong rubric."


class DiagnosticAgent(BaseAgent):
    def __init__(self, cot_engine: Optional[LocalCoTDiagnosticEngine] = None):
        super().__init__(
            name="DiagnosticAgent",
            role_description="Phân tích nguyên nhân lỗi sai (Root Cause Analysis), phát hiện lỗ hổng kiến thức và gắn nhãn hiểu lầm (Misconception Diagnosis)."
        )
        self.cot_engine = cot_engine or LocalCoTDiagnosticEngine()

    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input expects:
        - 'question': Dict containing Eedi diagnostic question format
        - 'selected_option': str (e.g. 'C')
        - 'use_llm': bool (optional, whether to use Local CoT LLM Engine, default False)
        - 'student_explanation': Optional[str]
        - 'learner_state': LearnerState
        """
        question = input_data.get("question", {})
        selected_option = input_data.get("selected_option", "").upper()
        use_llm = input_data.get("use_llm", False)

        if selected_option not in question.get("options", {}):
            raise ValueError(f"Invalid answer option: {selected_option}")

        correct_option = question.get("correct_option", "")
        concept_id = question.get("concept_id", "UNKNOWN")
        concept_name = question.get("concept_name", "Khái niệm chưa rõ")
        learner_state: Optional[LearnerState] = context.get("learner_state")

        is_correct = (selected_option.upper() == correct_option.upper())

        print(self.format_log(f"Đang phân tích bài làm cho câu hỏi '{question.get('question_id')}' (Khái niệm: {concept_name})..."))

        if is_correct:
            current_mastery = learner_state.get_concept_mastery(concept_id) if learner_state else 0.0
            new_mastery = min(1.0, current_mastery + 0.25)
            if learner_state:
                learner_state.set_concept_mastery(concept_id, new_mastery)

            reasoning_cot = f"Học sinh đã chọn đáp án chính xác '{selected_option}'. Năng lực khái niệm '{concept_name}' tăng lên {new_mastery:.2f}."
            diagnosis_result = {
                "is_correct": True,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "detected_misconception": None,
                "misconception_id": "unlabeled",
                "mastery_score": new_mastery,
                "cot_explanation": reasoning_cot,
                "confidence_score": 1.0,
                "is_valid_parse": None,
                "used_fallback": False,
                "error_reason": None,
                "engine": "rule_based"
            }
        else:
            misconception_map = question.get("misconception_map", {})
            if use_llm:
                schema_out, is_valid_parse, attempts, used_fallback, error_reason = self.cot_engine.diagnose(question, selected_option)
                misc_id = schema_out.misconception_id
                cot_explanation = schema_out.cot_reasoning
                confidence_score = schema_out.confidence_score

                # Resolve misconception name from rubric if ID matches, else fallback description
                matched_name = UNLABELED_MISCONCEPTION
                if misc_id != "unlabeled":
                    for opt, m in misconception_map.items():
                        if str(m.get("misconception_id")) == str(misc_id):
                            matched_name = m.get("name", UNLABELED_MISCONCEPTION)
                            break
                    if matched_name == UNLABELED_MISCONCEPTION:
                        matched_name = f"Misconception #{misc_id}"

                if learner_state:
                    learner_state.add_misconception(
                        concept_id=concept_id,
                        misconception_name=matched_name,
                        description=cot_explanation,
                        severity="unknown"
                    )

                mastery = learner_state.get_concept_mastery(concept_id) if learner_state else 0.0

                diagnosis_result = {
                    "is_correct": False,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "selected_option": selected_option,
                    "correct_option": correct_option,
                    "detected_misconception": matched_name,
                    "misconception_id": misc_id,
                    "severity": "unknown",
                    "mastery_score": mastery,
                    "cot_explanation": cot_explanation,
                    "confidence_score": confidence_score,
                    "is_valid_parse": is_valid_parse,
                    "parse_attempts": attempts,
                    "used_fallback": used_fallback,
                    "error_reason": error_reason,
                    "engine": "llm_cot"
                }
            else:
                misc_info = misconception_map.get(selected_option, {
                    "name": UNLABELED_MISCONCEPTION,
                    "description": UNLABELED_DESCRIPTION,
                    "severity": "unknown"
                })
                misc_id = misc_info.get("misconception_id", "unlabeled")

                cot_explanation = (
                    f"1. Quan sát: Học sinh chọn phương án '{selected_option}' thay vì đáp án đúng '{correct_option}'.\n"
                    f"2. Phân tích nguyên nhân (Root Cause): Học sinh mắc lỗi '{misc_info['name']}'.\n"
                    f"3. Diễn giải chi tiết: {misc_info['description']}\n"
                    f"4. Kết luận lỗ hổng: Cần kiểm tra thêm mức độ hiểu khái niệm '{concept_name}'."
                )

                if learner_state:
                    learner_state.add_misconception(
                        concept_id=concept_id,
                        misconception_name=misc_info["name"],
                        description=misc_info["description"],
                        severity=misc_info.get("severity", "unknown")
                    )

                mastery = learner_state.get_concept_mastery(concept_id) if learner_state else 0.0

                diagnosis_result = {
                    "is_correct": False,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "selected_option": selected_option,
                    "correct_option": correct_option,
                    "detected_misconception": misc_info["name"],
                    "misconception_id": misc_id,
                    "severity": misc_info.get("severity", "unknown"),
                    "mastery_score": mastery,
                    "cot_explanation": cot_explanation,
                    "confidence_score": 1.0,
                    "engine": "rule_based"
                }

        return diagnosis_result

