"""
Diagnostic Agent: Performs Chain-of-Thought (CoT) Misconception Analysis & Knowledge Gap Detection.
Analyzes student MCQ responses or free-form text answers against diagnostic question rubrics.
"""

from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.learner_state import LearnerState

UNLABELED_MISCONCEPTION = "Chưa có nhãn hiểu lầm cho lựa chọn này"
UNLABELED_DESCRIPTION = "Học sinh chọn đáp án sai nhưng chưa gán nhãn cụ thể trong rubric."


class DiagnosticAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DiagnosticAgent",
            role_description="Phân tích nguyên nhân lỗi sai (Root Cause Analysis), phát hiện lỗ hổng kiến thức và gắn nhãn hiểu lầm (Misconception Diagnosis)."
        )

    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input expects:
        - 'question': Dict containing Eedi diagnostic question format
        - 'selected_option': str (e.g. 'C')
        - 'student_explanation': Optional[str]
        - 'learner_state': LearnerState
        """
        question = input_data.get("question", {})
        selected_option = input_data.get("selected_option", "").upper()
        if selected_option not in question.get("options", {}):
            raise ValueError(f"Invalid answer option: {selected_option}")
        correct_option = question.get("correct_option", "")
        concept_id = question.get("concept_id", "UNKNOWN")
        concept_name = question.get("concept_name", "Khái niệm chưa rõ")
        learner_state: LearnerState = context.get("learner_state")

        is_correct = (selected_option.upper() == correct_option.upper())

        print(self.format_log(f"Đang phân tích bài làm cho câu hỏi '{question.get('question_id')}' (Khái niệm: {concept_name})..."))

        if is_correct:
            # Student answered correctly -> Boost mastery score
            current_mastery = learner_state.get_concept_mastery(concept_id)
            new_mastery = min(1.0, current_mastery + 0.25)
            learner_state.set_concept_mastery(concept_id, new_mastery)

            reasoning_cot = f"Học sinh đã chọn đáp án chính xác '{selected_option}'. Năng lực khái niệm '{concept_name}' tăng lên {new_mastery:.2f}."
            diagnosis_result = {
                "is_correct": True,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "detected_misconception": None,
                "mastery_score": new_mastery,
                "cot_explanation": reasoning_cot
            }
        else:
            # Student answered incorrectly -> Trigger Misconception Analysis
            misconception_map = question.get("misconception_map", {})
            misc_info = misconception_map.get(selected_option, {
                "name": UNLABELED_MISCONCEPTION,
                "description": UNLABELED_DESCRIPTION,
                "severity": "unknown"
            })

            # Chain-of-Thought (CoT) reasoning simulation
            cot_explanation = (
                f"1. Quan sát: Học sinh chọn phương án '{selected_option}' thay vì đáp án đúng '{correct_option}'.\n"
                f"2. Phân tích nguyên nhân (Root Cause): Học sinh mắc lỗi '{misc_info['name']}'.\n"
                f"3. Diễn giải chi tiết: {misc_info['description']}\n"
                f"4. Kết luận lỗ hổng: Cần kiểm tra thêm mức độ hiểu khái niệm '{concept_name}'."
            )

            # Record misconception in learner state
            learner_state.add_misconception(
                concept_id=concept_id,
                misconception_name=misc_info["name"],
                description=misc_info["description"],
                severity=misc_info.get("severity", "unknown")
            )

            diagnosis_result = {
                "is_correct": False,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "selected_option": selected_option,
                "correct_option": correct_option,
                "detected_misconception": misc_info["name"],
                "severity": misc_info.get("severity", "unknown"),
                "mastery_score": learner_state.get_concept_mastery(concept_id),
                "cot_explanation": cot_explanation
            }

        return diagnosis_result
