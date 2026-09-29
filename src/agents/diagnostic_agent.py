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

    @staticmethod
    def build_misconception_badge(
        question_id: str,
        selected_option: str,
        misconception_id: str,
        misconception_name: str,
        description: str,
        severity: str = "unknown",
        confidence_score: float = 1.0,
        cot_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates a standardized Misconception Badge schema dictionary."""
        is_labeled = misconception_id not in (None, "", "unlabeled") and misconception_name != UNLABELED_MISCONCEPTION
        sev = (severity or "unknown").lower()
        if sev in ("high", "critical"):
            color = "#EF4444"
            variant = "danger"
        elif sev == "medium":
            color = "#F59E0B"
            variant = "warning"
        elif sev == "low":
            color = "#3B82F6"
            variant = "info"
        elif is_labeled:
            color = "#8B5CF6"
            variant = "purple"
        else:
            color = "#6B7280"
            variant = "neutral"

        badge_id = f"BADGE_{misconception_id}" if is_labeled else f"BADGE_{question_id}_{selected_option}"
        label = misconception_name if is_labeled else "Lỗi chưa phân loại"
        badge_type = "misconception" if is_labeled else "unclassified"

        if not cot_summary:
            clean_desc = description.replace("\n", " ").strip()
            cot_summary = f"Lỗi: {label} — {clean_desc[:100]}..." if is_labeled else clean_desc[:100]

        return {
            "badge_id": badge_id,
            "misconception_id": str(misconception_id) if misconception_id else "unlabeled",
            "label": label,
            "badge_color": color,
            "badge_variant": variant,
            "badge_type": badge_type,
            "severity": sev,
            "cot_summary": cot_summary,
            "root_cause": description,
            "confidence_score": round(confidence_score, 4)
        }

    @staticmethod
    def build_root_cause_analysis(
        selected_option: str,
        correct_option: str,
        misconception_name: str,
        detailed_explanation: str,
        concept_name: str,
        cot_explanation: Optional[str] = None
    ) -> Dict[str, Any]:
        """Constructs structured Root Cause Analysis steps for the incorrect answer."""
        observation = f"Học sinh chọn phương án '{selected_option}' thay vì đáp án đúng '{correct_option}'."
        gap_conclusion = f"Cần củng cố và kiểm tra thêm mức độ hiểu khái niệm '{concept_name}'."

        if cot_explanation and "\n" in cot_explanation:
            cot_steps = [line.strip() for line in cot_explanation.split("\n") if line.strip()]
        else:
            cot_steps = [
                f"1. Quan sát: {observation}",
                f"2. Phân tích nguyên nhân (Root Cause): Học sinh mắc lỗi '{misconception_name}'.",
                f"3. Diễn giải chi tiết: {detailed_explanation}",
                f"4. Kết luận lỗ hổng: {gap_conclusion}"
            ]

        return {
            "observation": observation,
            "misconception": misconception_name,
            "detailed_explanation": detailed_explanation,
            "gap_conclusion": gap_conclusion,
            "cot_steps": cot_steps
        }

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
            reasoning_cot = f"Học sinh đã chọn đáp án chính xác '{selected_option}'."
            diagnosis_result = {
                "is_correct": True,
                "concept_id": concept_id,
                "concept_name": concept_name,
                "detected_misconception": None,
                "misconception_id": "unlabeled",
                "mastery_score": current_mastery,
                "cot_explanation": reasoning_cot,
                "confidence_score": 1.0,
                "is_valid_parse": None,
                "used_fallback": False,
                "error_reason": None,
                "engine": "rule_based",
                "misconception_badge": None,
                "root_cause_analysis": None
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

                badge = self.build_misconception_badge(
                    question_id=question.get("question_id", "UNKNOWN"),
                    selected_option=selected_option,
                    misconception_id=misc_id,
                    misconception_name=matched_name,
                    description=cot_explanation,
                    severity="unknown",
                    confidence_score=confidence_score
                )
                root_cause = self.build_root_cause_analysis(
                    selected_option=selected_option,
                    correct_option=correct_option,
                    misconception_name=matched_name,
                    detailed_explanation=cot_explanation,
                    concept_name=concept_name,
                    cot_explanation=cot_explanation
                )

                diagnosis_result = {
                    "is_correct": False,
                    "question_id": question.get("question_id", "UNKNOWN"),
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
                    "question_text": question.get("question_text", ""),
                    "options": question.get("options", {}),
                    "is_valid_parse": is_valid_parse,
                    "parse_attempts": attempts,
                    "used_fallback": used_fallback,
                    "error_reason": error_reason,
                    "engine": "llm_cot",
                    "misconception_badge": badge,
                    "root_cause_analysis": root_cause
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

                badge = self.build_misconception_badge(
                    question_id=question.get("question_id", "UNKNOWN"),
                    selected_option=selected_option,
                    misconception_id=misc_id,
                    misconception_name=misc_info["name"],
                    description=misc_info["description"],
                    severity=misc_info.get("severity", "unknown"),
                    confidence_score=1.0
                )
                root_cause = self.build_root_cause_analysis(
                    selected_option=selected_option,
                    correct_option=correct_option,
                    misconception_name=misc_info["name"],
                    detailed_explanation=misc_info["description"],
                    concept_name=concept_name,
                    cot_explanation=cot_explanation
                )

                diagnosis_result = {
                    "is_correct": False,
                    "question_id": question.get("question_id", "UNKNOWN"),
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
                    "question_text": question.get("question_text", ""),
                    "options": question.get("options", {}),
                    "engine": "rule_based",
                    "misconception_badge": badge,
                    "root_cause_analysis": root_cause
                }

        return diagnosis_result

