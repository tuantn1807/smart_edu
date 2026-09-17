"""
Pedagogical Planner Agent: Generates ZPD (Zone of Proximal Development)-guided dynamic learning paths.
Sequences foundational review steps, target concept practice, and scaffolding challenges.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.core.learner_state import LearnerState, LearningPathStep


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            role_description="Lập lộ trình học tập cá nhân hóa dựa trên lý thuyết Vùng phát triển gần nhất (ZPD - Zone of Proximal Development)."
        )

    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input expects:
        - 'target_concept_id': str
        - 'kg_analysis': Dict output from KGAgent
        - 'diagnosis_result': Dict output from DiagnosticAgent
        - 'learner_state': LearnerState
        """
        kg_analysis = input_data.get("kg_analysis", {})
        diagnosis_result = input_data.get("diagnosis_result", {})
        target_concept_id = input_data.get("target_concept_id", "UNKNOWN")
        learner_state: LearnerState = context.get("learner_state")

        print(self.format_log("Đang tổng hợp dữ liệu nhận thức và thiết lập lộ trình học tập cá nhân hóa (ZPD)..."))

        unmastered_prereqs = kg_analysis.get("unmastered_prerequisites", [])

        steps: List[LearningPathStep] = []
        step_counter = 1

        # Step Phase 1: Remediate unmastered foundational prerequisites (Scaffolding Phase)
        for prereq in unmastered_prereqs:
            step = LearningPathStep(
                step_id=step_counter,
                concept_id=prereq["concept_id"],
                concept_name=prereq["name"],
                action_type="review_prerequisite",
                description=f"Ôn tập lại khái niệm tiền đề '{prereq['name']}' (Mức độ hiện tại: {prereq['current_mastery']*100:.0f}%). {prereq['description']}"
            )
            steps.append(step)
            step_counter += 1

        # Step Phase 2: Address specific misconception if identified
        if diagnosis_result.get("detected_misconception"):
            misc_name = diagnosis_result.get("detected_misconception")
            step = LearningPathStep(
                step_id=step_counter,
                concept_id=target_concept_id,
                concept_name=diagnosis_result.get("concept_name", target_concept_id),
                action_type="remediate_misconception",
                description=f"Thực hành bài tập khắc phục hiểu lầm: '{misc_name}'."
            )
            steps.append(step)
            step_counter += 1

        # Step Phase 3: Targeted practice on primary concept (ZPD target)
        step = LearningPathStep(
            step_id=step_counter,
            concept_id=target_concept_id,
            concept_name=diagnosis_result.get("concept_name", target_concept_id),
            action_type="learn_concept",
            description=f"Luyện tập bài tập ứng dụng đạt chuẩn ở khái niệm mục tiêu '{target_concept_id}'."
        )
        steps.append(step)
        step_counter += 1

        # Update active learning path in centralized learner state
        learner_state.set_learning_path(steps)

        zpd_rationale = (
            f"Lộ trình được thiết kế gồm {len(steps)} bước: Ưu tiên củng cố {len(unmastered_prereqs)} khái niệm nền tảng bị đứt gãy "
            f"trước khi tiến lên làm bài tập mức độ nâng cao ở khái niệm mục tiêu, đảm bảo đúng vùng ZPD của học sinh."
        )

        if not kg_analysis.get("mapping_available", True):
            zpd_rationale = (
                "Lộ trình dựa trên nhãn lỗi Eedi; construct này chưa ánh xạ được sang node Junyi "
                "nên chưa duyệt cây tiên quyết."
            )
        elif not kg_analysis.get("prerequisites_available", True):
            zpd_rationale = "Lộ trình dựa trên khái niệm và nhãn lỗi; chưa đủ dữ liệu tiên quyết để đánh giá ZPD."

        return {
            "target_concept_id": target_concept_id,
            "total_steps": len(steps),
            "learning_path": [
                {
                    "step_id": s.step_id,
                    "action_type": s.action_type,
                    "concept": s.concept_name,
                    "description": s.description
                } for s in steps
            ],
            "zpd_rationale": zpd_rationale
        }
