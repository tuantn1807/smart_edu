"""
Pedagogical Planner Agent: Generates ZPD (Zone of Proximal Development)-guided dynamic learning paths.
Sequences foundational review steps, target concept practice, and scaffolding challenges with real Eedi questions.
"""

from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent
from src.core.learner_state import LearnerState, LearningPathStep
from src.data.item_repository import EediItemRepository


class PlannerAgent(BaseAgent):
    def __init__(self, item_repository: Optional[EediItemRepository] = None):
        super().__init__(
            name="PlannerAgent",
            role_description="Lập lộ trình học tập cá nhân hóa dựa trên lý thuyết Vùng phát triển gần nhất (ZPD - Zone of Proximal Development)."
        )
        self.item_repository = item_repository

    def _get_item_repo(self, context: Dict[str, Any]) -> EediItemRepository:
        if "item_repository" in context and context["item_repository"] is not None:
            return context["item_repository"]
        if self.item_repository is not None:
            return self.item_repository
        self.item_repository = EediItemRepository()
        return self.item_repository

    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input expects:
        - 'target_concept_id': str
        - 'kg_analysis': Dict output from KGAgent
        - 'diagnosis_result': Dict output from DiagnosticAgent
        - 'learner_state': LearnerState in context
        """
        kg_analysis = input_data.get("kg_analysis", {})
        diagnosis_result = input_data.get("diagnosis_result", {})
        target_concept_id = input_data.get("target_concept_id", "UNKNOWN")
        learner_state: Optional[LearnerState] = context.get("learner_state")
        item_repo = self._get_item_repo(context)

        print(self.format_log("Đang tổng hợp dữ liệu nhận thức và thiết lập lộ trình học tập cá nhân hóa (ZPD)..."))

        unmastered_prereqs = kg_analysis.get("unmastered_prerequisites", [])
        consecutive_correct = learner_state.consecutive_correct if learner_state else 0
        answered_questions = list(learner_state.answered_questions) if learner_state else []

        steps: List[LearningPathStep] = []
        step_counter = 1
        used_qids = list(answered_questions)

        # Dynamic ZPD Route Decision: Correct consecutive answers accelerate route
        if consecutive_correct >= 2:
            # Accelerated Path: Student has demonstrated consecutive mastery, advance to challenge
            q_item = item_repo.select_item(
                concept_id=target_concept_id,
                exclude_ids=used_qids,
                action_type="advanced_challenge"
            )
            used_qids.append(q_item["question_id"])

            step = LearningPathStep(
                step_id=step_counter,
                concept_id=target_concept_id,
                concept_name=q_item.get("concept_name", target_concept_id),
                action_type="advanced_challenge",
                description=f"Thách thức nâng cao: Bài tập vận dụng cao ở khái niệm mục tiêu '{target_concept_id}' (Đã đúng {consecutive_correct} câu liên tiếp).",
                question_id=q_item["question_id"],
                question_details=q_item
            )
            steps.append(step)
            step_counter += 1

            zpd_rationale = (
                f"Tăng độ khó lộ trình: Học sinh trả lời đúng {consecutive_correct} câu liên tiếp. "
                f"Hệ thống tự động đẩy nhanh tiến độ ZPD lên bài tập nâng cao (Advanced Challenge) "
                f"bằng câu hỏi thực tế {q_item['question_id']} và bỏ qua bước ôn tập tiên quyết."
            )
        else:
            # Standard / Remediation ZPD Path
            # Step Phase 1: Remediate unmastered foundational prerequisites (Scaffolding Phase)
            for prereq in unmastered_prereqs:
                prereq_cid = prereq["concept_id"]
                q_item = item_repo.select_item(
                    concept_id=prereq_cid,
                    exclude_ids=used_qids,
                    action_type="review_prerequisite"
                )
                used_qids.append(q_item["question_id"])

                step = LearningPathStep(
                    step_id=step_counter,
                    concept_id=prereq_cid,
                    concept_name=prereq["name"],
                    action_type="review_prerequisite",
                    description=f"Ôn tập khái niệm tiền đề '{prereq['name']}' (Mức độ hiện tại: {prereq['current_mastery']*100:.0f}%).",
                    question_id=q_item["question_id"],
                    question_details=q_item
                )
                steps.append(step)
                step_counter += 1

            # Step Phase 2: Address specific misconception if identified
            if diagnosis_result.get("detected_misconception"):
                misc_name = diagnosis_result.get("detected_misconception")
                misc_id = diagnosis_result.get("misconception_id")
                q_item = item_repo.select_item(
                    concept_id=target_concept_id,
                    misconception_id_or_name=misc_id or misc_name,
                    exclude_ids=used_qids,
                    action_type="remediate_misconception"
                )
                used_qids.append(q_item["question_id"])

                step = LearningPathStep(
                    step_id=step_counter,
                    concept_id=target_concept_id,
                    concept_name=diagnosis_result.get("concept_name", target_concept_id),
                    action_type="remediate_misconception",
                    description=f"Thực hành bài tập khắc phục hiểu lầm: '{misc_name}'.",
                    question_id=q_item["question_id"],
                    question_details=q_item
                )
                steps.append(step)
                step_counter += 1

            # Step Phase 3: Targeted practice on primary concept (ZPD target)
            q_item = item_repo.select_item(
                concept_id=target_concept_id,
                exclude_ids=used_qids,
                action_type="practice_exercise"
            )
            used_qids.append(q_item["question_id"])

            step = LearningPathStep(
                step_id=step_counter,
                concept_id=target_concept_id,
                concept_name=diagnosis_result.get("concept_name", target_concept_id),
                action_type="learn_concept",
                description=f"Luyện tập bài tập ứng dụng đạt chuẩn ở khái niệm mục tiêu '{target_concept_id}'.",
                question_id=q_item["question_id"],
                question_details=q_item
            )
            steps.append(step)
            step_counter += 1

            zpd_rationale = (
                f"Lộ trình được thiết kế gồm {len(steps)} bước: Ưu tiên củng cố {len(unmastered_prereqs)} khái niệm nền tảng bị đứt gãy "
                f"trước khi tiến lên làm bài tập ở khái niệm mục tiêu, đảm bảo đúng vùng ZPD của học sinh. "
                f"Mỗi bước đều gắn với question_id Eedi tồn tại thực tế."
            )

            if not kg_analysis.get("mapping_available", True):
                zpd_rationale = (
                    "Lộ trình dựa trên nhãn lỗi Eedi; construct này chưa ánh xạ được sang node Junyi "
                    "nên chưa duyệt cây tiên quyết. Đã gán câu hỏi Eedi thực tế."
                )
            elif not kg_analysis.get("prerequisites_available", True):
                zpd_rationale = "Lộ trình dựa trên khái niệm và nhãn lỗi; chưa đủ dữ liệu tiên quyết để đánh giá ZPD. Đã gán câu hỏi Eedi thực tế."

        if learner_state:
            learner_state.set_learning_path(steps)

        return {
            "target_concept_id": target_concept_id,
            "total_steps": len(steps),
            "consecutive_correct": consecutive_correct,
            "learning_path": [
                {
                    "step_id": s.step_id,
                    "action_type": s.action_type,
                    "concept": s.concept_name,
                    "description": s.description,
                    "status": s.status,
                    "question_id": s.question_id,
                    "question_details": s.question_details
                } for s in steps
            ],
            "zpd_rationale": zpd_rationale
        }
