"""
Interactive Tutor Agent: Handles multi-turn conversational remediation and pedagogical scaffolding.
Provides hint-based guidance, doubt resolution, and constructive feedback without giving direct answers.
"""

from typing import Dict, Any, Optional
from src.agents.base_agent import BaseAgent
from src.core.learner_state import LearnerState


class TutorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="TutorAgent",
            role_description="Trợ giảng hội thoại 2 chiều, thực hiện nâng đỡ sư phạm (Scaffolding), giải thích khái niệm và hỗ trợ vượt qua bế tắc."
        )

    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input expects:
        - 'student_query': str
        - 'diagnosis_result': Dict
        - 'planner_result': Dict
        - 'learner_state': LearnerState
        """
        student_query = input_data.get("student_query", "")
        diagnosis_result = input_data.get("diagnosis_result", {})
        planner_result = input_data.get("planner_result", {})
        learner_state: LearnerState = context.get("learner_state")

        print(self.format_log("Đang phản hồi hội thoại sư phạm (Scaffolding Dialogue)..."))

        # Extract active misconception or current step context
        detected_misc = diagnosis_result.get("detected_misconception")
        concept_name = diagnosis_result.get("concept_name", "Bài học")

        # Generate Scaffolding Response (Simulation of pedagogical LLM Prompt)
        if "tại sao" in student_query.lower() or "giải thích" in student_query.lower() or "không hiểu" in student_query.lower():
            if detected_misc:
                tutor_response = (
                    f"Chào bạn {learner_state.student_name}! Ở bài tập '{concept_name}', bạn đang vướng ở lỗi '{detected_misc}'.\n\n"
                    f"💡 **Gợi ý từng bước (Scaffolding Hint):**\n"
                    f"1. Hãy đọc lại yêu cầu bài và nêu quy tắc bạn đã dùng.\n"
                    f"2. Đối chiếu cách làm của bạn với nhãn lỗi: {detected_misc}.\n"
                    f"3. Bạn có thể chỉ ra bước nào cần kiểm tra lại không?"
                )
            else:
                tutor_response = (
                    f"Chào bạn {learner_state.student_name}! Để giải quyết bài tập này, bạn hãy xem bước 1 trong lộ trình cá nhân hóa vừa tạo.\n"
                    f"Hãy thử phát biểu lại quy tắc bạn đã áp dụng trong bài nhé!"
                )
        else:
            tutor_response = (
                f"Rất tốt! Thầy/cô thấy bạn đang tiến bộ ở lộ trình '{concept_name}'. "
                f"Hãy hoàn thành bước tiếp theo trong lộ trình cá nhân hóa nhé!"
            )

        # Log dialogue into learner state history
        learner_state.add_interaction(role="user", message=student_query)
        learner_state.add_interaction(role="assistant", message=tutor_response, agent_name=self.name)

        return {
            "student_query": student_query,
            "tutor_response": tutor_response,
            "scaffolding_technique": "Graduated Hinting / Guided Inquiry",
            "next_suggested_action": "Student response or step execution"
        }
