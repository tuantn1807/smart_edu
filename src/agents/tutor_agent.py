"""
Interactive Tutor Agent: Handles multi-turn conversational remediation and pedagogical scaffolding.
Provides hint-based guidance, doubt resolution, and constructive feedback without giving direct answers,
implementing 3-level Graduated Hinting (Nudge -> Hint -> Explanation).
"""

from typing import Dict, Any, Optional, List, Tuple
from src.agents.base_agent import BaseAgent
from src.agents.tutor_engine import (
    LocalScaffoldingTutorEngine,
    LEVEL_NUDGE,
    LEVEL_HINT,
    LEVEL_EXPLANATION,
)
from src.core.learner_state import LearnerState


class TutorAgent(BaseAgent):
    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        api_base: str = "http://localhost:11434",
        backend: str = "ollama",
        enable_ollama_fallback: bool = True,
    ):
        super().__init__(
            name="TutorAgent",
            role_description="Trợ giảng hội thoại 2 chiều, thực hiện nâng đỡ sư phạm (Scaffolding), giải thích khái niệm và hỗ trợ vượt qua bế tắc."
        )
        self.engine = LocalScaffoldingTutorEngine(
            model_name=model_name,
            api_base=api_base,
            backend=backend,
            enable_ollama_fallback=enable_ollama_fallback,
        )

    def _determine_scaffolding_level(
        self,
        interaction_history: List[Dict[str, Any]],
        current_misconception: Optional[str],
        current_question_id: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Determines the Graduated Hinting level strictly scoped to the tuple (question_id, misconception).
        Turn 1 for current (question_id, misconception) -> Nudge
        Turn 2 for current (question_id, misconception) -> Hint
        Turn 3+ for current (question_id, misconception) -> Explanation
        """
        if not current_question_id or not current_misconception:
            return LEVEL_NUDGE, 1

        curr_qid = str(current_question_id).strip()
        curr_misc = str(current_misconception).strip()

        matching_turns = 0
        for entry in interaction_history:
            if entry.get("agent") == self.name or entry.get("role") == "assistant":
                meta = entry.get("metadata", {})
                hist_qid = meta.get("question_id")
                hist_misc = meta.get("misconception")

                # Both question_id and misconception must be present and match strictly
                if hist_qid is not None and hist_misc is not None:
                    if str(hist_qid).strip() == curr_qid and str(hist_misc).strip() == curr_misc:
                        matching_turns += 1

        turn_index = matching_turns + 1

        if turn_index == 1:
            level = LEVEL_NUDGE
        elif turn_index == 2:
            level = LEVEL_HINT
        else:
            level = LEVEL_EXPLANATION

        return level, turn_index

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
        learner_state: Optional[LearnerState] = context.get("learner_state")

        print(self.format_log("Đang phản hồi hội thoại sư phạm (Scaffolding Dialogue Multi-turn)..."))

        student_name = learner_state.student_name if learner_state else "Học sinh"
        history = learner_state.interaction_history if learner_state else []

        # Extract context strictly without concept_id fallback for question_id
        question_id = diagnosis_result.get("question_id") or diagnosis_result.get("question", {}).get("question_id")
        detected_misc = diagnosis_result.get("detected_misconception")
        concept_name = diagnosis_result.get("concept_name", "Bài học")
        cot_explanation = diagnosis_result.get("cot_explanation", "")
        correct_option = diagnosis_result.get("correct_option", "")
        selected_option = diagnosis_result.get("selected_option", "")
        question_text = diagnosis_result.get("question_text", "")
        options = diagnosis_result.get("options", {})

        # Determine graduated hinting level scoped strictly to (question_id, misconception)
        scaffolding_level, turn_index = self._determine_scaffolding_level(
            history, current_misconception=detected_misc, current_question_id=question_id
        )

        # Generate Scaffolding Response via Tutor Engine
        tutor_response, used_llm = self.engine.generate_scaffolding_response(
            student_name=student_name,
            student_query=student_query,
            concept_name=concept_name,
            question_text=question_text,
            options=options,
            correct_option=correct_option,
            selected_option=selected_option,
            detected_misconception=detected_misc,
            cot_explanation=cot_explanation,
            scaffolding_level=scaffolding_level,
            interaction_history=history,
        )

        # Log dialogue into learner state history if state is present
        if learner_state:
            learner_state.add_interaction(
                role="user",
                message=student_query,
                agent_name="User",
                metadata={"turn_index": turn_index, "question_id": question_id, "misconception": detected_misc}
            )
            learner_state.add_interaction(
                role="assistant",
                message=tutor_response,
                agent_name=self.name,
                metadata={
                    "scaffolding_level": scaffolding_level,
                    "turn_index": turn_index,
                    "question_id": question_id,
                    "misconception": detected_misc,
                    "used_llm": used_llm,
                }
            )

        return {
            "student_query": student_query,
            "tutor_response": tutor_response,
            "scaffolding_level": scaffolding_level,
            "scaffolding_technique": f"Graduated Hinting ({scaffolding_level.capitalize()})",
            "turn_index": turn_index,
            "used_llm": used_llm,
            "next_suggested_action": "Student response or step execution"
        }
