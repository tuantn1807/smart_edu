"""
Unit tests for Interactive Scaffolding Tutor Loop (SE-02).
Tests Graduated Hinting (Nudge -> Hint -> Explanation), answer leakage guardrail,
and multi-turn context retention.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.agents.tutor_agent import TutorAgent
from src.agents.tutor_engine import (
    LocalScaffoldingTutorEngine,
    LEVEL_NUDGE,
    LEVEL_HINT,
    LEVEL_EXPLANATION,
)
from src.core.learner_state import LearnerState


class TestTutorScaffolding(unittest.TestCase):
    def setUp(self):
        self.tutor = TutorAgent(enable_ollama_fallback=True)
        self.learner_state = LearnerState(student_id="STU_100", student_name="Nguyễn Văn A")
        self.context = {"learner_state": self.learner_state}
        self.diagnosis = {
            "concept_id": "C_MATH_01",
            "concept_name": "Cộng phân số khác mẫu",
            "detected_misconception": "Cộng trực tiếp tử số và mẫu số không quy đồng",
            "selected_option": "A",
            "correct_option": "B",
            "question_text": "Tính 1/2 + 1/3.",
            "options": {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"},
            "cot_explanation": "Học sinh đã cộng tử với tử (1+1=2) và mẫu với mẫu (2+3=5)."
        }
        self.planner_result = {
            "learning_path": [
                {"step_id": 1, "concept": "Quy đồng mẫu số", "action_type": "review_prerequisite"}
            ]
        }

    def test_graduated_hinting_3_levels(self):
        """Verify Turn 1 -> Nudge, Turn 2 -> Hint, Turn 3 -> Explanation."""
        # Turn 1
        res1 = self.tutor.process(
            {
                "student_query": "Em chưa hiểu vì sao lại sai ạ.",
                "diagnosis_result": self.diagnosis,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        self.assertEqual(res1["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res1["turn_index"], 1)
        self.assertIn("Nudge", res1["scaffolding_technique"])

        # Turn 2
        res2 = self.tutor.process(
            {
                "student_query": "Em biến đổi thế nào tiếp theo?",
                "diagnosis_result": self.diagnosis,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        self.assertEqual(res2["scaffolding_level"], LEVEL_HINT)
        self.assertEqual(res2["turn_index"], 2)
        self.assertIn("Hint", res2["scaffolding_technique"])

        # Turn 3
        res3 = self.tutor.process(
            {
                "student_query": "Thầy giải thích rõ bản chất giúp em.",
                "diagnosis_result": self.diagnosis,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        self.assertEqual(res3["scaffolding_level"], LEVEL_EXPLANATION)
        self.assertEqual(res3["turn_index"], 3)
        self.assertIn("Explanation", res3["scaffolding_technique"])

    def test_turn2_more_specific_than_turn1(self):
        """Verify Turn 2 (Hint) provides more specific misconception information than Turn 1 (Nudge)."""
        res1 = self.tutor.process(
            {
                "student_query": "Giải thích giúp em.",
                "diagnosis_result": self.diagnosis,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        res2 = self.tutor.process(
            {
                "student_query": "Cho em gợi ý cụ thể hơn đi ạ.",
                "diagnosis_result": self.diagnosis,
                "planner_result": self.planner_result,
            },
            self.context,
        )

        # Level 2 hint should mention the specific detected misconception or step hint
        self.assertEqual(res1["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res2["scaffolding_level"], LEVEL_HINT)
        # Check text specificity or presence of misconception name in turn 2
        self.assertIn(self.diagnosis["detected_misconception"], res2["tutor_response"])

    def test_multi_turn_context_retention(self):
        """Verify dialogue context is preserved across 3+ turns in LearnerState."""
        for i in range(1, 4):
            self.tutor.process(
                {
                    "student_query": f"Thắc mắc lượt {i}",
                    "diagnosis_result": self.diagnosis,
                    "planner_result": self.planner_result,
                },
                self.context,
            )

        # 3 user turns + 3 assistant turns = 6 interactions logged
        history = self.learner_state.interaction_history
        self.assertEqual(len(history), 6)

        user_msgs = [h for h in history if h["role"] == "user"]
        assistant_msgs = [h for h in history if h["role"] == "assistant"]

        self.assertEqual(len(user_msgs), 3)
        self.assertEqual(len(assistant_msgs), 3)
        self.assertEqual(assistant_msgs[0]["metadata"]["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(assistant_msgs[1]["metadata"]["scaffolding_level"], LEVEL_HINT)
        self.assertEqual(assistant_msgs[2]["metadata"]["scaffolding_level"], LEVEL_EXPLANATION)

    def test_answer_leakage_guardrail(self):
        """Verify guardrail sanitizes responses that leak correct option text or explicit answers."""
        engine = LocalScaffoldingTutorEngine()
        options = {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"}
        correct_option = "B"

        leaked_raw = "Chào em! Đáp án đúng là B (5/6). Em hãy nhớ lấy 1/2 + 1/3 = 5/6 nhé."
        sanitized = engine.sanitize_answer_leakage(leaked_raw, correct_option, options)

        # Exact correct option text '5/6' and explicit 'Đáp án đúng là B' must be sanitized
        self.assertNotIn("Đáp án đúng là B", sanitized)
        self.assertNotIn("5/6", sanitized)

    def test_online_llm_with_mock_backend(self):
        """Verify backend calling when LLM returns response."""
        mock_response = "Chào bạn Nguyễn Văn A! Thầy/cô thấy bạn đang vướng ở bài toán này. Hãy kiểm tra lại quy tắc quy đồng mẫu số nhé!"
        with patch.object(LocalScaffoldingTutorEngine, "_call_backend", return_value=(mock_response, None)):
            res = self.tutor.process(
                {
                    "student_query": "Hỗ trợ em bài này với!",
                    "diagnosis_result": self.diagnosis,
                    "planner_result": self.planner_result,
                },
                self.context,
            )
            self.assertTrue(res["used_llm"])
            self.assertIn("Nguyễn Văn A", res["tutor_response"])


if __name__ == "__main__":
    unittest.main()
