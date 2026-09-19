"""
Unit tests for Interactive Scaffolding Tutor Loop (SE-02).
Tests Graduated Hinting (Nudge -> Hint -> Explanation), question/misconception scoping,
answer leakage guardrails, specificity progression, and multi-turn context retention.
"""

import unittest
from unittest.mock import patch

from src.agents.tutor_agent import TutorAgent
from src.agents.tutor_engine import (
    LocalScaffoldingTutorEngine,
    LEVEL_NUDGE,
    LEVEL_HINT,
    LEVEL_EXPLANATION,
    compute_specificity_score,
)
from src.core.learner_state import LearnerState


class TestTutorScaffolding(unittest.TestCase):
    def setUp(self):
        self.tutor = TutorAgent(enable_ollama_fallback=True)
        self.learner_state = LearnerState(student_id="STU_100", student_name="Nguyễn Văn A")
        self.context = {"learner_state": self.learner_state}
        self.diagnosis_q1 = {
            "question_id": "Q1",
            "concept_id": "C_MATH_01",
            "concept_name": "Cộng phân số khác mẫu",
            "detected_misconception": "Cộng trực tiếp tử số và mẫu số không quy đồng",
            "selected_option": "A",
            "correct_option": "B",
            "question_text": "Tính 1/2 + 1/3.",
            "options": {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"},
            "cot_explanation": "Học sinh đã cộng tử với tử (1+1=2) và mẫu với mẫu (2+3=5)."
        }
        self.diagnosis_q2 = {
            "question_id": "Q2",
            "concept_id": "C_MATH_02",
            "concept_name": "Giải phương trình bậc nhất",
            "detected_misconception": "Chuyển vế không đổi dấu",
            "selected_option": "A",
            "correct_option": "B",
            "question_text": "Giải x + 4 = 10.",
            "options": {"A": "x = 14", "B": "x = 6", "C": "x = 40", "D": "x = 2.5"},
            "cot_explanation": "Học sinh cộng 10 + 4 thay vì 10 - 4."
        }
        self.planner_result = {
            "learning_path": [
                {"step_id": 1, "concept": "Quy đồng mẫu số", "action_type": "review_prerequisite"}
            ]
        }

    def test_graduated_hinting_3_levels_same_question(self):
        """Verify Turn 1 -> Nudge, Turn 2 -> Hint, Turn 3 -> Explanation on same question."""
        # Turn 1
        res1 = self.tutor.process(
            {
                "student_query": "Em chưa hiểu vì sao lại sai ạ.",
                "diagnosis_result": self.diagnosis_q1,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        self.assertEqual(res1["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res1["turn_index"], 1)

        # Turn 2
        res2 = self.tutor.process(
            {
                "student_query": "Em biến đổi thế nào tiếp theo?",
                "diagnosis_result": self.diagnosis_q1,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        self.assertEqual(res2["scaffolding_level"], LEVEL_HINT)
        self.assertEqual(res2["turn_index"], 2)

        # Turn 3
        res3 = self.tutor.process(
            {
                "student_query": "Thầy giải thích rõ bản chất giúp em.",
                "diagnosis_result": self.diagnosis_q1,
                "planner_result": self.planner_result,
            },
            self.context,
        )
        self.assertEqual(res3["scaffolding_level"], LEVEL_EXPLANATION)
        self.assertEqual(res3["turn_index"], 3)

    def test_question_scoping_resets_turn_count(self):
        """Verify switching to a new question resets turn level back to Nudge (Level 1)."""
        # Question 1: Turn 1 (Nudge) & Turn 2 (Hint)
        self.tutor.process(
            {"student_query": "Lần 1 Q1", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )
        self.tutor.process(
            {"student_query": "Lần 2 Q1", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )

        # Switch to Question 2 -> Must reset to Turn 1 (Nudge)
        res_q2_t1 = self.tutor.process(
            {"student_query": "Lần 1 Q2", "diagnosis_result": self.diagnosis_q2},
            self.context,
        )
        self.assertEqual(res_q2_t1["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res_q2_t1["turn_index"], 1)

    def test_turn2_more_specific_than_turn1(self):
        """Verify Turn 2 (Hint) specificity score is strictly greater than Turn 1 (Nudge) specificity score."""
        res1 = self.tutor.process(
            {
                "student_query": "Giải thích giúp em.",
                "diagnosis_result": self.diagnosis_q1,
            },
            self.context,
        )
        res2 = self.tutor.process(
            {
                "student_query": "Cho em gợi ý cụ thể hơn đi ạ.",
                "diagnosis_result": self.diagnosis_q1,
            },
            self.context,
        )

        resp1 = res1["tutor_response"]
        resp2 = res2["tutor_response"]
        misc = self.diagnosis_q1["detected_misconception"]

        # Level 1 Nudge must NOT mention misconception name directly
        self.assertNotIn(misc, resp1)
        # Level 2 Hint MUST mention misconception name directly
        self.assertIn(misc, resp2)

        # Specificity score of Hint must be strictly higher than Nudge
        score1 = compute_specificity_score(resp1, misc)
        score2 = compute_specificity_score(resp2, misc)
        self.assertGreater(score2, score1)

    def test_multi_turn_context_retention(self):
        """Verify dialogue context is preserved across 3+ turns in LearnerState."""
        for i in range(1, 4):
            self.tutor.process(
                {
                    "student_query": f"Thắc mắc lượt {i}",
                    "diagnosis_result": self.diagnosis_q1,
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

        leaked_raw = "Chào em! Đáp án đúng là B (5/6). Em hãy nhớ kết quả là 5/6 nhé."
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
                    "diagnosis_result": self.diagnosis_q1,
                },
                self.context,
            )
            self.assertTrue(res["used_llm"])
            self.assertIn("Nguyễn Văn A", res["tutor_response"])


if __name__ == "__main__":
    unittest.main()
