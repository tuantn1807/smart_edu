"""
Unit tests for Interactive Scaffolding Tutor Loop (SE-02).
Tests Graduated Hinting (Nudge -> Hint -> Explanation), strict (question_id, misconception) scoping,
answer leakage guardrails with adversarial cases, specificity progression, and multi-turn context retention.
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
        self.diagnosis_q1_alt_misc = {
            "question_id": "Q1",
            "concept_id": "C_MATH_01",
            "concept_name": "Cộng phân số khác mẫu",
            "detected_misconception": "Quy đồng sai mẫu số chung",
            "selected_option": "D",
            "correct_option": "B",
            "question_text": "Tính 1/2 + 1/3.",
            "options": {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"},
            "cot_explanation": "Học sinh quy đồng mẫu chung là 6 nhưng nhân sai tử."
        }
        self.diagnosis_q2 = {
            "question_id": "Q2",
            "concept_id": "C_MATH_02",
            "concept_name": "Giải phương trình bậc nhất",
            "detected_misconception": "Cộng trực tiếp tử số và mẫu số không quy đồng",  # Same misc, different Q
            "selected_option": "A",
            "correct_option": "B",
            "question_text": "Giải x + 4 = 10.",
            "options": {"A": "x = 14", "B": "x = 6", "C": "x = 40", "D": "x = 2.5"},
            "cot_explanation": "Học sinh cộng 10 + 4 thay vì 10 - 4."
        }

    def test_graduated_hinting_3_levels_same_question(self):
        """Verify Turn 1 -> Nudge, Turn 2 -> Hint, Turn 3 -> Explanation on same (question_id, misconception)."""
        # Turn 1
        res1 = self.tutor.process(
            {"student_query": "Em chưa hiểu vì sao lại sai ạ.", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )
        self.assertEqual(res1["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res1["turn_index"], 1)

        # Turn 2
        res2 = self.tutor.process(
            {"student_query": "Em biến đổi thế nào tiếp theo?", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )
        self.assertEqual(res2["scaffolding_level"], LEVEL_HINT)
        self.assertEqual(res2["turn_index"], 2)

        # Turn 3
        res3 = self.tutor.process(
            {"student_query": "Thầy giải thích rõ bản chất giúp em.", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )
        self.assertEqual(res3["scaffolding_level"], LEVEL_EXPLANATION)
        self.assertEqual(res3["turn_index"], 3)

    def test_same_question_different_misconception_resets_nudge(self):
        """Verify same question but different misconception resets turn level back to Nudge (Level 1)."""
        self.tutor.process({"student_query": "Q1 M1 turn 1", "diagnosis_result": self.diagnosis_q1}, self.context)
        self.tutor.process({"student_query": "Q1 M1 turn 2", "diagnosis_result": self.diagnosis_q1}, self.context)

        # Same question Q1, but misconception changes -> Reset to Nudge (Level 1)
        res_alt = self.tutor.process(
            {"student_query": "Q1 M2 turn 1", "diagnosis_result": self.diagnosis_q1_alt_misc},
            self.context,
        )
        self.assertEqual(res_alt["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res_alt["turn_index"], 1)

    def test_different_question_same_misconception_resets_nudge(self):
        """Verify different question even with same misconception resets turn level back to Nudge (Level 1)."""
        self.tutor.process({"student_query": "Q1 M1 turn 1", "diagnosis_result": self.diagnosis_q1}, self.context)
        self.tutor.process({"student_query": "Q1 M1 turn 2", "diagnosis_result": self.diagnosis_q1}, self.context)

        # Different question Q2 -> Reset to Nudge (Level 1)
        res_q2 = self.tutor.process(
            {"student_query": "Q2 M1 turn 1", "diagnosis_result": self.diagnosis_q2},
            self.context,
        )
        self.assertEqual(res_q2["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res_q2["turn_index"], 1)

    def test_history_missing_metadata_handled_gracefully(self):
        """Verify interaction entries with missing or incomplete metadata do not cause false matching."""
        self.learner_state.add_interaction(role="user", message="Hi")
        self.learner_state.add_interaction(role="assistant", message="Hello", agent_name="TutorAgent", metadata={})

        res = self.tutor.process(
            {"student_query": "Giải thích giúp em.", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )
        self.assertEqual(res["scaffolding_level"], LEVEL_NUDGE)
        self.assertEqual(res["turn_index"], 1)

    def test_turn2_more_specific_than_turn1(self):
        """Verify Turn 2 (Hint) specificity score is strictly greater than Turn 1 (Nudge) specificity score."""
        res1 = self.tutor.process(
            {"student_query": "Giải thích giúp em.", "diagnosis_result": self.diagnosis_q1},
            self.context,
        )
        res2 = self.tutor.process(
            {"student_query": "Cho em gợi ý cụ thể hơn đi ạ.", "diagnosis_result": self.diagnosis_q1},
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

    def test_long_verbose_non_specific_response_fails_specificity(self):
        """Verify long verbose response without specific misconception clues gets lower score than targeted Hint."""
        verbose_generic = (
            "Chào bạn! Bạn hãy cố gắng lên nhé. Bài học này rất thú vị và quan trọng trong chương trình học. "
            "Nếu bạn gặp khó khăn thì đừng nản lòng, hãy xem lại lý thuyết ở phần sách giáo khoa và hỏi thầy cô nhé. "
            "Chúc bạn học tốt và luôn tự tin giải toán mỗi ngày!"
        )
        targeted_hint = "Chào bạn! Phân tích bài làm cho thấy bạn vướng ở nhãn lỗi 'Cộng trực tiếp tử số và mẫu số không quy đồng'. Hãy kiểm tra phép quy đồng."

        misc = "Cộng trực tiếp tử số và mẫu số không quy đồng"
        score_generic = compute_specificity_score(verbose_generic, misc)
        score_hint = compute_specificity_score(targeted_hint, misc)

        self.assertGreater(score_hint, score_generic)

    def test_multi_turn_prompt_context_building(self):
        """Verify build_scaffolding_prompt formats preceding user query and tutor hint into prompt context."""
        engine = LocalScaffoldingTutorEngine()

        history = [
            {"role": "user", "message": "Em chưa biết bắt đầu từ đâu"},
            {"role": "assistant", "message": "Hãy đọc kỹ lại đề bài nhé!"}
        ]

        prompt = engine.build_scaffolding_prompt(
            student_name="Nguyễn Văn A",
            student_query="Em đọc rồi nhưng chưa hiểu.",
            concept_name="Cộng phân số",
            question_text="Tính 1/2 + 1/3",
            options={"A": "2/5", "B": "5/6"},
            correct_option="B",
            selected_option="A",
            detected_misconception="Cộng tử với tử mẫu với mẫu",
            cot_explanation="CoT reasoning...",
            scaffolding_level=LEVEL_HINT,
            interaction_history=history,
        )

        self.assertIn("--- LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ ---", prompt)
        self.assertIn("Em chưa biết bắt đầu từ đâu", prompt)
        self.assertIn("Hãy đọc kỹ lại đề bài nhé!", prompt)

    def test_adversarial_answer_leakage_guardrail(self):
        """Verify guardrail sanitizes responses with LaTeX, paraphrasing, ordinal, and explicit letter leaks."""
        engine = LocalScaffoldingTutorEngine()
        options = {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"}
        correct_option = "B"

        test_cases = [
            "Đáp án đúng là B",
            "Chọn phương án B nhé",
            "Kết quả là 5/6",
            "Đáp án đúng là 5/6",
            "Key is B",
        ]

        for case in test_cases:
            sanitized = engine.sanitize_answer_leakage(case, correct_option, options)
            self.assertNotIn("Đáp án đúng là B", sanitized)
            self.assertNotIn(" 5/6", sanitized)

    def test_online_llm_with_mock_backend(self):
        """Verify backend calling when LLM returns response."""
        mock_response = "Chào bạn Nguyễn Văn A! Thầy/cô thấy bạn đang vướng ở bài toán này. Hãy kiểm tra lại quy tắc quy đồng mẫu số nhé!"
        with patch.object(LocalScaffoldingTutorEngine, "_call_backend", return_value=(mock_response, None)):
            res = self.tutor.process(
                {"student_query": "Hỗ trợ em bài này với!", "diagnosis_result": self.diagnosis_q1},
                self.context,
            )
            self.assertTrue(res["used_llm"])
            self.assertIn("Nguyễn Văn A", res["tutor_response"])


if __name__ == "__main__":
    unittest.main()
