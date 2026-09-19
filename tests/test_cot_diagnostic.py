"""
Unit tests for Local CoT Diagnostic Engine & Guardrail Validation.
"""

import json
import unittest
from typing import Dict, Any, List

from pydantic import ValidationError

from src.agents.cot_diagnostic_engine import (
    DiagnosticOutputSchema,
    LocalCoTDiagnosticEngine,
    UNLABELED_KEY,
)
from src.agents.diagnostic_agent import DiagnosticAgent
from src.core.learner_state import LearnerState


def compute_macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    """Computes Macro-F1 score for misconception classification."""
    classes = set(y_true).union(set(y_pred))
    if not classes:
        return 0.0

    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == cls and pr == cls)
        fp = sum(1 for gt, pr in zip(y_true, y_pred) if gt != cls and pr == cls)
        fn = sum(1 for gt, pr in zip(y_true, y_pred) if gt == cls and pr != cls)

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1

    return f1_sum / len(classes)


class TestLocalCoTDiagnosticEngine(unittest.TestCase):
    def setUp(self):
        self.engine = LocalCoTDiagnosticEngine(
            model_name="qwen2.5:7b-instruct",
            seed=42,
            temperature=0.1,
            max_retries=3,
            enable_ollama_fallback=True
        )
        self.agent = DiagnosticAgent(cot_engine=self.engine)
        self.sample_question = {
            "question_id": "EEDI_TEST_1",
            "concept_id": "EEDI_CONSTRUCT_100",
            "concept_name": "Phép cộng phân số",
            "question_text": "Tính 1/4 + 1/4",
            "options": {"A": "2/8", "B": "2/4", "C": "1/8", "D": "1/16"},
            "correct_option": "B",
            "misconception_map": {
                "A": {
                    "misconception_id": "1001",
                    "name": "Cộng cả tử và mẫu số",
                    "description": "Học sinh cộng cả mẫu số khi cộng hai phân số cùng mẫu.",
                    "severity": "medium"
                }
            }
        }

    def test_pydantic_schema_validation(self):
        valid_data = {
            "misconception_id": "1001",
            "cot_reasoning": "Học sinh cộng cả mẫu số.",
            "confidence_score": 0.9
        }
        schema = DiagnosticOutputSchema(**valid_data)
        self.assertEqual(schema.misconception_id, "1001")
        self.assertEqual(schema.confidence_score, 0.9)

        invalid_data = {
            "misconception_id": "1001",
            "cot_reasoning": "Missing confidence score",
            "confidence_score": 1.5  # Invalid, must be <= 1.0
        }
        with self.assertRaises(ValidationError):
            DiagnosticOutputSchema(**invalid_data)

    def test_json_extraction_markdown_fence(self):
        raw = """Here is the result:
```json
{
  "misconception_id": "1001",
  "cot_reasoning": "Test CoT",
  "confidence_score": 0.85
}
```
Done."""
        cleaned = self.engine._clean_and_extract_json(raw)
        data = json.loads(cleaned)
        schema = DiagnosticOutputSchema(**data)
        self.assertEqual(schema.misconception_id, "1001")

    def test_few_shot_prompt_generation(self):
        prompt = self.engine.build_few_shot_prompt(self.sample_question, "A")
        self.assertIn("1/4 + 1/4", prompt)
        self.assertIn("1001", prompt)
        self.assertIn("cot_reasoning", prompt)

    def test_engine_diagnosis_flow(self):
        schema_out, is_valid_parse, attempts = self.engine.diagnose(self.sample_question, "A")
        self.assertTrue(is_valid_parse)
        self.assertEqual(schema_out.misconception_id, "1001")
        self.assertGreater(len(schema_out.cot_reasoning), 10)
        self.assertGreaterEqual(attempts, 1)

    def test_diagnostic_agent_llm_mode(self):
        state = LearnerState("TEST_STUDENT", "Student Test")
        res = self.agent.process(
            {"question": self.sample_question, "selected_option": "A", "use_llm": True},
            {"learner_state": state}
        )
        self.assertFalse(res["is_correct"])
        self.assertEqual(res["engine"], "llm_cot")
        self.assertEqual(res["misconception_id"], "1001")
        self.assertTrue(res["is_valid_parse"])
        self.assertIn("cot_explanation", res)

    def test_diagnostic_agent_correct_answer(self):
        state = LearnerState("TEST_STUDENT", "Student Test")
        res = self.agent.process(
            {"question": self.sample_question, "selected_option": "B", "use_llm": True},
            {"learner_state": state}
        )
        self.assertTrue(res["is_correct"])
        self.assertIsNone(res["detected_misconception"])

    def test_macro_f1_calculation(self):
        y_true = ["101", "102", "101", "unlabeled"]
        y_pred = ["101", "102", "101", "unlabeled"]
        score = compute_macro_f1(y_true, y_pred)
        self.assertEqual(score, 1.0)

        y_pred_imperfect = ["101", "101", "101", "unlabeled"]
        score_imp = compute_macro_f1(y_true, y_pred_imperfect)
        self.assertLess(score_imp, 1.0)
        self.assertGreater(score_imp, 0.0)


if __name__ == "__main__":
    unittest.main()
