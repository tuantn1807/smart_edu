"""
Unit tests for Local CoT Diagnostic Engine & Guardrail Validation.
"""

import json
import unittest
from unittest.mock import patch
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
            api_base="http://localhost:11434",
            backend="ollama",
            diagnosis_mode="independent",
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

    def test_local_endpoint_validation(self):
        with self.assertRaises(ValueError):
            LocalCoTDiagnosticEngine(api_base="http://external-server.com:11434")

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
            "confidence_score": 1.5
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

    def test_few_shot_prompt_generation_independent_mode(self):
        prompt = self.engine.build_few_shot_prompt(self.sample_question, "A")
        self.assertIn("1/4 + 1/4", prompt)
        self.assertNotIn("Gợi ý nhãn rubric Eedi", prompt)
        self.assertNotIn("ID='1001'", prompt)

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnose_success_first_try(self, mock_backend):
        mock_response = json.dumps({
            "misconception_id": "1001",
            "cot_reasoning": "1. Quan sát: A. 2. Phân tích: cộng mẫu. 3. Kết luận: 1001",
            "confidence_score": 0.95
        })
        mock_backend.return_value = (mock_response, None)

        schema_out, is_valid_parse, attempts, used_fallback, err_reason = self.engine.diagnose(self.sample_question, "A")
        self.assertTrue(is_valid_parse)
        self.assertFalse(used_fallback)
        self.assertIsNone(err_reason)
        self.assertEqual(attempts, 1)
        self.assertEqual(schema_out.misconception_id, "1001")

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnose_retry_success(self, mock_backend):
        invalid_resp = ("Invalid JSON without braces", None)
        valid_resp = (json.dumps({
            "misconception_id": "1001",
            "cot_reasoning": "Fixed JSON after retry",
            "confidence_score": 0.90
        }), None)
        mock_backend.side_effect = [invalid_resp, valid_resp]

        schema_out, is_valid_parse, attempts, used_fallback, err_reason = self.engine.diagnose(self.sample_question, "A")
        self.assertTrue(is_valid_parse)
        self.assertFalse(used_fallback)
        self.assertIsNone(err_reason)
        self.assertEqual(attempts, 2)
        self.assertEqual(schema_out.misconception_id, "1001")

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnose_all_retries_failed(self, mock_backend):
        mock_backend.return_value = ("Broken JSON response", None)

        schema_out, is_valid_parse, attempts, used_fallback, err_reason = self.engine.diagnose(self.sample_question, "A")
        self.assertFalse(is_valid_parse)
        self.assertTrue(used_fallback)
        self.assertEqual(attempts, 3)
        self.assertEqual(err_reason, "RETRY_EXHAUSTED")

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnose_ollama_offline_fallback_enabled(self, mock_backend):
        mock_backend.return_value = (None, "CONNECTION_ERROR")

        schema_out, is_valid_parse, attempts, used_fallback, err_reason = self.engine.diagnose(self.sample_question, "A")
        self.assertFalse(is_valid_parse)
        self.assertTrue(used_fallback)
        self.assertEqual(attempts, 1)
        self.assertEqual(err_reason, "CONNECTION_ERROR")
        self.assertEqual(schema_out.misconception_id, "1001")

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnose_model_not_found(self, mock_backend):
        mock_backend.return_value = (None, "MODEL_NOT_FOUND")

        schema_out, is_valid_parse, attempts, used_fallback, err_reason = self.engine.diagnose(self.sample_question, "A")
        self.assertFalse(is_valid_parse)
        self.assertTrue(used_fallback)
        self.assertEqual(err_reason, "MODEL_NOT_FOUND")

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnose_fail_fast_when_fallback_disabled(self, mock_backend):
        engine_no_fallback = LocalCoTDiagnosticEngine(enable_ollama_fallback=False)
        with patch.object(engine_no_fallback, "_call_backend", return_value=(None, "MODEL_NOT_FOUND")):
            with self.assertRaises(RuntimeError):
                engine_no_fallback.diagnose(self.sample_question, "A")

    @patch.object(LocalCoTDiagnosticEngine, "_call_backend")
    def test_diagnostic_agent_llm_mode(self, mock_backend):
        mock_backend.return_value = (json.dumps({
            "misconception_id": "1001",
            "cot_reasoning": "Detailed CoT description",
            "confidence_score": 0.95
        }), None)
        state = LearnerState("TEST_STUDENT", "Student Test")
        res = self.agent.process(
            {"question": self.sample_question, "selected_option": "A", "use_llm": True},
            {"learner_state": state}
        )
        self.assertFalse(res["is_correct"])
        self.assertEqual(res["engine"], "llm_cot")
        self.assertEqual(res["misconception_id"], "1001")
        self.assertTrue(res["is_valid_parse"])
        self.assertFalse(res["used_fallback"])

    def test_diagnostic_agent_correct_answer(self):
        state = LearnerState("TEST_STUDENT", "Student Test")
        res = self.agent.process(
            {"question": self.sample_question, "selected_option": "B", "use_llm": True},
            {"learner_state": state}
        )
        self.assertTrue(res["is_correct"])
        self.assertIsNone(res["detected_misconception"])
        self.assertEqual(res["misconception_id"], "unlabeled")

    def test_macro_f1_calculation(self):
        y_true = ["101", "102", "101", "unlabeled"]
        y_pred = ["101", "102", "101", "unlabeled"]
        score = compute_macro_f1(y_true, y_pred)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
