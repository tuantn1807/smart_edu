"""
Unit tests for Scientific Benchmark & Ablation Study suite (SE-10).
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.eval.benchmark_suite import (
    compute_path_coherence,
    compute_zpd_alignment,
    generate_markdown_table,
    generate_latex_table,
    export_benchmark_artifacts,
)


class TestAblationBenchmarkSuite(unittest.TestCase):
    def test_compute_path_coherence_valid(self):
        valid_path = [
            {"step_id": 1, "action_type": "review_prerequisite", "question_id": "Q1"},
            {"step_id": 2, "action_type": "remediate_misconception", "question_id": "Q2"},
            {"step_id": 3, "action_type": "learn_concept", "question_id": "Q3"},
        ]
        self.assertTrue(compute_path_coherence(valid_path))

    def test_compute_path_coherence_inversion(self):
        inverted_path = [
            {"step_id": 1, "action_type": "learn_concept", "question_id": "Q1"},
            {"step_id": 2, "action_type": "review_prerequisite", "question_id": "Q2"},
        ]
        self.assertFalse(compute_path_coherence(inverted_path))

    def test_compute_path_coherence_missing_qid(self):
        missing_qid_path = [
            {"step_id": 1, "action_type": "review_prerequisite", "question_id": ""},
        ]
        self.assertFalse(compute_path_coherence(missing_qid_path))

    def test_compute_zpd_alignment_prerequisite_gap(self):
        scenario = {
            "consecutive_correct": 0,
            "has_unmastered_prereqs": True,
            "has_detected_misconception": True,
        }
        aligned_path = [
            {"action_type": "review_prerequisite"},
            {"action_type": "remediate_misconception"},
            {"action_type": "learn_concept"},
        ]
        self.assertTrue(compute_zpd_alignment(scenario, aligned_path))

        unaligned_path = [
            {"action_type": "remediate_misconception"},
            {"action_type": "learn_concept"},
        ]
        self.assertFalse(compute_zpd_alignment(scenario, unaligned_path))

    def test_compute_zpd_alignment_acceleration(self):
        scenario = {
            "consecutive_correct": 2,
            "has_unmastered_prereqs": False,
            "has_detected_misconception": False,
        }
        accelerated_path = [
            {"action_type": "advanced_challenge"},
        ]
        self.assertTrue(compute_zpd_alignment(scenario, accelerated_path))

        unaccelerated_path = [
            {"action_type": "review_prerequisite"},
            {"action_type": "advanced_challenge"},
        ]
        self.assertFalse(compute_zpd_alignment(scenario, unaccelerated_path))

    def test_table_generation(self):
        dummy_results = {
            "ablation_summary": {
                "full_paaf": {
                    "diagnostic_macro_f1": 1.0,
                    "path_coherence_rate": 1.0,
                    "zpd_alignment_rate": 1.0,
                    "scaffolding_rate": 1.0,
                    "answer_protection_rate": 1.0,
                    "dynamic_mastery_propagation_rate": 1.0,
                },
                "no_kg": {
                    "diagnostic_macro_f1": 1.0,
                    "path_coherence_rate": 1.0,
                    "zpd_alignment_rate": 0.182,
                    "scaffolding_rate": 1.0,
                    "answer_protection_rate": 1.0,
                    "dynamic_mastery_propagation_rate": 0.0,
                },
            }
        }
        md = generate_markdown_table(dummy_results)
        tex = generate_latex_table(dummy_results)
        self.assertIn("Full PAAF", md)
        self.assertIn("Full PAAF", tex)
        self.assertIn("1.0000", md)
        self.assertIn("1.0000", tex)

    def test_export_benchmark_artifacts(self):
        dummy_results = {"test": True, "ablation_summary": {}}
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "eval" / "results"
            exported = export_benchmark_artifacts(dummy_results, out_dir)
            self.assertTrue(exported["ablation_study_json"].exists())
            self.assertTrue(exported["benchmark_report_json"].exists())
            self.assertTrue(exported["comparison_table_md"].exists())
            self.assertTrue(exported["comparison_table.tex" if "comparison_table.tex" in exported else "comparison_table_tex"].exists())


if __name__ == "__main__":
    unittest.main()
