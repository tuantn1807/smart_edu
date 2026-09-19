import json
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate_all, format_report, write_report


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = evaluate_all()

    def test_report_has_required_sections(self):
        for key in ['datasets', 'diagnostic', 'mapping', 'knowledge_graph', 'planner_tutor', 'limitations']:
            self.assertIn(key, self.report)
        self.assertGreater(self.report['datasets']['eedi_questions'], 0)
        self.assertGreater(self.report['datasets']['junyi_edges'], 0)
        self.assertTrue(format_report(self.report))

    def test_diagnostic_lookup_matches_eedi_labels(self):
        diagnostic = self.report['diagnostic']
        self.assertEqual(diagnostic['correct_no_false_misconception_rate'], 1.0)
        self.assertEqual(diagnostic['labeled_wrong_exact_match_rate'], 1.0)
        self.assertEqual(diagnostic['unlabeled_uses_fallback_rate'], 1.0)
        self.assertIn('rule_based_macro_f1', diagnostic)
        self.assertIn('local_cot_llm_macro_f1', diagnostic)
        self.assertGreaterEqual(diagnostic['valid_json_parse_rate'], 0.98)
        self.assertEqual(diagnostic['model_version'], 'qwen2.5:7b-instruct')
        self.assertEqual(diagnostic['random_seed'], 42)

    def test_mapping_and_graph_integrity(self):
        mapping = self.report['mapping']
        kg = self.report['knowledge_graph']
        self.assertEqual(mapping['invalid_junyi_ids'], 0)
        self.assertGreater(mapping['mapped_constructs'], 0)
        self.assertEqual(kg['unmapped_no_invented_prerequisites_rate'], 1.0)
        self.assertGreater(kg['mapped_with_unmastered_prerequisites'], 0)

    def test_planner_does_not_invent_prereq_steps_when_unmapped(self):
        planner = self.report['planner_tutor']
        if planner['unmapped_wrong_questions']:
            self.assertEqual(planner['unmapped_wrong_no_review_prerequisite_rate'], 1.0)
        self.assertGreater(planner['mapped_wrong_with_3_phase_path'], 0)
        self.assertEqual(planner['tutor_uses_scaffolding_template_rate'], 1.0)

    def test_write_report_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            write_report(self.report, path)
            loaded = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(loaded['datasets']['eedi_questions'], self.report['datasets']['eedi_questions'])


if __name__ == '__main__':
    unittest.main()
