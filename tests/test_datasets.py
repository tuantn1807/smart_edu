import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import dataset_loaders as loaders
from src.data.concept_mapping import EediJunyiMapper
from src.orchestrator.paaf_framework import PAAFFramework


class RealDatasetTests(unittest.TestCase):
    def test_download_integrity(self):
        manifest = json.loads((loaders.DATA_DIR / 'sources.json').read_text())
        for name, metadata in manifest['files'].items():
            content = (loaders.DATA_DIR / 'raw' / 'eedi' / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), metadata['sha256'])

    def test_full_dataset_and_known_record(self):
        questions = loaders.EediDatasetLoader.load_questions()
        self.assertEqual(len(questions), 1869)
        self.assertEqual(len({q['question_id'] for q in questions}), 1869)
        first = loaders.EediDatasetLoader.get_question_by_id('EEDI_0')
        self.assertEqual(first['concept_id'], 'EEDI_CONSTRUCT_856')
        self.assertEqual(first['correct_option'], 'A')
        self.assertEqual(first['misconception_map']['D']['misconception_id'], '1672')
        self.assertNotIn('B', first['misconception_map'])
        for question in questions:
            self.assertEqual(set(question['options']), set('ABCD'))
            self.assertNotIn(question['correct_option'], question['misconception_map'])
            self.assertTrue(question['question_text'])

    def test_junyi_graph_has_hierarchy_edges(self):
        graph = loaders.JunyiGraphLoader.load_math_prerequisite_graph()
        self.assertTrue(graph.prerequisites_available)
        self.assertGreater(graph.edge_count(), 0)
        self.assertGreater(len(graph.nodes), 1330)
        sample = next(node for node in graph.nodes.values() if node.concept_id.startswith('JUNYI_') and not node.concept_id.startswith('JUNYI_LEVEL'))
        ancestors = graph.get_all_ancestors(sample.concept_id)
        self.assertGreater(len(ancestors), 0)
        for ancestor_id in ancestors:
            self.assertIn(ancestor_id, graph.nodes)

    def test_mapping_only_returns_existing_junyi_ids(self):
        questions = loaders.EediDatasetLoader.load_questions()
        graph = loaders.JunyiGraphLoader.load_math_prerequisite_graph()
        mapper = EediJunyiMapper(graph)
        mapped = 0
        for question in questions:
            result = mapper.map_question(question)
            if result.mapped:
                mapped += 1
                self.assertIn(result.junyi_concept_id, graph.nodes)
                self.assertTrue(result.junyi_concept_id.startswith('JUNYI_'))
            else:
                self.assertIsNone(result.junyi_concept_id)
        self.assertGreater(mapped, 0)
        first = mapper.map_question(loaders.EediDatasetLoader.get_question_by_id('EEDI_0'))
        self.assertTrue(first.mapped)

    def test_missing_data_and_unknown_id_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(loaders, 'DATA_DIR', Path(directory)):
                with patch.object(loaders, 'JUNYI_EXERCISE_TABLE', Path(directory) / 'missing_table.csv'):
                    with patch.object(loaders, 'JUNYI_INFO_CONTENT', Path(directory) / 'missing_info.csv'):
                        with self.assertRaises(FileNotFoundError):
                            loaders.EediDatasetLoader.load_questions()
                        with self.assertRaises(FileNotFoundError):
                            loaders.JunyiGraphLoader.load_math_prerequisite_graph()
        with self.assertRaises(KeyError):
            loaders.EediDatasetLoader.get_question_by_id('EEDI_Q101')

    def test_pipeline_uses_eedi_diagnosis_and_junyi_graph(self):
        graph = loaders.JunyiGraphLoader.load_math_prerequisite_graph()
        framework = PAAFFramework(graph)
        question = loaders.EediDatasetLoader.get_question_by_id('EEDI_0')
        for answer in ['A', 'D', 'B']:
            with contextlib.redirect_stdout(io.StringIO()):
                result = framework.run_full_pipeline('TEST', 'Test', question, answer)
            self.assertEqual(result['diagnosis_result']['is_correct'], answer == 'A')
            self.assertTrue(result['kg_analysis']['mapping_available'])
            self.assertTrue(result['kg_analysis']['prerequisites_available'])
            self.assertGreater(len(result['kg_analysis']['unmastered_prerequisites']), 0)
            self.assertIn(result['kg_analysis']['target_concept_id'], graph.nodes)
            if answer != 'A':
                self.assertEqual(result['diagnosis_result']['severity'], 'unknown')
        self.assertGreater(len(framework.knowledge_graph.nodes), len({
            q['concept_id'] for q in loaders.EediDatasetLoader.load_questions()}))

    def test_unmapped_construct_does_not_invent_prerequisites(self):
        graph = loaders.JunyiGraphLoader.load_math_prerequisite_graph()
        framework = PAAFFramework(graph)
        question = loaders.EediDatasetLoader.get_question_by_id('EEDI_0')
        question = dict(question)
        question['concept_id'] = 'EEDI_CONSTRUCT_UNMAPPED'
        question['concept_name'] = 'Unmapped synthetic construct xyz'
        question['subject'] = 'UnmappedSubjectXYZ'
        with contextlib.redirect_stdout(io.StringIO()):
            result = framework.run_full_pipeline('TEST', 'Test', question, 'A')
        self.assertFalse(result['kg_analysis']['mapping_available'])
        self.assertEqual(result['kg_analysis']['unmastered_prerequisites'], [])


if __name__ == '__main__':
    unittest.main()
