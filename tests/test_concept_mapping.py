"""Unit tests for Eedi->Junyi Concept Mapping and Gold Set Benchmark (SE-03)."""

import json
from pathlib import Path
import unittest

from src.data.dataset_loaders import JunyiGraphLoader
from src.data.concept_mapping import (
    EediJunyiMapper,
    SemanticEmbeddingMapper,
    HybridConceptMapper,
    ConceptMapping
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_SET_PATH = ROOT / 'data' / 'gold_concept_mapping.json'


class TestConceptMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.junyi_graph = JunyiGraphLoader.load_math_prerequisite_graph()
        assert GOLD_SET_PATH.is_file(), f"Gold set file missing: {GOLD_SET_PATH}"
        with GOLD_SET_PATH.open('r', encoding='utf-8') as f:
            cls.gold_dataset = json.load(f)

    def test_gold_set_validity_and_schema(self):
        """Verify Gold Set benchmark contains 100-200 valid records matching schema."""
        self.assertGreaterEqual(len(self.gold_dataset), 100)
        self.assertLessEqual(len(self.gold_dataset), 200)

        required_keys = {
            "construct_id", "eedi_concept_id", "eedi_concept_name",
            "eedi_subject", "gold_junyi_node_id", "gold_junyi_node_name",
            "acceptable_junyi_node_ids", "is_mapped"
        }

        for record in self.gold_dataset:
            self.assertTrue(required_keys.issubset(record.keys()))
            if record["is_mapped"]:
                self.assertIsNotNone(record["gold_junyi_node_id"])
                self.assertIsNotNone(record["gold_junyi_node_name"])
            else:
                self.assertIsNone(record["gold_junyi_node_id"])

    def test_baseline_heuristic_mapper_no_hallucinations(self):
        """Ensure baseline Lexicon mapper never invents node IDs outside Junyi Graph."""
        mapper = EediJunyiMapper(self.junyi_graph)
        sample_q = {
            'concept_id': 'EEDI_CONSTRUCT_856',
            'concept_name': 'Use the order of operations to carry out calculations involving powers (BIDMAS)',
            'subject': 'Number'
        }
        result = mapper.map_question(sample_q)
        self.assertIsInstance(result, ConceptMapping)
        if result.mapped:
            self.assertIn(result.junyi_concept_id, self.junyi_graph.nodes)

    def test_semantic_embedding_mapper_validity(self):
        """Ensure SemanticEmbeddingMapper returns valid mapping with top3 candidate lists."""
        mapper = SemanticEmbeddingMapper(self.junyi_graph)
        sample_q = {
            'concept_id': 'EEDI_CONSTRUCT_1612',
            'concept_name': 'Simplify an algebraic fraction by factorising the numerator',
            'subject': 'Algebra'
        }
        result = mapper.map_question(sample_q)
        self.assertIsInstance(result, ConceptMapping)
        if result.mapped:
            self.assertIn(result.junyi_concept_id, self.junyi_graph.nodes)
            self.assertIsNotNone(result.top3_junyi_ids)
            self.assertGreaterEqual(len(result.top3_junyi_ids), 1)
            for candidate_id in result.top3_junyi_ids:
                self.assertIn(candidate_id, self.junyi_graph.nodes)

    def test_hybrid_concept_mapper_validity(self):
        """Ensure HybridConceptMapper integrates lexicon & semantic mapping without invalid IDs."""
        mapper = HybridConceptMapper(self.junyi_graph)
        sample_q = {
            'concept_id': 'EEDI_CONSTRUCT_3387',
            'concept_name': 'Substitute positive integer values into formulae involving powers or roots',
            'subject': 'Algebra'
        }
        result = mapper.map_question(sample_q)
        self.assertIsInstance(result, ConceptMapping)
        self.assertTrue(result.mapped)
        self.assertIn(result.junyi_concept_id, self.junyi_graph.nodes)
        self.assertTrue(result.method.startswith('hybrid_'))

    def test_zero_hallucinated_ids_across_gold_set(self):
        """Critical acceptance criterion: 0% hallucinated/invalid Junyi IDs on Gold Set."""
        h_mapper = HybridConceptMapper(self.junyi_graph)
        invalid_ids = 0

        for r in self.gold_dataset:
            q = {
                'concept_id': r['eedi_concept_id'],
                'concept_name': r['eedi_concept_name'],
                'subject': r['eedi_subject']
            }
            res = h_mapper.map_question(q)
            if res.mapped and res.junyi_concept_id not in self.junyi_graph.nodes:
                invalid_ids += 1

        self.assertEqual(invalid_ids, 0)


if __name__ == '__main__':
    unittest.main()
