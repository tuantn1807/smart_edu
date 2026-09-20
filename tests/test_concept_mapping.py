"""Unit tests for Eedi->Junyi Concept Mapping and Gold Set Benchmark (SE-03)."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluate import evaluate_gold_set_mapping
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
        """Verify Gold Set benchmark contains 100-200 valid records matching schema and expert rationale."""
        self.assertGreaterEqual(len(self.gold_dataset), 100)
        self.assertLessEqual(len(self.gold_dataset), 200)

        required_keys = {
            "construct_id", "eedi_concept_id", "eedi_concept_name",
            "eedi_subject", "gold_junyi_node_id", "gold_junyi_node_name",
            "acceptable_junyi_node_ids", "is_mapped", "annotation_status", "notes"
        }

        for record in self.gold_dataset:
            self.assertTrue(required_keys.issubset(record.keys()))
            self.assertEqual(record["annotation_status"], "expert_verified")
            self.assertTrue(record["notes"].startswith("Expert verified"))
            if record["is_mapped"]:
                self.assertIsNotNone(record["gold_junyi_node_id"])
                self.assertIsNotNone(record["gold_junyi_node_name"])
                self.assertIn(record["gold_junyi_node_id"], self.junyi_graph.nodes)
            else:
                self.assertIsNone(record["gold_junyi_node_id"])

            for acc_id in record.get("acceptable_junyi_node_ids", []):
                self.assertIn(acc_id, self.junyi_graph.nodes)
                self.assertFalse(acc_id.startswith("JUNYI_LEVEL"))

    def test_baseline_heuristic_mapper_no_hallucinations(self):
        """Ensure baseline Lexicon mapper returns valid top3 and never invents node IDs outside Junyi Graph."""
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
            self.assertIsNotNone(result.top3_junyi_ids)
            for cand_id in result.top3_junyi_ids:
                self.assertIn(cand_id, self.junyi_graph.nodes)

    def test_semantic_embedding_mapper_validity(self):
        """Ensure SemanticEmbeddingMapper returns valid mapping with top3 candidate lists."""
        mapper = SemanticEmbeddingMapper(self.junyi_graph, require_model=True)
        self.assertIsNotNone(mapper.model, "SentenceTransformer model must be loaded for semantic mapper.")
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

    def test_semantic_mapper_requires_real_model(self):
        """Verify SemanticEmbeddingMapper raises RuntimeError when model is unavailable in strict mode."""
        with self.assertRaises(RuntimeError):
            SemanticEmbeddingMapper(self.junyi_graph, model_name="non_existent_invalid_model_12345", require_model=True)

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
        for cand_id in result.top3_junyi_ids or []:
            self.assertIn(cand_id, self.junyi_graph.nodes)

    def test_zero_hallucinated_ids_across_gold_set(self):
        """Critical acceptance criterion: 0% hallucinated/invalid Junyi IDs on Gold Set across all top3 candidates."""
        h_mapper = HybridConceptMapper(self.junyi_graph)
        invalid_ids = 0

        for r in self.gold_dataset:
            q = {
                'concept_id': r['eedi_concept_id'],
                'concept_name': r['eedi_concept_name'],
                'subject': r['eedi_subject']
            }
            res = h_mapper.map_question(q)
            suggested_ids = set()
            if res.mapped and res.junyi_concept_id:
                suggested_ids.add(res.junyi_concept_id)
            if res.mapped and res.top3_junyi_ids:
                suggested_ids.update(res.top3_junyi_ids)

            for nid in suggested_ids:
                if nid not in self.junyi_graph.nodes:
                    invalid_ids += 1

        self.assertEqual(invalid_ids, 0)

    def test_known_construct_mapping_fixtures(self):
        """Verify construct 856 (BIDMAS) maps to exact gold target node."""
        b_mapper = EediJunyiMapper(self.junyi_graph)
        sample_bidmas = {
            'concept_id': 'EEDI_CONSTRUCT_856',
            'concept_name': 'Use the order of operations to carry out calculations involving powers (BIDMAS)',
            'subject': 'Number'
        }
        res_bidmas = b_mapper.map_question(sample_bidmas)
        self.assertTrue(res_bidmas.mapped)
        self.assertEqual(res_bidmas.junyi_concept_id, "JUNYI_OByIaYY/56y05bpjY1K3Dg+OS32Y/fB3yVjKfBqFIIY=")

    def test_unmapped_construct_detection(self):
        """Ensure constructs with no matching concepts return unmapped status."""
        mapper = SemanticEmbeddingMapper(self.junyi_graph)
        unmapped_q = {
            'concept_id': 'EEDI_UNKNOWN_99999',
            'concept_name': 'Quantum cryptography and advanced astrophysics analysis',
            'subject': 'Unrelated Non-K12 Topic'
        }
        res = mapper.map_question(unmapped_q)
        self.assertFalse(res.mapped)
        self.assertIsNone(res.junyi_concept_id)

    def test_graph_fingerprint_includes_node_content(self):
        """Verify graph fingerprint changes when node title or description changes."""
        mapper = SemanticEmbeddingMapper(self.junyi_graph)
        original_fp = mapper._graph_fingerprint
        self.assertEqual(len(original_fp), 16)

        # Modify a non-level node title in a copy of the graph
        non_level_id = next(nid for nid in self.junyi_graph.nodes if not nid.startswith('JUNYI_LEVEL'))
        original_name = self.junyi_graph.nodes[non_level_id].name
        self.junyi_graph.nodes[non_level_id].name = original_name + "_MODIFIED_FINGERPRINT_TEST"

        new_fp = mapper._compute_graph_fingerprint()
        self.assertNotEqual(original_fp, new_fp)

        # Restore original node name
        self.junyi_graph.nodes[non_level_id].name = original_name

    def test_gold_set_mapping_latency(self):
        """Verify cold start latency is under 30s and warm-cache evaluation latency is under 5.0s."""
        # Warm query cache
        evaluate_gold_set_mapping(GOLD_SET_PATH, self.junyi_graph)
        result = evaluate_gold_set_mapping(GOLD_SET_PATH, self.junyi_graph)
        self.assertIn("latency_seconds", result)
        self.assertIn("cold_start_latency_seconds", result)
        self.assertLess(result["cold_start_latency_seconds"], 30.0)
        self.assertLess(result["latency_seconds"], 5.0)
        self.assertEqual(result["invalid_junyi_ids"], 0)


if __name__ == '__main__':
    unittest.main()
