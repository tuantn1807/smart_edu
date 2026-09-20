"""
Unit tests for SE-04: Dynamic Graph Traversal & Session Mastery Propagation.
"""

import unittest
from src.core.knowledge_graph import KnowledgeGraph
from src.core.learner_state import LearnerState
from src.agents.kg_agent import KGAgent
from src.orchestrator.paaf_framework import PAAFFramework


class TestKGMasteryPropagation(unittest.TestCase):
    def setUp(self):
        # Create a sample Junyi graph for testing
        # Node structure:
        # A (foundational, diff=0.1) -> B (intermediate, diff=0.4) -> C (target, diff=0.7)
        self.kg = KnowledgeGraph(subject_name="Toán", source="Junyi_Test", graph_kind="Junyi_Hierarchy")
        self.kg.add_concept("node_A", "Phép cộng cơ bản", difficulty=0.1, description="Cộng các số tự nhiên")
        self.kg.add_concept("node_B", "Quy đồng mẫu số", difficulty=0.4, description="Tìm mẫu số chung", prerequisites=["node_A"])
        self.kg.add_concept("node_C", "Cộng phân số", difficulty=0.7, description="Phép cộng phân số khác mẫu", prerequisites=["node_B"])

    def test_learner_state_update_and_propagation(self):
        state = LearnerState("S001", "Học sinh A")
        
        # Initial state: default mastery is 0.0
        self.assertEqual(state.get_concept_mastery("node_C"), 0.0)
        
        # Test update_mastery correct (+0.2)
        score = state.update_mastery("node_C", is_correct=True, delta_correct=0.2)
        self.assertAlmostEqual(score, 0.2)
        self.assertAlmostEqual(state.get_concept_mastery("node_C"), 0.2)

        # Test update_mastery incorrect (-0.3, floor at 0.0)
        score = state.update_mastery("node_C", is_correct=False, delta_incorrect=0.3)
        self.assertAlmostEqual(score, 0.0)

        # Set initial mastery for ancestors
        state.set_concept_mastery("node_A", 0.8)
        state.set_concept_mastery("node_B", 0.5)

        # Propagate loss from node_C
        updated = state.propagate_mastery_loss("node_C", self.kg, attenuation=0.1)
        self.assertIn("node_A", updated)
        self.assertIn("node_B", updated)
        self.assertAlmostEqual(state.get_concept_mastery("node_A"), 0.7)
        self.assertAlmostEqual(state.get_concept_mastery("node_B"), 0.4)

    def test_dynamic_kg_agent_traversal(self):
        agent = KGAgent(knowledge_graph=self.kg)
        state = LearnerState("S002", "Học sinh B")
        context = {"learner_state": state}

        # Lượt 1: Mastery ban đầu 0.0 -> cả node_A và node_B đều unmastered
        res1 = agent.process({"target_concept_id": "node_C", "mapping": {"mapped": True}}, context)
        unmastered_ids_1 = [n["concept_id"] for n in res1["unmastered_prerequisites"]]
        self.assertEqual(len(unmastered_ids_1), 2)
        self.assertIn("node_A", unmastered_ids_1)
        self.assertIn("node_B", unmastered_ids_1)

        # Lượt 2: Học sinh đạt mastery node_A = 0.8 (vượt ngưỡng 0.6)
        state.set_concept_mastery("node_A", 0.8)
        res2 = agent.process({"target_concept_id": "node_C", "mapping": {"mapped": True}}, context)
        unmastered_ids_2 = [n["concept_id"] for n in res2["unmastered_prerequisites"]]
        self.assertEqual(len(unmastered_ids_2), 1)
        self.assertNotIn("node_A", unmastered_ids_2)
        self.assertIn("node_B", unmastered_ids_2)

        # Lượt 3: Học sinh thành thạo cả node_B (mastery = 0.7)
        state.set_concept_mastery("node_B", 0.7)
        res3 = agent.process({"target_concept_id": "node_C", "mapping": {"mapped": True}}, context)
        unmastered_ids_3 = [n["concept_id"] for n in res3["unmastered_prerequisites"]]
        self.assertEqual(len(unmastered_ids_3), 0)

    def test_unmapped_concept_does_not_invent_edges(self):
        agent = KGAgent(knowledge_graph=self.kg)
        state = LearnerState("S003", "Học sinh C")
        context = {"learner_state": state}

        # Concept không tồn tại trong Junyi KG & unmapped
        res = agent.process({"target_concept_id": "NON_EXISTENT_NODE", "mapping": {"mapped": False}}, context)
        self.assertFalse(res["mapping_available"])
        self.assertEqual(res["all_prerequisites_count"], 0)
        self.assertEqual(len(res["unmastered_prerequisites"]), 0)
        self.assertIn("không duyệt tiên quyết giả", res["analysis_summary"])

    def test_paaf_multi_turn_session_mastery_propagation(self):
        framework = PAAFFramework(knowledge_graph=self.kg)
        question = {
            "question_id": "Q_TEST_1",
            "concept_id": "node_C",
            "options": ["A", "B", "C", "D"],
            "correct_option": "A"
        }

        # Session 1: Trả lời sai
        state = LearnerState("S004", "Học sinh D")
        res1 = framework.run_full_pipeline("S004", "Học sinh D", question, selected_option="B", learner_state=state)
        kg_res1 = res1["kg_analysis"]
        self.assertTrue(kg_res1["prerequisites_available"])
        self.assertGreater(len(kg_res1["unmastered_prerequisites"]), 0)
        self.assertIn("graph_kind", kg_res1)
        self.assertIn("graph_source", kg_res1)


if __name__ == "__main__":
    unittest.main()
