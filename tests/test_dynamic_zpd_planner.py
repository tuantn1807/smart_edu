"""
Unit tests for SE-05: Dynamic ZPD Path & Real Eedi Item Selection.
"""

import unittest
from src.core.knowledge_graph import KnowledgeGraph
from src.core.learner_state import LearnerState
from src.data.dataset_loaders import EediDatasetLoader, JunyiGraphLoader
from src.data.item_repository import EediItemRepository
from src.agents.planner_agent import PlannerAgent
from src.orchestrator.paaf_framework import PAAFFramework


class TestDynamicZPDPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = EediDatasetLoader.load_questions()
        cls.item_repo = EediItemRepository(cls.questions)
        cls.kg = JunyiGraphLoader.load_math_prerequisite_graph()
        cls.all_eedi_qids = {q['question_id'] for q in cls.questions}

    def setUp(self):
        self.planner = PlannerAgent(item_repository=self.item_repo)
        self.state = LearnerState("S_ZPD_001", "Học sinh ZPD")

    def test_every_step_has_real_eedi_question_id(self):
        """Tiêu chí 1: Mỗi bước practice/remediate phải gắn với question_id Eedi tồn tại thực tế."""
        kg_analysis = {
            "mapping_available": True,
            "prerequisites_available": True,
            "unmastered_prerequisites": [
                {
                    "concept_id": "EEDI_CONSTRUCT_856",
                    "name": "Order of operations",
                    "current_mastery": 0.2,
                    "description": "Prereq test"
                }
            ]
        }
        diagnosis_result = {
            "is_correct": False,
            "detected_misconception": "Disregards order of operations",
            "misconception_id": "1672",
            "concept_name": "Order of operations with brackets"
        }
        context = {"learner_state": self.state, "item_repository": self.item_repo}

        result = self.planner.process(
            {
                "target_concept_id": "EEDI_CONSTRUCT_856",
                "kg_analysis": kg_analysis,
                "diagnosis_result": diagnosis_result
            },
            context
        )

        learning_path = result["learning_path"]
        self.assertGreater(len(learning_path), 0)

        for step in learning_path:
            qid = step.get("question_id")
            self.assertIsNotNone(qid, f"Step {step['step_id']} is missing question_id")
            self.assertIn(qid, self.all_eedi_qids, f"Question ID '{qid}' does not exist in real Eedi dataset!")
            self.assertIsNotNone(step.get("question_details"), f"Step {step['step_id']} missing question_details")
            self.assertIn("question_text", step["question_details"])
            self.assertIn("options", step["question_details"])

    def test_dynamic_replanning_route_changes_on_performance(self):
        """Tiêu chí 2: Lộ trình lần tiếp theo phải thay đổi dựa trên kết quả câu hỏi trước."""
        context = {"learner_state": self.state, "item_repository": self.item_repo}
        kg_analysis = {
            "mapping_available": True,
            "prerequisites_available": True,
            "unmastered_prerequisites": [
                {
                    "concept_id": "EEDI_CONSTRUCT_856",
                    "name": "Order of operations",
                    "current_mastery": 0.2,
                    "description": "Prereq test"
                }
            ]
        }
        diag_wrong = {
            "is_correct": False,
            "detected_misconception": "Order mistake",
            "misconception_id": "1672",
            "concept_name": "Order of operations"
        }

        # Turn 1: Student answers wrong -> consecutive_correct = 0
        self.state.record_question_result("EEDI_0", is_correct=False)
        res1 = self.planner.process(
            {"target_concept_id": "EEDI_CONSTRUCT_856", "kg_analysis": kg_analysis, "diagnosis_result": diag_wrong},
            context
        )
        path1_actions = [s["action_type"] for s in res1["learning_path"]]
        self.assertIn("review_prerequisite", path1_actions)
        self.assertIn("remediate_misconception", path1_actions)
        self.assertIn("learn_concept", path1_actions)

        # Turn 2: Student answers correct -> consecutive_correct = 1
        self.state.record_question_result("EEDI_1", is_correct=True)
        self.assertEqual(self.state.consecutive_correct, 1)

        # Turn 3: Student answers correct again -> consecutive_correct = 2
        self.state.record_question_result("EEDI_2", is_correct=True)
        self.assertEqual(self.state.consecutive_correct, 2)

        # Re-plan route after 2 consecutive correct answers
        res3 = self.planner.process(
            {"target_concept_id": "EEDI_CONSTRUCT_856", "kg_analysis": kg_analysis, "diagnosis_result": diag_wrong},
            context
        )
        path3_actions = [s["action_type"] for s in res3["learning_path"]]
        # Route MUST dynamically change to advance to advanced_challenge and skip review_prerequisite!
        self.assertIn("advanced_challenge", path3_actions)
        self.assertNotIn("review_prerequisite", path3_actions)
        self.assertIn("Advanced Challenge", res3["zpd_rationale"])

    def test_remediation_misconception_item_selection(self):
        """Verifies item selection matches specific misconception when remediating."""
        q_target = self.item_repo.select_item(
            concept_id="EEDI_CONSTRUCT_856",
            misconception_id_or_name="1672"
        )
        self.assertIsNotNone(q_target)
        self.assertIn(q_target["question_id"], self.all_eedi_qids)

    def test_full_paaf_pipeline_dynamic_zpd_integration(self):
        """End-to-end integration test with PAAFFramework."""
        framework = PAAFFramework(self.kg, item_repository=self.item_repo)
        q_eedi = EediDatasetLoader.get_question_by_id("EEDI_0")

        # Turn 1: Wrong answer
        out1 = framework.run_full_pipeline("S_PAAF_01", "Student Multi-turn", q_eedi, selected_option="D")
        planner_res1 = out1["planner_result"]
        state_obj: LearnerState = out1["raw_learner_state_obj"]

        self.assertEqual(state_obj.consecutive_correct, 0)
        self.assertEqual(state_obj.consecutive_incorrect, 1)
        self.assertIn("EEDI_0", state_obj.answered_questions)

        for step in planner_res1["learning_path"]:
            self.assertIn(step["question_id"], self.all_eedi_qids)

        # Turn 2: Correct answer on step question
        next_q_id = planner_res1["learning_path"][0]["question_id"]
        next_q = EediDatasetLoader.get_question_by_id(next_q_id)
        out2 = framework.run_full_pipeline("S_PAAF_01", "Student Multi-turn", next_q, selected_option=next_q["correct_option"], learner_state=state_obj)

        self.assertEqual(state_obj.consecutive_correct, 1)
        self.assertEqual(state_obj.consecutive_incorrect, 0)
        self.assertIn(next_q_id, state_obj.answered_questions)

        # Turn 3: Second correct answer -> triggers route acceleration
        next_q_id2 = out2["planner_result"]["learning_path"][0]["question_id"]
        next_q2 = EediDatasetLoader.get_question_by_id(next_q_id2)
        out3 = framework.run_full_pipeline("S_PAAF_01", "Student Multi-turn", next_q2, selected_option=next_q2["correct_option"], learner_state=state_obj)

        self.assertEqual(state_obj.consecutive_correct, 2)
        planner_res3 = out3["planner_result"]
        path3_actions = [s["action_type"] for s in planner_res3["learning_path"]]
        self.assertIn("advanced_challenge", path3_actions)


if __name__ == "__main__":
    unittest.main()
