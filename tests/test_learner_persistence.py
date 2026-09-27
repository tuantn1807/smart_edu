"""
Unit and Integration Tests for Agent Memory Persistence using SQLite (SE-06).
Verifies:
1. SQLite repository CRUD operations for LearnerState (mastery, misconceptions, learning path, chat logs).
2. PAAF Framework multi-question sessions loading saved LearnerState by student_id.
3. Cumulative mastery preservation across questions without reset.
4. Tutor Agent access to historical interaction history across multiple questions.
"""

import unittest
import tempfile
import os
from src.core.learner_state import LearnerState, MisconceptionRecord, LearningPathStep
from src.core.knowledge_graph import KnowledgeGraph
from src.data.learner_repository import LearnerStateRepository
from src.data.item_repository import EediItemRepository
from src.orchestrator.paaf_framework import PAAFFramework


class TestLearnerStateRepository(unittest.TestCase):
    def setUp(self):
        # Use in-memory SQLite database for fast unit testing
        self.repo = LearnerStateRepository(db_path=":memory:")

    def test_save_and_load_basic_state(self):
        state = LearnerState(student_id="STU_001", student_name="Nguyễn Văn A")
        state.record_question_result("Q_101", is_correct=True)
        state.set_concept_mastery("CONCEPT_A", 0.85)
        
        self.repo.save_learner_state(state)
        loaded = self.repo.load_learner_state("STU_001")

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.student_id, "STU_001")
        self.assertEqual(loaded.student_name, "Nguyễn Văn A")
        self.assertEqual(loaded.consecutive_correct, 1)
        self.assertIn("Q_101", loaded.answered_questions)
        self.assertAlmostEqual(loaded.get_concept_mastery("CONCEPT_A"), 0.85)

    def test_save_and_load_complex_state(self):
        state = LearnerState(student_id="STU_002", student_name="Trần Thị B")
        state.set_concept_mastery("C1", 0.7)
        state.set_concept_mastery("C2", 0.4)

        state.add_misconception(
            concept_id="C2",
            misconception_name="Cộng vế với vế",
            description="Học sinh cộng trực tiếp tử với tử, mẫu với mẫu",
            severity="high"
        )

        step1 = LearningPathStep(
            step_id=1,
            concept_id="C1",
            concept_name="Quy đồng mẫu số",
            action_type="review_prerequisite",
            description="Ôn tập quy đồng",
            status="completed",
            question_id="Q_PREREQ_01",
            question_details={"title": "Quy đồng phân số"}
        )
        state.set_learning_path([step1])

        state.add_interaction(
            role="user",
            message="Em chưa hiểu chỗ này",
            agent_name="User",
            metadata={"question_id": "Q_102", "turn_index": 1}
        )
        state.add_interaction(
            role="assistant",
            message="Em hãy xem lại mẫu số chung nhé",
            agent_name="TutorAgent",
            metadata={"question_id": "Q_102", "turn_index": 1, "scaffolding_level": "nudge"}
        )

        self.repo.save_learner_state(state)
        loaded = self.repo.load_learner_state("STU_002")

        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.misconceptions), 1)
        self.assertEqual(loaded.misconceptions[0].misconception_name, "Cộng vế với vế")
        self.assertEqual(loaded.misconceptions[0].severity, "high")

        self.assertEqual(len(loaded.active_learning_path), 1)
        self.assertEqual(loaded.active_learning_path[0].question_id, "Q_PREREQ_01")
        self.assertEqual(loaded.active_learning_path[0].question_details["title"], "Quy đồng phân số")

        self.assertEqual(len(loaded.interaction_history), 2)
        self.assertEqual(loaded.interaction_history[0]["role"], "user")
        self.assertEqual(loaded.interaction_history[1]["agent"], "TutorAgent")
        self.assertEqual(loaded.interaction_history[1]["metadata"]["scaffolding_level"], "nudge")

    def test_load_nonexistent_student(self):
        loaded = self.repo.load_learner_state("NON_EXISTENT_ID")
        self.assertIsNone(loaded)

    def test_delete_learner_state(self):
        state = LearnerState(student_id="STU_DELETE", student_name="Xóa Test")
        self.repo.save_learner_state(state)
        self.assertIsNotNone(self.repo.load_learner_state("STU_DELETE"))

        self.repo.delete_learner_state("STU_DELETE")
        self.assertIsNone(self.repo.load_learner_state("STU_DELETE"))

    def test_file_based_sqlite_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            repo1 = LearnerStateRepository(db_path=tmp_path)
            state = LearnerState(student_id="STU_DISK", student_name="Học sinh Đĩa")
            state.set_concept_mastery("C_MATH", 0.9)
            repo1.save_learner_state(state)

            # Open with new repository instance to simulate app restart
            repo2 = LearnerStateRepository(db_path=tmp_path)
            loaded = repo2.load_learner_state("STU_DISK")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.student_name, "Học sinh Đĩa")
            self.assertAlmostEqual(loaded.get_concept_mastery("C_MATH"), 0.9)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPAAFPersistenceIntegration(unittest.TestCase):
    """Integration test suite for PAAF Framework with SQLite Persistence."""

    def setUp(self):
        self.kg = KnowledgeGraph()
        self.kg.add_concept("C_ADD", name="Cộng phân số", difficulty=0.3, description="Phép cộng phân số cùng mẫu")
        self.kg.add_concept("C_SUB", name="Trừ phân số", difficulty=0.4, description="Phép trừ phân số cùng mẫu")

        sample_questions = [
            {
                "question_id": "Q_PRACTICE_01",
                "construct_id": "C_ADD",
                "question_text": "Tính 1/3 + 1/3",
                "options": {"A": "2/3", "B": "2/6", "C": "1/6", "D": "1/3"},
                "correct_option": "A",
                "misconception_name": "Unlabeled"
            },
            {
                "question_id": "Q_PRACTICE_02",
                "construct_id": "C_SUB",
                "question_text": "Tính 2/3 - 1/3",
                "options": {"A": "1/3", "B": "1/0", "C": "3/3", "D": "0/3"},
                "correct_option": "A",
                "misconception_name": "Unlabeled"
            }
        ]
        self.item_repo = EediItemRepository(questions=sample_questions)
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_acceptance_criterion_1_and_2_multi_question_persistence(self):
        """
        Acceptance Criteria 1 & 2:
        - After completing Question 1 and moving to Question 2, system loads correct LearnerState by student_id.
        - Cumulative mastery is NOT reset across questions.
        """
        student_id = "STU_PERSIST_100"
        student_name = "Lê Văn C"

        # Framework instance 1 (Session 1 / Question 1)
        framework1 = PAAFFramework(
            knowledge_graph=self.kg,
            item_repository=self.item_repo,
            db_path=self.db_path
        )

        q1 = {
            "question_id": "Q_EEDI_001",
            "concept_id": "C_ADD",
            "question_text": "Tính 1/4 + 2/4",
            "options": {"A": "3/4", "B": "3/8", "C": "1/2", "D": "2/8"},
            "correct_option": "A"
        }

        # Student answers Question 1 CORRECTLY
        out1 = framework1.run_full_pipeline(student_id, student_name, q1, selected_option="A")
        state1: LearnerState = out1["raw_learner_state_obj"]

        mastery_after_q1 = state1.get_concept_mastery("C_ADD")
        self.assertGreater(mastery_after_q1, 0.0)
        self.assertEqual(state1.consecutive_correct, 1)
        self.assertIn("Q_EEDI_001", state1.answered_questions)

        # Framework instance 2 (New pipeline instance simulating app restart / Question 2)
        framework2 = PAAFFramework(
            knowledge_graph=self.kg,
            item_repository=self.item_repo,
            db_path=self.db_path
        )

        q2 = {
            "question_id": "Q_EEDI_002",
            "concept_id": "C_ADD",
            "question_text": "Tính 2/5 + 1/5",
            "options": {"A": "3/5", "B": "3/10", "C": "1/5", "D": "4/5"},
            "correct_option": "A"
        }

        # Run Question 2 without passing explicit learner_state -> system loads from SQLite
        out2 = framework2.run_full_pipeline(student_id, student_name, q2, selected_option="A")
        state2: LearnerState = out2["raw_learner_state_obj"]

        # Check Criterion 1: Loaded state matches student_id and includes Q1 history
        self.assertEqual(state2.student_id, student_id)
        self.assertEqual(len(state2.answered_questions), 2)
        self.assertIn("Q_EEDI_001", state2.answered_questions)
        self.assertIn("Q_EEDI_002", state2.answered_questions)

        # Check Criterion 2: Mastery score accumulated, NOT reset to 0.0
        mastery_after_q2 = state2.get_concept_mastery("C_ADD")
        self.assertGreater(mastery_after_q2, mastery_after_q1)
        self.assertEqual(state2.consecutive_correct, 2)

    def test_acceptance_criterion_3_tutor_interaction_history_retrieval(self):
        """
        Acceptance Criterion 3:
        Tutor Agent retrieves interaction history from previous questions across sessions.
        """
        student_id = "STU_PERSIST_200"
        student_name = "Hoàng Thị D"

        # Session 1: Question 1 and Tutor Interaction
        framework1 = PAAFFramework(
            knowledge_graph=self.kg,
            item_repository=self.item_repo,
            db_path=self.db_path
        )

        q1 = {
            "question_id": "Q_EEDI_101",
            "concept_id": "C_SUB",
            "question_text": "Tính 3/4 - 1/4",
            "options": {"A": "2/4", "B": "2/0", "C": "4/4", "D": "1/2"},
            "correct_option": "A"
        }

        out1 = framework1.run_full_pipeline(student_id, student_name, q1, selected_option="B")
        tutor_out1 = framework1.interact_with_tutor(out1, student_query="Em lấy 4 trừ 4 bằng 0 ạ?")

        self.assertIn("tutor_response", tutor_out1)
        state_s1: LearnerState = out1["raw_learner_state_obj"]
        self.assertEqual(len(state_s1.interaction_history), 2)

        # Session 2: New Framework instance (Question 2)
        framework2 = PAAFFramework(
            knowledge_graph=self.kg,
            item_repository=self.item_repo,
            db_path=self.db_path
        )

        q2 = {
            "question_id": "Q_EEDI_102",
            "concept_id": "C_SUB",
            "question_text": "Tính 5/7 - 2/7",
            "options": {"A": "3/7", "B": "3/0", "C": "7/7", "D": "1/7"},
            "correct_option": "A"
        }

        out2 = framework2.run_full_pipeline(student_id, student_name, q2, selected_option="B")
        state_s2: LearnerState = out2["raw_learner_state_obj"]

        # Verify previous interaction history from Q1 is loaded into state_s2
        self.assertGreaterEqual(len(state_s2.interaction_history), 2)
        self.assertEqual(state_s2.interaction_history[0]["message"], "Em lấy 4 trừ 4 bằng 0 ạ?")

        # Tutor interacts on Question 2
        tutor_out2 = framework2.interact_with_tutor(out2, student_query="Sao mẫu số lại không được trừ ạ?")
        self.assertIn("tutor_response", tutor_out2)

        # Verify updated interaction history contains entries from both Q1 and Q2
        updated_history = state_s2.interaction_history
        self.assertEqual(len(updated_history), 4)
        self.assertEqual(updated_history[0]["metadata"]["question_id"], "Q_EEDI_101")
        self.assertEqual(updated_history[2]["metadata"]["question_id"], "Q_EEDI_102")


if __name__ == "__main__":
    unittest.main()
