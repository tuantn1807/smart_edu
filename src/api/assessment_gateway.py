"""
Assessment Gateway Service for Online Assessment Platforms (Azota / Study4 / Tuyensinh247).
Handles multi-question test submissions, asynchronous diagnostic evaluation, 
mastery propagation, misconception badging, and automated ZPD remediation path generation.
"""

import time
from typing import Dict, List, Any, Optional
from src.core.knowledge_graph import KnowledgeGraph
from src.core.learner_state import LearnerState
from src.data.dataset_loaders import JunyiGraphLoader, EediDatasetLoader
from src.data.item_repository import EediItemRepository
from src.data.learner_repository import LearnerStateRepository
from src.data.concept_mapping import EediJunyiMapper
from src.orchestrator.paaf_framework import PAAFFramework
from src.api.schemas import (
    AssessmentSubmitRequest,
    AssessmentDiagnosticResponse,
    QuestionDiagnosisResultSchema,
    ZPDPathStepSchema,
    QuestionItemSchema,
)


class AssessmentGatewayService:
    def __init__(self, db_path: Optional[str] = None):
        try:
            self.knowledge_graph = JunyiGraphLoader.load_math_prerequisite_graph()
        except Exception:
            self.knowledge_graph = EediDatasetLoader.load_concept_graph()

        self.item_repository = EediItemRepository()
        self.learner_repository = LearnerStateRepository(db_path=db_path)
        self.concept_mapper = EediJunyiMapper(self.knowledge_graph)
        self.paaf_framework = PAAFFramework(
            knowledge_graph=self.knowledge_graph,
            concept_mapper=self.concept_mapper,
            item_repository=self.item_repository,
            learner_repository=self.learner_repository,
            db_path=db_path
        )

    def set_db_path(self, db_path: str):
        """Helper to re-target database path for testing or multi-tenant persistence."""
        self.learner_repository.set_db_path(db_path)
        self.paaf_framework.learner_repository = self.learner_repository

    def process_submission(self, request: AssessmentSubmitRequest) -> AssessmentDiagnosticResponse:
        """
        Processes full test submission asynchronously/synchronously from Azota / Study4 platform.
        Evaluates questions, updates mastery state, identifies misconceptions, 
        propagates mastery loss, and returns full diagnostic report & ZPD remediation plan.
        """
        start_time = time.perf_counter()

        # Step 1: Resolve question submissions list
        submission_items: List[Dict[str, Any]] = []

        if request.questions:
            for q_item in request.questions:
                q_dict = q_item.model_dump() if hasattr(q_item, "model_dump") else q_item.dict()
                submission_items.append({
                    "question_id": q_item.question_id,
                    "selected_option": q_item.selected_option,
                    "provided_question": q_dict
                })
        elif request.submissions:
            for sub in request.submissions:
                sub_dict = sub.model_dump() if hasattr(sub, "model_dump") else sub.dict()
                q_dict = sub_dict.get("question") or {}
                submission_items.append({
                    "question_id": sub.question_id,
                    "selected_option": sub.selected_option,
                    "provided_question": q_dict
                })
        else:
            raise ValueError("Payload missing 'questions' or 'submissions' list.")

        if not submission_items:
            raise ValueError("No questions submitted in exam payload.")

        # Step 2: Load or initialize LearnerState
        learner_state = self.learner_repository.load_learner_state(request.student_id)
        if learner_state is None:
            learner_state = LearnerState(student_id=request.student_id, student_name=request.student_name or "Học sinh")
        elif request.student_name and request.student_name != "Học sinh":
            learner_state.student_name = request.student_name

        question_diagnoses: List[QuestionDiagnosisResultSchema] = []
        incorrect_diagnoses: List[QuestionDiagnosisResultSchema] = []
        correct_count = 0

        context = {
            "learner_state": learner_state,
            "item_repository": self.item_repository
        }

        # Step 3: Process per-question diagnosis & mastery update
        for item in submission_items:
            qid = item["question_id"]
            selected_opt = (item.get("selected_option") or "").upper()
            provided_q = item.get("provided_question") or {}

            # Lookup question from item repository if fields missing
            repo_q = self.item_repository.get_question_by_id(qid)
            question_data = {}
            if repo_q:
                question_data.update(repo_q)
            if provided_q:
                for k, v in provided_q.items():
                    if v is not None and v != "" and v != {}:
                        question_data[k] = v

            # Fallback if question content still not populated
            if "question_id" not in question_data:
                question_data["question_id"] = qid
            if "concept_id" not in question_data:
                question_data["concept_id"] = provided_q.get("concept_id") or "UNKNOWN_CONCEPT"
            if "concept_name" not in question_data:
                question_data["concept_name"] = provided_q.get("concept_name") or "Khái niệm trắc nghiệm"
            if "options" not in question_data or not question_data["options"]:
                question_data["options"] = {"A": "Phương án A", "B": "Phương án B", "C": "Phương án C", "D": "Phương án D"}
            if "correct_option" not in question_data or not question_data["correct_option"]:
                question_data["correct_option"] = provided_q.get("correct_option") or "A"

            # Execute Diagnostic Agent
            diag_input = {
                "question": question_data,
                "selected_option": selected_opt,
                "use_llm": request.use_llm
            }
            diag_res = self.paaf_framework.diagnostic_agent.process(diag_input, context)

            is_correct = diag_res.get("is_correct", False)
            concept_id = diag_res.get("concept_id", "UNKNOWN")

            # Map concept to Junyi node for mastery propagation
            mapping = self.paaf_framework.concept_mapper.map_question(question_data)
            target_concept_id = mapping.junyi_concept_id if mapping.mapped else concept_id

            # Mastery update & history tracking
            learner_state.record_question_result(qid, is_correct=is_correct)
            mastery_score = learner_state.update_mastery(target_concept_id, is_correct=is_correct)

            if is_correct:
                correct_count += 1
            else:
                if mapping.mapped or target_concept_id in self.knowledge_graph.nodes:
                    learner_state.propagate_mastery_loss(target_concept_id, self.knowledge_graph)

            diag_schema = QuestionDiagnosisResultSchema(
                question_id=qid,
                concept_id=diag_res.get("concept_id", "UNKNOWN"),
                concept_name=diag_res.get("concept_name", "Khái niệm chưa rõ"),
                is_correct=is_correct,
                selected_option=selected_opt,
                correct_option=diag_res.get("correct_option", ""),
                detected_misconception=diag_res.get("detected_misconception"),
                misconception_id=diag_res.get("misconception_id", "unlabeled"),
                severity=diag_res.get("severity", "unknown"),
                cot_explanation=diag_res.get("cot_explanation", ""),
                mastery_score=mastery_score,
                engine=diag_res.get("engine", "rule_based")
            )

            question_diagnoses.append(diag_schema)
            if not is_correct:
                incorrect_diagnoses.append(diag_schema)

        # Step 4: Overall Test Scoring
        total_questions = len(submission_items)
        incorrect_count = total_questions - correct_count
        score_percentage = round((correct_count / total_questions) * 100.0, 2)
        score_scale_10 = round((correct_count / total_questions) * 10.0, 2)
        passed = score_scale_10 >= 5.0

        # Step 5: Knowledge Gap Detection & ZPD Remediation Path Generation
        unmastered_prereqs_set = set()
        zpd_steps: List[ZPDPathStepSchema] = []
        step_counter = 1

        for inc_diag in incorrect_diagnoses:
            c_id = inc_diag.concept_id
            mapping = self.paaf_framework.concept_mapper.map_question({
                "concept_id": c_id,
                "concept_name": inc_diag.concept_name
            })
            target_cid = mapping.junyi_concept_id if mapping.mapped else c_id

            kg_input = {
                "target_concept_id": target_cid,
                "mapping": mapping.to_dict()
            }
            kg_res = self.paaf_framework.kg_agent.process(kg_input, context)

            for prereq in kg_res.get("unmastered_prerequisites", []):
                pid = prereq.get("concept_id") or prereq.get("name")
                if pid:
                    unmastered_prereqs_set.add(pid)

            planner_input = {
                "target_concept_id": c_id,
                "kg_analysis": kg_res,
                "diagnosis_result": inc_diag.model_dump() if hasattr(inc_diag, "model_dump") else inc_diag.dict()
            }
            planner_res = self.paaf_framework.planner_agent.process(planner_input, context)

            for step in planner_res.get("learning_path", []):
                zpd_steps.append(ZPDPathStepSchema(
                    step_id=step_counter,
                    concept_id=step.get("concept_id", c_id),
                    concept_name=step.get("concept_name", "Khái niệm"),
                    action_type=step.get("action_type", "practice_exercise"),
                    description=step.get("description", ""),
                    question_id=step.get("question_id"),
                    question_details=step.get("question_details")
                ))
                step_counter += 1

        # Save updated LearnerState to SQLite persistence
        self.learner_repository.save_learner_state(learner_state)

        processing_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return AssessmentDiagnosticResponse(
            status="success",
            test_id=request.test_id,
            student_id=request.student_id,
            student_name=learner_state.student_name,
            total_questions=total_questions,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            score_percentage=score_percentage,
            score_scale_10=score_scale_10,
            passed=passed,
            processing_time_ms=processing_time_ms,
            question_diagnoses=question_diagnoses,
            incorrect_questions=incorrect_diagnoses,
            mastery_summary=learner_state.mastery_levels,
            unmastered_prerequisites=list(unmastered_prereqs_set),
            zpd_remediation_path=zpd_steps,
            learner_state_saved=True
        )
