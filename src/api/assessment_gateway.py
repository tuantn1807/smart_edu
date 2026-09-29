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
    MisconceptionBadgeSchema,
    RootCauseAnalysisSchema,
    PostSubmissionDiagnosticResponse,
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

    @staticmethod
    def render_misconception_badge_html(
        badge: MisconceptionBadgeSchema,
        root_cause: Optional[RootCauseAnalysisSchema] = None
    ) -> str:
        """
        Renders responsive, accessible HTML markup for the Misconception Badge
        and root-cause CoT breakdown beside an incorrect exam question.
        """
        badge_color = badge.badge_color or "#EF4444"
        sev_upper = badge.severity.upper() if badge.severity else "UNKNOWN"

        steps_html = ""
        if root_cause and root_cause.cot_steps:
            steps_items = "".join(f"<li style='margin-bottom: 6px;'>{step}</li>" for step in root_cause.cot_steps)
            steps_html = f"<ul style='margin: 8px 0 0 0; padding-left: 20px; font-size: 13px; color: #374151; line-height: 1.5;'>{steps_items}</ul>"
        elif root_cause:
            steps_html = (
                f"<div style='margin-top: 8px; font-size: 13px; color: #374151; line-height: 1.5;'>"
                f"<p style='margin: 4px 0;'><strong>🔍 Quan sát:</strong> {root_cause.observation}</p>"
                f"<p style='margin: 4px 0;'><strong>⚠️ Nguyên nhân gốc rễ:</strong> {root_cause.misconception}</p>"
                f"<p style='margin: 4px 0;'><strong>📖 Diễn giải:</strong> {root_cause.detailed_explanation}</p>"
                f"<p style='margin: 4px 0;'><strong>🎯 Kết luận:</strong> {root_cause.gap_conclusion}</p>"
                f"</div>"
            )

        html = (
            f'<div class="misconception-badge-card" style="border: 1px solid #E5E7EB; border-left: 4px solid {badge_color}; border-radius: 8px; padding: 12px 16px; background-color: #F9FAFB; margin-top: 10px; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif;">'
            f'<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">'
            f'<div style="display: flex; align-items: center; gap: 8px;">'
            f'<span class="badge-pill" style="background-color: {badge_color}; color: #FFFFFF; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 9999px; display: inline-flex; align-items: center;">'
            f'🏷️ {badge.label}'
            f'</span>'
            f'<span style="font-size: 12px; color: #6B7280; font-family: monospace;">#{badge.misconception_id}</span>'
            f'</div>'
            f'<span style="font-size: 11px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; color: #4B5563; background: #E5E7EB; padding: 2px 8px; border-radius: 4px;">'
            f'Mức độ: {sev_upper}'
            f'</span>'
            f'</div>'
            f'<div style="margin-top: 10px; border-top: 1px dashed #D1D5DB; padding-top: 8px;">'
            f'<div style="font-weight: 600; font-size: 13px; color: #1F2937; display: flex; align-items: center; gap: 4px;">'
            f'🧠 Phân tích Suy luận CoT (Root Cause):'
            f'</div>'
            f'{steps_html}'
            f'</div>'
            f'</div>'
        )
        return html

    def _parse_submission_items(self, request: AssessmentSubmitRequest) -> List[Dict[str, Any]]:
        """Parses and normalizes questions or submissions list from payload."""
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
        return submission_items

    def _resolve_question_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves complete question content merging item repository with payload."""
        qid = item["question_id"]
        provided_q = item.get("provided_question") or {}
        repo_q = self.item_repository.get_question_by_id(qid)
        question_data = {}
        if repo_q:
            question_data.update(repo_q)
        if provided_q:
            for k, v in provided_q.items():
                if v is not None and v != "" and v != {}:
                    question_data[k] = v

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
        return question_data

    def filter_incorrect_submissions(self, submission_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters all incorrect answers from exam submission items.
        Compares selected option against question's correct option.
        """
        incorrect_items: List[Dict[str, Any]] = []
        for item in submission_items:
            qid = item["question_id"]
            selected_opt = (item.get("selected_option") or "").strip().upper()
            provided_q = item.get("provided_question") or {}
            repo_q = self.item_repository.get_question_by_id(qid) or {}

            correct_opt = str(provided_q.get("correct_option") or repo_q.get("correct_option") or "A").strip().upper()
            if selected_opt != correct_opt:
                incorrect_items.append(item)
        return incorrect_items

    def process_submission(self, request: AssessmentSubmitRequest) -> AssessmentDiagnosticResponse:
        """
        Processes full test submission asynchronously/synchronously from Azota / Study4 platform.
        Evaluates questions, updates mastery state, identifies misconceptions, 
        propagates mastery loss, and returns full diagnostic report & ZPD remediation plan.
        """
        start_time = time.perf_counter()

        # Step 1: Resolve question submissions list
        submission_items = self._parse_submission_items(request)

        # Step 2: Load or initialize LearnerState
        learner_state = self.learner_repository.load_learner_state(request.student_id)
        if learner_state is None:
            learner_state = LearnerState(student_id=request.student_id, student_name=request.student_name or "Học sinh")
        elif request.student_name and request.student_name != "Học sinh":
            learner_state.student_name = request.student_name

        question_diagnoses: List[QuestionDiagnosisResultSchema] = []
        incorrect_diagnoses: List[QuestionDiagnosisResultSchema] = []
        misconception_badges: List[MisconceptionBadgeSchema] = []
        correct_count = 0

        context = {
            "learner_state": learner_state,
            "item_repository": self.item_repository
        }

        # Step 3: Process per-question diagnosis & mastery update
        for item in submission_items:
            qid = item["question_id"]
            selected_opt = (item.get("selected_option") or "").upper()
            question_data = self._resolve_question_data(item)

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

            badge_data = diag_res.get("misconception_badge")
            badge_schema = MisconceptionBadgeSchema(**badge_data) if badge_data else None

            rc_data = diag_res.get("root_cause_analysis")
            rc_schema = RootCauseAnalysisSchema(**rc_data) if rc_data else None

            badge_html = None
            if badge_schema and rc_schema:
                badge_html = self.render_misconception_badge_html(badge_schema, rc_schema)

            if is_correct:
                correct_count += 1
            else:
                if mapping.mapped or target_concept_id in self.knowledge_graph.nodes:
                    learner_state.propagate_mastery_loss(target_concept_id, self.knowledge_graph)
                if badge_schema:
                    misconception_badges.append(badge_schema)

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
                engine=diag_res.get("engine", "rule_based"),
                misconception_badge=badge_schema,
                root_cause_analysis=rc_schema,
                badge_html=badge_html
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
            misconception_badges=misconception_badges,
            mastery_summary=learner_state.mastery_levels,
            unmastered_prerequisites=list(unmastered_prereqs_set),
            zpd_remediation_path=zpd_steps,
            learner_state_saved=True
        )

    def diagnose_post_submission(self, request: AssessmentSubmitRequest) -> PostSubmissionDiagnosticResponse:
        """
        Post-Submission CoT Diagnostic & Misconception Badging (SE-12):
        1. Filters all incorrect answers in the test submission.
        2. Runs Diagnostic Agent to perform CoT root cause reasoning on each incorrect answer.
        3. Assigns misconception_id and attaches MisconceptionBadgeSchema & RootCauseAnalysisSchema to each wrong question.
        4. Returns structured post-submission diagnostic report.
        """
        start_time = time.perf_counter()
        submission_items = self._parse_submission_items(request)

        learner_state = self.learner_repository.load_learner_state(request.student_id)
        if learner_state is None:
            learner_state = LearnerState(student_id=request.student_id, student_name=request.student_name or "Học sinh")
        elif request.student_name and request.student_name != "Học sinh":
            learner_state.student_name = request.student_name

        context = {
            "learner_state": learner_state,
            "item_repository": self.item_repository
        }

        # Step 1: Filter incorrect answers
        incorrect_items = self.filter_incorrect_submissions(submission_items)
        incorrect_diagnoses: List[QuestionDiagnosisResultSchema] = []
        misconception_badges: List[MisconceptionBadgeSchema] = []

        # Step 2: Run Diagnostic Agent on each wrong answer
        for item in incorrect_items:
            qid = item["question_id"]
            selected_opt = (item.get("selected_option") or "").strip().upper()
            question_data = self._resolve_question_data(item)

            diag_input = {
                "question": question_data,
                "selected_option": selected_opt,
                "use_llm": request.use_llm
            }
            diag_res = self.paaf_framework.diagnostic_agent.process(diag_input, context)

            badge_data = diag_res.get("misconception_badge")
            badge_schema = MisconceptionBadgeSchema(**badge_data) if badge_data else None
            if badge_schema:
                misconception_badges.append(badge_schema)

            rc_data = diag_res.get("root_cause_analysis")
            rc_schema = RootCauseAnalysisSchema(**rc_data) if rc_data else None

            badge_html = None
            if badge_schema and rc_schema:
                badge_html = self.render_misconception_badge_html(badge_schema, rc_schema)

            concept_id = diag_res.get("concept_id", "UNKNOWN")
            mastery_score = learner_state.get_concept_mastery(concept_id) if learner_state else 0.0

            diag_schema = QuestionDiagnosisResultSchema(
                question_id=qid,
                concept_id=concept_id,
                concept_name=diag_res.get("concept_name", "Khái niệm chưa rõ"),
                is_correct=False,
                selected_option=selected_opt,
                correct_option=diag_res.get("correct_option", ""),
                detected_misconception=diag_res.get("detected_misconception"),
                misconception_id=diag_res.get("misconception_id", "unlabeled"),
                severity=diag_res.get("severity", "unknown"),
                cot_explanation=diag_res.get("cot_explanation", ""),
                mastery_score=mastery_score,
                engine=diag_res.get("engine", "rule_based"),
                misconception_badge=badge_schema,
                root_cause_analysis=rc_schema,
                badge_html=badge_html
            )
            incorrect_diagnoses.append(diag_schema)

        processing_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return PostSubmissionDiagnosticResponse(
            status="success",
            test_id=request.test_id,
            student_id=request.student_id,
            student_name=learner_state.student_name,
            total_questions=len(submission_items),
            total_incorrect=len(incorrect_diagnoses),
            incorrect_diagnoses=incorrect_diagnoses,
            misconception_badges=misconception_badges,
            processing_time_ms=processing_time_ms,
            engine="llm_cot" if request.use_llm else "rule_based"
        )
