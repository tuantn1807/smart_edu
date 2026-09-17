"""
Pedagogical Agentic AI Framework (PAAF) Orchestrator.
Main system framework coordinating Diagnostic Agent, Knowledge Graph Agent, Planner Agent, and Tutor Agent.
Manages workflow state transitions, centralized memory, and tool execution.
"""

from typing import Dict, Any, Optional
from src.core.learner_state import LearnerState
from src.core.knowledge_graph import KnowledgeGraph
from src.data.concept_mapping import EediJunyiMapper
from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.kg_agent import KGAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.tutor_agent import TutorAgent


class PAAFFramework:
    def __init__(self, knowledge_graph: KnowledgeGraph, concept_mapper: Optional[EediJunyiMapper] = None):
        self.knowledge_graph = knowledge_graph
        self.concept_mapper = concept_mapper or EediJunyiMapper(knowledge_graph)
        self.diagnostic_agent = DiagnosticAgent()
        self.kg_agent = KGAgent(knowledge_graph=self.knowledge_graph)
        self.planner_agent = PlannerAgent()
        self.tutor_agent = TutorAgent()

    def run_full_pipeline(self, student_id: str, student_name: str, diagnostic_question: Dict[str, Any], selected_option: str) -> Dict[str, Any]:
        """
        Executes end-to-end PAAF multi-agent pipeline:
        1. Initialize Centralized Learner State
        2. Diagnostic Agent -> CoT Misconception Diagnosis
        3. Knowledge Graph Agent -> Prerequisite Gap Traversal
        4. Planner Agent -> ZPD Personalized Learning Path
        5. Returns structured state and remediation package.
        """
        print("\n=== [PAAF FRAMEWORK] KHỞI ĐỘNG TIẾN TRÌNH MULTI-AGENT PIPELINE ===")

        # Step 1: Initialize / Load Learner State
        learner_state = LearnerState(student_id=student_id, student_name=student_name)
        context = {"learner_state": learner_state}

        # Step 2: Diagnostic Agent Execution
        diag_input = {
            "question": diagnostic_question,
            "selected_option": selected_option
        }
        diagnosis_result = self.diagnostic_agent.process(diag_input, context)

        # Step 3: Knowledge Graph Agent Execution on the Junyi node, if mapped
        mapping = self.concept_mapper.map_question(diagnostic_question)
        eedi_concept_id = diagnostic_question.get("concept_id", "UNKNOWN")
        target_concept_id = mapping.junyi_concept_id if mapping.mapped else eedi_concept_id
        kg_input = {
            "target_concept_id": target_concept_id,
            "mapping": mapping.to_dict(),
        }
        kg_result = self.kg_agent.process(kg_input, context)

        # Step 4: Planner Agent Execution
        planner_input = {
            "target_concept_id": eedi_concept_id,
            "kg_analysis": kg_result,
            "diagnosis_result": diagnosis_result
        }
        planner_result = self.planner_agent.process(planner_input, context)

        print("=== [PAAF FRAMEWORK] HOÀN THÀNH MULTI-AGENT PIPELINE ===\n")

        return {
            "learner_state": learner_state.to_dict(),
            "diagnosis_result": diagnosis_result,
            "kg_analysis": kg_result,
            "planner_result": planner_result,
            "raw_learner_state_obj": learner_state
        }

    def interact_with_tutor(self, pipeline_output: Dict[str, Any], student_query: str) -> Dict[str, Any]:
        """Trigger Tutor Agent for interactive 2-way scaffolding dialogue."""
        learner_state: LearnerState = pipeline_output["raw_learner_state_obj"]
        context = {"learner_state": learner_state}

        tutor_input = {
            "student_query": student_query,
            "diagnosis_result": pipeline_output["diagnosis_result"],
            "planner_result": pipeline_output["planner_result"]
        }

        tutor_result = self.tutor_agent.process(tutor_input, context)
        return tutor_result
