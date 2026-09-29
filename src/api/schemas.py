"""
Pydantic Schemas for Azota / Study4 / Assessment Integration REST API Gateway.
Defines input payloads and structured diagnostic response output models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class QuestionItemSchema(BaseModel):
    question_id: str = Field(..., description="Unique question identifier (e.g. EEDI_123 or Q001)")
    concept_id: Optional[str] = Field(None, description="Concept or construct identifier")
    concept_name: Optional[str] = Field(None, description="Human readable concept name")
    question_text: Optional[str] = Field("", description="Question text content")
    options: Optional[Dict[str, str]] = Field(default_factory=dict, description="Options dictionary, e.g. {'A': '...', 'B': '...'}")
    correct_option: Optional[str] = Field(None, description="Correct option character (e.g. 'A')")
    misconception_map: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Misconception mappings per option")
    selected_option: Optional[str] = Field(None, description="Option selected by student (e.g. 'C')")


class AnswerSubmissionSchema(BaseModel):
    question_id: str = Field(..., description="Target question ID")
    selected_option: str = Field(..., description="Option selected by student")
    question: Optional[QuestionItemSchema] = Field(None, description="Optional embedded question details")


class AssessmentSubmitRequest(BaseModel):
    student_id: str = Field(..., description="Student unique ID")
    test_id: str = Field(..., description="Test or exam session ID")
    student_name: Optional[str] = Field("Học sinh", description="Student full name")
    questions: Optional[List[QuestionItemSchema]] = Field(default=None, description="List of exam questions with selected options")
    submissions: Optional[List[AnswerSubmissionSchema]] = Field(default=None, description="List of item submissions")
    use_llm: bool = Field(False, description="Whether to invoke local LLM CoT engine for wrong answers")


class MisconceptionBadgeSchema(BaseModel):
    badge_id: str = Field(..., description="Unique badge identifier (e.g. BADGE_101 or BADGE_Q1_B)")
    misconception_id: str = Field("unlabeled", description="Eedi or identified misconception ID")
    label: str = Field(..., description="Human-readable misconception badge label")
    badge_color: str = Field("#EF4444", description="Hex color code for badge display (e.g. #EF4444 for red, #F59E0B for amber)")
    badge_variant: str = Field("danger", description="Badge UI variant: 'danger', 'warning', 'info', 'purple', 'neutral'")
    badge_type: str = Field("misconception", description="Type: 'misconception', 'procedural_error', 'conceptual_gap', 'unclassified'")
    severity: str = Field("unknown", description="Severity level: 'high', 'medium', 'low', 'critical', 'unknown'")
    cot_summary: str = Field("", description="One-line Chain-of-Thought summary for badge pill or tooltip")
    root_cause: str = Field("", description="Detailed root cause explanation for the student error")
    confidence_score: float = Field(1.0, description="Diagnostic model confidence score (0.0 - 1.0)")


class RootCauseAnalysisSchema(BaseModel):
    observation: str = Field(..., description="Step 1: Observation of student's choice vs correct answer")
    misconception: str = Field(..., description="Step 2: Identified misconception name")
    detailed_explanation: str = Field(..., description="Step 3: In-depth mathematical explanation of why the misconception occurs")
    gap_conclusion: str = Field(..., description="Step 4: Pedagogical conclusion and identified knowledge gap")
    cot_steps: List[str] = Field(default_factory=list, description="Step-by-step Chain-of-Thought reasoning breakdown")


class QuestionDiagnosisResultSchema(BaseModel):
    question_id: str
    concept_id: str
    concept_name: str
    is_correct: bool
    selected_option: str
    correct_option: str
    detected_misconception: Optional[str] = None
    misconception_id: Optional[str] = "unlabeled"
    severity: Optional[str] = "unknown"
    cot_explanation: str
    mastery_score: float
    engine: str = "rule_based"
    misconception_badge: Optional[MisconceptionBadgeSchema] = None
    root_cause_analysis: Optional[RootCauseAnalysisSchema] = None
    badge_html: Optional[str] = None


class ZPDPathStepSchema(BaseModel):
    step_id: int
    concept_id: str
    concept_name: str
    action_type: str
    description: str
    question_id: Optional[str] = None
    question_details: Optional[Dict[str, Any]] = None


class AssessmentDiagnosticResponse(BaseModel):
    status: str = "success"
    test_id: str
    student_id: str
    student_name: str
    total_questions: int
    correct_count: int
    incorrect_count: int
    score_percentage: float
    score_scale_10: float
    passed: bool
    processing_time_ms: float
    question_diagnoses: List[QuestionDiagnosisResultSchema]
    incorrect_questions: List[QuestionDiagnosisResultSchema]
    misconception_badges: List[MisconceptionBadgeSchema] = Field(default_factory=list, description="Aggregated misconception badges for incorrect answers")
    mastery_summary: Dict[str, float]
    unmastered_prerequisites: List[str]
    zpd_remediation_path: List[ZPDPathStepSchema]
    learner_state_saved: bool = True


class PostSubmissionDiagnosticRequest(AssessmentSubmitRequest):
    """Payload to trigger dedicated post-submission CoT diagnostic & misconception badging."""
    pass


class PostSubmissionDiagnosticResponse(BaseModel):
    status: str = "success"
    test_id: str
    student_id: str
    student_name: str
    total_questions: int
    total_incorrect: int
    incorrect_diagnoses: List[QuestionDiagnosisResultSchema]
    misconception_badges: List[MisconceptionBadgeSchema]
    processing_time_ms: float
    engine: str = "rule_based"


class TutorChatRequest(BaseModel):
    student_id: str = Field(..., description="Student unique ID")
    question_id: str = Field(..., description="Target question ID for AI tutor scaffolding guidance")
    student_query: str = Field(..., description="Student prompt or question text")
    test_id: Optional[str] = Field(None, description="Optional test ID context")
    use_llm: bool = Field(False, description="Whether to invoke local LLM tutor model")


class TutorScaffoldRequest(BaseModel):
    student_id: str = Field(..., description="Student unique ID")
    question_id: str = Field(..., description="Target question ID to request hint for")
    test_id: Optional[str] = Field(None, description="Optional test ID context")


class TutorChatResponse(BaseModel):
    status: str = "success"
    student_id: str
    question_id: str
    response_text: str = Field(..., description="Tutor AI response text")
    scaffolding_level: str = Field(..., description="Graduated Hinting level ('nudge', 'hint', 'explanation')")
    turn_index: int = Field(..., description="Scaffolding turn index")
    detected_misconception: Optional[str] = Field(None, description="Identified misconception label")
    answer_leakage_prevented: bool = Field(True, description="Answer leakage guardrail flag")
    engine: str = Field("rule_based", description="Tutor engine type")

