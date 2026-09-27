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
    mastery_summary: Dict[str, float]
    unmastered_prerequisites: List[str]
    zpd_remediation_path: List[ZPDPathStepSchema]
    learner_state_saved: bool = True
