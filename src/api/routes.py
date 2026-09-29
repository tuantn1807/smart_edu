"""
FastAPI Router for Online Assessment REST API Gateway.
Provides endpoints for Azota / Study4 / Tuyensinh247 exam payload submission & diagnosis.
"""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas import (
    AssessmentSubmitRequest,
    AssessmentDiagnosticResponse,
    PostSubmissionDiagnosticResponse,
)
from src.api.assessment_gateway import AssessmentGatewayService

router = APIRouter(prefix="/api/v1/assessment", tags=["Assessment Gateway"])

# Singleton gateway service instance
gateway_service = AssessmentGatewayService()


@router.post(
    "/submit",
    response_model=AssessmentDiagnosticResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit exam payload for automated grading & diagnostic analysis",
    description="Webhook/Integration endpoint receiving Azota/Study4 exam submissions. Returns per-question diagnosis, misconception badges, mastery levels, and ZPD remediation plan."
)
async def submit_assessment(request: AssessmentSubmitRequest) -> AssessmentDiagnosticResponse:
    try:
        response = gateway_service.process_submission(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Gateway Error: {str(e)}")


@router.post(
    "/diagnose-submission",
    response_model=AssessmentDiagnosticResponse,
    status_code=status.HTTP_200_OK,
    summary="Diagnose full exam submission with detailed CoT reasoning",
    description="Endpoint for deep diagnostic evaluation of exam submissions, supporting optional Local LLM CoT reasoning."
)
async def diagnose_submission(request: AssessmentSubmitRequest) -> AssessmentDiagnosticResponse:
    try:
        response = gateway_service.process_submission(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Diagnostic Error: {str(e)}")


@router.post(
    "/post-submission-diagnostic",
    response_model=PostSubmissionDiagnosticResponse,
    status_code=status.HTTP_200_OK,
    summary="Post-submission CoT Diagnostic & Misconception Badging",
    description="Filters incorrect answers from exam submission, runs Diagnostic Agent for CoT reasoning, and assigns misconception badges."
)
async def post_submission_diagnostic(request: AssessmentSubmitRequest) -> PostSubmissionDiagnosticResponse:
    try:
        response = gateway_service.diagnose_post_submission(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal Diagnostic Error: {str(e)}")


@router.get(
    "/health",
    summary="Health check & Gateway status",
    description="Returns API Gateway operational status and system metrics."
)
async def health_check():
    return {
        "status": "online",
        "service": "Online Assessment REST API Gateway (PAAF Framework)",
        "knowledge_graph_nodes": len(gateway_service.knowledge_graph.nodes),
        "indexed_questions": len(gateway_service.item_repository.questions),
        "gateway_version": "1.0.0"
    }
