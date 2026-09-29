"""
Comprehensive Test Suite for Post-Submission Diagnostic & Misconception Badging (SE-12).
Verifies:
- Filtering of incorrect MCQ answers from test submissions.
- CoT Root Cause Analysis generation and Misconception Badging.
- MisconceptionBadgeSchema, RootCauseAnalysisSchema, and responsive HTML card rendering.
- /api/v1/assessment/post-submission-diagnostic endpoint functionality and performance (<= 500ms).
- Edge cases: all correct submissions, all incorrect submissions, unlabeled misconceptions.
"""

import time
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.routes import gateway_service
from src.api.schemas import (
    MisconceptionBadgeSchema,
    RootCauseAnalysisSchema,
    AssessmentSubmitRequest,
)
from src.agents.diagnostic_agent import DiagnosticAgent

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    """Isolate learner state DB for each test."""
    db_file = str(tmp_path / "test_se12_post_submission.db")
    gateway_service.set_db_path(db_file)


def test_diagnostic_agent_badge_and_root_cause_helpers():
    """Verify DiagnosticAgent static builders for badges and root cause analysis."""
    # Test high severity badge
    badge_high = DiagnosticAgent.build_misconception_badge(
        question_id="Q1",
        selected_option="B",
        misconception_id="101",
        misconception_name="Cộng tử với tử, mẫu với mẫu",
        description="Học sinh cộng trực tiếp tử số và mẫu số.",
        severity="high",
        confidence_score=0.98
    )
    assert badge_high["badge_id"] == "BADGE_101"
    assert badge_high["badge_color"] == "#EF4444"
    assert badge_high["badge_variant"] == "danger"
    assert badge_high["severity"] == "high"
    assert badge_high["confidence_score"] == 0.98

    # Test medium severity badge
    badge_med = DiagnosticAgent.build_misconception_badge(
        question_id="Q2",
        selected_option="C",
        misconception_id="102",
        misconception_name="Nhân chéo nhầm lẫn",
        description="Học sinh nhân chéo thay vì nhân ngang.",
        severity="medium"
    )
    assert badge_med["badge_color"] == "#F59E0B"
    assert badge_med["badge_variant"] == "warning"

    # Test unlabeled badge fallback
    badge_unlabeled = DiagnosticAgent.build_misconception_badge(
        question_id="Q3",
        selected_option="D",
        misconception_id="unlabeled",
        misconception_name="Chưa có nhãn hiểu lầm cho lựa chọn này",
        description="Lựa chọn sai chưa có trong rubric.",
        severity="unknown"
    )
    assert badge_unlabeled["badge_id"] == "BADGE_Q3_D"
    assert badge_unlabeled["badge_color"] == "#6B7280"
    assert badge_unlabeled["badge_variant"] == "neutral"

    # Test Root Cause builder
    rc = DiagnosticAgent.build_root_cause_analysis(
        selected_option="B",
        correct_option="A",
        misconception_name="Cộng tử với tử, mẫu với mẫu",
        detailed_explanation="Học sinh cộng trực tiếp mà không quy đồng mẫu.",
        concept_name="Phép cộng phân số"
    )
    assert "B" in rc["observation"]
    assert "A" in rc["observation"]
    assert rc["misconception"] == "Cộng tử với tử, mẫu với mẫu"
    assert len(rc["cot_steps"]) == 4
    assert any("Quan sát" in step for step in rc["cot_steps"])
    assert any("Root Cause" in step for step in rc["cot_steps"])


def test_html_badge_card_rendering():
    """Verify render_misconception_badge_html produces clean, styled HTML markup."""
    badge = MisconceptionBadgeSchema(
        badge_id="BADGE_101",
        misconception_id="101",
        label="Cộng tử với tử, mẫu với mẫu",
        badge_color="#EF4444",
        badge_variant="danger",
        badge_type="misconception",
        severity="high",
        cot_summary="Học sinh cộng trực tiếp tử và mẫu.",
        root_cause="Không quy đồng mẫu trước khi cộng.",
        confidence_score=1.0
    )
    rc = RootCauseAnalysisSchema(
        observation="Học sinh chọn phương án 'B' thay vì đáp án đúng 'A'.",
        misconception="Cộng tử với tử, mẫu với mẫu",
        detailed_explanation="Không quy đồng mẫu trước khi cộng.",
        gap_conclusion="Cần kiểm tra thêm về quy đồng mẫu số.",
        cot_steps=[
            "1. Quan sát: Chọn B thay vì A",
            "2. Phân tích: Cộng tử với tử, mẫu với mẫu",
            "3. Diễn giải: Không quy đồng mẫu số",
            "4. Kết luận: Hổng kiến thức phân số"
        ]
    )

    html = gateway_service.render_misconception_badge_html(badge, rc)
    assert "misconception-badge-card" in html
    assert "#EF4444" in html
    assert "Cộng tử với tử, mẫu với mẫu" in html
    assert "#101" in html
    assert "Mức độ: HIGH" in html
    assert "1. Quan sát: Chọn B thay vì A" in html


def test_filter_incorrect_submissions():
    """Verify filter_incorrect_submissions isolates only wrong answers."""
    items = [
        {"question_id": "Q1", "selected_option": "B", "provided_question": {"correct_option": "A"}},
        {"question_id": "Q2", "selected_option": "A", "provided_question": {"correct_option": "A"}},
        {"question_id": "Q3", "selected_option": "C", "provided_question": {"correct_option": "D"}},
    ]
    incorrect = gateway_service.filter_incorrect_submissions(items)
    assert len(incorrect) == 2
    assert incorrect[0]["question_id"] == "Q1"
    assert incorrect[1]["question_id"] == "Q3"


def test_post_submission_diagnostic_endpoint_success():
    """
    Verify POST /api/v1/assessment/post-submission-diagnostic:
    - Filters wrong answers
    - Generates CoT reasoning and misconception badges
    - Returns <= 500ms
    """
    payload = {
        "student_id": "STUDENT_SE12_001",
        "test_id": "TEST_SE12_101",
        "student_name": "Lê Văn C",
        "use_llm": False,
        "questions": [
            {
                "question_id": "EEDI_Q1",
                "concept_id": "MATH_FRAC_ADD",
                "concept_name": "Phép cộng phân số",
                "question_text": "Tính 1/3 + 1/2",
                "options": {"A": "5/6", "B": "2/5", "C": "1/5", "D": "2/6"},
                "correct_option": "A",
                "misconception_map": {
                    "B": {
                        "misconception_id": "101",
                        "name": "Cộng tử với tử, mẫu với mẫu",
                        "description": "Học sinh cộng trực tiếp tử số và mẫu số không quy đồng.",
                        "severity": "high"
                    }
                },
                "selected_option": "B"  # Sai
            },
            {
                "question_id": "EEDI_Q2",
                "concept_id": "MATH_FRAC_SUB",
                "concept_name": "Phép trừ phân số",
                "question_text": "Tính 3/4 - 1/4",
                "options": {"A": "1/2", "B": "2/4", "C": "1/4", "D": "2/2"},
                "correct_option": "A",
                "selected_option": "A"  # Đúng
            },
            {
                "question_id": "EEDI_Q3",
                "concept_id": "MATH_FRAC_MUL",
                "concept_name": "Phép nhân phân số",
                "question_text": "Tính 1/2 * 3/4",
                "options": {"A": "3/8", "B": "4/6", "C": "2/6", "D": "4/8"},
                "correct_option": "A",
                "misconception_map": {
                    "B": {
                        "misconception_id": "103",
                        "name": "Nhân chéo nhầm lẫn",
                        "description": "Học sinh nhân chéo thay vì nhân tử với tử, mẫu với mẫu.",
                        "severity": "medium"
                    }
                },
                "selected_option": "B"  # Sai
            }
        ]
    }

    start_time = time.perf_counter()
    response = client.post("/api/v1/assessment/post-submission-diagnostic", json=payload)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    # Performance requirement
    assert elapsed_ms <= 500.0, f"Endpoint took {elapsed_ms:.2f}ms > 500ms"
    assert data["processing_time_ms"] <= 500.0

    # Summary
    assert data["status"] == "success"
    assert data["student_id"] == "STUDENT_SE12_001"
    assert data["test_id"] == "TEST_SE12_101"
    assert data["total_questions"] == 3
    assert data["total_incorrect"] == 2
    assert len(data["incorrect_diagnoses"]) == 2
    assert len(data["misconception_badges"]) == 2

    # Check 1st wrong question diagnosis & badge
    q1_diag = data["incorrect_diagnoses"][0]
    assert q1_diag["question_id"] == "EEDI_Q1"
    assert q1_diag["is_correct"] is False
    assert q1_diag["misconception_id"] == "101"
    assert q1_diag["detected_misconception"] == "Cộng tử với tử, mẫu với mẫu"
    assert q1_diag["severity"] == "high"

    # Misconception badge verification
    badge1 = q1_diag["misconception_badge"]
    assert badge1 is not None
    assert badge1["badge_id"] == "BADGE_101"
    assert badge1["label"] == "Cộng tử với tử, mẫu với mẫu"
    assert badge1["badge_color"] == "#EF4444"
    assert badge1["badge_variant"] == "danger"
    assert badge1["severity"] == "high"

    # Root Cause Analysis verification
    rc1 = q1_diag["root_cause_analysis"]
    assert rc1 is not None
    assert "B" in rc1["observation"]
    assert "A" in rc1["observation"]
    assert rc1["misconception"] == "Cộng tử với tử, mẫu với mẫu"
    assert len(rc1["cot_steps"]) >= 4

    # HTML visualization verification
    assert q1_diag["badge_html"] is not None
    assert "misconception-badge-card" in q1_diag["badge_html"]
    assert "#EF4444" in q1_diag["badge_html"]

    # Check 2nd wrong question badge
    badge2 = data["misconception_badges"][1]
    assert badge2["misconception_id"] == "103"
    assert badge2["badge_color"] == "#F59E0B"
    assert badge2["badge_variant"] == "warning"


def test_post_submission_diagnostic_all_correct():
    """Verify behavior when 100% of questions are answered correctly."""
    payload = {
        "student_id": "STUDENT_PERFECT_001",
        "test_id": "TEST_PERFECT_100",
        "questions": [
            {
                "question_id": "Q1",
                "concept_id": "C1",
                "options": {"A": "1", "B": "2"},
                "correct_option": "A",
                "selected_option": "A"
            },
            {
                "question_id": "Q2",
                "concept_id": "C2",
                "options": {"A": "X", "B": "Y"},
                "correct_option": "B",
                "selected_option": "B"
            }
        ]
    }

    response = client.post("/api/v1/assessment/post-submission-diagnostic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_questions"] == 2
    assert data["total_incorrect"] == 0
    assert len(data["incorrect_diagnoses"]) == 0
    assert len(data["misconception_badges"]) == 0


def test_post_submission_diagnostic_unlabeled_misconception():
    """Verify handling when wrong option has no predefined misconception in rubric."""
    payload = {
        "student_id": "STUDENT_UNLABELED_001",
        "test_id": "TEST_UNLABELED_101",
        "submissions": [
            {
                "question_id": "Q_UNLABELED",
                "selected_option": "D",
                "question": {
                    "question_id": "Q_UNLABELED",
                    "concept_id": "C_TEST",
                    "concept_name": "Kiểm tra tổng hợp",
                    "options": {"A": "10", "B": "20", "C": "30", "D": "40"},
                    "correct_option": "A",
                    "misconception_map": {}  # No labeled misconceptions
                }
            }
        ]
    }

    response = client.post("/api/v1/assessment/post-submission-diagnostic", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_incorrect"] == 1
    diag = data["incorrect_diagnoses"][0]
    assert diag["misconception_id"] == "unlabeled"
    assert diag["misconception_badge"] is not None
    assert diag["misconception_badge"]["badge_color"] == "#6B7280"
    assert diag["misconception_badge"]["badge_variant"] == "neutral"
    assert diag["root_cause_analysis"] is not None


def test_assessment_submit_includes_badges():
    """Verify standard /api/v1/assessment/submit also populates misconception_badges."""
    payload = {
        "student_id": "STUDENT_SUBMIT_001",
        "test_id": "TEST_SUBMIT_101",
        "questions": [
            {
                "question_id": "EEDI_SUB_1",
                "concept_id": "C_SUB",
                "concept_name": "Phép nhân phân số",
                "options": {"A": "1/2", "B": "1/4"},
                "correct_option": "A",
                "misconception_map": {
                    "B": {
                        "misconception_id": "301",
                        "name": "Nhân sai mẫu số",
                        "description": "Học sinh nhân sai mẫu số.",
                        "severity": "medium"
                    }
                },
                "selected_option": "B"
            }
        ]
    }

    response = client.post("/api/v1/assessment/submit", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["misconception_badges"]) == 1
    badge = data["misconception_badges"][0]
    assert badge["misconception_id"] == "301"
    assert badge["label"] == "Nhân sai mẫu số"

    wrong_q = data["incorrect_questions"][0]
    assert wrong_q["misconception_badge"]["badge_id"] == "BADGE_301"
    assert wrong_q["root_cause_analysis"] is not None
    assert wrong_q["badge_html"] is not None
