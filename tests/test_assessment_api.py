"""
Comprehensive Test Suite for Online Assessment REST API Gateway (SE-11).
Tests REST endpoints: /api/v1/assessment/submit, /api/v1/assessment/diagnose-submission, /health.
Verifies performance criteria (≤ 500ms rule-based), state persistence, misconception badging,
and ZPD remediation path structure.
"""

import time
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.routes import gateway_service

client = TestClient(app)


def test_health_check_endpoint():
    """Verify health check endpoint returns 200 and system details."""
    response = client.get("/api/v1/assessment/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "knowledge_graph_nodes" in data
    assert "indexed_questions" in data


def test_submit_assessment_rule_based_performance_and_diagnosis(tmp_path):
    """
    Verify POST /api/v1/assessment/submit processes payload ≤ 500ms for rule-based,
    returns valid scores, misconception badges, mastery levels, and ZPD remediation path.
    """
    db_file = str(tmp_path / "test_api_learner_state.db")
    gateway_service.set_db_path(db_file)

    payload = {
        "student_id": "STUDENT_AZOTA_001",
        "test_id": "TEST_AZOTA_101",
        "student_name": "Nguyễn Văn A",
        "use_llm": False,
        "questions": [
            {
                "question_id": "EEDI_1",
                "concept_id": "EEDI_CONSTRUCT_1",
                "concept_name": "Phép cộng phân số",
                "question_text": "Tính 1/2 + 1/3",
                "options": {"A": "5/6", "B": "2/5", "C": "1/5", "D": "3/5"},
                "correct_option": "A",
                "misconception_map": {
                    "B": {
                        "misconception_id": "101",
                        "name": "Cộng tử với tử, mẫu với mẫu",
                        "description": "Học sinh cộng trực tiếp tử số và mẫu số không quy đồng."
                    }
                },
                "selected_option": "B"  # Incorrect selection
            },
            {
                "question_id": "EEDI_2",
                "concept_id": "EEDI_CONSTRUCT_2",
                "concept_name": "Phép trừ phân số",
                "question_text": "Tính 3/4 - 1/4",
                "options": {"A": "1/2", "B": "2/4", "C": "1/4", "D": "2/2"},
                "correct_option": "A",
                "selected_option": "A"  # Correct selection
            }
        ]
    }

    start_time = time.perf_counter()
    response = client.post("/api/v1/assessment/submit", json=payload)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()

    # Performance requirement verification: ≤ 500ms for rule-based
    assert elapsed_ms <= 500.0, f"API response time ({elapsed_ms:.2f}ms) exceeded 500ms target!"
    assert data["processing_time_ms"] <= 500.0

    # Test summary verification
    assert data["test_id"] == "TEST_AZOTA_101"
    assert data["student_id"] == "STUDENT_AZOTA_001"
    assert data["total_questions"] == 2
    assert data["correct_count"] == 1
    assert data["incorrect_count"] == 1
    assert data["score_percentage"] == 50.0
    assert data["score_scale_10"] == 5.0
    assert data["passed"] is True

    # Question diagnoses verification
    assert len(data["question_diagnoses"]) == 2
    assert len(data["incorrect_questions"]) == 1
    wrong_q = data["incorrect_questions"][0]
    assert wrong_q["question_id"] == "EEDI_1"
    assert wrong_q["is_correct"] is False
    assert wrong_q["detected_misconception"] == "Cộng tử với tử, mẫu với mẫu"
    assert wrong_q["misconception_id"] == "101"
    assert "CoT" in wrong_q["cot_explanation"] or "Root Cause" in wrong_q["cot_explanation"] or wrong_q["cot_explanation"] != ""

    # ZPD Remediation path & mastery check
    assert isinstance(data["mastery_summary"], dict)
    assert isinstance(data["zpd_remediation_path"], list)
    assert len(data["zpd_remediation_path"]) >= 1


def test_diagnose_submission_endpoint():
    """Verify POST /api/v1/assessment/diagnose-submission endpoint functionality."""
    payload = {
        "student_id": "STUDENT_STUDY4_002",
        "test_id": "TEST_STUDY4_202",
        "student_name": "Trần Thị B",
        "use_llm": False,
        "submissions": [
            {
                "question_id": "EEDI_10",
                "selected_option": "C",
                "question": {
                    "question_id": "EEDI_10",
                    "concept_id": "EEDI_CONSTRUCT_5",
                    "concept_name": "Phép nhân phân số",
                    "options": {"A": "1/6", "B": "2/6", "C": "3/6", "D": "4/6"},
                    "correct_option": "A",
                    "misconception_map": {
                        "C": {
                            "misconception_id": "202",
                            "name": "Nhân chéo nhầm lẫn",
                            "description": "Học sinh nhân chéo phân số thay vì nhân tử với tử, mẫu với mẫu."
                        }
                    }
                }
            }
        ]
    }

    response = client.post("/api/v1/assessment/diagnose-submission", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "STUDENT_STUDY4_002"
    assert data["total_questions"] == 1
    assert data["correct_count"] == 0
    assert data["passed"] is False
    assert len(data["incorrect_questions"]) == 1
    assert data["incorrect_questions"][0]["misconception_id"] == "202"


def test_state_persistence_across_multiple_submissions(tmp_path):
    """Verify student state (mastery, answered questions) persists across multiple submissions."""
    db_file = str(tmp_path / "persistence_api.db")
    gateway_service.set_db_path(db_file)

    payload_1 = {
        "student_id": "STUDENT_PERSIST_001",
        "test_id": "EXAM_1",
        "student_name": "Lê Văn C",
        "questions": [
            {
                "question_id": "Q_P1",
                "concept_id": "CONCEPT_ALG_1",
                "concept_name": "Giải phương trình bậc nhất",
                "options": {"A": "x=1", "B": "x=2", "C": "x=3", "D": "x=4"},
                "correct_option": "A",
                "selected_option": "A"
            }
        ]
    }

    res1 = client.post("/api/v1/assessment/submit", json=payload_1)
    assert res1.status_code == 200
    mastery_after_test1 = res1.json()["mastery_summary"].get("CONCEPT_ALG_1", 0.0)
    assert mastery_after_test1 > 0.0

    payload_2 = {
        "student_id": "STUDENT_PERSIST_001",
        "test_id": "EXAM_2",
        "student_name": "Lê Văn C",
        "questions": [
            {
                "question_id": "Q_P2",
                "concept_id": "CONCEPT_ALG_1",
                "concept_name": "Giải phương trình bậc nhất",
                "options": {"A": "x=1", "B": "x=2", "C": "x=3", "D": "x=4"},
                "correct_option": "A",
                "selected_option": "A"
            }
        ]
    }

    res2 = client.post("/api/v1/assessment/submit", json=payload_2)
    assert res2.status_code == 200
    mastery_after_test2 = res2.json()["mastery_summary"].get("CONCEPT_ALG_1", 0.0)

    # Mastery should accumulate from previous submission
    assert mastery_after_test2 > mastery_after_test1


def test_invalid_payload_handling():
    """Verify validation errors for invalid payloads."""
    # Missing required fields student_id and test_id
    res = client.post("/api/v1/assessment/submit", json={})
    assert res.status_code == 422

    # Payload missing questions list
    res_empty = client.post("/api/v1/assessment/submit", json={
        "student_id": "S1",
        "test_id": "T1"
    })
    assert res_empty.status_code == 400
    assert "missing 'questions' or 'submissions'" in res_empty.json()["detail"]
