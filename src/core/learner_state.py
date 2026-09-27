"""
Centralized Learner State Management for PAAF (Pedagogical Agentic AI Framework)
Tracks student mastery levels, identified misconceptions, active learning path, and interaction memory history.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import datetime


@dataclass
class MisconceptionRecord:
    concept_id: str
    misconception_name: str
    description: str
    detected_at: str
    severity: str  # 'low', 'medium', 'high'
    resolved: bool = False


@dataclass
class LearningPathStep:
    step_id: int
    concept_id: str
    concept_name: str
    action_type: str  # 'review_prerequisite', 'learn_concept', 'practice_exercise', 'advanced_challenge', 'remediate_misconception'
    description: str
    status: str = 'pending'  # 'pending', 'in_progress', 'completed'
    question_id: Optional[str] = None
    question_details: Optional[Dict[str, Any]] = None


class LearnerState:
    def __init__(self, student_id: str, student_name: str = "Học sinh"):
        self.student_id = student_id
        self.student_name = student_name
        self.mastery_levels: Dict[str, float] = {}  # concept_id -> score in [0.0, 1.0]
        self.misconceptions: List[MisconceptionRecord] = []
        self.active_learning_path: List[LearningPathStep] = []
        self.interaction_history: List[Dict[str, Any]] = []
        self.consecutive_correct: int = 0
        self.consecutive_incorrect: int = 0
        self.answered_questions: List[str] = []
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now().isoformat()

    def record_question_result(self, question_id: Optional[str], is_correct: bool):
        """Record outcome of answering a question and update consecutive streaks."""
        if question_id and question_id not in self.answered_questions:
            self.answered_questions.append(question_id)

        if is_correct:
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
        else:
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0
        self.updated_at = datetime.datetime.now().isoformat()

    def set_concept_mastery(self, concept_id: str, score: float):
        """Update mastery score for a concept [0.0, 1.0]."""
        self.mastery_levels[concept_id] = max(0.0, min(1.0, score))
        self.updated_at = datetime.datetime.now().isoformat()

    def get_concept_mastery(self, concept_id: str) -> float:
        """Get mastery score for a concept, default to 0.0 if unassessed."""
        return self.mastery_levels.get(concept_id, 0.0)

    def update_mastery(self, concept_id: str, is_correct: bool, delta_correct: float = 0.2, delta_incorrect: float = 0.3) -> float:
        """Update mastery score for concept_id based on answer correctness."""
        current = self.get_concept_mastery(concept_id)
        if is_correct:
            new_score = min(1.0, current + delta_correct)
        else:
            new_score = max(0.0, current - delta_incorrect)
        self.set_concept_mastery(concept_id, new_score)
        return new_score

    def propagate_mastery_loss(self, target_concept_id: str, knowledge_graph: Any, attenuation: float = 0.1, threshold: float = 0.6) -> List[str]:
        """
        Propagate mastery reduction (-attenuation) to direct prerequisites
        that are not yet mastered (mastery < threshold) when student makes a mistake on target_concept_id.
        """
        if hasattr(knowledge_graph, "get_direct_prerequisites"):
            direct_prereqs = knowledge_graph.get_direct_prerequisites(target_concept_id)
        elif hasattr(knowledge_graph, "get_all_ancestors"):
            direct_prereqs = knowledge_graph.get_all_ancestors(target_concept_id)
        else:
            return []

        updated_prereqs = []
        for prereq_id in direct_prereqs:
            current = self.get_concept_mastery(prereq_id)
            if current < threshold:
                new_score = max(0.0, current - attenuation)
                self.set_concept_mastery(prereq_id, new_score)
                updated_prereqs.append(prereq_id)
        return updated_prereqs

    def add_misconception(self, concept_id: str, misconception_name: str, description: str, severity: str = 'medium'):
        """Record a detected knowledge gap / misconception."""
        record = MisconceptionRecord(
            concept_id=concept_id,
            misconception_name=misconception_name,
            description=description,
            detected_at=datetime.datetime.now().isoformat(),
            severity=severity
        )
        self.misconceptions.append(record)
        self.update_mastery(concept_id, is_correct=False, delta_incorrect=0.3)
        self.updated_at = datetime.datetime.now().isoformat()

    def set_learning_path(self, steps: List[LearningPathStep]):
        """Update active learning path."""
        self.active_learning_path = steps
        self.updated_at = datetime.datetime.now().isoformat()

    def add_interaction(self, role: str, message: str, agent_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Append to multi-turn dialogue history."""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "role": role,  # 'user', 'assistant', 'system'
            "agent": agent_name or "System",
            "message": message,
            "metadata": metadata or {}
        }
        self.interaction_history.append(entry)
        self.updated_at = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for inspection or persistence."""
        return {
            "student_id": self.student_id,
            "student_name": self.student_name,
            "mastery_levels": self.mastery_levels,
            "consecutive_correct": self.consecutive_correct,
            "consecutive_incorrect": self.consecutive_incorrect,
            "answered_questions": list(self.answered_questions),
            "misconceptions": [
                {
                    "concept_id": m.concept_id,
                    "name": m.misconception_name,
                    "misconception_name": m.misconception_name,
                    "description": m.description,
                    "detected_at": m.detected_at,
                    "severity": m.severity,
                    "resolved": m.resolved
                } for m in self.misconceptions
            ],
            "active_learning_path": [
                {
                    "step_id": s.step_id,
                    "concept_id": s.concept_id,
                    "concept_name": s.concept_name,
                    "action_type": s.action_type,
                    "description": s.description,
                    "status": s.status,
                    "question_id": s.question_id,
                    "question_details": s.question_details
                } for s in self.active_learning_path
            ],
            "history_count": len(self.interaction_history),
            "interaction_history": self.interaction_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearnerState":
        """Reconstruct a LearnerState instance from a dictionary."""
        state = cls(
            student_id=data["student_id"],
            student_name=data.get("student_name", "Học sinh")
        )
        state.mastery_levels = dict(data.get("mastery_levels", {}))
        state.consecutive_correct = data.get("consecutive_correct", 0)
        state.consecutive_incorrect = data.get("consecutive_incorrect", 0)
        state.answered_questions = list(data.get("answered_questions", []))
        state.created_at = data.get("created_at", datetime.datetime.now().isoformat())
        state.updated_at = data.get("updated_at", datetime.datetime.now().isoformat())

        state.misconceptions = []
        for m in data.get("misconceptions", []):
            rec = MisconceptionRecord(
                concept_id=m.get("concept_id", ""),
                misconception_name=m.get("misconception_name") or m.get("name", ""),
                description=m.get("description", ""),
                detected_at=m.get("detected_at", datetime.datetime.now().isoformat()),
                severity=m.get("severity", "medium"),
                resolved=m.get("resolved", False)
            )
            state.misconceptions.append(rec)

        state.active_learning_path = []
        for s in data.get("active_learning_path", []):
            step = LearningPathStep(
                step_id=s.get("step_id", 0),
                concept_id=s.get("concept_id", ""),
                concept_name=s.get("concept_name", ""),
                action_type=s.get("action_type", ""),
                description=s.get("description", ""),
                status=s.get("status", "pending"),
                question_id=s.get("question_id"),
                question_details=s.get("question_details")
            )
            state.active_learning_path.append(step)

        state.interaction_history = list(data.get("interaction_history", []))
        return state

