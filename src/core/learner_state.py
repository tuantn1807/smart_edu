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
    action_type: 'str'  # 'review_prerequisite', 'learn_concept', 'practice_exercise', 'advanced_challenge'
    description: str
    status: str = 'pending'  # 'pending', 'in_progress', 'completed'


class LearnerState:
    def __init__(self, student_id: str, student_name: str = "Học sinh"):
        self.student_id = student_id
        self.student_name = student_name
        self.mastery_levels: Dict[str, float] = {}  # concept_id -> score in [0.0, 1.0]
        self.misconceptions: List[MisconceptionRecord] = []
        self.active_learning_path: List[LearningPathStep] = []
        self.interaction_history: List[Dict[str, Any]] = []
        self.created_at = datetime.datetime.now().isoformat()
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

    def propagate_mastery_loss(self, target_concept_id: str, knowledge_graph: Any, attenuation: float = 0.1) -> List[str]:
        """
        Propagate mastery reduction (-attenuation) upstream to prerequisite ancestors
        when student makes a mistake on target_concept_id.
        """
        if not hasattr(knowledge_graph, "get_all_ancestors"):
            return []
        ancestors = knowledge_graph.get_all_ancestors(target_concept_id)
        updated_prereqs = []
        for prereq_id in ancestors:
            current = self.get_concept_mastery(prereq_id)
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
            "misconceptions": [
                {
                    "concept_id": m.concept_id,
                    "name": m.misconception_name,
                    "description": m.description,
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
                    "status": s.status
                } for s in self.active_learning_path
            ],
            "history_count": len(self.interaction_history)
        }
