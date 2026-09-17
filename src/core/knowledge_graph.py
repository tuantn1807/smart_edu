"""
Prerequisite Knowledge Graph for PAAF.
Models concept nodes, prerequisite dependency edges, and provides graph traversal for gap detection.
Inspired by Junyi Academy Concept Dependency Graph & XES3G5M schema.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ConceptNode:
    concept_id: str
    name: str
    subject: str
    difficulty: float  # 0.0 (easiest) to 1.0 (hardest)
    description: str
    prerequisites: Set[str] = field(default_factory=set)  # concept_ids of required prerequisites
    subconcepts: Set[str] = field(default_factory=set)


class KnowledgeGraph:
    def __init__(self, subject_name: str = "Toán Học", prerequisites_available: bool = True,
                 source: str = "", graph_kind: str = ""):
        self.subject_name = subject_name
        self.prerequisites_available = prerequisites_available
        self.source = source
        self.graph_kind = graph_kind
        self.nodes: Dict[str, ConceptNode] = {}

    def edge_count(self) -> int:
        return sum(len(node.prerequisites) for node in self.nodes.values())

    def add_concept(self, concept_id: str, name: str, difficulty: float, description: str, prerequisites: Optional[List[str]] = None) -> ConceptNode:
        """Add or update a concept in the Knowledge Graph."""
        prereqs = set(prerequisites) if prerequisites else set()
        node = ConceptNode(
            concept_id=concept_id,
            name=name,
            subject=self.subject_name,
            difficulty=difficulty,
            description=description,
            prerequisites=prereqs
        )
        self.nodes[concept_id] = node
        return node

    def add_prerequisite(self, concept_id: str, prerequisite_id: str):
        """Add a prerequisite relationship: prerequisite_id -> concept_id."""
        if concept_id in self.nodes and prerequisite_id in self.nodes:
            self.nodes[concept_id].prerequisites.add(prerequisite_id)

    def get_all_ancestors(self, concept_id: str) -> Set[str]:
        """Recursively find all prerequisite concepts required for concept_id."""
        ancestors: Set[str] = set()
        visited: Set[str] = set()

        def dfs(cid: str):
            if cid in visited or cid not in self.nodes:
                return
            visited.add(cid)
            for prereq_id in self.nodes[cid].prerequisites:
                ancestors.add(prereq_id)
                dfs(prereq_id)

        dfs(concept_id)
        return ancestors

    def find_unmastered_prerequisites(self, target_concept_id: str, mastery_levels: Dict[str, float], threshold: float = 0.6) -> List[ConceptNode]:
        """
        Traverse graph to find upstream prerequisites that student has NOT mastered (mastery < threshold).
        Ordered from foundational (root) to target concept.
        """
        all_prereqs = self.get_all_ancestors(target_concept_id)
        unmastered: List[ConceptNode] = []

        for p_id in all_prereqs:
            if p_id not in self.nodes:
                continue
            score = mastery_levels.get(p_id, 0.0)
            if score < threshold:
                unmastered.append(self.nodes[p_id])

        # Sort unmastered prerequisites by difficulty ascending (foundational first)
        unmastered.sort(key=lambda n: n.difficulty)
        return unmastered

    def to_dict(self) -> Dict[str, Any]:
        """Export graph structure for visualization or API responses."""
        return {
            "subject": self.subject_name,
            "concept_count": len(self.nodes),
            "concepts": {
                cid: {
                    "name": n.name,
                    "difficulty": n.difficulty,
                    "prerequisites": list(n.prerequisites)
                } for cid, n in self.nodes.items()
            }
        }
