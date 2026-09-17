"""
Knowledge Graph Agent: Traverses prerequisite concept graphs to identify unmastered upstream concepts.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.core.knowledge_graph import KnowledgeGraph, ConceptNode
from src.core.learner_state import LearnerState


class KGAgent(BaseAgent):
    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(
            name="KnowledgeGraphAgent",
            role_description="Truy vết cây phụ thuộc kiến thức (Prerequisite Tree Traversal) để phát hiện các khái niệm nền tảng bị đứt gãy."
        )
        self.knowledge_graph = knowledge_graph

    def process(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input expects:
        - 'target_concept_id': str
        - 'learner_state': LearnerState
        """
        target_concept_id = input_data.get("target_concept_id")
        mapping = input_data.get("mapping") or {}
        learner_state: LearnerState = context.get("learner_state")
        mapping_available = bool(mapping.get("mapped")) or (
            not mapping and target_concept_id in self.knowledge_graph.nodes
        )

        print(self.format_log(f"Đang truy vết cây phụ thuộc kiến thức Junyi cho '{target_concept_id}'..."))

        mastery_levels = learner_state.mastery_levels
        empty = {
            "prerequisites_available": self.knowledge_graph.prerequisites_available,
            "mapping_available": False,
            "graph_source": self.knowledge_graph.source,
            "graph_kind": self.knowledge_graph.graph_kind,
            "target_concept_id": target_concept_id,
            "all_prerequisites_count": 0,
            "unmastered_prerequisites": [],
            "mapping": mapping,
            "analysis_summary": "",
        }

        if not mapping_available or target_concept_id not in self.knowledge_graph.nodes:
            empty["analysis_summary"] = (
                "Eedi construct chưa ánh xạ được sang node Junyi có trong đồ thị; "
                "không duyệt tiên quyết giả."
            )
            return empty

        if not self.knowledge_graph.prerequisites_available:
            empty["mapping_available"] = True
            empty["analysis_summary"] = "Dataset chưa có quan hệ tiên quyết; không thể đánh giá lỗ hổng nền tảng."
            return empty

        unmastered_prereqs: List[ConceptNode] = self.knowledge_graph.find_unmastered_prerequisites(
            target_concept_id=target_concept_id,
            mastery_levels=mastery_levels,
            threshold=0.6
        )
        all_ancestors = self.knowledge_graph.get_all_ancestors(target_concept_id)
        return {
            "prerequisites_available": True,
            "mapping_available": True,
            "graph_source": self.knowledge_graph.source,
            "graph_kind": self.knowledge_graph.graph_kind,
            "target_concept_id": target_concept_id,
            "all_prerequisites_count": len(all_ancestors),
            "unmastered_prerequisites": [
                {
                    "concept_id": node.concept_id,
                    "name": node.name,
                    "difficulty": node.difficulty,
                    "current_mastery": mastery_levels.get(node.concept_id, 0.0),
                    "description": node.description
                } for node in unmastered_prereqs
            ],
            "mapping": mapping,
            "analysis_summary": (
                f"Đã kiểm tra tổng cộng {len(all_ancestors)} khái niệm tiên quyết Junyi cho '{target_concept_id}'. "
                f"Phát hiện {len(unmastered_prereqs)} khái niệm nền tảng chưa đạt yêu cầu (< 60%)."
            ),
        }
