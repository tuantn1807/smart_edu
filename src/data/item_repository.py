"""
Eedi Item Repository for Dynamic ZPD Question Selection.
Provides query and selection capabilities for real Eedi 2024 diagnostic questions.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
from src.data.dataset_loaders import EediDatasetLoader


class EediItemRepository:
    """Repository indexing real Eedi questions by construct, misconception, and question ID."""

    _instance = None

    def __init__(self, questions: Optional[List[Dict[str, Any]]] = None):
        if questions is None:
            questions = EediDatasetLoader.load_questions()
        self.questions = questions
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._by_construct: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._by_misconception_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._by_misconception_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._build_index()

    def _build_index(self):
        for q in self.questions:
            qid = q.get('question_id', '')
            if qid:
                self._by_id[qid] = q
            source_qid = q.get('source_question_id', '')
            if source_qid:
                self._by_id[source_qid] = q

            cid = q.get('concept_id', '')
            if cid:
                self._by_construct[cid].append(q)
            construct_id = q.get('construct_id', '')
            if construct_id:
                self._by_construct[construct_id].append(q)
                self._by_construct[f"EEDI_CONSTRUCT_{construct_id}"].append(q)

            misc_map = q.get('misconception_map', {})
            for opt, mdata in misc_map.items():
                mid = mdata.get('misconception_id', '')
                mname = mdata.get('name', '').lower()
                if mid:
                    self._by_misconception_id[mid].append(q)
                if mname:
                    self._by_misconception_name[mname].append(q)

    def get_question_by_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve question by Eedi question ID or source question ID."""
        return self._by_id.get(question_id)

    def find_by_construct(self, construct_id_or_concept_id: str) -> List[Dict[str, Any]]:
        """Find all Eedi questions matching a construct ID or concept ID."""
        return self._by_construct.get(construct_id_or_concept_id, [])

    def find_by_misconception(self, misconception_id_or_name: str) -> List[Dict[str, Any]]:
        """Find all Eedi questions matching a misconception ID or name."""
        if misconception_id_or_name in self._by_misconception_id:
            return self._by_misconception_id[misconception_id_or_name]
        query = misconception_id_or_name.lower()
        matches = []
        for name, qlist in self._by_misconception_name.items():
            if query in name or name in query:
                matches.extend(qlist)
        return matches

    def select_item(
        self,
        concept_id: Optional[str] = None,
        misconception_id_or_name: Optional[str] = None,
        exclude_ids: Optional[List[str]] = None,
        action_type: str = "practice"
    ) -> Dict[str, Any]:
        """
        Selects a real Eedi question based on concept, misconception, or fallback.
        Guarantees returning a real Eedi question existing in the dataset.
        """
        exclude_set = set(exclude_ids or [])

        # 1. Try misconception match if remediating
        if misconception_id_or_name:
            candidates = self.find_by_misconception(misconception_id_or_name)
            unseen = [q for q in candidates if q['question_id'] not in exclude_set]
            if unseen:
                return unseen[0]
            if candidates:
                return candidates[0]

        # 2. Try construct match
        if concept_id:
            candidates = self.find_by_construct(concept_id)
            unseen = [q for q in candidates if q['question_id'] not in exclude_set]
            if unseen:
                return unseen[0]
            if candidates:
                return candidates[0]

        # 3. Fallback: Any unseen question in dataset
        unseen_all = [q for q in self.questions if q['question_id'] not in exclude_set]
        if unseen_all:
            return unseen_all[0]

        # 4. Global fallback if all questions excluded
        return self.questions[0]
