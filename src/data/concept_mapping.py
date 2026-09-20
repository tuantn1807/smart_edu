"""Heuristic & Local Multilingual Semantic Embedding Eedi→Junyi concept mapping.
Never invents Junyi IDs that are not in the loaded graph."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import re
import numpy as np

from src.core.knowledge_graph import KnowledgeGraph

CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'cache'
EMBEDDING_CACHE_FILE = CACHE_DIR / 'junyi_embeddings.json'

STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'into', 'using', 'use', 'etc', 'nth', 'term',
    'simple', 'basic', 'others', 'list', 'data', 'amount', 'facts', 'other', 'between',
    'carry', 'out', 'involving', 'where', 'that', 'this', 'than', 'over', 'more',
}

# English needles in Eedi subject/construct → Chinese needles in Junyi pretty names.
LEXICON: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = [
    (('quadratic', 'inequalit'), ('二次不等式',)),
    (('quadratic', 'equation'), ('一元二次方程式', '二次方程式')),
    (('quadratic', 'graph'), ('二次函數',)),
    (('quadratic', 'sequence'), ('等差數列', '數列')),
    (('quadratic',), ('一元二次', '二次函數', '二次方程式')),
    (('simultaneous',), ('聯立方程式', '二元一次聯立')),
    (('linear', 'inequalit'), ('一元一次不等式', '不等式')),
    (('linear', 'equation'), ('一元一次方程式',)),
    (('linear', 'sequence'), ('等差數列', '數列的規律')),
    (('function', 'machine'), ('函數',)),
    (('completing', 'square'), ('配方', '完全平方式')),
    (('difference', 'squares'), ('平方差',)),
    (('expanding', 'bracket'), ('分配律', '去括號', '多項式乘法')),
    (('factoris',), ('因式分解', '提公因式')),
    (('factoriz',), ('因式分解', '提公因式')),
    (('writing', 'expression'), ('代數式', '列式')),
    (('collecting', 'like'), ('合併同類項', '化簡二元一次式')),
    (('substitut',), ('代入求值', '代數式的值', '代數')),
    (('substitution', 'formula'), ('代入求值', '代數式的值', '函數值')),
    (('bidmas',), ('先乘除後加減', '四則運算', '有括號')),
    (('order', 'operations'), ('先乘除後加減', '四則運算')),
    (('adding', 'subtracting', 'fraction'), ('異分母分數的加減', '同分母分數的加減', '分數的加減')),
    (('multiplying', 'fraction'), ('分數的乘法', '分數乘以')),
    (('dividing', 'fraction'), ('分數的除法', '分數除以')),
    (('equivalent', 'fraction'), ('等值分數', '約分', '擴分')),
    (('ordering', 'fraction'), ('分數的大小比較', '異分母分數的比較')),
    (('mixed', 'number'), ('帶分數與假分數', '帶分數')),
    (('improper', 'fraction'), ('帶分數與假分數', '假分數')),
    (('fraction', 'percent'), ('百分率與分數', '將小數、分數化成百分率')),
    (('fraction', 'decimal'), ('分數和小數', '分數轉換成小數')),
    (('fractions', 'percent'), ('百分率與分數',)),
    (('percentages',), ('百分率', '百分比', '百分')),
    (('percentage',), ('百分率', '百分比', '百分')),
    (('decimals', 'percent'), ('百分率與分數', '小數')),
    (('adding', 'subtracting', 'decimal'), ('小數的加', '小數的減', '多位小數的加減')),
    (('multiplying', 'dividing', 'decimal'), ('小數的乘法', '小數的除法', '小數乘以', '小數除以')),
    (('ordering', 'decimal'), ('小數的比較', '小數的大小')),
    (('place', 'value'), ('位值', '十進位')),
    (('rounding',), ('四捨五入', '概數')),
    (('negative', 'number'), ('正數與負數', '負數', '同號數相加', '異號數相加')),
    (('square', 'root'), ('平方根', '根號')),
    (('cube', 'root'), ('立方',)),
    (('squares', 'cubes'), ('乘方', '平方', '立方')),
    (('laws', 'indices'), ('指數律', '指數')),
    (('standard', 'form'), ('科學記號',)),
    (('area', 'simple'), ('長方形、正方形的面積', '三角形的面積', '面積')),
    (('compound', 'area'), ('複合圖形的面積',)),
    (('perimeter',), ('周長', '周界')),
    (('volume', 'prism'), ('長方體', '柱體的體積', '體積')),
    (('volume',), ('體積', '容積')),
    (('surface', 'area'), ('表面積',)),
    (('angle', 'parallel'), ('平行線', '截角')),
    (('angles', 'triangle'), ('三角形的內角', '三角形')),
    (('angles', 'polygon'), ('內角和', '多邊形')),
    (('measuring', 'angle'), ('量角器', '認識角', '角度')),
    (('basic', 'angle'), ('餘角與補角', '對頂角', '角度')),
    (('circle',), ('圓', '圓周', '扇形')),
    (('triangle', 'right'), ('直角三角形', '畢氏')),
    (('pythagoras',), ('畢氏定理',)),
    (('sohcahtoa',), ('直角三角形',)),
    (('properties', 'rectangle'), ('矩形', '長方形', '正方形與長方形')),
    (('properties', 'quadrilateral'), ('四邊形', '平行四邊形')),
    (('properties', 'triangle'), ('三角形',)),
    (('reflection',), ('線對稱', '對稱')),
    (('rotation',), ('旋轉',)),
    (('translation',), ('平移',)),
    (('enlargement',), ('縮放', '放大與縮小')),
    (('similar',), ('相似',)),
    (('congruen',), ('全等',)),
    (('ratio',), ('比與比例', '比值', '比率')),
    (('direct', 'proportion'), ('正比',)),
    (('inverse', 'proportion'), ('反比',)),
    (('average',), ('平均數', '中位數', '眾數')),
    (('mean', 'median'), ('平均數', '中位數', '眾數')),
    (('pie', 'chart'), ('圓形圖', '圓餅圖')),
    (('bar', 'chart'), ('長條圖',)),
    (('line', 'graph'), ('折線圖',)),
    (('probability',), ('機率',)),
    (('venn',), ('集合',)),
    (('tree', 'diagram'), ('樹狀圖',)),
    (('time',), ('時間', '時刻', '時鐘')),
    (('speed', 'distance'), ('速率', '速度')),
    (('coordinate',), ('坐標', '座標')),
    (('gradient',), ('線型函數', '斜率')),
    (('plotting', 'line'), ('線型函數圖形', '畫出二元一次')),
    (('horizontal', 'vertical'), ('水平線或鉛直線', '坐標')),
    (('algebraic', 'fraction'), ('分數型', '代數')),
    (('rearranging',), ('移項', '等量公理')),
    (('inequalities', 'number', 'line'), ('圖示一元一次不等式', '數線')),
    (('nets',), ('展開圖',)),
    (('polygon',), ('多邊形',)),
    (('parallel', 'line'), ('平行線',)),
    (('place',), ('位值',)),
]


@dataclass
class ConceptMapping:
    mapped: bool
    eedi_concept_id: str
    eedi_concept_name: str
    junyi_concept_id: Optional[str]
    junyi_concept_name: Optional[str]
    method: str
    score: float
    rule: Optional[str]
    top3_junyi_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _query_text(question: Dict[str, Any]) -> str:
    return f"{question.get('subject', '')} {question.get('concept_name', '')}".lower()


def _english_tokens(text: str) -> set:
    return {token for token in re.findall(r'[a-z]{3,}', text.lower()) if token not in STOPWORDS}


class EediJunyiMapper:
    """Map an Eedi construct onto an existing Junyi node, or return unmapped. (Baseline Heuristic)"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self._cache: Dict[str, ConceptMapping] = {}
        self._chinese_index: Dict[str, List[Any]] = {}
        self._node_tokens_cache: Dict[str, set] = {}
        self._build_index()

    def _build_index(self):
        for node in self.knowledge_graph.nodes.values():
            if node.concept_id.startswith('JUNYI_LEVEL'):
                continue
            haystack = f"{node.name} {node.description}"
            for english, chinese in LEXICON:
                for needle in chinese:
                    if needle in haystack:
                        if needle not in self._chinese_index:
                            self._chinese_index[needle] = []
                        self._chinese_index[needle].append(node)
            self._node_tokens_cache[node.concept_id] = _english_tokens(node.name.replace('_', ' '))

    def map_question(self, question: Dict[str, Any]) -> ConceptMapping:
        concept_id = question.get('concept_id', '')
        if concept_id not in self._cache:
            self._cache[concept_id] = self._map(question)
        return self._cache[concept_id]

    def coverage(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        mapped = {q['concept_id'] for q in questions if self.map_question(q).mapped}
        total = {q['concept_id'] for q in questions}
        return {'mapped_constructs': len(mapped), 'total_constructs': len(total)}

    def _map(self, question: Dict[str, Any]) -> ConceptMapping:
        eedi_id = question.get('concept_id', '')
        eedi_name = question.get('concept_name', '')
        unmatched = ConceptMapping(False, eedi_id, eedi_name, None, None, 'unmapped', 0.0, None, top3_junyi_ids=[])
        query = _query_text(question)

        candidates: Dict[str, Tuple[float, str, str, str]] = {}

        # 1. Lexicon matches
        for score, node_id, method, rule in self._lexicon_matches(query):
            if node_id not in candidates or score > candidates[node_id][0]:
                candidates[node_id] = (score, node_id, method, rule)

        # 2. Token overlap matches
        for score, node_id, method, rule in self._token_overlap_matches(query):
            if node_id not in candidates or score > candidates[node_id][0]:
                candidates[node_id] = (score, node_id, method, rule)

        if not candidates:
            return unmatched

        sorted_candidates = sorted(candidates.values(), key=lambda x: x[0], reverse=True)
        top_score, top_node_id, method, rule = sorted_candidates[0]

        node = self.knowledge_graph.nodes.get(top_node_id)
        if node is None:
            return unmatched

        top3 = [nid for _, nid, _, _ in sorted_candidates[:3] if nid in self.knowledge_graph.nodes]
        return ConceptMapping(
            mapped=True,
            eedi_concept_id=eedi_id,
            eedi_concept_name=eedi_name,
            junyi_concept_id=node.concept_id,
            junyi_concept_name=node.name,
            method=method,
            score=round(top_score, 4),
            rule=rule,
            top3_junyi_ids=top3
        )

    def _lexicon_matches(self, query: str) -> List[Tuple[float, str, str, str]]:
        results = []
        for english, chinese in LEXICON:
            if any(needle not in query for needle in english):
                continue
            specificity = (len(english), sum(len(token) for token in english))
            for needle in chinese:
                nodes = self._chinese_index.get(needle, [])
                for node in nodes:
                    haystack = f"{node.name} {node.description}"
                    matched = [c for c in chinese if c in haystack]
                    longest = max(len(c) for c in matched) if matched else len(needle)
                    haystack_mismatch = 0
                    if '分數' in haystack and 'fraction' not in query and not any('分數' in item for item in matched):
                        haystack_mismatch += 1
                    if '小數' in haystack and 'decimal' not in query and not any('小數' in item for item in matched):
                        haystack_mismatch += 1
                    rank = (specificity, longest, -haystack_mismatch, -node.difficulty, node.concept_id)
                    score = 100 + rank[0][0] * 10 + rank[1]
                    rule = '+'.join(english) + '→' + needle
                    results.append((score, node.concept_id, 'lexicon', rule))
        return results

    def _token_overlap_matches(self, query: str) -> List[Tuple[float, str, str, str]]:
        query_tokens = _english_tokens(query)
        if len(query_tokens) < 2:
            return []
        results = []
        for node_id, node_tokens in self._node_tokens_cache.items():
            overlap = query_tokens & node_tokens
            if len(overlap) < 2:
                continue
            ratio = len(overlap) / len(query_tokens)
            if ratio < 0.4:
                continue
            score = ratio * 50
            rule = ','.join(sorted(overlap))
            results.append((score, node_id, 'token_overlap', rule))
        return results


_MODEL_CACHE: Dict[str, Any] = {}
_QUERY_VECTOR_CACHE: Dict[Tuple[str, str], Any] = {}


class SemanticEmbeddingMapper:
    """Multilingual Semantic Embedding mapper using SentenceTransformer or pre-computed embeddings."""

    def __init__(self, knowledge_graph: KnowledgeGraph,
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 min_similarity_threshold: float = 0.15):
        self.knowledge_graph = knowledge_graph
        self.model_name = model_name
        self.min_similarity_threshold = min_similarity_threshold
        self._cache: Dict[str, ConceptMapping] = {}
        self._vector_cache: Dict[str, Any] = {}
        self.model = None
        self._graph_fingerprint = self._compute_graph_fingerprint()
        self._node_ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._init_model()
        self._init_vectors()
        self._build_matrix()

    def _compute_graph_fingerprint(self) -> str:
        node_keys = sorted([nid for nid in self.knowledge_graph.nodes.keys() if not nid.startswith('JUNYI_LEVEL')])
        content = json.dumps(node_keys, ensure_ascii=False)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def _init_model(self):
        if self.model_name not in _MODEL_CACHE:
            try:
                from sentence_transformers import SentenceTransformer
                _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
            except Exception:
                _MODEL_CACHE[self.model_name] = None
        self.model = _MODEL_CACHE[self.model_name]

    def _init_vectors(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        valid_cache = False
        if EMBEDDING_CACHE_FILE.is_file():
            try:
                with EMBEDDING_CACHE_FILE.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                meta = data.get('metadata', {})
                if (meta.get('model_name') == self.model_name and
                    meta.get('graph_fingerprint') == self._graph_fingerprint and
                    'vectors' in data and len(data['vectors']) > 0):
                    self._vector_cache = data['vectors']
                    valid_cache = True
            except Exception:
                valid_cache = False

        if not valid_cache:
            self._precompute_embeddings()

    def _build_matrix(self):
        if not self._vector_cache:
            return
        first_val = next(iter(self._vector_cache.values()))
        if isinstance(first_val, list):
            self._node_ids = [nid for nid in self._vector_cache.keys() if nid in self.knowledge_graph.nodes]
            vectors = [self._vector_cache[nid] for nid in self._node_ids]
            self._matrix = np.array(vectors, dtype=np.float32)
        else:
            self._matrix = None

    def _encode_text_single(self, text: str) -> Any:
        cache_key = (self.model_name, text)
        if cache_key not in _QUERY_VECTOR_CACHE:
            if self.model is not None:
                emb = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
                _QUERY_VECTOR_CACHE[cache_key] = emb[0].tolist()
            else:
                tokens = re.findall(r'\w+', text.lower())
                vec: Dict[str, float] = {}
                for token in tokens:
                    vec[token] = vec.get(token, 0.0) + 1.0
                norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
                _QUERY_VECTOR_CACHE[cache_key] = {k: v / norm for k, v in vec.items()}
        return _QUERY_VECTOR_CACHE[cache_key]

    def _encode_texts(self, texts: List[str]) -> List[Any]:
        return [self._encode_text_single(t) for t in texts]

    def _precompute_embeddings(self):
        self._vector_cache = {}
        node_items = [(nid, n) for nid, n in self.knowledge_graph.nodes.items() if not nid.startswith('JUNYI_LEVEL')]
        if not node_items:
            return

        texts = [f"{n.name} {n.description}" for _, n in node_items]
        if self.model is not None:
            embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()
        else:
            embeddings = []
            for text in texts:
                tokens = re.findall(r'\w+', text.lower())
                vec: Dict[str, float] = {}
                for token in tokens:
                    vec[token] = vec.get(token, 0.0) + 1.0
                norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
                embeddings.append({k: v / norm for k, v in vec.items()})

        for (nid, _), vec in zip(node_items, embeddings):
            self._vector_cache[nid] = vec

        cache_data = {
            "metadata": {
                "model_name": self.model_name,
                "model_version": "v1",
                "graph_fingerprint": self._graph_fingerprint,
                "embedding_dimension": len(embeddings[0]) if isinstance(embeddings[0], list) else 0
            },
            "vectors": self._vector_cache
        }
        with EMBEDDING_CACHE_FILE.open('w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)

    def map_question(self, question: Dict[str, Any]) -> ConceptMapping:
        concept_id = question.get('concept_id', '')
        if concept_id not in self._cache:
            self._cache[concept_id] = self._map(question)
        return self._cache[concept_id]

    def _map(self, question: Dict[str, Any]) -> ConceptMapping:
        eedi_id = question.get('concept_id', '')
        eedi_name = question.get('concept_name', '')
        query = _query_text(question)
        unmatched = ConceptMapping(False, eedi_id, eedi_name, None, None, 'semantic_unmapped', 0.0, None, top3_junyi_ids=[])

        q_vec = self._encode_text_single(query)
        scores: List[Tuple[float, str]] = []

        if self._matrix is not None and isinstance(q_vec, list):
            q_arr = np.array(q_vec, dtype=np.float32)
            sims = np.dot(self._matrix, q_arr)
            for idx, sim in enumerate(sims):
                if sim > 0.0:
                    scores.append((float(sim), self._node_ids[idx]))
        elif isinstance(q_vec, dict):
            for node_id, node_vec in self._vector_cache.items():
                if node_id not in self.knowledge_graph.nodes:
                    continue
                if isinstance(node_vec, dict):
                    sim = sum(v * node_vec.get(k, 0.0) for k, v in q_vec.items())
                else:
                    sim = 0.0
                if sim > 0.0:
                    scores.append((sim, node_id))

        if not scores:
            return unmatched

        scores.sort(key=lambda x: x[0], reverse=True)
        best_sim, best_node_id = scores[0]

        if best_sim < self.min_similarity_threshold:
            return unmatched

        node = self.knowledge_graph.nodes.get(best_node_id)
        if node is None:
            return unmatched

        top3 = [nid for _, nid in scores[:3] if nid in self.knowledge_graph.nodes]
        return ConceptMapping(
            mapped=True,
            eedi_concept_id=eedi_id,
            eedi_concept_name=eedi_name,
            junyi_concept_id=node.concept_id,
            junyi_concept_name=node.name,
            method='semantic_embedding',
            score=round(best_sim, 4),
            rule=f"CosineSim={best_sim:.4f}",
            top3_junyi_ids=top3
        )


class HybridConceptMapper:
    """Hybrid Mapper combining Lexicon rule specificity with Semantic Vector Similarity."""

    def __init__(self, knowledge_graph: KnowledgeGraph,
                 baseline_mapper: Optional[EediJunyiMapper] = None,
                 semantic_mapper: Optional[SemanticEmbeddingMapper] = None):
        self.knowledge_graph = knowledge_graph
        self.baseline_mapper = baseline_mapper or EediJunyiMapper(knowledge_graph)
        self.semantic_mapper = semantic_mapper or SemanticEmbeddingMapper(knowledge_graph)
        self._cache: Dict[str, ConceptMapping] = {}

    def map_question(self, question: Dict[str, Any]) -> ConceptMapping:
        concept_id = question.get('concept_id', '')
        if concept_id not in self._cache:
            self._cache[concept_id] = self._map(question)
        return self._cache[concept_id]

    def _map(self, question: Dict[str, Any]) -> ConceptMapping:
        base_res = self.baseline_mapper.map_question(question)
        sem_res = self.semantic_mapper.map_question(question)

        norm_base_score = min(base_res.score / 150.0, 1.0) if base_res.mapped else 0.0
        norm_sem_score = sem_res.score if sem_res.mapped else 0.0

        candidate_scores: Dict[str, float] = {}

        if base_res.mapped and base_res.top3_junyi_ids:
            for rank_idx, nid in enumerate(base_res.top3_junyi_ids):
                rank_weight = 1.0 - (rank_idx * 0.15)
                candidate_scores[nid] = candidate_scores.get(nid, 0.0) + norm_base_score * rank_weight * 0.6

        if sem_res.mapped and sem_res.top3_junyi_ids:
            for rank_idx, nid in enumerate(sem_res.top3_junyi_ids):
                rank_weight = 1.0 - (rank_idx * 0.15)
                candidate_scores[nid] = candidate_scores.get(nid, 0.0) + norm_sem_score * rank_weight * 0.4

        if not candidate_scores:
            return base_res

        ranked_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        best_nid, combined_score = ranked_candidates[0]

        top3 = [nid for nid, _ in ranked_candidates[:3] if nid in self.knowledge_graph.nodes]
        node = self.knowledge_graph.nodes.get(best_nid)
        if node is None:
            return base_res

        method = 'hybrid_lexicon_semantic' if base_res.mapped else 'hybrid_semantic_fallback'
        rule = base_res.rule if base_res.mapped else sem_res.rule

        return ConceptMapping(
            mapped=True,
            eedi_concept_id=question.get('concept_id', ''),
            eedi_concept_name=question.get('concept_name', ''),
            junyi_concept_id=node.concept_id,
            junyi_concept_name=node.name,
            method=method,
            score=round(combined_score, 4),
            rule=rule,
            top3_junyi_ids=top3
        )
