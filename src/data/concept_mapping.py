"""Heuristic & Local Semantic Embedding Eedi→Junyi concept mapping. Never invents Junyi IDs that are not in the loaded graph."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import re

from src.core.knowledge_graph import KnowledgeGraph

CACHE_DIR = Path(__file__).resolve().parents[2] / 'data' / 'cache'
EMBEDDING_CACHE_FILE = CACHE_DIR / 'junyi_embeddings.json'

STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'into', 'using', 'use', 'etc', 'nth', 'term',
    'simple', 'basic', 'others', 'list', 'data', 'amount', 'facts', 'other', 'between',
    'carry', 'out', 'involving', 'where', 'that', 'this', 'than', 'over', 'more',
}

# English needles in Eedi subject/construct → Chinese needles in Junyi pretty names.
# More specific rules must appear first.
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
        unmatched = ConceptMapping(False, eedi_id, eedi_name, None, None, 'unmapped', 0.0, None)
        query = _query_text(question)
        lexicon_hit = self._lexicon_match(query)
        overlap_hit = self._token_overlap_match(query)
        hit = lexicon_hit if lexicon_hit and (not overlap_hit or lexicon_hit[0] >= overlap_hit[0]) else overlap_hit
        if not hit:
            return unmatched
        score, node_id, method, rule = hit
        node = self.knowledge_graph.nodes.get(node_id)
        if node is None:
            return unmatched
        
        top3 = [node.concept_id]
        return ConceptMapping(True, eedi_id, eedi_name, node.concept_id, node.name, method, score, rule, top3_junyi_ids=top3)

    def _lexicon_match(self, query: str):
        best = None
        for english, chinese in LEXICON:
            if any(needle not in query for needle in english):
                continue
            specificity = (len(english), sum(len(token) for token in english))
            for node in self.knowledge_graph.nodes.values():
                haystack = f"{node.name} {node.description}"
                matched = [needle for needle in chinese if needle in haystack]
                if not matched:
                    continue
                if node.concept_id.startswith('JUNYI_LEVEL'):
                    continue
                longest = max(len(needle) for needle in matched)
                haystack_mismatch = 0
                if '分數' in haystack and 'fraction' not in query and not any('分數' in item for item in matched):
                    haystack_mismatch += 1
                if '小數' in haystack and 'decimal' not in query and not any('小數' in item for item in matched):
                    haystack_mismatch += 1
                rank = (specificity, longest, -haystack_mismatch, -node.difficulty, node.concept_id)
                if best is None or rank > best[0]:
                    best = (rank, node.concept_id, 'lexicon', '+'.join(english) + '→' + matched[0])
        if best is None:
            return None
        rank, node_id, method, rule = best
        score = 100 + rank[0][0] * 10 + rank[1]
        return score, node_id, method, rule

    def _token_overlap_match(self, query: str):
        query_tokens = _english_tokens(query)
        if len(query_tokens) < 2:
            return None
        best = None
        for node in self.knowledge_graph.nodes.values():
            node_tokens = _english_tokens(node.name.replace('_', ' '))
            overlap = query_tokens & node_tokens
            if len(overlap) < 2:
                continue
            score = len(overlap) / len(query_tokens)
            rank = (score, len(overlap), -node.difficulty, node.concept_id)
            if best is None or rank > best[0]:
                best = (rank, node.concept_id, 'token_overlap', ','.join(sorted(overlap)))
        if best is None or best[0][0] < 0.4:
            return None
        rank, node_id, method, rule = best
        return rank[0] * 50, node_id, method, rule


class SemanticEmbeddingMapper:
    """Local Semantic Embedding mapper combining pre-computed vectors and Cosine Similarity."""

    def __init__(self, knowledge_graph: KnowledgeGraph, min_similarity_threshold: float = 0.35):
        self.knowledge_graph = knowledge_graph
        self.min_similarity_threshold = min_similarity_threshold
        self._cache: Dict[str, ConceptMapping] = {}
        self._vector_cache: Dict[str, Any] = {}
        self._init_vectors()

    def _init_vectors(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if EMBEDDING_CACHE_FILE.is_file():
            try:
                with EMBEDDING_CACHE_FILE.open('r', encoding='utf-8') as f:
                    self._vector_cache = json.load(f)
            except Exception:
                self._vector_cache = {}

        if not self._vector_cache:
            self._precompute_embeddings()

    def _encode_text_sparse(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r'\w+', text.lower())
        vec: Dict[str, float] = {}
        for token in tokens:
            vec[token] = vec.get(token, 0.0) + 1.0
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    def _precompute_embeddings(self):
        for node_id, node in self.knowledge_graph.nodes.items():
            if node_id.startswith('JUNYI_LEVEL'):
                continue
            text = f"{node.name} {node.description}"
            self._vector_cache[node_id] = self._encode_text_sparse(text)

        with EMBEDDING_CACHE_FILE.open('w', encoding='utf-8') as f:
            json.dump(self._vector_cache, f, ensure_ascii=False)

    def map_question(self, question: Dict[str, Any]) -> ConceptMapping:
        concept_id = question.get('concept_id', '')
        if concept_id not in self._cache:
            self._cache[concept_id] = self._map(question)
        return self._cache[concept_id]

    def _map(self, question: Dict[str, Any]) -> ConceptMapping:
        eedi_id = question.get('concept_id', '')
        eedi_name = question.get('concept_name', '')
        query = _query_text(question)
        unmatched = ConceptMapping(False, eedi_id, eedi_name, None, None, 'semantic_unmapped', 0.0, None)

        scores: List[Tuple[float, str]] = []
        q_vec = self._encode_text_sparse(query)

        for node_id, node_vec in self._vector_cache.items():
            if node_id not in self.knowledge_graph.nodes:
                continue

            node = self.knowledge_graph.nodes[node_id]
            haystack = f"{node.name} {node.description}".lower()

            # Needle match score boost
            boost = 0.0
            for eng, chi in LEXICON:
                if all(term in query for term in eng):
                    if any(c in haystack for c in chi):
                        boost += 0.45

            # Cosine similarity calculation supporting dict/list formats
            if isinstance(node_vec, dict):
                dot = sum(v * node_vec.get(k, 0.0) for k, v in q_vec.items())
            elif isinstance(node_vec, list):
                dot = sum(v for k, v in q_vec.items() if k in haystack) / (len(q_vec) or 1.0)
            else:
                dot = 0.0

            sim = dot + boost

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
    """Hybrid Mapper combining Lexicon rule specificity boost with Semantic Vector Similarity."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.baseline_mapper = EediJunyiMapper(knowledge_graph)
        self.semantic_mapper = SemanticEmbeddingMapper(knowledge_graph)
        self._cache: Dict[str, ConceptMapping] = {}

    def map_question(self, question: Dict[str, Any]) -> ConceptMapping:
        concept_id = question.get('concept_id', '')
        if concept_id not in self._cache:
            self._cache[concept_id] = self._map(question)
        return self._cache[concept_id]

    def _map(self, question: Dict[str, Any]) -> ConceptMapping:
        base_res = self.baseline_mapper.map_question(question)
        sem_res = self.semantic_mapper.map_question(question)

        # If baseline lexicon matched, verify and rank
        if base_res.mapped:
            top3 = [base_res.junyi_concept_id]
            if sem_res.mapped and sem_res.junyi_concept_id not in top3:
                top3.append(sem_res.junyi_concept_id)
            if sem_res.top3_junyi_ids:
                for candidate in sem_res.top3_junyi_ids:
                    if candidate not in top3 and len(top3) < 3:
                        top3.append(candidate)
            return ConceptMapping(
                mapped=True,
                eedi_concept_id=base_res.eedi_concept_id,
                eedi_concept_name=base_res.eedi_concept_name,
                junyi_concept_id=base_res.junyi_concept_id,
                junyi_concept_name=base_res.junyi_concept_name,
                method='hybrid_lexicon_semantic',
                score=base_res.score + (sem_res.score * 10),
                rule=base_res.rule,
                top3_junyi_ids=top3
            )

        # Fallback to pure semantic match
        if sem_res.mapped:
            return ConceptMapping(
                mapped=True,
                eedi_concept_id=sem_res.eedi_concept_id,
                eedi_concept_name=sem_res.eedi_concept_name,
                junyi_concept_id=sem_res.junyi_concept_id,
                junyi_concept_name=sem_res.junyi_concept_name,
                method='hybrid_semantic_fallback',
                score=sem_res.score,
                rule=sem_res.rule,
                top3_junyi_ids=sem_res.top3_junyi_ids
            )

        return base_res
