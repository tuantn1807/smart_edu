"""Load real Eedi 2024 CSVs and Junyi concept graphs; never invent records."""

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from src.core.knowledge_graph import KnowledgeGraph

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

JUNYI_EXERCISE_TABLE = Path(DATA_DIR) / 'raw' / 'junyi' / 'junyi_Exercise_table.csv'
JUNYI_INFO_CONTENT = Path(DATA_DIR) / 'junyi' / 'archive' / 'Info_Content.csv'

JUNYI_DIFFICULTY = {'easy': 0.3, 'normal': 0.6, 'hard': 0.9, 'unset': 0.5}


def read_rows(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing. Run: python3 -m src.data.download_datasets')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def _strip_junyi_tag(name: str) -> str:
    return re.sub(r'^【[^】]+】', '', name or '').strip()


def _topic_label(names: List[str]) -> str:
    stems = [_strip_junyi_tag(name) for name in names if name]
    unique = []
    for stem in stems:
        if stem and stem not in unique:
            unique.append(stem)
        if len(unique) == 3:
            break
    if not unique:
        return 'Junyi topic'
    suffix = '…' if len({_strip_junyi_tag(name) for name in names}) > 3 else ''
    return ' / '.join(unique) + suffix


class EediDatasetLoader:
    """Eedi — Mining Misconceptions in Mathematics (2024), train split."""

    @staticmethod
    def get_sample_diagnostic_questions() -> List[Dict[str, Any]]:
        """Compatibility method: returns the complete downloaded training split."""
        return EediDatasetLoader.load_questions()

    @staticmethod
    def load_questions() -> List[Dict[str, Any]]:
        root = Path(DATA_DIR) / 'raw' / 'eedi'
        labels = {r['MisconceptionId']: r['MisconceptionName']
                  for r in read_rows(root / 'misconception_mapping.csv')}
        questions = []
        seen = set()
        for row in read_rows(root / 'train.csv'):
            qid = 'EEDI_' + row['QuestionId']
            if qid in seen or row['CorrectAnswer'] not in 'ABCD' or len(row['CorrectAnswer']) != 1:
                raise ValueError(f'Invalid/duplicate question: {qid}')
            seen.add(qid)
            misconceptions = {}
            for option in 'ABCD':
                value = row[f'Misconception{option}Id'].strip()
                if value:
                    mid = str(int(float(value)))
                    misconceptions[option] = {
                        'misconception_id': mid, 'name': labels[mid],
                        'description': labels[mid],
                        'severity': 'unknown',  # Not annotated by Eedi.
                    }
            questions.append({
                'question_id': qid, 'source_question_id': row['QuestionId'],
                'concept_id': 'EEDI_CONSTRUCT_' + row['ConstructId'],
                'concept_name': row['ConstructName'], 'construct_id': row['ConstructId'],
                'subject': row['SubjectName'], 'subject_id': row['SubjectId'],
                'question_text': row['QuestionText'],
                'options': {o: row[f'Answer{o}Text'] for o in 'ABCD'},
                'correct_option': row['CorrectAnswer'], 'misconception_map': misconceptions,
                'source': 'Eedi 2024 train',
            })
        return questions

    @staticmethod
    def get_question_by_id(question_id: str) -> Dict[str, Any]:
        for question in EediDatasetLoader.load_questions():
            if question['question_id'] == question_id:
                return question
        raise KeyError(f'Unknown Eedi question ID: {question_id}')

    @staticmethod
    def load_concept_graph() -> KnowledgeGraph:
        """Construct nodes only: Eedi does not supply prerequisite edges/difficulty."""
        graph = KnowledgeGraph(
            subject_name='Eedi 2024 constructs (no prerequisite annotations)',
            prerequisites_available=False,
            source='eedi_2024_train',
            graph_kind='construct_nodes',
        )
        for question in EediDatasetLoader.load_questions():
            graph.add_concept(question['concept_id'], question['concept_name'],
                              difficulty=0.5, description='Difficulty unannotated; neutral application default.')
        return graph


class JunyiGraphLoader:
    """Junyi concept graph for KG Agent. Prefers expert prerequisite table when present."""

    @staticmethod
    def load_math_prerequisite_graph() -> KnowledgeGraph:
        if JUNYI_EXERCISE_TABLE.is_file():
            return JunyiGraphLoader._from_exercise_table()
        if JUNYI_INFO_CONTENT.is_file():
            return JunyiGraphLoader._from_info_content()
        raise FileNotFoundError(
            f'Missing Junyi graph. Place junyi_Exercise_table.csv at {JUNYI_EXERCISE_TABLE} '
            f'or Info_Content.csv at {JUNYI_INFO_CONTENT}.'
        )

    @staticmethod
    def _from_exercise_table() -> KnowledgeGraph:
        rows = read_rows(JUNYI_EXERCISE_TABLE)
        graph = KnowledgeGraph(
            subject_name='Junyi Academy (expert prerequisite map)',
            source='junyi_exercise_table',
            graph_kind='expert_prerequisite',
        )
        names = {r['name'] for r in rows}
        prereq_field = 'prerequisite' if 'prerequisite' in rows[0] else 'prerequisites'
        for row in rows:
            raw = (row.get(prereq_field) or '').strip()
            prerequisites = [p.strip() for p in raw.split(',') if p.strip() and p.strip().lower() != 'nan']
            unknown = set(prerequisites) - names
            if unknown:
                raise ValueError(f"Unknown prerequisite for {row['name']}: {sorted(unknown)}")
            display = row.get('pretty_display_name') or row['name']
            topic = row.get('topic') or ''
            area = row.get('area') or ''
            description = ' | '.join(part for part in (display, topic, area) if part)
            graph.add_concept(
                'JUNYI_' + row['name'],
                display,
                0.5,
                description or 'Difficulty unannotated; neutral application default.',
                ['JUNYI_' + p for p in prerequisites if p != row['name']],
            )
        return graph

    @staticmethod
    def _from_info_content() -> KnowledgeGraph:
        """Hierarchy edges from Info_Content level2→level3→level4→exercise; not expert prerequisites."""
        rows = read_rows(JUNYI_INFO_CONTENT)
        graph = KnowledgeGraph(
            subject_name='Junyi Academy (2019 Info_Content hierarchy)',
            source='junyi_kaggle_2019_info_content',
            graph_kind='content_hierarchy',
        )
        by_level = {2: defaultdict(list), 3: defaultdict(list), 4: defaultdict(list)}
        for row in rows:
            by_level[2][row['level2_id']].append(row)
            by_level[3][row['level3_id']].append(row)
            by_level[4][row['level4_id']].append(row)

        for level2_id, members in by_level[2].items():
            graph.add_concept(
                'JUNYI_LEVEL2_' + level2_id,
                _topic_label([m['content_pretty_name'] for m in members]),
                0.2,
                'Junyi level-2 topic group from Info_Content; hierarchy parent, not expert prerequisite.',
            )
        for level3_id, members in by_level[3].items():
            parent = 'JUNYI_LEVEL2_' + members[0]['level2_id']
            graph.add_concept(
                'JUNYI_LEVEL3_' + level3_id,
                _topic_label([m['content_pretty_name'] for m in members]),
                0.35,
                'Junyi level-3 topic group from Info_Content; hierarchy parent, not expert prerequisite.',
                [parent],
            )
        for level4_id, members in by_level[4].items():
            parent = 'JUNYI_LEVEL3_' + members[0]['level3_id']
            graph.add_concept(
                'JUNYI_LEVEL4_' + level4_id,
                _topic_label([m['content_pretty_name'] for m in members]),
                0.45,
                'Junyi level-4 topic group from Info_Content; hierarchy parent, not expert prerequisite.',
                [parent],
            )
        seen = set()
        for row in rows:
            ucid = row['ucid']
            if ucid in seen:
                raise ValueError(f'Duplicate Junyi content id: {ucid}')
            seen.add(ucid)
            difficulty = JUNYI_DIFFICULTY.get(row['difficulty'], 0.5)
            graph.add_concept(
                'JUNYI_' + ucid,
                row['content_pretty_name'],
                difficulty,
                (
                    f"Junyi exercise ({row['difficulty']}/{row['learning_stage']}); "
                    'parent edge is content hierarchy, not an expert prerequisite annotation.'
                ),
                ['JUNYI_LEVEL4_' + row['level4_id']],
            )
        return graph
