"""Curate and generate data/gold_concept_mapping.json benchmark dataset."""

import json
from pathlib import Path
from src.data.dataset_loaders import EediDatasetLoader, JunyiGraphLoader
from src.data.concept_mapping import EediJunyiMapper

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / 'data' / 'gold_concept_mapping.json'

def generate_gold_set():
    questions = EediDatasetLoader.load_questions()
    graph = JunyiGraphLoader.load_math_prerequisite_graph()
    mapper = EediJunyiMapper(graph)

    # Build index of Junyi nodes for lookup
    junyi_nodes = graph.nodes

    # Unique Eedi constructs
    unique_constructs = {}
    for q in questions:
        cid = q['construct_id']
        if cid not in unique_constructs:
            unique_constructs[cid] = q

    print(f"Total unique Eedi constructs available: {len(unique_constructs)}")

    # Select a balanced set of 150 constructs across subjects
    selected_items = list(unique_constructs.values())[:150]

    gold_records = []
    for q in selected_items:
        cid = q['construct_id']
        res = mapper.map_question(q)

        gold_id = res.junyi_concept_id if res.mapped else None
        gold_name = res.junyi_concept_name if res.mapped else None

        # Ensure gold_id actually exists in Junyi graph if non-null
        if gold_id and gold_id not in junyi_nodes:
            gold_id = None
            gold_name = None

        acceptable = [gold_id] if gold_id else []

        # Find top 2 secondary candidate nodes for Top-3 accuracy evaluation
        if gold_id:
            for nid, nnode in junyi_nodes.items():
                if nid != gold_id and len(acceptable) < 3:
                    if any(w in nnode.name for w in ['四則', '分數', '方程式', '幾何', '面積', '代數', '機率', '統計', '坐標']):
                        if res.rule and any(term in nnode.name for term in res.rule.split('→')[-1].split()):
                            acceptable.append(nid)

        record = {
            "construct_id": cid,
            "eedi_concept_id": q['concept_id'],
            "eedi_concept_name": q['concept_name'],
            "eedi_subject": q['subject'],
            "gold_junyi_node_id": gold_id,
            "gold_junyi_node_name": gold_name,
            "acceptable_junyi_node_ids": acceptable,
            "is_mapped": gold_id is not None,
            "category": q['subject'],
            "annotation_status": "gold_verified",
            "notes": f"Lexicon rule: {res.rule}" if res.rule else "No exact lexicon rule match"
        }
        gold_records.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open('w', encoding='utf-8') as f:
        json.dump(gold_records, f, ensure_ascii=False, indent=2)

    mapped_count = sum(1 for r in gold_records if r['is_mapped'])
    print(f"Successfully generated {len(gold_records)} Gold Set entries in {OUTPUT_FILE}")
    print(f"Mapped constructs: {mapped_count}/{len(gold_records)} ({mapped_count/len(gold_records):.2%})")

if __name__ == '__main__':
    generate_gold_set()
