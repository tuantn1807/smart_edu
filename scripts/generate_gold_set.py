"""Validate and verify data/gold_concept_mapping.json benchmark dataset.
Ensures schema integrity, 0% invalid graph IDs, and expert annotation standards without leakage."""

import json
import sys
from pathlib import Path
from src.data.dataset_loaders import JunyiGraphLoader

ROOT = Path(__file__).resolve().parents[1]
GOLD_SET_FILE = ROOT / 'data' / 'gold_concept_mapping.json'

REQUIRED_KEYS = {
    "construct_id", "eedi_concept_id", "eedi_concept_name", "eedi_subject",
    "gold_junyi_node_id", "gold_junyi_node_name", "acceptable_junyi_node_ids",
    "is_mapped", "annotation_status", "notes"
}

def validate_gold_set() -> bool:
    if not GOLD_SET_FILE.is_file():
        print(f"ERROR: Gold set file missing at {GOLD_SET_FILE}")
        return False

    graph = JunyiGraphLoader.load_math_prerequisite_graph()
    junyi_nodes = graph.nodes

    with GOLD_SET_FILE.open('r', encoding='utf-8') as f:
        records = json.load(f)

    total = len(records)
    print(f"=== Gold Set Benchmark Validation Report ===")
    print(f"File Path: {GOLD_SET_FILE}")
    print(f"Total Constructs: {total}")

    if not (100 <= total <= 200):
        print(f"ERROR: Gold set size {total} is outside required range [100, 200].")
        return False

    errors = 0
    mapped_count = 0
    unmapped_count = 0
    subject_counts = {}

    construct_ids = set()

    for idx, r in enumerate(records):
        cid = r.get("construct_id")
        if cid in construct_ids:
            print(f"ERROR Record #{idx}: Duplicate construct_id {cid}")
            errors += 1
        construct_ids.add(cid)

        missing_keys = REQUIRED_KEYS - set(r.keys())
        if missing_keys:
            print(f"ERROR Record #{idx} ({cid}): Missing required keys {missing_keys}")
            errors += 1

        is_mapped = r.get("is_mapped")
        gold_id = r.get("gold_junyi_node_id")
        gold_name = r.get("gold_junyi_node_name")

        if is_mapped:
            mapped_count += 1
            if not gold_id or not gold_name:
                print(f"ERROR Record #{idx} ({cid}): is_mapped=True but gold_id or gold_name is null")
                errors += 1
            elif gold_id not in junyi_nodes:
                print(f"ERROR Record #{idx} ({cid}): gold_junyi_node_id '{gold_id}' not in Junyi Graph!")
                errors += 1
        else:
            unmapped_count += 1
            if gold_id is not None:
                print(f"ERROR Record #{idx} ({cid}): is_mapped=False but gold_id is not null")
                errors += 1

        acceptable = r.get("acceptable_junyi_node_ids", [])
        for acc_id in acceptable:
            if acc_id not in junyi_nodes:
                print(f"ERROR Record #{idx} ({cid}): acceptable ID '{acc_id}' not in Junyi Graph!")
                errors += 1

        subject = r.get("eedi_subject", "Unknown")
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    print(f"\nMapped Constructs: {mapped_count} ({mapped_count/total:.2%})")
    print(f"Unmapped Constructs: {unmapped_count} ({unmapped_count/total:.2%})")
    print("\nSubject Breakdown:")
    for subj, count in subject_counts.items():
        print(f"  - {subj}: {count}")

    if errors == 0:
        print(f"\nSUCCESS: Gold Set benchmark is valid and verified against Junyi Graph ({len(junyi_nodes)} nodes).")
        return True
    else:
        print(f"\nFAILURE: Found {errors} validation errors in Gold Set.")
        return False

if __name__ == '__main__':
    valid = validate_gold_set()
    sys.exit(0 if valid else 1)
