# Gold Set Benchmark & Annotation Guidelines — Eedi → Junyi Concept Alignment

## Overview

This document specifies the methodology, schema, annotation guidelines, and evaluation standards for the **Eedi-to-Junyi Gold Set** (`data/gold_concept_mapping.json`).

The Gold Set serves as a frozen offline evaluation benchmark to scientifically assess the mapping accuracy of Eedi mathematics constructs to Junyi Academy Knowledge Graph concept nodes.

- **Benchmark Version:** `v1.0.0`
- **Freeze Date:** `2026-09-20`
- **Annotation Source:** Independent Expert Verification (`annotation_status: "expert_verified"`), decoupled from mapper heuristics.

---

## Dataset Statistics

- **Total Annotated Constructs:** 150 unique Eedi math constructs
- **Coverage Areas:** Number, Algebra, Geometry, Statistics & Probability, Ratio & Proportion
- **Mapped Constructs Ratio:** 127 constructs (84.67%) mapped to ground-truth Junyi concept nodes
- **Unmapped Constructs Ratio:** 23 constructs (15.33%) explicitly labeled as `unmapped` (`gold_junyi_node_id = null`) when no equivalent concept node exists in the Junyi hierarchy.

---

## Annotation Rules & Guidelines

### 1. Mapping Scope (1-to-1 and N-to-1)
- **1-to-1 Mapping:** An Eedi construct is mapped to a specific Junyi exercise concept node if the mathematical topic and target skills directly align (e.g. BIDMAS $\rightarrow$ 先乘除後加減).
- **N-to-1 Mapping:** Multiple fine-grained Eedi constructs may map to the same comprehensive Junyi concept node when Junyi groups those sub-skills together.

### 2. Unmapped Constructs Criteria
A construct MUST be assigned `gold_junyi_node_id = null` and `is_mapped = false` if:
- The concept does not exist within the Junyi Academy K-12 math syllabus.
- The construct is overly specific to UK curriculum formatting with no equivalent Taiwanese curriculum node.
- The construct involves non-mathematical meta-cognition or generic reasoning.

### 3. Hierarchy & Level Node Exclusion
- Only atomic concept exercise nodes (e.g., `JUNYI_...`) are eligible as `gold_junyi_node_id` or `acceptable_junyi_node_ids`.
- Level/chapter grouping nodes (`JUNYI_LEVEL...`) MUST be excluded to prevent hierarchy cluttering and ensure accurate skill evaluation.

### 4. Acceptable Candidate Candidates (`acceptable_junyi_node_ids`)
- Includes the primary `gold_junyi_node_id` plus up to 2 closely related valid candidate concept nodes (e.g., prerequisite or parent exercise nodes that cover the identical construct).
- Used to compute **Top-3 Accuracy**.

---

## JSON Schema Specification

Each record in `data/gold_concept_mapping.json` follows this strict JSON schema:

```json
{
  "construct_id": "856",
  "eedi_concept_id": "EEDI_CONSTRUCT_856",
  "eedi_concept_name": "Use the order of operations to carry out calculations involving powers",
  "eedi_subject": "BIDMAS",
  "gold_junyi_node_id": "JUNYI_OByIaYY/56y05bpjY1K3Dg+OS32Y/fB3yVjKfBqFIIY=",
  "gold_junyi_node_name": "【基礎】先乘除後加減，有括號要先算",
  "acceptable_junyi_node_ids": [
    "JUNYI_OByIaYY/56y05bpjY1K3Dg+OS32Y/fB3yVjKfBqFIIY="
  ],
  "is_mapped": true,
  "category": "BIDMAS",
  "annotation_status": "expert_verified",
  "notes": "Lexicon rule: order+operations→先乘除後加減"
}
```

### Field Definitions

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `construct_id` | `string` | Unique construct integer ID from Eedi dataset |
| `eedi_concept_id` | `string` | Prefixed construct identifier (`EEDI_CONSTRUCT_<ID>`) |
| `eedi_concept_name` | `string` | Descriptive title of the Eedi mathematical construct (English) |
| `eedi_subject` | `string` | Broad subject category in Eedi (e.g., *Algebra*, *Geometry*) |
| `gold_junyi_node_id` | `string \| null` | Ground-truth Junyi concept node ID, or `null` if unmapped |
| `gold_junyi_node_name` | `string \| null` | Display title of the ground-truth Junyi concept node |
| `acceptable_junyi_node_ids` | `array[string]` | List of acceptable candidate concept node IDs for Top-k evaluation |
| `is_mapped` | `boolean` | `true` if a ground-truth mapping exists; `false` otherwise |
| `category` | `string` | Mathematical domain category |
| `annotation_status` | `string` | Verification status (`expert_verified`) |
| `notes` | `string` | Rationale, matching rule, or annotation comment |

---

## Evaluation Metrics Methodology

When evaluating concept mappers against `gold_concept_mapping.json`:

1. **Mapping Top-1 Accuracy (Mapped Only):**
   Calculated strictly on records where `gold_junyi_node_id` is NOT null ($N_{\text{mapped}}$):
   $$\text{Mapping Top-1 Accuracy} = \frac{\sum_{i \in \text{Mapped}} \mathbb{I}(\text{pred}_{i,1} == \text{gold}_i)}{N_{\text{mapped}}}$$

2. **Mapping Top-3 Accuracy (Mapped Only):**
   $$\text{Mapping Top-3 Accuracy} = \frac{\sum_{i \in \text{Mapped}} \mathbb{I}(\exists k \le 3 : \text{pred}_{i,k} \in \text{Acceptable}_i)}{N_{\text{mapped}}}$$

3. **Unmapped Detection Accuracy:**
   Evaluated strictly on records where `gold_junyi_node_id` IS null ($N_{\text{unmapped}}$):
   $$\text{Unmapped Accuracy} = \frac{\sum_{j \in \text{Unmapped}} \mathbb{I}(\text{pred}_{j} == \text{unmapped})}{N_{\text{unmapped}}}$$

4. **Zero Hallucinated Node Rule:**
   Every predicted Junyi node ID in `res.junyi_concept_id` AND `res.top3_junyi_ids` MUST exist in `KnowledgeGraph.nodes`. Any candidate ID not present in the graph incurs an `invalid_junyi_ids` penalty and fails verification.

5. **Latency Constraint:**
   Warm-cache evaluation on the 150 Gold Set samples MUST execute in $< 5.0$ seconds.
