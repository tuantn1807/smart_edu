"""
Demonstration of PAAF with two benchmark datasets:
- Eedi 2024: diagnostic questions and misconception labels (Diagnostic Agent)
- Junyi: concept graph for prerequisite traversal (Knowledge Graph Agent)
"""

from src.data.concept_mapping import EediJunyiMapper
from src.data.dataset_loaders import EediDatasetLoader, JunyiGraphLoader
from src.orchestrator.paaf_framework import PAAFFramework


def run_student_session(framework: PAAFFramework, student_id: str, student_name: str, question_id: str, selected_option: str, student_query: str):
    print("=" * 80)
    print(f"  PHIÊN THỰC NGHIỆM: HỌC SINH '{student_name}' (ID: {student_id})")
    print("=" * 80)

    question = EediDatasetLoader.get_question_by_id(question_id)
    print(f"📖 [DATASET EEDI - Câu hỏi {question['question_id']}]: {question['question_text']}")
    print(f"   Khái niệm: {question['concept_name']} (ID: {question['concept_id']})")
    print("   Các phương án lựa chọn:")
    for opt, text in question['options'].items():
        tag = " (Đáp án đúng)" if opt == question['correct_option'] else ""
        print(f"     [{opt}] {text}{tag}")
    print(f"   👉 Phương án học sinh chọn: Option [{selected_option}]\n")

    pipeline_result = framework.run_full_pipeline(
        student_id=student_id,
        student_name=student_name,
        diagnostic_question=question,
        selected_option=selected_option
    )

    print("-" * 80)
    print("📌 [DIAGNOSTIC AGENT - Misconception Analysis & CoT Reasoning]:")
    diag = pipeline_result["diagnosis_result"]
    print(f" - Kết quả làm bài: {'ĐÚNG' if diag['is_correct'] else 'SAI'}")
    if not diag['is_correct']:
        print(f" - Hiểu lầm phát hiện: {diag['detected_misconception']}")
        print(f" - Mức độ nghiêm trọng: {diag['severity'].upper()}")
        print(f" - Chuỗi suy luận (CoT Explanation):\n{diag['cot_explanation']}\n")

    print("-" * 80)
    print("📌 [KNOWLEDGE GRAPH AGENT - Prerequisite Tree Traversal (Junyi)]:")
    kg_res = pipeline_result["kg_analysis"]
    mapping = kg_res.get("mapping") or {}
    if mapping.get("mapped"):
        print(f" - Ánh xạ heuristic Eedi → Junyi: {mapping['eedi_concept_name']}")
        print(f"   → {mapping['junyi_concept_name']} ({mapping['junyi_concept_id']})")
        print(f"   Phương pháp: {mapping['method']} ({mapping.get('rule')})")
    else:
        print(" - Chưa ánh xạ được construct Eedi sang node Junyi.")
    print(f" - Tóm tắt: {kg_res['analysis_summary']}")
    for un_p in kg_res["unmastered_prerequisites"]:
        print(f"   * Khái niệm bị đứt gãy: {un_p['name']} (ID: {un_p['concept_id']}) - Mức độ hiện tại: {un_p['current_mastery']*100:.0f}%")
    print()

    print("-" * 80)
    print("📌 [PLANNER AGENT - ZPD Personalized Learning Path Recommendation]:")
    plan = pipeline_result["planner_result"]
    print(f" - Lý thuyết ZPD: {plan['zpd_rationale']}")
    print(" - Chuỗi các bước học tập được đề xuất:")
    for step in plan["learning_path"]:
        print(f"   [Bước {step['step_id']}] [{step['action_type'].upper()}] {step['concept']}: {step['description']}")
    print()

    print("-" * 80)
    print("📌 [TUTOR AGENT - Interactive Scaffolding Dialogue]:")
    print(f"Học sinh thắc mắc: '{student_query}'")
    tutor_res = framework.interact_with_tutor(pipeline_result, student_query)
    print(f"\nTutor Agent phản hồi:\n{tutor_res['tutor_response']}")
    print("=" * 80 + "\n\n")


def main():
    print("\n" + "█" * 80)
    print("     HỆ THỐNG AGENTIC AI (PAAF) - DEMO 2 DATASET: EEDI + JUNYI")
    print("█" * 80 + "\n")

    questions = EediDatasetLoader.load_questions()
    kg = JunyiGraphLoader.load_math_prerequisite_graph()
    mapper = EediJunyiMapper(kg)
    coverage = mapper.coverage(questions)
    print(f"Eedi: {len(questions)} câu hỏi chẩn đoán, {coverage['total_constructs']} constructs.")
    print(
        f"Junyi: {len(kg.nodes)} node, {kg.edge_count()} cạnh "
        f"(nguồn={kg.source}, loại={kg.graph_kind})."
    )
    print(
        f"Ánh xạ heuristic: {coverage['mapped_constructs']}/{coverage['total_constructs']} "
        "constructs Eedi khớp được node Junyi. Đây không phải ánh xạ chuyên gia."
    )
    print("Lựa chọn học sinh bên dưới là kịch bản demo, không phải nhật ký thật.\n")

    framework = PAAFFramework(knowledge_graph=kg, concept_mapper=mapper)
    examples = []
    for question in questions:
        if not question['misconception_map']:
            continue
        if mapper.map_question(question).mapped:
            examples.append(question)
        if len(examples) == 3:
            break
    if len(examples) < 3:
        raise RuntimeError('Không đủ câu Eedi vừa có nhãn hiểu lầm vừa ánh xạ được sang Junyi.')

    for index, question in enumerate(examples, 1):
        run_student_session(
            framework=framework,
            student_id=f"DEMO_{index}", student_name=f"Demo {index}",
            question_id=question['question_id'],
            selected_option=next(iter(question['misconception_map'])),
            student_query="Giải thích giúp em lỗi sai trong bài này.",
        )

    print("█" * 80)
    print("           HOÀN THÀNH TOÀN BỘ PHIÊN THỰC NGHIỆM DEMO 2 DATASET!")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    main()
