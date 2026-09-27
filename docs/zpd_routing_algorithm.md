# Thuật toán ZPD Routing & Lựa chọn Câu hỏi Thực tế (PAAF Framework)

> **Phục vụ Báo cáo NCKH & Bài báo Khoa học**
> **Mã Task:** `SE-05` — Dynamic ZPD Learning Path & Item Selection

---

## 1. Tổng quan & Đóng góp Khoa học

Trong khung làm việc **PAAF (Pedagogical Agentic AI Framework)**, Planner Agent đóng vai trò trung tâm điều phối lộ trình học tập cá nhân hóa dựa trên lý thuyết **Vùng phát triển gần nhất (Zone of Proximal Development - ZPD)** của Vygotsky.

Khác với các hệ thống quy tắc tĩnh hoặc tạo nội dung văn bản trừu tượng, thuật toán **ZPD Routing** trong SE-05 đạt được 2 tính năng cốt lõi:
1. **Lựa chọn Item Thực tế (Real Item Selection):** Mọi bước luyện tập (`practice_exercise`), khắc phục sai lầm (`remediate_misconception`), hoặc ôn tập tiền đề (`review_prerequisite`) đều được truy vấn và gắn trực tiếp với `question_id` câu hỏi trắc nghiệm tồn tại thực tế trong bộ dữ liệu Eedi 2024.
2. **Lập lại Lộ trình Động (Dynamic Re-planning Loop):** Sau mỗi lượt trả lời của học sinh, hệ thống tự động đánh giá trạng thái năng lực, điều chỉnh điểm thành thạo $M_c \in [0.0, 1.0]$, lan truyền đứt gãy kiến thức trên Đồ thị Kiến thức Junyi, và cập nhật lộ trình:
   - **Tăng tốc (Acceleration):** Trả lời đúng liên tiếp ($C_{correct} \ge 2$), hệ thống tự động đẩy nhanh lộ trình lên bài tập vận dụng cao (*Advanced Challenge*), bỏ qua các bước ôn tập kiến thức tiền đề.
   - **Lùi bước củng cố (Fallback & Scaffolding):** Trả lời sai ($C_{correct} = 0$), hệ thống lùi về ôn tập các nút tiên quyết bị đứt gãy ($M_{prereq} < 0.6$) và chọn bài tập Eedi chứa đúng nhãn lỗi (*misconception*) đã chẩn đoán.

---

## 2. Mô hình Trạng thái & Công thức Toán học

Trạng thái học sinh tại thời điểm $t$ được định nghĩa bằng ngũ giác trạng thái:

$$\mathcal{S}_t = \langle \mathbf{M}_t, \mathbf{m}_t, C_{correct}, C_{incorrect}, \mathcal{Q}_{answered} \rangle$$

Trong đó:
- $\mathbf{M}_t = \{c: M_t(c) \mid M_t(c) \in [0, 1]\}$: Vectơ điểm thành thạo các khái niệm.
- $\mathbf{m}_t$: Danh sách các hiểu lầm (misconceptions) chưa được khắc phục.
- $C_{correct} \in \mathbb{N}_0$: Số câu trả lời đúng liên tiếp trong phiên.
- $C_{incorrect} \in \mathbb{N}_0$: Số câu trả lời sai liên tiếp trong phiên.
- $\mathcal{Q}_{answered} \subset \mathcal{D}_{Eedi}$: Tập hợp các `question_id` đã thực hiện.

Ngưỡng đánh giá thành thạo: $\theta_{mastery} = 0.6$.  
Ngưỡng kích hoạt tăng tốc lộ trình ZPD: $\gamma_{accelerate} = 2$.

---

## 3. Giả mã Thuật toán ZPD Routing (Pseudocode)

```python
"""
ALGORITHM 1: Dynamic ZPD Path Routing & Real Item Selection
----------------------------------------------------------
Input:
  - c_target: Khái niệm mục tiêu (Eedi Construct ID / Junyi Node ID)
  - G_Junyi: Đồ thị kiến thức Junyi Academy
  - D_t: Kết quả chẩn đoán từ Diagnostic Agent {is_correct, detected_misconception, misconception_id}
  - S_t: Trạng thái học sinh LearnerState tại thời điểm t
  - R_Eedi: Repository ngân hàng câu hỏi Eedi 2024 (1,869 questions)

Output:
  - P_{t+1}: Danh sách các bước trong lộ trình học tập (LearningPathStep)
  - R_zpd: Rationale giải thích căn cứ sư phạm ZPD
"""

FUNCTION GenerateZPDLearningPath(c_target, G_Junyi, D_t, S_t, R_Eedi):
    // Bước 1: Khởi tạo danh sách bước và tập câu hỏi đã dùng
    P_next = []
    step_id = 1
    used_qids = COPY(S_t.answered_questions)
    
    // Bước 2: Kiểm tra điều kiện Tăng tốc Lộ trình ZPD (Acceleration Condition)
    IF S_t.consecutive_correct >= 2 THEN:
        // Đã đúng liên tiếp >= 2 câu: Đẩy thẳng lên bài tập vận dụng cao
        q_item = R_Eedi.SelectItem(
            concept_id = c_target,
            exclude_ids = used_qids,
            action_type = "advanced_challenge"
        )
        used_qids.APPEND(q_item.question_id)
        
        step = LearningPathStep(
            step_id = step_id,
            concept_id = c_target,
            concept_name = q_item.concept_name,
            action_type = "advanced_challenge",
            description = "Thách thức nâng cao: Bài tập vận dụng cao ở khái niệm mục tiêu (Đã đúng >= 2 câu).",
            status = "pending",
            question_id = q_item.question_id,
            question_details = q_item
        )
        P_next.APPEND(step)
        
        R_zpd = "Tăng tốc lộ trình ZPD: Học sinh trả lời đúng liên tiếp, hệ thống chuyển sang Advanced Challenge và bỏ qua bài tập tiên quyết."
        RETURN P_next, R_zpd
    END IF
    
    // Bước 3: Lộ trình Củng cố & Khắc phục Sai lầm (Standard / Remediation Path)
    // 3.1. Tìm các khái niệm tiền đề chưa thành thạo từ Knowledge Graph Agent
    unmastered_prereqs = G_Junyi.GetUnmasteredPrerequisites(c_target, S_t.mastery_levels, threshold = 0.6)
    
    FOR EACH prereq IN unmastered_prereqs DO:
        q_item = R_Eedi.SelectItem(
            concept_id = prereq.concept_id,
            exclude_ids = used_qids,
            action_type = "review_prerequisite"
        )
        used_qids.APPEND(q_item.question_id)
        
        step = LearningPathStep(
            step_id = step_id,
            concept_id = prereq.concept_id,
            concept_name = prereq.name,
            action_type = "review_prerequisite",
            description = FORMAT("Ôn tập khái niệm tiền đề '%s' (Thành thạo: %.0f%%).", prereq.name, prereq.current_mastery * 100),
            status = "pending",
            question_id = q_item.question_id,
            question_details = q_item
        )
        P_next.APPEND(step)
        step_id = step_id + 1
    END FOR
    
    // 3.2. Khắc phục hiểu lầm cụ thể nếu phát hiện lỗi sai
    IF D_t.detected_misconception IS NOT NULL THEN:
        misc_identifier = D_t.misconception_id OR D_t.detected_misconception
        q_item = R_Eedi.SelectItem(
            concept_id = c_target,
            misconception_id_or_name = misc_identifier,
            exclude_ids = used_qids,
            action_type = "remediate_misconception"
        )
        used_qids.APPEND(q_item.question_id)
        
        step = LearningPathStep(
            step_id = step_id,
            concept_id = c_target,
            concept_name = D_t.concept_name,
            action_type = "remediate_misconception",
            description = FORMAT("Thực hành bài tập khắc phục hiểu lầm: '%s'.", D_t.detected_misconception),
            status = "pending",
            question_id = q_item.question_id,
            question_details = q_item
        )
        P_next.APPEND(step)
        step_id = step_id + 1
    END IF
    
    // 3.3. Bài tập ứng dụng đạt chuẩn ở khái niệm mục tiêu
    q_item = R_Eedi.SelectItem(
        concept_id = c_target,
        exclude_ids = used_qids,
        action_type = "practice_exercise"
    )
    used_qids.APPEND(q_item.question_id)
    
    step = LearningPathStep(
        step_id = step_id,
        concept_id = c_target,
        concept_name = D_t.concept_name,
        action_type = "learn_concept",
        description = FORMAT("Luyện tập bài tập ứng dụng đạt chuẩn ở khái niệm mục tiêu '%s'.", c_target),
        status = "pending",
        question_id = q_item.question_id,
        question_details = q_item
    )
    P_next.APPEND(step)
    
    R_zpd = FORMAT("Lộ trình ZPD cá nhân hóa gồm %d bước. Mỗi bước gắn với question_id Eedi thực tế.", LEN(P_next))
    
    RETURN P_next, R_zpd
END FUNCTION
```

---

## 4. Ma trận Chuyển trạng thái Lộ trình (State Transition Matrix)

| Trạng thái $t$ | Kết quả câu làm $t$ | Chuyển đổi trạng thái | Hành động Planner ở $t+1$ | Loại bước & Question ID |
| :--- | :--- | :--- | :--- | :--- |
| Chẩn đoán ban đầu | Trả lời Sai | $C_{correct}=0, C_{incorrect}=1$, giảm $M_c$, lan truyền giảm $M_{prereq}$ | Lùi về củng cố tiền đề + sửa hiểu lầm | `review_prerequisite` + `remediate_misconception` (Gắn `question_id` Eedi thực) |
| Đang ôn tiền đề | Trả lời Đúng | $C_{correct}=1$, tăng $M_{prereq}$, cập nhật $G_{Junyi}$ | Loại bỏ tiền đề đã đạt ($M_{prereq} \ge 0.6$) | Chuyển sang câu tiền đề tiếp theo hoặc `learn_concept` |
| Ôn tiền đề / Mục tiêu | Trả lời Đúng 2 lần liên tiếp | $C_{correct} \ge 2$ | **Tăng tốc ZPD (Accelerated)** | Bỏ qua mọi tiền đề $\rightarrow$ `advanced_challenge` (Gắn `question_id` nâng cao) |
| Nâng cao | Trả lời Sai | $C_{correct}=0, C_{incorrect}=1$ | Re-plan lùi lại ZPD cơ bản | Quay lại `remediate_misconception` + `learn_concept` |

---

## 5. Kết quả Kiểm chứng Thực nghiệm

Thuật toán ZPD Routing đã được kiểm chứng tự động thông qua bộ unit test `tests/test_dynamic_zpd_planner.py`:
- **100% các bước** trong `active_learning_path` có `question_id` tồn tại thực tế trong Eedi 2024 dataset.
- **Tính động (Dynamic Re-planning):** Lộ trình tự động điều chỉnh cấu trúc bước sau mỗi câu nộp bài.
- **0 rò rỉ ID:** Không phát sinh ID giả nằm ngoài dataset.
