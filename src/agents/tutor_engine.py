"""
Local Scaffolding Tutor Engine for Multi-turn Pedagogical Dialogue.

Uses Local Open-Weights LLM (e.g. Qwen2.5-7B-Instruct / Llama-3.1-8B) via Ollama/vLLM REST API
with Graduated Hinting (Nudge -> Hint -> Explanation) and Answer Leakage Guardrails.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LEVEL_NUDGE = "nudge"
LEVEL_HINT = "hint"
LEVEL_EXPLANATION = "explanation"


class LocalScaffoldingTutorEngine:
    """Tutor Engine implementing Graduated Hinting and Multi-turn Scaffolding Dialogue."""

    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        api_base: str = "http://localhost:11434",
        backend: str = "ollama",
        seed: int = 42,
        temperature: float = 0.3,
        max_retries: int = 2,
        timeout: int = 10,
        enable_ollama_fallback: bool = True,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.backend = backend.lower()
        self.seed = seed
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_ollama_fallback = enable_ollama_fallback

        if self.backend not in ("ollama", "vllm"):
            raise ValueError(f"Unsupported backend '{self.backend}'. Must be 'ollama' or 'vllm'.")

        self._validate_local_endpoint(self.api_base)

    def _validate_local_endpoint(self, url: str) -> None:
        """Validates that api_base targets a local host for privacy & local execution constraints."""
        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        allowed_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
        if hostname not in allowed_hosts:
            raise ValueError(
                f"Local execution constraint violation: api_base '{url}' hostname '{hostname}' is not a permitted local endpoint ({allowed_hosts})."
            )

    def build_scaffolding_prompt(
        self,
        student_name: str,
        student_query: str,
        concept_name: str,
        question_text: str,
        options: Dict[str, str],
        correct_option: str,
        selected_option: str,
        detected_misconception: Optional[str],
        cot_explanation: Optional[str],
        scaffolding_level: str,
        interaction_history: List[Dict[str, Any]],
    ) -> str:
        """Constructs prompt for multi-turn scaffolding dialogue adhering to target level & answer masking."""
        prompt = [
            "Bạn là Tutor AI - Gia sư Toán học thông minh theo phương pháp sư phạm Scaffolding của Vygotsky.",
            "Nhiệm vụ của bạn là phản hồi học sinh theo mô hình Graduated Hinting (Gợi ý phân cấp) dựa trên lịch sử hội thoại.",
            "",
            "--- NGUYÊN TẮC QUAN TRỌNG NHẤT ---",
            "1. KHÔNG BAO GIỜ trực tiếp chỉ ra đáp án đúng (không nói phương án nào đúng, không nêu kết quả cuối cùng).",
            "2. Chỉ đưa ra gợi ý nâng đỡ để học sinh tự tư duy và tìm ra câu trả lời.",
            "3. Duy trì xưng hô thân thiện, mang tính động viên ('Chào bạn...', 'Thầy/Cô thấy...').",
            "",
            f"--- BÀI HỌC VÀ LỖI SAI ---",
            f"Tên học sinh: {student_name}",
            f"Chủ đề / Khái niệm: {concept_name}",
            f"Đề bài: {question_text}",
            f"Các lựa chọn: {', '.join([f'[{k}] {v}' for k, v in options.items()])}",
            f"Học sinh chọn: [{selected_option}]",
            f"Lỗi hiểu lầm phát hiện: {detected_misconception or 'Chưa xác định'}",
            f"Phân tích CoT: {cot_explanation or 'Chưa có phân tích'}",
            "",
        ]

        # Add recent conversation history (up to 6 items)
        if interaction_history:
            prompt.append("--- LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ ---")
            recent_history = interaction_history[-6:]
            for entry in recent_history:
                role = "Học sinh" if entry.get("role") == "user" else "Tutor AI"
                msg = entry.get("message", "")
                prompt.append(f"{role}: {msg}")
            prompt.append("")

        # Instructions based on scaffolding_level
        prompt.append(f"--- CÂU HỎI HIỆN TẠI VÀ YÊU CẦU PHẢN HỒI ---")
        prompt.append(f"Học sinh vừa hỏi: '{student_query}'")
        prompt.append(f"Cấp độ gợi ý hiện tại: LEVEL = {scaffolding_level.upper()}")

        if scaffolding_level == LEVEL_NUDGE:
            prompt.append(
                "Yêu cầu Level 1 (Nudge - Gợi mở): Nhắc nhở nhẹ nhàng, đặt 1-2 câu hỏi gợi mở yêu cầu học sinh tự đọc lại đề bài hoặc kiểm tra bước đầu tiên. KHÔNG chỉ ra chi tiết lỗi sai."
            )
        elif scaffolding_level == LEVEL_HINT:
            prompt.append(
                "Yêu cầu Level 2 (Hint - Gợi ý cụ thể): Chỉ ra chính xác khái niệm/công thức hoặc vị trí bước tính bị vướng (liên quan đến lỗi: "
                f"'{detected_misconception}'). Hướng dẫn học sinh cách kiểm tra bước đó nhưng VẪN KHÔNG cho đáp án."
            )
        else:  # LEVEL_EXPLANATION
            prompt.append(
                "Yêu cầu Level 3 (Explanation - Giải thích bản chất): Phân tích chi tiết nguyên nhân sư phạm vì sao cách suy nghĩ ban đầu dẫn đến lỗi sai, "
                "gợi ý quy tắc đúng cần áp dụng để tự giải lại bài toán. Tuyệt đối KHÔNG viết ra đáp án trắc nghiệm."
            )

        prompt.append("\nHãy viết phản hồi của Tutor AI:")
        return "\n".join(prompt)

    def _call_backend(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Calls local REST API endpoint (Ollama or vLLM). Returns (raw_text, error_reason)."""
        if self.backend == "ollama":
            url = f"{self.api_base}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "seed": self.seed},
            }
        else:
            url = f"{self.api_base}/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "seed": self.seed,
            }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    resp_obj = json.loads(response.read().decode("utf-8"))
                    if self.backend == "ollama":
                        return resp_obj.get("response", ""), None
                    choices = resp_obj.get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", ""), None
                    if choices:
                        return choices[0].get("text", ""), None
                    return "", None
                return None, f"HTTP_ERROR_{response.status}"
        except urllib.error.HTTPError as error:
            return None, "MODEL_NOT_FOUND" if error.code == 404 else f"HTTP_ERROR_{error.code}"
        except (TimeoutError, socket.timeout):
            return None, "TIMEOUT"
        except urllib.error.URLError:
            return None, "CONNECTION_ERROR"
        except OSError:
            return None, "CONNECTION_ERROR"
        except Exception:
            logger.exception("Unexpected local backend failure in Tutor Engine")
            return None, "UNEXPECTED_ERROR"

    def sanitize_answer_leakage(
        self, text: str, correct_option: str, options: Dict[str, str]
    ) -> str:
        """
        Guardrail: Ensures the correct answer option text or explicit statements like
        'Đáp án đúng là B' are not leaked to the student.
        """
        if not text:
            return text

        correct_text = options.get(correct_option, "").strip()

        # Sanitize explicit leaks of answer choice letter like "Đáp án đúng là B" or "Chọn B"
        if correct_option:
            pattern_letter = re.compile(
                rf"(đáp án đúng là|đáp án là|chọn|phương án đúng là)\s*\[?{re.escape(correct_option)}\]?",
                re.IGNORECASE,
            )
            text = pattern_letter.sub(
                r"\1 [bảo mật - hãy tự suy luận]", text
            )

        # Sanitize exact text of the correct option if it is specific and non-trivial (> 1 char)
        if correct_text and len(correct_text) > 1 and correct_text.lower() in text.lower():
            # Check if correct option text is leaked in context of telling student the answer
            leak_pattern = re.compile(re.escape(correct_text), re.IGNORECASE)
            # Only redact if it looks like a raw answer disclosure
            text = leak_pattern.sub("[bằng kết quả đúng mà bạn cần tự tính]", text)

        return text

    def generate_fallback_scaffolding(
        self,
        student_name: str,
        student_query: str,
        concept_name: str,
        detected_misconception: Optional[str],
        scaffolding_level: str,
        interaction_history: List[Dict[str, Any]],
    ) -> str:
        """Generates fallback scaffolding responses matching Nudge/Hint/Explanation levels offline."""
        misc_str = detected_misconception or "lỗi tính toán/áp dụng quy tắc"

        if scaffolding_level == LEVEL_NUDGE:
            return (
                f"Chào bạn {student_name}! Ở bài tập '{concept_name}', thầy/cô thấy bạn đang gặp chút vướng mắc.\n\n"
                f"💡 **Gợi mở (Nudge):**\n"
                f"1. Hãy đọc kỹ lại yêu cầu bài toán một lần nữa.\n"
                f"2. Bạn có thể nêu lại công thức hoặc quy tắc ban đầu mà bạn đã sử dụng để làm bài này không?"
            )
        elif scaffolding_level == LEVEL_HINT:
            return (
                f"Chào bạn {student_name}! Phân tích cho thấy bài làm của bạn dường như đang vướng ở lỗi '{misc_str}'.\n\n"
                f"🔍 **Gợi ý cụ thể (Hint):**\n"
                f"1. Hãy chú ý đến bước biến đổi liên quan đến '{concept_name}'.\n"
                f"2. Đối chiếu bước tính của bạn với nhãn lỗi '{misc_str}' để xem bạn đã bỏ sót điều kiện hay quy tắc quy đồng/đổi dấu nào.\n"
                f"3. Bạn thử tính lại từ vị trí đó xem sao nhé!"
            )
        else:  # LEVEL_EXPLANATION
            return (
                f"Chào bạn {student_name}! Thầy/cô sẽ giải thích rõ hơn về bản chất của lỗi '{misc_str}' nhé.\n\n"
                f"📘 **Giải thích chuyên sâu (Explanation):**\n"
                f"Khi giải bài toán thuộc chủ đề '{concept_name}', lỗi '{misc_str}' thường xảy ra khi áp dụng sai quy tắc cơ bản (ví dụ: thực hiện phép tính không đồng nhất hoặc chuyển vế sai dấu).\n"
                f"Để khắc phục, bạn cần tuân thủ từng bước:\n"
                f"- Bước 1: Xác định đúng tính chất/công thức chuẩn của bài toán.\n"
                f"- Bước 2: Thực hiện phép biến đổi chính xác theo từng hàng.\n"
                f"Hãy áp dụng quy tắc này và tự tìm lại kết quả chính xác nhé!"
            )

    def generate_scaffolding_response(
        self,
        student_name: str,
        student_query: str,
        concept_name: str,
        question_text: str,
        options: Dict[str, str],
        correct_option: str,
        selected_option: str,
        detected_misconception: Optional[str],
        cot_explanation: Optional[str],
        scaffolding_level: str,
        interaction_history: List[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """
        Generates scaffolding response using Local LLM with guardrails, falling back to rule-based scaffolding generator if unreachable.
        Returns (response_text, used_llm).
        """
        prompt = self.build_scaffolding_prompt(
            student_name=student_name,
            student_query=student_query,
            concept_name=concept_name,
            question_text=question_text,
            options=options,
            correct_option=correct_option,
            selected_option=selected_option,
            detected_misconception=detected_misconception,
            cot_explanation=cot_explanation,
            scaffolding_level=scaffolding_level,
            interaction_history=interaction_history,
        )

        raw, backend_error = self._call_backend(prompt)

        if not backend_error and raw and raw.strip():
            sanitized = self.sanitize_answer_leakage(raw.strip(), correct_option, options)
            return sanitized, True

        # Offline Fallback
        fallback = self.generate_fallback_scaffolding(
            student_name=student_name,
            student_query=student_query,
            concept_name=concept_name,
            detected_misconception=detected_misconception,
            scaffolding_level=scaffolding_level,
            interaction_history=interaction_history,
        )
        sanitized_fallback = self.sanitize_answer_leakage(fallback, correct_option, options)
        return sanitized_fallback, False


__all__ = [
    "LocalScaffoldingTutorEngine",
    "LEVEL_NUDGE",
    "LEVEL_HINT",
    "LEVEL_EXPLANATION",
]
