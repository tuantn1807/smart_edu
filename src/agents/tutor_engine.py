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


def compute_specificity_score(text: str, detected_misconception: Optional[str] = None) -> float:
    """
    Calculates quantitative specificity score for a tutor response.
    Considers text length, presence of specific misconception label, and step-by-step guidance indicators.
    """
    if not text:
        return 0.0

    score = len(text) / 100.0
    if detected_misconception and detected_misconception in text:
        score += 5.0
    if "🔍 **Gợi ý cụ thể" in text or "nhãn lỗi" in text.lower():
        score += 3.0
    if "📘 **Giải thích" in text or "bản chất" in text.lower():
        score += 4.0
    return score


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
                "Yêu cầu Level 1 (Nudge - Gợi mở): Đặt 1-2 câu hỏi gợi mở tổng quát để học sinh tự đọc lại đề bài và tự kiểm tra bước ban đầu. "
                "TUYỆT ĐỐI KHÔNG nêu tên lỗi hiểu lầm ('" + str(detected_misconception) + "') và KHÔNG ghi công thức chi tiết."
            )
        elif scaffolding_level == LEVEL_HINT:
            prompt.append(
                "Yêu cầu Level 2 (Hint - Gợi ý cụ thể): Nêu rõ tên nhãn lỗi phát hiện ('" + str(detected_misconception) + "') "
                "và chỉ ra bước tính/công thức cụ thể bị vướng để hướng dẫn học sinh cách tự sửa. VẪN KHÔNG tiết lộ đáp án."
            )
        else:  # LEVEL_EXPLANATION
            prompt.append(
                "Yêu cầu Level 3 (Explanation - Giải thích bản chất): Phân tích chi tiết nguyên nhân vì sao lỗi '" + str(detected_misconception) + "' "
                "dẫn đến kết quả sai, và giải thích từng bước logic toán học chuẩn để học sinh tự giải lại. VẪN KHÔNG tiết lộ đáp án trắc nghiệm."
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
        'Đáp án đúng là B' or raw option math expressions are not leaked to the student.
        """
        if not text:
            return text

        correct_text = options.get(correct_option, "").strip()

        # 1. Sanitize explicit leaks of answer choice letter like "Đáp án đúng là B", "Chọn [B]", "Đáp án B"
        if correct_option:
            pattern_letter = re.compile(
                rf"(đáp án đúng là|đáp án là|chọn phương án|chọn|phương án đúng là|kết quả là|kết quả đúng là|key is)\s*:?\s*\[?{re.escape(correct_option)}\]?",
                re.IGNORECASE,
            )
            text = pattern_letter.sub(
                r"\1 [bảo mật - hãy tự suy luận]", text
            )

        # 2. Sanitize exact text of the correct option if it is specific and non-trivial (> 1 char)
        if correct_text:
            clean_opt = re.sub(r"\\\(|\\\)|\\\$|\$|\\text\{|\}", "", correct_text).strip()
            if len(clean_opt) > 1:
                # Check for raw option text presence in context of telling answer
                leak_pattern = re.compile(re.escape(clean_opt), re.IGNORECASE)
                text = leak_pattern.sub("[kết quả mà bạn cần tự tính]", text)

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
                f"Chào bạn {student_name}! Ở bài tập thuộc chủ đề '{concept_name}', thầy/cô thấy bạn đang gặp chút vướng mắc.\n\n"
                f"💡 **Gợi mở (Nudge):**\n"
                f"1. Hãy đọc kỹ lại yêu cầu bài toán một lần nữa để xác định rõ thông tin đề bài cho.\n"
                f"2. Bạn có thể nêu lại công thức hoặc quy tắc ban đầu mà bạn đã áp dụng cho bài làm này không?"
            )
        elif scaffolding_level == LEVEL_HINT:
            return (
                f"Chào bạn {student_name}! Phân tích cho thấy bài làm của bạn dường như đang vướng ở lỗi '{misc_str}'.\n\n"
                f"🔍 **Gợi ý cụ thể (Hint):**\n"
                f"1. Nhãn lỗi phát hiện: '{misc_str}'.\n"
                f"2. Hãy chú ý đến bước biến đổi liên quan đến chủ đề '{concept_name}'. Xem lại quy tắc áp dụng và kiểm tra phép tính/phương pháp quy đồng/đổi dấu.\n"
                f"3. Bạn thử kiểm tra lại bước tính đó xem sao nhé!"
            )
        else:  # LEVEL_EXPLANATION
            return (
                f"Chào bạn {student_name}! Thầy/cô sẽ giải thích sâu hơn về bản chất của lỗi '{misc_str}' nhé.\n\n"
                f"📘 **Giải thích chuyên sâu (Explanation):**\n"
                f"Nguyên nhân mắc lỗi '{misc_str}' trong chủ đề '{concept_name}' thường do ngộ nhận hoặc áp dụng sai quy tắc biến đổi toán học.\n"
                f"Để khắc phục, bạn cần làm theo từng bước:\n"
                f"- Bước 1: Xác định đúng công thức/tính chất chuẩn của bài toán.\n"
                f"- Bước 2: Thực hiện phép biến đổi tương đương theo từng dòng một cách cẩn thận.\n"
                f"Hãy áp dụng quy tắc chuẩn này để tự tính lại kết quả nhé!"
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
    "compute_specificity_score",
]
