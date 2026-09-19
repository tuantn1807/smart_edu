"""
Local Chain-of-Thought (CoT) Diagnostic Engine.

Uses Few-shot CoT Prompting on Local Open-Weights LLM (e.g. Qwen2.5-7B-Instruct / Llama-3.1-8B)
via Ollama/vLLM API with Pydantic schema guardrails and automatic retry mechanisms.
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

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

UNLABELED_KEY = "unlabeled"


class DiagnosticOutputSchema(BaseModel):
    """Pydantic guardrail schema for Diagnostic Engine output."""

    misconception_id: str = Field(
        ...,
        description="ID of the identified misconception (e.g. '157', '1672') or 'unlabeled' if none applies.",
    )
    cot_reasoning: str = Field(
        ...,
        description="Detailed step-by-step Chain-of-Thought explanation of student's mistake.",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class LocalCoTDiagnosticEngine:
    """Diagnostic Engine powered by Local Open-Weights LLM with Few-shot CoT Prompting."""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "Tính 1/2 + 1/3.",
            "options": {"A": "2/5", "B": "5/6", "C": "1/6", "D": "2/6"},
            "correct_option": "B",
            "selected_option": "A",
            "concept_name": "Cộng phân số khác mẫu số",
            "cot_reasoning": (
                "1. Quan sát: Học sinh chọn phương án 'A' (2/5) cho phép tính 1/2 + 1/3.\n"
                "2. Phân tích lỗi: Học sinh đã lấy tử số cộng tử số (1+1=2) và mẫu số cộng mẫu số (2+3=5).\n"
                "3. Suy luận nguyên nhân: Học sinh mắc hiểu lầm phổ biến khi cộng phân số bằng cách cộng trực tiếp tử với tử, mẫu với mẫu mà không quy đồng mẫu số.\n"
                "4. Kết luận: Học sinh chưa nắm vững quy tắc quy đồng mẫu số trước khi cộng."
            ),
            "misconception_id": "157",
            "confidence_score": 0.95,
        },
        {
            "question": "Giải phương trình 2x + 4 = 10.",
            "options": {"A": "x = 7", "B": "x = 3", "C": "x = 5", "D": "x = 14"},
            "correct_option": "B",
            "selected_option": "A",
            "concept_name": "Giải phương trình bậc nhất một ẩn",
            "cot_reasoning": (
                "1. Quan sát: Học sinh chọn phương án 'A' (x = 7) cho phương trình 2x + 4 = 10.\n"
                "2. Phân tích lỗi: Học sinh đã tính 2x = 10 + 4 = 14, dẫn đến x = 7.\n"
                "3. Suy luận nguyên nhân: Học sinh mắc lỗi chuyển vế không đổi dấu (chuyển +4 thành +4 thay vì -4).\n"
                "4. Kết luận: Học sinh bị hổng kiến thức về quy tắc chuyển vế trong phương trình."
            ),
            "misconception_id": "312",
            "confidence_score": 0.92,
        },
        {
            "question": "Rút gọn biểu thức (x + 3)^2.",
            "options": {"A": "x^2 + 9", "B": "x^2 + 6x + 9", "C": "x^2 + 3x + 9", "D": "2x + 6"},
            "correct_option": "B",
            "selected_option": "A",
            "concept_name": "Hằng đẳng thức bình phương của một tổng",
            "cot_reasoning": (
                "1. Quan sát: Học sinh chọn phương án 'A' (x^2 + 9) cho (x + 3)^2.\n"
                "2. Phân tích lỗi: Học sinh bình phương từng số hạng bên trong ngoặc (x^2 + 3^2) mà bỏ qua tích 2ab (2*x*3 = 6x).\n"
                "3. Suy luận nguyên nhân: Học sinh ngộ nhận phép bình phương có tính chất phân phối qua phép cộng.\n"
                "4. Kết luận: Lỗi hằng đẳng thức cơ bản."
            ),
            "misconception_id": "845",
            "confidence_score": 0.96,
        },
    ]

    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        api_base: str = "http://localhost:11434",
        backend: str = "ollama",
        diagnosis_mode: str = "independent",
        seed: int = 42,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 10,
        enable_ollama_fallback: bool = True,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.backend = backend.lower()
        self.diagnosis_mode = diagnosis_mode.lower()
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

    def build_few_shot_prompt(
        self,
        question: Dict[str, Any],
        selected_option: str,
        feedback_error: Optional[str] = None,
    ) -> str:
        """Constructs zero-leakage Chain-of-Thought prompt."""
        question_text = question.get("question_text", "")
        options = question.get("options", {})
        correct_option = question.get("correct_option", "")
        concept_name = question.get("concept_name", "Chưa xác định")

        prompt = [
            "Bạn là Diagnostic AI Agent chuyên sâu về phân tích nguyên nhân lỗi sai và phát hiện hiểu lầm (misconception) của học sinh trong môn Toán.",
            "Nhiệm vụ của bạn là đọc đề bài, các phương án, đáp án đúng, và đáp án học sinh chọn. Hãy phân tích Chain-of-Thought (CoT) từng bước và xuất JSON duy nhất đúng theo schema.",
            "",
            "Schema JSON bắt buộc:",
            "```json",
            "{",
            '  "misconception_id": "<ID hiểu lầm dạng chuỗi hoặc \'unlabeled\'>",',
            '  "cot_reasoning": "<Diễn giải suy luận CoT theo các bước: 1. Quan sát -> 2. Phân tích -> 3. Nguyên nhân -> 4. Kết luận>",',
            '  "confidence_score": <Số thực từ 0.0 đến 1.0>',
            "}",
            "```",
            "",
            "Ví dụ minh họa (Few-Shot Examples):",
        ]

        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            prompt.append(f"\n--- VÍ DỤ {i} ---")
            prompt.append(f"Đề bài: {ex['question']}")
            prompt.append("Lựa chọn: " + ", ".join([f"[{k}] {v}" for k, v in ex["options"].items()]))
            prompt.append(f"Đáp án đúng: [{ex['correct_option']}]")
            prompt.append(f"Học sinh chọn: [{ex['selected_option']}]")
            prompt.append(f"Khái niệm: {ex['concept_name']}")
            prompt.append("Output JSON:")
            ex_json = {
                "misconception_id": ex["misconception_id"],
                "cot_reasoning": ex["cot_reasoning"],
                "confidence_score": ex["confidence_score"],
            }
            prompt.append(f"```json\n{json.dumps(ex_json, ensure_ascii=False, indent=2)}\n```")

        prompt.append("\n--- BÀI CẦN CHẨN ĐOÁN THỰC TẾ ---")
        prompt.append(f"Đề bài: {question_text}")
        prompt.append("Lựa chọn: " + ", ".join([f"[{k}] {v}" for k, v in options.items()]))
        prompt.append(f"Đáp án đúng: [{correct_option}]")
        prompt.append(f"Học sinh chọn: [{selected_option}]")
        prompt.append(f"Khái niệm: {concept_name}")
        prompt.append(
            "Lưu ý: Phân tích CoT độc lập hoàn toàn. Nếu không phát hiện hiểu lầm cụ thể hoặc đáp án không thuộc danh mục hiểu lầm đã biết, hãy gán misconception_id là 'unlabeled'."
        )

        if feedback_error:
            prompt.append(
                f"\n⚠️ CHÚ Ý: Lần thử trước xuất output không hợp lệ với lỗi: {feedback_error}. Hãy sửa lại và chỉ xuất duy nhất 1 JSON hợp lệ!"
            )

        prompt.append("\nHãy suy luận CoT và xuất JSON kết quả:")
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
            logger.exception("Unexpected local backend failure")
            return None, "UNEXPECTED_ERROR"

    def _call_ollama(self, prompt: str) -> Optional[str]:
        return self._call_backend(prompt)[0]

    @staticmethod
    def _clean_and_extract_json(raw_text: str) -> str:
        if not raw_text:
            return ""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        return match.group(0).strip() if match else raw_text.strip()

    @staticmethod
    def _error_output(error_reason: str, selected_option: str) -> DiagnosticOutputSchema:
        return DiagnosticOutputSchema(
            misconception_id=UNLABELED_KEY,
            cot_reasoning=(
                f"1. Quan sát: Học sinh chọn [{selected_option}].\n"
                f"2. Không thể hoàn tất chẩn đoán do lỗi backend: {error_reason}.\n"
                "3. Kết luận: Đây là kết quả lỗi, không phải chẩn đoán của mô hình."
            ),
            confidence_score=0.0,
        )

    def diagnose(
        self, question: Dict[str, Any], selected_option: str
    ) -> Tuple[DiagnosticOutputSchema, bool, int, bool, Optional[str]]:
        """Return an explicitly classified failure; never replace it with mock inference."""
        feedback_error: Optional[str] = None
        last_error: Optional[str] = None
        attempts = 0
        fatal_errors = {"MODEL_NOT_FOUND", "CONNECTION_ERROR", "SECURITY_VIOLATION"}

        for attempts in range(1, self.max_retries + 1):
            raw, backend_error = self._call_backend(
                self.build_few_shot_prompt(question, selected_option, feedback_error)
            )
            if backend_error:
                last_error = backend_error
                logger.warning("Attempt %s backend error: %s", attempts, backend_error)
                if backend_error in fatal_errors or backend_error.startswith("HTTP_ERROR"):
                    if not self.enable_ollama_fallback:
                        raise RuntimeError(f"Diagnostic Engine failed fast: {backend_error}")
                    return self._error_output(backend_error, selected_option), False, attempts, True, backend_error
                continue

            if not raw or not raw.strip():
                last_error = "MALFORMED_OUTPUT"
                logger.warning("Attempt %s empty output", attempts)
                continue

            try:
                parsed = DiagnosticOutputSchema(**json.loads(self._clean_and_extract_json(raw)))
                return parsed, True, attempts, False, None
            except (json.JSONDecodeError, ValidationError) as error:
                feedback_error = str(error)
                last_error = "SCHEMA_VALIDATION_ERROR"
                logger.warning("Attempt %s schema error: %s", attempts, error)

        error_reason = last_error or "RETRY_EXHAUSTED"
        logger.error("Diagnostic retries exhausted; last error was %s", last_error)
        if not self.enable_ollama_fallback:
            raise RuntimeError(f"Diagnostic Engine failed: {error_reason}; last error: {last_error}")
        return self._error_output(error_reason, selected_option), False, attempts, True, error_reason


__all__ = ["LocalCoTDiagnosticEngine", "DiagnosticOutputSchema"]
