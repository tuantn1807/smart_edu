"""Local CoT diagnostic engine with explicit backend-failure handling."""
from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)
UNLABELED_KEY = "unlabeled"


class DiagnosticOutputSchema(BaseModel):
    misconception_id: str = Field(...)
    cot_reasoning: str = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class LocalCoTDiagnosticEngine:
    """Diagnostic engine backed by a local Ollama or vLLM endpoint."""

    FEW_SHOT_EXAMPLES = [
        {"question": "Tính 1/2 + 1/3.", "options": {"A": "2/5", "B": "5/6"}, "selected_option": "A"},
        {"question": "Giải phương trình 2x + 4 = 10.", "options": {"A": "x = 7", "B": "x = 3"}, "selected_option": "A"},
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
        # Kept as a compatibility attribute; prompt construction is always independent.
        self.diagnosis_mode = diagnosis_mode.lower()
        self.seed = seed
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_ollama_fallback = enable_ollama_fallback
        if self.backend not in ("ollama", "vllm"):
            raise ValueError(f"Unsupported backend '{self.backend}'. Must be 'ollama' or 'vllm'.")
        self._validate_local_endpoint(self.api_base)

    @staticmethod
    def _validate_local_endpoint(url: str) -> None:
        hostname = (urllib.parse.urlparse(url).hostname or "").lower()
        if hostname not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            raise ValueError(f"api_base must be local; got hostname '{hostname}'")

    def build_few_shot_prompt(
        self, question: Dict[str, Any], selected_option: str, feedback_error: Optional[str] = None
    ) -> str:
        """Build an independent prompt; no answer, rubric, concept, or candidate IDs are injected."""
        question_text = question.get("question_text", "")
        options = question.get("options", {})
        prompt = [
            "Bạn là Diagnostic AI Agent phân tích nguyên nhân lỗi sai của học sinh môn Toán.",
            "Hãy chỉ dựa trên đề bài, các phương án và lựa chọn của học sinh.",
            "Không sử dụng đáp án đúng, rubric, concept_name, misconception_map, hoặc nhãn ứng viên.",
            "Chỉ trả về một JSON hợp lệ với các trường misconception_id, cot_reasoning, confidence_score.",
            'misconception_id phải là chuỗi hoặc "unlabeled"; confidence_score nằm trong [0.0, 1.0].',
            "",
            "Ví dụ minh họa không có nhãn rubric:",
        ]
        for i, example in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            prompt.extend([
                f"--- VÍ DỤ {i} ---",
                f"Đề bài: {example['question']}",
                "Lựa chọn: " + ", ".join(f"[{k}] {v}" for k, v in example["options"].items()),
                f"Học sinh chọn: [{example['selected_option']}]",
                'Output JSON: {"misconception_id":"unlabeled","cot_reasoning":"...","confidence_score":0.0}',
            ])
        prompt.extend([
            "--- BÀI CẦN CHẨN ĐOÁN ---",
            f"Đề bài: {question_text}",
            "Lựa chọn: " + ", ".join(f"[{k}] {v}" for k, v in options.items()),
            f"Học sinh chọn: [{selected_option}]",
        ])
        if feedback_error:
            prompt.append(f"Output trước không hợp lệ ({feedback_error}); hãy trả về JSON hợp lệ duy nhất.")
        prompt.append("Hãy xuất JSON kết quả:")
        return "\n".join(prompt)

    def _call_backend(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        if self.backend == "ollama":
            url = f"{self.api_base}/api/generate"
            payload = {"model": self.model_name, "prompt": prompt, "stream": False,
                       "options": {"temperature": self.temperature, "seed": self.seed}}
        else:
            url = f"{self.api_base}/v1/chat/completions"
            payload = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                       "temperature": self.temperature, "seed": self.seed}
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    return None, f"HTTP_ERROR_{response.status}"
                body = json.loads(response.read().decode())
                if self.backend == "ollama":
                    return body.get("response", ""), None
                choices = body.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", ""), None
                if choices:
                    return choices[0].get("text", ""), None
                return "", None
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
        for attempts in range(1, self.max_retries + 1):
            raw, backend_error = self._call_backend(
                self.build_few_shot_prompt(question, selected_option, feedback_error)
            )
            if backend_error:
                last_error = backend_error
                logger.warning("Attempt %s backend error: %s", attempts, backend_error)
                continue
            if not raw or not raw.strip():
                last_error = "MALFORMED_OUTPUT"
                continue
            try:
                parsed = DiagnosticOutputSchema(**json.loads(self._clean_and_extract_json(raw)))
                return parsed, True, attempts, False, None
            except (json.JSONDecodeError, ValidationError) as error:
                feedback_error = str(error)
                last_error = "SCHEMA_VALIDATION_ERROR"
                logger.warning("Attempt %s schema error: %s", attempts, error)

        error_reason = "RETRY_EXHAUSTED"
        logger.error("Diagnostic retries exhausted; last error was %s", last_error)
        if not self.enable_ollama_fallback:
            raise RuntimeError(f"Diagnostic Engine failed: {error_reason}; last error: {last_error}")
        return self._error_output(error_reason, selected_option), False, attempts, True, error_reason


__all__ = ["LocalCoTDiagnosticEngine", "DiagnosticOutputSchema"]
