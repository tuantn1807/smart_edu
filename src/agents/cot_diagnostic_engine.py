"""
Local Chain-of-Thought (CoT) Diagnostic Engine.

Uses Few-shot CoT Prompting on Local Open-Weights LLM (e.g. Qwen2.5-7B-Instruct / Llama-3.1-8B)
via Ollama/vLLM API with Pydantic schema guardrails and automatic retry mechanisms.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

UNLABELED_KEY = "unlabeled"


class DiagnosticOutputSchema(BaseModel):
    """Pydantic guardrail schema for Diagnostic Engine output."""
    misconception_id: str = Field(
        ...,
        description="ID of the identified misconception (e.g. '157', '1672') or 'unlabeled' if none applies."
    )
    cot_reasoning: str = Field(
        ...,
        description="Detailed step-by-step Chain-of-Thought explanation of student's mistake."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
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
            "confidence_score": 0.95
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
                "4. Kết luận: Học sinh bị rổng kiến thức về quy tắc chuyển vế trong phương trình."
            ),
            "misconception_id": "312",
            "confidence_score": 0.92
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
            "confidence_score": 0.96
        }
    ]

    def __init__(
        self,
        model_name: str = "qwen2.5:7b-instruct",
        api_base: str = "http://localhost:11434",
        seed: int = 42,
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 10,
        enable_ollama_fallback: bool = True
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.enable_ollama_fallback = enable_ollama_fallback

    def build_few_shot_prompt(
        self,
        question: Dict[str, Any],
        selected_option: str,
        feedback_error: Optional[str] = None
    ) -> str:
        """Construct prompt with system instructions, few-shot examples, and input context."""
        question_text = question.get("question_text", "")
        options = question.get("options", {})
        correct_option = question.get("correct_option", "")
        concept_name = question.get("concept_name", "Chưa xác định")
        misconception_map = question.get("misconception_map", {})

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
            "Ví dụ minh họa (Few-Shot Examples):"
        ]

        for i, ex in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            prompt.append(f"\n--- VÍ DỤ {i} ---")
            prompt.append(f"Đề bài: {ex['question']}")
            prompt.append("Lựa chọn: " + ", ".join([f"[{k}] {v}" for k, v in ex['options'].items()]))
            prompt.append(f"Đáp án đúng: [{ex['correct_option']}]")
            prompt.append(f"Học sinh chọn: [{ex['selected_option']}]")
            prompt.append(f"Khái niệm: {ex['concept_name']}")
            prompt.append("Output JSON:")
            ex_json = {
                "misconception_id": ex["misconception_id"],
                "cot_reasoning": ex["cot_reasoning"],
                "confidence_score": ex["confidence_score"]
            }
            prompt.append(f"```json\n{json.dumps(ex_json, ensure_ascii=False, indent=2)}\n```")

        prompt.append("\n--- BÀI CẦN CHẨN ĐOÁN THỰC TẾ ---")
        prompt.append(f"Đề bài: {question_text}")
        prompt.append("Lựa chọn: " + ", ".join([f"[{k}] {v}" for k, v in options.items()]))
        prompt.append(f"Đáp án đúng: [{correct_option}]")
        prompt.append(f"Học sinh chọn: [{selected_option}]")
        prompt.append(f"Khái niệm: {concept_name}")

        prompt.append("Lưu ý: Nếu không phát hiện hiểu lầm cụ thể hoặc đáp án không thuộc danh mục hiểu lầm đã biết, hãy gán misconception_id là 'unlabeled'.")

        if feedback_error:
            prompt.append(f"\n⚠️ CHÚ Ý: Lần thử trước xuất output không hợp lệ với lỗi: {feedback_error}. Hãy sửa lại và chỉ xuất duy nhất 1 JSON hợp lệ!")

        prompt.append("\nHãy suy luận CoT và xuất JSON kết quả:")
        return "\n".join(prompt)

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Calls local Ollama REST API endpoint with seed & temperature parameters."""
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    resp_obj = json.loads(response.read().decode("utf-8"))
                    return resp_obj.get("response", "")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            logger.debug(f"Ollama connection/HTTP call failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in Ollama call: {e}")
            return None

    def _clean_and_extract_json(self, raw_text: str) -> str:
        """Extracts JSON block from raw text (handling markdown fences or surrounding prose)."""
        if not raw_text:
            return ""
        # Match ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Match outermost { ... }
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()

        return raw_text.strip()

    def _mock_inference(
        self,
        question: Dict[str, Any],
        selected_option: str
    ) -> Dict[str, Any]:
        """Deterministic mock inference mode when local Ollama service is unavailable."""
        correct_option = question.get("correct_option", "")
        misconception_map = question.get("misconception_map", {})
        concept_name = question.get("concept_name", "Khái niệm Toán")
        opt_text = question.get("options", {}).get(selected_option, "")

        if selected_option.upper() == correct_option.upper():
            return {
                "misconception_id": UNLABELED_KEY,
                "cot_reasoning": f"1. Quan sát: Học sinh chọn đáp án đúng [{selected_option}] ({opt_text}).\n2. Phân tích: Không phát hiện lỗi sai.\n3. Kết luận: Học sinh làm đúng câu hỏi về '{concept_name}'.",
                "confidence_score": 1.0
            }

        misc_info = misconception_map.get(selected_option)
        if misc_info and "misconception_id" in misc_info:
            mid = str(misc_info["misconception_id"])
            mname = misc_info.get("name", "Hiểu lầm toán học")
            mdesc = misc_info.get("description", "")
            cot = (
                f"1. Quan sát: Học sinh chọn phương án [{selected_option}] ({opt_text}) thay vì đáp án đúng [{correct_option}].\n"
                f"2. Phân tích lỗi: Học sinh mắc hiểu lầm '{mname}' (ID: {mid}).\n"
                f"3. Diễn giải chi tiết: {mdesc}\n"
                f"4. Kết luận: Cần củng cố kiến thức về '{concept_name}'."
            )
            return {
                "misconception_id": mid,
                "cot_reasoning": cot,
                "confidence_score": 0.95
            }
        else:
            cot = (
                f"1. Quan sát: Học sinh chọn phương án [{selected_option}] thay vì đáp án đúng [{correct_option}].\n"
                f"2. Phân tích lỗi: Phương án này chưa được gán nhãn cụ thể trong Eedi rubric.\n"
                f"3. Suy luận: Học sinh có thể đoán mò hoặc mắc lỗi tính toán ngẫu nhiên.\n"
                f"4. Kết luận: Đánh dấu 'unlabeled' cho câu hỏi '{concept_name}'."
            )
            return {
                "misconception_id": UNLABELED_KEY,
                "cot_reasoning": cot,
                "confidence_score": 0.70
            }

    def diagnose(
        self,
        question: Dict[str, Any],
        selected_option: str
    ) -> Tuple[DiagnosticOutputSchema, bool, int, bool]:
        """
        Executes Local CoT Diagnosis with guardrails and retries.

        Returns:
            Tuple[DiagnosticOutputSchema, is_valid_parse, attempts_taken, used_fallback]
        """
        feedback_error: Optional[str] = None
        attempts = 0

        for attempt in range(1, self.max_retries + 1):
            attempts = attempt
            prompt = self.build_few_shot_prompt(question, selected_option, feedback_error)
            raw_output = self._call_ollama(prompt)

            if not raw_output:
                # If Ollama is offline/unreachable and fallback is allowed
                if self.enable_ollama_fallback:
                    mock_dict = self._mock_inference(question, selected_option)
                    parsed = DiagnosticOutputSchema(**mock_dict)
                    return parsed, False, 1, True
                else:
                    mock_dict = self._mock_inference(question, selected_option)
                    parsed = DiagnosticOutputSchema(**mock_dict)
                    return parsed, False, 1, False

            json_str = self._clean_and_extract_json(raw_output)
            try:
                data = json.loads(json_str)
                parsed = DiagnosticOutputSchema(**data)
                return parsed, True, attempt, False
            except (json.JSONDecodeError, ValidationError) as err:
                feedback_error = str(err)
                logger.warning(f"Attempt {attempt} failed schema validation: {err}")

        # If retries exceeded, fallback to deterministic mock representation
        fallback_dict = self._mock_inference(question, selected_option)
        used_fallback = self.enable_ollama_fallback
        return DiagnosticOutputSchema(**fallback_dict), False, attempts, used_fallback
