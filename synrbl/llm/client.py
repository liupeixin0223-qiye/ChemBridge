import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def _extract_message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_parts: list[str] = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text is not None:
                    text_parts.append(str(text))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if message is None:
        return ""
    return str(message)

from .prompts import DIAGNOSIS_SYSTEM_PROMPT, GENERATE_SYSTEM_PROMPT, SCORE_SYSTEM_PROMPT

DEFAULT_MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1/chat/completions"
DEFAULT_SCORE_MODEL = "kimi-k2.5"
DEFAULT_GENERATE_MODEL = "kimi-k2.5"


def _resolve_temperature(model: str, thinking_enabled: bool) -> Optional[float]:
    normalized = (model or "").strip().lower()
    if not thinking_enabled and normalized in {"kimi-k2.5", "kimi-k2.6"}:
        return 0.6
    return None


class LLMResponseParseError(ValueError):
    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class MoonshotLLMClient:
    def __init__(
        self,
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: str = DEFAULT_MOONSHOT_BASE_URL,
        score_model: str = DEFAULT_SCORE_MODEL,
        generate_model: str = DEFAULT_GENERATE_MODEL,
        timeout: int = 240,
        thinking_enabled: bool = False,
    ):
        self.api_key_env = api_key_env

        if not base_url:
            self.base_url = DEFAULT_MOONSHOT_BASE_URL
        elif not base_url.endswith("/chat/completions"):
            self.base_url = base_url.rstrip("/") + "/chat/completions"
        else:
            self.base_url = base_url

        self.score_model = score_model
        self.generate_model = generate_model
        self.timeout = timeout
        self.thinking_enabled = thinking_enabled

    def diagnose_reaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        content = self._chat(
            model=self.generate_model,
            system_prompt=DIAGNOSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            parsed = self._parse_json_response(content)
        except Exception as exc:
            raise LLMResponseParseError(str(exc), content) from exc
        if not isinstance(parsed, dict):
            raise LLMResponseParseError(
                "LLM diagnosis response must be a JSON object.", content
            )
        parsed["_raw_response"] = content
        return parsed

    def score_candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        content = self._chat(
            model=self.score_model,
            system_prompt=SCORE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            parsed = self._parse_json_response(content)
        except Exception as exc:
            raise LLMResponseParseError(str(exc), content) from exc
        scores = parsed.get("scores")
        if not isinstance(scores, list):
            raise LLMResponseParseError(
                "LLM score response must contain a 'scores' list.", content
            )
        return {"scores": [float(score) for score in scores], "_raw_response": content}

    def generate_candidate(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        content = self._chat(
            model=self.generate_model,
            system_prompt=GENERATE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            parsed = self._parse_json_response(content)
        except Exception as exc:
            raise LLMResponseParseError(str(exc), content) from exc
        if not isinstance(parsed, dict):
            raise LLMResponseParseError(
                "LLM generate response must be a JSON object.", content
            )
        parsed["_raw_response"] = content
        return parsed

    def _chat(self, model: str, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise EnvironmentError(
                "Missing API key. Please set environment variable '{}' .".format(
                    self.api_key_env
                )
            )

        body = {
            "model": model,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not self.thinking_enabled:
            body["thinking"] = {"type": "disabled"}
        temperature = _resolve_temperature(model, self.thinking_enabled)
        if temperature is not None:
            body["temperature"] = temperature
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(api_key),
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            },
            method="POST",
        )
        max_retries = 3
        retryable_status_codes = (429, 500, 502, 503, 504)
        last_error: str | None = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = "HTTP {}: {}".format(exc.code, error_body)
                if exc.code in retryable_status_codes and attempt < max_retries - 1:
                    base_sleep = 2 ** attempt
                    jitter = random.uniform(0.0, 0.5)
                    time.sleep(base_sleep + jitter)
                    continue
                raise RuntimeError(
                    "LLM API request failed with status {}: {}".format(
                        exc.code, error_body
                    )
                ) from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_error = str(exc)
                if attempt < max_retries - 1:
                    base_sleep = 2 ** attempt
                    jitter = random.uniform(0.0, 0.5)
                    time.sleep(base_sleep + jitter)
                    continue
                raise RuntimeError("LLM API connection failed: {}".format(exc)) from exc
        else:
            raise RuntimeError("LLM API request exhausted retries: {}".format(last_error or "unknown error"))

        data = json.loads(response_body)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = _extract_message_content(message.get("content"))
            if not content.strip():
                finish_reason = choice.get("finish_reason")
                refusal = message.get("refusal") if isinstance(message, dict) else None
                reasoning_content = (
                    message.get("reasoning_content") if isinstance(message, dict) else None
                )
                raise ValueError(
                    "Empty LLM message content. finish_reason={}, refusal={}, reasoning_content={}, response={}".format(
                        finish_reason,
                        refusal,
                        reasoning_content,
                        response_body,
                    )
                )
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Unexpected LLM response structure: {}".format(response_body)
            ) from exc

    @classmethod
    def _parse_json_response(cls, content: str) -> Dict[str, Any]:
        cleaned = cls._clean_response_text(content)
        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response JSON root must be an object.")
            return parsed
        except json.JSONDecodeError:
            extracted = cls._extract_first_json_object(cleaned)
            parsed = json.loads(extracted)
            if not isinstance(parsed, dict):
                raise ValueError("LLM response JSON root must be an object.")
            return parsed

    @staticmethod
    def _clean_response_text(content: str) -> str:
        text = content.strip().replace("\ufeff", "")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @classmethod
    def _extract_first_json_object(cls, content: str) -> str:
        start = content.find("{")
        if start == -1:
            raise ValueError("No JSON object found in LLM response.")

        in_string = False
        escape = False
        depth = 0
        for idx in range(start, len(content)):
            char = content[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : idx + 1]
        raise ValueError("Incomplete JSON object in LLM response.")
