import json
from typing import Any, Dict, Optional

from .client import LLMResponseParseError, MoonshotLLMClient
from .fallback_prompts import FALLBACK_GENERATE_SYSTEM_PROMPT


class FallbackGenerateClient(MoonshotLLMClient):
    def generate_candidate(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        content = self._chat(
            model=self.generate_model,
            system_prompt=FALLBACK_GENERATE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            parsed = self._parse_json_response(content)
        except Exception as exc:
            raise LLMResponseParseError(str(exc), content) from exc
        if not isinstance(parsed, dict):
            raise LLMResponseParseError(
                "Fallback generate response must be a JSON object.", content
            )
        parsed["_raw_response"] = content
        return parsed
