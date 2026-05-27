import json
from typing import Any, Dict

from .client import LLMResponseParseError, MoonshotLLMClient
from .species_prompts import SPECIES_DIAGNOSIS_SYSTEM_PROMPT


class SpeciesProposalClient(MoonshotLLMClient):
    def propose_side_species(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        content = self._chat(
            model=self.generate_model,
            system_prompt=SPECIES_DIAGNOSIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        try:
            parsed = self._parse_json_response(content)
        except Exception as exc:
            raise LLMResponseParseError(str(exc), content) from exc
        if not isinstance(parsed, dict):
            raise LLMResponseParseError(
                "Species proposal response must be a JSON object.", content
            )
        parsed["_raw_response"] = content
        return parsed
