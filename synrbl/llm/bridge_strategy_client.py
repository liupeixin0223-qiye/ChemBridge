"""
bridge_strategy_client.py — Bridge 策略选择 LLM 客户端

与原版 SpeciesProposalClient 不同，此客户端：
- 接收预筛查后的候选方案列表
- 让 LLM 选择最优策略并判断反应类型
- 返回仅含 selected_strategy 和 reaction_type 的 JSON
"""

import json
from typing import Any, Dict

from .client import LLMResponseParseError, MoonshotLLMClient
from .bridge_strategy_prompts import (
    BRIDGE_STRATEGY_SYSTEM_PROMPT,
    BRIDGE_STRATEGY_USER_TEMPLATE,
)


class BridgeStrategyClient(MoonshotLLMClient):
    """Bridge 策略选择客户端。

    LLM 从"物种生成者"转变为"反应分析专家/策略选择者"。
    输出仅包含两个字段：selected_strategy 和 reaction_type。
    """

    def select_strategy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        将候选方案提交给 LLM 进行策略选择。

        参数:
            payload: 包含以下字段的字典:
                - original_reaction: 原始反应 SMILES
                - imbalance_analysis: 原子收支分析文本
                - available_options: 格式化后的候选选项文本
                - reaction_id: 反应标识符（可选）

        返回:
            包含 selected_strategy 和 reaction_type 的字典
        """
        user_prompt = BRIDGE_STRATEGY_USER_TEMPLATE.format(
            original_reaction=payload.get("original_reaction", ""),
            imbalance_analysis=payload.get("imbalance_analysis", ""),
            available_options=payload.get("available_options", ""),
        )

        content = self._chat(
            model=self.generate_model,
            system_prompt=BRIDGE_STRATEGY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        try:
            parsed = self._parse_json_response(content)
        except Exception as exc:
            raise LLMResponseParseError(str(exc), content) from exc

        if not isinstance(parsed, dict):
            raise LLMResponseParseError(
                "Bridge strategy response must be a JSON object.", content
            )

        # 验证必要字段
        if "selected_strategy" not in parsed:
            raise LLMResponseParseError(
                "Bridge strategy response missing 'selected_strategy' field.",
                content,
            )
        if "reaction_type" not in parsed:
            raise LLMResponseParseError(
                "Bridge strategy response missing 'reaction_type' field.",
                content,
            )

        parsed["_raw_response"] = content
        return parsed
