"""
unified_decision.py — 统一决策中心（改进12）

包含：
  - AblationConfig：消融实验开关配置
  - CONFIDENCE_BASELINES：各路径的置信度基准值
  - result_source 追踪辅助函数
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================
#  置信度配置
# ============================================================

CONFIDENCE_BASELINES = {
    "template_match": 0.8,       # 模板匹配成功
    "bridge_verified": 2.0,      # Bridge 验证（LLM 审核后接受）
    "fallback_accepted": 3.0,    # Fallback 接受（LLM 直接生成）
}


# ============================================================
#  消融实验配置
# ============================================================

@dataclass
class AblationConfig:
    """
    消融实验开关配置。
    用于评估各改进模块的独立贡献。
    所有开关默认为 True（启用全部改进）。

    enable_advanced_scoring 是统一开关，同时控制：
      - Path A 的递进式投票（progressive_mcs_vote）
      - Path B 的穷举分配路径（exhaustive_allocation_path）
    关闭此开关后，Path A 退化为简单最大原子数选取，Path B 被完全禁用。
    """
    enable_advanced_scoring: bool = True       # 统一开关：递进投票 + 穷举路径 B
    enable_multi_fragment: bool = True         # 多碎片合并（改进 6c）
    enable_template_matching: bool = True      # 模板匹配（改进 11）


# ============================================================
#  result_source 追踪
# ============================================================

def determine_result_source(
    workflow_source: Optional[str],
    stage1_case: Optional[str],
    success: bool,
) -> str:
    """
    根据工作流来源和阶段判定确定 result_source 标签。

    可能的返回值：
      - "Prebalance"：预平衡直接通过
      - "Invalid"：输入无效
      - "SynRBL2.0"：Path A 或 Path B 成功
      - "Template"：模板匹配成功
      - "Bridge LLM"：Bridge 阶段成功
      - "Fallback"：Fallback 阶段成功
      - "False"：所有路径均失败
    """
    if not success:
        if stage1_case and "invalid" in str(stage1_case).lower():
            return "Invalid"
        return "False"

    if workflow_source == "prebalance":
        return "Prebalance"
    elif workflow_source in ("SynRBL", "exhaustive_allocation"):
        return "SynRBL2.0"
    elif workflow_source == "template_matching":
        return "Template"
    elif workflow_source == "bridge":
        return "Bridge LLM"
    elif workflow_source == "fallback":
        return "Fallback"
    else:
        return workflow_source or "Unknown"


def apply_confidence_penalty(
    confidence: float,
    num_fragments: int = 0,
    ablation: Optional[AblationConfig] = None,
) -> float:
    """
    对置信度施加惩罚因子（用于模型未重训时的临时措施）。

    当多碎片合并（≥3 碎片）且启用了多碎片合并功能时，
    将 XGBoost 输出的置信度乘以 0.8 惩罚因子。

    原版模型训练数据中从未出现过 ≥3 碎片场景，
    其置信度输出可能偏高或波动较大。
    """
    if num_fragments >= 3 and (ablation is None or ablation.enable_multi_fragment):
        return confidence * 0.8
    return confidence
