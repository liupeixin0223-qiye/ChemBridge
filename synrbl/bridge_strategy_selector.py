"""
bridge_strategy_selector.py — Bridge 策略选择器（Section六 Bridge 重设计）

将 LLM 角色从"物种生成者"转变为"反应分析专家/策略选择者"。
实现预筛查、策略选择、后处理的完整流程。

五级级联中的第四级（Bridge）核心组件。
"""

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from rdkit import Chem
from synrbl.llm.client import LLMResponseParseError

logger = logging.getLogger("synrbl")


# ============================================================
#  工具函数
# ============================================================

def _canonicalize_smiles(smiles: str) -> str:
    """将 SMILES 转为 canonical 形式用于比较。"""
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def _reaction_to_canonical(reaction_smiles: str) -> str:
    """将反应式两侧的所有 SMILES 规范化并排序，用于去重比较。"""
    if not reaction_smiles or ">>" not in reaction_smiles:
        return reaction_smiles or ""
    parts = reaction_smiles.split(">>", 1)
    reactants = sorted(_canonicalize_smiles(s) for s in parts[0].split(".") if s)
    products = sorted(_canonicalize_smiles(s) for s in parts[1].split(".") if s)
    return "{}>>{}".format(".".join(reactants), ".".join(products))


def _analyze_reaction_balance(rxn_smiles: str) -> Dict[str, Any]:
    """
    精确计算反应式的原子收支。
    返回与 LLMSpeciesBridge.analyze_reaction_balance 兼容的格式。
    """
    import collections

    if ">>" not in rxn_smiles:
        return {
            "is_balanced": False,
            "imbalance_text": "Invalid reaction format",
            "missing_on_products": {},
            "missing_on_reactants": {},
        }

    split_parts = rxn_smiles.split(">>", 1)
    if len(split_parts) != 2:
        return {
            "is_balanced": False,
            "imbalance_text": "Invalid reaction format",
            "missing_on_products": {},
            "missing_on_reactants": {},
        }
    reactants_str, products_str = split_parts

    try:
        def get_counts(smi: str) -> Dict[str, int]:
            counts = collections.defaultdict(int)
            if not smi:
                return dict(counts)
            for part in smi.split("."):
                if not part:
                    continue
                # 优先使用完整消毒（正确计算芳香环隐式氢）
                mol = Chem.MolFromSmiles(part)
                if mol is None:
                    # 完整消毒失败（可能是 LLM 产出的非标准 SMILES），降级处理
                    mol = Chem.MolFromSmiles(part, sanitize=False)
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                        except Exception:
                            try:
                                mol.UpdatePropertyCache(strict=False)
                            except Exception:
                                mol = None
                if mol:
                    for atom in mol.GetAtoms():
                        counts[atom.GetSymbol()] += 1
                        counts["H"] += atom.GetTotalNumHs()
            return dict(counts)

        r_counts = get_counts(reactants_str)
        p_counts = get_counts(products_str)
        missing_on_products: Dict[str, int] = {}
        missing_on_reactants: Dict[str, int] = {}
        for el in sorted(set(r_counts.keys()).union(set(p_counts.keys()))):
            diff = r_counts.get(el, 0) - p_counts.get(el, 0)
            if diff > 0:
                missing_on_products[el] = diff
            elif diff < 0:
                missing_on_reactants[el] = abs(diff)

        parts = []
        if missing_on_products:
            parts.append(
                "Missing on Products: "
                + " ".join(f"{el}:{count}" for el, count in missing_on_products.items())
            )
        if missing_on_reactants:
            parts.append(
                "Missing on Reactants: "
                + " ".join(f"{el}:{count}" for el, count in missing_on_reactants.items())
            )

        return {
            "is_balanced": not missing_on_products and not missing_on_reactants,
            "imbalance_text": "; ".join(parts) if parts else "Exactly Balanced",
            "missing_on_products": missing_on_products,
            "missing_on_reactants": missing_on_reactants,
        }
    except Exception as exc:
        return {
            "is_balanced": False,
            "imbalance_text": f"Analysis error: {type(exc).__name__}: {exc}",
            "missing_on_products": {},
            "missing_on_reactants": {},
            "error": str(exc),
        }


def species_cancellation(reaction_smiles: str) -> str:
    """
    物种消去（最小计数法）。
    如果某个分子同时出现在反应物侧和产物侧，
    从两侧各移除较小数量的该分子。

    例如：反应物 2×H₂O，产物 3×H₂O → 两侧各移除 2×H₂O，
    产物侧剩余 1×H₂O。
    """
    if not reaction_smiles or ">>" not in reaction_smiles:
        return reaction_smiles

    sc_parts = reaction_smiles.split(">>", 1)
    if len(sc_parts) != 2:
        return reaction_smiles
    reactants_str, products_str = sc_parts

    # 规范化并计数
    r_species: Dict[str, int] = {}
    r_original: Dict[str, str] = {}  # canonical → original SMILES
    for smi in reactants_str.split("."):
        if not smi:
            continue
        canon = _canonicalize_smiles(smi)
        r_species[canon] = r_species.get(canon, 0) + 1
        r_original[canon] = smi

    p_species: Dict[str, int] = {}
    p_original: Dict[str, str] = {}
    for smi in products_str.split("."):
        if not smi:
            continue
        canon = _canonicalize_smiles(smi)
        p_species[canon] = p_species.get(canon, 0) + 1
        p_original[canon] = smi

    # 最小计数法消去
    all_species = set(r_species.keys()) | set(p_species.keys())
    new_r: List[str] = []
    new_p: List[str] = []

    for sp in all_species:
        r_count = r_species.get(sp, 0)
        p_count = p_species.get(sp, 0)
        cancel_count = min(r_count, p_count)

        # 剩余部分保留
        for _ in range(r_count - cancel_count):
            new_r.append(r_original.get(sp, sp))
        for _ in range(p_count - cancel_count):
            new_p.append(p_original.get(sp, sp))

    if not new_r or not new_p:
        # 消去后某一侧为空，保留原始反应式
        return reaction_smiles

    return "{}>>{}".format(".".join(new_r), ".".join(new_p))


# ============================================================
#  预筛查
# ============================================================

class CandidateOption:
    """表示一个候选配平方案（选项 A 或 B）。"""

    def __init__(
        self,
        label: str,
        reaction_smiles: str,
        source: str,
        score: Optional[float] = None,
        score_type: str = "confidence",
        template_label: str = "",
    ):
        self.label = label
        self.reaction_smiles = reaction_smiles
        self.source = source
        self.score = score
        self.score_type = score_type
        self.template_label = template_label

    def format_for_llm(self) -> str:
        """格式化为 LLM 可读的选项文本（不含置信度/相似度分数）。"""
        template_str = ""
        if self.template_label:
            template_str = f"（模板类型: {self.template_label}）"

        return (
            f"选项 {self.label}: {self.reaction_smiles}"
            f" [来源: {self.source}{template_str}]"
        )


def _format_template_inference_summary(detail: Dict[str, Any]) -> str:
    """将模板推断结果格式化为简短文本，供追加到选项列表末尾作为 LLM 推理参考。"""
    rxn_type = detail.get("reaction_type", "未知")
    return f"根据模板匹配分析，该反应最可能的类型是 {rxn_type}"


def prescreen_candidates(
    path_a_result: Optional[Dict[str, Any]],
    path_b_result: Optional[Dict[str, Any]],
) -> List[CandidateOption]:
    """
    预筛查流程（Section六 6.2）：
    1. 移除失败策略
    2. 结果去重
    3. 构建选项列表

    返回候选选项列表（可能为空，此时直接进入 Fallback）。
    """
    raw_options: List[CandidateOption] = []

    # 选项 A: Path A (MCS 配平)
    if path_a_result and path_a_result.get("formal_output_reaction"):
        rxn = path_a_result["formal_output_reaction"]
        conf = path_a_result.get("workflow_confidence")
        raw_options.append(CandidateOption(
            label="A",
            reaction_smiles=rxn,
            source="确定性算法 (Path A - MCS配平)",
            score=conf,
            score_type="confidence",
        ))

    # 选项 B: Path B (穷举分配)
    if path_b_result and path_b_result.get("formal_output_reaction"):
        rxn = path_b_result["formal_output_reaction"]
        conf = path_b_result.get("workflow_confidence")
        raw_options.append(CandidateOption(
            label="B",
            reaction_smiles=rxn,
            source="确定性算法 (Path B - 穷举分配)",
            score=conf,
            score_type="confidence",
        ))

    # 去重：如果两个选项产生完全相同的配平结果，合并为一个
    seen_canonical: Dict[str, List[CandidateOption]] = {}
    for opt in raw_options:
        canon = _reaction_to_canonical(opt.reaction_smiles)
        if canon not in seen_canonical:
            seen_canonical[canon] = []
        seen_canonical[canon].append(opt)

    deduped_options: List[CandidateOption] = []
    for canon, group in seen_canonical.items():
        if len(group) == 1:
            # 保留原始标签
            deduped_options.append(group[0])
        else:
            # 合并标签（如 A/B），保留原始标签字母
            merged_label = "/".join(sorted(set(g.label for g in group)))
            best = max(group, key=lambda x: x.score if x.score is not None else -1)
            best.label = merged_label
            best.source = " + ".join(sorted(set(g.source for g in group)))
            deduped_options.append(best)

    return deduped_options


# ============================================================
#  Bridge 策略选择器
# ============================================================

class BridgeStrategySelector:
    """
    Bridge 策略选择器（Section六 Bridge 重设计）。

    将 LLM 定位为"反应分析专家"，从确定性算法的候选方案中选择最优解，
    并判断反应类型。

    五级级联中的第四级。
    """

    def __init__(
        self,
        id_col: str = "R-id",
        reaction_col: str = "reactions",
        select_strategy_fn: Optional[Callable] = None,
        log_col: str = "llm_species_bridge",
        max_workers: int = 20,
    ):
        self.id_col = id_col
        self.reaction_col = reaction_col
        self.select_strategy_fn = select_strategy_fn
        self.log_col = log_col
        self.max_workers = max(1, int(max_workers))

    @classmethod
    def from_moonshot(
        cls,
        id_col: str = "R-id",
        reaction_col: str = "reactions",
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: str = "https://api.moonshot.cn/v1/chat/completions",
        model: str = "kimi-k2.5",
        max_workers: int = 20,
        thinking_enabled: bool = False,
    ) -> "BridgeStrategySelector":
        from synrbl.llm.bridge_strategy_client import BridgeStrategyClient

        client = BridgeStrategyClient(
            api_key_env=api_key_env,
            base_url=base_url,
            score_model=model,
            generate_model=model,
            thinking_enabled=thinking_enabled,
        )
        return cls(
            id_col=id_col,
            reaction_col=reaction_col,
            select_strategy_fn=client.select_strategy,
            max_workers=max_workers,
        )

    def apply(
        self,
        bridge_inputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对需要 Bridge 处理的反应执行策略选择流程。

        每个 bridge_input 应包含:
          - reaction_col: 原始反应 SMILES
          - internal_candidate_1_reaction: Path A 的结果（可选）
          - internal_candidate_1_confidence: Path A 的置信度（可选）
          - template_inference_detail: 模板推测结果（可选，作为推理参考）

        返回处理后的结果列表。
        """
        if self.select_strategy_fn is None:
            return bridge_inputs

        results: List[Dict[str, Any]] = []
        request_stats = {
            "max_workers": self.max_workers,
            "requested_count": len(bridge_inputs),
            "completed_count": 0,
            "parse_error_count": 0,
            "request_error_count": 0,
        }

        # 构建 LLM 请求
        llm_requests: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for reaction in bridge_inputs:
            original_rxn = reaction.get("original_reaction") or reaction.get(self.reaction_col, "")

            # 从 bridge_input 构建候选选项
            path_a_result = None
            c1_rxn = reaction.get("internal_candidate_1_reaction")
            c1_conf = reaction.get("internal_candidate_1_confidence")
            if c1_rxn:
                path_a_result = {
                    "formal_output_reaction": c1_rxn,
                    "workflow_confidence": c1_conf,
                }

            # Path B（穷举分配）候选方案
            path_b_result = None
            c2_rxn = reaction.get("internal_candidate_2_reaction")
            c2_conf = reaction.get("internal_candidate_2_confidence")
            if c2_rxn:
                path_b_result = {
                    "formal_output_reaction": c2_rxn,
                    "workflow_confidence": c2_conf,
                }

            candidates = prescreen_candidates(path_a_result, path_b_result)

            if not candidates:
                # 无候选方案，直接跳过 Bridge，进入 Fallback
                result = dict(reaction)
                # BR-1 修复：确保 log_col 是字典（可能从 CSV 加载为字符串）
                existing_log = result.get(self.log_col)
                result[self.log_col] = existing_log if isinstance(existing_log, dict) else {}
                result[self.log_col]["bridge_strategy"] = {
                    "status": "no_candidates",
                    "selected_strategy": "C",
                    "reaction_type": "其他反应类型 — 无候选方案",
                }
                result["bridge_selected_strategy"] = "C"
                result["bridge_reaction_type"] = "其他反应类型 — 无候选方案"
                result["bridge_verified_reaction"] = None
                result["bridge_verified_confidence"] = None
                results.append(result)
                continue

            # 格式化选项
            options_text = "\n".join(opt.format_for_llm() for opt in candidates)

            # 添加 C 选项（原 D 选项，"以上皆非"）
            options_text += (
                "\n选项 C: 以上皆非（现有配平方案均不合理，由LLM直接生成配平方案）"
            )

            # 追加模板推测结果（如有），作为 LLM 推理参考，不作为选项
            template_detail = reaction.get("template_inference_detail")
            if template_detail and isinstance(template_detail, dict):
                options_text += "\n" + _format_template_inference_summary(template_detail)
            elif "template_context_id" in reaction:
                # 模板匹配已执行但无子集记录 → 兜底提示
                options_text += "\n[模板推测: 模板库中暂时没有能合理匹配的模板，该反应类型可能不常见]"

            # 原子收支分析
            imbalance = _analyze_reaction_balance(original_rxn)
            imbalance_text = imbalance.get("imbalance_text", "Unknown")

            payload = {
                "reaction_id": reaction.get(self.id_col),
                "original_reaction": original_rxn,
                "imbalance_analysis": imbalance_text,
                "available_options": options_text,
            }

            llm_requests.append((reaction, payload))

        # 并行调用 LLM
        llm_results: Dict[str, Dict[str, Any]] = {}
        if llm_requests:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(self.select_strategy_fn, payload): reaction
                    for reaction, payload in llm_requests
                }
                for future in as_completed(future_map):
                    reaction = future_map[future]
                    # BR-2 修复：用内部追踪 ID 替代用户 ID，避免重复 ID 导致结果覆盖
                    tracking_key = str(reaction.get("_workflow_tracking_id", reaction.get(self.id_col)))
                    try:
                        llm_results[tracking_key] = {
                            "status": "ok",
                            "payload": future.result(),
                        }
                        request_stats["completed_count"] += 1
                    except LLMResponseParseError as exc:
                        llm_results[tracking_key] = {
                            "status": "parse_error",
                            "raw_response": exc.raw_response,
                            "error": str(exc),
                        }
                        request_stats["parse_error_count"] += 1
                    except Exception as exc:
                        llm_results[tracking_key] = {
                            "status": "request_error",
                            "error": str(exc),
                        }
                        request_stats["request_error_count"] += 1

        # 处理 LLM 结果
        for reaction, payload in llm_requests:
            result = dict(reaction)
            # BR-2 修复
            tracking_key = str(reaction.get("_workflow_tracking_id", reaction.get(self.id_col)))
            llm_result = llm_results.get(tracking_key, {
                "status": "request_error",
                "error": "Missing LLM result",
            })

            # BR-1 修复：确保 log_col 是字典（可能从 CSV 加载为字符串）
            existing_log = result.get(self.log_col)
            bridge_log = existing_log if isinstance(existing_log, dict) else {}
            bridge_log["bridge_strategy_request"] = copy.deepcopy(payload)
            bridge_log["request_stats"] = copy.deepcopy(request_stats)

            if llm_result["status"] != "ok":
                bridge_log["bridge_strategy"] = {
                    "status": "error",
                    "error": llm_result.get("error"),
                }
                result[self.log_col] = bridge_log
                result["bridge_selected_strategy"] = None
                result["bridge_reaction_type"] = None
                result["bridge_verified_reaction"] = None
                result["bridge_verified_confidence"] = None
                results.append(result)
                continue

            llm_response = llm_result["payload"]
            selected = str(llm_response.get("selected_strategy", "")).strip().upper()
            reaction_type = str(llm_response.get("reaction_type", "")).strip()

            bridge_log["bridge_strategy"] = {
                "status": "ok",
                "selected_strategy": selected,
                "reaction_type": reaction_type,
                "raw_response": llm_response.get("_raw_response", ""),
            }

            result["bridge_selected_strategy"] = selected
            result["bridge_reaction_type"] = reaction_type

            # 根据 LLM 选择进行后处理
            if selected in ("A", "B", "A/B"):
                # 路径一：LLM 选择了策略 A 或 B
                chosen_reaction = self._find_chosen_reaction(
                    selected, reaction
                )
                if chosen_reaction:
                    # 物种消去
                    cancelled = species_cancellation(chosen_reaction)
                    # 原子守恒验证
                    balance = _analyze_reaction_balance(cancelled)
                    if balance.get("is_balanced"):
                        # 成功：置信度 2.0
                        result["bridge_verified_reaction"] = cancelled
                        result["bridge_verified_confidence"] = 2.0
                        bridge_log["bridge_strategy"]["post_processing"] = {
                            "species_cancellation_applied": cancelled != chosen_reaction,
                            "balance_verified": True,
                            "confidence": 2.0,
                        }
                    else:
                        # 物种消去后仍不守恒 → 需要 Fallback
                        result["bridge_verified_reaction"] = None
                        result["bridge_verified_confidence"] = None
                        bridge_log["bridge_strategy"]["post_processing"] = {
                            "species_cancellation_applied": cancelled != chosen_reaction,
                            "balance_verified": False,
                            "imbalance_after_cancel": balance.get("imbalance_text"),
                        }
                        bridge_log["bridge_strategy"]["needs_fallback"] = True
                        bridge_log["bridge_strategy"]["fallback_input"] = {
                            "source": "bridge_post_processing_failed",
                            "reaction": cancelled,
                            "imbalance": balance,
                        }
                else:
                    # 找不到对应的候选反应 → Fallback
                    result["bridge_verified_reaction"] = None
                    result["bridge_verified_confidence"] = None
                    bridge_log["bridge_strategy"]["needs_fallback"] = True
            elif selected == "C":
                # 路径二：LLM 选择了 C → 直接进入 Fallback
                result["bridge_verified_reaction"] = None
                result["bridge_verified_confidence"] = None
                bridge_log["bridge_strategy"]["needs_fallback"] = True
                bridge_log["bridge_strategy"]["fallback_input"] = {
                    "source": "llm_selected_C",
                    "reaction": reaction.get("original_reaction")
                        or reaction.get(self.reaction_col, ""),
                    "reaction_type": reaction_type,
                    "imbalance": _analyze_reaction_balance(
                        reaction.get("original_reaction")
                        or reaction.get(self.reaction_col, "")
                    ),
                }
            else:
                # 无效选择 → Fallback
                result["bridge_verified_reaction"] = None
                result["bridge_verified_confidence"] = None
                bridge_log["bridge_strategy"]["error"] = (
                    f"Invalid strategy selection: '{selected}'"
                )
                bridge_log["bridge_strategy"]["needs_fallback"] = True

            result[self.log_col] = bridge_log
            results.append(result)

        # 添加无候选方案的输入
        for reaction in bridge_inputs:
            # BR-2 修复：用内部追踪 ID 进行匹配
            tracking_key = str(reaction.get("_workflow_tracking_id", reaction.get(self.id_col)))
            if not any(
                str(r.get("_workflow_tracking_id", r.get(self.id_col))) == tracking_key for r in results
            ):
                # R-10 修复：为安全网追加的反应添加默认 bridge 字段，
                # 确保输出结构一致，下游编排器可统一读取
                reaction.setdefault("bridge_verified_reaction", None)
                reaction.setdefault("bridge_verified_confidence", None)
                reaction.setdefault("bridge_selected_strategy", None)
                reaction.setdefault("bridge_reaction_type", None)
                results.append(reaction)

        return results

    def _find_chosen_reaction(
        self,
        selected_label: str,
        reaction: Dict[str, Any],
    ) -> Optional[str]:
        """根据 LLM 选择的标签找到对应的候选反应 SMILES。"""
        # 从 bridge_input 重建候选选项
        path_a_result = None
        c1_rxn = reaction.get("internal_candidate_1_reaction")
        c1_conf = reaction.get("internal_candidate_1_confidence")
        if c1_rxn:
            path_a_result = {
                "formal_output_reaction": c1_rxn,
                "workflow_confidence": c1_conf,
            }

        path_b_result = None
        c2_rxn = reaction.get("internal_candidate_2_reaction")
        c2_conf = reaction.get("internal_candidate_2_confidence")
        if c2_rxn:
            path_b_result = {
                "formal_output_reaction": c2_rxn,
                "workflow_confidence": c2_conf,
            }

        candidates = prescreen_candidates(path_a_result, path_b_result)

        # 查找匹配的标签
        for opt in candidates:
            if selected_label in opt.label:
                return opt.reaction_smiles

        # C-BR1 修复：不再静默回退到第一个候选。
        # 当 LLM 返回非法策略标签（如 "D"）时返回 None，
        # 让调用者的"找不到对应的候选反应"分支正确处理，路由到 Fallback。
        return None
