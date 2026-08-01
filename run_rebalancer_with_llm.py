import argparse
import collections
import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synrbl import Balancer
from synrbl.evaluation_utils import write_accuracy_report, write_workflow_statistics
from synrbl.llm_fallback_postprocessor import LLMFallbackPostprocessor
from synrbl.preprocess import preprocess, input_sanitize_check
from synrbl.exhaustive_allocation import exhaustive_allocation_path
from synrbl.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from synrbl.template_matching import template_matching_for_reaction
from synrbl.unified_decision import (
    AblationConfig,
    CONFIDENCE_BASELINES,
    determine_result_source,
    apply_confidence_penalty,
)
from synrbl.bridge_strategy_selector import BridgeStrategySelector

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
WORKFLOW_THRESHOLD = 0.8
INTERNAL_TRACKING_ID = "_workflow_tracking_id"


@dataclass(frozen=True)
class RebalanceConfig:
    """主编排器专用配置。仅包含当前五级级联实际读取的字段。"""
    reaction_col: str = "reactions"
    id_col: str = "R-id"
    n_jobs: int = 1
    synrbl_confidence_threshold: float = 0.8
    enable_advanced_scoring: bool = True
    enable_multi_fragment: bool = True


def _load_table(input_path: Path, sep: str | None = None) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported input file format '{suffix}'. Supported formats: {sorted(SUPPORTED_SUFFIXES)}"
        )
    if suffix == ".csv":
        return pd.read_csv(input_path, sep=sep or ",")
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(input_path, sep=sep or "\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    if suffix == ".json":
        try:
            return pd.read_json(input_path)
        except ValueError:
            return pd.DataFrame(pd.read_json(input_path, orient="records"))
    raise ValueError(f"Unsupported input file format '{suffix}'.")


def _resolve_column(df: pd.DataFrame, requested: str, fallback_candidates: list[str]) -> str:
    if requested in df.columns:
        return requested
    lowered = {str(col).lower(): str(col) for col in df.columns}
    if requested.lower() in lowered:
        return lowered[requested.lower()]
    for candidate in fallback_candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Column '{requested}' not found. Available columns: {list(df.columns)}")


def _sanitize_filename_part(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = text.strip("-._")
    return text or "run"


def _build_run_label(start_row: int | None, end_row: int | None, row_count: int) -> str:
    if start_row is None and end_row is None:
        return f"rows-1-{row_count}"
    if start_row is None:
        return f"rows-1-{end_row}"
    if end_row is None:
        return f"rows-{start_row}-end"
    return f"rows-{start_row}-{end_row}"


def _slice_dataframe(df: pd.DataFrame, start_row: int | None, end_row: int | None) -> pd.DataFrame:
    if start_row is None and end_row is None:
        return df.copy()
    total_rows = len(df)
    start = 1 if start_row is None else start_row
    end = total_rows if end_row is None else end_row
    if start < 1:
        raise ValueError("--start-row 必须 >= 1（包含题头行之外的数据首行为第 1 行）")
    if end < 1:
        raise ValueError("--end-row 必须 >= 1（包含题头行之外的数据首行为第 1 行）")
    if start > end:
        raise ValueError("--start-row 不能大于 --end-row")
    if start > total_rows:
        raise ValueError(f"--start-row={start} 超出数据行数范围。当前数据共有 {total_rows} 行（不含题头）。")
    end = min(end, total_rows)
    return df.iloc[start - 1 : end].copy()


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace("％", "%")
        if not text or text.lower() in {"nan", "none"}:
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
            try:
                return round(float(text) / 100.0, 6)
            except (TypeError, ValueError):
                return None
        try:
            return float(text)
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_atom_counts(rxn_side: str) -> dict[str, int]:
    """统计反应一侧各元素的原子总数（含隐式氢）。"""
    from rdkit import Chem

    counts: dict[str, int] = collections.defaultdict(int)
    if not rxn_side:
        return dict(counts)
    for part in rxn_side.split("."):
        if not part:
            continue
        # 优先使用完整消毒（正确计算芳香环隐式氢）
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            mol = Chem.MolFromSmiles(part, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except Exception:
                    mol.UpdatePropertyCache(strict=False)
        if mol:
            for atom in mol.GetAtoms():
                counts[atom.GetSymbol()] += 1
                counts["H"] += atom.GetTotalNumHs()
    return dict(counts)


def _is_balanced(reaction: Any) -> bool:
    """判断反应是否原子守恒。"""
    text = str(reaction or "").strip()
    if not text or ">>" not in text:
        return False
    try:
        reactants, products = text.split(">>", 1)
        r_counts = _get_atom_counts(reactants)
        p_counts = _get_atom_counts(products)
        return r_counts == p_counts
    except Exception:
        return False


def _is_valid_reaction_smiles(reaction: Any) -> bool:
    """判断反应 SMILES 格式是否合法（两侧均有合法 SMILES）。"""
    from rdkit import Chem

    text = str(reaction or "").strip()
    if not text or ">>" not in text:
        return False
    try:
        reactants, products = text.split(">>", 1)
    except ValueError:
        return False
    if not reactants.strip() or not products.strip():
        return False
    for side in (reactants, products):
        for token in side.split("."):
            token = token.strip()
            if not token:
                return False
            try:
                if Chem.MolFromSmiles(token) is None:
                    return False
            except Exception:
                return False
    return True


def _serialize_rows_for_csv(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                out[key] = json.dumps(value, ensure_ascii=False)
            else:
                out[key] = value
        serialized.append(out)
    return serialized


def _write_json_and_csv(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        pd.DataFrame(_serialize_rows_for_csv(rows)).to_csv(csv_path, index=False, encoding="utf-8-sig")
    except PermissionError as exc:
        print(f"Warning: failed to write CSV because the file is in use or locked: {csv_path} ({exc})")


def _build_base_config(args: argparse.Namespace, reaction_col: str, id_col: str) -> RebalanceConfig:
    return RebalanceConfig(
        reaction_col=reaction_col,
        id_col=id_col,
        synrbl_confidence_threshold=args.synrbl_confidence_threshold,
        enable_advanced_scoring=not args.disable_advanced_scoring,
        enable_multi_fragment=not args.disable_multi_fragment,
    )


def _build_balancer(cfg: RebalanceConfig) -> Balancer:
    return Balancer(
        reaction_col=cfg.reaction_col,
        id_col=cfg.id_col,
        confidence_threshold=cfg.synrbl_confidence_threshold,
        llm_postprocessor=None,
        llm_species_bridge=None,
        llm_fallback_postprocessor=None,
        enable_advanced_scoring=cfg.enable_advanced_scoring,
        enable_multi_fragment=cfg.enable_multi_fragment,
    )


def _prepare_records(df: pd.DataFrame, reaction_col: str, id_col: str) -> list[dict[str, Any]]:
    records = df.to_dict("records")
    prepared: list[dict[str, Any]] = []
    for idx, row in enumerate(records):
        item = dict(row)
        item.setdefault("original_row_index", idx)
        item.setdefault("original_reaction", item.get(reaction_col))
        item[INTERNAL_TRACKING_ID] = f"workflow-{idx}"
        prepared.append(item)
    return prepared


def _build_round_summary(row: dict[str, Any], reaction_col: str, round_label: str) -> dict[str, Any]:
    solved_by = str(row.get("solved_by") or "").strip()
    confidence = _normalize_confidence(row.get("confidence"))
    balanced = _is_balanced(row.get(reaction_col))
    summary = {
        "round_label": round_label,
        "reaction": row.get(reaction_col),
        "balanced": balanced,
        "solved": bool(row.get("solved", False)),
        "solved_by": solved_by or None,
        "confidence": confidence,
        "confidence_available": confidence is not None,
        "used_rule_based_only": solved_by == "rule-based",
        "used_mcs": solved_by == "mcs-based",
        "issue": row.get("issue"),
    }
    row.setdefault("workflow_stage_summary", {})[f"{round_label}_summary"] = copy.deepcopy(summary)
    return summary


def _run_first_round(
    records: list[dict[str, Any]],
    cfg: RebalanceConfig,
    balancer: "Balancer | None" = None,
) -> list[dict[str, Any]]:
    if balancer is None:
        balancer = _build_balancer(cfg)
    working = copy.deepcopy(records)

    # === 步骤 0: 输入验证（input_sanitize_check）===
    valid_working = []
    invalid_working = []
    for row in working:
        rxn_smiles = str(row.get(cfg.reaction_col, ""))
        rxn_id = str(row.get(cfg.id_col, ""))
        check_result = input_sanitize_check(rxn_smiles, reaction_id=rxn_id)
        if check_result["valid"]:
            valid_working.append(row)
        else:
            row["processable"] = False
            row["preprocess_status"] = "invalid_smiles"
            row["issue"] = "; ".join(check_result["errors"])
            row["solved"] = False
            row.setdefault("workflow_stage_summary", {})["input_sanitize"] = {
                "valid": False,
                "errors": check_result["errors"],
            }
            invalid_working.append(row)

    working = preprocess(
        valid_working,
        cfg.reaction_col,
        cfg.id_col,
        "solved",
        "input_reaction",
        n_jobs=cfg.n_jobs,
        remove_aam=True,
    )
    for row in working:
        row.setdefault("cleaned_initial_reaction", row.get(cfg.reaction_col))
        row.setdefault("original_reaction", row.get(cfg.reaction_col))
    balancer.run_prebalance_check(working, stats={})
    remaining = [row for row in working if not row.get("prebalance_check", {}).get("short_circuited", False)]
    if remaining:
        balancer.run_core_pipeline(remaining, stats={}, allow_low_confidence_solved=True)
    for row in working:
        if row.get("prebalance_check", {}).get("short_circuited", False):
            row.setdefault("workflow_stage_summary", {})["prebalance_summary"] = {
                "round_label": "prebalance",
                "reaction": row.get(cfg.reaction_col),
                "balanced": True,
                "solved": True,
                "solved_by": "prebalanced",
                "confidence": 1.0,
                "confidence_available": True,
                "used_rule_based_only": False,
                "used_mcs": False,
                "issue": row.get("issue"),
            }
        else:
            _build_round_summary(row, cfg.reaction_col, "first_round")
    # 将未通过输入验证的反应追加回结果列表
    working.extend(invalid_working)
    return working


def _stage1_decision(row: dict[str, Any], reaction_col: str) -> dict[str, Any]:
    reaction = row.get(reaction_col)
    balanced = _is_balanced(reaction)
    reaction_valid = _is_valid_reaction_smiles(reaction)
    solved_by = str(row.get("solved_by") or "").strip()
    confidence = _normalize_confidence(row.get("confidence"))
    decision = {
        "formal_output_reaction": None,
        "workflow_confidence": None,
        "workflow_source": "SynRBL",
        "internal_candidate_1_reaction": None,
        "internal_candidate_1_confidence": None,
        "internal_candidate_1_source": None,
        "needs_bridge": False,
        "stage1_case": None,
    }
    if row.get("preprocess_status") == "invalid_smiles":
        decision["stage1_case"] = "invalid_smiles"
        decision["needs_bridge"] = False
        return decision
    if row.get("prebalance_check", {}).get("short_circuited", False):
        decision["formal_output_reaction"] = reaction if reaction_valid else None
        decision["workflow_confidence"] = 1.0 if reaction_valid else None
        decision["workflow_source"] = "prebalance"
        decision["stage1_case"] = "prebalance" if reaction_valid else "prebalance_invalid_reaction"
        decision["needs_bridge"] = not reaction_valid
        return decision
    if not balanced or not reaction_valid:
        decision["stage1_case"] = "unbalanced" if not balanced else "invalid_reaction"
        decision["needs_bridge"] = True
        return decision
    if solved_by == "rule-based":
        decision["formal_output_reaction"] = reaction
        decision["workflow_confidence"] = 1.0
        decision["stage1_case"] = "small_molecule"
        return decision
    if confidence is None:
        decision["stage1_case"] = "mcs_missing_confidence"
        decision["needs_bridge"] = True
        return decision
    if confidence >= WORKFLOW_THRESHOLD:
        decision["formal_output_reaction"] = reaction
        decision["workflow_confidence"] = confidence
        decision["stage1_case"] = "mcs_high_confidence"
        return decision
    decision["internal_candidate_1_reaction"] = reaction
    decision["internal_candidate_1_confidence"] = confidence
    decision["internal_candidate_1_source"] = "SynRBL"
    decision["needs_bridge"] = True
    decision["stage1_case"] = "mcs_low_confidence"
    return decision


def _try_exhaustive_allocation(
    row: dict[str, Any],
    reaction_col: str,
    decision: dict[str, Any],
    enable_advanced_scoring: bool = True,
    balancer: "Balancer | None" = None,
) -> dict[str, Any]:
    """在 _stage1_decision 判定 needs_bridge=True 之后、LLM Bridge 之前，
    尝试穷举分配路径（Path B）。

    流程：
      1. 调用 exhaustive_allocation_path 获取 top-N 排序后的候选分配方案
         （不传入 mcs_balance_func，仅做排名筛选）。
      2. 逐个候选方案调用 balancer.balance_allocation 进行完整配平。
      3. 对每个配平结果做 _is_balanced 原子守恒校验，首个通过者进入
         置信度判定。
      4. 置信度 >= WORKFLOW_THRESHOLD：直接输出，取消 Bridge。
      5. 置信度 < WORKFLOW_THRESHOLD：存为 Bridge 候选方案（B 选项），
         保留 needs_bridge=True。
      6. ≥3 碎片场景施加 apply_confidence_penalty 惩罚因子。
    """
    if not enable_advanced_scoring:
        return decision
    if not decision.get("needs_bridge"):
        return decision

    reaction = row.get(reaction_col, "")
    if not reaction or ">>" not in str(reaction):
        return decision

    n_jobs = -1
    ablation = None
    if balancer is not None:
        n_jobs = balancer.n_jobs
        ablation = AblationConfig(
            enable_advanced_scoring=enable_advanced_scoring,
            enable_multi_fragment=balancer.enable_multi_fragment,
        )

    try:
        parts = str(reaction).split(">>", 1)
        reactant_smiles = [s for s in parts[0].split(".") if s]
        product_smiles = [s for s in parts[1].split(".") if s]

        if not reactant_smiles or not product_smiles:
            return decision

        # 第一步：获取排序后的候选分配方案（不调用配平函数）
        result = exhaustive_allocation_path(
            reactants=reactant_smiles,
            products=product_smiles,
            mcs_balance_func=None,
            mcs_search_func=ensemble_mcs,
            n_jobs=n_jobs,
            enable_exhaustive_allocation=enable_advanced_scoring,
        )

        candidates = result.get("candidates", [])
        mcs_cache = result.get("mcs_cache", {})

        # 第二步：逐个候选方案调用配平 + 原子守恒校验
        best_balanced_rxn = None
        best_confidence = None
        best_alloc_details = {}
        tried_count = 0

        if balancer is not None and candidates:
            for candidate in candidates:
                try:
                    # 预检：检查每个分配单元的碎片数
                    # 0-2: 标准路径（正常处理）
                    # 3-4: 复杂路径（配平后施加惩罚）
                    # 5+:  超出合并能力，跳过此候选
                    max_frags = 0
                    skip_candidate = False
                    for detail in candidate.get("details", []):
                        eu_smiles = detail.get("eval_unit_smiles", "")
                        if ">>" not in eu_smiles:
                            continue
                        lhs, rhs = eu_smiles.split(">>", 1)
                        n_lhs = len([s for s in lhs.split(".") if s])
                        n_rhs = len([s for s in rhs.split(".") if s])
                        n_frags = n_lhs + n_rhs
                        if n_frags > 4:
                            skip_candidate = True
                            logger.debug(
                                "Skipping candidate: allocation unit has "
                                "%d fragments (>4): %s", n_frags, eu_smiles,
                            )
                            break
                        max_frags = max(max_frags, n_frags)

                    if skip_candidate:
                        continue

                    alloc_result = balancer.balance_allocation(
                        reactants=reactant_smiles,
                        products=product_smiles,
                        allocation=candidate,
                        swapped=candidate.get("swapped", False),
                        cached_mcs_data=mcs_cache,
                    )
                    if not isinstance(alloc_result, dict):
                        continue
                    if not alloc_result.get("success"):
                        continue

                    balanced_rxn = alloc_result.get("balanced_reaction")
                    confidence = alloc_result.get("confidence")
                    tried_count += 1
                    # 提取 Path B 内部详情（供输出日志使用）
                    best_alloc_details = {
                        "sub_reaction_details": alloc_result.get("sub_reaction_details"),
                        "ghost_reactants": alloc_result.get("ghost_reactants"),
                        "missing_reactant_parts": alloc_result.get("missing_reactant_parts"),
                        "missing_product_parts": alloc_result.get("missing_product_parts"),
                    }

                    if balanced_rxn and _is_balanced(balanced_rxn):
                        # 3-4 碎片的分配单元：对 XGBoost 置信度施加惩罚
                        if max_frags >= 3 and confidence is not None:
                            confidence = apply_confidence_penalty(
                                confidence, max_frags, ablation
                            )

                        best_balanced_rxn = balanced_rxn
                        best_confidence = confidence
                        break

                except Exception as inner_exc:
                    logger.debug(
                        "Path B candidate %d balance_allocation failed: %s",
                        tried_count, inner_exc,
                    )
                    continue

        # 第三步：决策
        if best_balanced_rxn is not None:
            # 无论置信度高低，都存为 Bridge 候选方案
            decision["internal_candidate_2_reaction"] = best_balanced_rxn
            decision["internal_candidate_2_confidence"] = best_confidence
            decision["internal_candidate_2_source"] = "exhaustive_allocation"
            # Path B 内部详情（供输出日志和消融实验分析）
            decision["exhaustive_sub_reaction_details"] = best_alloc_details.get(
                "sub_reaction_details"
            )
            decision["exhaustive_ghost_reactants"] = best_alloc_details.get(
                "ghost_reactants"
            )
            decision["exhaustive_missing_reactant_parts"] = best_alloc_details.get(
                "missing_reactant_parts"
            )
            decision["exhaustive_missing_product_parts"] = best_alloc_details.get(
                "missing_product_parts"
            )
            decision["exhaustive_candidates_tried"] = tried_count
            decision["exhaustive_candidates_total"] = len(candidates)

            if (
                best_confidence is not None
                and best_confidence >= WORKFLOW_THRESHOLD
            ):
                # 置信度达标：直接输出，取消 Bridge
                decision["formal_output_reaction"] = best_balanced_rxn
                decision["workflow_confidence"] = best_confidence
                decision["workflow_source"] = "exhaustive_allocation"
                decision["needs_bridge"] = False
                decision["stage1_case"] = (
                    decision.get("stage1_case", "") + "_exhaustive_solved"
                )
            else:
                # 置信度未达标：保留为候选，继续 Bridge
                decision["stage1_case"] = (
                    decision.get("stage1_case", "")
                    + "_exhaustive_low_confidence"
                )

        elif candidates:
            # 有候选但配平/守恒校验全部失败
            decision["stage1_case"] = (
                decision.get("stage1_case", "") + "_exhaustive_partial"
            )
            decision["exhaustive_candidates_tried"] = tried_count
            decision["exhaustive_candidates_total"] = len(candidates)
            decision["exhaustive_best_rank_score"] = candidates[0].get(
                "rank_score"
            )

    except Exception as exc:
        logger.warning(
            "Path B exhaustive allocation failed: %s: %s",
            type(exc).__name__, exc,
        )

    return decision


def _try_template_matching(
    row: dict[str, Any],
    reaction_col: str,
    decision: dict[str, Any],
    enable_template_matching: bool = True,
) -> dict[str, Any]:
    """在 Path B（穷举分配）失败之后、LLM Bridge 之前，
    尝试模板匹配兜底。

    如果模板匹配成功（评分 >= 阈值），更新 decision 并取消 needs_bridge。
    如果未成功，将模板推测信息附加到 decision，作为 Bridge 上下文提示。
    """
    if not enable_template_matching:
        return decision
    if not decision.get("needs_bridge"):
        return decision

    reaction = row.get(reaction_col, "")
    if not reaction or ">>" not in str(reaction):
        return decision

    try:
        result = template_matching_for_reaction(
            reaction_smiles=str(reaction),
            rb_method=None,
            similarity_threshold=0.5,
        )

        if result.get("success") and result.get("balanced_reaction"):
            decision["formal_output_reaction"] = result["balanced_reaction"]
            decision["workflow_confidence"] = CONFIDENCE_BASELINES.get(
                "template_match", 0.8
            )
            decision["workflow_source"] = "template_matching"
            decision["needs_bridge"] = False
            decision["stage1_case"] = (
                decision.get("stage1_case", "") + "_template_solved"
            )
            decision["template_id"] = result.get("template_id")
            decision["template_name"] = result.get("template_name")
            decision["template_label"] = result.get("template_label")
            decision["template_score"] = result.get("score")
            # 同步更新源行的 solved 状态，使 _build_output_row
            # 能读到模板匹配的成功结果，而非核心管线的遗留状态
            row["solved"] = True
            row["solved_by"] = "template_matching"
        elif result.get("as_bridge_hint"):
            # 模板匹配未成功，将推测信息传递给 Bridge 作为上下文提示
            decision["template_context_id"] = result.get("template_id")
            decision["template_context_name"] = result.get("template_name")
            decision["template_context_label"] = result.get("template_label")
            decision["template_context_score"] = result.get("score")
            decision["template_inference_detail"] = result.get(
                "inference_detail"
            )
            decision["stage1_case"] = (
                decision.get("stage1_case", "") + "_template_hint"
            )

    except Exception as exc:
        logger.warning(
            "Template matching raised exception: %s: %s",
            type(exc).__name__, exc,
        )

    return decision


def _run_bridge_strategy_round(
    bridge_inputs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """
    使用新版 Bridge 策略选择器处理需要 Bridge 的反应。
    LLM 角色从"物种生成者"转变为"反应分析专家/策略选择者"。
    """
    selector = BridgeStrategySelector(
        id_col=args.id_col,
        reaction_col=args.reaction_col,
        select_strategy_fn=None,  # 先检查 LLM 是否可用
        max_workers=25,
    )

    # 如果有 LLM API key，则使用 LLM 策略选择
    import os
    api_key = os.environ.get(args.llm_api_key_env)
    if api_key:
        selector = BridgeStrategySelector.from_moonshot(
            id_col=args.id_col,
            reaction_col=args.reaction_col,
            api_key_env=args.llm_api_key_env,
            base_url=args.llm_base_url,
            model=args.llm_generate_model,
            max_workers=25,
            thinking_enabled=args.enable_llm_thinking,
        )

    working = copy.deepcopy(bridge_inputs)
    results = selector.apply(working)
    return results


def _stage2_decision(stage1_row: dict[str, Any], stage2_row: dict[str, Any], reaction_col: str) -> dict[str, Any]:
    # === 检查新版 Bridge 策略选择器结果 ===
    bridge_verified_rxn = stage2_row.get("bridge_verified_reaction")
    bridge_verified_conf = stage2_row.get("bridge_verified_confidence")
    if bridge_verified_rxn and _is_valid_reaction_smiles(bridge_verified_rxn) and _is_balanced(bridge_verified_rxn):
        return {
            "formal_output_reaction": bridge_verified_rxn,
            "workflow_confidence": bridge_verified_conf or CONFIDENCE_BASELINES.get("bridge_verified", 2.0),
            "workflow_source": "bridge",
            "internal_candidate_2_reaction": None,
            "internal_candidate_2_confidence": None,
            "internal_candidate_2_source": None,
            "fallback_input_source": None,
            "fallback_input_reaction": None,
            "needs_fallback": False,
            "stage2_case": "bridge_strategy_verified",
            "stage2_case_reason": "Bridge strategy selector verified the reaction.",
            "bridge_raw_output_reaction": bridge_verified_rxn,
            "bridge_raw_balanced": True,
            "bridge_raw_valid": True,
            "bridge_accepted_reaction": bridge_verified_rxn,
            "bridge_accepted_balanced": True,
            "bridge_accepted_solved_by": "bridge_strategy",
            "bridge_accepted_workflow_route": "strategy_selector",
            "bridge_accepted_confidence": bridge_verified_conf,
            "bridge_accepted_round_summary": {},
            "bridge_reaction_type": stage2_row.get("bridge_reaction_type"),
            "bridge_selected_strategy": stage2_row.get("bridge_selected_strategy"),
        }

    # === 旧版 Bridge 已废弃 ===
    # 未验证的 bridge 结果在主循环中通过 direct_fallback_tids 直接路由到 Fallback，
    # 不会到达此函数。此分支仅作为防御性兜底。
    return {
        "formal_output_reaction": None,
        "workflow_confidence": None,
        "workflow_source": "bridge",
        "internal_candidate_2_reaction": None,
        "internal_candidate_2_confidence": None,
        "internal_candidate_2_source": None,
        "fallback_input_source": "original",
        "fallback_input_reaction": stage2_row.get("original_reaction"),
        "needs_fallback": True,
        "stage2_case": "bridge_unresolved",
        "stage2_case_reason": (
            "Bridge strategy selector did not verify any reaction; "
            "routing directly to Fallback."
        ),
        "bridge_raw_output_reaction": None,
        "bridge_raw_balanced": False,
        "bridge_raw_valid": False,
        "bridge_accepted_reaction": None,
        "bridge_accepted_balanced": False,
        "bridge_accepted_solved_by": None,
        "bridge_accepted_workflow_route": None,
        "bridge_accepted_confidence": None,
        "bridge_accepted_round_summary": {},
        "bridge_reaction_type": stage2_row.get("bridge_reaction_type"),
        "bridge_selected_strategy": stage2_row.get("bridge_selected_strategy"),
    }


def _build_fallback_runner(args: argparse.Namespace, cfg: RebalanceConfig) -> tuple[LLMFallbackPostprocessor, Balancer]:
    fallback = LLMFallbackPostprocessor.from_moonshot(
        id_col=cfg.id_col,
        reaction_col=cfg.reaction_col,
        api_key_env=args.llm_api_key_env,
        base_url=args.llm_base_url,
        model=args.llm_generate_model,
        max_workers=25,
        thinking_enabled=args.enable_llm_thinking,
    )
    balancer = _build_balancer(cfg)
    return fallback, balancer


def _run_fallback_round(fallback_inputs: list[dict[str, Any]], args: argparse.Namespace, cfg: RebalanceConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行 Fallback LLM 配平。

    返回:
        (working, fallback_stats) 元组。
        fallback_stats 包含 llm_fallback_retry_cnt（触发 LLM 调用的反应数）
        和 llm_fallback_recovered_cnt（通过重试恢复成功的反应数）。
    """
    fallback, balancer = _build_fallback_runner(args, cfg)
    working = copy.deepcopy(fallback_inputs)
    valid_rows: list[dict[str, Any]] = []
    # C-FB5 修复：使用持久化 stats 字典，不再传入一次性 {}
    fallback_stats: dict[str, Any] = {}
    for row in working:
        chosen_input = row.get("fallback_input_reaction") or row.get(cfg.reaction_col) or row.get("original_reaction")
        row[cfg.reaction_col] = chosen_input
        row["input_reaction"] = chosen_input
        row["workflow_confidence"] = None
        row["bridge_candidate_reaction"] = None
        row["bridge_best_reaction"] = None
        if _is_valid_reaction_smiles(chosen_input):
            valid_rows.append(row)
        else:
            row[cfg.reaction_col] = None
            row.setdefault("llm_fallback_postprocess", {})
            row["llm_fallback_postprocess"].update(
                {
                    "triggered": False,
                    "final_status": "invalid_fallback_input_filtered",
                    "failure_reason": "Fallback input reaction failed pre-pipeline validity filter.",
                    "chosen_input_source": row.get("fallback_input_source"),
                    "chosen_input_value": chosen_input,
                }
            )
    if valid_rows:
        fallback.apply(valid_rows, balancer, stats=fallback_stats)
    return working, fallback_stats


def _build_output_row(
    source_row: dict[str, Any],
    reaction_col: str,
    formal_output_reaction: Any,
    workflow_confidence: Any,
    workflow_source: str,
    success: bool,
    stage1: dict[str, Any],
    stage2: dict[str, Any] | None,
    fallback_input_source: Any,
    fallback_input_reaction: Any,
    internal_candidate_2_reaction: Any,
    internal_candidate_2_confidence: Any,
) -> dict[str, Any]:
    row = dict(source_row)

    # ── 标识列（显式写入，防止源行结构变化丢失） ──
    row["original_reaction"] = source_row.get("original_reaction")
    row["original_row_index"] = source_row.get("original_row_index")

    # ── 核心管线状态列（显式写入） ──
    row["pipeline_input_reaction"] = source_row.get(reaction_col)
    row["processable"] = source_row.get("processable")
    row["preprocess_status"] = source_row.get("preprocess_status")
    row["solved"] = source_row.get("solved")
    row["solved_by"] = source_row.get("solved_by")
    row["confidence"] = source_row.get("confidence")
    row["issue"] = source_row.get("issue")
    row["rules"] = source_row.get("rules")
    row["carbon_balance_check"] = source_row.get("carbon_balance_check")
    row["impute_direction"] = source_row.get("impute_direction")

    # ── MCS 投票策略（从嵌套 mcs 字典中提取为顶层字段） ──
    mcs_data = source_row.get("mcs")
    row["mcs_vote_method"] = mcs_data.get("vote_method") if isinstance(mcs_data, dict) else None

    # ── 流水线决策结果（已有字段，保持不变） ──
    row[reaction_col] = formal_output_reaction
    row["formal_output_reaction"] = formal_output_reaction
    row["workflow_confidence"] = workflow_confidence
    row["workflow_source"] = workflow_source
    row["success"] = success
    row["output_status"] = "True" if success else "False"
    row["final_result_source"] = determine_result_source(
        workflow_source, stage1.get("stage1_case"), success
    )

    # ── Path A 候选 ──
    row["internal_candidate_1_reaction"] = stage1.get("internal_candidate_1_reaction")
    row["internal_candidate_1_confidence"] = stage1.get("internal_candidate_1_confidence")
    row["internal_candidate_1_source"] = stage1.get("internal_candidate_1_source")
    row["stage1_case"] = stage1.get("stage1_case")

    # ── Path B 候选 ──
    row["internal_candidate_2_reaction"] = internal_candidate_2_reaction
    row["internal_candidate_2_confidence"] = internal_candidate_2_confidence
    # C2: 从 stage2 优先读取 candidate_2_source，回退到 stage1（Path B 直接输出时）
    c2_source = None
    if stage2 is not None:
        c2_source = stage2.get("internal_candidate_2_source")
    if c2_source is None:
        c2_source = stage1.get("internal_candidate_2_source")
    row["internal_candidate_2_source"] = c2_source

    # ── Path B 内部详情（新增：穷举分配诊断信息） ──
    row["exhaustive_sub_reaction_details"] = stage1.get("exhaustive_sub_reaction_details")
    row["exhaustive_ghost_reactants"] = stage1.get("exhaustive_ghost_reactants")
    row["exhaustive_missing_reactant_parts"] = stage1.get("exhaustive_missing_reactant_parts")
    row["exhaustive_missing_product_parts"] = stage1.get("exhaustive_missing_product_parts")
    row["exhaustive_candidates_tried"] = stage1.get("exhaustive_candidates_tried")
    row["exhaustive_candidates_total"] = stage1.get("exhaustive_candidates_total")
    row["exhaustive_best_rank_score"] = stage1.get("exhaustive_best_rank_score")

    # ── Fallback 输入 ──
    row["fallback_input_source"] = fallback_input_source
    row["fallback_input_reaction"] = fallback_input_reaction

    # ── 模板匹配字段 ──
    row["template_id"] = stage1.get("template_id")
    row["template_name"] = stage1.get("template_name")
    row["template_label"] = stage1.get("template_label")
    row["template_score"] = stage1.get("template_score")

    # ── Bridge / Stage 2 字段 ──
    # 先写 None 默认值，确保所有行都有统一的列结构
    row["stage2_case"] = None
    row["stage2_case_reason"] = None
    row["bridge_raw_output_reaction"] = None
    row["bridge_raw_balanced"] = None
    row["bridge_raw_valid"] = None
    row["bridge_reaction_type"] = None
    row["bridge_selected_strategy"] = None
    row["bridge_accepted_reaction"] = None
    row["bridge_accepted_balanced"] = None
    row["bridge_accepted_solved_by"] = None
    row["bridge_accepted_workflow_route"] = None
    row["bridge_accepted_confidence"] = None
    row["bridge_accepted_round_summary"] = None
    # 如果 stage2 存在，用实际值覆盖
    if stage2 is not None:
        row["stage2_case"] = stage2.get("stage2_case")
        row["stage2_case_reason"] = stage2.get("stage2_case_reason")
        row["bridge_raw_output_reaction"] = stage2.get("bridge_raw_output_reaction")
        row["bridge_raw_balanced"] = stage2.get("bridge_raw_balanced")
        row["bridge_raw_valid"] = stage2.get("bridge_raw_valid")
        row["bridge_reaction_type"] = stage2.get("bridge_reaction_type")
        row["bridge_selected_strategy"] = stage2.get("bridge_selected_strategy")
        row["bridge_accepted_reaction"] = stage2.get("bridge_accepted_reaction")
        row["bridge_accepted_balanced"] = stage2.get("bridge_accepted_balanced")
        row["bridge_accepted_solved_by"] = stage2.get("bridge_accepted_solved_by")
        row["bridge_accepted_workflow_route"] = stage2.get("bridge_accepted_workflow_route")
        row["bridge_accepted_confidence"] = stage2.get("bridge_accepted_confidence")
        row["bridge_accepted_round_summary"] = stage2.get("bridge_accepted_round_summary")

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged SynRBL workflow orchestrator.")
    parser.add_argument("input_file", help="Path to input CSV/TSV/Excel/JSON file.")
    parser.add_argument("--reaction-col", default="reactions", help="Reaction column name.")
    parser.add_argument("--id-col", default="R-id", help="ID column name.")
    parser.add_argument("--expected-col", default="expected_reaction", help="Expected reaction column for accuracy comparison.")
    parser.add_argument("--llm-api-key-env", default="MOONSHOT_API_KEY", help="Environment variable name for LLM API key.")
    parser.add_argument("--llm-base-url", default="https://api.moonshot.cn/v1/chat/completions", help="LLM chat completions endpoint.")
    parser.add_argument("--llm-generate-model", default="kimi-k2.5", help="LLM generation model name.")
    parser.add_argument("--enable-llm-thinking", action="store_true", help="Enable Kimi thinking mode for all LLM calls.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to input file directory.")
    parser.add_argument("--synrbl-confidence-threshold", type=float, default=0.8, help="Native SynRBL confidence threshold.")
    parser.add_argument("--sep", default=None, help="Optional separator for CSV/TSV/TXT input.")
    parser.add_argument("--head", type=int, default=None, help="只处理前 N 条数据（用于测试）")
    parser.add_argument("--start-row", type=int, default=None, help="从第几行数据开始处理（不含题头，1-based）")
    parser.add_argument("--end-row", type=int, default=None, help="处理到第几行数据结束（不含题头，1-based，包含该行）")

    # 消融实验开关
    parser.add_argument("--disable-advanced-scoring", action="store_true",
                        help="禁用高级评分统一开关（递进式投票 5A-2 + 穷举分配路径 B 5B）")
    parser.add_argument("--disable-multi-fragment", action="store_true",
                        help="禁用多碎片合并（改进 6c）")
    parser.add_argument("--disable-template-matching", action="store_true",
                        help="禁用模板匹配兜底（改进 11）")

    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    base_output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_path.parent
    df = _load_table(input_path, sep=args.sep)
    original_row_count = len(df)

    if args.head is not None and (args.start_row is not None or args.end_row is not None):
        raise ValueError("--head 不能与 --start-row / --end-row 同时使用。请选择一种切片方式。")

    if args.head is not None:
        df = df.head(args.head).copy()
        run_label = f"head-{args.head}"
    else:
        df = _slice_dataframe(df, args.start_row, args.end_row)
        run_label = _build_run_label(args.start_row, args.end_row, len(df))

    output_dir = base_output_dir / _sanitize_filename_part(run_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    reaction_col = _resolve_column(df, args.reaction_col, ["reaction", "reactions", "rxn", "rsmi"])
    id_col = _resolve_column(df, args.id_col, ["R-id", "id", "ID", "Id"])
    expected_col = (
        _resolve_column(df, args.expected_col, ["expected_reaction"])
        if args.expected_col in df.columns or args.expected_col.lower() in {str(col).lower() for col in df.columns} or "expected_reaction" in df.columns
        else args.expected_col
    )

    cfg = _build_base_config(args, reaction_col, id_col)
    prepared_records = _prepare_records(df, reaction_col, id_col)

    # 消融实验配置
    ablation = AblationConfig(
        enable_advanced_scoring=not args.disable_advanced_scoring,
        enable_multi_fragment=not args.disable_multi_fragment,
        enable_template_matching=not args.disable_template_matching,
    )

    # 创建共享 Balancer 实例：供 Path A（_run_first_round）和
    # Path B（_try_exhaustive_allocation）共用
    shared_balancer = _build_balancer(cfg)

    stage1_rows = _run_first_round(prepared_records, cfg, balancer=shared_balancer)
    original_stage_outputs: list[dict[str, Any]] = []
    bridge_inputs: list[dict[str, Any]] = []
    bridge_stage_outputs: list[dict[str, Any]] = []

    for row in stage1_rows:
        decision = _stage1_decision(row, reaction_col)
        # Path B：在 Bridge 之前尝试穷举分配
        decision = _try_exhaustive_allocation(
            row, reaction_col, decision,
            enable_advanced_scoring=ablation.enable_advanced_scoring,
            balancer=shared_balancer,
        )
        # 模板匹配兜底：在 Path B 之后、LLM Bridge 之前
        decision = _try_template_matching(
            row, reaction_col, decision,
            enable_template_matching=ablation.enable_template_matching,
        )
        stage1_output = _build_output_row(
            row,
            reaction_col,
            decision.get("formal_output_reaction"),
            decision.get("workflow_confidence"),
            decision.get("workflow_source"),
            bool(decision.get("formal_output_reaction")),
            decision,
            None,
            None,
            None,
            decision.get("internal_candidate_2_reaction"),
            decision.get("internal_candidate_2_confidence"),
        )
        original_stage_outputs.append(stage1_output)
        if decision.get("needs_bridge"):
            bridge_input = copy.deepcopy(row)
            bridge_input["stage1_case"] = decision.get("stage1_case")
            bridge_input["internal_candidate_1_reaction"] = decision.get("internal_candidate_1_reaction")
            bridge_input["internal_candidate_1_confidence"] = decision.get("internal_candidate_1_confidence")
            # 传递 Path B（穷举分配）候选方案
            bridge_input["internal_candidate_2_reaction"] = decision.get("internal_candidate_2_reaction")
            bridge_input["internal_candidate_2_confidence"] = decision.get("internal_candidate_2_confidence")
            bridge_input["internal_candidate_2_source"] = decision.get("internal_candidate_2_source")
            # 传递模板匹配上下文信息（供 Bridge 提示词注入）
            bridge_input["template_context_id"] = decision.get("template_context_id")
            bridge_input["template_context_name"] = decision.get("template_context_name")
            bridge_input["template_context_label"] = decision.get("template_context_label")
            bridge_input["template_context_score"] = decision.get("template_context_score")
            bridge_input["template_inference_detail"] = decision.get(
                "template_inference_detail"
            )
            bridge_inputs.append(bridge_input)

    # === Bridge 阶段：策略选择器 → 未解决则直接进入 Fallback ===
    bridge_strategy_results: dict[str, dict[str, Any]] = {}
    direct_fallback_tids: set[str] = set()

    if bridge_inputs:
        strategy_rows = _run_bridge_strategy_round(bridge_inputs, args)
        for sr in strategy_rows:
            tid = sr.get(INTERNAL_TRACKING_ID)
            verified_rxn = sr.get("bridge_verified_reaction")
            if verified_rxn and _is_valid_reaction_smiles(verified_rxn) and _is_balanced(verified_rxn):
                # 策略选择器成功解决了这条反应
                bridge_strategy_results[tid] = sr
            else:
                # 策略选择器未验证 → 直接进入 Fallback（旧版 Bridge 已废弃）
                direct_fallback_tids.add(tid)
                bridge_strategy_results[tid] = sr

    bridge_by_tracking: dict[str, dict[str, Any]] = dict(bridge_strategy_results)

    fallback_inputs: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []

    for stage1_output in original_stage_outputs:
        tracking_id = stage1_output[INTERNAL_TRACKING_ID]
        if stage1_output.get("formal_output_reaction") is not None:
            final_rows.append(stage1_output)
            continue

        bridge_row = bridge_by_tracking.get(tracking_id)
        if bridge_row is None:
            final_rows.append(stage1_output)
            continue

        # Bridge 未解决的反应：直接进入 Fallback（旧版 Bridge 已废弃）
        if tracking_id in direct_fallback_tids:
            original_rxn = bridge_row.get("original_reaction") or stage1_output.get(reaction_col)
            c1_rxn = stage1_output.get("internal_candidate_1_reaction")
            c1_valid = c1_rxn and _is_valid_reaction_smiles(c1_rxn)
            fallback_rxn = c1_rxn if c1_valid else original_rxn
            fallback_source = "internal_candidate_1" if c1_valid else "original"
            if fallback_rxn is not None and not _is_valid_reaction_smiles(fallback_rxn):
                fallback_rxn = None
                fallback_source = None
            strategy_sel = bridge_row.get("bridge_selected_strategy") or "unknown"
            d_stage2_decision = {
                "formal_output_reaction": None,
                "workflow_confidence": None,
                "workflow_source": "bridge",
                "needs_fallback": True,
                "stage2_case": "bridge_unresolved_direct_fallback",
                "stage2_case_reason": (
                    f"Bridge strategy selector did not verify any reaction "
                    f"(strategy={strategy_sel}); routing directly to Fallback."
                ),
                "internal_candidate_2_reaction": None,
                "internal_candidate_2_confidence": None,
                "fallback_input_source": fallback_source,
                "fallback_input_reaction": fallback_rxn,
                "bridge_selected_strategy": strategy_sel,
                "bridge_reaction_type": bridge_row.get("bridge_reaction_type"),
            }
            fallback_inputs.append(
                {
                    INTERNAL_TRACKING_ID: tracking_id,
                    id_col: bridge_row.get(id_col, ""),
                    reaction_col: fallback_rxn,
                    "original_reaction": original_rxn,
                    "issue": bridge_row.get("issue", ""),
                    "fallback_input_source": fallback_source,
                    "fallback_input_reaction": fallback_rxn,
                    "stage1_output": stage1_output,
                    "stage2_decision": d_stage2_decision,
                    "bridge_row": bridge_row,
                    "bridge_reaction_type": bridge_row.get("bridge_reaction_type"),
                    "bridge_selected_strategy": strategy_sel,
                    "fallback_case": (
                        "C_selected" if strategy_sel == "C" else "bridge_unresolved"
                    ),
                }
            )
            continue

        stage2_decision = _stage2_decision(stage1_output, bridge_row, reaction_col)
        bridge_stage_output = _build_output_row(
            bridge_row,
            reaction_col,
            stage2_decision.get("formal_output_reaction"),
            stage2_decision.get("workflow_confidence"),
            "bridge",
            bool(stage2_decision.get("formal_output_reaction")),
            stage1_output,
            stage2_decision,
            stage2_decision.get("fallback_input_source"),
            stage2_decision.get("fallback_input_reaction"),
            stage2_decision.get("internal_candidate_2_reaction"),
            stage2_decision.get("internal_candidate_2_confidence"),
        )
        bridge_stage_outputs.append(bridge_stage_output)

        if stage2_decision.get("formal_output_reaction") is not None:
            final_rows.append(bridge_stage_output)
            continue

        fallback_inputs.append(
            {
                INTERNAL_TRACKING_ID: tracking_id,
                id_col: bridge_row[id_col],
                reaction_col: stage2_decision.get("fallback_input_reaction"),
                "original_reaction": bridge_row.get("original_reaction"),
                "issue": bridge_row.get("issue", ""),
                "fallback_input_source": stage2_decision.get("fallback_input_source"),
                "fallback_input_reaction": stage2_decision.get("fallback_input_reaction"),
                "stage1_output": stage1_output,
                "stage2_decision": stage2_decision,
                "bridge_row": bridge_row,
                # 改进10：Bridge 反应类型上下文（用于 Fallback LLM 提示词增强）
                "bridge_reaction_type": (
                    stage2_decision.get("bridge_reaction_type")
                    or bridge_row.get("bridge_reaction_type")
                ),
                "bridge_selected_strategy": (
                    stage2_decision.get("bridge_selected_strategy")
                    or bridge_row.get("bridge_selected_strategy")
                ),
                # 区分两种 Fallback 情况（Section六 6.7）
                "fallback_case": (
                    "C_selected"
                    if (
                        stage2_decision.get("bridge_selected_strategy") == "C"
                        or bridge_row.get("bridge_selected_strategy") == "C"
                    )
                    else "post_processing_failed"
                ),
            }
        )

    fallback_stage_rows, fallback_stats = (
        _run_fallback_round(fallback_inputs, args, cfg)
        if fallback_inputs
        else ([], {})
    )
    fallback_by_tracking = {row[INTERNAL_TRACKING_ID]: row for row in fallback_stage_rows}

    for item in fallback_inputs:
        tracking_id = item[INTERNAL_TRACKING_ID]
        fallback_row = fallback_by_tracking.get(tracking_id)
        fallback_reaction = fallback_row.get(reaction_col) if fallback_row else None
        fallback_success = _is_valid_reaction_smiles(fallback_reaction) and _is_balanced(fallback_reaction)
        fallback_source_row = copy.deepcopy(item["bridge_row"])
        if fallback_row is not None:
            fallback_source_row.update(fallback_row)

        # ── Fallback 回调 ──
        # 当 LLM fallback 失败时，依次尝试确定性候选 A（Path A）和 B（Path B），
        # 采用第一个合法且原子守恒的候选。这是默认选择而非质量比较——
        # Bridge 已通过 strategy C 拒绝了这些候选，此处仅作为最终兜底。
        # 纯增强：成功的 fallback 路径不受影响。
        fb_callback_used = None
        fb_callback_source = None
        if not fallback_success:
            c1_rxn = item["stage1_output"].get("internal_candidate_1_reaction")
            if c1_rxn and _is_valid_reaction_smiles(c1_rxn) and _is_balanced(c1_rxn):
                fallback_reaction = c1_rxn
                fallback_success = True
                fb_callback_used = True
                fb_callback_source = "internal_candidate_1"
            else:
                c2_rxn = item["stage2_decision"].get("internal_candidate_2_reaction")
                if c2_rxn and _is_valid_reaction_smiles(c2_rxn) and _is_balanced(c2_rxn):
                    fallback_reaction = c2_rxn
                    fallback_success = True
                    fb_callback_used = True
                    fb_callback_source = "internal_candidate_2"

        final_rows.append(
            _build_output_row(
                fallback_source_row,
                reaction_col,
                fallback_reaction if fallback_success else None,
                CONFIDENCE_BASELINES.get("fallback_accepted", 3.0) if fallback_success else None,
                "fallback",
                fallback_success,
                item["stage1_output"],
                item["stage2_decision"],
                item.get("fallback_input_source"),
                item.get("fallback_input_reaction"),
                item["stage2_decision"].get("internal_candidate_2_reaction"),
                item["stage2_decision"].get("internal_candidate_2_confidence"),
            )
        )
        # 附加回调元数据（不影响既有列；供审计追踪）
        if fb_callback_used:
            final_rows[-1]["fallback_callback"] = True
            final_rows[-1]["fallback_callback_source"] = fb_callback_source

    sort_key = lambda x: int(x.get("original_row_index", 0))
    original_stage_outputs.sort(key=sort_key)
    bridge_stage_outputs.sort(key=sort_key)
    final_rows.sort(key=sort_key)
    fallback_stage_rows.sort(key=sort_key)

    # ═══ 输出文件（3 个结果文件 + 评估报告） ═══

    # 文件 1：流程状态文件 —— 所有反应的完整流水线经过
    _write_json_and_csv(
        final_rows,
        output_dir / "pipeline_status.json",
        output_dir / "pipeline_status.csv",
    )

    # 文件 2：配平成功文件 —— 仅 success=True，标注原始反应、来源、ID
    success_rows = [row for row in final_rows if row.get("success")]
    success_output = []
    for row in success_rows:
        success_output.append({
            "id": row.get(id_col),
            "original_row_index": row.get("original_row_index"),
            "original_reaction": row.get("original_reaction"),
            "pipeline_input_reaction": row.get("pipeline_input_reaction"),
            "balanced_reaction": row.get("formal_output_reaction"),
            "final_result_source": row.get("final_result_source"),
            "workflow_source": row.get("workflow_source"),
            "workflow_confidence": row.get("workflow_confidence"),
            "stage1_case": row.get("stage1_case"),
            "solved_by": row.get("solved_by"),
            "carbon_balance_check": row.get("carbon_balance_check"),
            "impute_direction": row.get("impute_direction"),
            "mcs_vote_method": row.get("mcs_vote_method"),
            "template_id": row.get("template_id"),
            "template_name": row.get("template_name"),
            "bridge_selected_strategy": row.get("bridge_selected_strategy"),
        })
    _write_json_and_csv(
        success_output,
        output_dir / "balanced_reactions.json",
        output_dir / "balanced_reactions.csv",
    )

    # 文件 3：配平失败文件 —— 仅 success=False，标注原始反应、来源、ID、诊断信息
    failed_rows = [row for row in final_rows if not row.get("success")]
    failed_output = []
    for row in failed_rows:
        failed_output.append({
            "id": row.get(id_col),
            "original_row_index": row.get("original_row_index"),
            "original_reaction": row.get("original_reaction"),
            "pipeline_input_reaction": row.get("pipeline_input_reaction"),
            "stage1_case": row.get("stage1_case"),
            "stage2_case": row.get("stage2_case"),
            "stage2_case_reason": row.get("stage2_case_reason"),
            "final_result_source": row.get("final_result_source"),
            "workflow_source": row.get("workflow_source"),
            "workflow_confidence": row.get("workflow_confidence"),
            "issue": row.get("issue"),
            "processable": row.get("processable"),
            "preprocess_status": row.get("preprocess_status"),
            "bridge_selected_strategy": row.get("bridge_selected_strategy"),
            "bridge_reaction_type": row.get("bridge_reaction_type"),
            "fallback_input_source": row.get("fallback_input_source"),
        })
    _write_json_and_csv(
        failed_output,
        output_dir / "failed_reactions.json",
        output_dir / "failed_reactions.csv",
    )

    # 评估报告（供论文使用，可注释掉）
    write_workflow_statistics(
        output_dir=output_dir,
        validation_df=df,
        original_stage_results=original_stage_outputs,
        with_llm_results=final_rows,
        reaction_col=reaction_col,
        target_col=expected_col,
    )
    write_accuracy_report(
        output_dir=output_dir,
        no_llm_results=original_stage_outputs,
        with_llm_results=final_rows,
        reaction_col=reaction_col,
        target_col=expected_col,
        group_key="workflow_source",
    )

    print(f"Processed {len(df)} / {original_row_count} rows. Output folder: {output_dir}")
    print(f"Finished. {len(success_rows)} balanced, {len(failed_rows)} failed.")
    # C-FB5 修复：输出 Fallback 统计统计（之前写入一次性字典被丢弃）
    if fallback_stats:
        fb_triggered = fallback_stats.get("llm_fallback_retry_cnt", 0)
        fb_rec = fallback_stats.get("llm_fallback_recovered_cnt", 0)
        print(
            f"Fallback LLM: {fb_triggered} triggered, {fb_rec} recovered via retry."
        )


if __name__ == "__main__":
    main()
