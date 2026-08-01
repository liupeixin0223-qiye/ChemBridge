import copy
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from synrbl.llm.client import LLMResponseParseError
from synrbl.llm.fallback_client import FallbackGenerateClient
from synrbl.llm_species_bridge import LLMSpeciesBridge

logger = logging.getLogger("synrbl")


class LLMFallbackPostprocessor:
    def __init__(
        self,
        id_col: str,
        reaction_col: str,
        solved_col: str = "solved",
        issue_col: str = "issue",
        confidence_col: str = "confidence",
        generate_candidate_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        log_col: str = "llm_fallback_postprocess",
        max_workers: int = 20,
    ):
        self.id_col = id_col
        self.reaction_col = reaction_col
        self.solved_col = solved_col
        self.issue_col = issue_col
        self.confidence_col = confidence_col
        self.generate_candidate_fn = generate_candidate_fn
        self.log_col = log_col
        self.max_workers = max(1, int(max_workers))

    @classmethod
    def from_moonshot(
        cls,
        id_col: str,
        reaction_col: str,
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: str = "https://api.moonshot.cn/v1/chat/completions",
        model: str = "kimi-k2.5",
        max_workers: int = 20,
        thinking_enabled: bool = False,
    ) -> "LLMFallbackPostprocessor":
        client = FallbackGenerateClient(
            api_key_env=api_key_env,
            base_url=base_url,
            score_model=model,
            generate_model=model,
            thinking_enabled=thinking_enabled,
        )
        return cls(
            id_col=id_col,
            reaction_col=reaction_col,
            generate_candidate_fn=client.generate_candidate,
            max_workers=max_workers,
        )

    def apply(self, reactions: List[Dict[str, Any]], balancer, stats=None):
        if self.generate_candidate_fn is None:
            return reactions

        triggered = 0
        recovered = 0
        triggered_reactions: list[Dict[str, Any]] = []
        request_stats = {
            "max_workers": self.max_workers,
            "requested_count": 0,
            "completed_count": 0,
            "parse_error_count": 0,
            "request_error_count": 0,
        }
        for reaction in reactions:
            explicit_fallback_input = reaction.get("fallback_input_reaction")
            # FB-1 修复：增加 NaN 检测（pandas 将 CSV 空值读为 float('nan')）
            if (
                explicit_fallback_input in {None, ""}
                or (
                    isinstance(explicit_fallback_input, float)
                    and math.isnan(explicit_fallback_input)
                )
            ):
                reaction.setdefault(self.log_col, {})
                reaction[self.log_col].update(
                    {
                        "triggered": False,
                        "final_status": "missing_explicit_fallback_input",
                        "failure_reason": "Fallback execution requires orchestrator-provided fallback_input_reaction.",
                        "fallback_input_source": reaction.get("fallback_input_source"),
                        "explicit_fallback_input_reaction": reaction.get("fallback_input_reaction"),
                    }
                )
                continue
            current_best_reaction = explicit_fallback_input
            current_balance = LLMSpeciesBridge.analyze_reaction_balance(str(current_best_reaction))
            if not self._should_retry_reaction(reaction, current_balance):
                continue
            triggered += 1

            imbalance = current_balance
            reaction[self.log_col] = {
                "triggered": True,
                "entry_id": reaction.get(self.id_col),
                "fallback_input_source": reaction.get("fallback_input_source"),
                "explicit_fallback_input_reaction": reaction.get("fallback_input_reaction"),
                "input_reaction": current_best_reaction,
                "input_balance_analysis": imbalance,
                "raw_response": None,
                "parse_error": None,
                "generated_candidate": None,
                "post_pipeline_reaction": None,
                "post_pipeline_solved": None,
                "post_pipeline_confidence": None,
                "post_pipeline_issue": None,
                "post_pipeline_balance_analysis": None,
                "final_status": "pending",
                "failure_reason": None,
                "request_stats": {
                    "phase": "fallback_postprocess",
                    "max_workers": self.max_workers,
                    "completed": False,
                },
            }

            # 构建 Fallback LLM payload（含反应类型上下文）
            bridge_reaction_type = reaction.get("bridge_reaction_type", "")
            fallback_case = reaction.get("fallback_case", "post_processing_failed")

            task_note_parts = [
                "Fallback task: produce the most plausible fully balanced reaction",
                "from the current reaction and exact atom imbalance.",
                "Prefer a complete best guess over a conservative empty result.",
            ]
            if bridge_reaction_type:
                task_note_parts.append(
                    f"Reaction type (identified by Bridge LLM): {bridge_reaction_type}."
                )
            if fallback_case == "C_selected":
                task_note_parts.append(
                    "Note: Bridge LLM judged all deterministic candidates as unreasonable. "
                    "You are generating from the original reaction."
                )
            elif fallback_case == "post_processing_failed":
                task_note_parts.append(
                    "Note: A deterministic candidate was close but failed species cancellation. "
                    "Try to fix the remaining imbalance."
                )

            payload = {
                "reaction_id": reaction.get(self.id_col),
                "input_reaction": current_best_reaction,
                "current_reaction": current_best_reaction,
                "calculated_imbalance": imbalance.get("imbalance_text", ""),
                "balance_analysis": imbalance,
                "issue": reaction.get(self.issue_col, ""),
                "bridge_reaction_type": bridge_reaction_type,
                "fallback_case": fallback_case,
                "task_note": " ".join(task_note_parts),
            }
            reaction[self.log_col]["payload"] = copy.deepcopy(payload)
            triggered_reactions.append(reaction)

        request_stats["requested_count"] = len(triggered_reactions)
        generated_results: dict[str, dict[str, Any]] = {}
        if triggered_reactions:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(self.generate_candidate_fn, reaction[self.log_col]["payload"]): reaction
                    for reaction in triggered_reactions
                }
                for future in as_completed(future_map):
                    reaction = future_map[future]
                    tracking_key = str(reaction.get(self.id_col))
                    try:
                        generated_results[tracking_key] = {"status": "ok", "payload": future.result()}
                        request_stats["completed_count"] += 1
                    except LLMResponseParseError as exc:
                        generated_results[tracking_key] = {
                            "status": "parse_error",
                            "raw_response": exc.raw_response,
                            "error": str(exc),
                        }
                        request_stats["parse_error_count"] += 1
                    except Exception as exc:
                        generated_results[tracking_key] = {"status": "request_error", "error": str(exc)}
                        request_stats["request_error_count"] += 1

        for reaction in triggered_reactions:
            reaction[self.log_col]["request_pool_stats"] = copy.deepcopy(request_stats)
            reaction[self.log_col]["request_stats"]["completed"] = True
            generated_result = generated_results.get(str(reaction.get(self.id_col)), {"status": "request_error", "error": "Missing generated result."})

            if generated_result["status"] == "parse_error":
                reaction[self.log_col]["raw_response"] = generated_result.get("raw_response")
                reaction[self.log_col]["parse_error"] = generated_result.get("error")
                reaction[self.log_col]["final_status"] = "parse_error"
                reaction[self.log_col]["failure_reason"] = generated_result.get("error")
                continue
            if generated_result["status"] != "ok":
                reaction[self.log_col]["parse_error"] = generated_result.get("error")
                reaction[self.log_col]["final_status"] = "request_exception"
                reaction[self.log_col]["failure_reason"] = generated_result.get("error")
                continue

            generated = generated_result.get("payload") or {}
            reaction[self.log_col]["raw_response"] = generated.get("_raw_response") if isinstance(generated, dict) else None
            generated_clean = {k: v for k, v in generated.items() if k != "_raw_response"} if isinstance(generated, dict) else {}
            reaction[self.log_col]["generated_candidate"] = generated_clean

            predicted_reaction = str(generated_clean.get("predicted_reaction_smiles") or "").strip()
            if not predicted_reaction:
                reaction[self.log_col]["final_status"] = "empty_prediction"
                reaction[self.log_col]["failure_reason"] = "LLM returned empty predicted reaction."
                continue
            basic_prediction_check = LLMSpeciesBridge.analyze_reaction_balance(predicted_reaction)
            is_valid_prediction = True
            if ">>" not in predicted_reaction or basic_prediction_check.get("error") == "Invalid reaction format":
                is_valid_prediction = False
            else:
                try:
                    reactants, products = predicted_reaction.split(">>", 1)
                    if not reactants.strip() or not products.strip():
                        is_valid_prediction = False
                    for side in (reactants, products):
                        for token in side.split("."):
                            token = token.strip()
                            if not token or not LLMSpeciesBridge._is_valid_smiles(token):
                                is_valid_prediction = False
                                break
                        if not is_valid_prediction:
                            break
                except Exception:
                    is_valid_prediction = False
            if not is_valid_prediction:
                reaction[self.log_col]["post_pipeline_reaction"] = predicted_reaction
                reaction[self.log_col]["post_pipeline_balance_analysis"] = basic_prediction_check
                reaction[self.log_col]["final_status"] = "invalid_prediction_filtered"
                reaction[self.log_col]["failure_reason"] = "Generated reaction failed molecule-level pre-pipeline validation."
                continue

            candidate = copy.deepcopy(reaction)
            candidate["input_reaction"] = predicted_reaction
            candidate[self.reaction_col] = predicted_reaction
            candidate[self.solved_col] = False
            candidate[self.issue_col] = ""
            candidate["solved_by"] = ""
            candidate.setdefault("carbon_balance_check", "balanced")
            candidate.setdefault("is_carbon_balance", True)
            candidate.setdefault("reactants", [])
            candidate.setdefault("products", [])
            candidate.setdefault("mcs", {})
            candidate.setdefault("rules", [])
            candidate.setdefault("confidence", 0.0)
            if self.confidence_col in candidate:
                candidate[self.confidence_col] = 0.0
            candidate["fallback_generated_reaction"] = predicted_reaction

            try:
                balancer.run_post_generation_pipeline([candidate], stats=stats)
            except Exception as exc:
                logger.exception("Fallback post-generation pipeline failed.")
                reaction[self.log_col]["final_status"] = "post_pipeline_exception"
                reaction[self.log_col]["failure_reason"] = str(exc)
                continue

            post_balance = LLMSpeciesBridge.analyze_reaction_balance(
                candidate.get(self.reaction_col, "") or candidate.get("input_reaction", "")
            )
            reaction[self.log_col]["post_pipeline_reaction"] = candidate.get(self.reaction_col)
            reaction[self.log_col]["post_pipeline_solved"] = candidate.get(self.solved_col)
            reaction[self.log_col]["post_pipeline_confidence"] = candidate.get(self.confidence_col)
            reaction[self.log_col]["post_pipeline_issue"] = candidate.get(self.issue_col)
            reaction[self.log_col]["post_pipeline_balance_analysis"] = post_balance

            # === 物种消去（Section六 6.7 步骤2）===
            post_rxn = candidate.get(self.reaction_col, "") or candidate.get("input_reaction", "")
            if post_rxn and ">>" in post_rxn:
                from synrbl.bridge_strategy_selector import species_cancellation
                cancelled_rxn = species_cancellation(post_rxn)
                if cancelled_rxn != post_rxn:
                    candidate[self.reaction_col] = cancelled_rxn
                    candidate["input_reaction"] = cancelled_rxn
                    reaction[self.log_col]["species_cancellation_applied"] = True
                    reaction[self.log_col]["cancelled_reaction"] = cancelled_rxn
                    # 重新验证消去后的原子守恒
                    post_balance = LLMSpeciesBridge.analyze_reaction_balance(cancelled_rxn)
                    reaction[self.log_col]["post_cancel_balance_analysis"] = post_balance
                else:
                    reaction[self.log_col]["species_cancellation_applied"] = False

            if not post_balance.get("is_balanced", False):
                # === 外部循环验证（Section六 6.7 步骤3）===
                # 将精确的原子差额反馈给 LLM，要求其修正，最多重试 2 次
                retry_success = False
                # R-4 修复：使用经过管线后处理（含物种消去）的版本作为重试起点，
                # 而非 LLM 的原始输出，让 LLM 在已修正的基础上继续优化
                retry_reaction = post_rxn
                max_retries = 2

                for retry_idx in range(max_retries):
                    retry_imbalance = LLMSpeciesBridge.analyze_reaction_balance(retry_reaction)

                    retry_payload = {
                        "reaction_id": reaction.get(self.id_col),
                        "input_reaction": current_best_reaction,
                        "current_reaction": retry_reaction,
                        "calculated_imbalance": retry_imbalance.get("imbalance_text", ""),
                        "balance_analysis": retry_imbalance,
                        "previous_attempt": retry_reaction,
                        "previous_imbalance": retry_imbalance.get("imbalance_text", ""),
                        "issue": reaction.get(self.issue_col, ""),
                        "bridge_reaction_type": reaction.get("bridge_reaction_type", ""),
                        "task_note": (
                            f"[CORRECTION REQUIRED — Retry {retry_idx + 1}/{max_retries}]\n"
                            f"Your previous output '{retry_reaction}' is NOT atom-balanced.\n"
                            f"Exact imbalance: {retry_imbalance.get('imbalance_text', 'unknown')}.\n"
                            f"This is a correction attempt — you MUST carefully address the "
                            f"specific imbalance above. Do NOT rush. Verify every atom on both "
                            f"sides before outputting.\n"
                            f"Output a corrected, fully balanced reaction SMILES."
                        ),
                    }

                    try:
                        retry_result = self.generate_candidate_fn(retry_payload)
                        if not retry_result:
                            continue
                        retry_predicted = str(
                            retry_result.get("predicted_reaction_smiles", "")
                        ).strip()
                        if not retry_predicted or ">>" not in retry_predicted:
                            continue

                        # C-11 修复：更新 retry_reaction 为最新重试输出，
                        # 确保下一次迭代的分析和反馈基于最新结果
                        retry_reaction = retry_predicted

                        # 验证重试结果
                        retry_check = LLMSpeciesBridge.analyze_reaction_balance(
                            retry_predicted
                        )
                        reaction[self.log_col][f"retry_{retry_idx + 1}_reaction"] = retry_predicted
                        reaction[self.log_col][f"retry_{retry_idx + 1}_balance"] = retry_check

                        if retry_check.get("is_balanced", False):
                            # 重试成功
                            retry_candidate = copy.deepcopy(reaction)
                            retry_candidate["input_reaction"] = retry_predicted
                            retry_candidate[self.reaction_col] = retry_predicted
                            retry_candidate[self.solved_col] = False
                            retry_candidate[self.issue_col] = ""
                            retry_candidate["solved_by"] = ""

                            try:
                                balancer.run_post_generation_pipeline(
                                    [retry_candidate], stats=stats
                                )
                            except Exception as retry_pipeline_exc:
                                # R-5 修复：记录重试管线异常，便于调试
                                logger.warning(
                                    "Fallback retry %d pipeline failed for %s: %s",
                                    retry_idx + 1,
                                    reaction.get(self.id_col, "?"),
                                    retry_pipeline_exc,
                                )
                                reaction[self.log_col][
                                    f"retry_{retry_idx + 1}_pipeline_error"
                                ] = str(retry_pipeline_exc)
                                continue

                            retry_post_balance = LLMSpeciesBridge.analyze_reaction_balance(
                                retry_candidate.get(self.reaction_col, "")
                            )

                            # 物种消去
                            retry_post_rxn = retry_candidate.get(
                                self.reaction_col, ""
                            )
                            if retry_post_rxn and ">>" in retry_post_rxn:
                                from synrbl.bridge_strategy_selector import species_cancellation
                                cancelled = species_cancellation(retry_post_rxn)
                                if cancelled != retry_post_rxn:
                                    retry_candidate[self.reaction_col] = cancelled
                                    retry_post_balance = LLMSpeciesBridge.analyze_reaction_balance(cancelled)

                            if retry_post_balance.get("is_balanced", False):
                                # 重试后通过验证
                                retry_candidate[self.solved_col] = True
                                retry_candidate["solved_by"] = "llm-fallback-retry"
                                retry_candidate["workflow_confidence"] = 3.0
                                fb_log_snap = copy.deepcopy(reaction[self.log_col])
                                cand_no_log = copy.deepcopy(retry_candidate)
                                cand_no_log.pop(self.log_col, None)
                                reaction.update(cand_no_log)
                                reaction[self.log_col] = fb_log_snap
                                reaction["bridge_best_reaction"] = reaction.get(
                                    self.reaction_col
                                )
                                reaction["workflow_confidence_origin"] = "fallback"
                                reaction[self.log_col]["final_status"] = "accepted_balanced_via_retry"
                                reaction[self.log_col][
                                    "retry_count"
                                ] = retry_idx + 1
                                recovered += 1
                                retry_success = True
                                break
                    except Exception as exc:
                        reaction[self.log_col][f"retry_{retry_idx + 1}_error"] = str(exc)
                        continue

                if retry_success:
                    continue

                reaction[self.log_col]["final_status"] = "post_pipeline_unbalanced"
                reaction[self.log_col]["failure_reason"] = (
                    "Generated reaction failed final balance audit after "
                    f"{max_retries} retry attempts."
                )
                continue

            candidate[self.solved_col] = True
            candidate[self.issue_col] = ""
            candidate["solved_by"] = "llm-fallback-postprocess"
            # 置信度 3.0：LLM 兜底直接生成（可靠性相对最低）
            candidate["workflow_confidence"] = 3.0
            fallback_log_snapshot = copy.deepcopy(reaction[self.log_col])
            candidate_without_log = copy.deepcopy(candidate)
            candidate_without_log.pop(self.log_col, None)
            reaction.update(candidate_without_log)
            reaction[self.log_col] = fallback_log_snapshot
            reaction["bridge_best_reaction"] = reaction.get(self.reaction_col)
            reaction["workflow_confidence_origin"] = "fallback"
            reaction[self.log_col]["final_status"] = "accepted_balanced"
            recovered += 1

        if stats is not None:
            stats["llm_fallback_retry_cnt"] = triggered
            stats["llm_fallback_recovered_cnt"] = recovered
        return reactions

    def _should_retry_reaction(self, reaction: Dict[str, Any], current_balance: Dict[str, Any]) -> bool:
        explicit_fallback_input = reaction.get("fallback_input_reaction")
        if explicit_fallback_input in {None, ""}:
            return False
        current_is_balanced = bool(current_balance.get("is_balanced", False))
        if current_is_balanced:
            return False
        return True
