import copy
import logging
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
        retry_confidence_threshold: float = 0.8,
        generate_candidate_fn: Optional[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = None,
        log_col: str = "llm_fallback_postprocess",
        max_workers: int = 20,
    ):
        self.id_col = id_col
        self.reaction_col = reaction_col
        self.solved_col = solved_col
        self.issue_col = issue_col
        self.confidence_col = confidence_col
        self.retry_confidence_threshold = retry_confidence_threshold
        self.generate_candidate_fn = generate_candidate_fn
        self.log_col = log_col
        self.max_workers = max(1, int(max_workers))

    @classmethod
    def from_moonshot(
        cls,
        id_col: str,
        reaction_col: str,
        retry_confidence_threshold: float = 0.8,
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
            retry_confidence_threshold=retry_confidence_threshold,
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
            if explicit_fallback_input in {None, ""}:
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
                "retry_confidence_threshold": self.retry_confidence_threshold,
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

            payload = {
                "reaction_id": reaction.get(self.id_col),
                "input_reaction": current_best_reaction,
                "current_reaction": current_best_reaction,
                "calculated_imbalance": imbalance.get("imbalance_text", ""),
                "balance_analysis": imbalance,
                "issue": reaction.get(self.issue_col, ""),
                "task_note": (
                    "Fallback task: produce the most plausible fully balanced reaction from the current reaction and exact atom imbalance. "
                    "Prefer a complete best guess over a conservative empty result."
                ),
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

            if not post_balance.get("is_balanced", False):
                reaction[self.log_col]["final_status"] = "post_pipeline_unbalanced"
                reaction[self.log_col]["failure_reason"] = "Generated reaction failed final balance audit."
                continue

            candidate[self.solved_col] = True
            candidate[self.issue_col] = ""
            candidate["solved_by"] = "llm-fallback-postprocess"
            candidate["workflow_confidence"] = 2.0
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
