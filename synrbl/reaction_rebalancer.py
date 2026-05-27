"""
reaction_rebalancer.py

Provides ReactionRebalancer for rebalancing chemical reactions pipeline: standardization,
balancing (which includes its own post-processing), neutralization, and deionization.
Uses internal 'R-id' copied from external id_col when they differ.
"""

import copy
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from synrbl import Balancer
from synrbl.evaluation_utils import write_accuracy_report, write_workflow_statistics
from synrbl.llm_fallback_postprocessor import LLMFallbackPostprocessor
from synrbl.llm_postprocessor import LLMPostprocessor
from synrbl.llm_species_bridge import LLMSpeciesBridge
from synkit.IO.debug import setup_logging
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.deionize import Deionize
from synkit.Chem.Reaction.neutralize import Neutralize

logger = setup_logging()


@dataclass(frozen=True)
class RebalanceConfig:
    reaction_col: str = "reactions"
    id_col: str = "R-id"
    n_jobs: int = 1
    batch_size: int = 1000
    raise_on_error: bool = False
    enable_logging: bool = True
    use_default_reduction: bool = False
    synrbl_confidence_threshold: float = 0.5
    llm_postprocessor: Optional[LLMPostprocessor] = None
    llm_species_bridge: Optional[LLMSpeciesBridge] = None
    llm_fallback_postprocessor: Optional[LLMFallbackPostprocessor] = None
    enable_llm_postprocess: bool = False
    enable_llm_species_bridge: bool = False
    llm_score_threshold: float = 0.5
    llm_retry_on_low_confidence: bool = True
    llm_retry_confidence_threshold: float = 0.5
    llm_species_bridge_confidence_threshold: float = 0.5
    llm_top_k_per_strategy: int = 1
    llm_enable_candidate_filter: bool = False
    llm_enable_two_stage_generation: bool = False
    llm_api_key_env: str = "MOONSHOT_API_KEY"
    llm_base_url: str = "https://api.moonshot.cn/v1/chat/completions"
    llm_score_model: str = "kimi-k2.5"
    llm_generate_model: str = "kimi-k2.5"
    llm_thinking_enabled: bool = False
    llm_max_workers: int = 25
    output_dir: Optional[str] = None
    expected_reaction_col: str = "expected_reaction"


class ReactionRebalancer:
    INTERNAL_ID: str = "R-id"
    INTERNAL_TRACKING_ID: str = "_synrbl_internal_id"

    def __init__(
        self,
        config: Optional[RebalanceConfig] = None,
        user_logger: Optional[logging.Logger] = None,
    ):
        if config is not None and not isinstance(config, RebalanceConfig):
            raise TypeError(f"config must be RebalanceConfig, got {type(config)}")
        if user_logger is not None and not isinstance(user_logger, logging.Logger):
            raise TypeError(f"logger must be logging.Logger, got {type(user_logger)}")

        self.config = config or RebalanceConfig()
        self.logger = user_logger if user_logger is not None else logger
        if not self.config.enable_logging:
            logging.disable(logging.CRITICAL)
        self.standardizer = Standardize()
        self._ensure_llm_postprocessor()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"

    @staticmethod
    def describe() -> None:
        print(
            "Usage examples for ReactionRebalancer:\n"
            "  rr = ReactionRebalancer()\n"
            "  result = rr.rebalance(data_frame_or_list)"
        )

    def rebalance(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], keep_extra: bool = False
    ) -> List[Dict[str, Any]]:
        cfg = self.config
        ext_id, int_id, rxn_col = cfg.id_col, self.INTERNAL_ID, cfg.reaction_col
        self.logger.info("Starting rebalancing pipeline.")

        records = self._load_records(data)
        self._init_ids(records, ext_id, int_id)
        std_data = self._standardize_records(records, rxn_col)
        balanced = self._balance_reactions(std_data, rxn_col, int_id)
        restored = self._restore_internal_id(std_data, balanced)
        restored = self._carry_original_metadata(records, restored)

        original_stage_results = [self._build_original_stage_result(entry.copy(), rxn_col) for entry in restored]

        no_llm_stage = [self._strip_llm_side_effects(entry.copy()) for entry in restored]
        no_llm_fixed = self._neutralize_deionize(no_llm_stage, rxn_col)
        no_llm_results = self._extract_results(no_llm_fixed, ext_id, int_id, rxn_col, True)

        with_llm_stage = [entry.copy() for entry in restored]
        fixed = self._neutralize_deionize(with_llm_stage, rxn_col)
        results = self._extract_results(fixed, ext_id, int_id, rxn_col, keep_extra)

        self._write_outputs(records, original_stage_results, no_llm_results, results, ext_id, rxn_col)
        return results

    def _ensure_llm_postprocessor(self) -> None:
        cfg = self.config
        if cfg.llm_postprocessor is None and cfg.enable_llm_postprocess:
            llm_postprocessor = LLMPostprocessor.from_moonshot(
                id_col=self.INTERNAL_ID,
                reaction_col=cfg.reaction_col,
                score_threshold=cfg.llm_score_threshold,
                retry_confidence_threshold=cfg.llm_retry_confidence_threshold,
                retry_on_low_confidence=cfg.llm_retry_on_low_confidence,
                top_k_per_strategy=cfg.llm_top_k_per_strategy,
                enable_candidate_filter=cfg.llm_enable_candidate_filter,
                enable_two_stage_generation=cfg.llm_enable_two_stage_generation,
                api_key_env=cfg.llm_api_key_env,
                base_url=cfg.llm_base_url,
                score_model=cfg.llm_score_model,
                generate_model=cfg.llm_generate_model,
                thinking_enabled=cfg.llm_thinking_enabled,
            )
        else:
            llm_postprocessor = cfg.llm_postprocessor

        if cfg.llm_species_bridge is None and cfg.enable_llm_species_bridge:
            llm_species_bridge = LLMSpeciesBridge.from_moonshot(
                id_col=self.INTERNAL_ID,
                reaction_col=cfg.reaction_col,
                confidence_threshold=cfg.llm_species_bridge_confidence_threshold,
                api_key_env=cfg.llm_api_key_env,
                base_url=cfg.llm_base_url,
                model=cfg.llm_generate_model,
                max_workers=cfg.llm_max_workers,
                thinking_enabled=cfg.llm_thinking_enabled,
            )
        else:
            llm_species_bridge = cfg.llm_species_bridge

        llm_fallback_postprocessor = cfg.llm_fallback_postprocessor
        if cfg.enable_llm_postprocess:
            llm_fallback_postprocessor = LLMFallbackPostprocessor.from_moonshot(
                id_col=self.INTERNAL_ID,
                reaction_col=cfg.reaction_col,
                retry_confidence_threshold=cfg.llm_retry_confidence_threshold,
                api_key_env=cfg.llm_api_key_env,
                base_url=cfg.llm_base_url,
                model=cfg.llm_generate_model,
                max_workers=cfg.llm_max_workers,
                thinking_enabled=cfg.llm_thinking_enabled,
            )

        if llm_postprocessor is None and llm_species_bridge is None and llm_fallback_postprocessor is None:
            return
        self.config = RebalanceConfig(
            reaction_col=cfg.reaction_col,
            id_col=cfg.id_col,
            n_jobs=cfg.n_jobs,
            batch_size=cfg.batch_size,
            raise_on_error=cfg.raise_on_error,
            enable_logging=cfg.enable_logging,
            use_default_reduction=cfg.use_default_reduction,
            synrbl_confidence_threshold=cfg.synrbl_confidence_threshold,
            llm_postprocessor=llm_postprocessor,
            llm_species_bridge=llm_species_bridge,
            llm_fallback_postprocessor=llm_fallback_postprocessor,
            enable_llm_postprocess=cfg.enable_llm_postprocess,
            enable_llm_species_bridge=cfg.enable_llm_species_bridge,
            llm_score_threshold=cfg.llm_score_threshold,
            llm_retry_on_low_confidence=cfg.llm_retry_on_low_confidence,
            llm_retry_confidence_threshold=cfg.llm_retry_confidence_threshold,
            llm_species_bridge_confidence_threshold=cfg.llm_species_bridge_confidence_threshold,
            llm_top_k_per_strategy=cfg.llm_top_k_per_strategy,
            llm_enable_candidate_filter=cfg.llm_enable_candidate_filter,
            llm_enable_two_stage_generation=cfg.llm_enable_two_stage_generation,
            llm_api_key_env=cfg.llm_api_key_env,
            llm_base_url=cfg.llm_base_url,
            llm_score_model=cfg.llm_score_model,
            llm_generate_model=cfg.llm_generate_model,
            llm_thinking_enabled=cfg.llm_thinking_enabled,
            llm_max_workers=cfg.llm_max_workers,
            output_dir=cfg.output_dir,
            expected_reaction_col=cfg.expected_reaction_col,
        )

    def _load_records(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(data, pd.DataFrame):
            self.logger.debug("Converted DataFrame to records, count=%d", len(data))
            return data.to_dict("records")
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return [x.copy() for x in data]
        raise ValueError(f"Unsupported data type: {type(data)}")

    def _init_ids(self, records: List[Dict[str, Any]], ext_id: str, int_id: str) -> None:
        for index, entry in enumerate(records):
            entry.setdefault("original_row_index", index)
            entry.setdefault("original_reaction", entry.get(self.config.reaction_col))
            entry[self.INTERNAL_TRACKING_ID] = entry.get(self.INTERNAL_TRACKING_ID) or (
                f"synrbl-internal-{index}-{uuid.uuid4().hex}"
            )
            entry.setdefault("processable", True)
            entry.setdefault("standardize_status", "pending")
            entry.setdefault("preprocess_status", "pending")
            entry.setdefault("issue", entry.get("issue", ""))
            if ext_id != int_id:
                entry[int_id] = entry.get(ext_id)

    def _standardize_records(
        self, records: List[Dict[str, Any]], rxn_col: str
    ) -> List[Dict[str, Any]]:
        out = []
        self.logger.info("Standardizing reactions.")
        for entry in records:
            raw = entry.get(rxn_col)
            if not raw:
                entry["std_rxn"] = None
                entry["standardize_status"] = "missing_reaction"
                entry["processable"] = False
                if not entry.get("issue"):
                    entry["issue"] = f"Missing '{rxn_col}'."
                out.append(entry)
                continue
            try:
                std = self.standardizer.fit(raw, remove_aam=True)
                entry["std_rxn"] = std
                entry[rxn_col] = std
                entry["standardize_status"] = "success"
            except Exception as exc:
                self.logger.exception("Std failed: %s", raw)
                entry["std_rxn"] = raw
                entry[rxn_col] = raw
                entry["standardize_status"] = "failed"
                entry["processable"] = False
                if not entry.get("issue"):
                    entry["issue"] = f"Standardization failed: {exc}"
            out.append(entry)
        if not out and self.config.raise_on_error:
            raise RuntimeError("Zero valid entries.")
        return out

    def _balance_reactions(
        self, data: List[Dict[str, Any]], rxn_col: str, int_id: str
    ) -> List[Dict[str, Any]]:
        self.logger.info("Balancing %d reactions.", len(data))
        processable = [entry for entry in data if entry.get("processable", True)]
        skipped = [entry.copy() for entry in data if not entry.get("processable", True)]

        for entry in skipped:
            entry.setdefault("input_reaction", entry.get(rxn_col))
            entry.setdefault("solved", False)
            entry.setdefault("solved_by", "")
            entry.setdefault("confidence", None)
            entry.setdefault("rules", [])
            entry.setdefault("mcs", None)
            entry.setdefault("prebalance_check", {"checked": False})
            if not entry.get("issue"):
                entry["issue"] = "Skipped before balancing because preprocessing or standardization failed."

        balanced_results: List[Dict[str, Any]] = []
        if processable:
            try:
                balancer = Balancer(
                    reaction_col=rxn_col,
                    id_col=int_id,
                    n_jobs=self.config.n_jobs,
                    batch_size=self.config.batch_size,
                    use_default_reduction=self.config.use_default_reduction,
                    confidence_threshold=self.config.synrbl_confidence_threshold,
                    llm_postprocessor=self.config.llm_postprocessor,
                    llm_species_bridge=self.config.llm_species_bridge,
                    llm_fallback_postprocessor=self.config.llm_fallback_postprocessor,
                )
                balanced_results = balancer.rebalance(reactions=processable, output_dict=True)
                for order_idx, balanced_entry in enumerate(balanced_results):
                    balanced_entry.setdefault("pipeline_pass_index", order_idx)
            except Exception:
                self.logger.exception("Balancer failed.")
                if self.config.raise_on_error:
                    raise
                for entry in processable:
                    fallback = entry.copy()
                    fallback.setdefault("input_reaction", fallback.get(rxn_col))
                    fallback["solved"] = False
                    fallback.setdefault("solved_by", "")
                    fallback.setdefault("confidence", None)
                    fallback.setdefault("rules", [])
                    fallback.setdefault("mcs", None)
                    fallback.setdefault("prebalance_check", {"checked": False})
                    if not fallback.get("issue"):
                        fallback["issue"] = "Balancer failed during processing."
                    skipped.append(fallback)

        combined = balanced_results + skipped
        result_by_tracking_id = {
            entry.get(self.INTERNAL_TRACKING_ID): entry for entry in combined
        }
        ordered_results = []
        for entry in data:
            tracking_id = entry.get(self.INTERNAL_TRACKING_ID)
            if tracking_id in result_by_tracking_id:
                ordered_results.append(result_by_tracking_id[tracking_id])
                continue
            fallback = entry.copy()
            fallback.setdefault("input_reaction", fallback.get(rxn_col))
            fallback["solved"] = False
            fallback.setdefault("solved_by", "")
            fallback.setdefault("confidence", None)
            fallback.setdefault("rules", [])
            fallback.setdefault("mcs", None)
            fallback.setdefault("prebalance_check", {"checked": False})
            if not fallback.get("issue"):
                fallback["issue"] = "No balancing result was produced."
            ordered_results.append(fallback)
        return ordered_results

    def _neutralize_deionize(
        self, data: List[Dict[str, Any]], rxn_col: str
    ) -> List[Dict[str, Any]]:
        processable = []
        passthrough = []
        for entry in data:
            entry.setdefault("neutralize_status", "pending")
            entry.setdefault("deionize_status", "pending")
            if entry.get("processable", True) and entry.get(rxn_col):
                processable.append(entry)
            else:
                entry["neutralize_status"] = "skipped"
                entry["deionize_status"] = "skipped"
                passthrough.append(entry)

        self.logger.info("Neutralization.")
        if processable:
            try:
                processable = Neutralize.parallel_fix_unbalanced_charge(
                    processable, rxn_col, self.config.n_jobs
                )
                for entry in processable:
                    entry["neutralize_status"] = "success"
            except Exception:
                self.logger.exception("Neutralization failed.")
                if self.config.raise_on_error:
                    raise
                for entry in processable:
                    entry["neutralize_status"] = "failed"
                    if not entry.get("issue"):
                        entry["issue"] = "Neutralization failed."

        self.logger.info("Deionization.")
        if processable:
            try:
                processable = Deionize.apply_uncharge_smiles_to_reactions(
                    processable, Deionize.uncharge_smiles, n_jobs=1
                )
                for entry in processable:
                    entry["deionize_status"] = "success"
            except Exception:
                self.logger.exception("Deionization failed.")
                if self.config.raise_on_error:
                    raise
                for entry in processable:
                    entry["deionize_status"] = "failed"
                    if not entry.get("issue"):
                        entry["issue"] = "Deionization failed."

        processed_by_tracking_id = {
            entry.get(self.INTERNAL_TRACKING_ID): entry
            for entry in processable + passthrough
        }
        return [processed_by_tracking_id.get(entry.get(self.INTERNAL_TRACKING_ID), entry) for entry in data]

    @staticmethod
    def _normalize_numeric_confidence(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            text = text.replace("％", "%")
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

    def _derive_original_stage_workflow(self, entry: Dict[str, Any], rxn_col: str) -> Dict[str, Any]:
        reaction = entry.get(rxn_col)
        balanced = bool(LLMSpeciesBridge.analyze_reaction_balance(str(reaction or "")).get("is_balanced", False))
        solved_by = str(entry.get("solved_by") or "").strip()
        confidence = self._normalize_numeric_confidence(entry.get("confidence"))
        result = {
            "formal_output_reaction": None,
            "workflow_confidence": None,
            "workflow_source": "prebalance" if entry.get("prebalance_check", {}).get("short_circuited", False) else "SynRBL",
            "internal_candidate_1_reaction": None,
            "internal_candidate_1_confidence": None,
            "internal_candidate_1_source": None,
        }
        if entry.get("prebalance_check", {}).get("short_circuited", False):
            result["formal_output_reaction"] = reaction
            result["workflow_confidence"] = 1.0
            return result
        if not balanced:
            return result
        if solved_by == "rule-based":
            result["formal_output_reaction"] = reaction
            result["workflow_confidence"] = 1.0
            return result
        if confidence is None:
            return result
        if confidence >= 0.8:
            result["formal_output_reaction"] = reaction
            result["workflow_confidence"] = confidence
            return result
        result["internal_candidate_1_reaction"] = reaction
        result["internal_candidate_1_confidence"] = confidence
        result["internal_candidate_1_source"] = "SynRBL"
        return result

    def _resolve_workflow_result(self, entry: Dict[str, Any], rxn_col: str) -> Dict[str, Any]:
        original_stage = self._build_original_stage_result(entry.copy(), rxn_col)
        original_view = self._derive_original_stage_workflow(original_stage, rxn_col)
        original_reaction = entry.get("original_reaction") or entry.get("input_reaction") or entry.get(rxn_col)
        bridge_candidate_reaction = entry.get("bridge_candidate_reaction")
        bridge_best_reaction = entry.get("bridge_best_reaction")
        bridge_raw_output = bridge_candidate_reaction or bridge_best_reaction
        resolved = {
            "final_reaction": original_view.get("formal_output_reaction"),
            "success": bool(original_view.get("formal_output_reaction")),
            "workflow_confidence": original_view.get("workflow_confidence"),
            "workflow_source": original_view.get("workflow_source"),
            "formal_output_reaction": original_view.get("formal_output_reaction"),
            "internal_candidate_1_reaction": original_view.get("internal_candidate_1_reaction"),
            "internal_candidate_1_confidence": original_view.get("internal_candidate_1_confidence"),
            "internal_candidate_1_source": original_view.get("internal_candidate_1_source"),
            "internal_candidate_2_reaction": None,
            "internal_candidate_2_confidence": None,
            "internal_candidate_2_source": None,
            "fallback_input_source": None,
            "fallback_input_reaction": None,
            "bridge_raw_output_reaction": bridge_raw_output,
        }

        bridge_log = entry.get("llm_species_bridge", {})
        bridge_triggered = isinstance(bridge_log, dict) and bridge_log.get("triggered") is True
        bridge_variant_reaction = None
        bridge_variant_confidence = None
        bridge_variant_kind = None
        bridge_direct_balance = LLMSpeciesBridge.analyze_reaction_balance(str(bridge_raw_output or ""))
        bridge_direct_balanced = bool(bridge_raw_output and bridge_direct_balance.get("is_balanced", False))
        if bridge_triggered:
            accepted_eval = bridge_log.get("accepted_variant_evaluation") if isinstance(bridge_log, dict) else None
            accepted_reaction = bridge_log.get("accepted_variant") if isinstance(bridge_log, dict) else None
            accepted_balance = bridge_log.get("accepted_variant_balance_analysis") if isinstance(bridge_log, dict) else None
            accepted_balanced = bool(isinstance(accepted_balance, dict) and accepted_balance.get("is_balanced", False))
            accepted_solved_by = ""
            if isinstance(accepted_eval, dict):
                accepted_solved_by = str(accepted_eval.get("solved_by") or "").strip()
                bridge_variant_confidence = self._normalize_numeric_confidence(accepted_eval.get("confidence"))
            if accepted_balanced and accepted_reaction:
                bridge_variant_reaction = accepted_reaction
                if accepted_solved_by == "rule-based":
                    bridge_variant_kind = "small_molecule"
                elif bridge_variant_confidence is not None:
                    bridge_variant_kind = "mcs"
                else:
                    bridge_variant_kind = "mcs_missing_confidence"
            elif bridge_direct_balanced:
                bridge_variant_reaction = bridge_raw_output
                bridge_variant_kind = "bridge_direct"

        if resolved.get("formal_output_reaction") is None and bridge_triggered:
            if bridge_variant_kind in {"bridge_direct", "small_molecule"} and bridge_variant_reaction:
                resolved["final_reaction"] = bridge_variant_reaction
                resolved["success"] = True
                resolved["workflow_confidence"] = 1.5
                resolved["workflow_source"] = "bridge"
                resolved["formal_output_reaction"] = bridge_variant_reaction
            elif bridge_variant_kind == "mcs" and bridge_variant_reaction and bridge_variant_confidence is not None:
                if bridge_variant_confidence >= 0.8:
                    resolved["final_reaction"] = bridge_variant_reaction
                    resolved["success"] = True
                    resolved["workflow_confidence"] = bridge_variant_confidence
                    resolved["workflow_source"] = "bridge"
                    resolved["formal_output_reaction"] = bridge_variant_reaction
                else:
                    candidate1_conf = resolved.get("internal_candidate_1_confidence")
                    candidate1_rxn = resolved.get("internal_candidate_1_reaction")
                    if candidate1_conf is None:
                        resolved["internal_candidate_2_reaction"] = bridge_variant_reaction
                        resolved["internal_candidate_2_confidence"] = bridge_variant_confidence
                        resolved["internal_candidate_2_source"] = "bridge"
                    elif bridge_variant_confidence >= candidate1_conf:
                        resolved["internal_candidate_2_reaction"] = bridge_variant_reaction
                        resolved["internal_candidate_2_confidence"] = bridge_variant_confidence
                        resolved["internal_candidate_2_source"] = "bridge"
                    else:
                        resolved["internal_candidate_2_reaction"] = candidate1_rxn
                        resolved["internal_candidate_2_confidence"] = candidate1_conf
                        resolved["internal_candidate_2_source"] = "SynRBL"
            elif bridge_variant_kind == "mcs_missing_confidence":
                candidate1_conf = resolved.get("internal_candidate_1_confidence")
                if candidate1_conf is not None:
                    resolved["internal_candidate_2_reaction"] = resolved.get("internal_candidate_1_reaction")
                    resolved["internal_candidate_2_confidence"] = candidate1_conf
                    resolved["internal_candidate_2_source"] = "SynRBL"
                    resolved["fallback_input_source"] = "original"
                    resolved["fallback_input_reaction"] = original_reaction
                else:
                    resolved["fallback_input_source"] = "bridge_raw" if bridge_raw_output else "original"
                    resolved["fallback_input_reaction"] = bridge_raw_output or original_reaction
            else:
                candidate1_conf = resolved.get("internal_candidate_1_confidence")
                if candidate1_conf is not None:
                    resolved["internal_candidate_2_reaction"] = resolved.get("internal_candidate_1_reaction")
                    resolved["internal_candidate_2_confidence"] = candidate1_conf
                    resolved["internal_candidate_2_source"] = "SynRBL"
                    resolved["fallback_input_source"] = "original"
                    resolved["fallback_input_reaction"] = original_reaction
                else:
                    resolved["fallback_input_source"] = "original"
                    resolved["fallback_input_reaction"] = original_reaction

        candidate2_conf = resolved.get("internal_candidate_2_confidence")
        if resolved.get("formal_output_reaction") is None and candidate2_conf is not None:
            if candidate2_conf >= 0.8:
                resolved["final_reaction"] = resolved.get("internal_candidate_2_reaction")
                resolved["success"] = True
                resolved["workflow_confidence"] = candidate2_conf
                resolved["workflow_source"] = resolved.get("internal_candidate_2_source") or ("bridge" if bridge_triggered else "SynRBL")
                resolved["formal_output_reaction"] = resolved.get("internal_candidate_2_reaction")
            elif resolved.get("fallback_input_reaction") is None:
                if resolved.get("internal_candidate_2_source") == "bridge":
                    resolved["fallback_input_source"] = "bridge"
                    resolved["fallback_input_reaction"] = resolved.get("internal_candidate_2_reaction")
                else:
                    resolved["fallback_input_source"] = "original"
                    resolved["fallback_input_reaction"] = original_reaction

        if resolved.get("formal_output_reaction") is None and resolved.get("fallback_input_reaction") is None:
            if bridge_raw_output and resolved.get("internal_candidate_1_confidence") is None and resolved.get("internal_candidate_2_confidence") is None:
                resolved["fallback_input_source"] = "bridge_raw"
                resolved["fallback_input_reaction"] = bridge_raw_output
            else:
                resolved["fallback_input_source"] = "original"
                resolved["fallback_input_reaction"] = original_reaction

        fallback_log = entry.get("llm_fallback_postprocess", {})
        fallback_triggered = isinstance(fallback_log, dict) and fallback_log.get("triggered") is True
        if resolved.get("formal_output_reaction") is None and fallback_triggered:
            fallback_reaction = fallback_log.get("post_pipeline_reaction") or entry.get(rxn_col)
            fallback_balanced = bool(LLMSpeciesBridge.analyze_reaction_balance(str(fallback_reaction or "")).get("is_balanced", False))
            if fallback_balanced:
                resolved["final_reaction"] = fallback_reaction
                resolved["formal_output_reaction"] = fallback_reaction
                resolved["success"] = True
                resolved["workflow_confidence"] = 2.0
                resolved["workflow_source"] = "fallback"

        if resolved.get("formal_output_reaction") is None:
            resolved["final_reaction"] = None
            resolved["success"] = False
            resolved["workflow_confidence"] = None
            if fallback_triggered:
                resolved["workflow_source"] = "fallback"
            elif bridge_triggered:
                resolved["workflow_source"] = "bridge"
            else:
                resolved["workflow_source"] = original_view.get("workflow_source")
        return resolved

    def _extract_results(
        self,
        data: List[Dict[str, Any]],
        ext_id: str,
        int_id: str,
        rxn_col: str,
        keep_extra: bool,
    ) -> List[Dict[str, Any]]:
        self.logger.info("Extracting results.")
        results = []
        for entry in data:
            if ext_id != int_id:
                entry[ext_id] = entry.get(int_id)
            llm_log = entry.get("llm_postprocess", {}) if isinstance(entry, dict) else {}
            bridge_log = entry.get("llm_species_bridge", {}) if isinstance(entry, dict) else {}
            resolved_workflow = self._resolve_workflow_result(entry, rxn_col)
            final_reaction = resolved_workflow.get("formal_output_reaction")
            final_balance = LLMSpeciesBridge.analyze_reaction_balance(str(final_reaction or ""))
            final_is_balanced = bool(final_balance.get("is_balanced"))
            entry["final_balance_check"] = final_balance
            entry["strictly_balanced_final"] = final_is_balanced
            entry["workflow_touched"] = bool(
                (isinstance(bridge_log, dict) and bridge_log.get("triggered"))
                or (isinstance(llm_log, dict) and llm_log)
                or entry.get("prebalance_check", {}).get("checked")
                or entry.get("llm_fallback_postprocess", {}).get("triggered")
            )
            entry["success"] = bool(resolved_workflow.get("success", False))
            entry["workflow_confidence"] = resolved_workflow.get("workflow_confidence")
            entry["workflow_source"] = resolved_workflow.get("workflow_source")
            entry["formal_output_reaction"] = resolved_workflow.get("formal_output_reaction")
            entry["internal_candidate_1_reaction"] = resolved_workflow.get("internal_candidate_1_reaction")
            entry["internal_candidate_1_confidence"] = resolved_workflow.get("internal_candidate_1_confidence")
            entry["internal_candidate_2_reaction"] = resolved_workflow.get("internal_candidate_2_reaction")
            entry["internal_candidate_2_confidence"] = resolved_workflow.get("internal_candidate_2_confidence")
            entry["internal_candidate_2_source"] = resolved_workflow.get("internal_candidate_2_source")
            entry["fallback_input_source"] = resolved_workflow.get("fallback_input_source")
            entry["fallback_input_reaction"] = resolved_workflow.get("fallback_input_reaction")
            output_status = "True" if entry["success"] else "False"
            out = {
                ext_id: entry.get(ext_id),
                self.INTERNAL_TRACKING_ID: entry.get(self.INTERNAL_TRACKING_ID),
                "original_row_index": entry.get("original_row_index"),
                "original_reaction": entry.get("original_reaction"),
                rxn_col: final_reaction,
                "output_status": output_status,
                "success": entry["success"],
                "final_result_source": entry.get("workflow_source") or entry.get("solved_by") or "",
            }
            if keep_extra:
                extras = {k: v for k, v in entry.items() if k not in (ext_id, rxn_col, "success")}
                out.update(extras)
            results.append(out)
        return results

    def _write_outputs(
        self,
        original_records: List[Dict[str, Any]],
        original_stage_results: List[Dict[str, Any]],
        no_llm_results: List[Dict[str, Any]],
        final_results: List[Dict[str, Any]],
        ext_id: str,
        rxn_col: str,
    ) -> None:
        output_dir = Path(self.config.output_dir or Path.cwd())
        output_dir.mkdir(parents=True, exist_ok=True)

        original_stage_path = output_dir / "synrbl_results_original_stage.json"
        original_stage_csv_path = output_dir / "synrbl_results_original_stage.csv"
        synrbl_only_path = output_dir / "synrbl_results_synrbl_only.json"
        synrbl_only_csv_path = output_dir / "synrbl_results_synrbl_only.csv"
        before_path = output_dir / "synrbl_results_no_llm.json"
        before_csv_path = output_dir / "synrbl_results_no_llm.csv"
        final_path = output_dir / "synrbl_results_with_llm.json"
        final_csv_path = output_dir / "synrbl_results_with_llm.csv"
        failed_path = output_dir / "synrbl_failed_cases.json"
        failed_csv_path = output_dir / "synrbl_failed_cases_flat.csv"

        pd.DataFrame(self._serialize_structured_fields_for_csv(original_stage_results)).to_csv(original_stage_csv_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(self._serialize_structured_fields_for_csv(original_stage_results)).to_csv(synrbl_only_csv_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(self._serialize_structured_fields_for_csv(no_llm_results)).to_csv(before_csv_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(self._serialize_structured_fields_for_csv(final_results)).to_csv(final_csv_path, index=False, encoding="utf-8-sig")
        original_stage_path.write_text(json.dumps(original_stage_results, ensure_ascii=False, indent=2), encoding="utf-8")
        synrbl_only_path.write_text(json.dumps(original_stage_results, ensure_ascii=False, indent=2), encoding="utf-8")
        before_path.write_text(json.dumps(no_llm_results, ensure_ascii=False, indent=2), encoding="utf-8")
        final_path.write_text(json.dumps(final_results, ensure_ascii=False, indent=2), encoding="utf-8")

        tracking_id = self.INTERNAL_TRACKING_ID
        before_map = {item.get(tracking_id): item for item in no_llm_results if item.get(tracking_id) is not None}
        original_stage_map = {item.get(tracking_id): item for item in original_stage_results if item.get(tracking_id) is not None}
        failed_cases = []
        failed_flat_rows = []
        for final_item in final_results:
            final_reaction = final_item.get(rxn_col)
            final_success = LLMSpeciesBridge.analyze_reaction_balance(str(final_reaction or "")).get("is_balanced", False)
            if final_success:
                continue
            case_tracking_id = final_item.get(tracking_id)
            before_item = before_map.get(case_tracking_id)
            original_stage_item = original_stage_map.get(case_tracking_id)
            case_id = final_item.get(ext_id)
            failed_cases.append(
                {
                    "id": case_id,
                    "tracking_id": case_tracking_id,
                    "original_row_index": final_item.get("original_row_index"),
                    "original_input_reaction": final_item.get("original_reaction"),
                    "original_stage_result": original_stage_item,
                    "no_llm_result": before_item,
                    "full_workflow_result": final_item,
                }
            )
            failed_flat_rows.append(
                {
                    "id": case_id,
                    "tracking_id": case_tracking_id,
                    "original_row_index": final_item.get("original_row_index"),
                    "original_input_reaction": final_item.get("original_reaction"),
                    "workflow_source": self._derive_workflow_source_for_case(final_item),
                    "original_stage_reaction": original_stage_item.get(rxn_col) if original_stage_item else None,
                    "original_stage_confidence": original_stage_item.get("confidence") if original_stage_item else None,
                    "original_stage_workflow_confidence": original_stage_item.get("workflow_confidence") if original_stage_item else None,
                    "original_stage_issue": original_stage_item.get("issue") if original_stage_item else None,
                    "original_stage_solved_by": original_stage_item.get("solved_by") if original_stage_item else None,
                    "no_llm_reaction": before_item.get(rxn_col) if before_item else None,
                    "no_llm_confidence": before_item.get("confidence") if before_item else None,
                    "no_llm_workflow_confidence": before_item.get("workflow_confidence") if before_item else None,
                    "no_llm_issue": before_item.get("issue") if before_item else None,
                    "no_llm_solved_by": before_item.get("solved_by") if before_item else None,
                    "full_reaction": final_reaction,
                    "full_confidence": final_item.get("confidence"),
                    "full_workflow_confidence": final_item.get("workflow_confidence"),
                    "full_issue": final_item.get("issue"),
                    "full_solved_by": final_item.get("solved_by"),
                    "full_output_status": final_item.get("output_status"),
                    ext_id: case_id,
                }
            )
        failed_path.write_text(json.dumps(failed_cases, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(self._serialize_structured_fields_for_csv(failed_flat_rows)).to_csv(failed_csv_path, index=False, encoding="utf-8-sig")

        write_accuracy_report(
            output_dir=output_dir,
            no_llm_results=no_llm_results,
            with_llm_results=final_results,
            reaction_col=rxn_col,
            target_col=self.config.expected_reaction_col,
            group_key="final_result_source",
        )
        sanitized_original_records = []
        for record in original_records:
            sanitized_record = copy.deepcopy(record)
            sanitized_record.pop(self.INTERNAL_TRACKING_ID, None)
            if self.INTERNAL_ID != ext_id:
                sanitized_record.pop(self.INTERNAL_ID, None)
            sanitized_original_records.append(sanitized_record)
        validation_df = pd.DataFrame(sanitized_original_records)
        write_workflow_statistics(
            output_dir=output_dir,
            validation_df=validation_df,
            original_stage_results=original_stage_results,
            with_llm_results=final_results,
            reaction_col=rxn_col,
            target_col=self.config.expected_reaction_col,
        )

    def _serialize_structured_fields_for_csv(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        serialized_rows: List[Dict[str, Any]] = []
        for row in rows:
            serialized_row: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    serialized_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serialized_row[key] = value
            serialized_rows.append(serialized_row)
        return serialized_rows

    def _strip_llm_side_effects(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        llm_override = entry.get("llm_species_bridge")
        if isinstance(llm_override, dict) and llm_override.get("pre_bridge_reaction"):
            entry[self.config.reaction_col] = llm_override.get("pre_bridge_reaction")
            entry["input_reaction"] = entry.get("input_reaction") or llm_override.get("pre_bridge_reaction")
            entry["solved"] = bool(llm_override.get("pre_bridge_solved", False))
            entry["confidence"] = llm_override.get("pre_bridge_confidence")
            entry["issue"] = llm_override.get("pre_bridge_issue", entry.get("issue", ""))
            entry["solved_by"] = llm_override.get("pre_bridge_solved_by", entry.get("solved_by", ""))
        entry.pop("llm_species_bridge", None)
        entry.pop("llm_postprocess", None)
        entry.pop("llm_fallback_postprocess", None)
        return entry

    def _build_original_stage_result(self, entry: Dict[str, Any], rxn_col: str) -> Dict[str, Any]:
        original = entry.copy()
        if original.get("prebalance_check", {}).get("short_circuited", False):
            original[rxn_col] = original.get("cleaned_initial_reaction", original.get(rxn_col))
            original["input_reaction"] = original.get("cleaned_initial_reaction", original.get("input_reaction", original.get(rxn_col)))
            original["solved"] = True
            original["solved_by"] = "prebalanced"
            original["confidence"] = 1.0
            original["workflow_confidence"] = 1.0
            original["issue"] = ""
            original["rules"] = []
            original["final_result_source"] = "prebalanced"
            original_workflow = self._derive_original_stage_workflow(original, rxn_col)
            original.update(original_workflow)
            return original

        original[rxn_col] = original.get("pre_llm_reaction", original.get(rxn_col))
        original["input_reaction"] = original.get(
            "input_reaction",
            original.get("cleaned_initial_reaction", original.get("pre_llm_reaction", original.get(rxn_col))),
        )
        original["solved"] = bool(original.get("pre_llm_solved", original.get("solved", False)))
        original["solved_by"] = original.get("pre_llm_solved_by", original.get("solved_by", ""))
        original["confidence"] = original.get("pre_llm_confidence", original.get("confidence"))
        original["issue"] = original.get("pre_llm_issue", original.get("issue", ""))
        original["rules"] = original.get("pre_llm_rules", original.get("rules", []))
        original_workflow = self._derive_original_stage_workflow(original, rxn_col)
        original["workflow_confidence"] = original_workflow.get("workflow_confidence")
        original["workflow_source"] = original_workflow.get("workflow_source")
        original["formal_output_reaction"] = original_workflow.get("formal_output_reaction")
        original["internal_candidate_1_reaction"] = original_workflow.get("internal_candidate_1_reaction")
        original["internal_candidate_1_confidence"] = original_workflow.get("internal_candidate_1_confidence")
        original["internal_candidate_1_source"] = original_workflow.get("internal_candidate_1_source")
        original["final_result_source"] = original.get("workflow_source") or original.get("solved_by") or ""
        return original

    @staticmethod
    def _derive_workflow_source_for_case(entry: Dict[str, Any]) -> str:
        prebalance_check = entry.get("prebalance_check", {})
        if isinstance(prebalance_check, dict) and prebalance_check.get("short_circuited", False):
            return "prebalance"
        fallback_log = entry.get("llm_fallback_postprocess", {})
        if isinstance(fallback_log, dict) and fallback_log.get("triggered") is True:
            return "fallback"
        bridge_log = entry.get("llm_species_bridge", {})
        if isinstance(bridge_log, dict) and bridge_log.get("triggered") is True:
            return "bridge"
        return "SynRBL"

    @staticmethod
    def _has_wrong_reactions(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, list):
            return any(str(item or "").strip() for item in value)
        text = str(value).strip()
        return bool(text and text not in {"[]", "nan", "None"})

    def _carry_original_metadata(
        self,
        original_list: List[Dict[str, Any]],
        processed_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tracking_id = ReactionRebalancer.INTERNAL_TRACKING_ID
        expected_col = self.config.expected_reaction_col
        original_by_tracking = {
            entry.get(tracking_id): entry for entry in original_list if entry.get(tracking_id)
        }
        out_list: List[Dict[str, Any]] = []
        for entry in processed_list:
            new_entry = entry.copy()
            original_entry = original_by_tracking.get(entry.get(tracking_id))
            if original_entry is not None:
                if expected_col in original_entry:
                    new_entry[expected_col] = original_entry.get(expected_col)
                if "wrong_reactions" in original_entry:
                    new_entry["wrong_reactions"] = original_entry.get("wrong_reactions")
                new_entry["has_wrong_reactions"] = self._has_wrong_reactions(original_entry.get("wrong_reactions"))
            else:
                new_entry.setdefault("has_wrong_reactions", False)
            out_list.append(new_entry)
        return out_list

    @staticmethod
    def _restore_internal_id(
        original_list: List[Dict[str, Any]], processed_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not all(isinstance(x, dict) for x in original_list + processed_list):
            raise ValueError("Both original_list and processed_list must be lists of dicts.")
        int_id = ReactionRebalancer.INTERNAL_ID
        tracking_id = ReactionRebalancer.INTERNAL_TRACKING_ID
        original_by_tracking = {
            entry.get(tracking_id): entry for entry in original_list if entry.get(tracking_id)
        }
        out_list: List[Dict[str, Any]] = []
        for entry in processed_list:
            new_entry = entry.copy()
            original_entry = original_by_tracking.get(entry.get(tracking_id))
            if original_entry is not None:
                new_entry[int_id] = original_entry.get(int_id)
            out_list.append(new_entry)
        return out_list
