import copy
import logging
from typing import Any, Callable, Dict, List, Optional

from synrbl.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict
from synrbl.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from synrbl.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from synrbl.llm.client import (
    DEFAULT_GENERATE_MODEL,
    DEFAULT_MOONSHOT_BASE_URL,
    DEFAULT_SCORE_MODEL,
    LLMResponseParseError,
    MoonshotLLMClient,
)
from synrbl.llm.models import LLMCandidate, LLMPostprocessorLogs

logger = logging.getLogger("synrbl")


class LLMPostprocessor:
    def __init__(
        self,
        id_col: str,
        reaction_col: str,
        solved_col: str = "solved",
        issue_col: str = "issue",
        mcs_data_col: str = "mcs",
        input_col: str = "input_reaction",
        confidence_col: str = "confidence",
        score_threshold: float = 0.8,
        retry_confidence_threshold: float = 0.8,
        retry_on_low_confidence: bool = True,
        top_k_per_strategy: int = 1,
        enable_candidate_filter: bool = False,
        enable_two_stage_generation: bool = False,
        log_col: str = "llm_postprocess",
        diagnose_reaction_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        score_candidates_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        generate_candidate_fn: Optional[
            Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
        ] = None,
    ):
        self.id_col = id_col
        self.reaction_col = reaction_col
        self.solved_col = solved_col
        self.issue_col = issue_col
        self.mcs_data_col = mcs_data_col
        self.input_col = input_col
        self.confidence_col = confidence_col
        self.score_threshold = score_threshold
        self.retry_confidence_threshold = retry_confidence_threshold
        self.retry_on_low_confidence = retry_on_low_confidence
        self.top_k_per_strategy = top_k_per_strategy
        self.enable_candidate_filter = enable_candidate_filter
        self.enable_two_stage_generation = enable_two_stage_generation
        self.log_col = log_col
        self.diagnose_reaction_fn = diagnose_reaction_fn
        self.score_candidates_fn = score_candidates_fn
        self.generate_candidate_fn = generate_candidate_fn

    @classmethod
    def from_moonshot(
        cls,
        id_col: str,
        reaction_col: str,
        score_threshold: float = 0.8,
        retry_confidence_threshold: float = 0.4,
        retry_on_low_confidence: bool = True,
        top_k_per_strategy: int = 1,
        enable_candidate_filter: bool = False,
        enable_two_stage_generation: bool = False,
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: str = DEFAULT_MOONSHOT_BASE_URL,
        score_model: str = DEFAULT_SCORE_MODEL,
        generate_model: str = DEFAULT_GENERATE_MODEL,
        thinking_enabled: bool = False,
    ) -> "LLMPostprocessor":
        client = MoonshotLLMClient(
            api_key_env=api_key_env,
            base_url=base_url,
            score_model=score_model,
            generate_model=generate_model,
            thinking_enabled=thinking_enabled,
        )
        return cls(
            id_col=id_col,
            reaction_col=reaction_col,
            score_threshold=score_threshold,
            retry_confidence_threshold=retry_confidence_threshold,
            retry_on_low_confidence=retry_on_low_confidence,
            top_k_per_strategy=top_k_per_strategy,
            enable_candidate_filter=enable_candidate_filter,
            enable_two_stage_generation=enable_two_stage_generation,
            diagnose_reaction_fn=client.diagnose_reaction,
            score_candidates_fn=client.score_candidates,
            generate_candidate_fn=client.generate_candidate,
        )

    def apply(self, reactions: List[Dict[str, Any]], balancer, stats=None):
        failed_reactions = [r for r in reactions if self._should_retry_reaction(r)]
        if len(failed_reactions) == 0:
            return reactions

        retried = 0
        recovered = 0
        for reaction in failed_reactions:
            retried += 1
            logs = LLMPostprocessorLogs()
            original_was_low_confidence = bool(reaction.get(self.solved_col, False)) and self._is_low_confidence_reaction(reaction)
            reaction["pre_llm_reaction"] = reaction.get(self.reaction_col)
            reaction["pre_llm_solved"] = reaction.get(self.solved_col)
            reaction["pre_llm_solved_by"] = reaction.get("solved_by")
            reaction["pre_llm_confidence"] = reaction.get(self.confidence_col)
            reaction["pre_llm_issue"] = reaction.get(self.issue_col)
            reaction["pre_llm_rules"] = copy.deepcopy(reaction.get("rules", []))
            reaction[self.log_col] = logs.to_dict()
            retry_reaction = self._prepare_retry_reaction(reaction)

            try:
                logs.pre_mcs_retry_triggered = True
                self._run_pre_mcs_retry_steps([retry_reaction], balancer)
                logs.pre_mcs_retry_solved = bool(retry_reaction.get(self.solved_col, False))
                logs.pre_mcs_retry_reaction = retry_reaction.get(self.reaction_col, "")
                logs.pre_mcs_retry_issue = str(retry_reaction.get(self.issue_col, "") or "")

                if retry_reaction.get(self.solved_col, False) and not original_was_low_confidence:
                    logs.selection_path = "pre_mcs_retry"
                    self._merge_retry_result(reaction, retry_reaction)
                    reaction[self.log_col] = logs.to_dict()
                    recovered += 1
                    continue

                logs.selection_path = "llm_generation"
                logs.generation_triggered = True
                selected_result = self.generate_candidate(retry_reaction, [], logs)

                if selected_result is None or "predicted_reaction" not in selected_result:
                    if original_was_low_confidence:
                        logs.force_pending_output = True
                    reaction[self.log_col] = logs.to_dict()
                    continue

                predicted_smiles = (selected_result.get("predicted_reaction") or "").strip()
                logs.generated_predicted_reaction = predicted_smiles

                if logs.input_reasonable is False:
                    reaction[self.solved_col] = False
                    reaction[self.issue_col] = selected_result.get(
                        self.issue_col,
                        "Input molecule judged unreasonable by LLM.",
                    )
                    reaction[self.confidence_col] = 0.0
                    reaction["solved_by"] = "llm-invalid-input"
                    logs.final_success = False
                    reaction[self.log_col] = logs.to_dict()
                    continue

                if not predicted_smiles:
                    if original_was_low_confidence and logs.input_reasonable is True:
                        logs.force_pending_output = True
                    reaction[self.issue_col] = selected_result.get(
                        self.issue_col,
                        "LLM returned empty predicted reaction.",
                    )
                    reaction[self.log_col] = logs.to_dict()
                    continue

                retry_reaction[self.reaction_col] = predicted_smiles
                retry_reaction[self.issue_col] = ""

                parts = predicted_smiles.split(">")
                if len(parts) >= 2:
                    retry_reaction["reactants"] = parts[0]
                    retry_reaction["products"] = parts[-1]
                else:
                    if original_was_low_confidence and logs.input_reasonable is True:
                        logs.force_pending_output = True
                    reaction[self.issue_col] = "LLM returned invalid reaction format."
                    reaction[self.log_col] = logs.to_dict()
                    continue

                balancer.mcs_validator.check(
                    [retry_reaction],
                    override_unsolved=True,
                    override_issue_msg="LLM generated reaction is unbalanced.",
                )

                logs.generated_reaction_after_validation = retry_reaction.get(self.reaction_col, "")
                logs.generated_issue_after_validation = str(retry_reaction.get(self.issue_col, "") or "")
                logs.final_success = bool(retry_reaction.get(self.solved_col, False))
                if logs.final_success:
                    retry_reaction["solved_by"] = "llm-end-to-end"
                    self._merge_retry_result(reaction, retry_reaction)
                    reaction[self.log_col] = logs.to_dict()
                    recovered += 1
                    continue

                retry_reaction[self.confidence_col] = 0.0
                if original_was_low_confidence and logs.input_reasonable is True:
                    logs.force_pending_output = True
                reaction[self.issue_col] = "LLM generated reaction is unbalanced."
                reaction[self.log_col] = logs.to_dict()
            except Exception as exc:
                logger.exception(
                    "LLM postprocessor failed for reaction %s",
                    reaction.get(self.id_col),
                )
                logs.pipeline_exception = str(exc)
                if original_was_low_confidence:
                    logs.force_pending_output = True
                reaction[self.log_col] = logs.to_dict()
                if reaction.get(self.issue_col, "") == "":
                    reaction[self.issue_col] = str(exc)

        if stats is not None:
            stats["llm_retry_cnt"] = retried
            stats["llm_recovered_cnt"] = recovered
        return reactions

    def _should_retry_reaction(self, reaction: Dict[str, Any]) -> bool:
        if not reaction.get(self.solved_col, False):
            return True
        if not self.retry_on_low_confidence:
            return False
        return self._is_low_confidence_reaction(reaction)

    def _is_low_confidence_reaction(self, reaction: Dict[str, Any]) -> bool:
        confidence = reaction.get(self.confidence_col)
        if confidence is None:
            return False
        try:
            return float(confidence) < self.retry_confidence_threshold
        except (TypeError, ValueError):
            return False

    def _prepare_retry_reaction(self, reaction: Dict[str, Any]) -> Dict[str, Any]:
        retry_reaction = copy.deepcopy(reaction)
        retry_reaction[self.reaction_col] = retry_reaction[self.input_col]
        retry_reaction[self.solved_col] = False
        retry_reaction[self.issue_col] = ""
        retry_reaction[self.confidence_col] = 0.0
        retry_reaction[self.mcs_data_col] = None
        return retry_reaction

    def _run_pre_mcs_retry_steps(self, reactions: List[Dict[str, Any]], balancer) -> None:
        balancer.input_validator.check(reactions)
        balancer.rb_method.run(reactions)
        balancer.rb_validator.check(reactions, override_unsolved=True)

    def _run_post_mcs_retry_steps(self, reactions: List[Dict[str, Any]], balancer) -> None:
        balancer.mcs_method.run(reactions)
        balancer.mcs_validator.check(reactions)
        balancer._Balancer__post_process(reactions)
        balancer.rb_method.run(reactions)
        balancer.mcs_validator.check(
            reactions,
            override_unsolved=True,
            override_issue_msg="Final reaction is unbalanced.",
        )
        balancer.conf_predictor.predict(
            reactions, threshold=balancer.confidence_threshold
        )

    def enumerate_mcs_candidates(
        self, reaction: Dict[str, Any], balancer
    ) -> List[LLMCandidate]:
        reaction_for_mcs = copy.deepcopy(reaction)
        reaction_for_mcs[self.issue_col] = ""
        condition_results = ensemble_mcs(
            [reaction_for_mcs],
            balancer.mcs_search.conditions,
            id_col=self.id_col,
            issue_col=self.issue_col,
            n_jobs=balancer.n_jobs,
        )

        candidates: List[LLMCandidate] = []
        for strategy_index, (condition, results) in enumerate(
            zip(balancer.mcs_search.conditions, condition_results)
        ):
            for result in results[: self.top_k_per_strategy]:
                if not result.get("mcs_results"):
                    continue
                mapping_summary = self._build_mapping_summary(result)
                candidates.append(
                    LLMCandidate(
                        strategy_index=strategy_index,
                        condition=condition,
                        reaction_id=reaction.get(self.id_col),
                        input_reaction=reaction[self.input_col],
                        mcs_results=result.get("mcs_results", []),
                        sorted_reactants=result.get("sorted_reactants", []),
                        issue=result.get(self.issue_col, ""),
                        mapping_summary=mapping_summary,
                    )
                )
        return candidates

    def filter_candidates(self, candidates: List[LLMCandidate]) -> List[LLMCandidate]:
        return [candidate for candidate in candidates if len(candidate.mcs_results) > 0]

    def deduplicate_candidates(self, candidates: List[LLMCandidate]) -> List[LLMCandidate]:
        unique_candidates: List[LLMCandidate] = []
        seen = set()
        for candidate in candidates:
            canonical_key = "|".join(candidate.mcs_results)
            if canonical_key in seen:
                continue
            seen.add(canonical_key)
            unique_candidates.append(candidate)
        return unique_candidates

    def select_or_generate_candidate(
        self,
        reaction: Dict[str, Any],
        candidates: List[LLMCandidate],
        logs: LLMPostprocessorLogs,
    ) -> Optional[Dict[str, Any]]:
        if len(candidates) == 0:
            logs.generation_triggered = True
            return self.generate_candidate(reaction, candidates, logs)

        if self.score_candidates_fn is None:
            logs.generation_triggered = True
            return self.generate_candidate(reaction, candidates, logs)

        payload = {
            "reaction_id": reaction.get(self.id_col),
            "input_reaction": reaction[self.input_col],
            "candidates": [candidate.to_payload() for candidate in candidates],
        }
        try:
            score_result = self.score_candidates_fn(payload)
        except LLMResponseParseError as exc:
            logs.score_raw_response = exc.raw_response
            logs.score_parse_error = str(exc)
            logs.generation_triggered = True
            return self.generate_candidate(reaction, candidates, logs)
        scores = score_result.get("scores", [])
        logs.score_raw_response = score_result.get("_raw_response")
        if len(scores) != len(candidates):
            raise ValueError("LLM score result must provide one score per candidate.")

        normalized_scores = [float(score) for score in scores]
        logs.candidate_scores = normalized_scores
        logs.top_score = max(normalized_scores) if normalized_scores else 0.0
        if normalized_scores and logs.top_score > self.score_threshold:
            top_index = normalized_scores.index(logs.top_score)
            return self._materialize_candidate(candidates[top_index])

        logs.generation_triggered = True
        return self.generate_candidate(reaction, candidates, logs)

    def diagnose_reaction(
        self,
        reaction: Dict[str, Any],
        candidates: List[LLMCandidate],
        logs: Optional[LLMPostprocessorLogs] = None,
    ) -> Dict[str, Any]:
        if not self.enable_two_stage_generation or self.diagnose_reaction_fn is None:
            return {
                "is_interpretable": True,
                "reaction_class": "",
                "imbalance_summary": "",
                "mechanistic_insight": "",
                "missing_reactants_smiles": "",
                "missing_products_smiles": "",
                "diagnosis_confidence": "",
            }
        payload = {
            "reaction_id": reaction.get(self.id_col),
            "input_reaction": reaction[self.input_col],
            "candidates": [candidate.to_payload() for candidate in candidates],
        }
        if logs is not None:
            logs.diagnosis_triggered = True
        try:
            diagnosis = self.diagnose_reaction_fn(payload)
        except LLMResponseParseError as exc:
            if logs is not None:
                logs.diagnosis_raw_response = exc.raw_response
                logs.diagnosis_parse_error = str(exc)
            raise
        if logs is not None:
            logs.diagnosis_raw_response = diagnosis.get("_raw_response")
        diagnosis = {k: v for k, v in diagnosis.items() if k != "_raw_response"}
        return self._normalize_diagnosis_result(diagnosis, logs)

    def generate_candidate(
        self,
        reaction: Dict[str, Any],
        candidates: List[LLMCandidate],
        logs: Optional[LLMPostprocessorLogs] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.generate_candidate_fn is None:
            return None
        diagnosis = self.diagnose_reaction(reaction, candidates, logs)
        payload = {
            "reaction_id": reaction.get(self.id_col),
            "input_reaction": reaction[self.input_col],
            "candidates": [candidate.to_payload() for candidate in candidates],
            "diagnosis": diagnosis,
        }
        try:
            generated = self.generate_candidate_fn(payload)
        except LLMResponseParseError as exc:
            if logs is not None:
                logs.generate_raw_response = exc.raw_response
                logs.generate_parse_error = str(exc)
            raise
        if generated is None:
            return None
        if logs is not None:
            logs.generate_raw_response = generated.get("_raw_response")
            logs.input_reasonable = generated.get("is_input_chemically_reasonable")
            logs.failure_reason = str(generated.get("failure_reason") or "")
            logs.atom_counting_scratchpad = str(generated.get("atom_counting_scratchpad") or "")
            logs.fragment_cutting_strategy = str(generated.get("fragment_cutting_strategy") or "")
        generated = {k: v for k, v in generated.items() if k != "_raw_response"}
        return self._normalize_generated_candidate(generated, reaction)

    def _materialize_candidate(self, candidate: LLMCandidate) -> Dict[str, Any]:
        largest_condition = ExtractMCS.get_largest_condition([candidate.to_payload()])[0]
        mcs_graph_result = find_graph_dict([largest_condition], n_jobs=1)[0]
        for key, value in largest_condition.items():
            mcs_graph_result[key] = value
        return mcs_graph_result

    def _normalize_diagnosis_result(
        self,
        diagnosis: Dict[str, Any],
        logs: Optional[LLMPostprocessorLogs] = None,
    ) -> Dict[str, Any]:
        normalized = copy.deepcopy(diagnosis)
        defaults = {
            "is_interpretable": True,
            "reaction_class": "",
            "imbalance_summary": "",
            "mechanistic_insight": "",
            "missing_reactants_smiles": "",
            "missing_products_smiles": "",
            "diagnosis_confidence": "",
        }
        for key, value in defaults.items():
            if key not in normalized:
                normalized[key] = value

        normalized["is_interpretable"] = bool(normalized.get("is_interpretable", True))
        for key in (
            "reaction_class",
            "imbalance_summary",
            "mechanistic_insight",
            "missing_reactants_smiles",
            "missing_products_smiles",
            "diagnosis_confidence",
        ):
            normalized[key] = str(normalized.get(key) or "")

        if logs is not None:
            logs.diagnosis_interpretable = normalized.get("is_interpretable")
            logs.diagnosis_reaction_class = normalized.get("reaction_class", "")
            logs.diagnosis_imbalance_summary = normalized.get("imbalance_summary", "")
            logs.diagnosis_mechanistic_insight = normalized.get("mechanistic_insight", "")
            logs.diagnosis_missing_reactants_smiles = normalized.get("missing_reactants_smiles", "")
            logs.diagnosis_missing_products_smiles = normalized.get("missing_products_smiles", "")
            logs.diagnosis_confidence = normalized.get("diagnosis_confidence", "")

        return normalized

    def _normalize_generated_candidate(
        self,
        generated: Dict[str, Any],
        reaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = copy.deepcopy(generated)

        if "predicted_reaction" not in normalized:
            raise ValueError("Generated response must include 'predicted_reaction'.")
        if "is_input_chemically_reasonable" not in normalized:
            raise ValueError("Generated response must include 'is_input_chemically_reasonable'.")
        if "failure_reason" not in normalized:
            raise ValueError("Generated response must include 'failure_reason'.")
        if "atom_counting_scratchpad" not in normalized:
            raise ValueError("Generated response must include 'atom_counting_scratchpad'.")
        if "fragment_cutting_strategy" not in normalized:
            raise ValueError("Generated response must include 'fragment_cutting_strategy'.")

        is_reasonable = normalized.get("is_input_chemically_reasonable")
        failure_reason = str(normalized.get("failure_reason") or "")
        allowed_failure_reasons = {
            "",
            "INVALID_FORMAT",
            "INVALID_VALENCE",
            "INVALID_BONDING",
            "MALFORMED_MOLECULE",
            "OTHER_INVALID_INPUT",
        }
        if failure_reason not in allowed_failure_reasons:
            failure_reason = "OTHER_INVALID_INPUT"
            normalized["failure_reason"] = failure_reason

        if isinstance(is_reasonable, bool) and not is_reasonable:
            normalized["predicted_reaction"] = ""
            normalized[self.issue_col] = (
                f"Input molecule judged unreasonable by LLM: {failure_reason}"
                if failure_reason
                else "Input molecule judged unreasonable by LLM."
            )
        else:
            normalized["failure_reason"] = ""

        if self.id_col not in normalized:
            normalized[self.id_col] = reaction.get(self.id_col)
        if self.issue_col not in normalized:
            normalized[self.issue_col] = ""

        return normalized

    def _build_mapping_summary(self, result: Dict[str, Any]) -> List[str]:
        mcs_results = result.get("mcs_results", [])
        sorted_reactants = result.get("sorted_reactants", [])
        return [
            "reactant={};mcs={}".format(reactant, mcs)
            for reactant, mcs in zip(sorted_reactants, mcs_results)
        ]

    def _merge_retry_result(
        self, original_reaction: Dict[str, Any], retry_reaction: Dict[str, Any]
    ) -> None:
        for key, value in retry_reaction.items():
            original_reaction[key] = value
