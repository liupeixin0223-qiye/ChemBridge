from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class LLMCandidate:
    strategy_index: int
    condition: Dict[str, Any]
    reaction_id: Any
    input_reaction: str
    mcs_results: List[str]
    sorted_reactants: List[str]
    issue: str = ""
    mapping_summary: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "strategy_index": self.strategy_index,
            "condition": self.condition,
            "reaction_id": self.reaction_id,
            "input_reaction": self.input_reaction,
            "mcs_results": self.mcs_results,
            "sorted_reactants": self.sorted_reactants,
            "mapping_summary": self.mapping_summary,
            "issue": self.issue,
        }


@dataclass
class LLMPostprocessorLogs:
    candidate_scores: List[float] = field(default_factory=list)
    top_score: float = 0.0
    diagnosis_triggered: bool = False
    generation_triggered: bool = False
    pre_mcs_retry_triggered: bool = False
    pre_mcs_retry_solved: bool = False
    pre_mcs_retry_reaction: str = ""
    pre_mcs_retry_issue: str = ""
    selection_path: str = ""
    generated_predicted_reaction: str = ""
    generated_reaction_after_validation: str = ""
    generated_issue_after_validation: str = ""
    final_success: bool = False
    input_reasonable: Optional[bool] = None
    force_pending_output: bool = False
    failure_reason: str = ""
    atom_counting_scratchpad: str = ""
    fragment_cutting_strategy: str = ""
    diagnosis_interpretable: Optional[bool] = None
    diagnosis_reaction_class: str = ""
    diagnosis_imbalance_summary: str = ""
    diagnosis_mechanistic_insight: str = ""
    diagnosis_missing_reactants_smiles: str = ""
    diagnosis_missing_products_smiles: str = ""
    diagnosis_confidence: str = ""
    diagnosis_raw_response: Optional[str] = None
    score_raw_response: Optional[str] = None
    generate_raw_response: Optional[str] = None
    diagnosis_parse_error: Optional[str] = None
    score_parse_error: Optional[str] = None
    generate_parse_error: Optional[str] = None
    pipeline_exception: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_scores": self.candidate_scores,
            "top_score": self.top_score,
            "diagnosis_triggered": self.diagnosis_triggered,
            "generation_triggered": self.generation_triggered,
            "pre_mcs_retry_triggered": self.pre_mcs_retry_triggered,
            "pre_mcs_retry_solved": self.pre_mcs_retry_solved,
            "pre_mcs_retry_reaction": self.pre_mcs_retry_reaction,
            "pre_mcs_retry_issue": self.pre_mcs_retry_issue,
            "selection_path": self.selection_path,
            "generated_predicted_reaction": self.generated_predicted_reaction,
            "generated_reaction_after_validation": self.generated_reaction_after_validation,
            "generated_issue_after_validation": self.generated_issue_after_validation,
            "final_success": self.final_success,
            "input_reasonable": self.input_reasonable,
            "force_pending_output": self.force_pending_output,
            "failure_reason": self.failure_reason,
            "atom_counting_scratchpad": self.atom_counting_scratchpad,
            "fragment_cutting_strategy": self.fragment_cutting_strategy,
            "diagnosis_interpretable": self.diagnosis_interpretable,
            "diagnosis_reaction_class": self.diagnosis_reaction_class,
            "diagnosis_imbalance_summary": self.diagnosis_imbalance_summary,
            "diagnosis_mechanistic_insight": self.diagnosis_mechanistic_insight,
            "diagnosis_missing_reactants_smiles": self.diagnosis_missing_reactants_smiles,
            "diagnosis_missing_products_smiles": self.diagnosis_missing_products_smiles,
            "diagnosis_confidence": self.diagnosis_confidence,
            "diagnosis_raw_response": self.diagnosis_raw_response,
            "score_raw_response": self.score_raw_response,
            "generate_raw_response": self.generate_raw_response,
            "diagnosis_parse_error": self.diagnosis_parse_error,
            "score_parse_error": self.score_parse_error,
            "generate_parse_error": self.generate_parse_error,
            "pipeline_exception": self.pipeline_exception,
        }
