import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from rdkit import Chem
from synrbl.llm.client import LLMResponseParseError
from synrbl.llm.species_client import SpeciesProposalClient

logger = logging.getLogger("synrbl")


@dataclass
class SpeciesProposal:
    missing_reactants_smiles: list[str]
    missing_products_smiles: list[str]
    raw_response: str = ""


def _build_bridge_round_summary(variant: Dict[str, Any], reaction_col: str, confidence_col: str, issue_col: str, solved_col: str) -> Dict[str, Any]:
    solved_by = str(variant.get("solved_by") or "").strip()
    confidence_raw = variant.get(confidence_col)
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    balanced = bool(LLMSpeciesBridge.analyze_reaction_balance(variant.get(reaction_col, "") or "").get("is_balanced", False))
    return {
        "round_label": "bridge_second_round",
        "reaction": variant.get(reaction_col),
        "balanced": balanced,
        "solved": bool(variant.get(solved_col, False)),
        "solved_by": solved_by or None,
        "confidence": confidence,
        "confidence_available": confidence is not None,
        "used_rule_based_only": solved_by == "rule-based",
        "used_mcs": solved_by == "mcs-based",
        "issue": variant.get(issue_col),
    }


class LLMSpeciesBridge:
    def __init__(
        self,
        id_col: str,
        reaction_col: str,
        solved_col: str = "solved",
        issue_col: str = "issue",
        confidence_col: str = "confidence",
        confidence_threshold: float = 0.5,
        propose_side_species_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        log_col: str = "llm_species_bridge",
        max_workers: int = 20,
    ):
        self.id_col = id_col
        self.reaction_col = reaction_col
        self.solved_col = solved_col
        self.issue_col = issue_col
        self.confidence_col = confidence_col
        self.confidence_threshold = confidence_threshold
        self.propose_side_species_fn = propose_side_species_fn
        self.log_col = log_col
        self.max_workers = max(1, int(max_workers))

    @classmethod
    def from_moonshot(
        cls,
        id_col: str,
        reaction_col: str,
        confidence_threshold: float = 0.5,
        api_key_env: str = "MOONSHOT_API_KEY",
        base_url: str = "https://api.moonshot.cn/v1/chat/completions",
        model: str = "kimi-k2.5",
        max_workers: int = 20,
        thinking_enabled: bool = False,
    ) -> "LLMSpeciesBridge":
        client = SpeciesProposalClient(
            api_key_env=api_key_env,
            base_url=base_url,
            score_model=model,
            generate_model=model,
            thinking_enabled=thinking_enabled,
        )
        return cls(
            id_col=id_col,
            reaction_col=reaction_col,
            confidence_threshold=confidence_threshold,
            propose_side_species_fn=client.propose_side_species,
            max_workers=max_workers,
        )

    def apply(self, reactions: List[Dict[str, Any]], balancer) -> List[Dict[str, Any]]:
        if self.propose_side_species_fn is None:
            return reactions

        triggered_reactions: list[Dict[str, Any]] = []
        request_stats = {
            "max_workers": self.max_workers,
            "requested_count": 0,
            "completed_count": 0,
            "parse_error_count": 0,
            "request_error_count": 0,
        }

        for reaction in reactions:
            is_solved = reaction.get(self.solved_col, False)
            conf = reaction.get(self.confidence_col, 0.0)
            if conf is None:
                conf = 0.0
            else:
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 0.0

            if is_solved and conf >= self.confidence_threshold:
                continue

            reaction[self.log_col] = {
                "triggered": True,
                "confidence_threshold": self.confidence_threshold,
                "entry_id": reaction.get(self.id_col),
                "pre_bridge_solved": bool(is_solved),
                "pre_bridge_confidence": conf,
                "pre_bridge_reaction": reaction.get(self.reaction_col),
                "pre_bridge_issue": reaction.get(self.issue_col, ""),
                "pre_bridge_solved_by": reaction.get("solved_by"),
                "fallback_applied": False,
                "fallback_reason": None,
                "input_issue": reaction.get(self.issue_col, ""),
                "cleaned_initial_reaction": None,
                "calculated_imbalance": None,
                "payload": None,
                "proposal_raw_response": None,
                "proposal_parse_error": None,
                "proposal": None,
                "query_resolution": None,
                "tested_variants": [],
                "variant_evaluations": [],
                "accepted_variant": None,
                "accepted_variant_evaluation": None,
                "accepted_variant_balance_analysis": None,
                "pre_bridge_fallback_suggested": False,
                "pre_bridge_fallback_payload": None,
                "failure_stage": None,
                "failure_reason": None,
                "final_status": "pending",
                "request_stats": {
                    "phase": "species_bridge",
                    "max_workers": self.max_workers,
                    "completed": False,
                },
            }

            clean_rxn = reaction.get("cleaned_initial_reaction") or reaction.get("input_reaction", "")
            exact_imbalance = self._calculate_atom_imbalance(clean_rxn)
            payload = {
                "reaction_id": reaction.get(self.id_col),
                "input_reaction": clean_rxn,
                "current_reaction": clean_rxn,
                "calculated_imbalance": exact_imbalance,
                "issue": reaction.get(self.issue_col, ""),
                "task_note": (
                    "Bridge task: infer the most likely missing large side-species from the original reaction and exact atom imbalance. "
                    "Prefer a decisive best guess over leaving the result empty."
                ),
            }
            reaction[self.log_col]["cleaned_initial_reaction"] = clean_rxn
            reaction[self.log_col]["calculated_imbalance"] = exact_imbalance
            reaction[self.log_col]["payload"] = copy.deepcopy(payload)
            triggered_reactions.append(reaction)

        request_stats["requested_count"] = len(triggered_reactions)
        proposal_results: dict[str, dict[str, Any]] = {}
        if triggered_reactions:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {
                    executor.submit(self.propose_side_species_fn, reaction[self.log_col]["payload"]): reaction
                    for reaction in triggered_reactions
                }
                for future in as_completed(future_map):
                    reaction = future_map[future]
                    tracking_key = str(reaction.get(self.id_col))
                    try:
                        proposal_results[tracking_key] = {"status": "ok", "payload": future.result()}
                        request_stats["completed_count"] += 1
                    except LLMResponseParseError as exc:
                        proposal_results[tracking_key] = {
                            "status": "parse_error",
                            "raw_response": exc.raw_response,
                            "error": str(exc),
                        }
                        request_stats["parse_error_count"] += 1
                    except Exception as exc:
                        proposal_results[tracking_key] = {"status": "request_error", "error": str(exc)}
                        request_stats["request_error_count"] += 1

        for reaction in triggered_reactions:
            reaction[self.log_col]["request_pool_stats"] = copy.deepcopy(request_stats)
            reaction[self.log_col]["request_stats"]["completed"] = True
            proposal_result = proposal_results.get(str(reaction.get(self.id_col)), {"status": "request_error", "error": "Missing proposal result."})

            if proposal_result["status"] == "parse_error":
                reaction[self.log_col]["proposal_raw_response"] = proposal_result.get("raw_response")
                reaction[self.log_col]["proposal_parse_error"] = proposal_result.get("error")
                reaction[self.log_col]["failure_stage"] = "proposal_parse"
                reaction[self.log_col]["failure_reason"] = proposal_result.get("error")
                reaction[self.log_col]["final_status"] = "proposal_parse_error"
                continue
            if proposal_result["status"] != "ok":
                reaction[self.log_col]["proposal_parse_error"] = proposal_result.get("error")
                reaction[self.log_col]["failure_stage"] = "proposal_request"
                reaction[self.log_col]["failure_reason"] = proposal_result.get("error")
                reaction[self.log_col]["final_status"] = "proposal_exception"
                continue

            proposal_payload = proposal_result["payload"]
            proposal = self._normalize_proposal(proposal_payload)
            reaction[self.log_col]["proposal_raw_response"] = proposal.raw_response
            reaction[self.log_col]["proposal"] = {
                "missing_reactants_smiles": proposal.missing_reactants_smiles,
                "missing_products_smiles": proposal.missing_products_smiles,
            }

            variants, resolution_log = self._build_variants(reaction, proposal)
            reaction[self.log_col]["query_resolution"] = resolution_log
            reaction[self.log_col]["tested_variants"] = [v[self.reaction_col] for v in variants]
            reaction[self.log_col]["post_llm_variant_count"] = len(variants)
            if variants:
                reaction["bridge_candidate_reaction"] = variants[0].get(self.reaction_col)
            if resolution_log.get("error"):
                reaction[self.log_col]["failure_stage"] = "variant_build"
                reaction[self.log_col]["failure_reason"] = resolution_log.get("error")
                reaction[self.log_col]["final_status"] = "variant_build_error"
                continue
            if not variants:
                reaction[self.log_col]["failure_stage"] = "variant_build"
                reaction[self.log_col]["failure_reason"] = "No variant could be built from resolved side-species queries."
                reaction[self.log_col]["final_status"] = "no_variant_built"
                continue

            accepted, accepted_evaluation, variant_evaluations = self._validate_variants(variants, balancer)
            reaction[self.log_col]["variant_evaluations"] = variant_evaluations
            if accepted is None:
                fallback_payload = self._build_pre_bridge_fallback_payload(reaction)
                reaction[self.log_col]["pre_bridge_fallback_payload"] = copy.deepcopy(fallback_payload)
                reaction[self.log_col]["pre_bridge_fallback_suggested"] = bool(fallback_payload)
                reaction[self.log_col]["fallback_applied"] = False
                if fallback_payload:
                    reaction[self.log_col]["fallback_reason"] = (
                        "Suggested fallback to pre-bridge low-confidence SynRBL result because bridge produced no accepted audited variant."
                    )
                    reaction[self.log_col]["failure_stage"] = "variant_validation"
                    reaction[self.log_col]["failure_reason"] = None
                    reaction[self.log_col]["final_status"] = "fallback_to_pre_bridge_suggested"
                    continue

                reaction[self.log_col]["failure_stage"] = "variant_validation"
                reaction[self.log_col]["failure_reason"] = (
                    "All variants were evaluated but none produced an accepted audited final reaction."
                    if variant_evaluations
                    else "No variant evaluation result was produced."
                )
                reaction[self.log_col]["final_status"] = (
                    "all_variants_rejected" if variant_evaluations else "variant_evaluation_missing"
                )
                continue

            reaction[self.log_col]["accepted_variant"] = accepted[self.reaction_col]
            reaction[self.log_col]["accepted_variant_evaluation"] = copy.deepcopy(accepted_evaluation)
            accepted_balance = self.analyze_reaction_balance(
                accepted.get(self.reaction_col, "") or accepted.get("input_reaction", "")
            )
            reaction[self.log_col]["accepted_variant_balance_analysis"] = accepted_balance
            candidate_confidence = accepted.get("workflow_confidence", accepted.get(self.confidence_col))
            try:
                candidate_confidence_value = float(candidate_confidence) if candidate_confidence not in {None, ""} else None
            except (TypeError, ValueError):
                candidate_confidence_value = None
            candidate_solved_by = str(accepted.get("solved_by") or "").strip()
            candidate_bridge_route = str(accepted.get("workflow_route") or "").strip()
            reaction[self.log_col]["accepted_variant_workflow_confidence"] = candidate_confidence_value
            reaction[self.log_col]["accepted_variant_solved_by"] = candidate_solved_by
            reaction[self.log_col]["accepted_variant_workflow_route"] = candidate_bridge_route

            if candidate_solved_by == "rule-based" and candidate_confidence_value is None:
                bridge_log_snapshot = copy.deepcopy(reaction.get(self.log_col))
                accepted_without_bridge_log = copy.deepcopy(accepted)
                accepted_without_bridge_log.pop(self.log_col, None)
                reaction.update(accepted_without_bridge_log)
                reaction[self.log_col] = bridge_log_snapshot
                reaction["bridge_best_reaction"] = reaction.get(self.reaction_col)
                reaction["workflow_confidence"] = 1.5
                reaction["workflow_confidence_origin"] = "bridge"
                reaction[self.log_col]["final_status"] = "accepted_balanced"
                continue

            if candidate_confidence_value is None or candidate_confidence_value < self.confidence_threshold:
                reaction[self.log_col]["final_status"] = "accepted_untrusted_low_confidence"
                reaction[self.log_col]["failure_stage"] = "variant_validation"
                reaction[self.log_col]["failure_reason"] = "Bridge accepted a balanced variant without sufficient workflow confidence."
                continue

            bridge_log_snapshot = copy.deepcopy(reaction.get(self.log_col))
            accepted_without_bridge_log = copy.deepcopy(accepted)
            accepted_without_bridge_log.pop(self.log_col, None)
            reaction.update(accepted_without_bridge_log)
            reaction[self.log_col] = bridge_log_snapshot
            reaction["bridge_best_reaction"] = reaction.get(self.reaction_col)
            reaction["workflow_confidence"] = candidate_confidence_value
            reaction["workflow_confidence_origin"] = "bridge"
            reaction[self.log_col]["final_status"] = "accepted_balanced"

        return reactions

    def _normalize_proposal(self, proposal_dict: Dict[str, Any]) -> SpeciesProposal:
        return SpeciesProposal(
            missing_reactants_smiles=self._normalize_smiles_list(
                proposal_dict.get("missing_reactants_smiles")
            ),
            missing_products_smiles=self._normalize_smiles_list(
                proposal_dict.get("missing_products_smiles")
            ),
            raw_response=str(proposal_dict.get("_raw_response") or ""),
        )

    def _build_pre_bridge_fallback_payload(self, reaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pre_bridge_solved = bool(reaction.get(self.log_col, {}).get("pre_bridge_solved", False))
        pre_bridge_confidence = reaction.get(self.log_col, {}).get("pre_bridge_confidence")
        pre_bridge_reaction = reaction.get(self.log_col, {}).get("pre_bridge_reaction")

        try:
            pre_bridge_confidence = float(pre_bridge_confidence) if pre_bridge_confidence is not None else 0.0
        except (TypeError, ValueError):
            pre_bridge_confidence = 0.0

        if (
            not pre_bridge_solved
            or pre_bridge_confidence >= self.confidence_threshold
            or not pre_bridge_reaction
        ):
            return None

        balance_check = self.analyze_reaction_balance(str(pre_bridge_reaction))
        if not balance_check.get("is_balanced", False):
            return None

        return {
            "reaction": pre_bridge_reaction,
            "confidence": pre_bridge_confidence,
            "issue": reaction.get(self.log_col, {}).get("pre_bridge_issue", ""),
            "solved_by": reaction.get(self.log_col, {}).get("pre_bridge_solved_by"),
            "balance_analysis": balance_check,
        }

    @staticmethod
    def analyze_reaction_balance(rxn_smiles: str) -> Dict[str, Any]:
        try:
            import collections
            from rdkit import Chem

            if ">>" not in rxn_smiles:
                return {
                    "is_balanced": False,
                    "imbalance_text": "Invalid reaction format",
                    "missing_on_products": {},
                    "missing_on_reactants": {},
                    "reactant_counts": {},
                    "product_counts": {},
                    "error": "Invalid reaction format",
                }

            reactants, products = rxn_smiles.split(">>", 1)

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

            r_counts = get_counts(reactants)
            p_counts = get_counts(products)
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
                    "Missing on Products: " + " ".join(f"{el}:{count}" for el, count in missing_on_products.items())
                )
            if missing_on_reactants:
                parts.append(
                    "Missing on Reactants: " + " ".join(f"{el}:{count}" for el, count in missing_on_reactants.items())
                )
            return {
                "is_balanced": not missing_on_products and not missing_on_reactants,
                "imbalance_text": "; ".join(parts) if parts else "Exactly Balanced",
                "missing_on_products": missing_on_products,
                "missing_on_reactants": missing_on_reactants,
                "reactant_counts": r_counts,
                "product_counts": p_counts,
                "error": "",
            }
        except Exception as exc:
            return {
                "is_balanced": False,
                "imbalance_text": f"Error computing exact imbalance: {str(exc)}",
                "missing_on_products": {},
                "missing_on_reactants": {},
                "reactant_counts": {},
                "product_counts": {},
                "error": str(exc),
            }

    @classmethod
    def _calculate_atom_imbalance(cls, rxn_smiles: str) -> str:
        return str(cls.analyze_reaction_balance(rxn_smiles).get("imbalance_text", ""))

    @classmethod
    def _normalize_smiles_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        for item in values:
            item_str = str(item or "").strip()
            if not item_str:
                continue
            if cls._is_valid_smiles(item_str):
                normalized.append(item_str)
        return normalized

    @staticmethod
    def _is_valid_smiles(value: str) -> bool:
        if not value:
            return False
        try:
            return Chem.MolFromSmiles(value) is not None
        except Exception:
            return False

    @classmethod
    def _split_candidate_tokens(cls, query: str) -> list[str]:
        normalized = str(query or "").replace("；", ";")
        parts = []
        for chunk in normalized.split(";"):
            token = chunk.strip()
            if token:
                parts.append(token)
        return parts or ([str(query).strip()] if str(query).strip() else [])

    @classmethod
    def name_to_smiles_plugin(cls, chemical_name: str) -> str:
        raw = str(chemical_name or "").strip()
        name = raw.lower()
        if not name:
            return ""
        if cls._is_valid_smiles(raw):
            return raw
        local_cache = {
            "water": "O", "hydrogen": "[H][H]", "oxygen": "O=O",
            "carbon dioxide": "O=C=O", "carbon monoxide": "[C-]#[O+]",
            "nitrogen": "N#N", "ammonia": "N", "methane": "C",
            "ethylene": "C=C", "isobutene": "C=C(C)C", "methanol": "CO",
            "ethanol": "CCO", "t-butanol": "CC(C)(C)O", "acetic acid": "CC(=O)O",
            "acetone": "CC(C)=O", "formic acid": "O=CO", "dimethylamine": "CNC",
            "bromoform": "BrC(Br)Br", "chloroform": "ClC(Cl)Cl",
            "hydrogen chloride": "Cl", "hydrochloric acid": "Cl",
            "hydrogen bromide": "Br", "hydrogen iodide": "I",
            "1-methyl-3-hydroxy-1H-pyrazole-4-carbaldehyde": "Cn1cc(C=O)c(O)n1",
            "1H-1,2,3-benzotriazin-4(3H)-one": "O=c1[nH]nnc2ccccc12",
            "diisopropylethylamine hydrochloride": "CCN(C(C)C)C(C)C.Cl",
            "4-chloro-N-(4-iodo-3-methylphenyl)nicotinamide": "Cc1cc(NC(=O)c2cnccc2Cl)ccc1I",
            "2-(hydroxymethyl)thiophene-5-boronic acid": "OB(O)c1ccc(CO)s1",
            "3-bromo-N-(quinolin-6-yl)isoquinolin-5-amine": "Brc1cc2c(cn1)c(Nc3ccc4ncccc4c3)ccc2",
            "1-(2-fluorophenyl)-2-isocyanatoethyl 2,2,2-trifluoroethyl carbonate": "O=C=NCC(OC(=O)OCC(F)(F)F)c1ccccc1F",
            "2-phenyl-2-(2-hydroxyethyl)ethylamine": "NCC(CCO)c1ccccc1",
            "1,2-dihydrophthalazine-1,2-dione": "O=c1[nH][nH]c(=O)c2ccccc12",
            "1,2-benzenedicarboxylic acid hydrazide": "O=c1[nH][nH]c(=O)c2ccccc12",
            "triethylamine hydrochloride": "CCN(CC)CC.Cl",
            "4-chlorobutanol": "ClCCCCO",
            "sodium chloride": "[Na+].[Cl-]",
            "2-nitroaniline": "Nc1ccccc1[N+](=O)[O-]",
            "Sodium sulfate": "[Na+].[Na+].[O-]S(=O)(=O)[O-]",
            "N-succinimidyl N,N-diethylcarbamate": "CCN(CC)C(=O)ON1C(=O)CCC1=O",
            "4-nitrophenol": "Oc1ccc([N+](=O)[O-])cc1",
            "manganese dioxide": "O=[Mn]=O",
            "potassium dihydrogen phosphate": "[K+].OP(=O)(O)[O-]",
            "fluoride anion": "[F-]",
            "methyl (dimethoxyphosphoryl)thioacetate": "CSC(=O)CP(=O)(OC)OC",
            "1-methyl-5-(hydroxymethyl)-1H-1,2,3-triazole-4-carboxylic acid": "Cn1nnc(C(=O)O)c1CO",
            "chloroformaldehyde": "O=CCl",
            "hexafluorophosphate anion": "F[P-](F)(F)(F)(F)F",
            "2-chloro-1-(chloromethyl)ethanone": "ClCC(=O)CCl",
            "N-hydroxy-2-(2-chloroacetamido)acetamide": "ClCC(=O)NCC(=O)NO",
            "4,5-diphenyl-1H-phosphole": "[PH]1C=CC(c2ccccc2)=C1c3ccccc3",
            "potassium 4-fluorophenoxide": "[K+].Fc1ccc([O-])cc1",
            "N-lithium diisopropylamine": "[Li+].CC(C)[N-]C(C)C",
            "1,3-dimethyl-3-(3-(methylamino)propyl)urea": "CNC(=O)N(C)CCCNC",
            "2-chloroethyl 2-(2-hydroxyethoxy)acetate": "ClCCOC(=O)COCCO",
            "1,2-dihydrophthalazine-1,4-dione": "O=c1[nH][nH]c(=O)c2ccccc12",
            "1,2-dihydrophthalazinedione": "O=c1[nH][nH]c(=O)c2ccccc12",
            "methyl 1-(4-methylphenyl)sulfonyl-1,2,3,4-tetrahydroquinoline-4-carboxylate": "COC(=O)C1CCN(S(=O)(=O)c2ccc(C)cc2)c3ccccc13",
            "1,1'-azobis(N-piperidine)": "C1CCN(CC1)N=NN2CCCCC2",
            "chloroboronic acid": "OB(O)Cl",
            "dihydroxybromoborane": "OB(O)Br",
            "bromodihydroxyborane": "OB(O)Br",
            "zinc chloride iodide": "Cl[Zn]I",
            "Fluoromagnesium(1+)": "F[Mg+]",
            "bromozinc chloride": "Cl[Zn]Br",
            "bromomagnesium(1+)": "Br[Mg+]",
            "bromozinc(1+)": "Br[Zn+]",
            "chloromagnesium": "Cl[Mg]",
            "zinc monochloride": "Cl[Zn]",
            "chlorozinc": "Cl[Zn]",
            "chlorodihydroxyborane": "OB(O)Cl",
            "dihydroxy(chloro)borane": "OB(O)Cl",
            "dibromomethylamine": "CN(Br)Br",
            "magnesium bromide dimethylamide": "CN(C)[Mg]Br",
            "tri(o-tolyl)phosphine hydrobromide": "Cc1ccccc1P(c2ccccc2C)c3ccccc3C.Br",
            "N,N-Diisopropylethylamine hydrobromide": "CCN(C(C)C)C(C)C.Br",
            "2,6-Lutidinium hexafluorophosphate": "Cc1cccc(C)[nH+]1.F[P-](F)(F)(F)(F)F",
            "2-Carboxybenzohydrazide": "O=C(NN)c1ccccc1C(=O)O",
            "1-chloromethyl-1,4-diazoniabicyclo[2.2.2]octane bis(tetrafluoroborate)": "ClC[N+]12CC[NH+](CC1)CC2.[B-](F)(F)(F)F.[B-](F)(F)(F)F",
            "Aminodibromomethane": "NC(Br)Br",
            "2,4-dimethyl-3-hexen-2-yl hydroperoxide": "CCC(C)=CC(C)(C)OO",
            "2,3-dichloro-5,6-dicyano-1,4-hydroquinone": "N#Cc1c(O)c(Cl)c(Cl)c(O)c1C#N",
            "B(OH)2Cl": "OB(O)Cl",
            "dihydroxy(iodo)borane": "OB(O)I",
            "dihydroxyiodoborane": "OB(O)I",
            "iododihydroxyborane": "OB(O)I",
            "dihydroxychloroborane": "OB(O)Cl",
            "dihydroxyboron chloride": "OB(O)Cl",
            "Dihydroxychloroborane": "OB(O)Cl",
            "Boron chloride dihydroxide": "OB(O)Cl",
            "zinc monoiodide": "I[Zn]",
            "zinc iodide(1+)": "I[Zn+]",
            "iodozinc(1+)": "I[Zn+]",
            "chlorozinc(1+)": "Cl[Zn+]",
            "ZnCl": "Cl[Zn]",
            "Zinc chloride (ZnCl)": "Cl[Zn]",
            "chloromagnesium(1+)": "Cl[Mg+]",
            "magnesium(1+) chloride": "Cl[Mg+]",
            "zinc monobromide": "Br[Zn]",
            "Bromozinc": "Br[Zn]",
            "zinc bromide cation": "Br[Zn+]",
            "CuBr": "Br[Cu]",
            "bromomagnesium": "Br[Mg]",
            "magnesium bromide cation": "Br[Mg+]",
            "bromochlorozinc": "Cl[Zn]Br",
            "methyl iodide": "CI",
            "tert-butanol": "CC(C)(C)O",
            "dimethyl sulfide": "CSC",
            "formaldehyde": "C=O",
            "ethylene oxide": "C1CO1",
            "allylamine hydrochloride": "C=CCN.Cl",
            "chloride ion": "[Cl-]",
            "methoxide": "C[O-]",
            "benzoic acid": "O=C(O)c1ccccc1",
            "succinimide": "O=C1CCC(=O)N1",
            "trimethylsilyl chloride": "C[Si](C)(C)Cl",
            "triphenylphosphine oxide": "O=P(c1ccccc1)(c2ccccc2)c3ccccc3",
            "ethene": "C=C",
            "iodomethane": "CI",
            "methoxide ion": "C[O-]",
            "Chloride": "[Cl-]",
            "triethylammonium chloride": "CCN(CC)CC.Cl",
            "toluene": "Cc1ccccc1",
            "butane": "CCCC",
            "dimethyl carbonate": "COC(=O)OC",
            "trimethylsilanol": "C[Si](C)(C)O",
            "methanesulfonic acid": "CS(=O)(=O)O",
            "1-bromobutane": "CCCCBr",
            "lithium bromide": "[Li+].[Br-]"
        }
        if name in local_cache:
            return local_cache[name]
        import urllib.parse
        import urllib.request
        try:
            safe_name = urllib.parse.quote(name)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/CanonicalSMILES/TXT"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.read().decode("utf-8").strip()
        except Exception as exc:
            logger.warning(f"PubChem 翻译 '{chemical_name}' 失败: {exc}")
            return ""

    @classmethod
    def resolve_pubchem_queries(cls, queries: List[str]) -> Dict[str, Any]:
        resolved_smiles = []
        items = []
        for query in queries:
            tokens = cls._split_candidate_tokens(query)
            token_results = []
            token_smiles = []
            for token in tokens:
                smi = cls.name_to_smiles_plugin(token)
                token_results.append({"token": token, "resolved_smiles": smi, "resolved": bool(smi)})
                if smi:
                    token_smiles.append(smi)
            combined = ".".join(token_smiles)
            items.append(
                {
                    "query": query,
                    "tokens": token_results,
                    "resolved_smiles": combined,
                    "resolved": bool(combined),
                }
            )
            if combined:
                resolved_smiles.append(combined)
        return {
            "queries": queries,
            "items": items,
            "resolved_smiles": ".".join(resolved_smiles),
            "resolved_count": len(resolved_smiles),
            "all_resolved": len(items) == len(resolved_smiles),
        }

    def _build_variants(
        self, reaction: Dict[str, Any], proposal: SpeciesProposal
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        base_reaction = reaction.get("cleaned_initial_reaction", reaction.get("input_reaction", ""))
        parts = base_reaction.split(">>", 1)
        if len(parts) != 2:
            return [], {
                "base_reaction": base_reaction,
                "error": "Invalid reaction format",
                "reactant_resolution": None,
                "product_resolution": None,
            }
        reactants, products = parts[0], parts[1]
        clean_reactants = [smi for smi in proposal.missing_reactants_smiles if self._is_valid_smiles(smi)]
        clean_products = [smi for smi in proposal.missing_products_smiles if self._is_valid_smiles(smi)]
        clean_add_r = ".".join(clean_reactants)
        clean_add_p = ".".join(clean_products)
        resolution_log = {
            "base_reaction": base_reaction,
            "reactant_smiles": proposal.missing_reactants_smiles,
            "product_smiles": proposal.missing_products_smiles,
            "resolved_reactant_smiles": clean_add_r,
            "resolved_product_smiles": clean_add_p,
            "reactant_smiles_count": len(proposal.missing_reactants_smiles),
            "product_smiles_count": len(proposal.missing_products_smiles),
            "invalid_reactant_smiles": [smi for smi in proposal.missing_reactants_smiles if not self._is_valid_smiles(smi)],
            "invalid_product_smiles": [smi for smi in proposal.missing_products_smiles if not self._is_valid_smiles(smi)],
        }

        if not clean_add_r and not clean_add_p:
            resolution_log["variant_count"] = 0
            resolution_log["variants"] = []
            resolution_log["error"] = "No valid side-species SMILES produced a non-empty variant."
            return [], resolution_log

        if clean_add_r and clean_add_p:
            new_rxn = f"{reactants}.{clean_add_r}>>{products}.{clean_add_p}"
        elif clean_add_r:
            new_rxn = f"{reactants}.{clean_add_r}>>{products}"
        else:
            new_rxn = f"{reactants}>>{products}.{clean_add_p}"

        variant = copy.deepcopy(reaction)
        variant["input_reaction"] = new_rxn
        variant[self.reaction_col] = new_rxn
        variant[self.solved_col] = False
        variant[self.issue_col] = ""
        variant["solved_by"] = ""
        variant["species_bridge_variant_source"] = {
            "added_reactants_smiles": clean_add_r,
            "added_products_smiles": clean_add_p,
            "reactant_smiles": proposal.missing_reactants_smiles,
            "product_smiles": proposal.missing_products_smiles,
        }
        if self.confidence_col in variant:
            variant[self.confidence_col] = 0.0
        for key in ["mcs", "rules", "unbalance_col", "carbon_balance_check"]:
            variant.pop(key, None)

        resolution_log["variant_count"] = 1
        resolution_log["variants"] = [new_rxn]
        return [variant], resolution_log

    def _validate_variants(self, variants: List[Dict[str, Any]], balancer) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        best_variant = None
        best_evaluation = None
        highest_score = -1.0
        evaluations: List[Dict[str, Any]] = []
        for variant in variants:
            evaluation = {
                "reaction": variant.get(self.reaction_col),
                "added_reactants_smiles": variant.get("species_bridge_variant_source", {}).get("added_reactants_smiles"),
                "added_products_smiles": variant.get("species_bridge_variant_source", {}).get("added_products_smiles"),
                "reactant_smiles": variant.get("species_bridge_variant_source", {}).get("reactant_smiles"),
                "product_smiles": variant.get("species_bridge_variant_source", {}).get("product_smiles"),
                "status": "pending",
                "solved": False,
                "confidence": None,
                "confidence_available": False,
                "issue": "",
                "combined_score": None,
                "selected_as_best": False,
                "audit_passed": False,
                "balance_analysis": None,
                "exception": None,
                "solved_by": None,
                "round_summary": None,
            }
            try:
                balancer.run_core_pipeline(
                    [variant], allow_low_confidence_solved=True
                )
                is_solved = variant.get(self.solved_col, False)
                conf_raw = variant.get(self.confidence_col)
                confidence_available = conf_raw is not None
                try:
                    conf = float(conf_raw) if conf_raw is not None else None
                except (TypeError, ValueError):
                    conf = None
                    confidence_available = False
                current_score = (conf if conf is not None else 0.0) + (10.0 if is_solved else 0.0)
                balance_analysis = self.analyze_reaction_balance(
                    variant.get(self.reaction_col, "") or variant.get("input_reaction", "")
                )
                audit_passed = bool(balance_analysis.get("is_balanced", False))
                round_summary = _build_bridge_round_summary(
                    variant,
                    self.reaction_col,
                    self.confidence_col,
                    self.issue_col,
                    self.solved_col,
                )
                variant.setdefault("workflow_stage_summary", {})["bridge_second_round_summary"] = copy.deepcopy(round_summary)
                evaluation.update({
                    "status": "evaluated",
                    "solved": bool(is_solved),
                    "confidence": conf,
                    "confidence_available": confidence_available,
                    "issue": str(variant.get(self.issue_col, "") or ""),
                    "combined_score": current_score,
                    "solved_by": variant.get("solved_by"),
                    "rules": copy.deepcopy(variant.get("rules", [])),
                    "audit_passed": audit_passed,
                    "balance_analysis": balance_analysis,
                    "round_summary": round_summary,
                })
                if audit_passed and (best_variant is None or current_score > highest_score):
                    highest_score = current_score
                    best_variant = variant
                    best_evaluation = evaluation
            except Exception as exc:
                logger.exception("Species bridge SynRBL re-run failed for variant.")
                evaluation.update({"status": "exception", "exception": str(exc)})
            evaluations.append(evaluation)
        if best_variant is not None:
            for item in evaluations:
                item["selected_as_best"] = item is best_evaluation
            best_variant["workflow_route"] = "llm-species-bridge"
            return best_variant, copy.deepcopy(best_evaluation), evaluations
        return None, None, evaluations
