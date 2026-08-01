import copy
import logging
import traceback

from synrbl.preprocess import preprocess, input_sanitize_check
from synrbl.postprocess import Validator
from synrbl.rule_based import RuleBasedMethod
from synrbl.mcs_search import MCSSearch
from synrbl.SynMCSImputer.mcs_based_method import MCSBasedMethod
from synrbl.SynChemImputer.post_process import PostProcess
from synrbl.SynChemImputer.molecule_standardizer import MoleculeStandardizer
from synrbl.confidence_prediction import ConfidencePredictor
from synrbl.llm_postprocessor import LLMPostprocessor
from synrbl.llm_species_bridge import LLMSpeciesBridge
from synrbl.llm_fallback_postprocessor import LLMFallbackPostprocessor
from synrbl.SynUtils.batching import Dataset, DataLoader, CacheManager

logger = logging.getLogger("synrbl")


def merge_stats(stats, new_stats):
    if stats is None:
        return
    if new_stats is None:
        return
    stats_keys = list(stats.keys())
    new_stats_keys = list(new_stats.keys())
    for k in stats.keys():
        if k in new_stats_keys:
            stats[k] += new_stats[k]
    for k, v in new_stats.items():
        if k not in stats_keys:
            stats[k] = v


class Balancer:
    def __init__(
        self,
        id_col="id",
        reaction_col="reaction",
        confidence_threshold=0,
        n_jobs=-1,
        batch_size=None,
        cache=False,
        cache_dir: str | None = "./cache",
        use_default_reduction: bool = False,
        llm_postprocessor: LLMPostprocessor | None = None,
        llm_species_bridge: LLMSpeciesBridge | None = None,
        llm_fallback_postprocessor: LLMFallbackPostprocessor | None = None,
        enable_advanced_scoring: bool = True,
        enable_multi_fragment: bool = True,
    ):
        self.__reaction_col = reaction_col
        self.__id_col = id_col
        self.__solved_col = "solved"
        self.__solved_by_col = "solved_by"
        self.__mcs_data_col = "mcs"
        self.__input_col = "input_reaction"
        self.__confidence_col = "confidence"
        self.__unbalance_col = "unbalance_col"
        self.__carbon_balance_col = "carbon_balance_check"
        self.__rules_col = "rules"
        self.__issue_col = "issue"
        self.__n_jobs = n_jobs

        self.use_default_reduction = use_default_reduction
        self.llm_postprocessor = llm_postprocessor
        self.llm_species_bridge = llm_species_bridge
        self.llm_fallback_postprocessor = llm_fallback_postprocessor

        self.remove_aam = True
        self.enable_advanced_scoring = enable_advanced_scoring
        self.enable_multi_fragment = enable_multi_fragment
        self.batch_size = batch_size
        self.cache = cache
        self.cache_dir = cache_dir
        self.columns = [
            self.__id_col,
            self.__input_col,
            reaction_col,
            self.__solved_col,
            self.__solved_by_col,
            self.__confidence_col,
            self.__rules_col,
            self.__issue_col,
            "_synrbl_internal_id",
            "original_row_index",
            "original_reaction",
            "preprocess_status",
            "standardize_status",
            "processable",
            "parse_issue",
            "can_parse_reaction",
            "neutralize_status",
            "deionize_status",
            "pipeline_failed",
            "pipeline_failure_stage",
            "llm_postprocess",
            "llm_species_bridge",
            "llm_fallback_postprocess",
            "prebalance_check",
            "pre_llm_reaction",
            "pre_llm_solved",
            "pre_llm_solved_by",
            "pre_llm_confidence",
            "pre_llm_issue",
            "pre_llm_rules",
            "bridge_best_reaction",
            "bridge_candidate_reaction",
            "fallback_generated_reaction",
            "wrong_reactions",
            "has_wrong_reactions",
            "expected_reaction",
            "workflow_stage_summary",
        ]

        self.confidence_threshold = confidence_threshold
        self.input_validator = Validator(
            reaction_col,
            "input-balanced",
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )
        self.rb_validator = Validator(
            reaction_col,
            "rule-based",
            check_carbon_balance=False,
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )
        self.mcs_validator = Validator(
            reaction_col,
            "mcs-based",
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )

        self.rb_method = RuleBasedMethod(
            id_col, reaction_col, reaction_col, n_jobs=n_jobs
        )
        self.mcs_search = MCSSearch(
            id_col,
            solved_col=self.__solved_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            n_jobs=n_jobs,
            enable_progressive_voting=enable_advanced_scoring,
            count_dot_components=enable_advanced_scoring,
        )
        self.mcs_method = MCSBasedMethod(
            reaction_col,
            reaction_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            rules_col=self.__rules_col,
            smiles_standardizer=[MoleculeStandardizer()],
            enable_multi_fragment=enable_multi_fragment,
        )
        self.post_processor = PostProcess(
            id_col=id_col,
            reaction_col=reaction_col,
            n_jobs=n_jobs,
            verbose=0,
            use_default=self.use_default_reduction,
        )
        self.conf_predictor = ConfidencePredictor(
            reaction_col=reaction_col,
            solved_by_method="mcs-based",
            input_reaction_col=self.__input_col,
            confidence_col=self.__confidence_col,
            solved_col=self.__solved_col,
            solved_by_col=self.__solved_by_col,
            issue_col=self.__issue_col,
            mcs_col=self.__mcs_data_col,
        )

    @property
    def n_jobs(self):
        return self.__n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self.__n_jobs = value

        self.input_validator.n_jobs = value
        self.rb_validator.n_jobs = value
        self.mcs_validator.n_jobs = value
        self.rb_method.n_jobs = value
        self.mcs_search.n_jobs = value
        self.post_processor.n_jobs = value

    def __post_process(self, reactions):
        key_index_map = {item[self.__id_col]: idx for idx, item in enumerate(reactions)}
        pp_data = [
            r
            for r in reactions
            if self.__solved_by_col in r.keys()
            and r[self.__solved_by_col] != "input-balanced"
        ]
        pp_results = self.post_processor.fit(pp_data)
        for pp_result in pp_results:
            if (
                pp_result["label"] != "unspecified"
                and "curated_reaction" in pp_result.keys()
            ):
                idx = key_index_map.get(pp_result.get(self.__id_col))
                if idx is None:
                    continue
                reactions[idx][self.__reaction_col] = pp_result["curated_reaction"]

    def run_prebalance_check(self, reactions, stats=None):
        prebalanced_cnt = 0
        for reaction in reactions:
            reaction.setdefault(
                "prebalance_check",
                {
                    "checked": False,
                    "input_reaction": reaction.get(self.__reaction_col, ""),
                    "cleaned_initial_reaction": reaction.get("cleaned_initial_reaction", reaction.get(self.__reaction_col, "")),
                    "analysis": None,
                    "short_circuited": False,
                    "assigned_confidence": None,
                    "notes": [],
                },
            )
            check_log = reaction["prebalance_check"]
            target_reaction = reaction.get("cleaned_initial_reaction", reaction.get(self.__reaction_col, ""))
            analysis = LLMSpeciesBridge.analyze_reaction_balance(target_reaction)
            check_log["checked"] = True
            check_log["input_reaction"] = reaction.get(self.__reaction_col, "")
            check_log["cleaned_initial_reaction"] = target_reaction
            check_log["analysis"] = analysis

            if analysis.get("is_balanced"):
                reaction[self.__solved_col] = True
                reaction[self.__solved_by_col] = "prebalanced"
                reaction[self.__confidence_col] = 1.0
                reaction["workflow_confidence"] = 1.0
                reaction["workflow_confidence_origin"] = "prebalanced"
                reaction["workflow_confidence_label"] = 1.0
                reaction[self.__issue_col] = ""
                reaction["bridge_best_reaction"] = target_reaction
                check_log["short_circuited"] = True
                check_log["assigned_confidence"] = 1.0
                check_log["notes"].append("Reaction is exactly balanced before SynRBL core pipeline; skipped downstream balancing and LLM bridge.")
                reaction.setdefault("workflow_stage_summary", {})["prebalanced"] = True
                prebalanced_cnt += 1

        if stats is not None:
            stats["prebalanced_cnt"] = prebalanced_cnt

    def _run_prebalance_shortcircuit(self, reactions, stats=None):
        self.run_prebalance_check(reactions, stats=stats)

    def run_core_pipeline(self, reactions, stats=None, allow_low_confidence_solved=False, skip_mcs=False):
        for reaction in reactions:
            stage_summary = reaction.setdefault("workflow_stage_summary", {})
            stage_summary.setdefault("core_pipeline_runs", 0)
            stage_summary["core_pipeline_runs"] += 1
            run_index = stage_summary["core_pipeline_runs"]
            if run_index == 1:
                stage_summary["first_core_input_counted"] = True
                stage_summary["first_rule_based_input"] = True
                stage_summary["first_mcs_input"] = True
            else:
                stage_summary["bridge_core_input_counted"] = True
                stage_summary["second_rule_based_input"] = True
                stage_summary["second_mcs_input"] = True
                if allow_low_confidence_solved:
                    stage_summary["bridge_low_confidence_solved_allowed"] = True

        self.input_validator.check(reactions)

        self.rb_method.run(reactions, stats=stats)
        self.rb_validator.check(reactions, override_unsolved=True)
        for reaction in reactions:
            stage_summary = reaction.setdefault("workflow_stage_summary", {})
            if stage_summary.get("core_pipeline_runs", 0) == 1 and reaction.get(self.__solved_by_col) == "rule-based":
                stage_summary["first_rule_based_solved"] = True
            elif stage_summary.get("core_pipeline_runs", 0) > 1 and reaction.get(self.__solved_by_col) == "rule-based":
                stage_summary["second_rule_based_solved"] = True

        # MCS 搜索：skip_mcs=True 时跳过（MCS 数据已由上游缓存注入）
        if not skip_mcs:
            self.mcs_search.find(reactions)

        self.mcs_method.run(reactions, stats=stats)
        self.mcs_validator.check(reactions)

        # 多候选选择：若 MCS 产生多个合并候选，逐个跑后续管线，取置信度最优
        for reaction in reactions:
            candidates = reaction.get("mcs_candidates")
            if candidates and len(candidates) > 1:
                self._select_best_mcs_candidate(
                    reaction, stats, allow_low_confidence_solved)
            else:
                self._run_post_mcs_pipeline(
                    [reaction], stats, allow_low_confidence_solved)

        # 清理临时候选数据
        for reaction in reactions:
            reaction.pop("mcs_candidates", None)

        for reaction in reactions:
            stage_summary = reaction.setdefault("workflow_stage_summary", {})
            if stage_summary.get("core_pipeline_runs", 0) == 1 and reaction.get(self.__solved_by_col) == "mcs-based":
                stage_summary["first_mcs_solved"] = True
            elif stage_summary.get("core_pipeline_runs", 0) > 1 and reaction.get(self.__solved_by_col) == "mcs-based":
                stage_summary["second_mcs_solved"] = True
        return reactions

    def _run_post_mcs_pipeline(self, reactions, stats, allow_low_confidence_solved):
        """MCS 之后的管线步骤：后处理 → 规则修补 → 全元素复检 → XGBoost。"""
        self.__post_process(reactions)
        self.rb_method.run(reactions)
        self.mcs_validator.check(
            reactions,
            override_unsolved=True,
            override_issue_msg="Final reaction is unbalanced.",
        )
        self.conf_predictor.predict(
            reactions,
            stats=stats,
            threshold=self.confidence_threshold,
            allow_low_confidence_solved=allow_low_confidence_solved,
        )

    def _select_best_mcs_candidate(self, reaction, stats, allow_low_confidence_solved):
        """遍历多个 MCS 合并候选，逐个跑后续管线，保留置信度最高的结果。"""
        candidates = reaction["mcs_candidates"]
        best_result = None
        best_score = -1.0
        # BA-2 修复：用临时 stats 收集最优候选的统计数据
        best_candidate_stats = None

        for result_rxn, rules, direction in candidates:
            candidate_reaction = copy.deepcopy(reaction)
            for col in self.mcs_method.output_col:
                candidate_reaction[col] = result_rxn
            candidate_reaction[self.__rules_col] = rules
            candidate_reaction["impute_direction"] = direction
            candidate_reaction.pop("mcs_candidates", None)

            # BA-2 修复：用临时字典收集每个候选的统计，
            # 而非传入 None 导致统计数据丢失
            temp_stats = {}
            self._run_post_mcs_pipeline(
                [candidate_reaction], stats=temp_stats,
                allow_low_confidence_solved=allow_low_confidence_solved,
            )

            confidence = candidate_reaction.get(self.__confidence_col)
            solved = candidate_reaction.get(self.__solved_col, False)
            # 可排序分数：有置信度用置信度值，已配平但无置信度用 0
            # （RB 提前解决时 conf_predictor 跳过，confidence=None），
            # 未配平不参与选择（score = -1）
            if confidence is not None:
                score = confidence
            elif solved:
                score = 0.0
            else:
                score = -1.0

            if score > best_score:
                best_score = score
                best_result = candidate_reaction
                best_candidate_stats = temp_stats

        if best_result is not None:
            # 将最优候选的状态回写到原始 reaction dict
            for key, value in best_result.items():
                reaction[key] = value
            # BA-2 修复：将最优候选的统计数据合并到主 stats
            if stats is not None and best_candidate_stats:
                merge_stats(stats, best_candidate_stats)
        else:
            # 所有候选均未通过，保持第一个候选的状态（已由 mcs_method.run 设置）
            self._run_post_mcs_pipeline(
                [reaction], stats, allow_low_confidence_solved)

    def balance_allocation(self, reactants, products, allocation,
                           swapped=False, cached_mcs_data=None):
        """Convert an exhaustive allocation strategy into a balanced reaction.

        For each product group in the allocation, constructs an allocation
        unit (assigned reactants -> product) and processes it through the
        full SynRBL core pipeline: input validation, rule-based method,
        MCS search, imputation, fragment merge, post-processing, and
        XGBoost confidence scoring.  All allocation units are batched
        together for efficiency.

        After all units are balanced, the final reaction is assembled
        symmetrically: missing fragments are extracted from both sides
        of each allocation unit, and ghost reactants (assigned to the
        empty set during enumeration) are copied to the product side.

        Parameters
        ----------
        reactants : list[str]
            SMILES list (as used in exhaustive allocation; may be swapped).
        products : list[str]
            SMILES list (as used in exhaustive allocation; may be swapped).
        allocation : dict
            ``best_solution`` dict from ``exhaustive_allocation_path``,
            containing ``allocation``, ``product_groups``,
            ``total_mcs_coverage``.
        swapped : bool
            Whether reactants/products were direction-swapped during
            exhaustive allocation.

        Returns
        -------
        dict
            ``success`` (bool), ``balanced_reaction`` (str | None),
            ``confidence`` (float | None), ``sub_reaction_details`` (list).
        """
        product_groups = allocation.get("product_groups", {})
        k_products = len(products)

        # ---- Build allocation units for each product group ----
        sub_reactions = []
        sub_details = []

        for j in range(k_products):
            # 兼容 int 和 string 键（JSON 反序列化可能将 int 转为 str）
            r_indices = product_groups.get(
                j, product_groups.get(str(j), [])
            )
            r_smiles_list = [reactants[i] for i in r_indices]

            if not r_smiles_list:
                # 无反应物分配到该产物：视为已平衡（产物可能来自
                # 溶剂、催化剂等隐式试剂），跳过管线处理
                sub_details.append({
                    "product_idx": j,
                    "reactant_indices": [],
                    "sub_reaction": f">>{products[j]}",
                    "solved": True,
                    "confidence": 1.0,
                    "balanced_reaction": f">>{products[j]}",
                    "solved_by": "skipped_no_reactants",
                })
                continue

            r_side = ".".join(r_smiles_list)
            p_side = products[j]
            sub_rxn = f"{r_side}>>{p_side}"

            check = input_sanitize_check(sub_rxn, reaction_id=f"path_b_sub_{j}")
            if not check["valid"]:
                return {
                    "success": False,
                    "error": f"Sub-reaction {j} invalid: {check['errors']}",
                    "balanced_reaction": None,
                    "confidence": None,
                    "sub_reaction_details": sub_details,
                }

            sub_reactions.append({
                self.__reaction_col: sub_rxn,
                self.__id_col: f"path_b_sub_{j}",
                # BA-1 修复：保存原始产物编号，用于回写 sub_details
                # （preprocess 后的 enumerate 索引 ≠ 原始产物编号，
                # 因为无反应物的产物不在 sub_reactions 中）
                "product_idx": j,
            })
            sub_details.append({
                "product_idx": j,
                "reactant_indices": list(r_indices),
                "sub_reaction": sub_rxn,
                "solved": False,
                "confidence": None,
                "balanced_reaction": None,
            })

        if not sub_reactions:
            return {
                "success": False,
                "error": "No sub-reactions constructed from allocation",
                "balanced_reaction": None,
                "confidence": None,
                "sub_reaction_details": sub_details,
            }

        # ---- Preprocess (standardise, neutralise, deionise, etc.) ----
        reactions = preprocess(
            sub_reactions,
            self.__reaction_col,
            self.__id_col,
            self.__solved_col,
            self.__input_col,
            n_jobs=self.__n_jobs,
            remove_aam=self.remove_aam,
        )

        if not reactions:
            return {
                "success": False,
                "error": "All sub-reactions failed preprocessing",
                "balanced_reaction": None,
                "confidence": None,
                "sub_reaction_details": sub_details,
            }

        # ---- Initialise per-reaction bookkeeping ----
        for idx, r in enumerate(reactions):
            r["_synrbl_internal_id"] = f"path_b_{idx}"
            r["original_row_index"] = idx
            # 保存原始 SMILES 作为 MCS 缓存键（preprocess 可能修改 reaction_col）
            r["_mcs_cache_key"] = r[self.__reaction_col]
            r["original_reaction"] = r[self.__reaction_col]
            r.setdefault("workflow_stage_summary", {})["allocation_path"] = True
            r["bridge_best_reaction"] = r[self.__reaction_col]
            r.setdefault("cleaned_initial_reaction", r[self.__reaction_col])

        # ---- Prebalance check (already atom-balanced?) ----
        self.run_prebalance_check(reactions)
        for r in reactions:
            if r.get("prebalance_check", {}).get("short_circuited", False):
                # BA-1 修复：用 product_idx 而非 original_row_index
                idx = r["product_idx"]
                sub_details[idx]["solved"] = True
                sub_details[idx]["confidence"] = 1.0
                sub_details[idx]["balanced_reaction"] = r[self.__reaction_col]

        remaining = [
            r for r in reactions
            if not r.get("prebalance_check", {}).get("short_circuited", False)
        ]

        # ---- Core pipeline (RB + MCS + merge + post-process + XGBoost) ----
        if remaining:
            # 注入缓存的 MCS 数据（来自穷举排名阶段的 MCES 搜索），
            # 避免 run_core_pipeline 重复执行三条件 MCS 搜索。
            # 仅当所有子反应均命中缓存时才跳过 MCS 搜索；
            # 若有任何未命中，回退到完整搜索以保证覆盖率。
            skip_mcs = False
            if cached_mcs_data:
                all_hit = True
                for r in remaining:
                    cache_key = r.get("_mcs_cache_key")
                    if cache_key and cache_key in cached_mcs_data:
                        r[self.__mcs_data_col] = cached_mcs_data[cache_key]
                    else:
                        all_hit = False
                if all_hit:
                    skip_mcs = True

            # M-1 修复：子反应允许低置信度保留，由编排器/ Bridge LLM 最终裁决
            self.run_core_pipeline(
                remaining,
                allow_low_confidence_solved=True,
                skip_mcs=skip_mcs,
            )
            for r in remaining:
                # BA-1 修复：用 product_idx 而非 original_row_index
                idx = r["product_idx"]
                sub_details[idx]["solved"] = r.get(self.__solved_col, False)
                sub_details[idx]["confidence"] = r.get(self.__confidence_col)
                sub_details[idx]["balanced_reaction"] = r[self.__reaction_col]
                sub_details[idx]["solved_by"] = r.get(self.__solved_by_col, "")

        # ---- Verify all sub-reactions solved ----
        if not all(d["solved"] for d in sub_details):
            failed = [d for d in sub_details if not d["solved"]]
            return {
                "success": False,
                "error": (
                    f"{len(failed)} allocation unit(s) failed to balance"
                ),
                "balanced_reaction": None,
                "confidence": None,
                "sub_reaction_details": sub_details,
            }

        # ---- Collect generated (missing) fragments — symmetric ----
        # 对每个分配单元，同时检查反应物侧和产物侧的新增碎片
        missing_reactant_parts = []
        missing_product_parts = []
        for detail in sub_details:
            bal_rxn = detail.get("balanced_reaction", "")
            if not bal_rxn or ">>" not in bal_rxn:
                continue
            bal_lhs, bal_rhs = bal_rxn.split(">>", 1)
            orig_lhs, orig_rhs = detail["sub_reaction"].split(">>", 1)

            # 产物侧：配平后多出来的碎片 → 加到总方程式右侧
            orig_p_set = set(
                p for p in orig_rhs.split(".") if p
            )
            for part in bal_rhs.split("."):
                if part and part not in orig_p_set:
                    missing_product_parts.append(part)

            # 反应物侧：配平后多出来的碎片 → 加到总方程式左侧
            orig_r_set = set(
                r for r in orig_lhs.split(".") if r
            )
            for part in bal_lhs.split("."):
                if part and part not in orig_r_set:
                    missing_reactant_parts.append(part)

        # ---- Ghost reactants: assigned to empty set (no product group) ----
        # 这些反应物不参与任何分配单元，复制到产物侧使左右抵消
        allocated_indices = set()
        for j in range(k_products):
            r_indices = product_groups.get(
                j, product_groups.get(str(j), [])
            )
            allocated_indices.update(r_indices)
        ghost_reactants = [
            reactants[i]
            for i in range(len(reactants))
            if i not in allocated_indices and reactants[i]
        ]

        # ---- Assemble the final balanced reaction ----
        all_lhs = [sm for sm in reactants if sm] + missing_reactant_parts
        all_rhs = (
            [sm for sm in products if sm]
            + missing_product_parts
            + ghost_reactants
        )
        balanced = ">>".join([".".join(all_lhs), ".".join(all_rhs)])

        # ---- Un-swap if needed ----
        if swapped:
            balanced = ">>".join(
                [".".join(all_rhs), ".".join(all_lhs)]
            )

        # ---- Overall confidence = minimum across allocation units ----
        confidences = [
            d["confidence"] for d in sub_details
            if d["confidence"] is not None
        ]
        overall_confidence = min(confidences) if confidences else None

        total_missing = (
            len(missing_reactant_parts)
            + len(missing_product_parts)
            + len(ghost_reactants)
        )
        logger.info(
            "Path B balance: %d allocation units, confidence=%.4f, "
            "missing_reactant=%d, missing_product=%d, ghost=%d, "
            "balanced=%s",
            len(sub_details),
            overall_confidence or 0.0,
            len(missing_reactant_parts),
            len(missing_product_parts),
            len(ghost_reactants),
            balanced,
        )

        return {
            "success": True,
            "balanced_reaction": balanced,
            "confidence": overall_confidence,
            "sub_reaction_details": sub_details,
            "ghost_reactants": ghost_reactants,
            "missing_reactant_parts": missing_reactant_parts,
            "missing_product_parts": missing_product_parts,
        }

    def run_post_generation_pipeline(self, reactions, stats=None):
        for reaction in reactions:
            stage_summary = reaction.setdefault("workflow_stage_summary", {})
            stage_summary["fallback_post_generation_input"] = True
        self.__post_process(reactions)
        self.rb_method.run(reactions, stats=stats)
        self.mcs_validator.check(
            reactions,
            override_unsolved=True,
            override_issue_msg="Final reaction is unbalanced.",
        )
        for reaction in reactions:
            stage_summary = reaction.setdefault("workflow_stage_summary", {})
            stage_summary["fallback_post_generation_output"] = True
            stage_summary["fallback_post_generation_balanced"] = bool(reaction.get(self.__solved_col, False))
        return reactions

    def __run_pipeline(self, reactions, stats=None):
        if stats is not None:
            stats["reaction_cnt"] = len(reactions)

        if stats is not None:
            stats["raw_input_cnt"] = len(reactions)

        # === 步骤 0: 输入验证（input_sanitize_check）===
        valid_reactions = []
        for reaction in reactions:
            rxn_smiles = reaction.get(self.__reaction_col, "")
            if rxn_smiles is None:
                rxn_smiles = ""
            rxn_id = reaction.get(self.__id_col, "")
            check_result = input_sanitize_check(
                str(rxn_smiles), reaction_id=str(rxn_id)
            )
            if check_result["valid"]:
                valid_reactions.append(reaction)
            else:
                reaction["processable"] = False
                reaction["preprocess_status"] = "invalid_smiles"
                reaction["issue"] = "; ".join(check_result["errors"])
                stage_summary = reaction.setdefault(
                    "workflow_stage_summary", {}
                )
                stage_summary["input_sanitize"] = {
                    "valid": False,
                    "errors": check_result["errors"],
                }

        if stats is not None:
            stats["post_input_sanitize_cnt"] = len(valid_reactions)
            stats["input_sanitize_invalid_cnt"] = (
                len(reactions) - len(valid_reactions)
            )

        # 保存未通过验证的反应，稍后追加回结果列表
        invalid_reactions = [
            r for r in reactions if not r.get("processable", True)
        ]
        total_input_cnt = len(reactions)

        # 仅对通过输入验证的反应执行 preprocess 及后续流程
        reactions = preprocess(
            valid_reactions,
            self.__reaction_col,
            self.__id_col,
            self.__solved_col,
            self.__input_col,
            n_jobs=self.__n_jobs,
            remove_aam=self.remove_aam,
        )

        if stats is not None:
            stats["post_preprocess_cnt"] = len(reactions)

        for r in reactions:
            stage_summary = r.setdefault("workflow_stage_summary", {})
            stage_summary.setdefault("raw_input_seen", True)
            stage_summary.setdefault("post_preprocess_seen", True)
            if "cleaned_initial_reaction" not in r:
                r["cleaned_initial_reaction"] = r[self.__reaction_col]
            r.setdefault("bridge_best_reaction", r[self.__reaction_col])

        rxn_cnt = len(reactions)

        self.run_prebalance_check(reactions, stats=stats)
        remaining_reactions = [r for r in reactions if not r.get("prebalance_check", {}).get("short_circuited", False)]
        if stats is not None:
            stats["post_prebalance_remaining_cnt"] = len(remaining_reactions)

        if remaining_reactions:
            self.run_core_pipeline(remaining_reactions, stats=stats)
            for r in remaining_reactions:
                r["bridge_best_reaction"] = r.get(self.__reaction_col)
                if "workflow_confidence" not in r and r.get(self.__confidence_col) is not None:
                    r["workflow_confidence"] = r.get(self.__confidence_col)

        if self.llm_species_bridge is not None and remaining_reactions:
            self.llm_species_bridge.apply(remaining_reactions, self)

        if self.llm_fallback_postprocessor is not None and remaining_reactions:
            self.llm_fallback_postprocessor.apply(remaining_reactions, self, stats=stats)

        if rxn_cnt != len(reactions):
            raise RuntimeError(
                "Reaction count changed during pipeline: {} -> {}".format(
                    rxn_cnt, len(reactions))
            )

        # 将未通过输入验证的反应追加回结果列表
        reactions.extend(invalid_reactions)
        if total_input_cnt != len(reactions):
            raise RuntimeError(
                "Reaction count changed after appending invalid: {} -> {}".format(
                    total_input_cnt, len(reactions))
            )

        logger.info("DONE")

        return reactions

    def __convert_to_dataset(self, data) -> Dataset:
        dataset = None
        if isinstance(data, str):
            data = [data]
        if isinstance(data, list):
            reaction_data = []
            for r in data:
                if isinstance(r, str):
                    reaction_data.append({self.__reaction_col: r})
                elif isinstance(r, dict):
                    reaction_data.append(r)
                else:
                    raise ValueError(
                        "Expected (a list of) SMILES or a data dictionary. "
                        + "Found '{}' instead.".format(type(r))
                    )
            dataset = Dataset(reaction_data)
        if isinstance(data, Dataset):
            dataset = data
        if dataset is None:
            raise ValueError(
                (
                    "Invalid type '{}' of reactions. "
                    + "Use a list of SMILES or a Dataset instead."
                ).format(type(data))
            )
        return dataset

    def __try_cache(self, cache_manager, batch):
        result = None
        batch_stats = {}
        cache_key = None
        if cache_manager:
            cache_key = cache_manager.get_hash_key(batch)
            if cache_manager.is_cached(cache_key):
                logger.info("Load cached results. (Key: {})".format(cache_key[:8]))
                cache_result = cache_manager.load_cache(cache_key)
                result = cache_result.get("result", None)
                batch_stats = cache_result.get("stats", None)
        return result, batch_stats, cache_key

    def __init_cache(self):
        cache_manager = None
        if self.cache:
            if self.cache_dir is None:
                raise ValueError(
                    "Undefined cache directory. "
                    + "Specify a directory with 'cache_dir' argument."
                )
            cache_manager = CacheManager(cache_dir=self.cache_dir)
        return cache_manager

    def __rebalance_batch(self, batch, cache_manager):
        result, batch_stats, cache_key = self.__try_cache(cache_manager, batch)
        if result is None or batch_stats is None:
            batch_stats = {}
            try:
                result = self.__run_pipeline(copy.deepcopy(batch), batch_stats)
                if cache_manager:
                    assert cache_key is not None
                    cache_manager.write_cache(
                        cache_key, {"stats": batch_stats, "result": result}
                    )
                    logger.info("Cached new results. (Key: {})".format(cache_key[:8]))
            except Exception as e:
                traceback.print_exc()
                logger.error("Pipeline execution failed: {}".format(type(e)))
                result = []
                for row_index, entry in enumerate(batch):
                    single_stats = {}
                    try:
                        single_result = self.__run_pipeline([copy.deepcopy(entry)], single_stats)
                        result.extend(single_result)
                        merge_stats(batch_stats, single_stats)
                    except Exception as single_exc:
                        fallback = copy.deepcopy(entry)
                        fallback.setdefault("original_row_index", row_index)
                        fallback.setdefault("original_reaction", fallback.get(self.__reaction_col))
                        fallback.setdefault(self.__input_col, fallback.get(self.__reaction_col))
                        fallback.setdefault("_synrbl_internal_id", f"single-fallback-{row_index}")
                        fallback[self.__solved_col] = False
                        fallback.setdefault(self.__solved_by_col, "")
                        fallback[self.__confidence_col] = None
                        fallback.setdefault(self.__rules_col, [])
                        fallback.setdefault(self.__issue_col, "")
                        fallback["processable"] = False
                        fallback["pipeline_failed"] = True
                        fallback["pipeline_failure_stage"] = "single_reaction_retry"
                        if not fallback[self.__issue_col]:
                            fallback[self.__issue_col] = (
                                f"Single reaction processing failed: {type(single_exc).__name__}: {single_exc}"
                            )
                        result.append(fallback)
                batch_stats["pipeline_failed_batch_cnt"] = 1
                batch_stats["pipeline_single_retry_cnt"] = len(batch)

        return result, batch_stats

    def rebalance(self, reactions, output_dict=False, stats=None, batch_size=None):
        dataset = self.__convert_to_dataset(reactions)
        batch_size = self.batch_size if batch_size is None else batch_size

        if batch_size is None:
            dataloader = iter([[e for e in dataset]])
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size)

        results = []
        cache_manager = self.__init_cache()

        for batch_i, batch in enumerate(dataloader):
            batch_i = batch_i + 1
            if len(batch) == 0:
                continue
            if batch_size is not None:
                logger.info("Start Batch {} | Size: {}".format(batch_i, len(batch)))

            result, batch_stats = self.__rebalance_batch(batch, cache_manager)

            if result is not None:
                results.extend(result)
                merge_stats(stats, batch_stats)

            if batch_size is not None:
                logger.info("Completed batch {}".format(batch_i))

        if output_dict:
            output = []
            for r in results:
                output.append({k: v for k, v in r.items() if k in self.columns})
            return output
        else:
            return [r[self.__reaction_col] for r in results]
