import copy
import logging
import traceback

from synrbl.preprocess import preprocess
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
        )
        self.mcs_method = MCSBasedMethod(
            reaction_col,
            reaction_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            rules_col=self.__rules_col,
            smiles_standardizer=[MoleculeStandardizer()],
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
                idx = key_index_map[pp_result[self.__id_col]]
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

    def run_core_pipeline(self, reactions, stats=None, allow_low_confidence_solved=False):
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

        self.mcs_search.find(reactions)

        self.mcs_method.run(reactions, stats=stats)
        self.mcs_validator.check(reactions)

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
        for reaction in reactions:
            stage_summary = reaction.setdefault("workflow_stage_summary", {})
            if stage_summary.get("core_pipeline_runs", 0) == 1 and reaction.get(self.__solved_by_col) == "mcs-based":
                stage_summary["first_mcs_solved"] = True
            elif stage_summary.get("core_pipeline_runs", 0) > 1 and reaction.get(self.__solved_by_col) == "mcs-based":
                stage_summary["second_mcs_solved"] = True
        return reactions

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

        reactions = preprocess(
            reactions,
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

        assert rxn_cnt == len(reactions)
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
