import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from synrbl import Balancer, RebalanceConfig
from synrbl.evaluation_utils import write_accuracy_report, write_workflow_statistics
from synrbl.llm_fallback_postprocessor import LLMFallbackPostprocessor
from synrbl.llm_species_bridge import LLMSpeciesBridge
from synrbl.preprocess import preprocess

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}
WORKFLOW_THRESHOLD = 0.8
INTERNAL_TRACKING_ID = "_workflow_tracking_id"


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


def _is_balanced(reaction: Any) -> bool:
    text = str(reaction or "").strip()
    if not text:
        return False
    return bool(LLMSpeciesBridge.analyze_reaction_balance(text).get("is_balanced", False))


def _is_valid_reaction_smiles(reaction: Any) -> bool:
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
            if not LLMSpeciesBridge._is_valid_smiles(token):
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


def _build_base_config(args: argparse.Namespace, output_dir: Path, reaction_col: str, id_col: str, expected_col: str) -> RebalanceConfig:
    return RebalanceConfig(
        reaction_col=reaction_col,
        id_col=id_col,
        synrbl_confidence_threshold=args.synrbl_confidence_threshold,
        enable_llm_postprocess=False,
        enable_llm_species_bridge=False,
        llm_score_threshold=args.score_threshold,
        llm_retry_on_low_confidence=not args.disable_low_confidence_retry,
        llm_retry_confidence_threshold=args.retry_confidence_threshold,
        llm_species_bridge_confidence_threshold=args.species_bridge_confidence_threshold,
        llm_top_k_per_strategy=args.top_k_per_strategy,
        llm_enable_two_stage_generation=args.enable_two_stage_llm,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
        llm_score_model=args.llm_score_model,
        llm_generate_model=args.llm_generate_model,
        llm_thinking_enabled=args.enable_llm_thinking,
        output_dir=str(output_dir),
        expected_reaction_col=expected_col,
    )


def _build_balancer(cfg: RebalanceConfig) -> Balancer:
    return Balancer(
        reaction_col=cfg.reaction_col,
        id_col=cfg.id_col,
        confidence_threshold=cfg.synrbl_confidence_threshold,
        llm_postprocessor=None,
        llm_species_bridge=None,
        llm_fallback_postprocessor=None,
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


def _run_first_round(records: list[dict[str, Any]], cfg: RebalanceConfig) -> list[dict[str, Any]]:
    balancer = _build_balancer(cfg)
    working = copy.deepcopy(records)
    working = preprocess(
        working,
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


def _build_bridge_runner(args: argparse.Namespace, cfg: RebalanceConfig) -> tuple[LLMSpeciesBridge, Balancer]:
    bridge = LLMSpeciesBridge.from_moonshot(
        id_col=cfg.id_col,
        reaction_col=cfg.reaction_col,
        confidence_threshold=args.species_bridge_confidence_threshold,
        api_key_env=args.llm_api_key_env,
        base_url=args.llm_base_url,
        model=args.llm_generate_model,
        max_workers=25,
        thinking_enabled=args.enable_llm_thinking,
    )
    balancer = _build_balancer(cfg)
    return bridge, balancer


def _run_bridge_round(bridge_inputs: list[dict[str, Any]], args: argparse.Namespace, cfg: RebalanceConfig) -> list[dict[str, Any]]:
    bridge, balancer = _build_bridge_runner(args, cfg)
    working = copy.deepcopy(bridge_inputs)
    bridge.apply(working, balancer)
    for row in working:
        bridge_log = row.get("llm_species_bridge") or {}
        for key in ("accepted_variant", "bridge_candidate_reaction", "bridge_best_reaction"):
            candidate = row.get(key) if key != "accepted_variant" else (bridge_log.get("accepted_variant") if isinstance(bridge_log, dict) else None)
            if candidate and not _is_valid_reaction_smiles(candidate):
                if key == "accepted_variant" and isinstance(bridge_log, dict):
                    bridge_log["accepted_variant"] = None
                    bridge_log["accepted_variant_evaluation"] = None
                    bridge_log["accepted_variant_balance_analysis"] = None
                    bridge_log["final_status"] = "invalid_accepted_variant_filtered"
                    bridge_log["failure_reason"] = "Accepted bridge variant failed pre-core reaction validity filter."
                    row["llm_species_bridge"] = bridge_log
                else:
                    row[key] = None
    return working


def _stage2_decision(stage1_row: dict[str, Any], stage2_row: dict[str, Any], reaction_col: str) -> dict[str, Any]:
    candidate1_reaction = stage1_row.get("internal_candidate_1_reaction")
    candidate1_confidence = _normalize_confidence(stage1_row.get("internal_candidate_1_confidence"))
    bridge_log = stage2_row.get("llm_species_bridge") or {}
    bridge_raw_output = stage2_row.get("bridge_candidate_reaction") or stage2_row.get("bridge_best_reaction")
    accepted_reaction = bridge_log.get("accepted_variant") if isinstance(bridge_log, dict) else None
    accepted_eval = bridge_log.get("accepted_variant_evaluation") if isinstance(bridge_log, dict) else None
    accepted_round_summary = (accepted_eval or {}).get("round_summary") or {}
    accepted_final_summary = accepted_round_summary.get("final") or {}
    accepted_solved_by = str(
        accepted_final_summary.get("solved_by")
        or (accepted_eval or {}).get("solved_by")
        or ""
    ).strip()
    accepted_workflow_route = str(
        bridge_log.get("accepted_variant_workflow_route")
        or stage2_row.get("workflow_route")
        or ""
    ).strip()
    accepted_confidence = _normalize_confidence(
        accepted_final_summary.get("confidence")
        if accepted_final_summary.get("confidence") is not None
        else (accepted_eval or {}).get("confidence")
    )
    accepted_balanced = _is_balanced(accepted_reaction)
    accepted_valid = _is_valid_reaction_smiles(accepted_reaction)
    bridge_raw_balanced = _is_balanced(bridge_raw_output)
    bridge_raw_valid = _is_valid_reaction_smiles(bridge_raw_output)

    accepted_rule_based = bool(accepted_round_summary.get("used_rule_based_only", False))
    accepted_mcs_solved = bool(accepted_round_summary.get("used_mcs", False))

    decision = {
        "formal_output_reaction": None,
        "workflow_confidence": None,
        "workflow_source": "bridge",
        "internal_candidate_2_reaction": None,
        "internal_candidate_2_confidence": None,
        "internal_candidate_2_source": None,
        "fallback_input_source": None,
        "fallback_input_reaction": None,
        "needs_fallback": False,
        "stage2_case": None,
        "stage2_case_reason": None,
        "bridge_raw_output_reaction": bridge_raw_output,
        "bridge_raw_balanced": bridge_raw_balanced,
        "bridge_raw_valid": bridge_raw_valid,
        "bridge_accepted_output_reaction": accepted_reaction,
        "bridge_accepted_output_balanced": accepted_balanced,
        "bridge_accepted_output_valid": accepted_valid,
        "bridge_accepted_output_solved_by": accepted_solved_by,
        "bridge_accepted_output_workflow_route": accepted_workflow_route,
        "bridge_accepted_output_confidence": accepted_confidence,
        "bridge_accepted_output_round_summary": accepted_round_summary,
        "bridge_accepted_reaction": accepted_reaction,
        "bridge_accepted_balanced": accepted_balanced,
        "bridge_accepted_solved_by": accepted_solved_by,
        "bridge_accepted_workflow_route": accepted_workflow_route,
        "bridge_accepted_confidence": accepted_confidence,
        "bridge_accepted_round_summary": accepted_round_summary,
    }

    if bridge_raw_output and not bridge_raw_valid:
        bridge_raw_output = None
        decision["bridge_raw_output_reaction"] = None
        decision["bridge_raw_balanced"] = False
    if accepted_reaction and not accepted_valid:
        accepted_reaction = None
        accepted_balanced = False
        decision["bridge_accepted_output_reaction"] = None
        decision["bridge_accepted_output_balanced"] = False
        decision["bridge_accepted_output_valid"] = False
        decision["bridge_accepted_output_solved_by"] = None
        decision["bridge_accepted_output_workflow_route"] = None
        decision["bridge_accepted_output_confidence"] = None
        decision["bridge_accepted_reaction"] = None
        decision["bridge_accepted_balanced"] = False
        decision["bridge_accepted_solved_by"] = None
        decision["bridge_accepted_workflow_route"] = None
        decision["bridge_accepted_confidence"] = None

    if bridge_raw_balanced and bridge_raw_valid and not accepted_balanced:
        decision["formal_output_reaction"] = bridge_raw_output
        decision["workflow_confidence"] = 1.5
        decision["stage2_case"] = "bridge_direct"
        decision["stage2_case_reason"] = "Bridge raw output is atom-balanced before accepted second-round SynRBL result exists."
        return decision

    if accepted_balanced and accepted_valid and accepted_rule_based and not accepted_mcs_solved:
        decision["formal_output_reaction"] = accepted_reaction
        decision["workflow_confidence"] = 1.5
        decision["stage2_case"] = "bridge_small_molecule"
        decision["stage2_case_reason"] = "Bridge-triggered second-round SynRBL solved using only rule-based small molecules."
        return decision

    if accepted_balanced and accepted_valid and accepted_mcs_solved and accepted_confidence is not None:
        if accepted_confidence >= WORKFLOW_THRESHOLD:
            decision["formal_output_reaction"] = accepted_reaction
            decision["workflow_confidence"] = accepted_confidence
            decision["stage2_case"] = "bridge_mcs_high_confidence"
            decision["stage2_case_reason"] = "Bridge-triggered second-round SynRBL reached an MCS result with confidence >= workflow threshold."
            return decision
        if candidate1_confidence is None:
            decision["internal_candidate_2_reaction"] = accepted_reaction
            decision["internal_candidate_2_confidence"] = accepted_confidence
            decision["internal_candidate_2_source"] = "bridge_accepted_output"
        elif accepted_confidence >= candidate1_confidence:
            decision["internal_candidate_2_reaction"] = accepted_reaction
            decision["internal_candidate_2_confidence"] = accepted_confidence
            decision["internal_candidate_2_source"] = "bridge_accepted_output"
        else:
            decision["internal_candidate_2_reaction"] = candidate1_reaction
            decision["internal_candidate_2_confidence"] = candidate1_confidence
            decision["internal_candidate_2_source"] = "SynRBL"
        decision["stage2_case"] = "bridge_mcs_low_confidence"
        decision["stage2_case_reason"] = "Bridge-triggered second-round MCS result is balanced but below workflow threshold, so it enters candidate comparison."
    elif accepted_balanced and accepted_valid and accepted_mcs_solved and accepted_confidence is None:
        if candidate1_confidence is None:
            decision["stage2_case"] = "both_balanced_no_confidence"
            decision["stage2_case_reason"] = "Both first-round and bridge second-round balanced outputs lack confidence, so fallback uses the direct bridge raw output rather than the second-round accepted output."
            decision["fallback_input_source"] = "bridge_raw_output" if bridge_raw_output else "original"
            decision["fallback_input_reaction"] = bridge_raw_output or stage2_row.get("original_reaction")
        else:
            decision["internal_candidate_2_reaction"] = candidate1_reaction
            decision["internal_candidate_2_confidence"] = candidate1_confidence
            decision["internal_candidate_2_source"] = "SynRBL"
            decision["stage2_case"] = "bridge_mcs_missing_confidence"
            decision["stage2_case_reason"] = "Second-round bridge MCS output is balanced but missing confidence, so the trusted first-round candidate is restored."
            decision["fallback_input_source"] = "original"
            decision["fallback_input_reaction"] = stage2_row.get("original_reaction")
    else:
        decision["stage2_case"] = "bridge_failed"
        if accepted_balanced and accepted_rule_based and accepted_mcs_solved:
            decision["stage2_case_reason"] = "Bridge second-round result contains both rule-based and MCS traces; framework treats it as an MCS-route result and requires confidence-based handling."
        else:
            decision["stage2_case_reason"] = "Bridge route did not produce any balanced accepted output, so fallback reverts to original reaction or first-round trusted candidate."
        if candidate1_confidence is not None:
            decision["internal_candidate_2_reaction"] = candidate1_reaction
            decision["internal_candidate_2_confidence"] = candidate1_confidence
            decision["internal_candidate_2_source"] = "SynRBL"
            decision["fallback_input_source"] = "original"
            decision["fallback_input_reaction"] = stage2_row.get("original_reaction")
        else:
            decision["fallback_input_source"] = "original"
            decision["fallback_input_reaction"] = stage2_row.get("original_reaction")

    if decision["internal_candidate_2_reaction"] is not None and not _is_valid_reaction_smiles(decision["internal_candidate_2_reaction"]):
        decision["internal_candidate_2_reaction"] = None
        decision["internal_candidate_2_confidence"] = None
        decision["internal_candidate_2_source"] = None

    if decision["fallback_input_reaction"] is not None and not _is_valid_reaction_smiles(decision["fallback_input_reaction"]):
        if decision.get("fallback_input_source") != "original":
            decision["fallback_input_source"] = "original"
            decision["fallback_input_reaction"] = stage2_row.get("original_reaction")
            if decision["fallback_input_reaction"] is not None and not _is_valid_reaction_smiles(decision["fallback_input_reaction"]):
                decision["fallback_input_reaction"] = None
        else:
            decision["fallback_input_reaction"] = None

    if decision["formal_output_reaction"] is None:
        decision["needs_fallback"] = True
        if decision["fallback_input_reaction"] is None:
            if decision["internal_candidate_2_source"] == "bridge_accepted_output" and _is_valid_reaction_smiles(decision["internal_candidate_2_reaction"]):
                decision["fallback_input_source"] = "bridge_accepted_output"
                decision["fallback_input_reaction"] = decision["internal_candidate_2_reaction"]
            else:
                original_reaction = stage2_row.get("original_reaction")
                decision["fallback_input_source"] = "original"
                decision["fallback_input_reaction"] = original_reaction if _is_valid_reaction_smiles(original_reaction) else None
    return decision


def _build_fallback_runner(args: argparse.Namespace, cfg: RebalanceConfig) -> tuple[LLMFallbackPostprocessor, Balancer]:
    fallback = LLMFallbackPostprocessor.from_moonshot(
        id_col=cfg.id_col,
        reaction_col=cfg.reaction_col,
        retry_confidence_threshold=args.retry_confidence_threshold,
        api_key_env=args.llm_api_key_env,
        base_url=args.llm_base_url,
        model=args.llm_generate_model,
        max_workers=25,
        thinking_enabled=args.enable_llm_thinking,
    )
    balancer = _build_balancer(cfg)
    return fallback, balancer


def _run_fallback_round(fallback_inputs: list[dict[str, Any]], args: argparse.Namespace, cfg: RebalanceConfig) -> list[dict[str, Any]]:
    fallback, balancer = _build_fallback_runner(args, cfg)
    working = copy.deepcopy(fallback_inputs)
    valid_rows: list[dict[str, Any]] = []
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
        fallback.apply(valid_rows, balancer, stats={})
    return working


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
    row[reaction_col] = formal_output_reaction
    row["formal_output_reaction"] = formal_output_reaction
    row["workflow_confidence"] = workflow_confidence
    row["workflow_source"] = workflow_source
    row["success"] = success
    row["output_status"] = "True" if success else "False"
    row["final_result_source"] = workflow_source
    row["internal_candidate_1_reaction"] = stage1.get("internal_candidate_1_reaction")
    row["internal_candidate_1_confidence"] = stage1.get("internal_candidate_1_confidence")
    row["internal_candidate_1_source"] = stage1.get("internal_candidate_1_source")
    row["stage1_case"] = stage1.get("stage1_case")
    row["internal_candidate_2_reaction"] = internal_candidate_2_reaction
    row["internal_candidate_2_confidence"] = internal_candidate_2_confidence
    row["internal_candidate_2_source"] = stage2.get("internal_candidate_2_source") if stage2 is not None else None
    row["fallback_input_source"] = fallback_input_source
    row["fallback_input_reaction"] = fallback_input_reaction
    if stage2 is not None:
        row["stage2_case"] = stage2.get("stage2_case")
        row["stage2_case_reason"] = stage2.get("stage2_case_reason")
        row["bridge_raw_output_reaction"] = stage2.get("bridge_raw_output_reaction")
        row["bridge_raw_balanced"] = stage2.get("bridge_raw_balanced")
        row["bridge_raw_valid"] = stage2.get("bridge_raw_valid")
        row["bridge_accepted_output_reaction"] = stage2.get("bridge_accepted_output_reaction")
        row["bridge_accepted_output_balanced"] = stage2.get("bridge_accepted_output_balanced")
        row["bridge_accepted_output_valid"] = stage2.get("bridge_accepted_output_valid")
        row["bridge_accepted_output_solved_by"] = stage2.get("bridge_accepted_output_solved_by")
        row["bridge_accepted_output_workflow_route"] = stage2.get("bridge_accepted_output_workflow_route")
        row["bridge_accepted_output_confidence"] = stage2.get("bridge_accepted_output_confidence")
        row["bridge_accepted_output_round_summary"] = stage2.get("bridge_accepted_output_round_summary")
        row["bridge_accepted_reaction"] = stage2.get("bridge_accepted_output_reaction")
        row["bridge_accepted_balanced"] = stage2.get("bridge_accepted_output_balanced")
        row["bridge_accepted_solved_by"] = stage2.get("bridge_accepted_output_solved_by")
        row["bridge_accepted_workflow_route"] = stage2.get("bridge_accepted_output_workflow_route")
        row["bridge_accepted_confidence"] = stage2.get("bridge_accepted_output_confidence")
        row["bridge_accepted_round_summary"] = stage2.get("bridge_accepted_output_round_summary")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged SynRBL workflow orchestrator.")
    parser.add_argument("input_file", help="Path to input CSV/TSV/Excel/JSON file.")
    parser.add_argument("--reaction-col", default="reactions", help="Reaction column name.")
    parser.add_argument("--id-col", default="R-id", help="ID column name.")
    parser.add_argument("--expected-col", default="expected_reaction", help="Expected reaction column for accuracy comparison.")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM postprocessing.")
    parser.add_argument("--llm-api-key-env", default="MOONSHOT_API_KEY", help="Environment variable name for LLM API key.")
    parser.add_argument("--llm-base-url", default="https://api.moonshot.cn/v1/chat/completions", help="LLM chat completions endpoint.")
    parser.add_argument("--llm-score-model", default="kimi-k2.5", help="LLM scoring model name.")
    parser.add_argument("--llm-generate-model", default="kimi-k2.5", help="LLM generation model name.")
    parser.add_argument("--enable-llm-thinking", action="store_true", help="Enable Kimi thinking mode for all LLM calls.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to input file directory.")
    parser.add_argument("--synrbl-confidence-threshold", type=float, default=0.8, help="Native SynRBL confidence threshold.")
    parser.add_argument("--score-threshold", type=float, default=0.8, help="Score threshold for switching from scoring LLM to generation LLM.")
    parser.add_argument("--retry-confidence-threshold", type=float, default=0.8, help="Retry with LLM when SynRBL confidence is below this threshold.")
    parser.add_argument("--species-bridge-confidence-threshold", type=float, default=0.8, help="Retry with LLM species bridge when SynRBL confidence is below this threshold.")
    parser.add_argument("--disable-low-confidence-retry", action="store_true", help="Disable retrying solved low-confidence reactions with LLM.")
    parser.add_argument("--top-k-per-strategy", type=int, default=1, help="Top-k candidates per MCS strategy.")
    parser.add_argument("--enable-two-stage-llm", action="store_true", help="Enable diagnosis-guided two-stage LLM generation.")
    parser.add_argument("--enable-llm-species-bridge", action="store_true", help="Enable LLM side-species proposal bridge before full LLM generation.")
    parser.add_argument("--sep", default=None, help="Optional separator for CSV/TSV/TXT input.")
    parser.add_argument("--head", type=int, default=None, help="只处理前 N 条数据（用于测试）")
    parser.add_argument("--start-row", type=int, default=None, help="从第几行数据开始处理（不含题头，1-based）")
    parser.add_argument("--end-row", type=int, default=None, help="处理到第几行数据结束（不含题头，1-based，包含该行）")
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

    cfg = _build_base_config(args, output_dir, reaction_col, id_col, expected_col)
    prepared_records = _prepare_records(df, reaction_col, id_col)

    stage1_rows = _run_first_round(prepared_records, cfg)
    stage1_by_tracking = {row[INTERNAL_TRACKING_ID]: row for row in stage1_rows}
    original_stage_outputs: list[dict[str, Any]] = []
    bridge_inputs: list[dict[str, Any]] = []
    bridge_stage_outputs: list[dict[str, Any]] = []

    for row in stage1_rows:
        decision = _stage1_decision(row, reaction_col)
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
            None,
            None,
        )
        original_stage_outputs.append(stage1_output)
        if decision.get("needs_bridge"):
            bridge_input = copy.deepcopy(row)
            bridge_input["stage1_case"] = decision.get("stage1_case")
            bridge_input["internal_candidate_1_reaction"] = decision.get("internal_candidate_1_reaction")
            bridge_input["internal_candidate_1_confidence"] = decision.get("internal_candidate_1_confidence")
            bridge_inputs.append(bridge_input)

    bridge_stage_rows = _run_bridge_round(bridge_inputs, args, cfg) if bridge_inputs else []
    bridge_by_tracking = {row[INTERNAL_TRACKING_ID]: row for row in bridge_stage_rows}

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
            }
        )

    fallback_stage_rows = _run_fallback_round(fallback_inputs, args, cfg) if fallback_inputs else []
    fallback_by_tracking = {row[INTERNAL_TRACKING_ID]: row for row in fallback_stage_rows}

    for item in fallback_inputs:
        tracking_id = item[INTERNAL_TRACKING_ID]
        fallback_row = fallback_by_tracking.get(tracking_id)
        fallback_reaction = fallback_row.get(reaction_col) if fallback_row else None
        fallback_success = _is_valid_reaction_smiles(fallback_reaction) and _is_balanced(fallback_reaction)
        fallback_source_row = copy.deepcopy(item["bridge_row"])
        if fallback_row is not None:
            fallback_source_row.update(fallback_row)
        final_rows.append(
            _build_output_row(
                fallback_source_row,
                reaction_col,
                fallback_reaction if fallback_success else None,
                2.0 if fallback_success else None,
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

    sort_key = lambda x: int(x.get("original_row_index", 0))
    original_stage_outputs.sort(key=sort_key)
    bridge_stage_rows.sort(key=sort_key)
    bridge_stage_outputs.sort(key=sort_key)
    final_rows.sort(key=sort_key)
    fallback_stage_rows.sort(key=sort_key)

    _write_json_and_csv(
        original_stage_outputs,
        output_dir / "synrbl_results_original_stage.json",
        output_dir / "synrbl_results_original_stage.csv",
    )
    _write_json_and_csv(
        bridge_stage_outputs,
        output_dir / "synrbl_results_bridge_stage.json",
        output_dir / "synrbl_results_bridge_stage.csv",
    )
    _write_json_and_csv(
        final_rows,
        output_dir / "synrbl_results_with_llm.json",
        output_dir / "synrbl_results_with_llm.csv",
    )

    failed_rows = [row for row in final_rows if not row.get("success")]
    _write_json_and_csv(
        failed_rows,
        output_dir / "synrbl_failed_cases.json",
        output_dir / "synrbl_failed_cases_flat.csv",
    )

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
    print(f"Finished. Generated {len(final_rows)} final result rows.")


if __name__ == "__main__":
    main()
