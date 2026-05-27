from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False

from synrbl.SynUtils import normalize_smiles, wc_similarity
from synrbl.llm_species_bridge import LLMSpeciesBridge


WORKFLOW_STATISTICS_FILENAME = "workflow_statistics.csv"
WORKFLOW_STATISTICS_XLSX_FILENAME = "workflow_statistics.xlsx"


def _safe_normalize(smiles: Any) -> str | None:
    if smiles is None or pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    try:
        return normalize_smiles(text)
    except Exception:
        return None


def _safe_parse_json_like(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _is_prebalance_row(row: dict[str, Any]) -> bool:
    prebalance_check = _safe_parse_json_like(row.get("prebalance_check"))
    return bool(isinstance(prebalance_check, dict) and prebalance_check.get("short_circuited") is True)


def _derive_workflow_source(row: dict[str, Any]) -> str:
    explicit_source = str(row.get("workflow_source") or "").strip()
    if explicit_source:
        return explicit_source
    if _is_prebalance_row(row):
        return "prebalance"
    fallback_log = _safe_parse_json_like(row.get("llm_fallback_postprocess"))
    if isinstance(fallback_log, dict) and fallback_log.get("triggered") is True:
        return "fallback"
    bridge_log = _safe_parse_json_like(row.get("llm_species_bridge"))
    if isinstance(bridge_log, dict) and bridge_log.get("triggered") is True:
        return "bridge"
    return "SynRBL"


def _is_atom_balanced(reaction: Any) -> bool:
    text = str(reaction or "").strip()
    if not text:
        return False
    analysis = LLMSpeciesBridge.analyze_reaction_balance(text)
    return bool(analysis.get("is_balanced", False))


def _compute_accuracy_or_blank(actual: Any, expected: Any) -> Any:
    expected_norm = _safe_normalize(expected)
    if expected_norm is None:
        return ""
    actual_norm = _safe_normalize(actual)
    if actual_norm is None:
        return False
    try:
        return bool(wc_similarity(expected_norm, actual_norm, "pathway") >= 1)
    except Exception:
        return False


def _is_known_wrong_reaction(actual: Any, wrong_reactions: Any) -> bool:
    actual_norm = _safe_normalize(actual)
    if actual_norm is None:
        return False
    parsed = _safe_parse_json_like(wrong_reactions)
    if parsed is None:
        parsed = wrong_reactions
    if isinstance(parsed, (set, tuple)):
        values = list(parsed)
    elif isinstance(parsed, list):
        values = parsed
    elif isinstance(parsed, str):
        values = [parsed]
    else:
        return False
    normalized_wrongs = {norm for norm in (_safe_normalize(item) for item in values) if norm is not None}
    return actual_norm in normalized_wrongs


def _normalize_confidence_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    return value


def _derive_workflow_confidence(row: dict[str, Any], workflow_source: str) -> Any:
    explicit_confidence = _normalize_confidence_value(row.get("workflow_confidence"))
    if explicit_confidence != "":
        return explicit_confidence
    if workflow_source == "prebalance":
        return 1.0 if _is_atom_balanced(row.get("formal_output_reaction") or row.get("reaction")) else ""

    confidence = row.get("confidence")
    solved_by = str(row.get("solved_by") or "").strip()
    confidence_value = _normalize_confidence_value(confidence)

    if workflow_source == "SynRBL":
        if confidence_value != "":
            return confidence_value
        if solved_by in {"rule-based", "prebalanced"}:
            return 1.0
        return ""

    if workflow_source == "bridge":
        if confidence_value != "":
            return confidence_value
        if solved_by in {"rule-based", "prebalanced"}:
            return 1.5
        return 1.5 if _is_atom_balanced(row.get("formal_output_reaction") or row.get("reaction")) else ""

    if workflow_source == "fallback":
        return 2.0 if _is_atom_balanced(row.get("formal_output_reaction") or row.get("reaction")) else ""

    return confidence_value


def _ordered_validation_columns(validation_df: pd.DataFrame) -> list[str]:
    return list(validation_df.columns)


def _write_workflow_statistics_xlsx(workflow_df: pd.DataFrame, destination: Path) -> None:
    if not HAS_OPENPYXL:
        return
    workflow_df.to_excel(destination, index=False, engine="openpyxl")


def build_workflow_statistics(
    validation_df: pd.DataFrame,
    original_stage_results: list[dict[str, Any]],
    with_llm_results: list[dict[str, Any]],
    reaction_col: str,
    target_col: str = "expected_reaction",
) -> pd.DataFrame:
    validation_rows = validation_df.to_dict("records")
    if len(validation_rows) != len(original_stage_results) or len(validation_rows) != len(with_llm_results):
        raise ValueError(
            "Workflow statistics requires validation rows, original-stage results, and with-llm results to have identical lengths."
        )

    def _row_index(row: dict[str, Any], fallback_index: int) -> int:
        value = row.get("original_row_index", fallback_index)
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback_index

    validation_columns = _ordered_validation_columns(validation_df)
    original_by_index = {
        _row_index(row, idx): row for idx, row in enumerate(original_stage_results)
    }
    full_by_index = {
        _row_index(row, idx): row for idx, row in enumerate(with_llm_results)
    }

    output_rows: list[dict[str, Any]] = []
    for idx, validation_row in enumerate(validation_rows):
        orig_row = original_by_index.get(idx, {})
        full_row = full_by_index.get(idx, {})

        workflow_source = _derive_workflow_source(full_row)
        orig_output = orig_row.get("formal_output_reaction") or orig_row.get(reaction_col)
        full_output = full_row.get("formal_output_reaction") or full_row.get(reaction_col)

        row_output: dict[str, Any] = {col: validation_row.get(col, "") for col in validation_columns}
        row_output["orig_output_reaction"] = orig_row.get("formal_output_reaction") or orig_output
        row_output["full_output_reaction"] = full_row.get("formal_output_reaction") or full_output

        orig_success = bool(orig_row.get("success", _is_atom_balanced(orig_output)))
        full_success = bool(full_row.get("success", _is_atom_balanced(full_output)))
        row_output["orig_success"] = orig_success
        row_output["orig_accuracy"] = _compute_accuracy_or_blank(orig_output, validation_row.get(target_col))
        row_output["full_success"] = full_success
        row_output["full_accuracy"] = _compute_accuracy_or_blank(full_output, validation_row.get(target_col))
        row_output["orig_known_wrong_match"] = _is_known_wrong_reaction(orig_output, validation_row.get("wrong_reactions"))
        row_output["full_known_wrong_match"] = _is_known_wrong_reaction(full_output, validation_row.get("wrong_reactions"))
        row_output["workflow_source"] = workflow_source
        row_output["orig_confidence"] = _normalize_confidence_value(orig_row.get("workflow_confidence", _derive_workflow_confidence(orig_row, _derive_workflow_source(orig_row))))
        row_output["workflow_confidence"] = _normalize_confidence_value(full_row.get("workflow_confidence", _derive_workflow_confidence(full_row, workflow_source)))

        output_rows.append(row_output)

    final_columns = validation_columns + [
        "orig_output_reaction",
        "full_output_reaction",
        "orig_success",
        "orig_accuracy",
        "full_success",
        "full_accuracy",
        "orig_known_wrong_match",
        "full_known_wrong_match",
        "workflow_source",
        "orig_confidence",
        "workflow_confidence",
    ]
    return pd.DataFrame(output_rows, columns=final_columns)


def write_workflow_statistics(
    output_dir: str | Path,
    validation_df: pd.DataFrame,
    original_stage_results: list[dict[str, Any]],
    with_llm_results: list[dict[str, Any]],
    reaction_col: str,
    target_col: str = "expected_reaction",
) -> Path:
    output_path = Path(output_dir)
    workflow_df = build_workflow_statistics(
        validation_df=validation_df,
        original_stage_results=original_stage_results,
        with_llm_results=with_llm_results,
        reaction_col=reaction_col,
        target_col=target_col,
    )
    destination = output_path / WORKFLOW_STATISTICS_FILENAME
    workflow_df.to_csv(destination, index=False, encoding="utf-8-sig")
    xlsx_destination = output_path / WORKFLOW_STATISTICS_XLSX_FILENAME
    _write_workflow_statistics_xlsx(workflow_df, xlsx_destination)
    return destination


def _evaluate_row(row: dict[str, Any], reaction_col: str, target_col: str) -> dict[str, Any]:
    success = bool(row.get("success", False))
    actual_value = row.get("formal_output_reaction") or row.get(reaction_col)
    expected_norm = _safe_normalize(row.get(target_col))
    actual_norm = _safe_normalize(actual_value)
    has_expected = expected_norm is not None
    comparable = expected_norm is not None and actual_norm is not None
    exact_match = comparable and expected_norm == actual_norm
    similarity = None
    if comparable:
        try:
            similarity = float(wc_similarity(expected_norm, actual_norm, "pathway"))
        except Exception:
            similarity = None
    known_wrong_match = _is_known_wrong_reaction(actual_value, row.get("wrong_reactions"))
    correct = exact_match
    return {
        "raw_solved": success,
        "success": success,
        "has_expected": has_expected,
        "comparable": comparable,
        "exact_match": bool(exact_match),
        "correct": bool(correct),
        "similarity": similarity,
        "known_wrong_match": bool(known_wrong_match),
    }


def _summarize(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    total = len(rows)
    success_cnt = sum(1 for row in rows if row["success"])
    correct_cnt = sum(1 for row in rows if row["correct"])
    comparable_success_cnt = sum(1 for row in rows if row["success"] and row["comparable"])
    missing_expected_cnt = sum(1 for row in rows if not row["has_expected"])
    metrics = {
        "total": total,
        "success_cnt": success_cnt,
        "correct_cnt": correct_cnt,
        "comparable_success_cnt": comparable_success_cnt,
        "missing_expected_cnt": missing_expected_cnt,
        "success_rate": success_cnt / total if total else None,
        "accuracy": correct_cnt / success_cnt if success_cnt else None,
        "strict_accuracy_on_comparable_success": (
            correct_cnt / comparable_success_cnt if comparable_success_cnt else None
        ),
    }
    grouped: dict[str, dict[str, Any]] = {}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(group_key) or "")].append(row)
    for key, bucket in sorted(buckets.items()):
        grouped[key] = {
            "total": len(bucket),
            "success_cnt": sum(1 for row in bucket if row["success"]),
            "correct_cnt": sum(1 for row in bucket if row["correct"]),
            "success_rate": (
                sum(1 for row in bucket if row["success"]) / len(bucket) if bucket else None
            ),
            "accuracy": (
                sum(1 for row in bucket if row["correct"]) / sum(1 for row in bucket if row["success"])
                if sum(1 for row in bucket if row["success"])
                else None
            ),
        }
    return {"overall": metrics, "by_group": grouped}


def build_accuracy_report(
    no_llm_results: list[dict[str, Any]],
    with_llm_results: list[dict[str, Any]],
    reaction_col: str,
    target_col: str = "expected_reaction",
    group_key: str = "solved_by",
) -> dict[str, Any]:
    no_llm_rows = []
    with_llm_rows = []
    for row in no_llm_results:
        merged = dict(row)
        merged.update(_evaluate_row(row, reaction_col, target_col))
        no_llm_rows.append(merged)
    for row in with_llm_results:
        merged = dict(row)
        merged.update(_evaluate_row(row, reaction_col, target_col))
        with_llm_rows.append(merged)

    for idx, row in enumerate(with_llm_rows):
        no_llm_row = no_llm_rows[idx] if idx < len(no_llm_rows) else {}
        is_new_success = (not bool(no_llm_row.get("success", False))) and bool(row.get("success", False))
        row["is_new_success"] = is_new_success
        row["new_success_exact_match"] = bool(is_new_success and row.get("exact_match", False))
        row["new_success_comparable"] = bool(is_new_success and row.get("comparable", False))
        row["new_success_source"] = str(row.get(group_key) or "") if is_new_success else ""

    report = {
        "target_col": target_col,
        "reaction_col": reaction_col,
        "mode_summary": {
            "no_llm": _summarize(no_llm_rows, group_key),
            "with_llm": _summarize(with_llm_rows, group_key),
        },
    }

    new_success_rows = [row for row in with_llm_rows if row.get("is_new_success")]
    report["new_success_summary"] = {
        "new_success_cnt": len(new_success_rows),
        "new_success_exact_match_cnt": sum(1 for row in new_success_rows if row.get("new_success_exact_match")),
        "new_success_comparable_cnt": sum(1 for row in new_success_rows if row.get("new_success_comparable")),
        "new_success_exact_match_ratio": (
            sum(1 for row in new_success_rows if row.get("new_success_exact_match")) / len(new_success_rows)
            if new_success_rows else None
        ),
        "by_source": {
            key: {
                "new_success_cnt": len(bucket),
                "new_success_exact_match_cnt": sum(1 for row in bucket if row.get("new_success_exact_match")),
                "new_success_comparable_cnt": sum(1 for row in bucket if row.get("new_success_comparable")),
                "new_success_exact_match_ratio": (
                    sum(1 for row in bucket if row.get("new_success_exact_match")) / len(bucket)
                    if bucket else None
                ),
            }
            for key, bucket in sorted(
                {
                    source: [row for row in new_success_rows if row.get("new_success_source") == source]
                    for source in {str(row.get("new_success_source") or "") for row in new_success_rows}
                }.items()
            )
        },
    }

    deltas = {}
    for key in ("success_rate", "accuracy", "strict_accuracy_on_comparable_success"):
        left = report["mode_summary"]["no_llm"]["overall"].get(key)
        right = report["mode_summary"]["with_llm"]["overall"].get(key)
        deltas[key] = None if left is None or right is None else right - left
    report["delta_with_llm_minus_no_llm"] = deltas

    return report


def write_accuracy_report(
    output_dir: str | Path,
    no_llm_results: list[dict[str, Any]],
    with_llm_results: list[dict[str, Any]],
    reaction_col: str,
    target_col: str = "expected_reaction",
    group_key: str = "solved_by",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    report = build_accuracy_report(
        no_llm_results=no_llm_results,
        with_llm_results=with_llm_results,
        reaction_col=reaction_col,
        target_col=target_col,
        group_key=group_key,
    )
    (output_path / "accuracy_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    detail_rows = []
    evaluated_no_llm_rows = []
    evaluated_with_llm_rows = []
    for row in no_llm_results:
        detail = dict(row)
        detail.update(_evaluate_row(row, reaction_col, target_col))
        evaluated_no_llm_rows.append(detail)
    for row in with_llm_results:
        detail = dict(row)
        detail.update(_evaluate_row(row, reaction_col, target_col))
        evaluated_with_llm_rows.append(detail)

    for idx, row in enumerate(evaluated_no_llm_rows):
        detail = dict(row)
        detail["mode"] = "no_llm"
        detail["is_new_success"] = False
        detail["new_success_exact_match"] = False
        detail["new_success_source"] = ""
        detail["new_success_comparable"] = False
        detail_rows.append(detail)

    for idx, row in enumerate(evaluated_with_llm_rows):
        no_llm_row = evaluated_no_llm_rows[idx] if idx < len(evaluated_no_llm_rows) else {}
        is_new_success = (not bool(no_llm_row.get("success", False))) and bool(row.get("success", False))
        detail = dict(row)
        detail["mode"] = "with_llm"
        detail["is_new_success"] = is_new_success
        detail["new_success_exact_match"] = bool(is_new_success and row.get("exact_match", False))
        detail["new_success_source"] = str(row.get(group_key) or "") if is_new_success else ""
        detail["new_success_comparable"] = bool(is_new_success and row.get("comparable", False))
        detail_rows.append(detail)

    pd.DataFrame(detail_rows).to_csv(
        output_path / "accuracy_comparison_detail.csv", index=False, encoding="utf-8-sig"
    )
    return report
