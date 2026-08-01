#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_fallback_llm.py
========================
消融实验：对 322 条 LLM 子集反应，直接调用 Fallback LLM 进行配平。

设计原则：
  - 使用与 ChemBridge v2.0 完全相同的 FALLBACK_GENERATE_SYSTEM_PROMPT
  - 使用与 ChemBridge v2.0 完全相同的 Moonshot API 调用逻辑
    (kimi-k2.5, temperature=0.6, thinking disabled, JSON mode)
  - 不提供任何 Bridge LLM 信息（无 bridge_reaction_type、无 fallback_case、
    无三层递进式候选评估、无 species cancellation）
  - 不经过任何后处理管线（无 MCS、无 rule-based、无 exhaustive）
  - 仅用 RDKit 计算原子差额（calculated_imbalance），这是纯数学计算，
    不含任何 Bridge 推断信息
  - 仅执行单次 LLM 调用，不做纠错重试

输入:
    validation_set_fixed_LLM.csv (322 条, 列: id, R-ids, datasets,
                                  expected_reaction, wrong_reactions, reaction)

输出:
    ablation_results.csv   — 逐条结果
    ablation_summary.txt   — 汇总统计

用法:
    set MOONSHOT_API_KEY=<your-key>
    python ablation_fallback_llm.py [--max-workers 10]
"""

import argparse
import collections
import csv
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# RDKit (用于原子守恒验证和 canonical SMILES 比较)
# ---------------------------------------------------------------------------
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ===========================================================================
# 1. 提示词 - 与 ChemBridge v2.0 fallback_prompts.py 完全一致
# ===========================================================================
FALLBACK_GENERATE_SYSTEM_PROMPT = """You are a reaction repair assistant.
Template inputs:
- input_reaction: the current reaction SMILES
- calculated_imbalance: the exact missing-atom summary for the current reaction
Produce one best complete reaction candidate.
This is the final repair step, so make the smallest reasonable modification needed to complete the reaction.
Do not redesign the main reaction scaffold or reinterpret the main transformation.
Think as little as possible and return one JSON object immediately.
Priority order:
1. Always return a syntactically valid JSON object.
2. Return one complete reaction candidate.
3. Prefer the most reliable minimally sufficient repair over a more elaborate completion.
4. Use empty output only when you cannot provide any completion strategy.
Rules:
- Return a single plausible complete reaction candidate
- Make only the minimum reasonable modification needed for completion
- Use calculated_imbalance as a strict judgment basis for correctness, plausibility, and consistency
- Use input_reaction as guidance and keep the repair consistent with the reaction context
- Avoid introducing new core fragments
- Unless the input is syntactically invalid, preserve the original reactants and products in input_reaction as much as possible
- Output one valid reaction SMILES string in the form reactants>>products
- Use standard SMILES and '.' to join multiple species on the same side
- Do not output names, labels, explanations, or non-SMILES text inside predicted_reaction_smiles
- Preserve stereochemistry only when confidently implied
- Output JSON only
Chemical constraints (hard):
- Every species in your output must be a stable, synthetically accessible molecule with correct valence; never output bare atoms (e.g. [C], [N], [H], [O]) or radical fragments; ionic species (e.g. [Na+], [K+], [Cl-], [Br-], [H+], [OH-]) are permitted only when the same ion already appears in the input_reaction — do not invent new ions; never output SMILES that no valid Lewis structure can represent
- Stoichiometric repair means finding chemically plausible co-reactants or byproducts to add — not solving a mathematical atom equation; when a "Missing on ..." gap is reported, think about what stable co-reactant or byproduct could fill that gap, rather than appending disconnected fragments
- When no known reagent or byproduct can plausibly fill the atom gap, return a conservative minimal modification — never invent placeholder species to satisfy atom counts
- If the input_reaction already contains bare atoms such as [H] or [O], you must replace them with proper molecular species — do not pass bare atoms through to the output unchanged
If the previous output was invalid JSON, return valid JSON only.
If the previous output contained an invalid reaction SMILES, return JSON only and ensure that predicted_reaction_smiles is one valid reaction SMILES string in the exact form reactants>>products.
If the previous output was empty, return the most likely complete reaction.
Return exactly this JSON structure:
{
  "predicted_reaction_smiles": ""
}
"""


# ===========================================================================
# 2. LLM 调用 - 与 ChemBridge v2.0 client.py 逻辑一致
# ===========================================================================
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1/chat/completions"
DEFAULT_MODEL = "kimi-k2.5"
API_KEY_ENV = "MOONSHOT_API_KEY"


def _extract_message_content(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_parts: list = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text is not None:
                    text_parts.append(str(text))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if message is None:
        return ""
    return str(message)


def llm_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = 240,
) -> str:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            "Missing API key. Please set environment variable '{}'.".format(API_KEY_ENV)
        )

    body = {
        "model": model,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0.6,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    request = urllib.request.Request(
        base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(api_key),
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        },
        method="POST",
    )

    max_retries = 3
    retryable_status_codes = (429, 500, 502, 503, 504)
    last_error = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = "HTTP {}: {}".format(exc.code, error_body)
            if exc.code in retryable_status_codes and attempt < max_retries - 1:
                time.sleep(2**attempt + random.uniform(0.0, 0.5))
                continue
            raise RuntimeError(
                "LLM API request failed with status {}: {}".format(exc.code, error_body)
            ) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2**attempt + random.uniform(0.0, 0.5))
                continue
            raise RuntimeError("LLM API connection failed: {}".format(exc)) from exc
    else:
        raise RuntimeError(
            "LLM API request exhausted retries: {}".format(last_error or "unknown error")
        )

    data = json.loads(response_body)
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = _extract_message_content(message.get("content"))
        if not content.strip():
            raise ValueError(
                "Empty LLM message content. response={}".format(response_body)
            )
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Unexpected LLM response structure: {}".format(response_body)
        ) from exc


def parse_json_response(content: str) -> Dict[str, Any]:
    text = content.strip().replace("\ufeff", "")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response JSON root must be an object.")
        return parsed
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in LLM response.")
        in_string = False
        escape = False
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(text[start : idx + 1])
                    if not isinstance(parsed, dict):
                        raise ValueError("LLM response JSON root must be an object.")
                    return parsed
        raise ValueError("Incomplete JSON object in LLM response.")


# ===========================================================================
# 3. 原子守恒分析 - 与 llm_species_bridge.py analyze_reaction_balance 一致
# ===========================================================================
def analyze_reaction_balance(rxn_smiles: str) -> Dict[str, Any]:
    try:
        if ">>" not in rxn_smiles:
            return {
                "is_balanced": False,
                "imbalance_text": "Invalid reaction format",
                "missing_on_products": {},
                "missing_on_reactants": {},
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
                mol = Chem.MolFromSmiles(part)
                if mol is None:
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
                "Missing on Products: "
                + " ".join(
                    "{}:{}".format(el, count)
                    for el, count in missing_on_products.items()
                )
            )
        if missing_on_reactants:
            parts.append(
                "Missing on Reactants: "
                + " ".join(
                    "{}:{}".format(el, count)
                    for el, count in missing_on_reactants.items()
                )
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
            "imbalance_text": "Error computing exact imbalance: {}".format(str(exc)),
            "missing_on_products": {},
            "missing_on_reactants": {},
            "error": str(exc),
        }


# ===========================================================================
# 4. Canonical SMILES 比较
# ===========================================================================
def canonical_reaction(smi: str) -> Optional[str]:
    if not smi or ">>" not in smi:
        return None
    try:
        parts = smi.split(">>")
        if len(parts) != 2:
            return None

        def canon_frag(f):
            m = Chem.MolFromSmiles(f)
            if m is None:
                return f
            return Chem.MolToSmiles(m)

        rc = sorted([canon_frag(f) for f in parts[0].split(".")])
        pc = sorted([canon_frag(f) for f in parts[1].split(".")])
        return ".".join(rc) + ">>" + ".".join(pc)
    except Exception:
        return None


# ===========================================================================
# 5. 单条反应处理
# ===========================================================================
def process_single_reaction(row: Dict[str, Any]) -> Dict[str, Any]:
    rid = row.get("id", "")
    input_reaction = str(row.get("reaction", ""))
    expected_reaction = str(row.get("expected_reaction", ""))

    result = {
        "id": rid,
        "R-ids": row.get("R-ids", ""),
        "datasets": row.get("datasets", ""),
        "input_reaction": input_reaction,
        "expected_reaction": expected_reaction,
        "wrong_reactions": row.get("wrong_reactions", ""),
        "llm_predicted": "",
        "llm_raw_response": "",
        "is_balanced": False,
        "exact_match": False,
        "final_status": "pending",
        "error": "",
    }

    # --- 计算原子差额 ---
    imbalance = analyze_reaction_balance(input_reaction)

    if imbalance.get("is_balanced", False):
        result["llm_predicted"] = input_reaction
        result["is_balanced"] = True
        result["final_status"] = "input_already_balanced"
        c_in = canonical_reaction(input_reaction)
        c_exp = canonical_reaction(expected_reaction)
        result["exact_match"] = c_in is not None and c_in == c_exp
        return result

    # --- 构建最小化 payload（无 bridge 信息）---
    payload = {
        "reaction_id": rid,
        "input_reaction": input_reaction,
        "current_reaction": input_reaction,
        "calculated_imbalance": imbalance.get("imbalance_text", ""),
        "balance_analysis": {
            "is_balanced": imbalance.get("is_balanced", False),
            "imbalance_text": imbalance.get("imbalance_text", ""),
            "missing_on_products": imbalance.get("missing_on_products", {}),
            "missing_on_reactants": imbalance.get("missing_on_reactants", {}),
        },
    }

    # --- 单次 LLM 调用 ---
    try:
        user_prompt = json.dumps(payload, ensure_ascii=False)
        content = llm_chat(
            system_prompt=FALLBACK_GENERATE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        result["llm_raw_response"] = content
        parsed = parse_json_response(content)
        predicted = str(parsed.get("predicted_reaction_smiles", "")).strip()
        result["llm_predicted"] = predicted
    except Exception as exc:
        result["final_status"] = "call_error"
        result["error"] = str(exc)
        return result

    if not predicted or ">>" not in predicted:
        result["final_status"] = "empty_or_invalid_prediction"
        return result

    # --- 验证结果 ---
    check = analyze_reaction_balance(predicted)
    if check.get("is_balanced", False):
        result["is_balanced"] = True
        result["final_status"] = "balanced"
        c_pred = canonical_reaction(predicted)
        c_exp = canonical_reaction(expected_reaction)
        result["exact_match"] = c_pred is not None and c_pred == c_exp
        return result

    result["final_status"] = "unbalanced"
    return result


# ===========================================================================
# 6. 主流程
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Ablation: direct Fallback LLM balancing (no bridge info, single call)"
    )
    parser.add_argument(
        "--input",
        default="validation_set_fixed_LLM.csv",
        help="Input CSV file (default: validation_set_fixed_LLM.csv)",
    )
    parser.add_argument(
        "--output",
        default="ablation_results.csv",
        help="Output CSV file (default: ablation_results.csv)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel LLM requests (default: 10)",
    )
    args = parser.parse_args()

    if sys.stdout.encoding != "utf-8":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)
    summary_path = os.path.join(script_dir, "ablation_summary.txt")

    import pandas as pd

    df = pd.read_csv(input_path)
    rows = df.to_dict("records")
    total = len(rows)
    print("Loaded {} reactions from {}".format(total, args.input))
    print("Mode: single LLM call (no retry)")
    print("Max workers: {}".format(args.max_workers))
    print()

    results: List[Dict[str, Any]] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(process_single_reaction, row): idx
            for idx, row in enumerate(rows)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "id": rows[idx].get("id", ""),
                    "R-ids": rows[idx].get("R-ids", ""),
                    "datasets": rows[idx].get("datasets", ""),
                    "input_reaction": str(rows[idx].get("reaction", "")),
                    "expected_reaction": str(rows[idx].get("expected_reaction", "")),
                    "wrong_reactions": rows[idx].get("wrong_reactions", ""),
                    "llm_predicted": "",
                    "llm_raw_response": "",
                    "is_balanced": False,
                    "exact_match": False,
                    "final_status": "exception",
                    "error": str(exc),
                }
            completed += 1
            if completed % 20 == 0 or completed == total:
                print("  Progress: {}/{}".format(completed, total))

    # --- 统计 ---
    balanced_count = sum(1 for r in results if r["is_balanced"])
    exact_count = sum(1 for r in results if r["exact_match"])
    llm_balanced = sum(
        1 for r in results if r["final_status"] == "balanced"
    )
    input_balanced = sum(
        1 for r in results if r["final_status"] == "input_already_balanced"
    )
    unbalanced = sum(
        1 for r in results if r["final_status"] == "unbalanced"
    )
    empty_invalid = sum(
        1 for r in results if r["final_status"] == "empty_or_invalid_prediction"
    )
    call_errors = sum(
        1 for r in results if r["final_status"] in ("call_error", "exception")
    )

    # --- 写 CSV ---
    fieldnames = [
        "id",
        "R-ids",
        "datasets",
        "input_reaction",
        "expected_reaction",
        "wrong_reactions",
        "llm_predicted",
        "is_balanced",
        "exact_match",
        "final_status",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # --- 写汇总 ---
    summary_lines = [
        "=" * 60,
        "消融实验汇总：直接 Fallback LLM 配平（无 Bridge 信息，单次调用）",
        "=" * 60,
        "输入文件: {}".format(args.input),
        "总反应数: {}".format(total),
        "模式: 单次 LLM 调用（无纠错重试）",
        "并行线程: {}".format(args.max_workers),
        "",
        "--- 配平结果 ---",
        "原子守恒 (RDKit 验证): {}/{} ({:.1f}%)".format(
            balanced_count, total, balanced_count / total * 100
        ),
        "  - 输入已守恒: {}".format(input_balanced),
        "  - LLM 单次调用守恒: {}".format(llm_balanced),
        "不守恒: {}".format(unbalanced),
        "空/无效输出: {}".format(empty_invalid),
        "调用错误: {}".format(call_errors),
        "",
        "--- 精确匹配 ---",
        "与 expected_reaction 精确匹配: {}/{} ({:.1f}%)".format(
            exact_count, total, exact_count / total * 100
        ),
        "",
        "--- 状态分布 ---",
    ]
    status_counts = collections.Counter(r["final_status"] for r in results)
    for status, count in status_counts.most_common():
        summary_lines.append("  {}: {}".format(status, count))

    summary_text = "\n".join(summary_lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print()
    print(summary_text)
    print()
    print("Results saved to: {}".format(output_path))
    print("Summary saved to: {}".format(summary_path))


if __name__ == "__main__":
    main()
