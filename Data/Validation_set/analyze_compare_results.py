import json
import os
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from rdkit import Chem
from rdkit import RDLogger

BASE_DIR = Path(r"D:\SynRBL-main_old\Data\Validation_set")
INPUT_CANDIDATES = [
    BASE_DIR / "compare results.xls",
    BASE_DIR / "compare results.xlsx",
    BASE_DIR / "对比结果.xls",
    BASE_DIR / "对比结果.xlsx",
]
CSV_FILE = BASE_DIR / "对比结果_llm_friendly.csv"
REVIEW_FILE = BASE_DIR / "对比结果_llm_review.csv"
SUMMARY_FILE = BASE_DIR / "对比结果_summary.json"
DETAIL_FILE = BASE_DIR / "对比结果_detailed_report.json"

MODEL_NAME = "kimi-k2-0905-preview"
BASE_URL = "https://api.moonshot.cn/v1"
MAX_WORKERS = min(100, max(1, int(os.getenv("KIMI_MAX_WORKERS", "20"))))
REQUEST_TIMEOUT = int(os.getenv("KIMI_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("KIMI_MAX_RETRIES", "2"))
MAX_ROWS = int(os.getenv("KIMI_MAX_ROWS", "0"))

EXPECTED_COLUMNS = ["序号", "原输入", "模型结果", "标准答案", "问题"]
ARROW_PATTERN = re.compile(r">>|=>|->|→|⟶|⟹|=\s*>")
ATOM_MAP_PATTERN = re.compile(r":\d+(?=[\]\)])")
WHITESPACE_PATTERN = re.compile(r"\s+")
SYSTEM_PROMPT = (
    "You evaluate reaction equivalence and completion quality. Ignore atom mapping, reactant/product order, SMILES writing style, and minor reagent/byproduct differences. "
    "Focus on the main transformation only. Do not count atoms. Return JSON only: {\"success\": true/false}."
)

HEADER_ALIASES = {
    "序号": ["序号", "id", "index", "编号"],
    "原输入": ["原输入", "原始输入", "input", "原始数据"],
    "模型结果": ["模型结果", "结果", "model_result", "模型输出"],
    "标准答案": ["标准答案", "人工标注", "answer", "gold", "golden", "参考答案"],
    "问题": ["问题", "问题（与标准答案互斥）", "备注", "issue", "comment", "reason", "异常原因"],
}

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


def normalize_header_name(name: str) -> str:
    text = str(name).strip()
    lowered = text.lower()
    for target, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            if lowered == alias.lower():
                return target
    return text


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def strip_atom_mapping(text: str) -> str:
    return ATOM_MAP_PATTERN.sub("", clean_text(text))


def normalize_reaction_text(text: str) -> str:
    value = strip_atom_mapping(text)
    value = ARROW_PATTERN.sub(">>", value)
    value = WHITESPACE_PATTERN.sub("", value)
    return value


def split_reaction(reaction: str) -> Tuple[str, str]:
    if ">>" not in reaction:
        return reaction, ""
    return reaction.split(">>", 1)


def canonicalize_molecule(mol_text: str) -> str:
    mol_text = mol_text.strip()
    if not mol_text:
        return ""
    mol = Chem.MolFromSmiles(mol_text)
    if mol is None:
        return mol_text
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_side(side: str) -> str:
    parts = [x.strip() for x in side.split(".") if x.strip()]
    parts = [canonicalize_molecule(x) for x in parts]
    parts.sort()
    return ".".join(parts)


def canonicalize_reaction(text: str) -> str:
    reaction = normalize_reaction_text(text)
    if not reaction:
        return ""
    left, right = split_reaction(reaction)
    return f"{canonicalize_side(left)}>>{canonicalize_side(right)}"


def choose_target(answer: str, issue: str) -> Tuple[str, str]:
    answer = clean_text(answer)
    issue = clean_text(issue)
    if answer:
        return "标准答案", answer
    if issue:
        return "问题", issue
    return "缺失", ""


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("未找到输入文件，请检查 compare results.xls/xlsx 或 对比结果.xls/xlsx")


def load_excel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xls":
        return pd.read_excel(path, engine="xlrd")
    return pd.read_excel(path)


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={col: normalize_header_name(col) for col in df.columns})
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}；当前列: {list(df.columns)}")
    return df[EXPECTED_COLUMNS].copy()


def atom_counter_for_side(side: str) -> Tuple[Counter, List[str]]:
    total = Counter()
    bad = []
    for part in [x for x in side.split(".") if x.strip()]:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            bad.append(part)
            continue
        for atom in mol.GetAtoms():
            total[atom.GetSymbol()] += 1
    return total, bad


def balance_check(reaction: str) -> Dict[str, object]:
    reaction = normalize_reaction_text(reaction)
    if not reaction or ">>" not in reaction:
        return {
            "is_parseable": False,
            "is_balanced": False,
            "reason": "missing_arrow",
            "pattern": "missing_arrow",
            "parse_errors": [],
        }
    left, right = split_reaction(reaction)
    lc, lb = atom_counter_for_side(left)
    rc, rb = atom_counter_for_side(right)
    if lb or rb:
        return {
            "is_parseable": False,
            "is_balanced": False,
            "reason": "smiles_parse_failed",
            "pattern": "smiles_parse_failed",
            "parse_errors": lb + rb,
        }
    left_only = {k: lc[k] - rc[k] for k in lc if lc[k] > rc[k]}
    right_only = {k: rc[k] - lc[k] for k in rc if rc[k] > lc[k]}
    if not left_only and not right_only:
        return {
            "is_parseable": True,
            "is_balanced": True,
            "reason": "ok",
            "pattern": "balanced",
            "parse_errors": [],
        }
    pattern = []
    if left_only:
        pattern.append("L:" + ",".join(f"{k}+{v}" for k, v in sorted(left_only.items())))
    if right_only:
        pattern.append("R:" + ",".join(f"{k}+{v}" for k, v in sorted(right_only.items())))
    return {
        "is_parseable": True,
        "is_balanced": False,
        "reason": "atom_count_mismatch",
        "pattern": " | ".join(pattern),
        "parse_errors": [],
    }


def build_llm_prompt(row: pd.Series) -> str:
    if row["evaluation_target_type"] == "标准答案":
        return (
            "Judge whether model and gold mean the same reaction. "
            "Ignore atom mapping, SMILES style differences, molecule order, and omission/addition of obvious small byproducts or reagents such as H2O, O, H, simple acids/alcohols when the main transformation is the same. "
            "Return true if the core chemical transformation matches. Do not count atoms.\n"
            f"input={row['原输入_规范化']}\n"
            f"model={row['模型结果_规范化']}\n"
            f"gold={row['标准答案_规范化']}\n"
            '{"success": true/false}'
        )
    return (
        "Judge whether the model completion is chemically reasonable. "
        "Ignore atom mapping, SMILES style differences, and minor reagent/byproduct differences. "
        "Use input and issue as context. Return true if the completion is a plausible and useful correction/completion of the reaction. Do not count atoms.\n"
        f"input={row['原输入_规范化']}\n"
        f"model={row['模型结果_规范化']}\n"
        f"issue={row['问题']}\n"
        f"parse_problem={row['模型结果_配平检查原因']}\n"
        f"parse_fragments={row['配平解析失败片段']}\n"
        '{"success": true/false}'
    )


def safe_json_success(content: str) -> bool:
    data = json.loads(content)
    return bool(data.get("success", False))


def call_kimi(prompt: str, client: OpenAI) -> bool:
    last_error = None
    for retry in range(MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                timeout=REQUEST_TIMEOUT,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return safe_json_success(completion.choices[0].message.content)
        except Exception as exc:
            last_error = exc
            if retry < MAX_RETRIES:
                time.sleep(1.5 * (retry + 1))
    raise last_error


def progress_line(done: int, total: int) -> str:
    width = 30
    filled = 0 if total == 0 else int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {done}/{total}"


def save_outputs(df: pd.DataFrame) -> None:
    review_cols = [
        "序号", "原输入", "模型结果", "标准答案", "问题",
        "evaluation_target_type", "程序判定_exact_hit",
        "模型结果_配平检查可解析", "模型结果_是否配平", "模型结果_配平检查原因",
        "未配平_模式", "配平解析失败片段",
        "llm_needed", "llm_success", "llm_done", "llm_error"
    ]
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    df.loc[df["llm_needed"], review_cols].to_csv(REVIEW_FILE, index=False, encoding="utf-8-sig")


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    standard_rows = df[df["evaluation_target_type"] == "标准答案"]
    issue_rows = df[df["evaluation_target_type"] == "问题"]
    need_llm = df[df["llm_needed"]]
    unbalanced = df[df["模型结果_是否配平"] == False]
    parse_failed = df[df["模型结果_配平检查可解析"] == False]
    return {
        "total_rows": int(len(df)),
        "rows_with_standard_answer": int(len(standard_rows)),
        "rows_with_issue_only": int(len(issue_rows)),
        "rows_missing_both": int((df["evaluation_target_type"] == "缺失").sum()),
        "exact_hit_by_program": int(df["程序判定_exact_hit"].sum()),
        "need_llm_review": int(len(need_llm)),
        "llm_done_count": int(df["llm_done"].sum()),
        "llm_true_count": int((df["llm_success"] == True).sum()),
        "llm_false_count": int((df["llm_success"] == False).sum()),
        "llm_error_count": int(df["llm_error"].astype(bool).sum()),
        "standard_answer_success_total": int(((df["evaluation_target_type"] == "标准答案") & (df["llm_success"] == True)).sum() + df["程序判定_exact_hit"].sum()),
        "issue_only_success_total": int(((df["evaluation_target_type"] == "问题") & (df["llm_success"] == True)).sum()),
        "unbalanced_model_result_count": int(len(unbalanced)),
        "balance_parse_failed_count": int(len(parse_failed)),
        "top_unbalanced_patterns": unbalanced["未配平_模式"].value_counts().head(20).to_dict(),
        "top_parse_fail_fragments": parse_failed["配平解析失败片段"].value_counts().head(20).to_dict(),
        "max_rows_used": int(len(df)),
    }


def build_detailed_report(df: pd.DataFrame) -> Dict[str, object]:
    return {
        "standard_answer": {
            "total": int((df["evaluation_target_type"] == "标准答案").sum()),
            "program_exact_hit": int(df["程序判定_exact_hit"].sum()),
            "llm_true": int(((df["evaluation_target_type"] == "标准答案") & (df["llm_success"] == True)).sum()),
            "llm_false": int(((df["evaluation_target_type"] == "标准答案") & (df["llm_success"] == False)).sum()),
        },
        "issue_only": {
            "total": int((df["evaluation_target_type"] == "问题").sum()),
            "llm_true": int(((df["evaluation_target_type"] == "问题") & (df["llm_success"] == True)).sum()),
            "llm_false": int(((df["evaluation_target_type"] == "问题") & (df["llm_success"] == False)).sum()),
        },
        "balance": {
            "balanced": int((df["模型结果_是否配平"] == True).sum()),
            "unbalanced": int((df["模型结果_是否配平"] == False).sum()),
            "parse_failed": int((df["模型结果_配平检查可解析"] == False).sum()),
            "pattern_counts": df[df["模型结果_是否配平"] == False]["未配平_模式"].value_counts().to_dict(),
        },
        "llm_errors": df[df["llm_error"].astype(bool)][["序号", "llm_error"]].to_dict(orient="records"),
    }


def main() -> None:
    input_file = find_input_file()
    df = validate_columns(load_excel(input_file))
    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS).copy()

    for col in EXPECTED_COLUMNS:
        df[col] = df[col].map(clean_text)

    df["原输入_规范化"] = df["原输入"].map(normalize_reaction_text)
    df["模型结果_规范化"] = df["模型结果"].map(normalize_reaction_text)
    df["标准答案_规范化"] = df["标准答案"].map(normalize_reaction_text)
    df["模型结果_签名"] = df["模型结果"].map(canonicalize_reaction)
    df["标准答案_签名"] = df["标准答案"].map(canonicalize_reaction)

    targets = df.apply(lambda row: choose_target(row["标准答案"], row["问题"]), axis=1)
    df["evaluation_target_type"] = [x[0] for x in targets]
    df["evaluation_target_text"] = [x[1] for x in targets]

    df["程序判定_exact_hit"] = df.apply(
        lambda row: row["evaluation_target_type"] == "标准答案"
        and bool(row["模型结果_签名"])
        and row["模型结果_签名"] == row["标准答案_签名"],
        axis=1,
    )

    balance_results = df["模型结果"].map(balance_check)
    df["模型结果_配平检查可解析"] = balance_results.map(lambda x: x["is_parseable"])
    df["模型结果_是否配平"] = balance_results.map(lambda x: x["is_balanced"])
    df["模型结果_配平检查原因"] = balance_results.map(lambda x: x["reason"])
    df["未配平_模式"] = balance_results.map(lambda x: x["pattern"])
    df["配平解析失败片段"] = balance_results.map(lambda x: json.dumps(x["parse_errors"], ensure_ascii=False))

    df["llm_needed"] = True
    df["llm_prompt"] = df.apply(build_llm_prompt, axis=1)
    df["llm_success"] = pd.NA
    df["llm_done"] = False
    df["llm_error"] = ""

    api_key = os.getenv("MOONSHOT_API_KEY", "").strip()
    if api_key:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        review_index = df.index[df["llm_needed"]].tolist()
        total = len(review_index)
        if total:
            print(progress_line(0, total))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(call_kimi, df.at[idx, "llm_prompt"], client): idx
                    for idx in review_index
                }
                done = 0
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        df.at[idx, "llm_success"] = bool(future.result())
                        df.at[idx, "llm_done"] = True
                    except Exception as exc:
                        df.at[idx, "llm_error"] = str(exc)
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(progress_line(done, total), flush=True)
                        save_outputs(df)
    else:
        df.loc[df["llm_needed"], "llm_error"] = "MOONSHOT_API_KEY not set"

    save_outputs(df)
    summary = summarize(df)
    detail = build_detailed_report(df)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    DETAIL_FILE.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成")
    print(f"输入文件: {input_file}")
    print(f"并发数: {MAX_WORKERS}")
    print(f"本次处理条数: {len(df)}")
    print(f"主结果: {CSV_FILE}")
    print(f"复核结果: {REVIEW_FILE}")
    print(f"汇总结果: {SUMMARY_FILE}")
    print(f"详细报表: {DETAIL_FILE}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
