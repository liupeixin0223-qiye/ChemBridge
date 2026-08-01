import pandas as pd
import re

from rdkit import Chem
from typing import List, Dict, Tuple, Optional

from synrbl.SynProcessor import RSMIProcessing
from synrbl.SynUtils import remove_atom_mapping


def input_sanitize_check(
    reaction_smiles: str,
    reaction_id: Optional[str] = None,
) -> Dict:
    """对输入的反应 SMILES 进行系统性验证。

    在 preprocess 之前调用，作为管线的第一步。

    参数:
        reaction_smiles: 输入的反应 SMILES 字符串，格式为 "反应物>>产物"
        reaction_id: 可选的反应标识符，用于错误追踪

    返回:
        包含验证结果的字典:
        - valid (bool): 是否通过验证
        - reactants (list): 解析后的反应物 SMILES 列表
        - products (list): 解析后的产物 SMILES 列表
        - errors (list): 错误信息列表
        - reaction_id (str): 反应标识符
    """
    result: Dict = {
        "valid": False,
        "reactants": [],
        "products": [],
        "errors": [],
        "reaction_id": reaction_id or "",
    }

    # 检查基本格式：必须包含 ">>" 分隔符
    if ">>" not in reaction_smiles:
        result["errors"].append(
            f"反应 SMILES 格式错误：缺少 '>>' 分隔符。"
            f"输入值: '{reaction_smiles}'"
        )
        return result

    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        result["errors"].append(
            f"反应 SMILES 格式错误：'>>' 分隔符数量不正确，"
            f"期望 1 个，实际 {len(parts) - 1} 个。"
        )
        return result

    reactants_str, products_str = parts[0], parts[1]

    # 检查反应物侧或产物侧是否为空
    if not reactants_str.strip():
        result["errors"].append("反应物侧为空。")
        return result
    if not products_str.strip():
        result["errors"].append("产物侧为空。")
        return result

    def _remove_atom_mapping(smiles_token: str) -> str:
        """去除单个 SMILES token 中的原子映射编号。"""
        return re.sub(r":\d+", "", smiles_token)

    # 逐一验证反应物侧的每个 SMILES 片段
    reactant_tokens = reactants_str.split(".")
    for i, token in enumerate(reactant_tokens):
        token = token.strip()
        if not token:
            result["errors"].append(
                f"反应物侧第 {i + 1} 个片段为空字符串。"
            )
            continue

        cleaned_token = _remove_atom_mapping(token)
        mol = Chem.MolFromSmiles(cleaned_token)
        if mol is None:
            result["errors"].append(
                f"SMILES parsing error: 反应物侧第 {i + 1} 个片段 "
                f"'{token}' 无法被 RDKit 解析为有效分子。"
            )
        else:
            canonical = Chem.MolToSmiles(mol)
            result["reactants"].append(canonical)

    # 逐一验证产物侧的每个 SMILES 片段
    product_tokens = products_str.split(".")
    for i, token in enumerate(product_tokens):
        token = token.strip()
        if not token:
            result["errors"].append(
                f"产物侧第 {i + 1} 个片段为空字符串。"
            )
            continue

        cleaned_token = _remove_atom_mapping(token)
        mol = Chem.MolFromSmiles(cleaned_token)
        if mol is None:
            result["errors"].append(
                f"SMILES parsing error: 产物侧第 {i + 1} 个片段 "
                f"'{token}' 无法被 RDKit 解析为有效分子。"
            )
        else:
            canonical = Chem.MolToSmiles(mol)
            result["products"].append(canonical)

    # 如果没有任何错误，标记为有效
    if not result["errors"]:
        result["valid"] = True

    return result


def batch_input_sanitize_check(
    reactions_df: pd.DataFrame,
    reaction_column: str = "reactions",
) -> Tuple[List[Dict], List[Dict]]:
    """批量验证数据框中的所有反应 SMILES。

    参数:
        reactions_df: 包含反应数据的 DataFrame
        reaction_column: 反应 SMILES 所在的列名

    返回:
        (valid_reactions, invalid_reactions) 元组:
        - valid_reactions: 通过验证的反应列表
        - invalid_reactions: 未通过验证的反应列表（含错误信息）
    """
    valid_reactions: List[Dict] = []
    invalid_reactions: List[Dict] = []

    for idx, row in reactions_df.iterrows():
        rxn_smiles = str(row[reaction_column])
        rxn_id = str(row.get("id", row.get("reaction_id", f"row_{idx}")))

        check_result = input_sanitize_check(rxn_smiles, reaction_id=rxn_id)

        if check_result["valid"]:
            valid_reactions.append(
                {
                    "original_smiles": rxn_smiles,
                    "reactants": check_result["reactants"],
                    "products": check_result["products"],
                    "reaction_id": rxn_id,
                    "row_index": idx,
                }
            )
        else:
            invalid_reactions.append(
                {
                    "original_smiles": rxn_smiles,
                    "reaction_id": rxn_id,
                    "row_index": idx,
                    "errors": check_result["errors"],
                    "stage": "input_sanitize",
                    "status": "SMILES parsing error",
                }
            )

    return valid_reactions, invalid_reactions


def preprocess(
    reactions,
    reaction_col,
    index_col,
    solved_col,
    input_col,
    n_jobs=1,
    remove_aam=False,
):
    normalized_reactions = []
    for row_index, reaction in enumerate(reactions):
        entry = reaction.copy()
        entry.setdefault("original_row_index", row_index)
        entry.setdefault("original_reaction", entry.get(reaction_col))
        entry.setdefault("preprocess_status", "pending")
        entry.setdefault("processable", True)
        entry.setdefault("issue", entry.get("issue", ""))

        if remove_aam:
            raw_reaction = entry.get(reaction_col)
            try:
                if raw_reaction is not None:
                    entry[reaction_col] = remove_atom_mapping(raw_reaction)
                    entry["atom_mapping_removed"] = True
                    entry["preprocess_status"] = "atom_mapping_removed"
                else:
                    entry["atom_mapping_removed"] = False
            except Exception as exc:
                entry[reaction_col] = raw_reaction
                entry["atom_mapping_removed"] = False
                entry["preprocess_status"] = "atom_mapping_removal_failed"
                entry["processable"] = False
                if not entry["issue"]:
                    entry["issue"] = f"Atom mapping removal failed: {exc}"
        else:
            entry["atom_mapping_removed"] = False
            entry["preprocess_status"] = "skipped"

        normalized_reactions.append(entry)

    df = pd.DataFrame(normalized_reactions)
    df[solved_col] = False

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    process = RSMIProcessing(
        data=df,
        rsmi_col=reaction_col,
        parallel=True,
        n_jobs=n_jobs,
        data_name="internal",  # type: ignore
        index_col=index_col,
        drop_duplicates=False,
        save_json=False,
        save_path_name=None,  # type: ignore
        verbose=0,
    )
    reactions_df = process.data_splitter()
    reactions_df[input_col] = reactions_df[reaction_col]

    if "can_parse_reaction" in reactions_df.columns:
        unparsable_mask = ~reactions_df["can_parse_reaction"].fillna(False)
        reactions_df.loc[unparsable_mask, "processable"] = False
        reactions_df.loc[unparsable_mask, "preprocess_status"] = "unparseable"
        empty_issue_mask = unparsable_mask & reactions_df["issue"].fillna("").eq("")
        reactions_df.loc[empty_issue_mask, "issue"] = reactions_df.loc[
            empty_issue_mask, "parse_issue"
        ].fillna("Reaction SMILES cannot be parsed.")

    return reactions_df.to_dict("records")
