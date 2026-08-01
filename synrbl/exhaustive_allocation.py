# -*- coding: utf-8 -*-
"""改进 5B：穷举分配路径（Path B）。

Path A 的穷举升级版。Path A 使用贪心搜索（三组 MCS 条件各找一个结果，
投票选最优），Path B 遍历所有合法的反应物→产物分配方案，对每个方案
执行与 Path A 相同的 MCS 搜索和结构特征提取，然后：

  1. 碎片过滤：每个分配单元的碎片数 ≤ 4（超出后续合并能力）
  2. Rank-based 排名：使用与 Path A 相同的加权排名公式对候选排序
  3. 返回评分最优的前 N 个候选分配（默认 10），
     调用方依次对每个候选调用 balance_allocation，
     选择首个原子守恒的结果

当 mcs_search_func 不可用时，退化为基于原子不平衡量的简单评分。
"""

import time
import itertools
import logging
from typing import List, Dict, Optional, Any

from rdkit import Chem

logger = logging.getLogger("synrbl")


def exhaustive_allocation_path(
    reactants: List[str],
    products: List[str],
    mcs_balance_func=None,
    mcs_search_func=None,
    n_jobs: int = -1,
    budget_safety_factor: float = 1.5,
    min_budget: float = 90.0,
    max_budget: float = 600.0,
    enable_exhaustive_allocation: bool = True,
    top_n: int = 10,
    max_combinations: int = 1_000_000,
) -> Dict:
    """穷举分配路径（路径 B）的完整实现。

    Parameters
    ----------
    reactants : list[str]
        反应物 SMILES 列表（共 k 个）
    products : list[str]
        产物 SMILES 列表（共 n 个）
    mcs_balance_func : callable, optional
        配平函数，签名: (reactants, products, allocation, swapped) -> dict
    mcs_search_func : callable, optional
        MCS 搜索函数（通常为 ensemble_mcs）。
        签名: (data, conditions, id_col, issue_col, n_jobs) -> List[List[Dict]]
    n_jobs : int
        并行线程数（传给 ensemble_mcs 和 find_graph_dict）。
    budget_safety_factor : float
        预算安全系数。
    min_budget, max_budget : float
        预算上下限（秒）。
    enable_exhaustive_allocation : bool
        消融开关。
    top_n : int
        返回评分最优的前 N 个候选分配方案（默认 10）。
        调用方可依次尝试每个候选，选择首个原子守恒的结果。
    max_combinations : int
        组合数硬上限（默认 1_000_000）。当 (n+1)^k 超过此值时，
        直接返回 success=False 并跳过穷举，防止内存耗尽。

    Returns
    -------
    dict
        包含配平结果的字典。当 success=True 时额外包含
        ``candidates`` 字段（按评分降序排列的 top N 候选列表）。
    """
    if not enable_exhaustive_allocation:
        return {
            "success": False,
            "method": "exhaustive_allocation",
            "error": "[消融] 穷举分配路径已禁用",
            "ablation_disabled": True,
        }

    start_time = time.time()

    n = len(products)
    k = len(reactants)

    if n == 0 or k == 0:
        return {
            "success": False,
            "method": "exhaustive_allocation",
            "error": "反应物或产物为空",
        }

    # 方向判断：当反应物少于产物时交换（如分解反应 A→B+C+D），
    # 让每个产物选择反应物，确保能探索多产物对应同一反应物的组合。
    swapped = False
    if k < n:
        reactants, products = products, reactants
        n, k = k, n
        swapped = True

    # 判断评分模式
    use_mcs_scoring = mcs_search_func is not None

    # 动态时间预算
    total_combinations = (n + 1) ** k

    # 组合数硬上限：超过此阈值直接跳过，防止内存耗尽（MemoryError）
    if total_combinations > max_combinations:
        logger.warning(
            "Exhaustive allocation skipped: %d combinations "
            "(n=%d, k=%d) exceeds limit %d (combinatorial explosion)",
            total_combinations, n, k, max_combinations,
        )
        return {
            "success": False,
            "method": "exhaustive_allocation",
            "error": (
                f"组合爆炸：{total_combinations} 种组合超过上限 "
                f"{max_combinations}，跳过穷举分配"
            ),
            "reason": "combinatorial_explosion",
            "total_combinations": total_combinations,
            "max_combinations": max_combinations,
            "time_elapsed": time.time() - start_time,
        }

    raw_budget = total_combinations * 0.001 * budget_safety_factor
    budget = max(min_budget, min(raw_budget, max_budget))

    logger.info(
        "Exhaustive allocation: n=%d, k=%d, budget=%.1fs, "
        "combinations=%d, mcs_scoring=%s",
        n, k, budget, total_combinations, use_mcs_scoring,
    )

    # ---- 预计算原子数 ----
    product_mols = [Chem.MolFromSmiles(p) for p in products]
    reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
    product_atom_counts = [
        mol.GetNumAtoms() if mol else 0 for mol in product_mols
    ]
    reactant_atom_counts = [
        mol.GetNumAtoms() if mol else 0 for mol in reactant_mols
    ]

    # ================================================================
    # 阶段一：(n+1)^k 笛卡尔积枚举 + 分配单元生成
    # ================================================================
    # 每个"反应物"（交换空间中）有 (n+1) 个选择：产物 0..n-1 或空集(n)
    # 空集表示该反应物不分配给任何产物（可能是催化剂/溶剂）
    allocations: List[Dict[str, Any]] = []
    all_eval_units: Dict[str, Dict] = {}  # SMILES -> {product_idx, ...}
    empty_skipped = 0

    choices = list(range(n + 1))  # 0=空集, 1..n=产物索引+1

    for assignment in itertools.product(choices, repeat=k):
        elapsed = time.time() - start_time
        if elapsed >= budget:
            break

        # 构建产物分组：产物 j -> 分配给它的反应物列表
        pg = {j: [] for j in range(n)}
        for i, choice in enumerate(assignment):
            if choice > 0:  # 非空集
                pg[choice - 1].append(i)

        details = []
        total_imbalance = 0

        for j in range(n):
            r_indices = pg[j]
            if not r_indices:
                continue  # 该产物无反应物分配，跳过

            r_side = ".".join(reactants[i] for i in r_indices)
            p_side = products[j]
            eu_smiles = "{}>>{}".format(r_side, p_side)

            atom_imb = abs(
                product_atom_counts[j]
                - sum(reactant_atom_counts[i] for i in r_indices)
            )
            total_imbalance += atom_imb

            details.append({
                "product_idx": j,
                "reactant_indices": list(r_indices),
                "eval_unit_smiles": eu_smiles,
                "atom_imbalance": atom_imb,
            })

            # 注册分配单元（去重）
            if eu_smiles not in all_eval_units:
                all_eval_units[eu_smiles] = {
                    "smiles": eu_smiles,
                    "product_idx": j,
                    "reactant_indices": list(r_indices),
                }

        # 过滤全空分配：所有反应物都选空集时无意义
        if not details:
            empty_skipped += 1
            continue

        allocations.append({
            "details": details,
            "product_groups": {
                j: sorted(pg[j]) for j in range(n)
            },
            "total_imbalance": total_imbalance,
            "swapped": swapped,
        })

    if not allocations:
        return {
            "success": False,
            "error": "未找到有效分配方案（预算可能不足）",
            "method": "exhaustive_allocation",
            "time_elapsed": time.time() - start_time,
        }

    logger.info(
        "Enumerated %d allocations, %d unique eval units, "
        "%d empty skipped",
        len(allocations), len(all_eval_units), empty_skipped,
    )

    # ================================================================
    # 阶段二：MCS 搜索 + 特征提取 + 碎片过滤 + 排名
    # ================================================================
    if use_mcs_scoring:
        score_result = _score_with_mcs(
            allocations, all_eval_units,
            mcs_search_func, n_jobs,
            start_time, budget,
            top_n=top_n,
        )
        if score_result is None:
            # _score_with_mcs 内部失败（MCS 搜索异常或无候选通过碎片过滤），
            # 回退到原子不平衡量排序
            logger.warning(
                "MCS scoring failed, falling back to atom-imbalance ranking"
            )
            use_mcs_scoring = False
        else:
            candidates, mcs_cache = score_result

    if not use_mcs_scoring:
        # 退化模式：用原子不平衡量排序，取 top N
        sorted_allocs = sorted(
            allocations, key=lambda a: a["total_imbalance"]
        )
        candidates = sorted_allocs[:top_n]
        mcs_cache = {}
        for i, c in enumerate(candidates):
            c["rank_score"] = float(c["total_imbalance"])

    elapsed = time.time() - start_time

    if not candidates:
        return {
            "success": False,
            "error": "所有分配方案均未通过筛选",
            "method": "exhaustive_allocation",
            "time_elapsed": elapsed,
            "n_allocations": len(allocations),
            "mcs_cache": mcs_cache,
        }

    best_allocation = candidates[0]

    logger.info(
        "Best allocation: imbalance=%d, score=%.2f, "
        "%d candidates returned, time=%.1fs",
        best_allocation["total_imbalance"],
        best_allocation.get("rank_score", 0.0),
        len(candidates), elapsed,
    )

    # ---- swapped 时变换 product_groups 为原始方向 ----
    # 枚举在交换空间中进行：product_groups[j] 映射
    # "交换后的反应物 j"（即原始产物 j）到"交换后的产物索引"
    # （即原始反应物索引）。balance_allocation 期望的格式是
    # "产物索引 -> 反应物索引列表"，因此需要反转映射。
    for candidate in candidates:
        if swapped:
            original_pg = {}
            for j in range(k):  # k = 原始产物数 = 交换后的反应物数
                for orig_prod_idx in candidate["product_groups"].get(
                    j, []
                ):
                    original_pg.setdefault(orig_prod_idx, []).append(j)
            candidate["product_groups"] = {
                j: sorted(original_pg.get(j, []))
                for j in range(n)  # n = 原始反应物数
            }

    # ================================================================
    # 阶段三：调用 balance_allocation
    # ================================================================
    if mcs_balance_func is not None:
        # product_groups 已转回原始空间，reactants/products 也需要
        # 转回，否则 balance_allocation 会用原始空间索引访问交换
        # 空间的列表，导致分解反应(k<n)时产物丢失。
        orig_reactants = products if swapped else reactants
        orig_products = reactants if swapped else products
        try:
            result = mcs_balance_func(
                reactants=orig_reactants,
                products=orig_products,
                allocation=best_allocation,
                swapped=swapped,
            )

            if isinstance(result, dict):
                if result.get("success"):
                    return {
                        "success": True,
                        "balanced_reaction": result.get(
                            "balanced_reaction"
                        ),
                        "confidence": result.get("confidence"),
                        "allocation_strategy": best_allocation,
                        "candidates": candidates,
                        "mcs_cache": mcs_cache,
                        "method": "exhaustive_allocation",
                        "time_elapsed": elapsed,
                        "sub_reaction_details": result.get(
                            "sub_reaction_details"
                        ),
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get(
                            "error", "配平函数返回失败状态"
                        ),
                        "method": "exhaustive_allocation",
                        "time_elapsed": elapsed,
                        "allocation_strategy": best_allocation,
                        "candidates": candidates,
                        "mcs_cache": mcs_cache,
                        "sub_reaction_details": result.get(
                            "sub_reaction_details"
                        ),
                    }
            else:
                # 向后兼容：纯字符串返回值
                return {
                    "success": True,
                    "balanced_reaction": result,
                    "confidence": None,
                    "allocation_strategy": best_allocation,
                    "candidates": candidates,
                    "mcs_cache": mcs_cache,
                    "method": "exhaustive_allocation",
                    "time_elapsed": elapsed,
                }
        except Exception as e:
            return {
                "success": False,
                "error": "MCS 配平函数执行失败: {}".format(str(e)),
                "method": "exhaustive_allocation",
                "time_elapsed": elapsed,
                "allocation_strategy": best_allocation,
                "candidates": candidates,
                "mcs_cache": mcs_cache,
            }
    else:
        return {
            "success": False,
            "error": "找到最优分配方案但 MCS 配平函数未提供",
            "method": "exhaustive_allocation",
            "time_elapsed": elapsed,
            "allocation_strategy": best_allocation,
            "candidates": candidates,
            "mcs_cache": mcs_cache,
        }


def _score_with_mcs(
    allocations,
    all_eval_units,
    mcs_search_func,
    n_jobs,
    start_time,
    budget,
    top_n=10,
):
    """使用 MCS 搜索 + 结构特征对分配方案进行评分和排名。

    流程：
      1. 将所有分配单元批量送入 ensemble_mcs（单 MCES 条件）
      2. 运行 find_graph_dict 提取碎片信息
      3. 碎片过滤（≤ 4）
      4. 聚合分配级别的特征
      5. 调用 _compute_rank_scores 排名
      6. 构建 MCS 缓存（供下游复用）

    Returns: (top_candidates, mcs_cache) tuple, or None on failure
    """
    from synrbl.SynMCSImputer.SubStructure.extract_common_mcs import (
        ExtractMCS,
    )
    from synrbl.SynMCSImputer.MissingGraph.find_graph_dict import (
        find_graph_dict,
    )
    from synrbl.mcs_search import _compute_rank_scores

    # ---- 单 MCES 条件 ----
    mces_condition = {"method": "MCES", "sort": "MCES"}

    # ---- 构建分配单元数据（供 ensemble_mcs 使用）----
    eu_list = list(all_eval_units.values())
    eu_index_map = {eu["smiles"]: idx for idx, eu in enumerate(eu_list)}
    num_eu = len(eu_list)

    data_for_mcs = []
    for idx, eu in enumerate(eu_list):
        data_for_mcs.append({
            "id": "pathb_eu_{}".format(idx),
            "reactants": eu["smiles"].split(">>")[0],
            "products": eu["smiles"].split(">>")[1],
            "carbon_balance_check": "products",
        })

    # ---- 批量 MCS 搜索（单 MCES 条件）----
    try:
        condition_results = mcs_search_func(
            data_for_mcs, [mces_condition],
            id_col="id", issue_col="issue", n_jobs=n_jobs,
        )
    except Exception as e:
        logger.warning("ensemble_mcs failed for Path B: %s", e)
        return None

    # condition_results[0] = 单条件下所有分配单元的结果
    mcs_results = condition_results[0]

    # ---- 运行 find_graph_dict ----
    mcs_dicts = []
    for eu_idx in range(num_eu):
        entry = mcs_results[eu_idx]
        mcs_dicts.append({
            "id": "pathb_eu_{}".format(eu_idx),
            "mcs_results": entry.get("mcs_results", []),
            "sorted_reactants": entry.get("sorted_reactants", []),
        })

    try:
        fg_results = find_graph_dict(mcs_dicts, n_jobs=n_jobs)
    except Exception as e:
        logger.warning(
            "find_graph_dict failed for Path B, "
            "falling back to zero features: %s", e,
        )
        fg_results = [{}] * num_eu

    # ---- 提取每个分配单元的特征 ----
    eu_features = []  # [eu_idx] = feature dict or None

    for eu_idx in range(num_eu):
        entry = mcs_results[eu_idx]
        fg = fg_results[eu_idx]

        # 原子数：从 SMARTS 计算
        smarts_list = entry.get("mcs_results", [])
        atom_count = sum(
            ExtractMCS.get_num_atoms(s) for s in smarts_list
        )

        # 碎片数：统计 smiles 中的碎片（含点分隔的多部件）
        smiles_list = fg.get("smiles", [])
        fragment_count = 0
        for s in smiles_list:
            if s is not None:
                fragment_count += len(s.split("."))

        # 边界原子数：展平所有反应物的边界原子列表后计数
        ba_list = fg.get("boundary_atoms_products", [])
        boundary_count = sum(
            len(inner) for inner in ba_list
            if isinstance(inner, list)
        ) if isinstance(ba_list, list) else 0

        eu_features.append({
            "atom_count": atom_count,
            "fragment_count": fragment_count,
            "boundary_count": boundary_count,
        })

    # ---- 评估每个分配方案 ----
    valid_candidates = []  # [(alloc_idx, total_atoms, total_frags, total_bnd)]

    for alloc_idx, alloc in enumerate(allocations):
        max_frag = 0
        total_atoms = 0
        total_frags = 0
        total_bnd = 0
        valid = True

        for detail in alloc["details"]:
            eu_smiles = detail["eval_unit_smiles"]
            eu_idx = eu_index_map.get(eu_smiles)
            if eu_idx is None or eu_features[eu_idx] is None:
                valid = False
                break

            feat = eu_features[eu_idx]
            max_frag = max(max_frag, feat["fragment_count"])
            total_atoms += feat["atom_count"]
            total_frags += feat["fragment_count"]
            total_bnd += feat["boundary_count"]

        # 碎片过滤：每个分配单元 ≤ 4 个碎片
        if not valid or max_frag > 4:
            continue

        valid_candidates.append(
            (alloc_idx, total_atoms, total_frags, total_bnd)
        )

    if not valid_candidates:
        logger.info(
            "No allocations passed fragment filter (<=4)"
        )
        return None

    # ---- Rank-based 排名（与 Path A 相同公式）----
    if len(valid_candidates) == 1:
        scores = [0.0]
    else:
        atom_list = [c[1] for c in valid_candidates]
        frag_list = [c[2] for c in valid_candidates]
        bnd_list = [c[3] for c in valid_candidates]

        scores, _, _, _ = _compute_rank_scores(
            atom_list, frag_list, bnd_list,
        )

    # 将评分附加到每个候选并排序（分数越低越好）
    scored = []
    for i, (alloc_idx, _, _, _) in enumerate(valid_candidates):
        alloc = allocations[alloc_idx]
        alloc["rank_score"] = float(scores[i])
        scored.append(alloc)

    scored.sort(key=lambda a: a["rank_score"])
    top_candidates = scored[:top_n]

    # ---- 构建 MCS 缓存（供下游 build_compounds 复用，避免重复搜索）----
    # EA-1 修复：eu_list 元素是字典（含 "smiles" 等键），
    # 需用其中的 SMILES 字符串作为缓存键，而非整个字典
    mcs_cache = {}  # eval_unit_smiles → mcs_data dict
    for eu_idx, eu_entry in enumerate(eu_list):
        eu_smiles = eu_entry["smiles"]
        entry = mcs_results[eu_idx]
        fg = fg_results[eu_idx] if eu_idx < len(fg_results) else {}
        cached = dict(fg)
        for k, v in entry.items():
            cached[k] = v
        mcs_cache[eu_smiles] = cached

    logger.info(
        "Ranked %d valid allocations, returning top %d "
        "(best score=%.2f, worst score=%.2f)",
        len(valid_candidates),
        len(top_candidates),
        top_candidates[0]["rank_score"] if top_candidates else 0.0,
        top_candidates[-1]["rank_score"] if top_candidates else 0.0,
    )

    return top_candidates, mcs_cache
