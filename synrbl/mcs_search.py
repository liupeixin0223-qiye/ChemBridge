import logging

from typing import List, Dict, Optional
from collections import Counter

from synrbl.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from synrbl.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from synrbl.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict

logger = logging.getLogger(__name__)


def _extract_vote_features(
    condition_results: List[List[Dict]],
    condition_names: Optional[List[str]] = None,
    find_graph_results: Optional[List[List[Dict]]] = None,
    count_dot_components: bool = True,
) -> List[List[Dict]]:
    """将 ensemble_mcs 的输出转换为 progressive_mcs_vote 所需的格式。

    对每个反应 × 每个条件，从 SMARTS 和 find_graph_dict 结果中提取
    投票所需的特征字段。

    参数:
        condition_results: ensemble_mcs 的返回值，
            condition_results[c][r] = 条件 c 对反应 r 的结果字典
        condition_names: 条件名称列表（可选）
        find_graph_results: 每个条件的 find_graph_dict 输出，
            find_graph_results[c] = 条件 c 对所有反应的碎片提取结果列表。
            用于获取精确的 fragment_count 和 boundary_atom_count。
        count_dot_components: 是否将点号分隔的多组分 SMILES 按独立分子
            计数（True 时 "A.B" 计为 2，False 时计为 1）。默认 True。
            受 enable_advanced_scoring 消融开关控制：关闭时退化为
            原始计数方式以保持消融实验可比性。

    返回:
        按反应索引重组织的列表:
            result[r] = [条件0的特征字典, 条件1的..., 条件2的...]
    """
    if not condition_results:
        return []

    num_conditions = len(condition_results)
    num_reactions = min(len(c) for c in condition_results)

    if condition_names is None:
        condition_names = [
            "mcis_strict_ring", "mcis_relaxed", "mces"
        ][:num_conditions]

    per_reaction: List[List[Dict]] = []
    for r_idx in range(num_reactions):
        candidates = []
        for c_idx in range(num_conditions):
            entry = condition_results[c_idx][r_idx]
            smarts_list = entry.get("mcs_results", [])
            # 将 SMARTS 列表转为不可变集合（用于多数投票比较）
            smarts_key = frozenset(smarts_list) if smarts_list else frozenset()
            # 计算 MCS 总原子数（使用 RDKit 解析 SMARTS）
            atom_count = sum(
                ExtractMCS.get_num_atoms(s) for s in smarts_list
            )

            # 精确的 fragment_count 和 boundary_atom_count
            # 来自 find_graph_dict 的输出
            fragment_count = 0
            boundary_atom_count = 0
            if (find_graph_results is not None
                    and c_idx < len(find_graph_results)
                    and r_idx < len(find_graph_results[c_idx])):
                fg_result = find_graph_results[c_idx][r_idx]
                # smiles 字段 = 切割后的残余碎片 SMILES 列表
                smiles_list = fg_result.get("smiles", [])
                if count_dot_components:
                    # 精确计数：点号分隔的多组分按独立分子计
                    # （"A.B" 计为 2），与 Path B 一致
                    fragment_count = sum(
                        len(s.split(".")) for s in smiles_list
                        if s is not None
                    )
                else:
                    # 原始计数：每个非 None 条目计为 1
                    fragment_count = sum(
                        1 for s in smiles_list if s is not None
                    )
                # boundary_atoms_products = 每个反应物的边界原子列表
                ba = fg_result.get("boundary_atoms_products", [])
                if isinstance(ba, list):
                    boundary_atom_count = sum(
                        len(inner) for inner in ba
                        if isinstance(inner, list)
                    )

            candidates.append({
                "condition_name": condition_names[c_idx]
                if c_idx < len(condition_names)
                else f"condition_{c_idx}",
                "smarts_key": smarts_key,
                "atom_count": atom_count,
                "fragment_count": fragment_count,
                "boundary_atom_count": boundary_atom_count,
                "original_entry": entry,  # 保留原始数据供后续使用
                "condition_index": c_idx,
            })
        per_reaction.append(candidates)
    return per_reaction


def _rank_avg(values, reverse=False):
    """手动计算平均排名（average method），处理并列值。"""
    indexed = sorted(
        enumerate(values), key=lambda x: x[1], reverse=reverse
    )
    n_v = len(values)
    ranks = [0.0] * n_v
    i = 0
    while i < n_v:
        j = i
        while j < n_v - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _compute_rank_scores(
    atom_counts,
    fragment_counts,
    boundary_counts,
    weight_atom_count=3.0,
    weight_fragment_count=1.5,
    weight_boundary_atoms=1.0,
):
    """Compute rank-based weighted scores for a list of candidates.

    Reusable across Path A (progressive_mcs_vote) and Path B
    (exhaustive_allocation).  Lower score = better candidate.

    Parameters
    ----------
    atom_counts : list[int]
        MCS atom counts (higher is better → ranked descending).
    fragment_counts : list[int]
        Fragment counts (lower is better → ranked ascending).
    boundary_counts : list[int]
        Boundary atom counts (lower is better → ranked ascending).
    weight_atom_count, weight_fragment_count, weight_boundary_atoms : float
        Weights for each dimension.

    Returns
    -------
    tuple[list[float], list[float], list[float], list[float]]
        (scores, rank_atom, rank_frag, rank_bnd)
    """
    try:
        from scipy.stats import rankdata as _scipy_rankdata

        rank_atom = _scipy_rankdata([-x for x in atom_counts])
        rank_frag = _scipy_rankdata(fragment_counts)
        rank_bnd = _scipy_rankdata(boundary_counts)
    except ImportError:
        rank_atom = _rank_avg(atom_counts, reverse=True)
        rank_frag = _rank_avg(fragment_counts, reverse=False)
        rank_bnd = _rank_avg(boundary_counts, reverse=False)

    n = len(atom_counts)
    scores = [
        weight_atom_count * rank_atom[i]
        + weight_fragment_count * rank_frag[i]
        + weight_boundary_atoms * rank_bnd[i]
        for i in range(n)
    ]
    return scores, rank_atom, rank_frag, rank_bnd


def progressive_mcs_vote(
    mcs_results: List[Dict],
    weight_atom_count: float = 3.0,
    weight_fragment_count: float = 1.5,
    weight_boundary_atoms: float = 1.0,
    enable_progressive_voting: bool = True,
) -> Dict:
    """递进式两层投票机制，用于从多组 MCS 搜索结果中选择最优方案。

    参数:
        mcs_results: MCS 搜索结果列表，每个元素为一个字典，包含:
            - condition_name (str): 搜索条件名称
            - smarts_key (frozenset): SMARTS 集合的唯一标识（用于多数投票）
            - atom_count (int): MCS 的原子数目
            - fragment_count (int): 切割后的残余碎片数目
            - boundary_atom_count (int): MCS 边界上的原子数目
            - original_entry (dict): 原始 ensemble_mcs 输出条目
            - condition_index (int): 条件索引
        weight_atom_count: MCS 原子数的评分权重（默认 3.0）
        weight_fragment_count: 碎片数的评分权重（默认 1.5）
        weight_boundary_atoms: 边界原子数的评分权重（默认 1.0）
        enable_progressive_voting: 是否启用递进式投票（False 时退化为
            简单最大原子数选取）

    返回:
        被选中的 MCS 结果字典，附加 'vote_method' 字段说明选择方式
    """
    if not mcs_results:
        return {"error": "MCS 搜索结果列表为空", "vote_method": "none"}

    if len(mcs_results) == 1:
        result = mcs_results[0].copy()
        result["vote_method"] = "single_result"
        return result

    # ========== 消融开关：退化为简单最大原子数选取 ==========
    if not enable_progressive_voting:
        best = max(mcs_results, key=lambda m: m["atom_count"])
        result = best.copy()
        result["vote_method"] = "largest_atom_count"
        return result

    # ========== 第一层：多数投票 ==========
    smarts_votes = Counter()
    smarts_to_result = {}

    for mcs in mcs_results:
        key = mcs.get("smarts_key", frozenset())
        if not key:
            continue
        smarts_votes[key] += 1
        if key not in smarts_to_result:
            smarts_to_result[key] = mcs

    majority_threshold = 2
    majority_candidates = [
        (key, count)
        for key, count in smarts_votes.items()
        if count >= majority_threshold
    ]

    if majority_candidates:
        best_key, best_count = max(
            majority_candidates, key=lambda x: x[1]
        )
        result = smarts_to_result[best_key].copy()
        result["vote_method"] = "majority_vote"
        result["vote_count"] = best_count
        result["total_voters"] = len(mcs_results)
        return result

    # ========== 第二层：Rank-based 加权排名 ==========
    atom_counts = [mcs["atom_count"] for mcs in mcs_results]
    fragment_counts = [mcs["fragment_count"] for mcs in mcs_results]
    boundary_counts = [mcs["boundary_atom_count"] for mcs in mcs_results]

    # 过滤全零候选：三项指标均为0说明MCS为空，不应参与排名
    non_zero_indices = [
        i for i in range(len(mcs_results))
        if not (atom_counts[i] == 0
                and fragment_counts[i] == 0
                and boundary_counts[i] == 0)
    ]
    if not non_zero_indices:
        return {
            "error": "所有MCS候选均为全零",
            "vote_method": "none",
        }
    if len(non_zero_indices) < len(mcs_results):
        mcs_results = [mcs_results[i] for i in non_zero_indices]
        atom_counts = [atom_counts[i] for i in non_zero_indices]
        fragment_counts = [fragment_counts[i] for i in non_zero_indices]
        boundary_counts = [boundary_counts[i] for i in non_zero_indices]

    n_candidates = len(mcs_results)
    scores, rank_atom, rank_frag, rank_bnd = _compute_rank_scores(
        atom_counts, fragment_counts, boundary_counts,
        weight_atom_count, weight_fragment_count, weight_boundary_atoms,
    )

    best_idx = scores.index(min(scores))
    result = mcs_results[best_idx].copy()
    result["vote_method"] = "weighted_ranking"
    result["total_score"] = float(scores[best_idx])
    result["score_breakdown"] = {
        "atom_rank": float(rank_atom[best_idx]),
        "fragment_rank": float(rank_frag[best_idx]),
        "boundary_rank": float(rank_bnd[best_idx]),
        "weighted_sum": float(scores[best_idx]),
    }
    result["ranking_details"] = [
        {
            "condition": mcs_results[i]["condition_name"],
            "atom_rank": float(rank_atom[i]),
            "fragment_rank": float(rank_frag[i]),
            "boundary_rank": float(rank_bnd[i]),
            "weighted_sum": float(scores[i]),
        }
        for i in range(n_candidates)
    ]

    return result


class MCSSearch:
    def __init__(
        self,
        id_col,
        solved_col="solved",
        mcs_data_col="mcs",
        issue_col="issue",
        n_jobs=-1,
        enable_progressive_voting=True,
        count_dot_components=True,
    ):
        self.id_col = id_col
        self.solved_col = solved_col
        self.mcs_data_col = mcs_data_col
        self.issue_col = issue_col
        self.n_jobs = n_jobs
        self.enable_progressive_voting = enable_progressive_voting
        self.count_dot_components = count_dot_components

        self.conditions = [
            {
                "RingMatchesRingOnly": True,
                "CompleteRingsOnly": True,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            # {
            #     "RingMatchesRingOnly": True,
            #     "CompleteRingsOnly": True,
            #     "method": "MCIS",
            #     "sort": "MCIS",
            #     "ignore_bond_order": False,
            # },
            {
                "RingMatchesRingOnly": False,
                "CompleteRingsOnly": False,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            # {
            #     "RingMatchesRingOnly": False,
            #     "CompleteRingsOnly": False,
            #     "method": "MCIS",
            #     "sort": "MCIS",
            #     "ignore_bond_order": False,
            # },
            {"method": "MCES", "sort": "MCES"},
        ]

    def find(self, reactions):
        id2idx_map = {}
        mcs_reactions = []
        for idx, reaction in enumerate(reactions):
            if reaction[self.solved_col]:
                continue
            id2idx_map[reaction[self.id_col]] = idx
            reaction[self.mcs_data_col] = None
            reaction[self.issue_col] = "No MCS identified."
            mcs_reactions.append(reaction)

        if len(mcs_reactions) == 0:
            return reactions

        logger.info(
            "Find maximum-common-substructure for {} reactions.".format(
                len(mcs_reactions)
            )
        )

        condition_results = ensemble_mcs(
            mcs_reactions,
            self.conditions,
            id_col=self.id_col,
            issue_col=self.issue_col,
            n_jobs=self.n_jobs,
        )

        # ---- 对每个条件分别运行碎片提取（find_graph_dict）----
        # 获取精确的 fragment_count 和 boundary_atom_count 用于投票
        num_conditions = len(self.conditions)
        condition_names = ["mcis_strict_ring", "mcis_relaxed", "mces"]
        find_graph_results = []  # find_graph_results[c] = 条件c的碎片提取结果

        for c_idx in range(num_conditions):
            logger.info(
                "Running fragment extraction for condition {}/{}: {}".format(
                    c_idx + 1, num_conditions,
                    condition_names[c_idx]
                    if c_idx < len(condition_names)
                    else f"condition_{c_idx}",
                )
            )
            mcs_dicts_for_condition = []
            for r_idx in range(len(mcs_reactions)):
                entry = condition_results[c_idx][r_idx]
                mcs_dicts_for_condition.append({
                    self.id_col: mcs_reactions[r_idx][self.id_col],
                    "mcs_results": entry.get("mcs_results", []),
                    "sorted_reactants": entry.get("sorted_reactants", []),
                })
            fg_results = find_graph_dict(
                mcs_dicts_for_condition, n_jobs=self.n_jobs
            )
            find_graph_results.append(fg_results)

        # ---- 递进式投票选择最优 MCS 条件 ----
        per_reaction_features = _extract_vote_features(
            condition_results, condition_names,
            find_graph_results=find_graph_results,
            count_dot_components=self.count_dot_components,
        )

        for r_idx, candidates in enumerate(per_reaction_features):
            vote_result = progressive_mcs_vote(
                candidates,
                enable_progressive_voting=self.enable_progressive_voting,
            )
            winning_c_idx = vote_result.get("condition_index")
            winning_entry = vote_result.get("original_entry")

            if (winning_entry is not None
                    and winning_c_idx is not None
                    and winning_c_idx < len(find_graph_results)
                    and r_idx < len(find_graph_results[winning_c_idx])):
                # 使用获胜条件的碎片提取结果
                fg_result = find_graph_results[winning_c_idx][r_idx]
                mcs_result = dict(fg_result)
                for k, v in winning_entry.items():
                    mcs_result[k] = v
                mcs_result["vote_method"] = vote_result.get(
                    "vote_method", "unknown"
                )
                _id = mcs_reactions[r_idx][self.id_col]
                _idx = id2idx_map[_id]
                reactions[_idx][self.mcs_data_col] = mcs_result
                reactions[_idx][self.issue_col] = mcs_result.get(
                    self.issue_col, ""
                )
            else:
                # 极端边界：所有候选均为空，构造默认条目
                fallback_id = (
                    mcs_reactions[r_idx][self.id_col]
                    if r_idx < len(mcs_reactions)
                    else f"unknown_{r_idx}"
                )
                _idx = id2idx_map.get(fallback_id, r_idx)
                reactions[_idx][self.mcs_data_col] = {
                    self.id_col: fallback_id,
                    "mcs_results": [],
                    "sorted_reactants": [],
                    self.issue_col: "No valid MCS candidates.",
                    "vote_method": vote_result.get("vote_method", "none"),
                }
                reactions[_idx][self.issue_col] = "No valid MCS candidates."

        return reactions
