"""改进 6c：多碎片合并器。

实现两阶段整合合并流程：
  阶段一 — 穷举直接配对（扣减式自由基匹配）
  阶段二 — ExpandRule 扩展（插入中间原子）

所有通过规则检查的候选均返回，由下游管线（全元素校验 + XGBoost）做最终选择。

支持 2、3、4 个碎片的合并场景。
"""

import json
import logging
import importlib.resources
from typing import List, Dict, Optional
from copy import deepcopy

from rdkit import Chem

import synrbl.SynMCSImputer
from synrbl.SynMCSImputer.rules import ThreeLayerBondRuleEngine

logger = logging.getLogger("synrbl")


class MultiFragmentMerger:
    """多碎片合并器：实现三阶段整合合并流程。

    支持 2、3、4 个碎片的合并场景。
    """

    def __init__(
        self,
        bond_rule_engine: Optional[ThreeLayerBondRuleEngine] = None,
        expand_rules: Optional[List[Dict]] = None,
        enable_multi_fragment: bool = True,
    ):
        self.bond_engine = bond_rule_engine or ThreeLayerBondRuleEngine()
        self.expand_rules = expand_rules or self._default_expand_rules()
        self.enable_multi_fragment = enable_multi_fragment

    @staticmethod
    def _default_expand_rules() -> List[Dict]:
        """从 expand_rules.json 加载 type=bridge 的桥接原子规则。

        如果 JSON 加载失败，回退到硬编码的 O/S/N/C 四种桥接原子。
        """
        try:
            json_data = (
                importlib.resources.files(synrbl.SynMCSImputer)
                .joinpath("expand_rules.json")
                .read_text(encoding="utf-8")
            )
            all_rules = json.loads(json_data)
            bridge_rules = [
                r for r in all_rules if r.get("type") == "bridge"
            ]
            if bridge_rules:
                return bridge_rules
        except Exception as exc:
            logger.debug(
                "Failed to load bridge rules from expand_rules.json, "
                "using hardcoded defaults: %s", exc,
            )
        # 硬编码回退
        return [
            {"intermediate_atom": "O", "intermediate_smiles": "[O]",
             "valence": 2, "priority": 1, "multi_site": True,
             "description": "插入氧原子（醚键/酯键形成）"},
            {"intermediate_atom": "S", "intermediate_smiles": "[S]",
             "valence": 2, "priority": 2, "multi_site": True,
             "description": "插入硫原子（硫醚键形成）"},
            {"intermediate_atom": "N", "intermediate_smiles": "[NH]",
             "valence": 3, "priority": 3, "multi_site": True,
             "description": "插入氮原子（胺键形成）"},
            {"intermediate_atom": "C", "intermediate_smiles": "[CH2]",
             "valence": 2, "priority": 4, "multi_site": True,
             "description": "插入亚甲基（碳链延长）"},
        ]

    def merge_fragments(self, fragments: List) -> List[Dict]:
        """三阶段整合合并流程的入口函数。"""
        n_frags = len(fragments)
        if n_frags < 2:
            return []
        if n_frags > 4:
            return [{"error": f"碎片数量 {n_frags} 超出支持范围（最多 4 个）"}]

        if not self.enable_multi_fragment and n_frags > 2:
            return [{"error": f"[消融] 多碎片合并已禁用，当前 {n_frags} 个碎片",
                      "merge_failed_reason": "ablation_multi_fragment_disabled"}]

        candidates: List[Dict] = []

        # ========== 阶段一：穷举直接配对 ==========
        direct_strategies = self._enumerate_direct_pairings(fragments)
        merge_cache: Dict[int, Optional[Dict]] = {}
        for i, strategy in enumerate(direct_strategies):
            merged = self._execute_direct_merge(fragments, strategy)
            merge_cache[i] = merged
            if merged is not None:
                candidates.append({
                    "merged_mol": merged["mol"],
                    "merged_smiles": merged["smiles"],
                    "used_expand_rule": False,
                    "residual_capped_h": merged["residual_h"],
                    "pairing_strategy": strategy,
                    "stage_origin": "direct",
                })

        # ========== 阶段二：ExpandRule 扩展 ==========
        for i, s in enumerate(direct_strategies):
            cached = merge_cache.get(i)
            if cached is not None and cached["residual_h"] > 0:
                expanded_results = self._apply_expand_rules(
                    fragments, s, cached)
                for expanded in expanded_results:
                    candidates.append({
                        "merged_mol": expanded["mol"],
                        "merged_smiles": expanded["smiles"],
                        "used_expand_rule": True,
                        "residual_capped_h": expanded["residual_h"],
                        "pairing_strategy": s,
                        "stage_origin": "expand",
                        "expand_rule_used": expanded.get("rule_description", ""),
                    })

        # ========== 阶段 1.5：未配对边界的补充桥接 ==========
        # 针对阶段一中未能参与任何配对的边界原子（自由基不足或键型限制
        # 导致无法直接成键），尝试通过插入中间原子（O/S/N/CH2）建立
        # 连接。当阶段一零配对时覆盖所有边界；当阶段一有部分配对时
        # 仅覆盖被遗漏的边界。纯增强：已有候选不受影响。
        paired_boundary_keys = set()
        for s in direct_strategies:
            for pair in s["pairs"]:
                paired_boundary_keys.add((
                    pair["boundary_1"]["fragment_idx"],
                    pair["boundary_1"]["boundary"].atom_idx))
                paired_boundary_keys.add((
                    pair["boundary_2"]["fragment_idx"],
                    pair["boundary_2"]["boundary"].atom_idx))

        unpaired = self._enumerate_unpaired_boundaries(
            fragments, paired_boundary_keys)

        # 守卫：当所有未配对边界都没有自由基时，阶段 1.5 无法产生
        # 桥接候选（_apply_unpaired_expand_rules 的 residual_sites 要求
        # radical_electrons > 0 AND capped_hydrogens > 0）。显式跳过，
        # 确保"零配对且无自由基"场景原样流入 Concat 回退路径，
        # 维持修改前行为（纯增强，无决策路径变化）。
        total_unpaired_radical = sum(
            u["initial_radical"] for u in unpaired)

        if unpaired and total_unpaired_radical > 0:
            # 复用 _execute_direct_merge 的空配对路径构建组合分子与
            # offset_map，供桥接原子插入使用
            empty_strategy = {"pairs": [], "total_pairs": 0,
                              "coverage": "none"}
            base_merge = self._execute_direct_merge(fragments, empty_strategy)
            if base_merge is not None:
                unpaired_results = self._apply_unpaired_expand_rules(
                    fragments, base_merge, unpaired)
                for result in unpaired_results:
                    candidates.append({
                        "merged_mol": result["mol"],
                        "merged_smiles": result["smiles"],
                        "used_expand_rule": True,
                        "residual_capped_h": result["residual_h"],
                        "pairing_strategy": None,
                        "stage_origin": "expand_unpaired",
                        "expand_rule_used": result.get(
                            "rule_description", ""),
                    })

        # ========== Concat 回退：无自由基时拼接碎片 ==========
        if not candidates:
            all_boundaries = []
            for frag in fragments:
                all_boundaries.extend(frag.boundaries)
            has_radicals = any(
                bd.radical_electrons > 0 for bd in all_boundaries
            )
            if all_boundaries and not has_radicals:
                concat_result = self._concat_fragments(fragments)
                if concat_result is not None:
                    candidates.append({
                        "merged_mol": concat_result["mol"],
                        "merged_smiles": concat_result["smiles"],
                        "used_expand_rule": False,
                        "residual_capped_h": 0,
                        "pairing_strategy": None,
                        "stage_origin": "concat",
                        "low_confidence_flag": True,
                        "low_confidence_reason": (
                            "No radical electrons on boundary atoms; "
                            "fragments concatenated without bonds."
                        ),
                    })

        # ========== 候选筛选（UFF 已移除，返回所有合法候选） ==========
        if not candidates:
            return []

        fully_resolved = [c for c in candidates if c["residual_capped_h"] == 0]
        if fully_resolved:
            return fully_resolved

        partially_resolved = [c for c in candidates if c["residual_capped_h"] > 0]
        for c in partially_resolved:
            c["low_confidence_flag"] = True
            c["low_confidence_reason"] = (
                f"ExpandRule 未能消耗全部残余封顶氢"
                f"（残余 {c['residual_capped_h']} 个）")
        return partially_resolved

    # ------------------------------------------------------------------
    # 阶段一辅助方法
    # ------------------------------------------------------------------
    def _enumerate_direct_pairings(self, fragments: List) -> List[Dict]:
        all_boundaries = []
        for frag_idx, frag in enumerate(fragments):
            for bd in frag.boundaries:
                all_boundaries.append({
                    "fragment_idx": frag_idx,
                    "boundary": bd,
                    "initial_radical": bd.radical_electrons,
                })

        possible_pairs = []
        for i in range(len(all_boundaries)):
            for j in range(i + 1, len(all_boundaries)):
                b1, b2 = all_boundaries[i], all_boundaries[j]
                if b1["fragment_idx"] == b2["fragment_idx"]:
                    continue
                decision = self.bond_engine.determine_bond_type(
                    atom1_element=b1["boundary"].element,
                    atom1_radical_electrons=b1["initial_radical"],
                    atom2_element=b2["boundary"].element,
                    atom2_radical_electrons=b2["initial_radical"])
                if decision.bond_type != "NONE":
                    possible_pairs.append({
                        "idx_1": i, "idx_2": j,
                        "boundary_1": b1, "boundary_2": b2,
                        "restriction_check": decision})

        valid_strategies: List[Dict] = []
        self._enumerate_pairing_combinations(
            possible_pairs, [], {}, valid_strategies, all_boundaries,
            bond_engine=self.bond_engine)

        # 去重：消除因选择顺序不同而产生的完全相同策略（仅在顶层执行一次）
        seen = set()
        unique_strategies = []
        for s in valid_strategies:
            key = tuple(sorted(
                (p["idx_1"], p["idx_2"], p["consumption"])
                for p in s["pairs"]
            ))
            full_key = (key, s["coverage"])
            if full_key not in seen:
                seen.add(full_key)
                unique_strategies.append(s)
        return unique_strategies

    def _enumerate_pairing_combinations(
        self, possible_pairs, current_strategy, usage_counts,
        valid_strategies, all_boundaries, bond_engine=None):
        """穷举所有合法的配对策略（含不同顺序产生的不同消耗分配）。

        与原版不同，此方法不限制配对的选择顺序（无 start_index 约束），
        而是允许每一步从所有配对中自由选择，确保不遗漏因处理顺序不同
        而产生的合法策略。去重由调用方 _enumerate_direct_pairings 统一处理。

        当 bond_engine 可用时，consumption 会受 determine_bond_type 返回的
        允许键型约束：若键型被限制层降级（如 TRIPLE→SINGLE），consumption
        取 min(bond_type_order, remaining_b1, remaining_b2)，确保降级后
        多余的自由基保留在 remaining 中继续参与后续配对。
        """
        remaining = {}
        for b in all_boundaries:
            bid = (b["fragment_idx"], b["boundary"].atom_idx)
            remaining[bid] = b["initial_radical"] - usage_counts.get(bid, 0)

        # 所有碎片的自由基已完全消耗 → 记录完整策略
        if all(v <= 0 for v in remaining.values()):
            valid_strategies.append({
                "pairs": deepcopy(current_strategy),
                "total_pairs": len(current_strategy),
                "coverage": "full"})
            return

        # 尝试所有可用配对（不限制起始索引，允许任意顺序）
        for idx in range(len(possible_pairs)):
            pair = possible_pairs[idx]
            b1_id = (pair["boundary_1"]["fragment_idx"],
                      pair["boundary_1"]["boundary"].atom_idx)
            b2_id = (pair["boundary_2"]["fragment_idx"],
                      pair["boundary_2"]["boundary"].atom_idx)
            if remaining.get(b1_id, 0) <= 0 or remaining.get(b2_id, 0) <= 0:
                continue

            # 计算键型上限：用当前剩余自由基重新评估允许的键型，
            # 防止限制层降级（如 TRIPLE→SINGLE）后消耗过多自由基
            consumption = min(remaining[b1_id], remaining[b2_id])
            if bond_engine is not None:
                decision = bond_engine.determine_bond_type(
                    atom1_element=pair["boundary_1"]["boundary"].element,
                    atom1_radical_electrons=remaining[b1_id],
                    atom2_element=pair["boundary_2"]["boundary"].element,
                    atom2_radical_electrons=remaining[b2_id],
                )
                _bond_order = {"SINGLE": 1, "DOUBLE": 2,
                               "TRIPLE": 3, "NONE": 0}
                bond_type_order = _bond_order.get(
                    decision.bond_type, consumption)
                consumption = min(consumption, bond_type_order)
                if consumption <= 0:
                    continue

            new_counts = dict(usage_counts)
            new_counts[b1_id] = new_counts.get(b1_id, 0) + consumption
            new_counts[b2_id] = new_counts.get(b2_id, 0) + consumption
            self._enumerate_pairing_combinations(
                possible_pairs,
                current_strategy + [{**pair, "consumption": consumption}],
                new_counts, valid_strategies, all_boundaries,
                bond_engine=bond_engine)

        # 当前策略非空但未达到全消耗 → 记录为部分覆盖策略
        if current_strategy:
            valid_strategies.append({
                "pairs": deepcopy(current_strategy),
                "total_pairs": len(current_strategy),
                "coverage": "partial"})

    def _execute_direct_merge(self, fragments, strategy) -> Optional[Dict]:
        working_mols = [Chem.RWMol(Chem.Mol(frag.fragment_mol)) for frag in fragments]
        atom_offset = 0
        offset_map: Dict[int, Dict[int, int]] = {}
        for frag_idx, mol in enumerate(working_mols):
            offset_map[frag_idx] = {}
            for atom in mol.GetAtoms():
                offset_map[frag_idx][atom.GetIdx()] = atom.GetIdx() + atom_offset
            atom_offset += mol.GetNumAtoms()

        combined = Chem.RWMol(working_mols[0])
        for mol in working_mols[1:]:
            combined = Chem.RWMol(Chem.CombineMols(combined.GetMol(), mol.GetMol()))
        combined = Chem.RWMol(combined)

        total_residual_h = 0
        boundary_total_consumption: Dict[tuple, int] = {}
        for pair in strategy["pairs"]:
            b1, b2 = pair["boundary_1"], pair["boundary_2"]
            c = pair["consumption"]
            b1_key = (b1["fragment_idx"], b1["boundary"].atom_idx)
            b2_key = (b2["fragment_idx"], b2["boundary"].atom_idx)
            boundary_total_consumption[b1_key] = boundary_total_consumption.get(b1_key, 0) + c
            boundary_total_consumption[b2_key] = boundary_total_consumption.get(b2_key, 0) + c

        boundary_h_deducted: Dict[tuple, bool] = {}
        for pair in strategy["pairs"]:
            b1, b2 = pair["boundary_1"], pair["boundary_2"]
            consumption = pair["consumption"]
            idx1 = offset_map[b1["fragment_idx"]].get(b1["boundary"].atom_idx, b1["boundary"].atom_idx)
            idx2 = offset_map[b2["fragment_idx"]].get(b2["boundary"].atom_idx, b2["boundary"].atom_idx)

            b1_key = (b1["fragment_idx"], b1["boundary"].atom_idx)
            b2_key = (b2["fragment_idx"], b2["boundary"].atom_idx)

            if b1_key not in boundary_h_deducted:
                h_to_remove = min(b1["boundary"].capped_hydrogens,
                                  boundary_total_consumption.get(b1_key, consumption))
                h_removed = self._remove_capped_hydrogens(combined, idx1, h_to_remove)
                total_residual_h += b1["boundary"].capped_hydrogens - h_removed
                boundary_h_deducted[b1_key] = True

            if b2_key not in boundary_h_deducted:
                h_to_remove = min(b2["boundary"].capped_hydrogens,
                                  boundary_total_consumption.get(b2_key, consumption))
                h_removed = self._remove_capped_hydrogens(combined, idx2, h_to_remove)
                total_residual_h += b2["boundary"].capped_hydrogens - h_removed
                boundary_h_deducted[b2_key] = True

            bond_type_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
            rdkit_bond_type = bond_type_map.get(consumption, Chem.BondType.SINGLE)
            if idx1 < combined.GetNumAtoms() and idx2 < combined.GetNumAtoms():
                combined.AddBond(idx1, idx2, rdkit_bond_type)

        try:
            return {"mol": combined.GetMol(),
                    "smiles": Chem.MolToSmiles(combined.GetMol()),
                    "residual_h": total_residual_h,
                    "offset_map": offset_map}
        except Exception:
            return None

    @staticmethod
    def _remove_capped_hydrogens(mol, atom_idx, capped_h_count):
        if capped_h_count <= 0 or atom_idx >= mol.GetNumAtoms():
            return 0
        atom = mol.GetAtomWithIdx(atom_idx)
        current_h = atom.GetNumExplicitHs()
        removable = min(current_h, capped_h_count)
        if removable > 0:
            atom.SetNumExplicitHs(max(0, current_h - removable))
            return removable
        return 0

    @staticmethod
    def _concat_fragments(fragments):
        """将所有碎片通过 CombineMols 拼接为断开超级分子（不增键）。

        仅在所有边界原子均无自由基电子时使用，作为最后手段。
        """
        try:
            combined = fragments[0].fragment_mol
            for frag in fragments[1:]:
                combined = Chem.CombineMols(combined, frag.fragment_mol)
            smiles = Chem.MolToSmiles(combined)
            return {"mol": combined, "smiles": smiles}
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 阶段二辅助方法
    # ------------------------------------------------------------------
    def _apply_expand_rules(self, fragments, strategy, direct_result):
        expanded_results = []
        offset_map = direct_result.get("offset_map") if direct_result else None
        residual_sites = self._find_residual_capped_sites(
            fragments, strategy, offset_map)
        if not residual_sites:
            return expanded_results

        for rule in self.expand_rules:
            valence = rule["valence"]
            total_demand = sum(s["remaining_radical"] for s in residual_sites)

            if total_demand == valence:
                mol = self._insert_intermediate_atom(direct_result["mol"], residual_sites, rule)
                if mol is not None:
                    try:
                        expanded_results.append({
                            "mol": mol, "smiles": Chem.MolToSmiles(mol),
                            "residual_h": 0, "rule_description": rule["description"]})
                    except Exception:
                        pass
            elif total_demand > valence:
                mol = self._insert_intermediate_chain(direct_result["mol"], residual_sites, rule, total_demand)
                if mol is not None:
                    try:
                        remaining = total_demand - valence * (total_demand // valence)
                        expanded_results.append({
                            "mol": mol, "smiles": Chem.MolToSmiles(mol),
                            "residual_h": remaining,
                            "rule_description": f"{rule['description']}（链式插入）"})
                    except Exception:
                        pass
        return expanded_results

    @staticmethod
    def _find_residual_capped_sites(fragments, strategy, offset_map=None):
        consumption: Dict[tuple, int] = {}
        for pair in strategy["pairs"]:
            b1_key = (pair["boundary_1"]["fragment_idx"],
                      pair["boundary_1"]["boundary"].atom_idx)
            b2_key = (pair["boundary_2"]["fragment_idx"],
                      pair["boundary_2"]["boundary"].atom_idx)
            c = pair["consumption"]
            consumption[b1_key] = consumption.get(b1_key, 0) + c
            consumption[b2_key] = consumption.get(b2_key, 0) + c
        sites = []
        for frag_idx, frag in enumerate(fragments):
            for bd in frag.boundaries:
                key = (frag_idx, bd.atom_idx)
                total_consumed = consumption.get(key, 0)
                remaining_radical = bd.radical_electrons - total_consumed
                remaining_capped_h = bd.capped_hydrogens - total_consumed
                if remaining_radical > 0 and remaining_capped_h > 0:
                    # 将局部索引转为全局索引
                    global_idx = bd.atom_idx
                    if offset_map:
                        global_idx = offset_map.get(
                            frag_idx, {}).get(bd.atom_idx, bd.atom_idx)
                    sites.append({
                        "fragment_idx": frag_idx,
                        "atom_idx": global_idx,
                        "capped_hydrogens": remaining_capped_h,
                        "remaining_radical": remaining_radical})
        return sites

    @staticmethod
    def _enumerate_unpaired_boundaries(fragments, paired_keys: set) -> List[Dict]:
        """收集未参与任何阶段一配对的边界原子。

        返回的边界须有 capped_hydrogens > 0（桥接原子插入需要移除封顶
        氢来成键），且满足以下至少一项：
          - radical_electrons > 0（有自由基但因键型限制未能配对）
          - 存在来自不同碎片的其它未配对边界（可尝试跨碎片桥接）

        纯增强：不影响已配对边界的处理路径。
        """
        unpaired = []
        for frag_idx, frag in enumerate(fragments):
            for bd in frag.boundaries:
                key = (frag_idx, bd.atom_idx)
                if key not in paired_keys and bd.capped_hydrogens > 0:
                    unpaired.append({
                        "fragment_idx": frag_idx,
                        "boundary": bd,
                        "initial_radical": bd.radical_electrons,
                    })

        if len(unpaired) < 2:
            # 单个未配对边界无法桥接（桥接需至少两个位点）；
            # 仅当该边界有自由基时保留供后续尝试
            return [u for u in unpaired if u["initial_radical"] > 0]

        # 多碎片场景：至少保留来自不同碎片的未配对边界
        distinct_frags = set(u["fragment_idx"] for u in unpaired)
        if len(distinct_frags) < 2:
            return []
        return unpaired

    def _apply_unpaired_expand_rules(
        self, fragments, base_merge: Dict,
        unpaired_boundaries: List[Dict],
    ) -> List[Dict]:
        """对未配对边界尝试桥接原子插入。

        构造一个仅包含未配对边界的虚拟策略，复用
        ``_find_residual_capped_sites`` 定位残余封顶位点，再调用
        ``_insert_intermediate_atom``（需求 == 价态）或
        ``_insert_intermediate_chain``（需求 > 价态）插入中间原子。

        与阶段二的 ``_apply_expand_rules`` 不同，此方法不依赖阶段一
        的配对策略；当阶段一零配对时仍能生成候选。
        """
        offset_map = base_merge.get("offset_map")
        # 直接基于 unpaired_boundaries 构造 residual_sites（不依赖
        # _find_residual_capped_sites，因其 pairs 为空时不会建立
        # consumption，显式构造更清晰且可控制筛选条件）
        residual_sites = []
        for u in unpaired_boundaries:
            frag_idx = u["fragment_idx"]
            bd = u["boundary"]
            if bd.radical_electrons > 0 and bd.capped_hydrogens > 0:
                global_idx = bd.atom_idx
                if offset_map:
                    global_idx = offset_map.get(
                        frag_idx, {}).get(bd.atom_idx, bd.atom_idx)
                residual_sites.append({
                    "fragment_idx": frag_idx,
                    "atom_idx": global_idx,
                    "capped_hydrogens": bd.capped_hydrogens,
                    "remaining_radical": bd.radical_electrons,
                })

        if not residual_sites:
            return []

        expanded_results = []
        total_demand = sum(s["remaining_radical"] for s in residual_sites)

        for rule in self.expand_rules:
            valence = rule["valence"]
            if total_demand == valence:
                mol = self._insert_intermediate_atom(
                    base_merge["mol"], residual_sites, rule)
                if mol is not None:
                    try:
                        expanded_results.append({
                            "mol": mol,
                            "smiles": Chem.MolToSmiles(mol),
                            "residual_h": 0,
                            "rule_description": rule["description"],
                        })
                    except Exception:
                        pass
            elif total_demand > valence:
                mol = self._insert_intermediate_chain(
                    base_merge["mol"], residual_sites, rule, total_demand)
                if mol is not None:
                    try:
                        remaining = (
                            total_demand
                            - valence * (total_demand // valence))
                        expanded_results.append({
                            "mol": mol,
                            "smiles": Chem.MolToSmiles(mol),
                            "residual_h": remaining,
                            "rule_description": (
                                f"{rule['description']}（链式插入）"),
                        })
                    except Exception:
                        pass
        return expanded_results

    @staticmethod
    def _insert_intermediate_atom(mol, residual_sites, rule):
        try:
            rw = Chem.RWMol(mol)
            mid = rw.AddAtom(Chem.Atom(rule["intermediate_atom"]))
            for site in residual_sites:
                idx = site["atom_idx"]
                if idx < rw.GetNumAtoms():
                    atom = rw.GetAtomWithIdx(idx)
                    atom.SetNumExplicitHs(max(0, atom.GetNumExplicitHs() - site["capped_hydrogens"]))
                    rw.AddBond(idx, mid, Chem.BondType.SINGLE)
            Chem.SanitizeMol(rw.GetMol())
            return rw.GetMol()
        except Exception:
            return None

    @staticmethod
    def _insert_intermediate_chain(mol, residual_sites, rule, total_demand):
        try:
            rw = Chem.RWMol(mol)
            n = total_demand // rule["valence"]
            chain = []
            for _ in range(n):
                chain.append(rw.AddAtom(Chem.Atom(rule["intermediate_atom"])))
            for i in range(len(chain) - 1):
                rw.AddBond(chain[i], chain[i + 1], Chem.BondType.SINGLE)
            if chain and residual_sites:
                # 连接链的起始端到第一个残余位点
                s1 = residual_sites[0]
                if s1["atom_idx"] < rw.GetNumAtoms():
                    a1 = rw.GetAtomWithIdx(s1["atom_idx"])
                    a1.SetNumExplicitHs(max(0, a1.GetNumExplicitHs() - s1["capped_hydrogens"]))
                    rw.AddBond(s1["atom_idx"], chain[0], Chem.BondType.SINGLE)
                # 当有 ≥2 个残余位点时，连接链的末端到最后一个位点
                if len(residual_sites) >= 2:
                    s2 = residual_sites[-1]
                    if s2["atom_idx"] < rw.GetNumAtoms():
                        a2 = rw.GetAtomWithIdx(s2["atom_idx"])
                        a2.SetNumExplicitHs(max(0, a2.GetNumExplicitHs() - s2["capped_hydrogens"]))
                        rw.AddBond(s2["atom_idx"], chain[-1], Chem.BondType.SINGLE)
            Chem.SanitizeMol(rw.GetMol())
            return rw.GetMol()
        except Exception:
            return None
