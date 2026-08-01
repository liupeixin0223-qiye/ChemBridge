from .rules import MergeRule, ExpandRule, CompoundRule
from .rules import ThreeLayerBondRuleEngine, BondTypeDecision
from .structure import Boundary, Compound, CompoundSet
from .merge_multifrag import MultiFragmentMerger

import logging
import numpy as np
from typing import List
from types import SimpleNamespace
import rdkit.Chem as Chem
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolops as rdmolops
import synrbl.SynMCSImputer.utils as utils

logger = logging.getLogger("synrbl")

# 三层合并规则引擎的单例实例
_bond_rule_engine = None
_multifrag_merger = None


def _get_bond_rule_engine() -> ThreeLayerBondRuleEngine:
    """获取三层合并规则引擎的单例。"""
    global _bond_rule_engine
    if _bond_rule_engine is None:
        _bond_rule_engine = ThreeLayerBondRuleEngine()
    return _bond_rule_engine


def _get_multifrag_merger(
    enable_multi_fragment: bool = True,
) -> MultiFragmentMerger:
    """获取多碎片合并器。

    当 enable_multi_fragment=True（默认）时使用单例缓存；
    当 enable_multi_fragment=False 时每次创建新实例，
    避免单例缓存以错误配置返回。
    """
    global _multifrag_merger
    if not enable_multi_fragment:
        return MultiFragmentMerger(
            bond_rule_engine=_get_bond_rule_engine(),
            enable_multi_fragment=False,
        )
    if _multifrag_merger is None:
        _multifrag_merger = MultiFragmentMerger(
            bond_rule_engine=_get_bond_rule_engine()
        )
    return _multifrag_merger


_BOND_TYPE_MAP = {
    "SINGLE": (rdchem.BondType.SINGLE, 1),
    "DOUBLE": (rdchem.BondType.DOUBLE, 2),
    "TRIPLE": (rdchem.BondType.TRIPLE, 3),
    "AROMATIC": (rdchem.BondType.AROMATIC, 1),
}


class NoExpandRule(Exception):
    def __str__(self):
        return "No expand rule found."


def expand_boundary(boundary: Boundary) -> Compound:
    compound = None
    for rule in ExpandRule.get_all():
        if rule.can_apply(boundary):
            compound = rule.apply()
            break
    if compound is None:
        raise NoExpandRule()
    return compound


def _fix_explicit_h_for_bond(atom, bond_order: int):
    """合并前移除封顶氢原子，腾出价键用于形成新键。

    氢封顶时添加的显式氢原子数等于 radical_electrons（键合需求数）。
    当两个边界原子形成 bond_order 阶的键时，各自需移除
    min(封顶氢数, bond_order) 个显式氢原子。
    """
    n_exp = atom.GetNumExplicitHs()
    if n_exp > 0:
        atom.SetNumExplicitHs(int(max(0, n_exp - bond_order)))


def merge_boundaries(boundary1: Boundary, boundary2: Boundary) -> Compound | None:
    """合并两个碎片的边界原子。

    优先尝试三层规则引擎（当至少一个边界原子具有自由基信息时）；
    若引擎无法决策或自由基信息缺失，则回退到原版 MergeRule 配置规则。
    """
    # === 三层规则引擎：当至少一个边界原子有自由基信息时激活 ===
    if boundary1.radical_electrons > 0 or boundary2.radical_electrons > 0:
        engine = _get_bond_rule_engine()
        decision = engine.determine_bond_type(
            boundary1.symbol, boundary1.radical_electrons,
            boundary2.symbol, boundary2.radical_electrons,
        )

        if decision.bond_type == "NONE":
            # 限制规则层明确禁止此配对（如 F-F、Cl-Cl），直接返回 None，
            # 阻止后续 MergeRule 回退绕过禁止规则。
            logger.debug(
                "ThreeLayerBondRuleEngine: forbidden pair %s-%s "
                "(layer=%s), returning None",
                boundary1.symbol, boundary2.symbol,
                decision.decision_layer,
            )
            return None

        bond_info = _BOND_TYPE_MAP.get(decision.bond_type)
        if bond_info is not None:
            bond_type, bond_nr = bond_info
            _fix_explicit_h_for_bond(boundary1.get_atom(), bond_nr)
            _fix_explicit_h_for_bond(boundary2.get_atom(), bond_nr)

            try:
                merge_result = utils.merge_two_mols(
                    boundary1.compound.mol,
                    boundary2.compound.mol,
                    boundary1.index,
                    boundary2.index,
                    bond_type,
                )
                mol = merge_result["mol"]
                rdmolops.SanitizeMol(mol)

                boundary1.compound.update(mol, boundary1)
                boundary1.compound.rules.extend(
                    boundary2.compound.rules
                )
                logger.debug(
                    "ThreeLayerBondRuleEngine: %s bond between %s-%s "
                    "(layer=%s)",
                    decision.bond_type,
                    boundary1.symbol,
                    boundary2.symbol,
                    decision.decision_layer,
                )
                return boundary1.compound
            except Exception as e:
                logger.warning(
                    "ThreeLayerBondRuleEngine merge failed "
                    "(%s), falling back to MergeRule: %s",
                    decision.bond_type, str(e),
                )

    # === 回退：原版基于 merge_rules.json 配置的合并规则 ===
    for rule in MergeRule.get_all():
        if not rule.can_apply(boundary1, boundary2):
            continue
        return rule.apply(boundary1, boundary2)
    return None


def update_compound(compound: Compound):
    for rule in CompoundRule.get_all():
        if rule.can_apply(compound):
            rule.apply(compound)
            break


def _merge_one_compound(compound: Compound) -> Compound:
    merged_compound = compound
    while len(merged_compound.boundaries) > 0:
        boundary1 = merged_compound.boundaries[0]
        try:
            compound2 = expand_boundary(boundary1)
            if len(compound2.boundaries) != 1:
                raise NotImplementedError(
                    "Compound expansion and merge is only supported for "
                    + "compounds with a single boundary atom."
                )
            merged_compound = merge_boundaries(boundary1, compound2.boundaries[0])
            if merged_compound is None:
                raise ValueError("No merge rule found.")
        except NoExpandRule:
            # If no compound rule is found, leave the compound as is
            merged_compound.update(merged_compound.mol, boundary1)
            pass
        if merged_compound is None:
            raise ValueError("No merge rule found.")
    return merged_compound


def _merge_two_compounds(compound1: Compound, compound2: Compound) -> Compound:
    boundaries1 = compound1.boundaries
    boundaries2 = compound2.boundaries
    merged_compound = None
    if len(boundaries1) != 1:
        raise NotImplementedError(
            ("Can only merge compounds with single boundary atom. ({})").format(
                len(boundaries1)
            )
        )
    if len(boundaries1) != len(boundaries2):
        if compound1.num_compounds == 1 and compound2.num_compounds == 1:
            # If boundaries don't match, try to expand-merge them.
            # This is only safe if the smiles contains only one compound,
            # otherwise MCS was probably wrong.
            compound1 = _merge_one_compound(compound1)
            compound2 = _merge_one_compound(compound2)
            compound1.concat(compound2)
            merged_compound = compound1
        else:
            raise ValueError(
                (
                    "Can not merge compounds with unequal "
                    + "number of boundaries. ({} != {})."
                ).format(len(boundaries1), len(boundaries2))
            )
    else:
        merged_compound = merge_boundaries(boundaries1[0], boundaries2[0])
    if merged_compound is None:
        raise ValueError("No merge rule found.")
    return merged_compound


def merge(compound_set: CompoundSet, enable_multi_fragment: bool = True) -> List[Compound]:
    merged_compound = None
    merged_compounds = []  # 多候选结果（仅 ≥3 碎片时使用）

    comps_with_boundaries, comps_without_boundaries = [], []
    removed_rules = []
    for c in compound_set.compounds:
        update_compound(c)
        if not c.active:
            removed_rules.extend(c.rules)
            continue
        if len(c.boundaries) == 0:
            comps_without_boundaries.append(c)
        else:
            comps_with_boundaries.append(c)

    if len(comps_with_boundaries) == 0 and len(comps_without_boundaries) > 0:
        merged_compound = comps_without_boundaries.pop()
    elif len(comps_with_boundaries) == 1:
        merged_compound = _merge_one_compound(comps_with_boundaries[0])
    elif len(comps_with_boundaries) == 2:
        try:
            merged_compound = _merge_two_compounds(
                comps_with_boundaries[0], comps_with_boundaries[1]
            )
        except (NotImplementedError, ValueError) as exc:
            # 旧逻辑无法处理（典型场景：多边界 compound，如
            # "Can only merge compounds with single boundary atom"）。
            # 降级到 MultiFragmentMerger 处理 2 碎片：其阶段一直接
            # 配对 + 阶段 1.5 补充桥接 + Concat 回退，能处理多边界
            # 与跨碎片桥接场景。纯增强：成功路径完全不变。
            logger.info(
                "_merge_two_compounds failed (%s), "
                "falling back to MultiFragmentMerger for 2 fragments.",
                exc,
            )
            multifrag = _get_multifrag_merger(enable_multi_fragment)
            fragment_infos = []
            for comp in comps_with_boundaries:
                frag_boundaries = []
                for bd in comp.boundaries:
                    frag_boundaries.append(SimpleNamespace(
                        atom_idx=bd.index,
                        original_atom_idx=bd.index,
                        bond_type_severed="SINGLE",
                        radical_electrons=bd.radical_electrons,
                        capped_hydrogens=bd.capped_hydrogens,
                        element=bd.symbol,
                    ))
                frag_info = SimpleNamespace(
                    fragment_mol=comp.mol,
                    fragment_smiles=comp.smiles,
                    boundaries=frag_boundaries,
                    radical_sites=[],
                    original_atom_map={},
                )
                fragment_infos.append(frag_info)

            candidates = multifrag.merge_fragments(fragment_infos)
            if candidates and not candidates[0].get("error"):
                for cand in candidates:
                    merged_mol = cand["merged_mol"]
                    mc = compound_set.add_compound(merged_mol)
                    mc.rules = list(removed_rules)
                    if cand.get("low_confidence_flag"):
                        logger.warning(
                            "2-frag fallback MultiFragmentMerger: %s",
                            cand.get("low_confidence_reason",
                                     "low confidence"),
                        )
                    if comps_without_boundaries:
                        smiles_parts = [mc.smiles]
                        src_parts = []
                        if mc.src_smiles:
                            src_parts.append(mc.src_smiles)
                        for c in comps_without_boundaries:
                            smiles_parts.append(c.smiles)
                            if c.src_smiles:
                                src_parts.append(c.src_smiles)
                            mc.rules.extend(c.rules)
                        combined_smiles = ".".join(smiles_parts)
                        mc.mol = Chem.MolFromSmiles(combined_smiles)
                        if src_parts:
                            mc.src_mol = Chem.MolFromSmiles(
                                ".".join(src_parts))
                    merged_compounds.append(mc)
                # merged_compound 保持 None，跳过单候选路径
            else:
                # MultiFragmentMerger 也未产出合法候选，重新抛出原异常，
                # 行为等同修改前（_merge_two_compounds 的 NotImplementedError）
                raise
    elif len(comps_with_boundaries) >= 3:
        # 多碎片合并：使用 MultiFragmentMerger
        multifrag = _get_multifrag_merger(enable_multi_fragment)

        # 将 Compound 对象转换为 FragmentInfo-like 对象
        fragment_infos = []
        for comp in comps_with_boundaries:
            frag_boundaries = []
            for bd in comp.boundaries:
                frag_boundaries.append(SimpleNamespace(
                    atom_idx=bd.index,
                    original_atom_idx=bd.index,
                    bond_type_severed="SINGLE",
                    radical_electrons=bd.radical_electrons,
                    capped_hydrogens=bd.capped_hydrogens,
                    element=bd.symbol,
                ))
            frag_info = SimpleNamespace(
                fragment_mol=comp.mol,
                fragment_smiles=comp.smiles,
                boundaries=frag_boundaries,
                radical_sites=[],
                original_atom_map={},
            )
            fragment_infos.append(frag_info)

        candidates = multifrag.merge_fragments(fragment_infos)

        if candidates and not candidates[0].get("error"):
            # 为每个合并候选创建独立的 Compound
            for cand in candidates:
                merged_mol = cand["merged_mol"]
                mc = compound_set.add_compound(merged_mol)
                mc.rules = list(removed_rules)
                if cand.get("low_confidence_flag"):
                    logger.warning(
                        "Multi-fragment merge: %s",
                        cand.get("low_confidence_reason", "low confidence"),
                    )
                # 手动合并无边界碎片（避免 concat 对 compound_set 的破坏性修改）
                if comps_without_boundaries:
                    smiles_parts = [mc.smiles]
                    src_parts = []
                    if mc.src_smiles:
                        src_parts.append(mc.src_smiles)
                    for c in comps_without_boundaries:
                        smiles_parts.append(c.smiles)
                        if c.src_smiles:
                            src_parts.append(c.src_smiles)
                        mc.rules.extend(c.rules)
                    combined_smiles = ".".join(smiles_parts)
                    mc.mol = Chem.MolFromSmiles(combined_smiles)
                    if src_parts:
                        mc.src_mol = Chem.MolFromSmiles(".".join(src_parts))
                merged_compounds.append(mc)
        else:
            error_msg = (
                candidates[0].get("error", "No valid merge found")
                if candidates
                else "No candidates produced"
            )
            raise NotImplementedError(
                f"Multi-fragment merge failed for {len(comps_with_boundaries)} "
                f"compounds: {error_msg}"
            )

    # 单候选路径（≤2 碎片）：包装为列表
    if merged_compound is not None:
        merged_compound.rules = removed_rules + merged_compound.rules
        for c in comps_without_boundaries:
            merged_compound.concat(c)
        return [merged_compound]

    if not merged_compounds:
        raise NotImplementedError(
            "Merging {} compounds is not supported.".format(len(comps_with_boundaries))
        )

    return merged_compounds
