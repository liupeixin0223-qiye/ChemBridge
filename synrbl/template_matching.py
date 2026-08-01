"""
template_matching.py — 模板匹配兜底模块（改进11，统一方案）

在确定性算法（Path A + Path B）失败之后、LLM Bridge 之前，
尝试利用预定义的反应模板库（查询推测表）进行配平。

核心架构：
  查询推测表 + 回溯遍历匹配 → 三种分类（全满足/子集/不匹配）
  全满足 → 路径 A：模板验证（添加物种 + 守恒检查）
  子集   → 路径 B：评分推测（注入 Bridge 提示词）
  不匹配 → 跳过（Bridge 兜底提示）

包含核心类：
  - TemplateMatcher：查询推测表 + 回溯遍历 + 三分类 + 模板验证
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from rdkit import Chem


# ============================================================
#  原子守恒检查（独立工具函数）
# ============================================================

def _check_atom_balance(reactants, products):
    """
    检查反应式是否原子守恒。
    使用 Chem.AddHs 将隐式氢转为显式氢原子对象后遍历计数，
    确保显式氢和隐式氢都被准确统计。
    """
    reactant_counts: Dict[str, int] = {}
    product_counts: Dict[str, int] = {}

    for smi in reactants:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
        if mol is None:
            # C-2 修复：无效 SMILES 不再静默跳过，直接报告不平衡
            return False, {"parse_error": f"无法解析反应物 SMILES: {smi}"}
        mol_h = Chem.AddHs(mol)
        for atom in mol_h.GetAtoms():
            elem = atom.GetSymbol()
            reactant_counts[elem] = reactant_counts.get(elem, 0) + 1

    for smi in products:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is not None:
                mol.UpdatePropertyCache(strict=False)
        if mol is None:
            # C-2 修复：无效 SMILES 不再静默跳过，直接报告不平衡
            return False, {"parse_error": f"无法解析产物 SMILES: {smi}"}
        mol_h = Chem.AddHs(mol)
        for atom in mol_h.GetAtoms():
            elem = atom.GetSymbol()
            product_counts[elem] = product_counts.get(elem, 0) + 1

    all_elements = set(reactant_counts.keys()) | set(product_counts.keys())
    imbalance: Dict[str, int] = {}
    balanced = True
    for elem in all_elements:
        diff = reactant_counts.get(elem, 0) - product_counts.get(elem, 0)
        if diff != 0:
            imbalance[elem] = diff
            balanced = False

    return balanced, imbalance


def _detect_halogen(smiles: str) -> List[str]:
    """
    从 SMILES 分子中检测所有卤素原子（Cl/Br/I）。

    用于卤素依赖型模板：模板副产物中的 "{X}" 占位符需要在
    应用时替换为输入分子中的实际卤素元素符号。

    返回检测到的卤素元素符号列表（去重），如 ["Cl", "Br"]。
    未检测到则返回空列表。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    halogens: List[str] = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ("Cl", "Br", "I") and sym not in halogens:
            halogens.append(sym)
    return halogens


def _match_feature_on_mol(
    mol,
    pattern: str,
    exclude_atom_indices: Set[int] = None,
) -> Set[int]:
    """
    在分子上执行 SMARTS 子结构匹配，排除指定原子索引。

    参数:
        mol: RDKit Mol 对象
        pattern: SMARTS 模式字符串
        exclude_atom_indices: 需要排除的原子索引集合（来自 exclude_atoms_from）

    返回:
        匹配到的原子索引集合（已排除指定原子）。匹配失败返回空集合。
    """
    if exclude_atom_indices is None:
        exclude_atom_indices = set()
    pat_mol = Chem.MolFromSmarts(pattern)
    if pat_mol is None:
        return set()
    match = mol.GetSubstructMatch(pat_mol)
    if not match:
        return set()
    matched_atoms = set(match)
    return matched_atoms - exclude_atom_indices


# ============================================================
#  模板匹配器（查询推测表 + 回溯遍历 + 三分类）
# ============================================================

class TemplateMatcher:
    """
    模板匹配器（查询推测表 + 回溯遍历 + 三分类）。

    核心数据结构为查询推测表（query inference table），每条记录
    定义了一种反应类型的结构特征签名，同时关联一个配平模板。

    对每条记录执行回溯遍历，将输入分子分配给特征要求，根据分配
    结果分类为全满足/子集/不匹配，分别走路径 A（模板验证）或
    路径 B（评分推测）。
    """

    def __init__(
        self,
        template_db_path: str = "reaction_templates.json",
        similarity_threshold: float = 0.5,
    ):
        # similarity_threshold 保留参数以兼容管线调用，当前逻辑不使用
        self.similarity_threshold = similarity_threshold
        self.inference_table = self._load_inference_table(template_db_path)

    # ----------------------------------------------------------
    #  加载查询推测表
    # ----------------------------------------------------------

    def _load_inference_table(self, path: str) -> list:
        """加载查询推测表。优先从文件加载，否则使用内置默认表。"""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return self._default_inference_table()

    def _default_inference_table(self) -> list:
        """
        内置默认查询推测表，涵盖常见有机反应类型。

        每条记录包含：
          - reaction_type: 反应类型标识
          - description: 反应类型描述
          - reactant_requirements: 反应物侧特征要求列表
          - product_requirements: 产物侧特征要求列表
          - template: 关联的配平模板（含 byproducts, coreactants 等）
          - commonness_rank: 反应类型常见度等级（1~5，1 为最常见）

        特征声明顺序规则：对于 scope: "same_molecule" 的多特征要求，
        特征必须按"大基团优先"顺序声明（包含原子数更多的排在前面），
        exclude_atoms_from 只能引用当前特征之前声明的特征。
        """
        return [
            # ---- 1. 酰胺偶联反应 ----
            {
                "reaction_type": "amide_coupling",
                "description": "羧酸与胺脱水缩合生成酰胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "amide_coupling",
                    "name": "酰胺偶联反应",
                    "label": "酰胺偶联",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": "[C:1](=[O:2])[OX2H1].[NX3;H3,H2,H1:3]>>[C:1](=[O:2])[NX3:3].[OH2]",
                },
                "commonness_rank": 1,
            },
            # ---- 2. 酯化反应 ----
            {
                "reaction_type": "ester_formation",
                "description": "羧酸与醇脱水缩合生成酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ester_formation",
                    "name": "酯化反应",
                    "label": "酯化",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": "[C:1](=[O:2])[OX2H1].[OH1:3]>>[C:1](=[O:2])[O:3].[OH2]",
                },
                "commonness_rank": 1,
            },
            # ---- 3. Wittig 反应 ----
            {
                "reaction_type": "wittig_reaction",
                "description": "醛酮与 Wittig 试剂反应生成烯烃",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[PX3]=[CX2]",
                             "label": "phosphorus_ylide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2]=[CX2]",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "wittig_reaction",
                    "name": "Wittig 反应",
                    "label": "Wittig",
                    "byproducts": ["[P](=O)(c1ccccc1)(c1ccccc1)c1ccccc1"],
                    "coreactants": [],
                    "reaction_smarts": "[C:1](=[O:2]).[P:3]=[C:4]>>[C:1]=[C:4].[O:2]=[P:3]",
                },
                "commonness_rank": 2,
            },
            # ---- 4. Suzuki 偶联反应 ----
            {
                "reaction_type": "suzuki_coupling",
                "description": "卤代芳烃与芳基硼酸的偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c[X;Cl,Br,I]",
                             "label": "aryl_halide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c[B](O)O",
                             "label": "boronic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_coupling",
                    "name": "Suzuki 偶联反应",
                    "label": "Suzuki偶联",
                    "byproducts": ["[B](O)(O){X}"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": "[c:1][X;Cl,Br,I:4].[c:2][B:5]([O:6])[O:7]>>[c:1][c:2].[B:5]([O:6])([O:7])[X:4]",
                },
                "commonness_rank": 2,
            },
            # ---- 5. Boc 保护基反应 ----
            {
                "reaction_type": "protection_boc",
                "description": "胺与 Boc 酸酐反应生成 Boc 保护胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
                             "label": "boc_anhydride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]C(=O)OC(C)(C)C",
                             "label": "boc_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "protection_boc",
                    "name": "Boc 保护基反应",
                    "label": "Boc保护",
                    "byproducts": ["CC(C)(C)O", "O=C=O"],
                    "coreactants": [],
                    # reaction_smarts 设为 None：RDKit RunReactants 无法正确处理
                    # Boc2O 的复杂变换（未映射原子被保留导致结构错误）。
                    # 该模板依赖原子守恒检查（_apply_template_for_full_match）。
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 6. Heck 反应 ----
            {
                "reaction_type": "heck_reaction",
                "description": "卤代芳烃与烯烃的偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c[X;Cl,Br,I]",
                             "label": "aryl_halide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX2]=[CX2]",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[CX2]=[CX2]",
                             "label": "aryl_alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "heck_reaction",
                    "name": "Heck 反应",
                    "label": "Heck",
                    "byproducts": ["[Pd]", "[H+].[{X}-]"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": "[c:1][X;Cl,Br,I:3].[C:2]=[C:4]>>[c:1][C:2]=[C:4].[H:5][X:3]",
                },
                "commonness_rank": 3,
            },
            # ---- 7. Grignard 加成反应 ----
            {
                "reaction_type": "grignard_addition",
                "description": "醛酮与 Grignard 试剂的加成反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Mg][CX4][X;Cl,Br,I]",
                             "label": "grignard_reagent",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H1]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "grignard_addition",
                    "name": "Grignard 加成反应",
                    "label": "Grignard",
                    "byproducts": ["[Mg](O){X}"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": "[C:1](=[O:2]).[C:6][Mg:3][X;Cl,Br,I:4]>>[C:1]([O:2])([C:6])[Mg:3][X:4]",
                },
                "commonness_rank": 2,
            },
            # ---- 8. Buchwald-Hartwig 胺化反应 ----
            {
                "reaction_type": "buchwald_hartwig",
                "description": "卤代芳烃与胺的交叉偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c[X;Cl,Br,I]",
                             "label": "aryl_halide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[NX3]",
                             "label": "aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "buchwald_hartwig",
                    "name": "Buchwald-Hartwig 胺化反应",
                    "label": "Buchwald-Hartwig",
                    "byproducts": ["[Na]{X}", "[Pd]"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": "[c:1][X;Cl,Br,I:3].[NH2:2]>>[c:1][NH:2].[H:4][X:3]",
                },
                "commonness_rank": 3,
            },
            # ---- 9. Wittig 反应（三苯基磷鎓盐） ----
            {
                "reaction_type": "wittig_phosphonium",
                "description": "三苯基磷鎓叶立德与醛酮反应生成烯烃和三苯基氧膦",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[P+;X4](c1ccccc1)(c1ccccc1)(c1ccccc1)",
                             "label": "triphenylphosphonium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CH0,CH1,CH2]=[CH0,CH1,CH2]",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "wittig_phosphonium",
                    "name": "Wittig 反应（三苯基磷鎓盐）",
                    "label": "Wittig-Ph3P",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "[H+]"],
                    "coreactants": [],
                    # reaction_smarts 设为 None：磷鎓盐 P+-C 在
                    # SMARTS 中难以准确表达 P=C 叶立德共振式，
                    # RunReactants 无法正确模拟该变换。
                    # 依赖原子守恒检查（_apply_template_for_full_match）。
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 10. Wittig 反应（三甲基磷鎓盐） ----
            {
                "reaction_type": "wittig_trimethyl",
                "description": "三甲基磷鎓叶立德与醛酮反应生成烯烃和三甲基氧膦",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C[P+](C)(C)",
                             "label": "trimethylphosphonium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CH0,CH1,CH2]=[CH0,CH1,CH2]",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "wittig_trimethyl",
                    "name": "Wittig 反应（三甲基磷鎓盐）",
                    "label": "Wittig-Me3P",
                    "byproducts": ["CP(C)(C)=O", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 11. Appel 溴化反应（CBr4 / PPh3 体系） ----
            {
                "reaction_type": "appel_bromination",
                "description": "醇与四溴化碳/PPh3反应生成烷基溴化物，副产TPPO和溴仿",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([OX2H1])",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "BrC(Br)(Br)Br",
                             "label": "cbr4",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][Br]",
                             "label": "alkyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "appel_bromination",
                    "name": "Appel 溴化反应（CBr4）",
                    "label": "Appel-Br",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "BrC(Br)Br"],
                    "coreactants": ["c1ccc(P(c2ccccc2)c2ccccc2)cc1"],
                    # reaction_smarts 设为 None：醇→烷基溴的
                    # 变换依赖整个分子骨架改写，单一 SMARTS 变换
                    # 规则难以通用表达。依赖原子守恒检查。
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 12. Appel 氯化反应（CCl4 / PPh3 体系） ----
            {
                "reaction_type": "appel_chlorination",
                "description": "醇与四氯化碳/PPh3反应生成烷基氯化物，副产TPPO和氯仿",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([OX2H1])",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "ClC(Cl)(Cl)Cl",
                             "label": "ccl4",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][Cl]",
                             "label": "alkyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "appel_chlorination",
                    "name": "Appel 氯化反应（CCl4）",
                    "label": "Appel-Cl",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "ClC(Cl)Cl"],
                    "coreactants": ["c1ccc(P(c2ccccc2)c2ccccc2)cc1"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 13. N-溴代琥珀酰亚胺替代：Br2CHBr 溴化胺类 ----
            {
                "reaction_type": "dibromomethane_amination_bromination",
                "description": "胺类与二溴甲烷（溴仿）反应，胺基被溴取代",
                "reactant_requirements": [
                    {
                        "features": [
                            {
                                "pattern": "[NX3;H2,H1]",
                                "label": "amine",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {
                                "pattern": "BrC(Br)Br",
                                "label": "bromoform",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "N[CX4](Br)Br",
                             "label": "nc_dibromo",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "n_br2_bromination",
                    "name": "Br2CHBr 胺溴化反应",
                    "label": "NBr2-Br",
                    "byproducts": ["NC(Br)Br"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 14. Wacker 氧化偶联（双烯烃 → 酮 + 乙醛） ----
            {
                "reaction_type": "wacker_oxidative_coupling",
                "description": "两个烯烃分子在氧化条件下偶联，生成偶联产物和乙醛",
                "reactant_requirements": [
                    {
                        "features": [
                            {
                                "pattern": "[CH0,CH1,CH2]=[CH0,CH1,CH2]",
                                "label": "alkene_1",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {
                                "pattern": "[CH0,CH1,CH2]=[CH0,CH1,CH2]",
                                "label": "alkene_2",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "wacker_oxidative_coupling",
                    "name": "Wacker 氧化偶联",
                    "label": "Wacker",
                    "byproducts": ["CC=O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 15. NaH/MeI N-烷基化 ----
            {
                "reaction_type": "nah_mei_n_alkylation",
                "description": "胺类在 NaH 存在下与碘甲烷反应，发生 N-甲基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {
                                "pattern": "[NX3;H2,H1]",
                                "label": "amine",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {
                                "pattern": "CI",
                                "label": "mei",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4H3]",
                             "label": "n_methyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "nah_mei_alkylation",
                    "name": "NaH/MeI N-甲基化",
                    "label": "NaH-MeI",
                    "byproducts": ["[Na+]", "[I-]", "[H][H]"],
                    "coreactants": ["[H-]", "[Na+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 16. ICl 碘化反应 ----
            {
                "reaction_type": "icl_iodination",
                "description": "胺类与一氯化碘反应，胺基被碘取代（亲电芳香取代）",
                "reactant_requirements": [
                    {
                        "features": [
                            {
                                "pattern": "[NX3;H2,H1]",
                                "label": "amine",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {
                                "pattern": "ICI",
                                "label": "icl",
                                "exclude_atoms_from": [],
                            }
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[IX1]",
                             "label": "iodo_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "icl_iodination",
                    "name": "ICl 碘化反应",
                    "label": "ICl-I",
                    "byproducts": ["NCI"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 17. LDA介导的N-甲基化反应 ----
            {
                "reaction_type": "lda_mei_alkylation",
                "description": "LDA介导的N-甲基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(C)[N-]C(C)C",
                             "label": "lda",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1]",
                             "label": "secondary_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4H3]",
                             "label": "n_methyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "lda_mei_alkylation",
                    "name": "LDA/MeI N-甲基化",
                    "label": "LDA-MeI-N",
                    "byproducts": ["CI", "CC(C)NC(C)C"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 18. Swern氧化反应（旁观者模式） ----
            {
                "reaction_type": "swern_spectator",
                "description": "Swern氧化反应（旁观者模式）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H1]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "swern_spectator",
                    "name": "Swern氧化",
                    "label": "Swern",
                    "byproducts": ["[H][H]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 19. 锂卤交换反应 ----
            {
                "reaction_type": "li_halogen_exchange",
                "description": "锂卤交换反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "li_halogen_exchange",
                    "name": "锂卤交换",
                    "label": "Li-X-Exch",
                    "byproducts": ["Br", "[Li]CCCC"],
                    "coreactants": ["[H+]", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 20. POCl3脱水反应 ----
            {
                "reaction_type": "pocl3_dehydration",
                "description": "POCl3脱水反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=P(Cl)(Cl)Cl",
                             "label": "pocl3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]=[CX3]",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "pocl3_dehydration",
                    "name": "POCl3脱水",
                    "label": "POCl3-DH",
                    "byproducts": ["O", "O=P(Cl)(Cl)Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 21. 邻苯二甲酰亚胺肼解反应 ----
            {
                "reaction_type": "phthalimide_hydrazinolysis",
                "description": "邻苯二甲酰亚胺肼解反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]1C(=O)c2ccccc2C1=O",
                             "label": "phthalimide_core",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "NN",
                             "label": "hydrazine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2]",
                             "label": "primary_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "phthalimide_hydrazinolysis",
                    "name": "邻苯二甲酰肼解",
                    "label": "Phthal-Hyd",
                    "byproducts": ["O=C(O)c1ccccc1C(=O)O", "CNN"],
                    "coreactants": ["O", "O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 22. 格氏试剂羧基化反应 ----
            {
                "reaction_type": "grignard_carboxylation",
                "description": "格氏试剂羧基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=C=O",
                             "label": "co2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Mg]",
                             "label": "magnesium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX4][Br]",
                             "label": "alkyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "grignard_carboxylation",
                    "name": "格氏羧基化",
                    "label": "Grign-CO2",
                    "byproducts": ["[MgH]Br", "O"],
                    "coreactants": ["[H]", "[H]", "[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 23. Mitsunobu反应（PBu3体系） ----
            {
                "reaction_type": "mitsunobu_pbu3",
                "description": "Mitsunobu反应（PBu3体系）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCCCP(CCCC)CCCC",
                             "label": "pbu3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N(=NC(=O)OCC)C(=O)OCC",
                             "label": "diad",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "mitsunobu_pbu3",
                    "name": "PBu3-Mitsunobu",
                    "label": "Mit-PBu3",
                    "byproducts": ["O=P(CCCC)(CCCC)CCCC", "CCOC(=O)NNC(=O)OCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 24. K2CO3/DMF介导的N-甲基化 ----
            {
                "reaction_type": "k2co3_dmf_methylation",
                "description": "K2CO3/DMF介导的N-甲基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]-[CX4H3]",
                             "label": "n_methyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "k2co3_dmf_methylation",
                    "name": "K2CO3/DMF甲基化",
                    "label": "K2CO3-DMF-Me",
                    "byproducts": ["CI"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 25. K2CO3介导的酯化反应 ----
            {
                "reaction_type": "k2co3_esterification",
                "description": "K2CO3介导的酯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX4][Br]",
                             "label": "alkyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "k2co3_esterification",
                    "name": "K2CO3酯化",
                    "label": "K2CO3-Ester",
                    "byproducts": ["Br"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 26. HMDS钠盐/MeI烷基化 ----
            {
                "reaction_type": "hmds_mei_alkylation",
                "description": "HMDS钠盐/MeI烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C[Si](C)(C)[N-][Si](C)(C)C",
                             "label": "hmds_na",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4H3]",
                             "label": "n_methyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "hmds_mei_alkylation",
                    "name": "HMDS/MeI烷基化",
                    "label": "HMDS-MeI",
                    "byproducts": ["CI", "C[Si](C)(C)[N-][Si](C)(C)O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 27. NaH/MeI/NaOAc烷基化 ----
            {
                "reaction_type": "nah_mei_naoac_alkylation",
                "description": "NaH/MeI/NaOAc烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[H-]",
                             "label": "hydride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[O-]",
                             "label": "carboxylate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4H3]",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "nah_mei_naoac_alkylation",
                    "name": "NaH/MeI/NaOAc烷基化",
                    "label": "NaH-MeI-Ac",
                    "byproducts": ["CI", "O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 28. Appel碘化反应（PPh3/I2） ----
            {
                "reaction_type": "appel_iodination_pph3",
                "description": "Appel碘化反应（PPh3/I2）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "pph3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "II",
                             "label": "iodine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][IX1]",
                             "label": "alkyl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "appel_iodination_pph3",
                    "name": "Appel碘化",
                    "label": "Appel-I",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "I"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 29. 乙烯基碘/硫醇烷基化 ----
            {
                "reaction_type": "vinyl_iodide_thio_alkylation",
                "description": "乙烯基碘/硫醇烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCI",
                             "label": "vinyl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[SX2H1]",
                             "label": "thiol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C[SX2]",
                             "label": "vinyl_sulfide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "vinyl_iodide_thio_alkylation",
                    "name": "乙烯基碘硫醚化",
                    "label": "VinylI-SH",
                    "byproducts": ["I"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 30. TBHP介导的酰胺偶联 ----
            {
                "reaction_type": "tbhp_amide_coupling",
                "description": "TBHP介导的酰胺偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(C)(C)OOC(C)(C)C",
                             "label": "tbhp",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3H1]=[OX1]",
                             "label": "aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "tbhp_amide_coupling",
                    "name": "TBHP酰胺偶联",
                    "label": "TBHP-Amide",
                    "byproducts": ["CC(C)(C)OOC(C)(C)C", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 31. DMSO/炔烃/胺氧化反应 ----
            {
                "reaction_type": "dmso_alkyne_amine_oxidation",
                "description": "DMSO/炔烃/胺氧化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CS(C)=O",
                             "label": "dmso",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C#C",
                             "label": "alkyne",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "dmso_alkyne_amine_oxidation",
                    "name": "DMSO炔胺氧化",
                    "label": "DMSO-Alkyn",
                    "byproducts": ["C[SH]=O"],
                    "coreactants": ["[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 32. 锍叶立德氧化反应 ----
            {
                "reaction_type": "sulfoxonium_oxidation",
                "description": "锍叶立德氧化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C[S+](C)(C)=O",
                             "label": "dmso",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]1[OX2][CX3]1",
                             "label": "epoxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sulfoxonium_oxidation",
                    "name": "锍叶立德氧化",
                    "label": "Sulfox-Ox",
                    "byproducts": ["C[S+](C)(=O)O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 33-34. [已删除] EDC/HOBt模板与amide_coupling重复，无实际EDC/HOBt检测 ----
            # ---- 35. 锂卤交换反应（扩展） ----
            {
                "reaction_type": "li_halogen_extended",
                "description": "锂卤交换反应（扩展）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][I]",
                             "label": "aryl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[Li]",
                             "label": "aryl_lithium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "li_halogen_extended",
                    "name": "锂卤交换扩展",
                    "label": "Li-X-Ext",
                    "byproducts": ["I", "[Li]CCCC"],
                    "coreactants": ["[H+]", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 36. 氯代物甲氧基取代 ----
            {
                "reaction_type": "cl_to_ome_substitution",
                "description": "氯代物甲氧基取代",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Cl]",
                             "label": "aryl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CO",
                             "label": "methanol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[cX3][OX2][CX4]", "label": "methyl_aryl_ether", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "cl_to_ome_substitution",
                    "name": "Cl→OMe取代",
                    "label": "Cl-OMe",
                    "byproducts": ["Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 37. 磺酰氯转化为磺酰胺 ----
            {
                "reaction_type": "so2cl_to_so2nme2",
                "description": "磺酰氯转化为磺酰胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)([cX3])Cl",
                             "label": "sulfonyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CN(C)C",
                             "label": "dimehtylamine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)([cX3])[NX3]C",
                             "label": "sulfonamide_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "so2cl_to_so2nme2",
                    "name": "SO2Cl→SO2NMe2",
                    "label": "SO2Cl-NMe2",
                    "byproducts": ["Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 38. 异氰酸酯与胺反应（TEA） ----
            {
                "reaction_type": "isocyanate_amine_tea",
                "description": "异氰酸酯与胺反应（TEA）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX2]=[CX2]=[OX1]",
                             "label": "isocyanate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCN(CC)CC",
                             "label": "tea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX3](=[OX1])[NX3]",
                             "label": "urea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "isocyanate_amine_tea",
                    "name": "NCO/胺/TEA",
                    "label": "NCO-NH-TEA",
                    "byproducts": ["CCN(CC)CC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 39. 异氰酸酯与胺反应（DIPEA） ----
            {
                "reaction_type": "isocyanate_amine_dipea",
                "description": "异氰酸酯与胺反应（DIPEA）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX2]=[CX2]=[OX1]",
                             "label": "isocyanate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCN(C(C)C)C(C)C",
                             "label": "dipea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX3](=[OX1])[NX3]",
                             "label": "urea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "isocyanate_amine_dipea",
                    "name": "NCO/胺/DIPEA",
                    "label": "NCO-NH-DIPEA",
                    "byproducts": ["CCN(C(C)C)C(C)C"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 40. 异氰酸酯与胺反应（DIPEA+Cl） ----
            {
                "reaction_type": "isocyanate_amine_dipea_cl",
                "description": "异氰酸酯与胺反应（DIPEA+Cl）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX2]=[CX2]=[OX1]",
                             "label": "isocyanate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCN(C(C)C)C(C)C",
                             "label": "dipea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX3](=[OX1])[NX3]",
                             "label": "urea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "isocyanate_amine_dipea_cl",
                    "name": "NCO/胺/DIPEA-Cl",
                    "label": "NCO-DIPEA-Cl",
                    "byproducts": ["CCN(C(C)C)C(C)C", "Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 41. LDA/MeI介导的alpha-甲基化反应（O2氧化） ----
            {
                "reaction_type": "lda_mei_o2_alkylation",
                "description": "LDA/MeI介导的alpha-甲基化反应（O2氧化）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4;H2,H1]~[CX3]=[OX1]",
                             "label": "acidic_ch_carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)[N-]C(C)C",
                             "label": "lda",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4H3][CX4][CX3]=[OX1]",
                             "label": "alpha_methyl_carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "lda_mei_o2_alkylation",
                    "name": "LDA/MeI/O2 甲基化",
                    "label": "LDA-MeI-O2",
                    "byproducts": ["CC(C)[N-]C(C)O", "CI"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 42. 烯烃臭氧裂解生成羰基化合物 ----
            {
                "reaction_type": "ozonolysis",
                "description": "烯烃臭氧裂解生成羰基化合物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2]=[CX2]",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_fragment",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ozonolysis",
                    "name": "臭氧裂解反应",
                    "label": "Ozonolysis",
                    "byproducts": ["CO", "CO"],
                    "coreactants": ["O", "O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 43. 碳酸二甲酯/K2CO3介导的羧酸甲酯化 ----
            {
                "reaction_type": "k2co3_dmc_methylation",
                "description": "碳酸二甲酯/K2CO3介导的羧酸甲酯化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S(=O)(OC)OC",
                             "label": "dmc",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "k2co3_dmc_methylation",
                    "name": "DMC/K2CO3 甲酯化",
                    "label": "DMC-K2CO3",
                    "byproducts": ["COS(=O)(=O)OC", "O", "[K+]", "[K+]", "[OH-]", "[OH-]"],
                    "coreactants": ["[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 44. 三乙胺/氯醛介导的氧化环化反应 ----
            {
                "reaction_type": "tea_chloral_cyclization",
                "description": "三乙胺/氯醛介导的氧化环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCN(CC)CC",
                             "label": "tea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "ClC(Cl)(Cl)OC(OC(Cl)(Cl)Cl)=O",
                             "label": "chloral_orthoformate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4](Cl)(Cl)Cl",
                             "label": "trichloromethyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "tea_chloral_cyclization",
                    "name": "TEA/氯醛氧化环化",
                    "label": "TEA-Chloral",
                    "byproducts": ["CCN(CC)CC(OC(Cl)(Cl)Cl)OC(Cl)(Cl)Cl", "O", "O"],
                    "coreactants": ["[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 45. 四氯化碳与烯醇反应生成氯乙烯基化合物 ----
            {
                "reaction_type": "ccl4_enol_chlorination",
                "description": "四氯化碳与烯醇反应生成氯乙烯基化合物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]=[CX3]([OX2H1])",
                             "label": "enol_oh",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "ClC(Cl)(Cl)Cl",
                             "label": "ccl4",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4][Cl]",
                             "label": "alpha_chloro_carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ccl4_enol_chlorination",
                    "name": "CCl4烯醇氯化",
                    "label": "CCl4-Enol",
                    "byproducts": ["OC(Cl)(Cl)Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 46. HMDS钠盐/MeI介导的烷基化（O2氧化） ----
            {
                "reaction_type": "hmds_mei_o2_alkylation",
                "description": "HMDS钠盐/MeI介导的烷基化（O2氧化）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C[Si](C)(C)[N-][Si](C)(C)C",
                             "label": "hmds_na",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][CX4H3]",
                             "label": "methyl_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "hmds_mei_o2_alkylation",
                    "name": "HMDS/MeI/O2 烷基化",
                    "label": "HMDS-MeI-O2",
                    "byproducts": ["CI", "C[Si](C)(C)[N-][Si](C)(C)O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 47. 乙烯基碘化物与硫醇的氧化偶联 ----
            {
                "reaction_type": "vinyl_iodide_thiol_coupling_o2",
                "description": "乙烯基碘化物与硫醇的氧化偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCI",
                             "label": "vinyl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[SX2H1]",
                             "label": "thiol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C[SX2]",
                             "label": "vinyl_sulfide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "vinyl_iodide_thiol_coupling_o2",
                    "name": "乙烯基碘/硫醇偶联",
                    "label": "VinylI-SH-O2",
                    "byproducts": ["OCI"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 48. 过氧化氢氧化脱甲酰基反应 ----
            {
                "reaction_type": "h2o2_aldehyde_deformylation",
                "description": "过氧化氢氧化脱甲酰基反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1]=[OX1]",
                             "label": "aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "OO",
                             "label": "hydrogen_peroxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "h2o2_aldehyde_deformylation",
                    "name": "H2O2脱甲酰基",
                    "label": "H2O2-Deformyl",
                    "byproducts": ["O=CO", "OO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 49. 芳胺的Miyaura硼化反应 ----
            {
                "reaction_type": "miyaura_borylation_aryl_amine",
                "description": "芳胺的Miyaura硼化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cH,cX3][NX3;H2]",
                             "label": "aniline",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC1(C)C(C)(C)OB(O1)B2OC(C)(C)C(C)(C)O2",
                             "label": "bis_pinacol_diboron",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]B",
                             "label": "boryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "miyaura_borylation_aryl_amine",
                    "name": "Miyaura硼化",
                    "label": "Borylation",
                    "byproducts": ["CC1(C)OB(N)OC1(C)C", "O=N[O-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 50. 氨基酸三氯乙酯内酰胺环化 ----
            {
                "reaction_type": "lactam_cyclization_tce_ester",
                "description": "氨基酸三氯乙酯内酰胺环化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []},
                            {"pattern": "ClC(Cl)(Cl)C(O)OCC",
                             "label": "tce_ester",
                             "exclude_atoms_from": ["amine"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "lactam",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "lactam_cyclization_tce_ester",
                    "name": "TCE酯内酰胺化",
                    "label": "Lactam-TCE",
                    "byproducts": ["CCO", "O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 51. TBHP介导的醛与仲胺氧化酰胺化 ----
            {
                "reaction_type": "tbhp_oxidative_amidation",
                "description": "TBHP介导的醛与仲胺氧化酰胺化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1]=[OX1]",
                             "label": "aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1]",
                             "label": "secondary_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)(C)OO",
                             "label": "tbhp",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "tbhp_oxidative_amidation",
                    "name": "TBHP氧化酰胺化",
                    "label": "TBHP-Amide-Ox",
                    "byproducts": ["CC(C)(C)OO", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 52. 活性亚甲基与羰基化合物的Knoevenagel缩合 ----
            {
                "reaction_type": "knoevenagel_condensation",
                "description": "活性亚甲基与羰基化合物的Knoevenagel缩合",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4][CX3](=[OX1])",
                             "label": "active_methylene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]=[CX3][CX3]=[OX1]",
                             "label": "unsaturated_dicarbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "knoevenagel_condensation",
                    "name": "Knoevenagel缩合",
                    "label": "Knoevenagel",
                    "byproducts": ["CCO", "O"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 53. 逆烯反应脱羰基 ----
            {
                "reaction_type": "retro_ene_decarbonylation",
                "description": "逆烯反应脱羰基",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([OX2])[CX4]",
                             "label": "epoxide_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "retro_ene_decarbonylation",
                    "name": "逆烯脱羰",
                    "label": "RetroEne",
                    "byproducts": ["CO", "CO", "O"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 54. 硅基烯醇醚环化脱硅醇 ----
            {
                "reaction_type": "silyl_enol_ether_cyclization",
                "description": "硅基烯醇醚环化脱硅醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Si][OX2]",
                             "label": "silyl_ether",
                             "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": ["silyl_ether"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[Si][OX2]",
                             "label": "silyl_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "silyl_enol_ether_cyclization",
                    "name": "硅烯醇醚环化",
                    "label": "SiEnol-Cycl",
                    "byproducts": ["C[Si](C)(C)O", "O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 55. 内酯和酰胺的完全还原 ----
            {
                "reaction_type": "lactone_amide_full_reduction",
                "description": "内酯和酰胺的完全还原",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])O[CX4]",
                             "label": "lactone",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": ["lactone"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "lactone_amide_full_reduction",
                    "name": "内酯酰胺全还原",
                    "label": "LactAm-Red",
                    "byproducts": ["CO", "O"],
                    "coreactants": ["[H][H]", "[H][H]", "[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 56. 二酯的Dieckmann缩合环化 ----
            {
                "reaction_type": "dieckmann_cyclization",
                "description": "二酯的Dieckmann缩合环化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2]",
                             "label": "ester1",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=[OX1])[OX2]",
                             "label": "ester2",
                             "exclude_atoms_from": ["ester1"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX3](=[OX1])[CX4][CX3](=[OX1])", "label": "beta_keto_ester", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "dieckmann_cyclization",
                    "name": "Dieckmann环化",
                    "label": "Dieckmann",
                    "byproducts": ["CCO", "CCOC(=O)O"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 57. 腈与胍的嘧啶环合成 ----
            {
                "reaction_type": "pyrimidine_nitrile_guanidine",
                "description": "腈与胍的嘧啶环合成",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2]#[NX1]",
                             "label": "nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "NC(N)=N",
                             "label": "guanidine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cncnc1",
                             "label": "pyrimidine_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "pyrimidine_nitrile_guanidine",
                    "name": "腈/胍嘧啶合成",
                    "label": "Pyrimidine",
                    "byproducts": ["CO", "N"],
                    "coreactants": ["N"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 58. 乙酸乙烯酯选择性乙酰化 ----
            {
                "reaction_type": "vinyl_acetate_selective_acylation",
                "description": "乙酸乙烯酯选择性乙酰化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(OC=C)=O",
                             "label": "vinyl_acetate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "acetate_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "vinyl_acetate_selective_acylation",
                    "name": "乙酸乙烯酯乙酰化",
                    "label": "VAc-Acyl",
                    "byproducts": ["CC=O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 59. 多酯/酰胺的全局水解脱保护 ----
            {
                "reaction_type": "global_deprotection_hydrolysis",
                "description": "多酯/酰胺的全局水解脱保护",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2C]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "global_deprotection_hydrolysis",
                    "name": "全局水解脱保护",
                    "label": "Global-Deprot",
                    "byproducts": ["CC(=O)O", "CO", "CO", "O", "O"],
                    "coreactants": ["O", "O", "O", "[H][H]", "[H][H]", "[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 60. 端炔与芳基溴的Sonogashira环化 ----
            {
                "reaction_type": "sonogashira_annulation",
                "description": "端炔与芳基溴的Sonogashira环化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C#C",
                             "label": "terminal_alkyne",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2]#[CX2]c",
                             "label": "aryl_alkyne",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "sonogashira_annulation",
                    "name": "Sonogashira环化",
                    "label": "Sonogashira-Ann",
                    "byproducts": ["BrCc1ccccc1"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 61. 关环复分解反应伴随氧化 ----
            {
                "reaction_type": "rcm_with_oxidation",
                "description": "关环复分解反应伴随氧化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CC[NX3]",
                             "label": "allyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "rcm_with_oxidation",
                    "name": "RCM氧化环化",
                    "label": "RCM-Ox",
                    "byproducts": ["CC=O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 62. 苯偶姻缩合变体 ----
            {
                "reaction_type": "benzoin_condensation_variant",
                "description": "苯偶姻缩合变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1C=O",
                             "label": "benzaldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OH2]",
                             "label": "water",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4][OX2H1]",
                             "label": "alpha_hydroxy_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "benzoin_condensation_variant",
                    "name": "苯偶姻缩合",
                    "label": "Benzoin",
                    "byproducts": ["O", "O=Cc1ccccc1"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 63. 酯水解伴随碘化 ----
            {
                "reaction_type": "ester_hydrolysis_iodination",
                "description": "酯水解伴随碘化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2C]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ester_hydrolysis_iodination",
                    "name": "酯水解碘化",
                    "label": "Ester-I-hydro",
                    "byproducts": ["CO", "I"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 64. DMSO介导的环化反应 ----
            {
                "reaction_type": "dmso_mediated_cyclization",
                "description": "DMSO介导的环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C[S+](C)(C)=O",
                             "label": "dmso",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "dmso_mediated_cyclization",
                    "name": "DMSO环化",
                    "label": "DMSO-Cycl",
                    "byproducts": ["C[S+](C)(=O)O", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 65. 硫酯裂解生成硫醇 ----
            {
                "reaction_type": "thioester_cleavage",
                "description": "硫酯裂解生成硫醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(=S)O[CX4]",
                             "label": "thioester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "thioester_cleavage",
                    "name": "硫酯裂解",
                    "label": "ThioEster-Cleave",
                    "byproducts": ["CC(O)=S"],
                    "coreactants": ["[H+]", "[H+]", "[S-2]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 66. 多组分杂环合成反应 ----
            {
                "reaction_type": "multicomponent_heterocycle_synthesis",
                "description": "多组分杂环合成反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cH,cX3][NH1]",
                             "label": "pyrrole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "multicomponent_heterocycle_synthesis",
                    "name": "多组分杂环合成",
                    "label": "Multi-Het",
                    "byproducts": ["C=O", "O", "O"],
                    "coreactants": ["O", "O", "[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 67. 缩醛的氧化脱保护 ----
            {
                "reaction_type": "acetal_oxidative_deprotection",
                "description": "缩醛的氧化脱保护",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([OX2])([OX2])",
                             "label": "acetal",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "acetal_oxidative_deprotection",
                    "name": "缩醛氧化脱保护",
                    "label": "Acetal-OxDp",
                    "byproducts": ["C=O", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 68. 多酯的还原水解 ----
            {
                "reaction_type": "multi_ester_reductive_hydrolysis",
                "description": "多酯的还原水解",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2C]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "multi_ester_reductive_hydrolysis",
                    "name": "多酯还原水解",
                    "label": "MultiEster-RedHy",
                    "byproducts": ["CCO", "CCO", "O", "O"],
                    "coreactants": ["O", "O", "[H][H]", "[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 69. 硒氧化物消除反应 ----
            {
                "reaction_type": "selenoxide_elimination",
                "description": "硒氧化物消除反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Se](=O)c",
                             "label": "selenoxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "selenoxide_elimination",
                    "name": "硒氧化物消除",
                    "label": "SeOx-Elim",
                    "byproducts": ["O=[SeH]c1ccccc1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 70. 复杂底物的酰胺偶联（NaOH） ----
            {
                "reaction_type": "amide_coupling_complex_naoh",
                "description": "复杂底物的酰胺偶联（NaOH）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(OC(=O)C)=O",
                             "label": "acetic_anhydride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "amide_coupling_complex_naoh",
                    "name": "酰胺偶联NaOH",
                    "label": "Amide-NaOH",
                    "byproducts": ["CC(=O)OC(C)N(C(C)C)C(C)C", "Cl", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 71. 正丁基锂/芳基溴/胺偶联 ----
            {
                "reaction_type": "buli_arbr_amination",
                "description": "正丁基锂/芳基溴/胺偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]c",
                             "label": "n_aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "buli_arbr_amination",
                    "name": "nBuLi/ArBr胺化",
                    "label": "nBuLi-ArBr-NH",
                    "byproducts": ["Br", "[Li]CCCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 72. 叠氮化钠开环氮丙啶 ----
            {
                "reaction_type": "nan3_aziridine_ring_opening",
                "description": "叠氮化钠开环氮丙啶",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N-]=[N+]=N",
                             "label": "azide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C1CN1",
                             "label": "aziridine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "nan3_aziridine_ring_opening",
                    "name": "NaN3开环",
                    "label": "NaN3-Azirid",
                    "byproducts": ["B=C=[N-]", "O", "O"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 73. 异氰酸酯与胺反应（TEA/Cl旁观） ----
            {
                "reaction_type": "isocyanate_amine_tea_cl_spectator",
                "description": "异氰酸酯与胺反应（TEA/Cl旁观）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX2]=[CX2]=[OX1]",
                             "label": "isocyanate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCN(CC)CC",
                             "label": "tea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX3](=[OX1])[NX3]",
                             "label": "urea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "isocyanate_amine_tea_cl_spectator",
                    "name": "NCO/胺/TEA-Cl",
                    "label": "NCO-NH-TEA-Cl",
                    "byproducts": ["CCN(CC)CC", "Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 74. 酯的氨解生成内酰胺 ----
            {
                "reaction_type": "ester_aminolysis_lactam",
                "description": "酯的氨解生成内酰胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2C]",
                             "label": "ester",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": ["ester"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ester_aminolysis_lactam",
                    "name": "酯氨解内酰胺化",
                    "label": "Ester-NH-Lactam",
                    "byproducts": ["CO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 75. 正丁基锂介导的双芳基偶联 ----
            {
                "reaction_type": "buli_diaryl_coupling",
                "description": "正丁基锂介导的双芳基偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "buli_diaryl_coupling",
                    "name": "nBuLi双芳基偶联",
                    "label": "nBuLi-BiAr",
                    "byproducts": ["Br", "Br", "[Li]CCCC"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 76. Suzuki偶联变体（芳基硼酸+芳基溴） ----
            {
                "reaction_type": "suzuki_coupling_variant",
                "description": "Suzuki偶联变体（芳基硼酸+芳基溴）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "OB(O)c",
                             "label": "aryl_boronic",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_coupling_variant",
                    "name": "Suzuki偶联变体",
                    "label": "Suzuki-Var",
                    "byproducts": ["Br", "Br", "OCc1ccc(B(O)O)s1"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 77. 二氯乙烷介导的环化反应 ----
            {
                "reaction_type": "dichloroethane_cyclization",
                "description": "二氯乙烷介导的环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "ClCCC(Cl)",
                             "label": "dichloroethane",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C=C(C)C(=C)",
                             "label": "diene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "dichloroethane_cyclization",
                    "name": "二氯乙烷环化",
                    "label": "DCE-Cycl",
                    "byproducts": ["ClCCCl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 78. 复杂硼酸酯的Suzuki偶联 ----
            {
                "reaction_type": "suzuki_complex_boronate",
                "description": "复杂硼酸酯的Suzuki偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC1(C)C(C)(C)OB(O1)",
                             "label": "pinacol_boronate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_complex_boronate",
                    "name": "复杂Suzuki偶联",
                    "label": "Suzuki-Complex",
                    "byproducts": ["Br", "Br", "CC(=O)[O-]", "CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C", "O=C([O-])[O-]", "[K+]", "[K+]", "[K+]"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 79. 三丁基膦Mitsunobu反应变体 ----
            {
                "reaction_type": "mitsunobu_pbu3_variant",
                "description": "三丁基膦Mitsunobu反应变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCCCP(CCCC)CCCC",
                             "label": "pbu3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2]c",
                             "label": "aryl_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "mitsunobu_pbu3_variant",
                    "name": "PBu3-Mitsunobu变体",
                    "label": "PBu3-Mit-Var",
                    "byproducts": ["O=P(CCCC)(CCCC)CCCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 80. Wittig反应与硅醚底物 ----
            {
                "reaction_type": "wittig_silyl_variant",
                "description": "Wittig反应与硅醚底物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "triphenylphosphine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX4][Br]",
                             "label": "alkyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "wittig_silyl_variant",
                    "name": "Wittig硅醚变体",
                    "label": "Wittig-Si",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "[Br-]", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 81. 碳酸铯介导的Suzuki偶联 ----
            {
                "reaction_type": "suzuki_cs2co3",
                "description": "碳酸铯介导的Suzuki偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_cs2co3",
                    "name": "Cs2CO3-Suzuki偶联",
                    "label": "Suzuki-Cs",
                    "byproducts": ["O=C(O)Br", "O=C([O-])[O-]", "[Cs+]", "[Cs+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 82. 脱羧偶联反应 ----
            {
                "reaction_type": "decarboxylative_coupling",
                "description": "脱羧偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C=CC(=O)O",
                             "label": "acrylate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "decarboxylative_coupling",
                    "name": "脱羧偶联",
                    "label": "Decarbox-Coup",
                    "byproducts": ["C=CC(=O)O", "CBr"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 83. 碳酸钠/二氧六环Suzuki偶联 ----
            {
                "reaction_type": "suzuki_na2co3_dioxane",
                "description": "碳酸钠/二氧六环Suzuki偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_na2co3_dioxane",
                    "name": "Na2CO3/dioxane-Suzuki",
                    "label": "Suzuki-Na2CO3",
                    "byproducts": ["COCCOC", "O=C(O)Br", "O=C([O-])[O-]", "[Na+]", "[Na+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 84. 碳酸钠/二氧戊环Suzuki偶联 ----
            {
                "reaction_type": "suzuki_na2co3_dioxolane",
                "description": "碳酸钠/二氧戊环Suzuki偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_na2co3_dioxolane",
                    "name": "Na2CO3/dioxolane-Suzuki",
                    "label": "Suzuki-Na2CO3-2",
                    "byproducts": ["C1COCCO1", "O=C(O)Br", "O=C([O-])[O-]", "[Na+]", "[Na+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 85. Buchwald-Hartwig偶联（邻甲苯基膦） ----
            {
                "reaction_type": "buchwald_otol3p_variant",
                "description": "Buchwald-Hartwig偶联（邻甲苯基膦）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]c",
                             "label": "n_aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "buchwald_otol3p_variant",
                    "name": "Buchwald-oTol3P",
                    "label": "BH-oTol3P",
                    "byproducts": ["Br", "CCN(C(C)C)C(C)C", "Cc1ccccc1P(=O)(c1ccccc1C)c1ccccc1C"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 86. 复杂底物的硫酸盐氧化反应 ----
            {
                "reaction_type": "complex_oxidation_sulfate",
                "description": "复杂底物的硫酸盐氧化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=S(=O)(OC)[CX4]",
                             "label": "sulfate_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "complex_oxidation_sulfate",
                    "name": "硫酸盐氧化",
                    "label": "Sulfate-Ox",
                    "byproducts": ["CC(=O)OC(C)O", "O", "O=C([O-])O", "O=S(=O)(O)O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 87. nBuLi/HMDS/LDA/MeI多试剂体系 ----
            {
                "reaction_type": "buli_hmds_lda_mei",
                "description": "nBuLi/HMDS/LDA/MeI多试剂体系",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)NC(C)C",
                             "label": "diisopropylamine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#7][CX4H3]",
                             "label": "n_methyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "buli_hmds_lda_mei",
                    "name": "nBuLi/HMDS/MeI",
                    "label": "nBuLi-HMDS-MeI",
                    "byproducts": ["CC(C)NC(C)C", "CI", "[Li]CCCO"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 88. 多组分复杂烷基化级联反应 ----
            {
                "reaction_type": "complex_alkylation_cascade",
                "description": "多组分复杂烷基化级联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)[N-]C(C)C",
                             "label": "lda",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX2]#[NX1]",
                             "label": "nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "complex_alkylation_cascade",
                    "name": "复杂烷基化级联",
                    "label": "Alkyl-Cascade",
                    "byproducts": ["CC(C)N", "CC(C)[N-]C(C)C", "CCCI", "[Li+]", "[Li]CCCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 89. 钙盐介导的酯化反应 ----
            {
                "reaction_type": "calcium_salt_esterification",
                "description": "钙盐介导的酯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Ca+2]",
                             "label": "calcium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[O-]",
                             "label": "carboxylate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "calcium_salt_esterification",
                    "name": "钙盐酯化",
                    "label": "Ca-Ester",
                    "byproducts": ["CO", "O=C([O-])C(O)CCS", "O=S(=O)(O)O", "[Ca+2]"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 90. Ipc2BCl介导的Suzuki偶联 ----
            {
                "reaction_type": "suzuki_ipc2bcl",
                "description": "Ipc2BCl介导的Suzuki偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "B(Cl)",
                             "label": "chloroborane",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_ipc2bcl",
                    "name": "Ipc2BCl-Suzuki",
                    "label": "Suzuki-Ipc",
                    "byproducts": ["CC1C(B(Cl)C2CC3CC(C2O)C3(C)C)CC2CC1C2(C)C", "CCO", "OO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 91. DIAD/PPh3 Mitsunobu反应 ----
            {
                "reaction_type": "mitsunobu_diad",
                "description": "DIAD/PPh3 Mitsunobu反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N(=NC(=O)OCC)C(=O)OCC",
                             "label": "diad",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "pph3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "mitsunobu_diad",
                    "name": "DIAD-Mitsunobu",
                    "label": "Mit-DIAD",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "CCOC(=O)NNC(=O)OCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 92. 环氧开环伴随甲基化 ----
            {
                "reaction_type": "epoxide_opening_methylation",
                "description": "环氧开环伴随甲基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C1CO1",
                             "label": "epoxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1]",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4][CX4][OX2H1]",
                             "label": "beta_amino_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "epoxide_opening_methylation",
                    "name": "环氧开环甲基化",
                    "label": "Epox-Me",
                    "byproducts": ["CC"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 93. 糖苷的全局脱保护 ----
            {
                "reaction_type": "glycoside_global_deprotection",
                "description": "糖苷的全局脱保护",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "OC(C)=O",
                             "label": "acetate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "glycoside_global_deprotection",
                    "name": "糖苷全局脱保护",
                    "label": "Glyco-Deprot",
                    "byproducts": ["CC(=O)O", "CC(=O)O", "CC(=O)O", "CC(=O)O", "OCC1OC=CC(O)C1O"],
                    "coreactants": ["O", "O", "O", "O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 94. 复杂磺酰胺级联反应 ----
            {
                "reaction_type": "complex_sulfonamide_cascade",
                "description": "复杂磺酰胺级联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)(c)N",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(=O)O",
                             "label": "acetic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "complex_sulfonamide_cascade",
                    "name": "磺酰胺级联",
                    "label": "Sulfonam-Casc",
                    "byproducts": ["CC(=O)O", "CCOC(=O)O", "COC(=O)C1CCN(S(=O)(=O)c2ccc(C)cc2)c2ccccc2C1=O", "Cl"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 95. 含三氟甲基底物的复杂还原反应 ----
            {
                "reaction_type": "cf3_complex_reduction",
                "description": "含三氟甲基底物的复杂还原反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "FC(F)(F)",
                             "label": "cf3_group",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "cf3_complex_reduction",
                    "name": "CF3复杂还原",
                    "label": "CF3-Red",
                    "byproducts": ["F", "F", "F", "O", "O=C(O)Cn1nc(-c2ccc(Cl)cc2)n(C2CC2)c1=O"],
                    "coreactants": ["[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 96. 叠氮还原伴随复杂环化 ----
            {
                "reaction_type": "azide_reduction_complex",
                "description": "叠氮还原伴随复杂环化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C12=CC=CC=C1N(O)N=N2",
                             "label": "benzotriazole_noxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "azide_reduction_complex",
                    "name": "叠氮还原环化",
                    "label": "Azide-Red-Cplx",
                    "byproducts": ["CN1CCOCC1", "Cl", "N#N", "O", "O", "On1nnc2ccccc21", "[Cl-]", "[Na+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 97. 硫酰胺的甲基化反应 ----
            {
                "reaction_type": "thioamide_methylation",
                "description": "硫酰胺的甲基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=S)N",
                             "label": "thioamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX2][CX4]",
                             "label": "thioether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "thioamide_methylation",
                    "name": "硫酰胺甲基化",
                    "label": "ThioAm-Me",
                    "byproducts": ["C=O", "CI", "Cl", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 98. 混合酸酐酯化反应 ----
            {
                "reaction_type": "mixed_anhydride_esterification",
                "description": "混合酸酐酯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)CC(=O)C",
                             "label": "ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "mixed_anhydride_esterification",
                    "name": "混合酸酐酯化",
                    "label": "MixAnh-Ester",
                    "byproducts": ["CC(C)CC(=O)O", "COS(=O)(=O)OC", "O=C([O-])[O-]", "[K+]", "[K+]"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 99. DABCO氟化试剂氟化反应 ----
            {
                "reaction_type": "dabco_fluorination",
                "description": "DABCO氟化试剂氟化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N+]([CX4]Cl)",
                             "label": "dabco_salt",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "F[B-](F)(F)F",
                             "label": "bf4",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[FX1][#6]",
                             "label": "alkyl_fluoride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dabco_fluorination",
                    "name": "DABCO氟化",
                    "label": "DABCO-F",
                    "byproducts": ["CO", "ClC[N+]12CC[NH+](CC1)CC2", "F[B-](F)(F)F", "F[B-](F)(F)F"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 100. N-羟基邻苯二甲酰亚胺介导的醇保护 ----
            {
                "reaction_type": "nhpi_alcohol_protection",
                "description": "N-羟基邻苯二甲酰亚胺介导的醇保护",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C1CC(=O)N(C1=O)OC(=O)",
                             "label": "nhpi_carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCN(CC)CC",
                             "label": "tea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][CX3](=[OX1])[NX3]",
                             "label": "protected_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "nhpi_alcohol_protection",
                    "name": "NHPI醇保护",
                    "label": "NHPI-Prot",
                    "byproducts": ["CCN(CC)CC", "O", "O=C1CCC(=O)N1"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 101. 磺酰胺醚偶联反应 ----
            {
                "reaction_type": "sulfonamide_ether_coupling",
                "description": "磺酰胺醚偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)(c)C",
                             "label": "mesyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX4][Br]",
                             "label": "alkyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "sulfonamide_ether_coupling",
                    "name": "磺酰胺醚偶联",
                    "label": "Sulfon-Ether",
                    "byproducts": ["NCC1CC1", "O", "O=[SH](=O)c1cc(F)cc(OCCBr)c1"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 102. 多组分复杂偶联反应 ----
            {
                "reaction_type": "complex_multicomponent_coupling",
                "description": "多组分复杂偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[N+]([O-])(=O)c",
                             "label": "nitro_arene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)N(C(C)C)CC",
                             "label": "dipea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "complex_multicomponent_coupling",
                    "name": "多组分偶联",
                    "label": "Multi-Coup",
                    "byproducts": ["CCN(C(C)C)C(C)C", "CCN(CC)CC", "Cl", "Cl", "Cl", "O=C([O-])[O-]", "O=[N+]([O-])c1ccc(O)cc1", "[Na+]", "[Na+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 103. 过氧化物介导的氧化反应 ----
            {
                "reaction_type": "peroxide_mediated_oxidation",
                "description": "过氧化物介导的氧化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "OO",
                             "label": "peroxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "peroxide_mediated_oxidation",
                    "name": "过氧化物氧化",
                    "label": "Perox-Ox",
                    "byproducts": ["C=COC(C)=O", "CC=O", "CCCCCCCCO", "CCCCCCO", "O=C([O-])[O-]", "[Na+]", "[Na+]"],
                    "coreactants": ["OO"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 104. 乙酰氯/DMF乙酰化反应 ----
            {
                "reaction_type": "acetylation_accl_dmf",
                "description": "乙酰氯/DMF乙酰化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(=O)Cl",
                             "label": "acetyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "acetamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "acetylation_accl_dmf",
                    "name": "AcCl/DMF乙酰化",
                    "label": "AcCl-Acyl",
                    "byproducts": ["CC(=O)Cl", "CNC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 105. Negishi偶联反应（有机锌试剂） ----
            {
                "reaction_type": "negishi_coupling",
                "description": "Negishi偶联反应（有机锌试剂）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Zn]",
                             "label": "zinc_reagent",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Cl]",
                             "label": "aryl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "negishi_coupling",
                    "name": "Negishi偶联",
                    "label": "Negishi",
                    "byproducts": ["CCCCCl", "CC[Zn]CC", "CN(C)CCO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 106. 钠介导的酯交换反应 ----
            {
                "reaction_type": "na_transesterification",
                "description": "钠介导的酯交换反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Na]",
                             "label": "sodium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2C]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCO",
                             "label": "ethanol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "na_transesterification",
                    "name": "钠酯交换",
                    "label": "Na-Transester",
                    "byproducts": ["C=O", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 107. 复杂底物的酰胺偶联变体 ----
            {
                "reaction_type": "complex_amide_coupling_variant",
                "description": "复杂底物的酰胺偶联变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(Cl)=O",
                             "label": "acid_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "complex_amide_coupling_variant",
                    "name": "复杂酰胺偶联",
                    "label": "Amide-Cplx",
                    "byproducts": ["CNC(CCO)CCCc1ccccc1N", "CO", "Cl", "O=C(Cl)C=Cc1ccccc1"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 108. nBuLi/锂吡啶偶联反应 ----
            {
                "reaction_type": "buli_lipyridine_coupling",
                "description": "nBuLi/锂吡啶偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]CCCC",
                             "label": "nbuli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Li]",
                             "label": "lithium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccncc1",
                             "label": "pyridine_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "buli_lipyridine_coupling",
                    "name": "nBuLi/LiPyr偶联",
                    "label": "nBuLi-LiPyr",
                    "byproducts": ["Br", "[Li]CCCC", "[Li]c1ccccn1"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 109. Vilsmeier型氯化反应 ----
            {
                "reaction_type": "vilsmeier_type_chlorination",
                "description": "Vilsmeier型氯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[N+]([O-])=O",
                             "label": "nitro",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[Cl]",
                             "label": "acid_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "vilsmeier_type_chlorination",
                    "name": "Vilsmeier氯化",
                    "label": "Vilsmeier-Cl",
                    "byproducts": ["O"],
                    "coreactants": ["[Cl-]", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 110. Balz-Schiemann氟化反应 ----
            {
                "reaction_type": "balz_schiemann_fluorination",
                "description": "Balz-Schiemann氟化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N+]#N",
                             "label": "diazonium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[F-]",
                             "label": "fluoride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4,c][F]",
                             "label": "fluorinated_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "balz_schiemann_fluorination",
                    "name": "Balz-Schiemann氟化",
                    "label": "Balz-Sch",
                    "byproducts": ["N#N", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 111. 盐酸介导的双氯化反应 ----
            {
                "reaction_type": "dichlorination_hcl",
                "description": "盐酸介导的双氯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H1]",
                             "label": "cyclic_imide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX4,c][Cl]", "label": "chlorinated", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "dichlorination_hcl",
                    "name": "HCl双氯化",
                    "label": "HCl-DiCl",
                    "byproducts": ["O", "O"],
                    "coreactants": ["[Cl-]", "[Cl-]", "[H+]", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 112. 内酯的酸性水解开环 ----
            {
                "reaction_type": "lactone_hydrolysis_acid",
                "description": "内酯的酸性水解开环",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])O[CX4]",
                             "label": "lactone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Li+]",
                             "label": "lithium_ion",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "lactone_hydrolysis_acid",
                    "name": "内酯酸性水解",
                    "label": "Lactone-Hy",
                    "byproducts": ["[Li+]"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 113. Ullmann型C-N偶联反应 ----
            {
                "reaction_type": "ullmann_cn_coupling",
                "description": "Ullmann型C-N偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Cl]",
                             "label": "aryl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OH2]",
                             "label": "water",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]c",
                             "label": "n_aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ullmann_cn_coupling",
                    "name": "Ullmann C-N偶联",
                    "label": "Ullmann-CN",
                    "byproducts": ["Cl", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 114. Rosenmund-von Braun氰化反应 ----
            {
                "reaction_type": "rosenmund_von_braun",
                "description": "Rosenmund-von Braun氰化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cX3][Br]",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NH3]",
                             "label": "ammonia",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[CX2]#[NX1]",
                             "label": "aryl_nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "rosenmund_von_braun",
                    "name": "RvB氰化",
                    "label": "RvB-CN",
                    "byproducts": ["Br", "O", "O", "O"],
                    "coreactants": ["[O]", "[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 115. 硫代磷酸酯S-烷基化 ----
            {
                "reaction_type": "phosphorothioate_alkylation",
                "description": "硫代磷酸酯S-烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[P](=S)(O)(O)",
                             "label": "phosphorothioate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX4][K]",
                             "label": "potassium_alkyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[P](=S)([OX2])[OX2]",
                             "label": "phosphorothioate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "phosphorothioate_alkylation",
                    "name": "硫代磷酸酯烷基化",
                    "label": "PS-Alkyl",
                    "byproducts": ["[KH]"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 116. 磺酰胺N,N-二甲基化 ----
            {
                "reaction_type": "sulfonamide_n_dialkylation",
                "description": "磺酰胺N,N-二甲基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)(c)Cl",
                             "label": "sulfonyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NH3]",
                             "label": "ammonia",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][SX4](=[OX1])(=[OX1])",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sulfonamide_n_dialkylation",
                    "name": "磺酰胺二甲基化",
                    "label": "SO2Cl-NMe2-2",
                    "byproducts": ["Cl", "O", "O"],
                    "coreactants": ["[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 117. Chichibabin型胺化反应 ----
            {
                "reaction_type": "chichibabin_amination",
                "description": "Chichibabin型胺化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccncc1",
                             "label": "pyridine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N[Na]",
                             "label": "sodium_amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "Nc1ccccn1",
                             "label": "aminopyridine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "chichibabin_amination",
                    "name": "Chichibabin胺化",
                    "label": "Chichibabin",
                    "byproducts": ["O[Na]"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 118. 高锰酸钾氧化反应 ----
            {
                "reaction_type": "kmno4_oxidation",
                "description": "高锰酸钾氧化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=[Mn](=O)(=O)[O-]",
                             "label": "permanganate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[cH,cX3][CX4]",
                             "label": "benzylic_ch",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[CX3](=[OX1])[OX2H]",
                             "label": "benzoic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "kmno4_oxidation",
                    "name": "KMnO4氧化",
                    "label": "KMnO4-Ox",
                    "byproducts": ["O=[N+]([O-])[Mn](=O)(=O)[O-]", "[H]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 119. 硼氢化钠还原胺化 ----
            {
                "reaction_type": "nabh4_reductive_amination",
                "description": "硼氢化钠还原胺化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[BH4-]",
                             "label": "borohydride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4;H2,H1,H0]",
                             "label": "amine_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "nabh4_reductive_amination",
                    "name": "NaBH4还原胺化",
                    "label": "NaBH4-RedAm",
                    "byproducts": ["O", "[BH4-]", "[Na+]"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 120. 硼介导的还原反应 ----
            {
                "reaction_type": "boron_mediated_reduction",
                "description": "硼介导的还原反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "B",
                             "label": "boron",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "boron_mediated_reduction",
                    "name": "硼还原",
                    "label": "B-Red",
                    "byproducts": ["B", "O"],
                    "coreactants": ["[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 121. DMSO介导的Corey-Chaykovsky环氧化 ----
            {
                "reaction_type": "corey_chaykovsky_dmso",
                "description": "DMSO介导的Corey-Chaykovsky环氧化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C[S+](C)(C)=O",
                             "label": "dmso",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]1[OX2][CX3]1",
                             "label": "epoxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "corey_chaykovsky_dmso",
                    "name": "Corey-Chaykovsky",
                    "label": "Corey-Chay",
                    "byproducts": ["C[SH+](C)=O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 122. 简单脱羧反应 ----
            {
                "reaction_type": "simple_decarboxylation",
                "description": "简单脱羧反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "simple_decarboxylation",
                    "name": "简单脱羧",
                    "label": "Decarbox",
                    "byproducts": ["O=C(O)C1CCN(C(=O)CO)CC1"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 123. 脱羧反应变体 ----
            {
                "reaction_type": "decarboxylation_variant",
                "description": "脱羧反应变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "decarboxylation_variant",
                    "name": "脱羧变体",
                    "label": "Decarbox-Var",
                    "byproducts": ["O=C(O)Cn1c(-c2ccc(Cl)c(Cl)c2)nc2cccnc21"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 124. Chan-Lam偶联反应（硼酸+胺） ----
            {
                "reaction_type": "chan_lam_coupling",
                "description": "Chan-Lam偶联反应（硼酸+胺）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "Fc1ccc(B2OB(c3ccc(F)cc3)OB(c3ccc(F)cc3)O2)cc1",
                             "label": "boroxine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]c",
                             "label": "n_aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "chan_lam_coupling",
                    "name": "Chan-Lam偶联",
                    "label": "Chan-Lam",
                    "byproducts": ["OB1OB(c2ccc(F)cc2)OB(c2ccc(F)cc2)O1"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 125. H2O2氧化脱醛反应 ----
            {
                "reaction_type": "h2o2_oxidative_deformylation",
                "description": "H2O2氧化脱醛反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)[c]",
                             "label": "aryl_aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "OO",
                             "label": "h2o2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]c",
                             "label": "phenol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "h2o2_oxidative_deformylation",
                    "name": "H2O2氧化脱醛",
                    "label": "H2O2-Deformyl",
                    "byproducts": ["O=CO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 126. B2pin2/亚硝酸盐硼化偶联反应 ----
            {
                "reaction_type": "b2pin2_nitrite_borylation",
                "description": "B2pin2/亚硝酸盐硼化偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]c",
                             "label": "aniline",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC1(C)C(C)(C)OB(O1)B2OC(C)(C)C(C)(C)O2",
                             "label": "b2pin2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N(=O)[O-]",
                             "label": "nitrite",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]B",
                             "label": "boryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "b2pin2_nitrite_borylation",
                    "name": "B2pin2硼化偶联",
                    "label": "B2pin2-Boryl",
                    "byproducts": ["CC1(C)OB(N)OC1(C)C", "O=N[O-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 127. 吲哚C3位草酸酯缩合反应 ----
            {
                "reaction_type": "indole_oxalate_condensation",
                "description": "吲哚C3位草酸酯缩合反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1c[nH]c2ccccc12",
                             "label": "indole_nh",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)C",
                             "label": "acetyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCOC(=O)C(=O)OCC",
                             "label": "diethyl_oxalate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1c[nH]c2ccccc12",
                             "label": "indole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "indole_oxalate_condensation",
                    "name": "吲哚草酸酯缩合",
                    "label": "Indole-Oxal",
                    "byproducts": ["CCO", "O"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 128. 硅基烯醇醚环化芳香化反应 ----
            {
                "reaction_type": "silyl_enol_cyclization_aromatization",
                "description": "硅基烯醇醚环化芳香化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C[Si](C)(C)OC=C",
                             "label": "silyl_enol_ether",
                             "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": ["silyl_enol_ether"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "silyl_enol_cyclization_aromatization",
                    "name": "硅基烯醇醚环化",
                    "label": "Si-Enol-Cyc",
                    "byproducts": ["C[Si](C)(C)O", "O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 129. 邻苯二甲酰亚胺内酯还原反应 ----
            {
                "reaction_type": "phthalimide_lactone_reduction",
                "description": "邻苯二甲酰亚胺内酯还原反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C1(=O)c2ccccc2C(=O)N1",
                             "label": "phthalimide",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)O",
                             "label": "lactone_or_ester",
                             "exclude_atoms_from": ["phthalimide"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "phthalimide_lactone_reduction",
                    "name": "邻苯二甲酰亚胺还原",
                    "label": "Phth-Red",
                    "byproducts": ["CO", "O"],
                    "coreactants": ["[H][H]", "[H][H]", "[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 130. 多酯全局还原脱保护反应 ----
            {
                "reaction_type": "global_reduction_deprotection_multiester",
                "description": "多酯全局还原脱保护反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3H1]",
                             "label": "amide_nh",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "global_reduction_deprotection_multiester",
                    "name": "多酯全局还原",
                    "label": "Multi-Est-Red",
                    "byproducts": ["CC(=O)O", "CO", "CO", "O", "O"],
                    "coreactants": ["O", "O", "O", "[H][H]", "[H][H]", "[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 131. 酯裂解重排反应(氧气参与) ----
            {
                "reaction_type": "ester_cleavage_rearrangement_o2",
                "description": "酯裂解重排反应(氧气参与)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []},
                            {"pattern": "C=C",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ester_cleavage_rearrangement_o2",
                    "name": "酯裂解重排",
                    "label": "Est-Cleave",
                    "byproducts": ["CO", "CO"],
                    "coreactants": ["O", "O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 132. 苯甲醛还原偶联反应 ----
            {
                "reaction_type": "benzaldehyde_reductive_coupling",
                "description": "苯甲醛还原偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)c1ccccc1",
                             "label": "benzaldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)c1ccccc1",
                             "label": "benzaldehyde_2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "benzaldehyde_reductive_coupling",
                    "name": "苯甲醛还原偶联",
                    "label": "BzAld-Couple",
                    "byproducts": ["O=Cc1ccccc1"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 133. 酯水解反应(HI参与) ----
            {
                "reaction_type": "ester_hydrolysis_hi",
                "description": "酯水解反应(HI参与)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ester_hydrolysis_hi",
                    "name": "酯水解(HI)",
                    "label": "Est-Hydro-HI",
                    "byproducts": ["CO"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 134. DMSO介导喹啉合成反应 ----
            {
                "reaction_type": "dmso_quinoline_synthesis",
                "description": "DMSO介导喹啉合成反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C#Cc1ccccc1",
                             "label": "aryl_alkyne",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]c",
                             "label": "aniline_nh2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CS(C)=O",
                             "label": "dmso",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccncc1",
                             "label": "pyridine_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dmso_quinoline_synthesis",
                    "name": "DMSO喹啉合成",
                    "label": "DMSO-Quin",
                    "byproducts": ["C[SH]=O", "O", "O"],
                    "coreactants": ["[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 135. DMSO介导复杂环化反应 ----
            {
                "reaction_type": "dmso_annulation_complex",
                "description": "DMSO介导复杂环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)",
                             "label": "aldehyde",
                             "exclude_atoms_from": []},
                            {"pattern": "n[nH]",
                             "label": "pyrazole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)C",
                             "label": "acetyl",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccc(OC)cc1",
                             "label": "anisole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C[S+](C)(C)=O",
                             "label": "dmso_sulfoxonium",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "dmso_annulation_complex",
                    "name": "DMSO环化反应",
                    "label": "DMSO-Annul",
                    "byproducts": ["C[S+](C)(=O)O", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 136. 环戊二烯酮与吡咯缩合反应 ----
            {
                "reaction_type": "cyclopentadienone_pyrrole_condensation",
                "description": "环戊二烯酮与吡咯缩合反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)c1ccccc1",
                             "label": "benzoyl",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=C(c1ccccc1)c1ccccc1)",
                             "label": "diphenyl_alkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc[nH]c1",
                             "label": "pyrrole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "cyclopentadienone_pyrrole_condensation",
                    "name": "环戊二烯酮吡咯缩合",
                    "label": "Cp-Pyrrole",
                    "byproducts": ["C=O", "O", "O"],
                    "coreactants": ["O", "O", "[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 137. 复杂酯水解还原反应 ----
            {
                "reaction_type": "complex_ester_hydrolysis_reduction",
                "description": "复杂酯水解还原反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)OCC",
                             "label": "ethyl_ester",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccc([N+](=O)[O-])cc1",
                             "label": "nitro_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "complex_ester_hydrolysis_reduction",
                    "name": "复杂酯水解还原",
                    "label": "Cx-Est-HR",
                    "byproducts": ["CCO", "CCO", "O", "O"],
                    "coreactants": ["O", "O", "[H][H]", "[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 138. Ac2O/DIPEA酰化反应 ----
            {
                "reaction_type": "ac2o_dipea_acylation",
                "description": "Ac2O/DIPEA酰化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)OC(=O)C",
                             "label": "acetic_anhydride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)N(C(C)C)CC",
                             "label": "dipea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]c",
                             "label": "aniline_nh2",
                             "exclude_atoms_from": []},
                            {"pattern": "c1cc(F)ccc1",
                             "label": "fluoro_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(=O)[NX3]",
                             "label": "acetamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ac2o_dipea_acylation",
                    "name": "Ac2O/DIPEA酰化",
                    "label": "Ac2O-DIPEA",
                    "byproducts": ["CC(=O)OC(C)N(C(C)C)C(C)C", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 139. 含CF3杂环环化反应 ----
            {
                "reaction_type": "cf3_heterocycle_cyclization",
                "description": "含CF3杂环环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "FC(F)(F)C=O",
                             "label": "trifluoroacetaldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc([N+](=O)[O-])cc1",
                             "label": "nitro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "NC(CO)",
                             "label": "amino_alcohol",
                             "exclude_atoms_from": ["nitro_aryl"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "FC(F)(F)",
                             "label": "trifluoromethyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "cf3_heterocycle_cyclization",
                    "name": "CF3杂环环化",
                    "label": "CF3-Het-Cyc",
                    "byproducts": ["O"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 140. 酯氨解反应(氨) ----
            {
                "reaction_type": "ester_aminolysis_nh3",
                "description": "酯氨解反应(氨)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NH3]",
                             "label": "ammonia",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NH2]",
                             "label": "primary_amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ester_aminolysis_nh3",
                    "name": "酯氨解",
                    "label": "Est-NH3",
                    "byproducts": ["CO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 141. 锂介导芳基偶联(双溴代物) ----
            {
                "reaction_type": "lithium_aryl_coupling_dibr",
                "description": "锂介导芳基偶联(双溴代物)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(CC[Li])C",
                             "label": "buli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "C#N",
                             "label": "nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide_2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "lithium_aryl_coupling_dibr",
                    "name": "锂介导芳基偶联",
                    "label": "Li-Ar-Ar",
                    "byproducts": ["Br", "Br", "[Li][CH2]CCC"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 142. Suzuki偶联(硼酸变体) ----
            {
                "reaction_type": "suzuki_coupling_boronic_acid_variant",
                "description": "Suzuki偶联(硼酸变体)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)C",
                             "label": "enone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide_2",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3H1](=O)",
                             "label": "aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "OB(O)c1ccc(CO)s1",
                             "label": "boronic_acid_thiophene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_coupling_boronic_acid_variant",
                    "name": "Suzuki偶联变体",
                    "label": "Suz-BorVar",
                    "byproducts": ["Br", "Br", "OCc1ccc(B(O)O)s1"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 143. 糖精二烯环加成反应 ----
            {
                "reaction_type": "saccharin_diene_cycloaddition",
                "description": "糖精二烯环加成反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "ClCCCl",
                             "label": "dce",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C=C(C)C(=C)C",
                             "label": "diene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S1(=O)NC(=C1)",
                             "label": "saccharin_sultam",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "saccharin_diene_cycloaddition",
                    "name": "糖精二烯环加成",
                    "label": "Sac-Diene",
                    "byproducts": ["ClCCCl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 144. 复杂Suzuki偶联(K2CO3/B2pin2) ----
            {
                "reaction_type": "suzuki_b2pin2_k2co3_complex",
                "description": "复杂Suzuki偶联(K2CO3/B2pin2)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Br)cnc1",
                             "label": "bromo_heterocycle",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccccc1",
                             "label": "phenyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC([O-])=O",
                             "label": "acetate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC1(C)C(C)(C)OB(O1)B2OC(C)(C)C(C)(C)O2",
                             "label": "b2pin2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_b2pin2_k2co3_complex",
                    "name": "复杂Suzuki偶联",
                    "label": "Suz-B2pin2-K",
                    "byproducts": ["Br", "Br", "CC(=O)[O-]", "CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C", "O=C([O-])[O-]"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 145. 复杂Wittig烯化反应(PPh3) ----
            {
                "reaction_type": "wittig_olefination_complex_variant",
                "description": "复杂Wittig烯化反应(PPh3)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccc(Br)cc1",
                             "label": "benzyl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)OCC",
                             "label": "ethyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "triphenylphosphine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)",
                             "label": "aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "wittig_olefination_complex_variant",
                    "name": "复杂Wittig烯化",
                    "label": "Wittig-Cx",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "[Br-]", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 146. 脱羧交叉偶联(Cs2CO3) ----
            {
                "reaction_type": "decarboxylative_cross_coupling_cs2co3",
                "description": "脱羧交叉偶联(Cs2CO3)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3H1]C(=O)O",
                             "label": "carbamate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccnc1",
                             "label": "thiazole_or_pyridine",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "decarboxylative_cross_coupling_cs2co3",
                    "name": "脱羧交叉偶联",
                    "label": "Dcarbox-Cs",
                    "byproducts": ["O=C(O)Br", "O=C([O-])[O-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 147. Heck烯化反应(旁观者丙烯酸酯) ----
            {
                "reaction_type": "heck_olefination_with_spectator",
                "description": "Heck烯化反应(旁观者丙烯酸酯)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCC(=O)O",
                             "label": "crotonic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)cc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "c1nn[nH]c1",
                             "label": "pyrazole_nh",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[CX3]=[CX3]",
                             "label": "styrene_type",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "heck_olefination_with_spectator",
                    "name": "Heck烯化(旁观者)",
                    "label": "Heck-Spec",
                    "byproducts": ["CBr"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 148. Suzuki偶联(Na2CO3/二甘醇二甲醚) ----
            {
                "reaction_type": "suzuki_na2co3_diglyme",
                "description": "Suzuki偶联(Na2CO3/二甘醇二甲醚)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(COC)OC",
                             "label": "diglyme",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []},
                            {"pattern": "C(F)(F)F",
                             "label": "cf3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(C=O)cc1",
                             "label": "benzaldehyde",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_na2co3_diglyme",
                    "name": "Suzuki偶联(二甘醇)",
                    "label": "Suz-Diglyme",
                    "byproducts": ["COCCOC", "O=C(O)Br", "O=C([O-])[O-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 149. Suzuki偶联(二氧六环/Na2CO3) ----
            {
                "reaction_type": "suzuki_dioxane_na2co3",
                "description": "Suzuki偶联(二氧六环/Na2CO3)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Br)cc(Cl)c1",
                             "label": "bromo_chloro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Cl)ccc1",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid_2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_dioxane_na2co3",
                    "name": "Suzuki偶联(二氧六环)",
                    "label": "Suz-Diox-Na",
                    "byproducts": ["O=C(O)Br", "O=C([O-])[O-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 150. Buchwald偶联(P(o-tol)3/DIPEA) ----
            {
                "reaction_type": "buchwald_p_otol3_dipea",
                "description": "Buchwald偶联(P(o-tol)3/DIPEA)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "phosphine_ligand",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)N(C)C",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)N(C(C)C)CC",
                             "label": "dipea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)cnc1",
                             "label": "bromo_heteroaryl",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)N",
                             "label": "urea_or_amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "buchwald_p_otol3_dipea",
                    "name": "Buchwald偶联",
                    "label": "Bw-Potol3",
                    "byproducts": ["Br", "CCN(C(C)C)C(C)C", "Cc1ccccc1P(=O)(c1ccccc1C)c1ccccc1C"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 151. Ac2O/HCOONa乙酰化反应 ----
            {
                "reaction_type": "ac2o_hcoona_acetylation",
                "description": "Ac2O/HCOONa乙酰化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)OC(=O)C",
                             "label": "acetic_anhydride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S(=O)(OC)c1ccccc1",
                             "label": "aryl_sulfonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C(=O)O",
                             "label": "bicarbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(=O)[NX3]",
                             "label": "acetamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ac2o_hcoona_acetylation",
                    "name": "Ac2O/HCOONa乙酰化",
                    "label": "Ac2O-HCOO",
                    "byproducts": ["CC(=O)OC(C)O", "O=C([O-])O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 152. MeI/K2CO3/DMF N-甲基化反应 ----
            {
                "reaction_type": "mei_k2co3_dmf_n_methylation",
                "description": "MeI/K2CO3/DMF N-甲基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H1,H0]",
                             "label": "amine",
                             "exclude_atoms_from": []},
                            {"pattern": "c1cc(Cl)nnc1",
                             "label": "chloro_heterocycle",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CN(C)C=O",
                             "label": "dmf",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4H3]",
                             "label": "n_methyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "mei_k2co3_dmf_n_methylation",
                    "name": "MeI/K2CO3甲基化",
                    "label": "MeI-K2CO3",
                    "byproducts": ["CI", "CNC=O", "O=C([O-])[O-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 153. BuLi/ArI/DIPEA/MeI复杂反应 ----
            {
                "reaction_type": "buli_ari_dipea_mei_complex",
                "description": "BuLi/ArI/DIPEA/MeI复杂反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(CC[Li])C",
                             "label": "buli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc([Si](C)(C)C(C)(C)C)cc1",
                             "label": "silyl_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#7][CX4H3]",
                             "label": "n_methyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "buli_ari_dipea_mei_complex",
                    "name": "BuLi/ArI复杂反应",
                    "label": "BuLi-ArI-Cx",
                    "byproducts": ["CI", "[Li][CH2]CCO"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 154. 钙盐酸化酯化反应变体 ----
            {
                "reaction_type": "calcium_salt_esterification_variant",
                "description": "钙盐酸化酯化反应变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[O-]C(=O)C(O)",
                             "label": "calcium_carboxylate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C(=O)C(O)",
                             "label": "calcium_carboxylate_2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "calcium_salt_esterification_variant",
                    "name": "钙盐酸化酯化",
                    "label": "Ca-Est-V",
                    "byproducts": ["O=C([O-])C(O)CCS"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 155. Suzuki-Miyaura偶联(Ipc硼烷) ----
            {
                "reaction_type": "suzuki_ipc_borane_coupling",
                "description": "Suzuki-Miyaura偶联(Ipc硼烷)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCO",
                             "label": "ethanol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccccc1",
                             "label": "phenyl",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)C",
                             "label": "ketone",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "B(Cl)",
                             "label": "chloroborane",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OH2]",
                             "label": "water",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_ipc_borane_coupling",
                    "name": "Suzuki Ipc硼烷偶联",
                    "label": "Suz-Ipc",
                    "byproducts": ["CC1C(B(Cl)C2CC3CC(C2O)C3(C)C)CC2CC1C2(C)C", "CCO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 156. 硫酸二甲酯/K2CO3甲基化反应 ----
            {
                "reaction_type": "dms_k2co3_methylation",
                "description": "硫酸二甲酯/K2CO3甲基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S(=O)(OC)OC",
                             "label": "dimethyl_sulfate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4H3]",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dms_k2co3_methylation",
                    "name": "DMS/K2CO3甲基化",
                    "label": "DMS-K2CO3",
                    "byproducts": ["COS(=O)(=O)OC", "O", "[OH-]", "[OH-]"],
                    "coreactants": ["[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 157. Mitsunobu反应(DIAD/PPh3变体) ----
            {
                "reaction_type": "mitsunobu_diad_pph3_variant",
                "description": "Mitsunobu反应(DIAD/PPh3变体)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[NX3]([OX2])",
                             "label": "hydroxy_phthalimide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "triphenylphosphine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N(=NC(=O)OCC)C(=O)OCC",
                             "label": "diad",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]C(=O)O",
                             "label": "amide_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "mitsunobu_diad_pph3_variant",
                    "name": "Mitsunobu(DIAD)",
                    "label": "Mits-DIAD-V",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "CCOC(=O)NNC(=O)OCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 158. 糖苷化脱乙酰化反应 ----
            {
                "reaction_type": "glycosylation_deacetylation",
                "description": "糖苷化脱乙酰化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)C",
                             "label": "acetyl",
                             "exclude_atoms_from": []},
                            {"pattern": "c1cc(Cl)ccc1",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "glycosylation_deacetylation",
                    "name": "糖苷化脱乙酰化",
                    "label": "Glyc-DeAc",
                    "byproducts": ["CC(=O)O", "CC(=O)O", "CC(=O)O", "CC(=O)O"],
                    "coreactants": ["O", "O", "O", "O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 159. 磺内酰胺环化反应(AcOH) ----
            {
                "reaction_type": "sultam_cyclization_acoh",
                "description": "磺内酰胺环化反应(AcOH)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(=O)O",
                             "label": "acetic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)(c1ccc(C)cc1)",
                             "label": "tosyl",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3]",
                             "label": "amine",
                             "exclude_atoms_from": ["tosyl"]},
                            {"pattern": "[CX3](=O)OCC",
                             "label": "ethyl_ester",
                             "exclude_atoms_from": ["tosyl", "amine"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)(c1ccccc1)",
                             "label": "sulfonyl",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": ["sulfonyl"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "sultam_cyclization_acoh",
                    "name": "磺内酰胺环化",
                    "label": "Sultam-Cyc",
                    "byproducts": ["CC(=O)O", "CCOC(=O)O", "COC(=O)C1CCN(S(=O)(=O)c2ccc(C)cc2)c2ccccc2C1=O"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 160. CF3三唑复杂环化反应 ----
            {
                "reaction_type": "cf3_triazole_complex_cyclization",
                "description": "CF3三唑复杂环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Cl)ccc1",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)N",
                             "label": "amide",
                             "exclude_atoms_from": ["chloro_aryl"]},
                            {"pattern": "[NX3]",
                             "label": "triazole",
                             "exclude_atoms_from": ["chloro_aryl", "amide"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                    {
                        "features": [
                            {"pattern": "FC(F)(F)C(O)",
                             "label": "cf3_alcohol",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)N",
                             "label": "amide_2",
                             "exclude_atoms_from": ["cf3_alcohol"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "FC(F)(F)",
                             "label": "trifluoromethyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "cf3_triazole_complex_cyclization",
                    "name": "CF3三唑环化",
                    "label": "CF3-Triaz",
                    "byproducts": ["F", "F", "F", "O", "O=C(O)Cn1nc(-c2ccc(Cl)cc2)n(C2CC2)c1=O"],
                    "coreactants": ["[H]", "[H]", "[H]", "[H]", "[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 161. DMS/K2CO3氨基酸甲基化反应 ----
            {
                "reaction_type": "dms_k2co3_amino_acid_methylation",
                "description": "DMS/K2CO3氨基酸甲基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Cl)ccc1",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2]",
                             "label": "primary_amine",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S(=O)(OC)OC",
                             "label": "dimethyl_sulfate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4H3]",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dms_k2co3_amino_acid_methylation",
                    "name": "DMS氨基酸甲基化",
                    "label": "DMS-AA-Me",
                    "byproducts": ["COS(=O)(=O)OC", "O", "[OH-]", "[OH-]"],
                    "coreactants": ["[H][H]", "[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 162. DMS/K2CO3甲基化(酮旁观者) ----
            {
                "reaction_type": "dms_k2co3_methylation_ketone_spectator",
                "description": "DMS/K2CO3甲基化(酮旁观者)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Cl)ccc1",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2]",
                             "label": "primary_amine",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)O",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)C",
                             "label": "ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S(=O)(OC)OC",
                             "label": "dimethyl_sulfate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4H3]",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dms_k2co3_methylation_ketone_spectator",
                    "name": "DMS甲基化(酮旁观)",
                    "label": "DMS-Ket-Spec",
                    "byproducts": ["CC(C)CC(=O)O", "COS(=O)(=O)OC", "O=C([O-])[O-]"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 163. DABCO氟化反应变体 ----
            {
                "reaction_type": "dabco_fluorination_variant",
                "description": "DABCO氟化反应变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N+]1(CCNCC1)",
                             "label": "dabco_salt",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C(=O)OC(C)(C)",
                             "label": "boc",
                             "exclude_atoms_from": []},
                            {"pattern": "c1c(Cl)nc(N)c1",
                             "label": "chloro_pyrimidine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[FX1][#6]",
                             "label": "alkyl_fluoride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dabco_fluorination_variant",
                    "name": "DABCO氟化变体",
                    "label": "DABCO-F-V",
                    "byproducts": ["ClC[N+]12CC[NH+](CC1)CC2"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 164. 复杂多组分酰胺偶联反应 ----
            {
                "reaction_type": "complex_multicomponent_amide_coupling",
                "description": "复杂多组分酰胺偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCN(CC)CC",
                             "label": "tea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C1CC(CCN1)",
                             "label": "piperazine",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)NC",
                             "label": "urea_or_carbamate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)N",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(C)N(C(C)C)CC",
                             "label": "dipea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[N+](=O)([O-])c1ccc(OC(Cl)=O)cc1",
                             "label": "nitro_chloroformate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "complex_multicomponent_amide_coupling",
                    "name": "多组分酰胺偶联",
                    "label": "MC-Amide",
                    "byproducts": ["CCN(C(C)C)C(C)C", "CCN(CC)CC", "Cl", "O=C([O-])[O-]", "O=[N+]([O-])c1ccc(O)cc1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 165. Baeyer-Villiger氧化(Na2CO3变体) ----
            {
                "reaction_type": "bv_oxidation_na2co3_variant",
                "description": "Baeyer-Villiger氧化(Na2CO3变体)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "OC=C",
                             "label": "vinyl_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCCCO",
                             "label": "alcohol_long_chain",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CC(OC=C)=O",
                             "label": "vinyl_acetate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[O-]C([O-])=O",
                             "label": "carbonate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2]",
                             "label": "ester_lactone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "bv_oxidation_na2co3_variant",
                    "name": "BV氧化(Na2CO3)",
                    "label": "BV-Na2CO3",
                    "byproducts": ["C=COC(C)=O", "CC=O", "CCCCCCCCO", "CCCCCCO", "O=C([O-])[O-]"],
                    "coreactants": ["OO"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 166. AcCl/DMF仲胺乙酰化反应 ----
            {
                "reaction_type": "accl_dmf_secondary_amine_acetylation",
                "description": "AcCl/DMF仲胺乙酰化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(C)(Cl)=O",
                             "label": "acetyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CN(C)C(C)=O",
                             "label": "dmf",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1]c1nccc1",
                             "label": "secondary_aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "acetamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "accl_dmf_secondary_amine_acetylation",
                    "name": "AcCl/DMF乙酰化",
                    "label": "AcCl-DMF-Ac",
                    "byproducts": ["CC(=O)Cl", "CNC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 167. 有机锌吡啶偶联反应 ----
            {
                "reaction_type": "organozinc_pyridyl_coupling",
                "description": "有机锌吡啶偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(C)[Zn]CC",
                             "label": "dietzinc",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCCCC",
                             "label": "pentane",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Cl)ncc1",
                             "label": "chloro_pyridine",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)C",
                             "label": "acetyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "organozinc_pyridyl_coupling",
                    "name": "有机锌吡啶偶联",
                    "label": "Zn-Py-Coup",
                    "byproducts": ["CCCCCl", "C[CH2][Zn][CH2]C"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 168. Na/EtOH酯交换反应 ----
            {
                "reaction_type": "transesterification_na_etoh",
                "description": "Na/EtOH酯交换反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCO",
                             "label": "ethanol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)OC",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2]",
                             "label": "amino_heterocycle",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "transesterification_na_etoh",
                    "name": "Na/EtOH酯交换",
                    "label": "Trans-NaEt",
                    "byproducts": ["C=O", "O"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 169. 复杂酰胺偶联(多旁观者) ----
            {
                "reaction_type": "complex_amide_coupling_multi_spectator",
                "description": "复杂酰胺偶联(多旁观者)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(Cl)=O",
                             "label": "acyl_chloride",
                             "exclude_atoms_from": []},
                            {"pattern": "C=Cc1ccccc1",
                             "label": "cinnamoyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(N)cc1",
                             "label": "aniline",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1]C(=O)c1ccc(OC)cc1",
                             "label": "methoxy_benzamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "complex_amide_coupling_multi_spectator",
                    "name": "复杂酰胺偶联",
                    "label": "CxAmd-Multi",
                    "byproducts": ["CNC(CCO)CCCc1ccccc1N", "CO", "O=C(Cl)C=Cc1ccccc1"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 170. BuLi/ArBr/锂吡啶偶联反应 ----
            {
                "reaction_type": "buli_arbr_lipyridine_coupling",
                "description": "BuLi/ArBr/锂吡啶偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(CC[Li])C",
                             "label": "buli",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)",
                             "label": "aldehyde",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)Nc1ccccc1",
                             "label": "benzamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)cnc1",
                             "label": "bromo_pyridine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Li]c1ccccn1",
                             "label": "lithium_pyridine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccncc1",
                             "label": "pyridine_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "buli_arbr_lipyridine_coupling",
                    "name": "BuLi/ArBr/吡啶锂偶联",
                    "label": "BuLi-PyLi",
                    "byproducts": ["Br", "[Li][CH2]CCC", "[Li][c]1ccccn1"],
                    "coreactants": ["[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 171. 羧酸转化为酰氯 ----
            {
                "reaction_type": "carboxylic_acid_to_acid_chloride",
                "description": "羧酸与氯化亚砜(SOCl2)反应转化为酰氯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[Cl]",
                             "label": "acid_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "carboxylic_acid_to_acid_chloride",
                    "name": "羧酸酰氯化",
                    "label": "Acid-Cl",
                    "byproducts": ["O=S=O", "[H+].[Cl-]"],
                    "coreactants": ["ClS(=O)Cl"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---- 172. 重氮盐氟化(Balz-Schiemann变体) ----
            {
                "reaction_type": "diazonium_to_fluoride_balz_schiemann",
                "description": "重氮盐氟化(Balz-Schiemann变体)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N+]#N",
                             "label": "diazonium",
                             "exclude_atoms_from": []},
                            {"pattern": "c1nccnc1",
                             "label": "pyrimidine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[F]c",
                             "label": "fluoro_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "diazonium_to_fluoride_balz_schiemann",
                    "name": "重氮盐氟化",
                    "label": "Diazo-F",
                    "byproducts": ["N#N", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 173. 环酰亚胺二氯化反应 ----
            {
                "reaction_type": "cyclic_imide_dichlorination",
                "description": "环酰亚胺二氯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=C1NC(=O)CC1",
                             "label": "cyclic_imide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
                             "label": "imide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "cyclic_imide_dichlorination",
                    "name": "环酰亚胺二氯化",
                    "label": "Imide-Cl2",
                    "byproducts": ["O", "O"],
                    "coreactants": ["[H+]", "[H+]", "[Cl-]", "[Cl-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 174. 格氏反应(MgO变体) ----
            {
                "reaction_type": "grignard_mgo_variant",
                "description": "格氏反应(MgO变体)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc(Br)cc(C)c1",
                             "label": "ortho_tolyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=C1CCCCN1",
                             "label": "lactam_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Mg]=O",
                             "label": "mgo",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "lactam",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "grignard_mgo_variant",
                    "name": "格氏反应(MgO)",
                    "label": "Grign-MgO",
                    "byproducts": ["[MgH][Br]", "O"],
                    "coreactants": ["[H]", "[H]", "[H]", "[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 175. 氨基甲酸酯开环反应(LiOH) ----
            {
                "reaction_type": "carbamate_ring_opening_lioh",
                "description": "氨基甲酸酯开环反应(LiOH)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)([NX3])",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []},
                            {"pattern": "O=C1OCCC1",
                             "label": "lactone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OH-]",
                             "label": "hydroxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "carbamate_ring_opening_lioh",
                    "name": "氨基甲酸酯开环",
                    "label": "Carb-Open",
                    "byproducts": [],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 176. 甲基化/氯取代反应(MeI) ----
            {
                "reaction_type": "methylation_cl_displacement_mei",
                "description": "甲基化/氯取代反应(MeI)",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1c(Cl)ncnc1",
                             "label": "chloro_purine_or_pyrimidine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#7][CX4H3]",
                             "label": "n_methyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "methylation_cl_displacement_mei",
                    "name": "甲基化氯取代",
                    "label": "Me-Cl-Disp",
                    "byproducts": ["Cl"],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 177. Rosenmund-von Braun氰化反应变体 ----
            {
                "reaction_type": "rosenmund_von_braun_cyanation_variant",
                "description": "Rosenmund-von Braun氰化反应变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1cc(Br)ccc1",
                             "label": "aryl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NH3]",
                             "label": "ammonia",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2]#[NX1]",
                             "label": "nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "rosenmund_von_braun_cyanation_variant",
                    "name": "RVB氰化变体",
                    "label": "RVB-CN-V",
                    "byproducts": ["Br", "O", "O", "O"],
                    "coreactants": ["[O]", "[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---- 178. 磺酰胺N,N-二甲基化反应 ----
            {
                "reaction_type": "sulfonamide_nn_dimethylation",
                "description": "磺酰胺N,N-二甲基化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "mei_2",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NH3]",
                             "label": "amine_nh3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)(Cl)",
                             "label": "sulfonyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4H3]",
                             "label": "n_methyl_sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sulfonamide_nn_dimethylation",
                    "name": "磺酰胺N,N-二甲基化",
                    "label": "Sulfon-NMe2",
                    "byproducts": ["Cl", "O", "O"],
                    "coreactants": ["[O]", "[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 179. NaBH4还原胺化反应变体 ----
            {
                "reaction_type": "reductive_amination_nabh4_variant",
                "description": "NaBH4还原胺化反应变体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)",
                             "label": "ketone_or_aldehyde",
                             "exclude_atoms_from": []},
                            {"pattern": "C1CCCCC1",
                             "label": "cyclohexane",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1,H2]CC",
                             "label": "secondary_or_primary_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4]",
                             "label": "alkylated_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "reductive_amination_nabh4_variant",
                    "name": "NaBH4还原胺化",
                    "label": "RedAm-NaBH4",
                    "byproducts": ["O"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 180. POCl3噁二唑环化反应 ----
            {
                "reaction_type": "pocl3_oxadiazole_cyclization",
                "description": "POCl3噁二唑环化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "ClP(Cl)(Cl)=O",
                             "label": "pocl3",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C(=O)[NX3][NX3]",
                             "label": "acyl_hydrazide",
                             "exclude_atoms_from": []},
                            {"pattern": "c1cc(Cl)ccc1",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": ["acyl_hydrazide"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#8;R]~[#6;R]~[#7;R]",
                             "label": "oxadiazole_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "pocl3_oxadiazole_cyclization",
                    "name": "POCl3噁二唑环化",
                    "label": "POCl3-OxDia",
                    "byproducts": ["O", "O=P(Cl)(Cl)Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 181. Remaining template for ID=174 ----
            {
                "reaction_type": "remaining_lactam_decarboxylation",
                "description": "内酰胺还原脱羧反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C1(C)OC2C(O1)C(N(OC)C2=O)",
                             "label": "isoxazolidinone_lactam",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "lactam",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_lactam_decarboxylation",
                    "name": "内酰胺还原脱羧",
                    "label": "Lact-DecO2",
                    "byproducts": ["O=C=O"],
                    "coreactants": ["[H][H]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 182. Remaining template for ID=427 ----
            {
                "reaction_type": "remaining_saccharin_diene_cyclo",
                "description": "糖精与二烯环加成（二氯乙烷为溶剂）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C(C)C=C",
                             "label": "diene",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=S1(=O)NC(=O)C=C1",
                             "label": "saccharin",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "remaining_saccharin_diene_cyclo",
                    "name": "糖精二烯环加成",
                    "label": "Sac-Diene",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 183. Remaining template for ID=478 ----
            {
                "reaction_type": "remaining_decarboxylative_cs2co3",
                "description": "碳酸铯促进的脱羧偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "Brc1ccccc1",
                             "label": "bromo_aryl",
                             "exclude_atoms_from": []},
                            {"pattern": "[Si](C)(C)C",
                             "label": "silyl_group",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ncsc1",
                             "label": "thiazole",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "remaining_decarboxylative_cs2co3",
                    "name": "Cs2CO3脱羧偶联",
                    "label": "Decarbox-Cs",
                    "byproducts": ["O=C=O", "Br"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 184. Remaining template for ID=482 ----
            {
                "reaction_type": "remaining_heck_crotonic",
                "description": "Heck偶联（巴豆酸参与）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCC(=O)O",
                             "label": "crotonic_acid",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "Brc1ccc(C)cc1",
                             "label": "bromo_toluene",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[CX3]=[CX3]",
                             "label": "styrene_type",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_heck_crotonic",
                    "name": "Heck-巴豆酸偶联",
                    "label": "Heck-Crot",
                    "byproducts": ["CBr"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 185. Remaining template for ID=808 ----
            {
                "reaction_type": "remaining_balz_schiemann_v2",
                "description": "Balz-Schiemann氟化（重氮盐脱N2加F）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N+]#N",
                             "label": "diazonium",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[F]c",
                             "label": "fluoro_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "remaining_balz_schiemann_v2",
                    "name": "Balz-Schiemann氟化",
                    "label": "BalzSch-F",
                    "byproducts": ["N#N"],
                    "coreactants": ["[F-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 186. Remaining v2 for ID=149 ----
            {
                "reaction_type": "remaining_knoevenagel_malonate_v2",
                "description": "Knoevenagel缩合：乙酰基吡咯与丙二酸二乙酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[CX4]",
                             "label": "acetyl_ketone",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CCOC(=O)C(OCC)=O",
                             "label": "diethyl_malonate_exact",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]=[CX3][CX3]=[OX1]",
                             "label": "unsaturated_dicarbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_knoevenagel_malonate_v2",
                    "name": "Knoevenagel丙二酸酯缩合v2",
                    "label": "Knoev-Mal-v2",
                    "byproducts": ["CCO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 187. Remaining v2 for ID=336 ----
            {
                "reaction_type": "remaining_cyclopentadienone_amine_v2",
                "description": "环戊二烯酮与氮杂芳烃缩合脱CO",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)C=C",
                             "label": "enone_in_ring",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccccc1",
                             "label": "phenyl_substituted",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX2]",
                             "label": "divalent_nitrogen",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "remaining_cyclopentadienone_amine_v2",
                    "name": "环戊二烯酮缩合v2",
                    "label": "CpD-Amine-v2",
                    "byproducts": ["[C-]#[O+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 188. Remaining v2 for ID=449 ----
            {
                "reaction_type": "remaining_wittig_pph3_v2",
                "description": "Wittig烯烃化（三苯基膦+溴代底物+硅基醛）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CBr",
                             "label": "bromo_methylene",
                             "exclude_atoms_from": []},
                            {"pattern": "C(=O)OCC",
                             "label": "ethyl_ester",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
                             "label": "pph3",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Si]",
                             "label": "silyl_group",
                             "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)",
                             "label": "carbonyl",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_wittig_pph3_v2",
                    "name": "Wittig-PPh3烯烃化v2",
                    "label": "Wittig-PPh3-v2",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "Br"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 189. Remaining v2 for ID=506 ----
            {
                "reaction_type": "remaining_n_methylation_mei_v2",
                "description": "MeI/K2CO3体系N-甲基化（芳香族SMARTS）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1nc2c(n1Cc3ccccc3)c(Cl)nnc2O",
                             "label": "chloro_benzodiazepine_ar",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "methyl_iodide",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#7][CX4H3]",
                             "label": "n_methyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_n_methylation_mei_v2",
                    "name": "MeI/K2CO3甲基化v2",
                    "label": "MeI-K2CO3-v2",
                    "byproducts": ["I"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 190. Remaining v2 for ID=812 ----
            {
                "reaction_type": "remaining_benzodiazepine_dichloro_v2",
                "description": "苯二氮卓二酮二氯化（修正版）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c(=O)c(=O)",
                             "label": "diketone_aromatic",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccc(OC)cc1",
                             "label": "methoxy_aryl",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "lactam",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_benzodiazepine_dichloro_v2",
                    "name": "苯二氮卓二氯化v2",
                    "label": "BzD-Cl2-v2",
                    "byproducts": ["O", "O"],
                    "coreactants": ["Cl", "Cl"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 191. Remaining v2 for ID=829 ----
            {
                "reaction_type": "remaining_grignard_mgo_v2",
                "description": "MgO促进的Grignard加成（修正版）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "Brc1cc(C)ccc1",
                             "label": "bromo_tolyl",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N1CCC(=O)CC1",
                             "label": "lactam_6membered",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[CX3](=[OX1])",
                             "label": "aryl_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_grignard_mgo_v2",
                    "name": "Grignard-MgO加成v2",
                    "label": "Grig-MgO-v2",
                    "byproducts": ["Br"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 192. Remaining v2 for ID=838 ----
            {
                "reaction_type": "remaining_cl_displacement_v3",
                "description": "氯原子取代反应（三分子消耗版）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CH4]",
                             "label": "methane",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c1nc(Cl)cnc1",
                             "label": "chloro_pyrimidine",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OH2]",
                             "label": "water",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]c",
                             "label": "hydroxy_arene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_cl_displacement_v3",
                    "name": "氯取代反应v3",
                    "label": "Cl-Disp-v3",
                    "byproducts": ["Cl", "[H][H]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 193. Remaining v3 for ID=703 ----
            {
                "reaction_type": "remaining_acetylation_accl_v3",
                "description": "乙酰氯/DMAc体系乙酰化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CC(=O)Cl",
                             "label": "acetyl_chloride",
                             "exclude_atoms_from": []},
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H1]",
                             "label": "secondary_amine",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccccc1",
                             "label": "aryl_substituent",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "acetamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_acetylation_accl_v3",
                    "name": "AcCl/DMAc乙酰化",
                    "label": "AcCl-DMAc",
                    "byproducts": ["Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 194. Remaining for ID=407 (Suzuki coupling) ----
            {
                "reaction_type": "remaining_suzuki_boronate_v2",
                "description": "Suzuki偶联（硼酸酯参与，脱Br2）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)",
                             "label": "ketone",
                             "exclude_atoms_from": []},
                            {"pattern": "OC",
                             "label": "methoxy",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "Brc1ccccc1",
                             "label": "bromo_phenyl",
                             "exclude_atoms_from": []},
                            {"pattern": "C=O",
                             "label": "aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_suzuki_boronate_v2",
                    "name": "Suzuki硼酸酯偶联",
                    "label": "Suz-Bor-v2",
                    "byproducts": ["Br", "Br"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 195. Remaining for ID=509 (organolithium alkylation) ----
            {
                "reaction_type": "remaining_organolithium_alkylation_v2",
                "description": "有机锂烷基化反应（BuLi/MeI体系）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]",
                             "label": "lithium_reagent",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []},
                            {"pattern": "[Si]",
                             "label": "silyl_group",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "IC",
                             "label": "methyl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_organolithium_alkylation_v2",
                    "name": "有机锂烷基化",
                    "label": "OrgLi-Alk",
                    "byproducts": ["[Li]I", "CCCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 196. Remaining for ID=718 (organolithium pyridine coupling) ----
            {
                "reaction_type": "remaining_organolithium_pyridine_v2",
                "description": "有机锂-吡啶偶联反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Li]",
                             "label": "lithium_reagent",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[Li]",
                             "label": "lithium_pyridine",
                             "exclude_atoms_from": []},
                            {"pattern": "c1ccncc1",
                             "label": "pyridine_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)",
                             "label": "carbonyl",
                             "exclude_atoms_from": []},
                            {"pattern": "OC",
                             "label": "methoxy",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccncc1",
                             "label": "pyridine_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_organolithium_pyridine_v2",
                    "name": "有机锂吡啶偶联",
                    "label": "OrgLi-Py",
                    "byproducts": ["[Li]", "[Li]", "C=CCC"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 197. Remaining for ID=588 (CF3 deprotection) ----
            {
                "reaction_type": "remaining_cf3_deprotection_v2",
                "description": "三氟甲基脱除反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)",
                             "label": "carbonyl",
                             "exclude_atoms_from": []},
                            {"pattern": "C(F)(F)F",
                             "label": "cf3_group",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_cf3_deprotection_v2",
                    "name": "CF3脱除反应",
                    "label": "CF3-Dep",
                    "byproducts": ["F", "F", "F", "[O]"],
                    "coreactants": ["[H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 198. Remaining for ID=634 (quinoline esterification) ----
            {
                "reaction_type": "remaining_quinoline_esterification_v2",
                "description": "喹啉羧酸甲酯化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CO",
                             "label": "methanol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4H3]",
                             "label": "methyl_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "remaining_quinoline_esterification_v2",
                    "name": "喹啉甲酯化",
                    "label": "Quin-Est",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---- 199. Remaining for ID=680 (cyclopropylamine substitution) ----
            {
                "reaction_type": "remaining_cyclopropyl_substitution_v2",
                "description": "环丙基胺取代反应（脱HBr）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C1CC1",
                             "label": "cyclopropyl",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2]",
                             "label": "primary_amine",
                             "exclude_atoms_from": ["cyclopropyl"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                    {
                        "features": [
                            {"pattern": "[#6][Br]",
                             "label": "alkyl_bromide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]1[#6][#6]1[NX3]",
                             "label": "cyclopropyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "remaining_cyclopropyl_substitution_v2",
                    "name": "环丙基取代反应",
                    "label": "CyPr-Sub",
                    "byproducts": ["Br"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 200. 硝基芳烃还原羰基化（CO参与，生成杂环并释放2 CO2）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "reductive_carbonylation_nitroarene",
                "description": "硝基芳烃与CO发生还原羰基化，硝基被CO还原环化，生成含氮杂环并释放2分子CO2",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3+](=O)[O-]", "label": "nitro_aryl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]c",
                             "label": "aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "reductive_carbonylation_nitroarene",
                    "name": "硝基芳烃还原羰基化",
                    "label": "Nitro-CO",
                    "byproducts": ["O=C=O", "O=C=O"],
                    "coreactants": ["[C-]#[O+]", "[C-]#[O+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 201. N-甲基化（tBuOK/MeI，用于内酰胺、酰胺、仲胺的N-H甲基化）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "n_methylation_tBuOK_MeI",
                "description": "内酰胺/酰胺/仲胺的N-H在tBuOK碱作用下与MeI发生N-甲基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H1,H2]", "label": "nh", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4H3]",
                             "label": "n_methyl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "n_methylation_tBuOK_MeI",
                    "name": "N-甲基化（tBuOK/MeI）",
                    "label": "N-Me",
                    "byproducts": ["[K+].[I-]", "CC(C)(C)O"],
                    "coreactants": ["IC", "CC(C)(C)[O-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 202. 芳香族硝化反应（添加HNO3，释放H2O）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_nitration",
                "description": "芳香环上发生硝化反应，引入硝基，水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1", "label": "aryl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3+](=O)[O-]",
                             "label": "nitro_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_nitration",
                    "name": "芳香族硝化",
                    "label": "Ar-NO2",
                    "byproducts": ["O"],
                    "coreactants": ["O=[N+]([O-])O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 203. 甲基酮α-溴化（Br2，生成α-溴代甲基酮）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "ketone_alpha_bromination",
                "description": "甲基酮的α-甲基被Br2溴化，生成α-溴代甲基酮，副产物HBr",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CH3][CX3](=O)[#6]", "label": "methyl_ketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([Br])[CX3](=[OX1])",
                             "label": "alpha_bromo_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ketone_alpha_bromination",
                    "name": "甲基酮α-溴化",
                    "label": "α-Br-ketone",
                    "byproducts": ["[H][Br]"],
                    "coreactants": ["BrBr"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 204. 芳香族溴化（Br2，生成溴代芳烃，副产物HBr）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_bromination",
                "description": "芳香环上发生溴化反应，引入溴原子，副产物HBr",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1", "label": "aryl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[Br]c",
                             "label": "bromo_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_bromination",
                    "name": "芳香族溴化",
                    "label": "Ar-Br",
                    "byproducts": ["[H][Br]"],
                    "coreactants": ["BrBr"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 205. Corey-Fuchs反应（醛+CBr4→二溴烯烃，副产物H2O）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "corey_fuchs",
                "description": "醛与CBr4反应生成二溴烯烃，水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C(Br)(Br)=[#6]",
                             "label": "gem_dibromoalkene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "corey_fuchs",
                    "name": "Corey-Fuchs反应",
                    "label": "Corey-Fuchs",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1", "[H+]", "[Br-]"],
                    "coreactants": ["BrC(Br)(Br)Br", "c1ccc(P(c2ccccc2)c2ccccc2)cc1"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 206. 醛氧化为羧酸（需要氧化剂）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aldehyde_oxidation",
                "description": "醛基被氧化为羧基，水参与氧化过程",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aldehyde_oxidation",
                    "name": "醛氧化为羧酸",
                    "label": "R-CHO→COOH",
                    "byproducts": [],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 207. 芳环氯磺化（SO2Cl2作为氯磺化试剂）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_chlorosulfonation",
                "description": "芳环上发生氯磺化反应，引入-SO2Cl基团，HCl为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1", "label": "aryl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX4](=[OX1])(=[OX1])([Cl])c",
                             "label": "chlorosulfonyl_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_chlorosulfonation",
                    "name": "芳环氯磺化",
                    "label": "Ar-SO2Cl",
                    "byproducts": ["[H][Cl]"],
                    "coreactants": ["ClS(=O)(=O)Cl"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 208. 叠氮-炔环加成（分子内，生成三唑）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "azide_alkyne_cycloaddition",
                "description": "叠氮与炔烃发生分子内环加成，生成三唑环",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N]=[N+]=[N-]", "label": "azide", "exclude_atoms_from": []},
                            {"pattern": "C#C", "label": "alkyne", "exclude_atoms_from": ["azide"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "n1nnc[c]1",
                             "label": "triazole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "azide_alkyne_cycloaddition",
                    "name": "叠氮-炔环加成",
                    "label": "Azide-Alkyne",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 209. 烯烃复分解关环（RCM，生成环状烯烃）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "ring_closing_metathesis",
                "description": "双烯化合物发生烯烃复分解关环反应，生成环状烯烃，释放乙烯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C", "label": "alkene1", "exclude_atoms_from": []},
                            {"pattern": "C=C", "label": "alkene2", "exclude_atoms_from": ["alkene1"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "alkene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "ring_closing_metathesis",
                    "name": "烯烃复分解关环（RCM）",
                    "label": "RCM",
                    "byproducts": ["C=C"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 210. Dieckmann缩合（二酯分子内缩合，生成环酮）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "dieckmann_condensation",
                "description": "二酯分子内缩合形成环酮，醇为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[OX2][#6;!$(C=O)]", "label": "ester1", "exclude_atoms_from": []},
                            {"pattern": "C(=O)[OX2][#6;!$(C=O)]", "label": "ester2", "exclude_atoms_from": ["ester1"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[#6][CX3](=[OX1])",
                             "label": "beta_dicarbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "dieckmann_condensation",
                    "name": "Dieckmann缩合",
                    "label": "Dieckmann",
                    "byproducts": ["CO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 211. 烯丙基乙烯基醚Claisen重排
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "claisen_rearrangement",
                "description": "烯丙基乙烯基醚发生[3,3]-重排，生成不饱和酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCOCC=C", "label": "allyl_vinyl_ether", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4][CX4][CX3]=[CX3]",
                             "label": "unsaturated_carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "claisen_rearrangement",
                    "name": "Claisen重排",
                    "label": "Claisen",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 212. 醛酮Strecker反应（生成α-氨基腈）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "strecker_reaction",
                "description": "醛与HCN和胺发生Strecker反应，生成α-氨基腈",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([NX3])[CX2]#[NX1]",
                             "label": "amino_nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "strecker_reaction",
                    "name": "Strecker反应",
                    "label": "Strecker",
                    "byproducts": ["O"],
                    "coreactants": ["[NH4+].[Cl-]", "C#N"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 213. 醇磺酰化（引入磺酰基）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "alcohol_sulfonation",
                "description": "醇与磺酰化试剂反应生成磺酸酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][SX4](=[OX1])(=[OX1])[OX2H1]",
                             "label": "sulfonate_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "alcohol_sulfonation",
                    "name": "醇磺酰化",
                    "label": "Alcohol-OSO2R",
                    "byproducts": ["O"],
                    "coreactants": ["O=S(=O)(O)O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 214. 醇的磺酰胺化
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "alcohol_sulfonamidation",
                "description": "醇与磺酰胺反应生成磺酰胺衍生物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfonamide_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "alcohol_sulfonamidation",
                    "name": "醇的磺酰胺化",
                    "label": "Alcohol-SO2NH2",
                    "byproducts": ["O"],
                    "coreactants": ["O=S(=O)([NH2])"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 215. 分子内Aldol缩合（生成环状不饱和酮）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "intramolecular_aldol",
                "description": "二醛/二酮分子内Aldol缩合，生成环状不饱和酮，水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)", "label": "carbonyl", "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)", "label": "carbonyl2", "exclude_atoms_from": ["carbonyl"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H1]",
                             "label": "aldol_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "intramolecular_aldol",
                    "name": "分子内Aldol缩合",
                    "label": "Intra-Aldol",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 216. 芳烃磺化反应（引入磺酸基）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_sulfonation",
                "description": "芳环上发生磺化反应，引入-SO3H基团，水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1", "label": "aryl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[c][S](=O)(=O)[OX2H1]", "label": "sulfonic_acid", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "aromatic_sulfonation",
                    "name": "芳烃磺化",
                    "label": "Ar-SO3H",
                    "byproducts": ["O"],
                    "coreactants": ["O=S(=O)(O)O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 217. 芳环氯化反应（引入Cl）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_chlorination",
                "description": "芳环上发生氯化反应，引入Cl原子，HCl为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1", "label": "aryl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[Cl]c",
                             "label": "chloro_aryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_chlorination",
                    "name": "芳环氯化",
                    "label": "Ar-Cl",
                    "byproducts": ["[H][Cl]"],
                    "coreactants": ["ClCl"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 218. Boc脱保护（酸催化脱除Boc基团，生成CO2和异丁烯）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "boc_deprotection",
                "description": "Boc保护的胺在酸性条件下脱除，释放CO2和异丁烯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)OC(C)(C)C", "label": "boc", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "free_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "boc_deprotection",
                    "name": "Boc脱保护",
                    "label": "Boc-De",
                    "byproducts": ["O=C=O", "CC(C)=C"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 219. 缩醛/缩酮保护（二醇与二甲基丙烯酸反应生成缩醛）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "acetal_protection",
                "description": "二醇与丙酮二甲基缩醛反应生成环状缩醛，甲醇为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]", "label": "oh1", "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]", "label": "oh2", "exclude_atoms_from": ["oh1"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([OX2])([OX2])",
                             "label": "acetal_carbon",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "acetal_protection",
                    "name": "缩醛保护",
                    "label": "Acetal-Prot",
                    "byproducts": ["CO", "CO"],
                    "coreactants": ["COC(OC)(OC)C"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 220. 亚胺形成（胺+醛/酮→亚胺，水为副产物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "imine_formation",
                "description": "伯胺与醛/酮缩合生成亚胺，水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2]", "label": "primary_amine", "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)", "label": "carbonyl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]=[NX2]",
                             "label": "imine_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "imine_formation",
                    "name": "亚胺形成",
                    "label": "Imine",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 221. Baeyer-Villiger氧化（过氧酸氧化酮生成酯，羧酸为副产物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "baeyer_villiger_oxidation",
                "description": "过氧酸氧化酮或环酮生成酯或内酯，羧酸为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[#6]", "label": "ketone", "exclude_atoms_from": []},
                            {"pattern": "[OX2][OX2H1]", "label": "peracid", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX3](=[OX1])[OX2][CX4]", "label": "ester_lactone", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "baeyer_villiger_oxidation",
                    "name": "Baeyer-Villiger氧化",
                    "label": "BV-Oxid",
                    "byproducts": ["C(=O)(O)c1ccccc1Cl"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 222. Darzens缩合（α-卤代酯+醛/酮→缩水甘油酸酯，碱催化）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "darzens_condensation",
                "description": "α-卤代酯与醛/酮在碱作用下生成缩水甘油酸酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=O)", "label": "aldehyde", "exclude_atoms_from": []},
                            {"pattern": "[Cl,Br][CX3](=O)[OX2]", "label": "haloester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]1[OX2][#6]1",
                             "label": "epoxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "darzens_condensation",
                    "name": "Darzens缩合",
                    "label": "Darzens",
                    "byproducts": ["CC(C)(C)O", "[Cl-]"],
                    "coreactants": ["CC(C)(C)[O-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 223. 香豆素合成（酚+β-酮酯→香豆素，乙醇+水为副产物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "coumarin_synthesis",
                "description": "酚与β-酮酯缩合生成香豆素环，乙醇和水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1", "label": "phenol", "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)C[C](=O)[OX2][#6;!$(C=O)]", "label": "beta_keto_ester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6](=[OX1])[#8]",
                             "label": "lactone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "coumarin_synthesis",
                    "name": "香豆素合成",
                    "label": "Coumarin",
                    "byproducts": ["CCO", "O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 224. 炔丙基醚环化（生成苯并呋喃衍生物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "propargyl_ether_cyclization",
                "description": "炔丙基醚在吡啶环上发生环化，生成苯并呋喃衍生物，HCN为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C#C", "label": "alkyne", "exclude_atoms_from": []},
                            {"pattern": "c1ccncc1", "label": "pyridine", "exclude_atoms_from": ["alkyne"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1cc2ncccc2cc1",
                             "label": "fused_heterocycle",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "propargyl_ether_cyclization",
                    "name": "炔丙基醚环化",
                    "label": "Propargyl-Cyc",
                    "byproducts": ["C#N"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 225. 酯/内酯水解（酸或碱催化水解）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "lactone_hydrolysis",
                "description": "内酯/酯水解为羟基酸衍生物，水参与反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[OX2]", "label": "lactone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "lactone_hydrolysis",
                    "name": "内酯水解",
                    "label": "Lactone-Hydro",
                    "byproducts": [],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 226. 碳负离子烷基化（负碳离子+卤代烃→烷基化产物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "carbanion_alkylation",
                "description": "碳负离子与卤代烃发生亲核取代，生成烷基化产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[C-]", "label": "carbanion", "exclude_atoms_from": []},
                            {"pattern": "[Cl,Br,I][CX4]", "label": "alkyl_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][CX4]",
                             "label": "alkylated_carbon",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "carbanion_alkylation",
                    "name": "碳负离子烷基化",
                    "label": "C-Alkyl",
                    "byproducts": ["[Cl-]", "[Br-]", "[I-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 227. 碳负离子加成（负碳离子+羰基→醇盐）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "carbanion_addition",
                "description": "碳负离子对羰基的亲核加成，生成醇盐中间体",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[C-]", "label": "carbanion", "exclude_atoms_from": []},
                            {"pattern": "[CX3](=O)", "label": "carbonyl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H1]",
                             "label": "alcohol_from_addition",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "carbanion_addition",
                    "name": "碳负离子加成",
                    "label": "C-Add",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 228. 缩醛/缩酮去保护（酸催化脱除缩醛基团）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "acetal_deprotection",
                "description": "酸催化下缩醛/缩酮去保护，生成羰基化合物和醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][CX4][OX2]", "label": "acetal", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX4][OX2H1]", "label": "alcohol_from_addition", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "acetal_deprotection",
                    "name": "缩醛去保护",
                    "label": "Acetal-DeProt",
                    "byproducts": ["CC(C)=O"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 229. 烯醇硅醚Mukaiyama反应（硅烯醇醚+亲电体，氟离子催化）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "mukaiyama_aldol",
                "description": "硅烯醇醚与亲电体在氟离子催化下反应，生成加成产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C[OX2][Si]", "label": "silyl_enol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[#6][#6][OX2H1]",
                             "label": "beta_hydroxy_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "mukaiyama_aldol",
                    "name": "Mukaiyama反应",
                    "label": "Mukaiyama",
                    "byproducts": ["C[SiH](C)(C)C"],
                    "coreactants": ["[F-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 230. 芳基卤代烃Suzuki偶联（已存在，但扩展匹配更多卤代烃类型）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "suzuki_extended",
                "description": "芳基卤代烃与硼酸/硼酸酯的Suzuki偶联扩展版",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Cl,Br,I]c1ccccc1", "label": "aryl_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c-[c]",
                             "label": "biaryl_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_extended",
                    "name": "Suzuki偶联（扩展）",
                    "label": "Suzuki-Ext",
                    "byproducts": ["O", "[F-]", "[K+]", "[Na+]"],
                    "coreactants": ["OB(O)C=1C=CC=CC=1"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 231. 硅醚去保护（氟离子催化去硅保护基）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "silyl_deprotection",
                "description": "氟离子催化硅醚去保护，生成游离醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][Si]", "label": "silyl_ether", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "free_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "silyl_deprotection",
                    "name": "硅醚去保护",
                    "label": "Si-DeProt",
                    "byproducts": ["C[SiH](C)(C)C"],
                    "coreactants": ["[F-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 232. 胺与醛缩合（生成亚胺或Schiff碱）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "imine_schiff",
                "description": "伯胺或仲胺与醛/酮缩合生成亚胺，水为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H1,H2][#6;!$(C=O)]", "label": "amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)", "label": "carbonyl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX3]=[NX2]", "label": "imine_product", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "imine_schiff",
                    "name": "亚胺缩合",
                    "label": "Imine-Schiff",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 233. 碳负离子亲核取代（负碳离子+卤代烃→烷基化）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "sn2_carbanion",
                "description": "碳负离子对卤代烃的SN2取代反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[C-][#6]", "label": "carbanion", "exclude_atoms_from": []},
                            {"pattern": "[Cl,Br,I][CX4H2,CX4H3]", "label": "primary_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][CX4]",
                             "label": "alkylated_carbon",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sn2_carbanion",
                    "name": "SN2碳负离子取代",
                    "label": "SN2-C",
                    "byproducts": ["[Cl-]", "[Br-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 234. 环氧化反应（烯烃+过氧酸→环氧化物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "epoxidation",
                "description": "烯烃与过氧酸反应生成环氧化物，羧酸为副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CH0,CH1,CH2]=[CH0,CH1,CH2]", "label": "alkene", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]1[OX2][#6]1",
                             "label": "epoxide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "epoxidation",
                    "name": "环氧化反应",
                    "label": "Epoxidation",
                    "byproducts": ["C(=O)(O)c1ccccc1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 235. 叠氮还原为胺（ Curtius型反应，羧酸→胺）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "curtius_rearrangement",
                "description": "羧酸与叠氮钠反应，经Curtius重排生成胺，释放N2和CO2",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[OX2H1]", "label": "acid", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX2]=[CX2]=[OX1]",
                             "label": "isocyanate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "curtius_rearrangement",
                    "name": "Curtius重排",
                    "label": "Curtius",
                    "byproducts": ["[N-]=[N+]=[N-]", "O=C=O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "stille_coupling",
                "description": "有机锡试剂与芳基/乙烯基卤化物在Pd催化下偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Sn]", "label": "organotin", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[#6][Br,I]", "label": "aryl_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "coupled_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "stille_coupling",
                    "name": "Stille偶联",
                    "label": "Stille",
                    "byproducts": ["[Sn]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "sonogashira_coupling",
                "description": "端炔与芳基/乙烯基卤化物在Pd/Cu催化下偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2H][#6]", "label": "terminal_alkyne", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[#6][Br,I]", "label": "aryl_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX2]#[CX2]c",
                             "label": "internal_alkyne",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sonogashira_coupling",
                    "name": "Sonogashira偶联",
                    "label": "Sonogashira",
                    "byproducts": ["[I-]", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "diels_alder",
                "description": "共轭双烯与亲双烯体发生[4+2]环加成",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CC=C", "label": "diene", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "C=C", "label": "dienophile", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C1=CCCCC1",
                             "label": "cyclohexene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "diels_alder",
                    "name": "Diels-Alder环加成",
                    "label": "DA",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "esterification_fisher",
                "description": "羧酸与醇在酸催化下酯化，脱水",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[OX2H1]", "label": "acid", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2][CX4]",
                             "label": "ester_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "esterification_fisher",
                    "name": "Fisher酯化",
                    "label": "Fisher",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "snar_aryl",
                "description": "亲核芳香取代，芳基卤素被胺/醇/酚等亲核试剂取代",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6X3;R][Cl,F,Br,I]", "label": "aryl_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3]", "label": "nucleophile_n", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]c",
                             "label": "n_aryl_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "snar_aryl",
                    "name": "亲核芳香取代",
                    "label": "SNAr",
                    "byproducts": ["[{X}-]", "[H+]"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "claisen_general",
                "description": "烯丙基乙烯基醚发生[3,3]重排生成γ,δ-不饱和羰基",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCOC=C", "label": "allyl_vinyl_ether", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4][CX4][CX3]=[CX3]",
                             "label": "unsaturated_carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "claisen_general",
                    "name": "Claisen重排(通用)",
                    "label": "Claisen",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "alkene_oxidative_cleavage_ozonolysis",
                "description": "烯烃臭氧氧化裂解生成醛/酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C", "label": "alkene", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_fragment",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "alkene_oxidative_cleavage_ozonolysis",
                    "name": "烯烃臭氧氧化裂解",
                    "label": "O3",
                    "byproducts": ["O"],
                    "coreactants": ["[O][O][O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "alcohol_oxidation_pcc",
                "description": "醇被PCC氧化为醛/酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "alcohol_oxidation_pcc",
                    "name": "醇PCC氧化",
                    "label": "PCC",
                    "byproducts": ["[H][O]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "ketone_reduction_nabh4",
                "description": "酮被NaBH4还原为醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[#6]", "label": "ketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4;H1][OX2H1]",
                             "label": "secondary_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ketone_reduction_nabh4",
                    "name": "酮NaBH4还原",
                    "label": "NaBH4",
                    "byproducts": ["[BH4-]"],
                    "coreactants": ["[H+]", "[H-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "nitrile_hydrolysis",
                "description": "腈水解为羧酸",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C#N", "label": "nitrile", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "nitrile_hydrolysis",
                    "name": "腈水解",
                    "label": "CN-H2O",
                    "byproducts": ["[NH4+]"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "amide_hydrolysis",
                "description": "酰胺酸性水解为羧酸和铵盐",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[NX3]", "label": "amide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "amide_hydrolysis",
                    "name": "酰胺水解",
                    "label": "Amide-H2O",
                    "byproducts": ["[NH4+]"],
                    "coreactants": ["O", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "lactonization",
                "description": "羟基酸分子内酯化闭环",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[OX2H1]", "label": "acid", "exclude_atoms_from": []},
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": ["acid"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX3](=[OX1])[OX2][CX4]", "label": "lactone", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "lactonization",
                    "name": "内酯化",
                    "label": "Lactone",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "lactamization",
                "description": "氨基酸分子内酰胺化闭环",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)[OX2H1]", "label": "acid", "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2,H1]", "label": "amine", "exclude_atoms_from": ["acid"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX3](=[OX1])[NX3]", "label": "lactam", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "lactamization",
                    "name": "内酰胺化",
                    "label": "Lactam",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "decarboxylation_beta_keto",
                "description": "β-酮酸脱羧生成酮+CO2",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C(=O)CC(=O)[OX2H1]", "label": "beta_keto_acid", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX3](=[OX1])[CX4]", "label": "ketone_product", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "decarboxylation_beta_keto",
                    "name": "β-酮酸脱羧",
                    "label": "beta-DC",
                    "byproducts": ["O=C=O"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "aldol_condensation",
                "description": "醛/酮发生Aldol缩合，脱水",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3;H1](=O)[#6]", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[#6]", "label": "ketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX3]=[CX3]",
                             "label": "enone_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aldol_condensation",
                    "name": "Aldol缩合",
                    "label": "Aldol",
                    "byproducts": ["O"],
                    "coreactants": ["[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "michael_addition",
                "description": "亲核试剂对α,β-不饱和羰基的Michael加成",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CC(=O)[#6]", "label": "enone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "nucleophile", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4][CX4][CX3](=[OX1])",
                             "label": "michael_adduct",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "michael_addition",
                    "name": "Michael加成",
                    "label": "Michael",
                    "byproducts": ["[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "imine_general",
                "description": "醛/酮与胺缩合生成亚胺，脱水",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[#6]", "label": "carbonyl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]", "label": "primary_amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]=[NX2]",
                             "label": "imine_bond",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "imine_general",
                    "name": "亚胺形成(通用)",
                    "label": "Imine",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "acetal_formation",
                "description": "醛/酮与醇缩合生成缩醛/缩酮，脱水",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[#6]", "label": "carbonyl", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]([OX2])([OX2])",
                             "label": "acetal_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "acetal_formation",
                    "name": "缩醛/缩酮形成",
                    "label": "Acetal",
                    "byproducts": ["O", "O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "epoxide_opening",
                "description": "环氧化物被亲核试剂开环",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C1OC1", "label": "epoxide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "nucleophile", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4][CX4][OX2H]",
                             "label": "amino_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "epoxide_opening",
                    "name": "环氧化物开环",
                    "label": "Epoxide",
                    "byproducts": ["[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "suzuki_miyaura_aryl_boron",
                "description": "芳基硼酸与芳基卤化物的Suzuki-Miyaura偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[BX3](O)O", "label": "boronic_acid", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[#6X3;R][Br,I]", "label": "aryl_halide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_miyaura_aryl_boron",
                    "name": "Suzuki-Miyaura偶联",
                    "label": "Suzuki",
                    "byproducts": ["[OH-]", "[OH-]", "[H+]"],
                    "coreactants": ["[OH-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "ibx_dmp_alcohol_oxidation",
                "description": "醇被IBX或DMP氧化为醛/酮，氧化剂被还原为亚碘酰苯甲酸",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=I", "label": "hypervalent_iodine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ibx_dmp_alcohol_oxidation",
                    "name": "IBX/DMP醇氧化",
                    "label": "IBX",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "ozonolysis_to_aldehyde",
                "description": "烯烃被臭氧氧化裂解为醛",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C", "label": "alkene", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=[OX1])",
                             "label": "aldehyde_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ozonolysis_to_aldehyde",
                    "name": "烯烃臭氧裂解成醛",
                    "label": "O3-Ald",
                    "byproducts": ["C=O"],
                    "coreactants": ["[O][O][O]", "[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "tetraphenylcyclopentadienone_decaboxylation",
                "description": "四苯基环戊二烯酮与炔/腈/叠氮[4+2]环加成后脱羰",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=C1C(=C(C(=C1[#6])[#6])[#6])[#6]", "label": "tetraphenyl_cpdk", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "C=C",
                             "label": "diene_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "tetraphenylcyclopentadienone_decaboxylation",
                    "name": "四苯基环戊二烯酮脱羰",
                    "label": "Cp-CO",
                    "byproducts": ["[C-]#[O+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "carbamate_from_isocyanate",
                "description": "醇与异氰酸酯加成生成氨基甲酸酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "N=C=O", "label": "isocyanate", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][CX3](=[OX1])[NX3]",
                             "label": "carbamate_linkage",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "carbamate_from_isocyanate",
                    "name": "氨基甲酸酯形成(异氰酸酯)",
                    "label": "Carb-Iso",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "cdi_amide_coupling",
                "description": "羰基二咪唑(CDI)与胺反应形成酰胺/脲",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "O=C(n1ccnc1)n1ccnc1", "label": "cdi", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "cdi_amide_coupling",
                    "name": "CDI酰胺偶联",
                    "label": "CDI",
                    "byproducts": ["c1c[nH]cc1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "desulfurization_reductive",
                "description": "硫醚/硫代羰基还原脱硫",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[SX2]", "label": "sulfur", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "desulfurization_reductive",
                    "name": "还原脱硫",
                    "label": "Desulf",
                    "byproducts": ["O=S=O"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "mesylate_tosylate_substitution",
                "description": "甲磺酸酯/对甲苯磺酸酯被亲核试剂取代",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)O[#6]", "label": "sulfonate_ester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "nucleophile_n", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H1,H0][CX4]",
                             "label": "alkylated_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "mesylate_tosylate_substitution",
                    "name": "磺酸酯取代",
                    "label": "Ms/Ts",
                    "byproducts": ["[O-]S(=O)(=O)[C]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            # [已删除] boc2o_carbamate_formation与protection_boc重复 ----
            {
                "reaction_type": "swern_oxidation_general",
                "description": "Swern氧化:醇+DMSO+草酰氯→醛/酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2H]", "label": "alcohol", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "CS(=O)C", "label": "dmso", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])",
                             "label": "carbonyl_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "swern_oxidation_general",
                    "name": "Swern氧化(通用)",
                    "label": "Swern",
                    "byproducts": ["CS(=O)C", "[H][H]", "[H][H]"],
                    "coreactants": ["ClC(=O)C(=O)Cl"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "azide_pyrazole_cyclization",
                "description": "叠氮基吡唑分子内环化生成吡咯并吡唑，释放N2",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N]=[N+]=[N-]", "label": "azide", "exclude_atoms_from": []},
                            {"pattern": "n1c[nH]c1", "label": "pyrazole", "exclude_atoms_from": ["azide"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#7;R]~[#7;R]~[#7;R]",
                             "label": "triazole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "azide_pyrazole_cyclization",
                    "name": "叠氮吡唑环化",
                    "label": "Az-Pyr",
                    "byproducts": ["[N-]=[N+]=[N-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "claisen_ester_rearrangement",
                "description": "烯丙基酯的Claisen重排生成γ,δ-不饱和羧酸衍生物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CCOC(=O)[#6]", "label": "allyl_ester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4][CX4][CX3]=[CX3]",
                             "label": "unsaturated_carbonyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "claisen_ester_rearrangement",
                    "name": "烯丙基酯Claisen重排",
                    "label": "Claisen-ester",
                    "byproducts": ["[H]O[H]"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "pyrazole_condensation_multicomponent",
                "description": "α-卤代酮+醛+二胺(肼)多组分缩合生成吡唑并稠杂环",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]C(=O)C[Cl,Br]", "label": "haloketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[#6]C=O", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#7]1[#7][#6][#6][#6]1",
                             "label": "pyrazole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "pyrazole_condensation_multicomponent",
                    "name": "吡唑多组分缩合",
                    "label": "Pyr-MC",
                    "byproducts": ["[H]O[H]", "[H][Cl]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "orthoester_condensation",
                "description": "原甲酸三乙酯与胺/醇缩合，释放乙醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCOC(OCC)OCC", "label": "orthoester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[NX2])[OX2]",
                             "label": "imidate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "orthoester_condensation",
                    "name": "原甲酸酯缩合",
                    "label": "Ortho",
                    "byproducts": ["CCO", "CCO", "CCO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "wittig_stabilized_ylide",
                "description": "稳定Wittig试剂(α-酯基磷叶立德)与醛反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=[P+]", "label": "phosphonium_ylide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3;H1](=O)", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3]=[CX3][CX3](=[OX1])[OX2]",
                             "label": "unsaturated_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "wittig_stabilized_ylide",
                    "name": "稳定Wittig烯烃化",
                    "label": "Wittig-stab",
                    "byproducts": ["O=P(c1ccccc1)(c1ccccc1)c1ccccc1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "knoevenagel_active_methylene",
                "description": "活泼亚甲基化合物与醛/酮Knoevenagel缩合",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]C(=O)CC(=O)[#6]", "label": "active_methylene", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3;H1](=O)", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX3]=[CX3]",
                             "label": "knoevenagel_product",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "knoevenagel_active_methylene",
                    "name": "Knoevenagel缩合",
                    "label": "Knoev",
                    "byproducts": ["[H]O[H]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "reductive_alkylation_amine",
                "description": "胺与醛还原烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3;H1](=O)", "label": "aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H0,H1,H2][CX4]",
                             "label": "alkylated_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "reductive_alkylation_amine",
                    "name": "还原烷基化",
                    "label": "Red-Alk",
                    "byproducts": ["[H]O[H]"],
                    "coreactants": ["[H][H]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "urea_formation_carbonyldi",
                "description": "环己基异氰酸酯(或双异氰酸酯)与胺反应生成脲",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "N=C=O", "label": "isocyanate", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]", "label": "primary_amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX3](=[OX1])[NX3]",
                             "label": "urea",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "urea_formation_carbonyldi",
                    "name": "脲形成(异氰酸酯)",
                    "label": "Urea-Iso",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "luche_reduction",
                "description": "Luche还原:烯酮被NaBH4+CeCl3选择性还原为烯丙醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "C=CC(=O)[#6]", "label": "enone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4;H1][OX2H1]",
                             "label": "allylic_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "luche_reduction",
                    "name": "Luche还原",
                    "label": "Luche",
                    "byproducts": [],
                    "coreactants": ["[H-]", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "grignard_addition_to_ketone",
                "description": "格氏试剂对酮的加成生成叔醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Mg]", "label": "grignard", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[#6]", "label": "ketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "[CX4][OX2H1]", "label": "alcohol_or_amine", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "grignard_addition_to_ketone",
                    "name": "格氏试剂加成到酮",
                    "label": "RMgX-K",
                    "byproducts": ["[Mg][OH]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            {
                "reaction_type": "suzuki_variant_boronic_ester",
                "description": "Suzuki偶联变体:频哪醇硼酸酯+芳基三氟甲磺酸酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "B1OC(C)(C)C(C)(C)O1", "label": "pinacol_boronate", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c[OX2]S(=O)(=O)C(F)(F)F", "label": "aryl_triflate", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "suzuki_variant_boronic_ester",
                    "name": "Suzuki偶联(硼酸酯+三氟甲磺酸酯)",
                    "label": "Suz-Bor-OTf",
                    "byproducts": ["CC1(C)OB(OS(=O)(=O)C(F)(F)F)OC1(C)C"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "sulfonyl_chloride_hydrolysis",
                "description": "芳基磺酰氯水解生成芳基磺酸和HCl",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "cS(=O)(=O)Cl", "label": "aryl_sulfonyl_chloride", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[SX4](=[OX1])(=[OX1])[OX2H]",
                             "label": "sulfonic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sulfonyl_chloride_hydrolysis",
                    "name": "磺酰氯水解",
                    "label": "SO2Cl-H2O",
                    "byproducts": ["[Cl-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "benzoin_condensation",
                "description": "两分子芳香醛在氰化物催化下发生安息香缩合，生成α-羟基芳基酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3;H1](=O)c", "label": "aromatic_aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3;H1](=O)c", "label": "aromatic_aldehyde", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[CX4]([OX2H])[c]",
                             "label": "alpha_hydroxy_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "benzoin_condensation",
                    "name": "安息香缩合",
                    "label": "Benzoin",
                    "byproducts": [],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            {
                "reaction_type": "suzuki_pinacol_triflate",
                "description": "Suzuki偶联:芳基频哪醇硼酸酯与芳基三氟甲磺酸酯偶联，生成联芳基和频哪醇硼酸酯-三氟甲磺酸酯副产物",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "B1OC(C)(C)C(C)(C)O1", "label": "pinacol_boronate", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "S(=O)(=O)OC(F)(F)F", "label": "triflate", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "B[OX2]S(=O)(=O)",
                             "label": "boron_triflate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "suzuki_pinacol_triflate",
                    "name": "Suzuki偶联(频哪醇+三氟甲磺酸酯)",
                    "label": "Suz-pinB-OTf",
                    "byproducts": ["CC1(C)OB(OS(=O)(=O)C(F)(F)F)OC1(C)C"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "aromatic_formylation",
                "description": "芳香环甲酰化:在芳环上引入甲酰基(CHO)，以甲酸为甲酰化试剂",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cH]", "label": "aromatic_CH", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3H1](=[OX1])c",
                             "label": "aromatic_aldehyde",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_formylation",
                    "name": "芳香环甲酰化",
                    "label": "Ar-CHO",
                    "byproducts": ["O"],
                    "coreactants": ["OC=O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "n_acetylation",
                "description": "N-乙酰化:胺与乙酸反应生成N-乙酰基胺和水",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]", "label": "amine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "acetamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "n_acetylation",
                    "name": "N-乙酰化",
                    "label": "N-Ac",
                    "byproducts": ["O"],
                    "coreactants": ["CC(=O)O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "c_acetylation",
                "description": "C-乙酰化:芳环上Friedel-Crafts型乙酰化，以乙酸为乙酰化试剂",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[cH]", "label": "aromatic_CH", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])([CX4H3])c",
                             "label": "methyl_aryl_ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "c_acetylation",
                    "name": "C-乙酰化",
                    "label": "C-Ac",
                    "byproducts": ["O"],
                    "coreactants": ["CC(=O)O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "chloromethyl_etherification",
                "description": "N-氯甲基醚化:胺上氯甲基与甲醇反应生成甲氧甲基，释放HCl",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4][Cl]", "label": "chloromethylamine", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3][CX4][OX2]",
                             "label": "amino_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "chloromethyl_etherification",
                    "name": "氯甲基醚化",
                    "label": "ClMe→OMe",
                    "byproducts": ["Cl"],
                    "coreactants": ["CO"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "gewald_thiophene",
                "description": "Gewald噻吩合成:二烷基酮+氰基乙酸酯+硫化物→氨基噻吩",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)([CX4])[CX4]", "label": "dialkyl_ketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CH2][CX2]#[NX1]", "label": "active_methylene_nitrile", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[S-2]", "label": "sulfide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[#6]1[#6][#6][#16][#6]1",
                             "label": "thiophene_ring",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "gewald_thiophene",
                    "name": "Gewald噻吩合成",
                    "label": "Gewald",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "paal_knorr_thiophene",
                "description": "Paal-Knorr噻吩合成:1,4-二酮与硫化物反应生成噻吩和水",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)CC[CX3](=O)", "label": "1,4-diketone", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[S-]", "label": "sulfide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccsc1",
                             "label": "thiophene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "paal_knorr_thiophene",
                    "name": "Paal-Knorr噻吩合成",
                    "label": "Paal-Knorr",
                    "byproducts": [],
                    "coreactants": ["[H+]", "[H+]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            {
                "reaction_type": "benzothiopyrone_synthesis",
                "description": "苯并硫代吡喃酮合成:氟碘芳烃+CO+硫化物+端炔→苯并硫代吡喃酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "Fc", "label": "aryl_fluoride", "exclude_atoms_from": []},
                            {"pattern": "Ic", "label": "aryl_iodide", "exclude_atoms_from": ["aryl_fluoride"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                    {
                        "features": [
                            {"pattern": "[C-]#[O+]", "label": "carbon_monoxide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[S-]", "label": "sulfide", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX2]#[CH1]", "label": "terminal_alkyne", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [],
                "template": {
                    "id": "benzothiopyrone_synthesis",
                    "name": "苯并硫代吡喃酮合成",
                    "label": "BzThPyr",
                    "byproducts": ["[F-]", "[I-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 236. 芳基甲基醚 O-脱甲基化（HI介导脱保护）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aryl_odemethylation",
                "description": "芳基甲基醚HI介导脱甲基:Ar-OCH3+HI→Ar-OH+CH3I",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c[OX2][CH3]", "label": "aryl_methyl_ether", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [{"features": [{"pattern": "c[OX2H1]", "label": "phenol", "exclude_atoms_from": []}], "scope": "separate_molecule", "base_score": 1.0, "multi_feature_bonus": 0}],
                "template": {
                    "id": "aryl_odemethylation",
                    "name": "芳基O-脱甲基化",
                    "label": "Ar-ODem",
                    "byproducts": ["CI"],
                    "coreactants": ["[H+]", "[I-]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---------------------------------------------------------------------------
            # 237. 原酸酯+两分子伯胺→脒/亚胺（释放3当量乙醇和氨）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "orthoester_amidine",
                "description": "原酸酯与两分子伯胺缩合生成脒/亚胺，释放乙醇和氨",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCOC(C)(OCC)OCC", "label": "orthoester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]", "label": "amine1", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]", "label": "amine2", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[NX2])[NX3]",
                             "label": "amidine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    }
                ],
                "template": {
                    "id": "orthoester_amidine",
                    "name": "原酸酯-双胺脒化",
                    "label": "Ortho-Am",
                    "byproducts": ["CCO", "CCO", "CCO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 238. 原酸酯+多元醇→螺环/环状缩醛（释放3当量乙醇）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "orthoester_polyol_acetal",
                "description": "原酸酯与多元醇(≥2个OH)发生缩醛交换生成螺环或环状缩醛，释放乙醇",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "CCOC(C)(OCC)OCC", "label": "orthoester", "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1]", "label": "oh1", "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]", "label": "oh2", "exclude_atoms_from": ["oh1"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4]1[OX2][CX4][OX2][CX4][CX4]1",
                             "label": "cyclic_acetal",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "orthoester_polyol_acetal",
                    "name": "原酸酯-多元醇缩醛交换",
                    "label": "Ortho-Pol",
                    "byproducts": ["CCO", "CCO", "CCO"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---------------------------------------------------------------------------
            # 286. N-甲酰化（1-甲酰基苯并三唑为甲酰化试剂）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "n_formylation",
                "description": "仲胺与1-甲酰基苯并三唑反应生成N-甲酰胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H1;!$(N=*)]",
                             "label": "secondary_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "O=Cn1nnc2ccccc12",
                             "label": "formyl_benzotriazole",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3]C=O",
                             "label": "formamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "n_formylation",
                    "name": "N-甲酰化",
                    "label": "N-Formyl",
                    "byproducts": ["c1ccc2[nH]ncc2c1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 287. 喹唑啉二酮二氯化（POCl3介导）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "quinazolinedione_dichlorination",
                "description": "喹唑啉-2,4-二酮与POCl3反应生成2,4-二氯喹唑啉",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[nH]1c2ccccc2c(=O)[nH]c1=O",
                             "label": "quinazolinedione",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccc2c(c1)c(Cl)nc(Cl)n2",
                             "label": "dichloroquinazoline",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "quinazolinedione_dichlorination",
                    "name": "喹唑啉二酮二氯化",
                    "label": "Quin-Cl2",
                    "byproducts": ["OP(=O)(O)O", "Cl", "Cl"],
                    "coreactants": ["P(=O)(Cl)(Cl)Cl"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 288. 芳香羧基化（CO2引入羧基）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_carboxylation",
                "description": "芳烃经金属化后与CO2反应引入羧基",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1",
                             "label": "arene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1C(=O)O",
                             "label": "aryl_carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_carboxylation",
                    "name": "芳香羧基化",
                    "label": "Ar-CO2H",
                    "byproducts": ["[H+]"],
                    "coreactants": ["O=C=O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 289. 格氏试剂对腈的加成（生成酮）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "grignard_addition_to_nitrile",
                "description": "格氏试剂或有机锂试剂对腈加成，水解后生成酮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[C]#[N]",
                             "label": "nitrile",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[CX4]",
                             "label": "ketone",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "grignard_addition_to_nitrile",
                    "name": "格氏试剂对腈加成",
                    "label": "Grignard-CN",
                    "byproducts": ["N", "O[Mg]Br"],
                    "coreactants": ["O"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 290. 脱羧交叉偶联（杂芳羧酸+芳基卤化物）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "decarboxylative_cross_coupling_general",
                "description": "杂芳羧酸脱羧后与芳基卤化物发生交叉偶联",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[c]C(=O)O",
                             "label": "heteroaryl_carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "c[X;Cl,Br,I]",
                             "label": "aryl_halide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cc",
                             "label": "biaryl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "decarboxylative_cross_coupling_general",
                    "name": "脱羧交叉偶联",
                    "label": "Decarb-XC",
                    "byproducts": ["O=C=O", "[H+].[{X}-]"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 291. N-烷基化（伯烷基卤化物，通用）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "nh_alkylation_alkyl_halide",
                "description": "含NH的杂环或胺与伯烷基卤化物发生N-烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[nH]",
                             "label": "nh_group",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H2,H1]",
                             "label": "nh_group_aliphatic",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CH2][X;Cl,Br,I]",
                             "label": "alkyl_halide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[n;!H]",
                             "label": "n_alkyl_aromatic",
                             "exclude_atoms_from": []},
                            {"pattern": "[NX3;H1,H0;!$(N=*)]",
                             "label": "n_alkyl_aliphatic",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "nh_alkylation_alkyl_halide",
                    "name": "N-烷基化(卤代烷)",
                    "label": "N-Alkyl",
                    "byproducts": ["[H+].[{X}-]"],
                    "coreactants": [],
                    "halogen_dependent": True,
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---------------------------------------------------------------------------
            # 292. 羟胺O-烷基化（N-羟基化合物+醇）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "hydroxylamine_o_alkylation",
                "description": "N-羟基化合物(如N-羟基邻苯二甲酰亚胺)与醇发生O-烷基化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[N]([OX2H1])[C](=O)",
                             "label": "n_hydroxy_imide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1][CH2]",
                             "label": "primary_alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[N]([OX2])[C](=O)",
                             "label": "n_alkoxy_imide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "hydroxylamine_o_alkylation",
                    "name": "羟胺O-烷基化",
                    "label": "NOH-Alk",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 293. 酰基叠氮形成（DPPA介导的Curtius重排第一步）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "acyl_azide_formation_dppa",
                "description": "羧酸与DPPA(叠氮磷酸二苯酯)反应生成酰基叠氮",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=O)[OX2H1]",
                             "label": "carboxylic_acid",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[N-]=[N+]=[N]P(=O)",
                             "label": "dppa",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX1]~[NX2]~[NX2]C(=O)",
                             "label": "acyl_azide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "acyl_azide_formation_dppa",
                    "name": "酰基叠氮形成(DPPA)",
                    "label": "DPPA-N3",
                    "byproducts": ["OP(=O)(O)c1ccccc1c1ccccc1"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 294. 磺酰胺形成（磺酰氯+胺）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "sulfonamide_formation",
                "description": "磺酰氯与胺反应生成磺酰胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[S](=O)(=O)[Cl]",
                             "label": "sulfonyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2,H1]",
                             "label": "amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[S](=O)(=O)[NX3]",
                             "label": "sulfonamide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sulfonamide_formation",
                    "name": "磺酰胺形成",
                    "label": "SO2N-Form",
                    "byproducts": ["[H+].[Cl-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 1,
            },
            # ---------------------------------------------------------------------------
            # 295. 芳香碘化（ICl为碘化试剂）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "aromatic_iodination_icl",
                "description": "富电子芳烃与ICl发生亲电芳香碘化反应",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1ccccc1",
                             "label": "arene",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "ICl",
                             "label": "icl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "cI",
                             "label": "aryl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "aromatic_iodination_icl",
                    "name": "芳香碘化(ICl)",
                    "label": "Ar-I",
                    "byproducts": ["[H+].[Cl-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 296. 醇与四氯化碳直接氯化（无PPh3体系）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "ccl4_direct_chlorination",
                "description": "醇与四氯化碳直接反应生成烷基氯化物、氯仿和水（无PPh3参与的直接氯化）",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][OX2H1]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "ClC(Cl)(Cl)Cl",
                             "label": "ccl4",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX4][Cl]",
                             "label": "alkyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ccl4_direct_chlorination",
                    "name": "CCl4直接氯化（无PPh3）",
                    "label": "CCl4-Cl-direct",
                    "byproducts": ["ClC(Cl)Cl", "[O]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 297. 吲哚氧化为吲哚酮（含异氰基底物的吲哚C2位氧化）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "indole_oxidation_oxindole",
                "description": "吲哚衍生物氧化为吲哚酮（oxindole），常见于含异氰基吲哚生物碱的氧化转化",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c1c[nH]c2ccccc12",
                             "label": "indole_nh",
                             "exclude_atoms_from": []},
                            {"pattern": "[N+]#[C-]",
                             "label": "isocyanide",
                             "exclude_atoms_from": ["indole_nh"]}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;R][CX3](=[OX1])[CX4;R]",
                             "label": "cyclic_amide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "indole_oxidation_oxindole",
                    "name": "吲哚氧化为吲哚酮",
                    "label": "Ind-Ox",
                    "byproducts": [],
                    "coreactants": ["[O]"],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },            # ---------------------------------------------------------------------------
            # 298. 氯甲酸酯与伯胺生成氨基甲酸酯（carbamate formation）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "chloroformate_carbamate_formation",
                "description": "氯甲酸酯与伯胺反应生成氨基甲酸酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[Cl][CX3](=[OX1])[OX2]",
                             "label": "chloroformate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[NX3;H2]",
                             "label": "primary_amine",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]",
                             "label": "carbamate",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "chloroformate_carbamate_formation",
                    "name": "氯甲酸酯氨基甲酸酯化",
                    "label": "Carb-F",
                    "byproducts": ["[Cl-]", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 3,
            },
            # ---------------------------------------------------------------------------
            # 299. 脱水醚化（酚 + 醇 → 芳基醚）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "ether_formation_dehydrative",
                "description": "酚与醇脱水生成芳基醚",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c",
                             "label": "ring",
                             "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]",
                             "label": "hydroxyl",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                    {
                        "features": [
                            {"pattern": "[CX4]",
                             "label": "alkyl",
                             "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[OX2][CX4]",
                             "label": "aryl_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "ether_formation_dehydrative",
                    "name": "脱水醚化",
                    "label": "Ether-DH",
                    "byproducts": ["O"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---------------------------------------------------------------------------
            # 300. O-烷基化（酚 + 碘代烷 → 芳基醚）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "o_alkylation_alkyl_iodide",
                "description": "酚与碘代烷O-烷基化生成芳基醚",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "c",
                             "label": "ring",
                             "exclude_atoms_from": []},
                            {"pattern": "[OX2H1]",
                             "label": "phenol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "same_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0.5,
                    },
                    {
                        "features": [
                            {"pattern": "[IX1][CX4]",
                             "label": "alkyl_iodide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "c[OX2][CX4]",
                             "label": "aryl_ether",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "o_alkylation_alkyl_iodide",
                    "name": "O-烷基化（碘代烷）",
                    "label": "O-Alk-I",
                    "byproducts": ["[H+]", "[I-]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },
            # ---------------------------------------------------------------------------
            # 301. 氨基磺酰氯与醇生成磺酸酯（sulfamate ester formation）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "sulfamate_ester_formation",
                "description": "氨基磺酰氯与醇反应生成磺酸酯",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "Cl[SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfamoyl_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[OX2H1]",
                             "label": "alcohol",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[OX2][SX4](=[OX1])(=[OX1])[NX3]",
                             "label": "sulfamate_ester",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "sulfamate_ester_formation",
                    "name": "氨基磺酸酯化",
                    "label": "Sulf-E",
                    "byproducts": ["[Cl-]", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 4,
            },
            # ---------------------------------------------------------------------------
            # 302. Schotten-Baumann酰胺化（苯胺 + 酰氯 → 酰胺）
            # ---------------------------------------------------------------------------
            {
                "reaction_type": "schotten_baumann_amide",
                "description": "苯胺与酰氯Schotten-Baumann反应生成酰胺",
                "reactant_requirements": [
                    {
                        "features": [
                            {"pattern": "[NX3;H2]c",
                             "label": "aniline",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[Cl]",
                             "label": "acid_chloride",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "product_requirements": [
                    {
                        "features": [
                            {"pattern": "[CX3](=[OX1])[NX3]c",
                             "label": "anilide",
                             "exclude_atoms_from": []}
                        ],
                        "scope": "separate_molecule",
                        "base_score": 1.0,
                        "multi_feature_bonus": 0,
                    },
                ],
                "template": {
                    "id": "schotten_baumann_amide",
                    "name": "Schotten-Baumann酰胺化",
                    "label": "SB-Amide",
                    "byproducts": ["[Cl-]", "[H+]"],
                    "coreactants": [],
                    "reaction_smarts": None,
                },
                "commonness_rank": 2,
            },

        ]

    # ----------------------------------------------------------
    #  特征匹配（支持 exclude_atoms_from）
    # ----------------------------------------------------------

    def _match_features_on_single_molecule(
        self, mol, features: list
    ) -> Tuple[bool, List[Set[int]]]:
        """
        检查一个分子是否满足一个特征要求中的所有特征。

        按声明顺序依次匹配每个特征，处理 exclude_atoms_from 的
        原子级排除（先匹配的特征先占用原子，后匹配的特征排除已
        占用的原子）。

        返回:
            (all_matched, matched_atom_sets)
            all_matched: 所有特征是否都匹配成功
            matched_atom_sets: 每个特征匹配到的原子索引列表
        """
        occupied: Dict[str, Set[int]] = {}
        matched_sets: List[Set[int]] = []

        for feat in features:
            pattern = feat["pattern"]
            label = feat.get("label", "")
            exclude_refs = feat.get("exclude_atoms_from", [])

            # 收集被引用特征已占用的原子
            exclude_atoms: Set[int] = set()
            for ref_label in exclude_refs:
                exclude_atoms |= occupied.get(ref_label, set())

            # 执行 SMARTS 匹配（排除已占用原子）
            matched = _match_feature_on_mol(mol, pattern, exclude_atoms)
            if not matched:
                return False, []

            occupied[label] = matched
            matched_sets.append(matched)

        return True, matched_sets

    def _match_single_feature(
        self, mol, feature: dict
    ) -> bool:
        """
        检查一个分子是否匹配单个 SMARTS 特征。

        用于 separate_molecule 类型（每个特征在不同分子上，
        无需原子级排除）。
        """
        pattern = feature["pattern"]
        pat_mol = Chem.MolFromSmarts(pattern)
        if pat_mol is None:
            return False
        return bool(mol.GetSubstructMatch(pat_mol))

    # ----------------------------------------------------------
    #  回溯遍历分配算法
    # ----------------------------------------------------------

    def _traverse_requirements(
        self,
        requirements: list,
        molecules: List[str],
        allow_skip: bool = False,
    ) -> Dict[str, Any]:
        """
        回溯遍历：将输入分子分配给特征要求。

        分子按大小（原子数）降序排列。依次为每个特征要求尝试
        分配未被使用的分子。分配策略必须是一对一的（每个分子
        只能服务于一个特征要求）。

        参数:
            requirements: 特征要求列表
            molecules: 输入分子 SMILES 列表
            allow_skip: 是否允许跳过要求。
                False = 全满足模式：所有要求都必须满足。
                True = 子集模式：允许跳过部分要求，
                       寻找评分最高的部分分配方案。

        返回字典:
            full_strategies: 全满足的分配策略列表（allow_skip=False 时使用）
            best_score: 最高评分（allow_skip=True 时使用）
            best_allocation: 最高评分对应的分配方案
        """
        if not requirements:
            return {"full_strategies": [{}], "best_score": 0,
                    "best_allocation": {}}
        if not molecules:
            return {"full_strategies": [], "best_score": 0,
                    "best_allocation": {}}

        # 分子预处理：按原子数降序排列
        mol_with_size = []
        for idx, smi in enumerate(molecules):
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                n_atoms = m.GetNumAtoms()
                mol_with_size.append((idx, smi, m, n_atoms))
        mol_with_size.sort(key=lambda x: -x[3])

        result = {
            "full_strategies": [],
            "full_index_strategies": [],  # C-TM1: 索引版全满足策略
            "best_score": 0,
            "best_allocation": {},
            "best_index_alloc": {},       # C-TM1: 索引版最佳分配
        }

        def _recurse(req_idx, used_indices, allocation, score,
                     alloc_indices):
            """
            递归回溯。

            req_idx: 当前处理的特征要求索引
            used_indices: 已被使用的分子索引集合
            allocation: {req_index: [(idx, smi)] 或 [smi]}
            score: 当前累计评分
            alloc_indices: {req_index: set(indices)} — C-3 精确释放用
            """
            # 所有要求都处理完毕 → 记录结果
            if req_idx == len(requirements):
                # C-3: 转换为纯 SMILES 列表以保持下游兼容
                smiles_alloc = {
                    k: ([s for _, s in v] if v and isinstance(v[0], tuple)
                        else list(v))
                    for k, v in allocation.items()
                }
                # C-TM1 修复：同时保留索引版分配方案，
                # 用于精确追踪重复分子（相同 SMILES 不同位置）的使用情况
                index_alloc = {}
                for k, v in allocation.items():
                    if v and isinstance(v[0], tuple):
                        index_alloc[k] = [idx_i for idx_i, _ in v]
                    else:
                        index_alloc[k] = list(range(len(v)))
                # 仅当所有要求都被分配时才记为全满足策略
                if len(smiles_alloc) == len(requirements):
                    result["full_strategies"].append(smiles_alloc)
                    result["full_index_strategies"].append(index_alloc)
                # 更新最高评分（同分取先发现的，由分子大小降序保证确定性）
                if score > result["best_score"]:
                    result["best_score"] = score
                    result["best_allocation"] = smiles_alloc
                    result["best_index_alloc"] = index_alloc
                return

            req = requirements[req_idx]
            features = req["features"]
            scope = req.get("scope", "separate_molecule")
            req_score = (req.get("base_score", 1.0)
                         + req.get("multi_feature_bonus", 0))

            # ===== 尝试为当前要求分配分子 =====
            tried_mols = set()  # 避免对相同 SMILES 重复尝试
            for idx, smi, mol, _ in mol_with_size:
                if idx in used_indices or smi in tried_mols:
                    continue
                tried_mols.add(smi)

                assigned = False
                if scope == "same_molecule":
                    # same_molecule: 所有特征必须在同一个分子上
                    all_ok, _ = self._match_features_on_single_molecule(
                        mol, features
                    )
                    if all_ok:
                        assigned = True
                        used_indices.add(idx)
                        allocation[req_idx] = [(idx, smi)]
                        alloc_indices[req_idx] = {idx}

                else:  # separate_molecule
                    if len(features) == 1:
                        # 单特征: 分配一个分子
                        if self._match_single_feature(mol, features[0]):
                            assigned = True
                            used_indices.add(idx)
                            allocation[req_idx] = [(idx, smi)]
                            alloc_indices[req_idx] = {idx}
                    else:
                        # 多特征: 每个特征分配不同的分子（含特征级回溯）
                        # 贪心匹配可能错过有效组合，使用回溯探索
                        # 所有可能的特征-分子分配方案
                        feat_result = {"found": False, "assignments": []}

                        def _feat_recurse(feat_idx, temp_assignments,
                                          temp_used_indices):
                            if feat_result["found"]:
                                return  # 找到一个有效分配即可
                            if feat_idx == len(features):
                                feat_result["found"] = True
                                feat_result["assignments"] = list(
                                    temp_assignments
                                )
                                return
                            feat = features[feat_idx]
                            tried = set()
                            for idx2, s2, m2, _ in mol_with_size:
                                if idx2 in used_indices or idx2 in temp_used_indices:
                                    continue
                                if s2 in tried:
                                    continue
                                tried.add(s2)
                                if self._match_single_feature(m2, feat):
                                    # C-3: 存储 (idx, smi) 元组
                                    temp_assignments.append((idx2, s2))
                                    temp_used_indices.add(idx2)
                                    _feat_recurse(
                                        feat_idx + 1,
                                        temp_assignments,
                                        temp_used_indices,
                                    )
                                    temp_assignments.pop()
                                    temp_used_indices.discard(idx2)
                                    if feat_result["found"]:
                                        return

                        _feat_recurse(0, [], set())
                        if feat_result["found"]:
                            assigned = True
                            assigned_idx_set = set()
                            for idx2, s2 in feat_result["assignments"]:
                                used_indices.add(idx2)
                                assigned_idx_set.add(idx2)
                            allocation[req_idx] = feat_result["assignments"]
                            alloc_indices[req_idx] = assigned_idx_set

                if assigned:
                    _recurse(req_idx + 1, used_indices,
                             allocation, score + req_score,
                             alloc_indices)
                    # 回溯：释放分子 — C-3 使用精确索引释放
                    allocation.pop(req_idx)
                    released_indices = alloc_indices.pop(req_idx, set())
                    for ridx in released_indices:
                        used_indices.discard(ridx)

            # ===== 跳过当前要求（仅子集模式）=====
            if allow_skip:
                _recurse(req_idx + 1, used_indices, allocation, score,
                         alloc_indices)

        _recurse(0, set(), {}, 0, {})
        return result

    # ----------------------------------------------------------
    #  三分类判定
    # ----------------------------------------------------------

    def _classify_side(
        self,
        side_result: Dict[str, Any],
        n_molecules: int,
        n_requirements: int,
    ) -> str:
        """
        对单侧（反应物侧或产物侧）进行分类。

        参数:
            side_result: _traverse_requirements 的返回字典
            n_molecules: 该侧的输入分子数
            n_requirements: 该侧的特征要求数

        返回:
            "full"（全满足）、"subset"（子集）、"mismatch"（不匹配）

        判定逻辑:
          - full_strategies 非空 → full（全满足）
          - 评分为 0 → mismatch（前提条件不满足）
          - 有分子未被分配 → mismatch（存在"搭不上边的"分子）
          - 所有分子都被分配但有要求未满足 → subset（分子不够多）
        """
        # 全满足模式有结果 → 所有要求都满足
        if side_result["full_strategies"]:
            return "full"

        # 未满足的要求数
        unsatisfied = n_requirements - len(side_result["best_allocation"])

        # 所有要求都满足（理论上应被 full_strategies 捕获）
        if unsatisfied == 0:
            return "full"

        # 评分为 0 → 无效（前提条件不满足）
        if side_result["best_score"] <= 0:
            return "mismatch"

        # C-TM1 修复：使用索引追踪计算被分配的分子数
        # 旧代码用 SMILES 字符串集合统计，重复分子只算一个，
        # 导致"子集"被误判为"不匹配"。
        # 新代码用位置索引集合，相同 SMILES 的不同位置各算一个。
        used_indices: Set[int] = set()
        best_idx_alloc = side_result.get("best_index_alloc", {})
        if best_idx_alloc:
            for indices in best_idx_alloc.values():
                used_indices.update(indices)
        else:
            # 向后兼容：如果 best_index_alloc 不存在，回退到旧逻辑
            for mols in side_result["best_allocation"].values():
                used_indices.update(range(len(mols)))
        n_used = len(used_indices)

        # 关键判定：是否有分子不匹配任何特征要求
        # 如果所有分子都被分配 → 子集（分子不够多，但没有"搭不上边的"）
        # 如果有分子未被分配 → 不匹配（存在"完全搭不上边的"分子）
        if n_used < n_molecules:
            return "mismatch"

        # 所有分子都被分配，但有要求未满足 → 子集
        return "subset"

    # ----------------------------------------------------------
    #  评分（子集路径 B 使用）
    # ----------------------------------------------------------

    def _calculate_score(
        self,
        allocation: dict,
        requirements: list,
    ) -> float:
        """
        计算给定分配方案的评分。

        全有或全无：每个被满足的要求得满分
        (base_score + multi_feature_bonus)，未满足得 0 分。
        """
        total = 0.0
        for req_idx in allocation:
            if req_idx < len(requirements):
                req = requirements[req_idx]
                total += req.get("base_score", 1.0)
                total += req.get("multi_feature_bonus", 0)
        return total

    # ----------------------------------------------------------
    #  模板应用（全满足路径 A）
    # ----------------------------------------------------------

    def _apply_template_for_full_match(
        self,
        reactants: List[str],
        products: List[str],
        record: dict,
        r_allocation: dict,
        detected_halogen: Optional[str],
        r_index_alloc: Optional[dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        对全满足记录应用配平模板。

        1. 确定旁观者分子（未被分配给任何特征要求的输入分子）
        2. 添加模板定义的共反应物和副产物
        3. 旁观者复制到产物侧（原子守恒计算中自然抵消）
        4. 验证原子守恒

        返回:
            验证通过的候选字典，或 None（验证失败）
        """
        template = record["template"]
        byproducts = list(template.get("byproducts", []))
        coreactants = list(template.get("coreactants", []))

        # 卤素占位符解析
        if detected_halogen:
            byproducts = [
                bp.replace("{X}", detected_halogen) for bp in byproducts
            ]
            coreactants = [
                cs.replace("{X}", detected_halogen) for cs in coreactants
            ]

        # 标准化 SMILES（含未解析占位符 → 拒绝该候选）
        all_bp_smiles = []
        for bp in byproducts:
            if "{X}" in bp:
                # 卤素占位符未解析，模板无法产出有效方程
                return None
            bp_mol = Chem.MolFromSmiles(bp)
            if bp_mol is not None:
                all_bp_smiles.append(Chem.MolToSmiles(bp_mol))

        all_cs_smiles = []
        for cs in coreactants:
            if "{X}" in cs:
                return None
            cs_mol = Chem.MolFromSmiles(cs)
            if cs_mol is not None:
                all_cs_smiles.append(Chem.MolToSmiles(cs_mol))

        # 确定旁观者分子（反应物侧未被分配的分子）
        # C-TM1 修复：使用索引追踪，避免 SMILES 集合过滤掉重复分子的旁观者副本
        if r_index_alloc:
            assigned_idx_set: Set[int] = set()
            for idx_list in r_index_alloc.values():
                assigned_idx_set.update(idx_list)
            spectators = [
                r for i, r in enumerate(reactants)
                if i not in assigned_idx_set
            ]
        else:
            # 向后兼容：无索引信息时回退到旧逻辑
            assigned_reactants: Set[str] = set()
            for mols in r_allocation.values():
                assigned_reactants.update(mols)
            spectators = [r for r in reactants if r not in assigned_reactants]

        # 构建增强方程
        augmented_reactants = reactants + all_cs_smiles
        augmented_products = products + all_bp_smiles + spectators

        # 原子守恒验证
        balanced, imbalance = _check_atom_balance(
            augmented_reactants, augmented_products
        )

        if not balanced:
            return None

        # 额外补全的物质数量（用于排序）
        added_count = len(all_cs_smiles) + len(all_bp_smiles)

        # 构建完整的配平反应 SMILES
        balanced_reaction = "{}>>{}".format(
            ".".join(augmented_reactants),
            ".".join(augmented_products),
        )

        return {
            "balanced_reaction": balanced_reaction,
            "added_species": all_cs_smiles + all_bp_smiles,
            "added_count": added_count,
            "balanced": True,
            "spectators": spectators,
        }

    # ----------------------------------------------------------
    #  RunReactants 产物验证（预留接口）
    # ----------------------------------------------------------

    def _verify_with_run_reactants(
        self,
        reactants: List[str],
        products: List[str],
        record: dict,
        r_allocation: dict,
        detected_halogen: Optional[str] = None,
    ) -> bool:
        """
        通过 RunReactants 验证模板生成的产物是否包含输入产物。

        验证条件：模板完整产物集合 ⊇ 输入产物集合。
        先尝试 canonical SMILES 精确匹配，失败时回退到 InChI
        分子级比较（处理同一分子不同 canonical 表示的情况）。

        对含 {X} 占位符的卤素依赖型模板，在解析前将 {X} 替换为
        实际检测到的卤素元素符号。

        注意：异常视为验证失败（fail-fast 原则）。
        """
        template = record["template"]
        reaction_smarts = template.get("reaction_smarts")

        if reaction_smarts is None:
            # 模板未提供 reaction_smarts，跳过 RunReactants 验证
            # 依赖原子守恒检查（在 _apply_template_for_full_match 中完成）
            return True

        # 卤素占位符替换
        if detected_halogen and "{X}" in reaction_smarts:
            reaction_smarts = reaction_smarts.replace("{X}", detected_halogen)

        try:
            from rdkit.Chem import AllChem

            rxn = AllChem.ReactionFromSmarts(reaction_smarts)
            if rxn is None:
                return False

            # 确定参与模板反应的分子（被分配的分子，按模板要求的顺序）
            assigned_reactants_ordered = []
            for req_idx in sorted(r_allocation.keys()):
                assigned_reactants_ordered.extend(r_allocation[req_idx])

            reactant_mols = []
            for smi in assigned_reactants_ordered:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    return False
                reactant_mols.append(m)

            # 运行 RunReactants
            product_sets = rxn.RunReactants(reactant_mols)
            if not product_sets:
                return False

            # 输入产物的 canonical SMILES 集合
            input_product_set = set()
            for p in products:
                pm = Chem.MolFromSmiles(p)
                if pm is not None:
                    input_product_set.add(Chem.MolToSmiles(pm))

            # C-TM3 修复：空集守卫
            # 如果所有输入产物都无法解析，required_products 为空集，
            # 后续超集检查（generated_set >= set()）恒为 True，
            # 会导致验证假通过。此处直接拒绝。
            if not input_product_set:
                return False

            # 仅验证输入产物是否出现在 RunReactants 输出中
            # 旁观者不参与模板反应，不应要求出现在生成产物中
            required_products = input_product_set

            # 第一轮：canonical SMILES 字符串精确匹配
            for product_tuple in product_sets:
                generated_set = set()
                for pm in product_tuple:
                    generated_set.add(Chem.MolToSmiles(pm))
                if generated_set >= required_products:
                    return True

            # 第二轮：InChI 分子级比较（回退）
            # RunReactants 生成的分子可能因内部原子编号不同而产生不同的
            # canonical SMILES，但 InChI 是分子唯一标识，不受编号影响。
            try:
                from rdkit.Chem.inchi import MolToInchi

                required_inchis = set()
                for smi in required_products:
                    m = Chem.MolFromSmiles(smi)
                    if m is not None:
                        inchi = MolToInchi(m)
                        if inchi:
                            required_inchis.add(inchi)

                if required_inchis:
                    for product_tuple in product_sets:
                        generated_inchis = set()
                        for pm in product_tuple:
                            inchi = MolToInchi(pm)
                            if inchi:
                                generated_inchis.add(inchi)
                        if generated_inchis >= required_inchis:
                            return True
            except ImportError:
                pass  # InChI 不可用时跳过回退比较

            return False

        except Exception:
            # 异常视为验证失败
            return False

    # ----------------------------------------------------------
    #  构建 inference_detail（子集路径 B 使用）
    # ----------------------------------------------------------

    def _build_inference_detail(
        self,
        record: dict,
        r_result: Dict[str, Any],
        p_result: Dict[str, Any],
        r_score: float,
        p_score: float,
        reactants: List[str],
        products: List[str],
    ) -> Dict[str, Any]:
        """
        构建子集路径的 inference_detail 字典。

        包含：已满足的要求、未满足的要求、推断的缺失物质。
        供 Bridge 提示词生成使用。
        """
        detail: Dict[str, Any] = {
            "reaction_type": record["reaction_type"],
            "total_score": r_score + p_score,
            "satisfied_requirements": [],
            "unsatisfied_requirements": [],
            "inferred_missing": [],
        }

        # ---- 反应物侧 ----
        r_reqs = record.get("reactant_requirements", [])
        for req_idx in range(len(r_reqs)):
            req = r_reqs[req_idx]
            if req_idx in r_result["best_allocation"]:
                detail["satisfied_requirements"].append({
                    "side": "reactant",
                    "requirement_index": req_idx,
                    "matched_molecules": r_result["best_allocation"][req_idx],
                })
            else:
                feat_labels = [f.get("label", "unknown")
                               for f in req.get("features", [])]
                desc = "含 {} 的反应物".format(
                    ", ".join(feat_labels) if feat_labels else "特定结构"
                )
                detail["unsatisfied_requirements"].append({
                    "side": "reactant",
                    "requirement_index": req_idx,
                    "expected_feature": (
                        feat_labels[0] if feat_labels else "unknown"
                    ),
                })
                detail["inferred_missing"].append({
                    "side": "reactant",
                    "description": desc,
                })

        # ---- 产物侧 ----
        p_reqs = record.get("product_requirements", [])
        for req_idx in range(len(p_reqs)):
            req = p_reqs[req_idx]
            if req_idx in p_result["best_allocation"]:
                detail["satisfied_requirements"].append({
                    "side": "product",
                    "requirement_index": req_idx,
                    "matched_molecules": p_result["best_allocation"][req_idx],
                })
            else:
                feat_labels = [f.get("label", "unknown")
                               for f in req.get("features", [])]
                desc = "含 {} 的产物".format(
                    ", ".join(feat_labels) if feat_labels else "特定结构"
                )
                detail["unsatisfied_requirements"].append({
                    "side": "product",
                    "requirement_index": req_idx,
                    "expected_feature": (
                        feat_labels[0] if feat_labels else "unknown"
                    ),
                })
                detail["inferred_missing"].append({
                    "side": "product",
                    "description": desc,
                })

        return detail

    # ----------------------------------------------------------
    #  主入口：模板匹配完整流程
    # ----------------------------------------------------------

    def run_template_matching(
        self,
        reactants: List[str],
        products: List[str],
        rb_method=None,
    ) -> Dict[str, Any]:
        """
        模板匹配的主入口。

        遍历查询推测表中的所有记录，对每条记录执行回溯遍历
        并分类为全满足/子集/不匹配。

        决策流程（按优先级从高到低）：
          1. 收集所有全满足记录 → 路径 A 模板验证 → 排序输出
          2. 全满足都未通过 → 收集所有子集记录 → 评分排名 → Bridge 提示
          3. 无子集 → 兜底提示

        参数:
            reactants: 反应物 SMILES 列表
            products: 产物 SMILES 列表
            rb_method: 保留参数以兼容管线调用

        返回字典（成功路径——路径 A 验证通过）:
            success=True, balanced_reaction, template_id/name/label,
            score=0.5 (固定), method="template_matching"

        返回字典（失败/推测路径——路径 B 或路径 A 验证失败）:
            success=False, reason, template_id/name/label,
            score (子集总分或 None), as_bridge_hint=True,
            inference_detail (子集详情或 None)
        """
        full_match_candidates = []
        subset_records = []
        full_match_attempted = False

        for record in self.inference_table:
            r_reqs = record.get("reactant_requirements", [])
            p_reqs = record.get("product_requirements", [])

            # ---- 反应物侧遍历（全满足模式） ----
            r_result = self._traverse_requirements(
                r_reqs, reactants, allow_skip=False
            )
            r_class = self._classify_side(
                r_result, len(reactants), len(r_reqs)
            )

            # ---- 产物侧遍历（全满足模式） ----
            p_result = self._traverse_requirements(
                p_reqs, products, allow_skip=False
            )
            p_class = self._classify_side(
                p_result, len(products), len(p_reqs)
            )

            # ---- 判断是否进入路径 A（模板验证）----
            # 分类表：反应物 full + 产物 full/subset → Path A
            # （反应物完整即可运行 RunReactants，产物缺失由模板补全）
            if r_class == "full" and p_class != "mismatch":
                # 路径 A：模板验证
                detected_halogens: List[str] = []
                if record.get("template", {}).get("halogen_dependent"):
                    for r in reactants:
                        for h in _detect_halogen(r):
                            if h not in detected_halogens:
                                detected_halogens.append(h)

                # 对每种反应物分配策略尝试模板应用
                # 产物侧为 subset 时 full_strategies 为空，用空分配兜底
                r_strategies = r_result["full_strategies"] or [{}]
                r_index_strategies = r_result.get("full_index_strategies", [])
                for strat_idx, r_alloc in enumerate(r_strategies):
                    # C-TM1: 取出对应的索引版分配方案
                    r_idx_alloc = (
                        r_index_strategies[strat_idx]
                        if strat_idx < len(r_index_strategies)
                        else None
                    )
                    # 卤素依赖型模板：为每种卤素各生成一个候选方案
                    halogen_variants = (
                        detected_halogens if detected_halogens else [None]
                    )
                    for hal in halogen_variants:
                        candidate = self._apply_template_for_full_match(
                            reactants, products, record,
                            r_alloc, hal,
                            r_index_alloc=r_idx_alloc,
                        )
                        if candidate is not None:
                            # RunReactants 验证（传入卤素用于 {X} 替换）
                            verified = self._verify_with_run_reactants(
                                reactants, products, record,
                                r_alloc, hal,
                            )
                            if verified:
                                full_match_candidates.append({
                                    "candidate": candidate,
                                    "record": record,
                                    "r_allocation": r_alloc,
                                })
                            else:
                                full_match_attempted = True
                        else:
                            full_match_attempted = True

                continue  # 已处理为路径 A，无需子集检查

            # ---- 非全满足 → 子集模式重新遍历 ----
            r_result_sub = self._traverse_requirements(
                r_reqs, reactants, allow_skip=True
            )
            r_class_sub = self._classify_side(
                r_result_sub, len(reactants), len(r_reqs)
            )

            p_result_sub = self._traverse_requirements(
                p_reqs, products, allow_skip=True
            )
            p_class_sub = self._classify_side(
                p_result_sub, len(products), len(p_reqs)
            )

            # 前提条件：任何有效匹配必须两侧都至少得 1 分
            if r_result_sub["best_score"] <= 0 or \
               p_result_sub["best_score"] <= 0:
                continue  # 无效，等同于不匹配

            # 混合情况：一侧子集 + 另一侧不匹配 → 整个记录不匹配
            if r_class_sub == "mismatch" or p_class_sub == "mismatch":
                continue

            # 两侧都是子集（或一侧子集一侧全满足）→ 子集
            if r_class_sub == "subset" or p_class_sub == "subset":
                r_score = self._calculate_score(
                    r_result_sub["best_allocation"], r_reqs
                )
                p_score = self._calculate_score(
                    p_result_sub["best_allocation"], p_reqs
                )
                total_score = r_score + p_score

                detail = self._build_inference_detail(
                    record, r_result_sub, p_result_sub,
                    r_score, p_score, reactants, products,
                )

                subset_records.append({
                    "record": record,
                    "total_score": total_score,
                    "r_score": r_score,
                    "p_score": p_score,
                    "inference_detail": detail,
                    "r_result": r_result_sub,
                    "p_result": p_result_sub,
                })

        # ============================================================
        #  决策流程
        # ============================================================

        # ---- 1. 优先处理全满足 ----
        if full_match_candidates:
            # 排序：额外补全物质数量升序 → commonness_rank 升序
            full_match_candidates.sort(
                key=lambda c: (
                    c["candidate"]["added_count"],
                    c["record"]["commonness_rank"],
                )
            )
            best = full_match_candidates[0]
            template = best["record"]["template"]
            return {
                "success": True,
                "balanced_reaction": best["candidate"]["balanced_reaction"],
                "template_id": template["id"],
                "template_name": template["name"],
                "template_label": template.get("label", template["name"]),
                "score": 0.5,
                "method": "template_matching",
            }

        # ---- 1b. 全满足候选存在但全部验证失败 ----
        if not full_match_candidates and full_match_attempted and not subset_records:
            return {
                "success": False,
                "balanced_reaction": None,
                "reason": "套用模板无法得到包含原始输入的反应物和产物的方程",
                "template_id": None,
                "template_name": None,
                "template_label": None,
                "score": None,
                "as_bridge_hint": True,
                "inference_detail": None,
            }

        # ---- 2. 全满足都未通过 → 处理子集 ----
        if subset_records:
            # 排序：总分降序 → commonness_rank 升序
            subset_records.sort(
                key=lambda s: (
                    -s["total_score"],
                    s["record"]["commonness_rank"],
                )
            )
            best_sub = subset_records[0]
            template = best_sub["record"]["template"]
            return {
                "success": False,
                "balanced_reaction": None,
                "reason": "反应物侧或产物侧特征要求未完全满足",
                "template_id": template["id"],
                "template_name": template["name"],
                "template_label": template.get("label", template["name"]),
                "score": best_sub["total_score"],
                "as_bridge_hint": True,
                "inference_detail": best_sub["inference_detail"],
            }

        # ---- 3. 没有子集 → 兜底提示 ----
        return {
            "success": False,
            "balanced_reaction": None,
            "reason": "模板库中无可用模板",
            "template_id": None,
            "template_name": None,
            "template_label": None,
            "score": None,
            "as_bridge_hint": True,
            "inference_detail": None,
        }


# ============================================================
#  模块级便捷函数（供 run_rebalancer_with_llm.py 调用）
# ============================================================

_global_matcher: Optional[TemplateMatcher] = None


def get_template_matcher(
    template_db_path: str = "reaction_templates.json",
    similarity_threshold: float = 0.5,
) -> TemplateMatcher:
    """获取全局单例 TemplateMatcher。"""
    global _global_matcher
    if _global_matcher is None:
        _global_matcher = TemplateMatcher(
            template_db_path=template_db_path,
            similarity_threshold=similarity_threshold,
        )
    return _global_matcher


def template_matching_for_reaction(
    reaction_smiles: str,
    rb_method=None,
    similarity_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    对单条反应执行模板匹配。管线唯一入口。

    参数:
        reaction_smiles: "反应物>>产物" 格式的反应 SMILES
        rb_method: 保留参数以兼容管线调用
        similarity_threshold: 保留参数以兼容管线调用

    返回字典（成功路径——路径 A 验证通过）:
        success: True
        balanced_reaction: 配平后的完整反应 SMILES
        template_id / template_name / template_label: 模板标识
        score: 固定为 0.5
        method: "template_matching"

    返回字典（失败/推测路径——路径 B 或路径 A 验证失败）:
        success: False
        reason: 失败原因字符串
        template_id / template_name / template_label: 子集记录的模板标识
        score: 子集总分（无子集时 None）
        as_bridge_hint: True
        inference_detail: 推测详情字典（无子集时 None）
    """
    if not reaction_smiles or ">>" not in reaction_smiles:
        return {
            "success": False,
            "balanced_reaction": None,
            "reason": "无效反应 SMILES",
            "template_id": None,
            "template_name": None,
            "template_label": None,
            "score": None,
            "as_bridge_hint": True,
            "inference_detail": None,
        }

    parts = reaction_smiles.split(">>", 1)
    if len(parts) != 2:
        return {
            "success": False,
            "balanced_reaction": None,
            "reason": "无效反应格式（未找到 '>>' 分隔符）",
            "template_id": None,
            "template_name": None,
            "template_label": None,
            "score": None,
            "as_bridge_hint": True,
            "inference_detail": None,
        }
    reactants = [s for s in parts[0].split(".") if s]
    products = [s for s in parts[1].split(".") if s]

    if not reactants or not products:
        return {
            "success": False,
            "balanced_reaction": None,
            "reason": "反应物或产物为空",
            "template_id": None,
            "template_name": None,
            "template_label": None,
            "score": None,
            "as_bridge_hint": True,
            "inference_detail": None,
        }

    matcher = get_template_matcher(
        similarity_threshold=similarity_threshold
    )
    return matcher.run_template_matching(reactants, products, rb_method)
