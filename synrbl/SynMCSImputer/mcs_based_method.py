from synrbl.SynMCSImputer.structure import CompoundSet
from synrbl.SynMCSImputer.utils import is_carbon_balanced
from synrbl.SynMCSImputer.merge import merge
from rdkit.rdBase import BlockLogs

import logging

logger = logging.getLogger("synrbl")


def build_compounds(data_dict) -> CompoundSet:
    src_smiles = data_dict["sorted_reactants"]
    smiles = data_dict["smiles"]
    boundaries = data_dict["boundary_atoms_products"]
    neighbors = data_dict["nearest_neighbor_products"]
    mcs_results = data_dict["mcs_results"]
    radical_sites_list = data_dict.get("radical_sites", [[] for _ in smiles])
    n = len(smiles)
    if n != len(src_smiles):
        raise ValueError(
            "Smiles and sorted reactants are not of the same length. ({} != {})".format(
                len(smiles), len(src_smiles)
            )
        )
    if n != len(boundaries) or n != len(neighbors):
        raise ValueError(
            "Boundaries and nearest neighbors must be of same length as compounds."
        )
    if n != len(mcs_results):
        raise ValueError("MCS results must be of same length as compounds.")
    cset = CompoundSet()
    for s, ss, b, n_nbr, mcs, rad_sites in zip(
        smiles, src_smiles, boundaries, neighbors, mcs_results, radical_sites_list
    ):
        # Build radical lookup: atom_idx -> num_radical_electrons
        radical_map = {}
        for site in (rad_sites or []):
            radical_map[site["atom_idx"]] = site["num_radical_electrons"]

        if s is None:
            if mcs == "":
                # TODO use compound rule for that
                if ss == "O":
                    # water is not catalyst -> binds to other compound
                    c = cset.add_compound(ss, src_mol=ss)
                    c.add_boundary(0, symbol="O")
                else:
                    # catalysis compound
                    c = cset.add_compound(ss, src_mol=ss)
            else:
                # empty compound
                pass
        else:
            c = cset.add_compound(s, src_mol=ss)
            if len(b) != len(n_nbr):
                raise ValueError(
                    (
                        "Boundary and neighbor missmatch. "
                        + "(boundary={}, neighbor={})"
                    ).format(b, n_nbr)
                )
            for bi, ni in zip(b, n_nbr):
                bi_s, bi_i = list(bi.items())[0]
                ni_s, ni_i = list(ni.items())[0]
                rad_e = radical_map.get(bi_i, 0)
                c.add_boundary(
                    bi_i, symbol=bi_s, neighbor_index=ni_i, neighbor_symbol=ni_s,
                    radical_electrons=rad_e,
                    capped_hydrogens=rad_e,
                )
    return cset


def impute_reaction(
    reaction_dict,
    reaction_col,
    issue_col,
    carbon_balance_col,
    mcs_data_col,
    smiles_standardizer=[],
    enable_multi_fragment=True,
):
    issue = reaction_dict[issue_col] if issue_col in reaction_dict.keys() else ""
    if issue != "":
        raise ValueError("Skip reaction because of previous issue.\n" + issue)
    compound_set = build_compounds(reaction_dict[mcs_data_col])
    if len(compound_set) == 0:
        raise ValueError("Empty compound set.")
    merge_results = merge(compound_set, enable_multi_fragment=enable_multi_fragment)
    carbon_balance = reaction_dict[carbon_balance_col]

    # impute_direction: 记录 MCS 补全方向
    if carbon_balance == "reactants":
        impute_direction = "coreactant"
    elif carbon_balance == "products":
        impute_direction = "byproduct"
    else:
        impute_direction = "balanced"

    original_reaction = reaction_dict[reaction_col]
    results = []

    for merge_result in merge_results:
        merged_smiles = merge_result.smiles
        for standardizer in smiles_standardizer:
            merged_smiles = standardizer(merged_smiles)

        if carbon_balance == "reactants":
            # 产物侧碳多于反应物侧 → 缺失的是共反应物
            parts = original_reaction.split(">>", 1)
            if len(parts) == 2:
                imputed_reaction = "{}.{}>>{}".format(
                    merged_smiles, parts[0], parts[1]
                )
            else:
                raise ValueError(
                    "Cannot impute reactant side: "
                    "reaction format missing '>>' separator."
                )
        elif carbon_balance in ["products", "balanced"]:
            # 反应物侧碳多于或等于产物侧 → 缺失的是副产物
            imputed_reaction = "{}.{}".format(original_reaction, merged_smiles)
        else:
            raise ValueError(
                "Invalid value '{}' for carbon balance.".format(carbon_balance)
            )

        rules = [r.name for r in merge_result.rules]

        # 碳校验：快速过滤碳不平衡的候选
        if not is_carbon_balanced(imputed_reaction):
            logger.debug(
                "Merge candidate skipped: carbon not balanced. SMILES: %s",
                imputed_reaction,
            )
            continue

        results.append((imputed_reaction, rules, impute_direction))

    if not results:
        raise RuntimeError(
            "All merge candidates failed carbon balance check. "
            "No valid imputed reaction produced."
        )
    return results


class MCSBasedMethod:
    def __init__(
        self,
        reaction_col,
        output_col,
        mcs_data_col="mcs",
        issue_col="issue",
        rules_col="rules",
        carbon_balance_col="carbon_balance_check",
        smiles_standardizer=[],
        enable_multi_fragment=True,
    ):
        self.reaction_col = reaction_col
        self.output_col = output_col if isinstance(output_col, list) else [output_col]
        self.mcs_data_col = mcs_data_col
        self.issue_col = issue_col
        self.rules_col = rules_col
        self.carbon_balance_col = carbon_balance_col
        self.smiles_standardizer = smiles_standardizer
        self.enable_multi_fragment = enable_multi_fragment

    def run(self, reactions: list[dict], stats=None):
        mcs_applied = 0
        mcs_solved = 0
        block_logs = BlockLogs()
        for reaction in reactions:
            if self.mcs_data_col not in reaction.keys():
                continue
            mcs_applied += 1
            if reaction[self.mcs_data_col] is None:
                continue
            try:
                candidates = impute_reaction(
                    reaction,
                    mcs_data_col=self.mcs_data_col,
                    reaction_col=self.reaction_col,
                    issue_col=self.issue_col,
                    carbon_balance_col=self.carbon_balance_col,
                    smiles_standardizer=self.smiles_standardizer,
                    enable_multi_fragment=self.enable_multi_fragment,
                )
                # 存储所有候选供下游管线选择（多碎片穷举场景）
                reaction["mcs_candidates"] = candidates
                # 使用第一个候选作为主结果（单候选路径不变）
                result, rules, impute_direction = candidates[0]
                for col in self.output_col:
                    reaction[col] = result
                reaction[self.rules_col] = rules
                reaction["impute_direction"] = impute_direction
                mcs_solved += 1
            except Exception as e:
                reaction[self.issue_col] = str(e)

        del block_logs
        if stats is not None:
            stats["mcs_applied"] = mcs_applied
            stats["mcs_solved"] = mcs_solved
        return reactions
