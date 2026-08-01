import logging

import joblib
import pandas as pd
import numpy as np
import importlib.resources
import synrbl.SynAnalysis

logger = logging.getLogger("synrbl")

from synrbl.SynAnalysis.analysis_utils import (
    calculate_chemical_properties,
    count_boundary_atoms_products_and_calculate_changes,
)
from synrbl.SynUtils.common import update_reactants_and_products


class ConfidencePredictor:
    def __init__(
        self,
        reaction_col="reaction",
        input_reaction_col="input_reaction",
        confidence_col="confidence",
        solved_col="solved",
        solved_by_col="solved_by",
        solved_by_method="mcs-based",
        issue_col="issue",
        mcs_col="mcs",
        id_col="id",
    ):
        model_path = str(
            importlib.resources.files(synrbl.SynAnalysis)
            .joinpath("scoring_function.dump")
        )
        try:
            self.model = joblib.load(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"置信度预测模型加载失败: {model_path}。"
                f"请确认模型文件存在且未损坏。原始错误: {exc}"
            ) from exc
        self.reaction_col = reaction_col
        self.input_reaction_col = input_reaction_col
        self.confidence_col = confidence_col
        self.solved_col = solved_col
        self.solved_by_col = solved_by_col
        self.solved_by_method = solved_by_method
        self.issue_col = issue_col
        self.mcs_col = mcs_col
        self.id_col = id_col

    def predict(self, reactions, stats=None, threshold=0, allow_low_confidence_solved=False):
        reactions = [
            r
            for r in reactions
            if self.solved_by_col in r.keys()
            and r[self.solved_by_col] == self.solved_by_method
        ]
        conf_success = 0
        low_confidence_preserved_cnt = 0
        if len(reactions) > 0:
            _reactions = count_boundary_atoms_products_and_calculate_changes(
                reactions, self.reaction_col, self.mcs_col
            )
            update_reactants_and_products(_reactions, self.input_reaction_col)
            _reactions = calculate_chemical_properties(_reactions)

            FEATURE_COLS = [
                "carbon_difference",
                "fragment_count",
                "total_carbons",
                "total_bonds",
                "total_rings",
                "num_boundary",
                "ring_change_merge",
                "bond_change_merge",
            ]

            # C-5 修复：分离有效/无效反应
            # calculate_chemical_properties 对无法解析的 SMILES 将 4 个特征
            # 设为字符串 "Invalid SMILES"，XGBoost 要求纯数值输入，
            # 因此将这些反应标记为 unsolved，仅对有效子集运行 predict_proba
            valid_indices = []
            for i, entry in enumerate(_reactions):
                if all(
                    isinstance(entry.get(col), (int, float))
                    for col in FEATURE_COLS
                ):
                    valid_indices.append(i)
                else:
                    r = reactions[i]
                    r[self.solved_col] = False
                    r[self.confidence_col] = 0.0
                    r["workflow_confidence"] = 0.0
                    issue_msg = (
                        "Confidence prediction skipped: "
                        "invalid SMILES in reaction."
                    )
                    if r.get(self.issue_col):
                        r[self.issue_col] += "; " + issue_msg
                    else:
                        r[self.issue_col] = issue_msg

            if valid_indices:
                valid_reactions = [_reactions[i] for i in valid_indices]
                df = pd.DataFrame(valid_reactions)
                X_pred = df[FEATURE_COLS]

                confidence = np.round(
                    self.model.predict_proba(X_pred)[:, 1], 3
                )
                assert len(valid_reactions) == len(confidence)
                for idx, c in zip(valid_indices, confidence):
                    r = reactions[idx]
                    c_value = c.item()
                    r[self.confidence_col] = c_value
                    r["workflow_confidence"] = c_value
                    if c >= threshold:
                        conf_success += 1
                    elif allow_low_confidence_solved:
                        low_confidence_preserved_cnt += 1
                    else:
                        if r.get(self.issue_col, "") != "":
                            logger.warning(
                                "Solved reaction %s has non-empty issue "
                                "('%s'); clearing before rejection.",
                                r.get(self.id_col, "?"),
                                r[self.issue_col],
                            )
                            r[self.issue_col] = ""
                        r[self.solved_col] = False
                        r[self.issue_col] = (
                            "Confidence is below the threshold of "
                            "{:.2%}.".format(threshold)
                        )
        if stats is not None:
            stats["confident_cnt"] = stats.get("confident_cnt", 0) + conf_success
            if allow_low_confidence_solved:
                stats["low_confidence_preserved_cnt"] = (
                    stats.get("low_confidence_preserved_cnt", 0)
                    + low_confidence_preserved_cnt
                )
        return reactions
