import pandas as pd

from synrbl.SynProcessor import RSMIProcessing
from synrbl.SynUtils import remove_atom_mapping


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
