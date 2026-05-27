import pandas as pd
import rdkit.Chem as Chem

from joblib import Parallel, delayed
from synrbl.rsmi_utils import save_database

from typing import Tuple


class RSMIProcessing:
    """
    A class to process reaction SMILES (RSMI) data.

    Parameters
    ----------
    rsmi : str, optional
        The reaction SMILES string to be processed.
    data : DataFrame, optional
        The DataFrame containing the RSMI data.
    rsmi_col : str, optional
        The column name in the DataFrame that contains the RSMI data.
    symbol : str, optional
        The symbol that separates reactants and products in the RSMI string
        (default is '>>').
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 10).
    verbose : int, optional
        The verbosity level (default is 1).
    parallel : bool, optional
        Whether to use parallel processing (default is True).
    save_json : bool, optional
        Whether to save the processed data to a JSON file (default is True).
    save_path_name : str, optional
        The path and name of the JSON file to save (default is 'reaction.json.gz').
    orient : str, optional
        The orientation of the JSON file (default is 'records').
    compression : str, optional
        The compression type for the JSON file (default is 'gzip').

    Example
    -------
    # Create an instance of the RSMIProcessing class with a DataFrame
    >>> data = pd.DataFrame({'rsmi': ['CCO>>CCOCC', 'CC>>C']})
    >>> processor = RSMIProcessing(data=data, rsmi_col='rsmi', parallel=False)
    # Split the RSMI data into reactants and products
    >>> processed_data = processor.data_splitter()
    # Display the processed data
    >>> print(processed_data)
    """

    def __init__(
        self,
        reaction_smiles: str = None,
        data: pd.DataFrame = None,
        data_name: str = "USPTO_50K",
        index_col: str = "R-id",
        rsmi_col: str = None,
        symbol: str = ">>",
        n_jobs: int = 4,
        drop_duplicates: bool = True,
        verbose: int = 1,
        parallel: bool = True,
        save_json: bool = False,
        save_path_name: str = "reaction.json.gz",
        orient: str = "records",
        compression: str = "gzip",
    ) -> None:
        self.reaction_smiles = reaction_smiles
        self.symbol = symbol
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.parallel = parallel
        self.data = data
        self.data_name = data_name
        self.index_col = index_col
        self.drop_duplicates = drop_duplicates
        self.rsmi_col = rsmi_col
        self.save_json = save_json
        self.save_path_name = save_path_name
        self.orient = orient
        self.compression = compression

    @staticmethod
    def smi_splitter(rsmi: str, symbol: str = ">>") -> Tuple[str, str]:
        """
        Split a RSMI string into reactants and products.

        Returns
        -------
        tuple
            The reactants and products as separate strings, if the RSMI string
            can be parsed.
        str
            A message indicating the RSMI string can't be parsed, if applicable.
        """

        # Check if the RSMI string can be parsed
        if RSMIProcessing.can_parse(rsmi, symbol):
            # Split the RSMI string into reactants and products
            return rsmi.split(symbol)
        else:
            # Return a message if the RSMI string cannot be parsed
            return "Can't parse"

    def data_splitter(self) -> pd.DataFrame:
        """
        Split the RSMI data in the DataFrame into reactants and products.

        Returns
        -------
        DataFrame
            The processed DataFrame with separate columns for reactants and products.

        Example
        -------
        >>> data = pd.DataFrame({'rsmi': ['C>>CC', 'CC>>CCC']})
        >>> processor = RSMIProcessing(data=data, rsmi_col='rsmi', parallel=False)
        >>> processed_data = processor.data_splitter()
        >>> print(processed_data)
        """
        working_data = self.data.copy()

        if self.rsmi_col not in working_data.columns:
            raise KeyError(f"Column '{self.rsmi_col}' not found in input data.")

        if self.parallel:
            parsable = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(RSMIProcessing.can_parse)(rsmi, self.symbol)
                for rsmi in working_data[self.rsmi_col].values
            )
        else:
            parsable = working_data[self.rsmi_col].apply(
                lambda rsmi: RSMIProcessing.can_parse(rsmi, self.symbol)
            )

        working_data["can_parse_reaction"] = parsable

        reactants = []
        products = []
        parse_issues = []
        for rsmi, can_parse in zip(working_data[self.rsmi_col].values, parsable):
            if can_parse:
                split_parts = str(rsmi).split(self.symbol, 1)
                reactants.append(split_parts[0])
                products.append(split_parts[1])
                parse_issues.append("")
            else:
                reactants.append(None)
                products.append(None)
                parse_issues.append("Reaction SMILES cannot be parsed.")

        working_data["reactants"] = reactants
        working_data["products"] = products
        working_data["parse_issue"] = parse_issues

        if self.drop_duplicates:
            working_data.drop_duplicates(subset=self.rsmi_col, inplace=True)
        working_data.reset_index(drop=True, inplace=True)

        if self.index_col not in working_data.columns:
            prefix = f"{self.data_name}_" if self.data_name is not None else "internal_"
            working_data[self.index_col] = [
                f"{prefix}{i}" for i in working_data.index
            ]

        self.data = working_data

        if self.save_json:
            data = self.data.to_dict(orient=self.orient)
            save_database(data, self.save_path_name)

        return self.data

    @staticmethod
    def can_parse(rsmi: str, symbol: str = ">>") -> bool:
        """
        Check if a RSMI string can be parsed into reactants and products.

        Parameters
        ----------
        rsmi : str
            The reaction smiles (RSMI) string to be checked for parsability.
        symbol : str, optional
            The symbol used to separate reactants and products in the RSMI
            string (default is '>>').

        Returns
        -------
        bool
            True if the RSMI string can be parsed into valid reactant and
            product SMILES, False otherwise.
        """
        if not isinstance(rsmi, str) or symbol not in rsmi:
            return False

        parts = rsmi.split(symbol)
        if len(parts) != 2:
            return False

        react, prod = parts
        return (
            Chem.MolFromSmiles(prod) is not None
            and Chem.MolFromSmiles(react) is not None
        )
