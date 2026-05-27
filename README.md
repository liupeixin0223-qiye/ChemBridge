# ChemBridge: LLM-Enhanced Reaction Rebalancing Workflow

ChemBridge is a computational chemistry workflow for correcting atom-imbalanced reaction SMILES. It combines the deterministic reaction-rebalancing components from SynRBL with an independently organized staged workflow that introduces LLM-assisted species bridging, fallback generation, auditing, routing, and result reporting.

> **Contribution and attribution note.** The repository name is ChemBridge, while the Python import/package name remains `synrbl` for compatibility with the modified SynRBL codebase. SynRBL provides the deterministic rule-based and MCS-based rebalancing components used in this project. Peixin Liu built the ChemBridge workflow around those components, including the staged orchestration in `run_rebalancer_with_llm.py`, the LLM species-bridge and fallback modules, the bridge/fallback validation logic, workflow-level confidence routing, output reporting, and format/interface adjustments needed to connect SynRBL outputs with the ChemBridge workflow. These interface adjustments are intended to preserve the deterministic SynRBL logic while making its outputs usable inside the full ChemBridge pipeline. Please cite the original SynRBL work when using its deterministic rebalancing components.

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [LLM Configuration](#llm-configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [Benchmark and Analysis](#benchmark-and-analysis)
- [Attribution, Citation, and License](#attribution-citation-and-license)

## Overview

ChemBridge separates the deterministic rebalancing function from the workflow logic that decides when and how to use it. In this repository, SynRBL contributes the deterministic rule-based and MCS-based reaction rebalancing methods. ChemBridge contributes the surrounding workflow framework that prepares inputs, routes cases across stages, calls LLM-assisted modules when needed, re-audits generated candidates, and writes traceable outputs.

The current workflow has three main decision layers:

1. **Deterministic SynRBL round**: preprocesses reaction SMILES, removes atom mapping, checks already-balanced reactions, then runs the SynRBL rule-based/MCS-based rebalancing components. Some output-field and interface formatting has been adjusted so that SynRBL results, including fields such as confidence, can be consumed consistently by the ChemBridge workflow.
2. **LLM species bridge**: for failed, invalid, or low-confidence cases, asks an LLM to infer likely missing side species from the original reaction and exact atom imbalance. Candidate variants are then audited through deterministic checks and the SynRBL-based rebalancing route before acceptance.
3. **LLM fallback generation**: if the bridge route cannot produce an accepted balanced result, asks the LLM for a complete reaction candidate and validates it again through deterministic balance checks and workflow-level routing.

The intended design is not to directly trust the LLM as the final judge. LLM outputs are treated as candidate generators. The workflow records validity, atom balance, source route, confidence, stage-level diagnostics, and bridge/fallback traces so that each result can be inspected.

## Workflow

The main entry point is:

```bash
python run_rebalancer_with_llm.py <input_file> [options]
```

At a high level, the workflow performs the following steps:

1. Load tabular input from CSV, TSV/TXT, Excel, or JSON.
2. Resolve reaction, ID, and optional expected-reaction columns.
3. Run the deterministic SynRBL round.
4. Classify each first-round result:
   - already prebalanced;
   - rule-based small-molecule solution;
   - high-confidence MCS solution;
   - low-confidence MCS candidate;
   - invalid/unbalanced/unsolved case requiring the bridge route.
5. Run the LLM species bridge for cases requiring additional side-species proposals.
6. Re-run deterministic validation on bridge-generated variants.
7. Run fallback LLM generation only for cases still unresolved after the bridge route.
8. Write staged JSON/CSV outputs, failed cases, workflow statistics, and accuracy comparison reports.

## Repository Structure

```text
ChemBridge/
├── run_rebalancer_with_llm.py        # Main staged ChemBridge workflow entry point
├── synrbl/                           # Modified SynRBL package and ChemBridge extensions
│   ├── balancing.py                  # Core deterministic Balancer
│   ├── reaction_rebalancer.py        # Batch rebalancing API/configuration
│   ├── llm_species_bridge.py         # LLM side-species proposal bridge
│   ├── llm_fallback_postprocessor.py # Final LLM fallback generator/auditor
│   ├── llm/                          # Moonshot/Kimi client and prompts
│   │   ├── species_prompts.py         # Active species-bridge prompts used by code
│   │   ├── fallback_prompts.py        # Active fallback prompts used by code
│   │   ├── species_prompts_strict.py  # Strict prompt variant used in paper experiments
│   │   └── fallback_prompts_strict.py # Strict prompt variant used in paper experiments
│   ├── SynRuleImputer/               # Rule-based imputation components
│   ├── SynMCSImputer/                # MCS-based imputation components
│   ├── SynChemImputer/               # Chemical post-processing utilities
│   └── SynProcessor/                 # Reaction SMILES preprocessing utilities
├── Data/                             # Validation/raw data and generated experiment outputs
├── Test/                             # Original/modified SynRBL tests
├── Scripts/                          # Utility scripts
├── pyproject.toml                    # Python package metadata
├── requirements.txt                  # Runtime dependencies
├── LICENSE                           # License inherited from SynRBL unless replaced
└── CITATION.cff                      # SynRBL citation metadata
```

This repository intentionally keeps the validation artifact folder `Data/Validation_set/rows-1-5032/`. Other generated `rows-*` output folders are ignored by `.gitignore` to avoid publishing stale experiment runs.

## Installation

ChemBridge requires Python 3.11 or later.

### Create an environment

```bash
python -m venv chembridge-env
chembridge-env\Scripts\activate
```

Or with Conda:

```bash
conda create --name chembridge-env python=3.11
conda activate chembridge-env
```

### Install dependencies

From the repository root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Core dependencies include:

- `synkit`
- `rdkit` through the chemistry dependency stack used by SynRBL/SynKit
- `xgboost`
- `imbalanced_learn`
- `scikit_learn`
- `reportlab`
- `fgutils`
- `pandas`

The development environment used for the current validation run used Python 3.11.15. Important package versions included RDKit 2026.3.1, SynKit 1.3, pandas 3.0.2, scikit-learn 1.7.2, imbalanced-learn 0.14.1, fgutils 0.2.3, reportlab 4.4.10, and xgboost 3.0.0. Equivalent newer compatible versions may also work, but these versions document the tested local environment.

If `pandas` or Excel support packages are missing in your environment, install them explicitly:

```bash
pip install pandas openpyxl
```

## LLM Configuration

ChemBridge currently uses a Moonshot/Kimi-compatible chat-completions endpoint by default.

Set your API key as an environment variable before running LLM-enabled workflows:

```powershell
$env:MOONSHOT_API_KEY="your_api_key_here"
```

Default LLM parameters in `run_rebalancer_with_llm.py`:

- API key environment variable: `MOONSHOT_API_KEY`
- Base URL: `https://api.moonshot.cn/v1/chat/completions`
- Score model: `kimi-k2.5`
- Generate model: `kimi-k2.5`

You can override them from the command line:

```bash
python run_rebalancer_with_llm.py data.csv \
  --llm-api-key-env MOONSHOT_API_KEY \
  --llm-base-url https://api.moonshot.cn/v1/chat/completions \
  --llm-generate-model kimi-k2.5
```

Do not commit API keys, `.env` files, or private service credentials to GitHub.

## Usage

### Input format

Input files may be `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, or `.json`.

By default, ChemBridge expects:

- reaction column: `reactions`
- ID column: `R-id`
- expected/reference reaction column: `expected_reaction` if available

Common fallback column names such as `reaction`, `rxn`, `rsmi`, `id`, and `ID` are also resolved automatically.

### Run on a CSV file

```bash
python run_rebalancer_with_llm.py Data/Validation_set/validation_set.csv \
  --reaction-col reactions \
  --id-col R-id \
  --expected-col expected_reaction
```

### Process only a subset

```bash
python run_rebalancer_with_llm.py Data/Validation_set/validation_set.csv --head 100
```

Or use one-based row slicing, excluding the header row:

```bash
python run_rebalancer_with_llm.py Data/Validation_set/validation_set.csv \
  --start-row 1 \
  --end-row 500
```

### LLM-enabled validation run with manual thresholds

The validation artifact currently kept in this repository corresponds to the `rows-1-5032` style run. A representative command is:

```bash
python run_rebalancer_with_llm.py "Data/Validation_set/validation_set.csv" \
  --enable-llm \
  --enable-llm-species-bridge \
  --synrbl-confidence-threshold 0.8 \
  --score-threshold 0.8 \
  --retry-confidence-threshold 0.8 \
  --species-bridge-confidence-threshold 0.8 \
  --start-row 1 \
  --end-row 5032
```

These threshold values are exposed as command-line arguments in `run_rebalancer_with_llm.py`, so they can be manually adjusted to match the desired confidence policy for a specific experiment.

### Useful options

```text
--output-dir                         Output directory; defaults to input file directory
--synrbl-confidence-threshold        Confidence threshold for native SynRBL
--score-threshold                    Score threshold for LLM candidate scoring/generation policy
--retry-confidence-threshold         Threshold for fallback retry decisions
--species-bridge-confidence-threshold Threshold for bridge acceptance
--enable-llm-thinking                Enable Kimi thinking mode
--sep                                Custom separator for CSV/TSV/TXT input
```

## Outputs

For each run, ChemBridge creates a run folder such as `rows-1-500` or `head-100` under the selected output directory.

Main output files:

- `synrbl_results_original_stage.json/csv`: first deterministic SynRBL-stage decisions.
- `synrbl_results_bridge_stage.json/csv`: bridge-stage decisions for records that entered the species bridge, including bridge-related trace fields.
- `synrbl_results_with_llm.json/csv`: final formal workflow output, including bridge/fallback route information when applicable.
- `synrbl_failed_cases.json` and `synrbl_failed_cases_flat.csv`: unresolved cases.
- `workflow_statistics.csv`: aggregate route/status statistics.
- `accuracy_comparison.json` and `accuracy_comparison_detail.csv`: comparison against the expected reaction column, when available.

Important final-result columns include:

- `formal_output_reaction`: final accepted reaction SMILES.
- `workflow_confidence`: workflow-level confidence/priority score.
- `workflow_source`: final source route, such as `SynRBL`, `bridge`, `fallback`, or `prebalance`.
- `success` / `output_status`: whether a valid balanced final output was produced.
- `stage1_case` and `stage2_case`: branch labels useful for debugging.
- `internal_candidate_1_reaction` and `internal_candidate_2_reaction`: retained low-confidence candidates.
- `bridge_raw_output_reaction` and `bridge_accepted_output_reaction`: bridge diagnostics.

## Benchmark and Analysis

If your dataset contains a reference column such as `expected_reaction`, ChemBridge automatically writes workflow statistics and accuracy comparison reports after each run.

Additional validation helpers are available under `Data/Validation_set/`, including scripts for merging and comparing experiment outputs.

The published validation artifact is `Data/Validation_set/rows-1-5032/`. It intentionally keeps the original-stage, bridge-stage, final-result, failed-case, workflow-statistics, and accuracy-comparison files so that bridge and fallback traces are available for inspection.

## Attribution, Citation, and License

ChemBridge-specific workflow modifications are authored by Peixin Liu. In this repository, SynRBL provides the deterministic rule-based and MCS-based rebalancing components. ChemBridge provides the staged workflow framework that coordinates preprocessing, SynRBL invocation, LLM species bridging, fallback generation, candidate auditing, confidence-based routing, and result reporting. Some SynRBL-facing interfaces and output formats were adjusted so that deterministic outputs such as confidence and solved-route information can be used consistently inside the ChemBridge workflow; these changes are intended as integration changes rather than changes to the underlying deterministic rebalancing idea.

Please retain attribution to the original SynRBL authors and cite their paper when using the SynRBL-derived deterministic components:

```bibtex
@Article{Phan2024,
  author={Phan, Tieu-Long and Weinbauer, Klaus and G{\"a}rtner, Thomas and Merkle, Daniel and Andersen, Jakob L. and Fagerberg, Rolf and Stadler, Peter F.},
  title={Reaction rebalancing: a novel approach to curating reaction databases},
  journal={Journal of Cheminformatics},
  year={2024},
  volume={16},
  number={1},
  pages={82},
  doi={10.1186/s13321-024-00875-4},
  url={https://doi.org/10.1186/s13321-024-00875-4}
}
```

This project currently retains the MIT License file from SynRBL. If you publish ChemBridge as a derivative work, keep the original copyright/license notice and add your own copyright notice for your modifications where appropriate.
