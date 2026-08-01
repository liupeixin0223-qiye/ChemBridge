# ChemBridge: A Five-Level Cascade Workflow for Chemical Reaction Balancing and Completion

ChemBridge is a five-level cascade workflow for balancing and completing incomplete chemical reaction equations. It integrates deterministic graph-matching algorithms (inherited from SynRBL) with large language model (LLM)-assisted strategy selection and generative repair, coordinated through confidence-driven hierarchical routing.

> **Contribution and attribution note.** The repository name is ChemBridge, while the Python import/package name remains `synrbl` for compatibility with the modified SynRBL codebase. SynRBL provides the deterministic rule-based and MCS-based rebalancing components. ChemBridge contributes the five-level cascade architecture, including progressive voting MCS selection, global exhaustive allocation, multi-fragment merging, template matching, Bridge LLM strategy selection, Fallback LLM generative completion with error-correction retry, and the confidence-driven routing that coordinates all levels. Please cite the original SynRBL work when using its deterministic rebalancing components.

## Table of Contents

- [Overview](#overview)
- [Five-Level Cascade Architecture](#five-level-cascade-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [LLM Configuration](#llm-configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [Benchmark and Analysis](#benchmark-and-analysis)
- [Attribution, Citation, and License](#attribution-citation-and-license)

## Overview

ChemBridge addresses the prevalent issue of missing co-reactants or by-products in chemical reaction databases. The workflow implements a five-level cascade in which each level processes only reactions that all preceding levels have failed to resolve, and the outputs of different levels do not overlap. This sequential structure ensures that deterministic methods handle the majority of reactions with guaranteed atom conservation, while LLM-assisted modules are invoked only for the most challenging cases.

Key algorithmic improvements over the original SynRBL include:

1. **Progressive voting MCS selection**: A two-layer voting mechanism (majority vote + rank-based weighted scoring) replaces the original atom-count-only comparison for selecting among multiple MCS search results.
2. **Global exhaustive allocation (Path B)**: Cartesian-product exhaustive enumeration of all valid reactant–product allocation combinations, activated when the greedy Path A fails, with direction swap optimization and safety mechanisms.
3. **Multi-fragment merging**: Extension from two-fragment to four-fragment merging, with radical retention, a three-layer bond-type decision engine, and a three-stage merging workflow.
4. **Template matching**: A structured knowledge base of 303 reaction templates with backtracking traversal assignment and three-outcome classification (Full Match / Subset Match / Mismatch).
5. **LLM role redesign**: Bridge LLM repositioned from an unconstrained species generator to a constrained strategy selector (output space: A/B/C); Fallback LLM augmented with an error-correction retry mechanism (up to 2 retries with recalculated atom deficit feedback).

## Five-Level Cascade Architecture

```
Input reaction
    │
    ▼
┌─────────────────────────────────────────────┐
│ Level 1: Greedy Graph Matching (Path A)     │
│   MCS-based matching + progressive voting   │
└─────────────────┬───────────────────────────┘
                  │ (failed / low confidence)
                  ▼
┌─────────────────────────────────────────────┐
│ Level 2: Exhaustive Global Matching (Path B)│
│   Cartesian-product exhaustive allocation   │
└─────────────────┬───────────────────────────┘
                  │ (failed / low confidence)
                  ▼
┌─────────────────────────────────────────────┐
│ Level 3: Template Matching                  │
│   303 templates + backtracking assignment   │
└─────────────────┬───────────────────────────┘
                  │ (unresolved / type inference)
                  ▼
┌─────────────────────────────────────────────┐
│ Level 4: Bridge LLM Adjudication            │
│   Strategy selector (A / B / C)             │
└─────────────────┬───────────────────────────┘
                  │ (C selected / validation failed)
                  ▼
┌─────────────────────────────────────────────┐
│ Level 5: Fallback LLM Generative Completion │
│   End-to-end repair + error-correction retry│
└─────────────────────────────────────────────┘
                  │
                  ▼
            Final output
```

The main entry point is:

```bash
python run_rebalancer_with_llm.py <input_file> [options]
```

## Repository Structure

```text
ChemBridge/
├── run_rebalancer_with_llm.py            # Main entry point: five-level cascade workflow
├── per_dataset_benchmark.py              # Per-dataset benchmark script
├── synrbl/                               # Core package (SynRBL-compatible + ChemBridge extensions)
│   ├── __init__.py / __main__.py         # Package init and CLI entry (python -m synrbl)
│   ├── balancing.py                      # Core deterministic Balancer
│   ├── mcs_search.py                     # Level 1: Progressive voting MCS search
│   ├── exhaustive_allocation.py          # Level 2: Global exhaustive allocation (Path B)
│   ├── template_matching.py              # Level 3: Template matching (303 built-in templates)
│   ├── bridge_strategy_selector.py       # Level 4: Bridge LLM strategy selector
│   ├── llm_fallback_postprocessor.py     # Level 5: Fallback LLM with error-correction retry
│   ├── llm_postprocessor.py              # LLM post-processing pipeline
│   ├── llm_species_bridge.py             # LLM species bridge utilities
│   ├── unified_decision.py               # Confidence-driven cascade routing
│   ├── confidence_prediction.py          # XGBoost confidence prediction
│   ├── evaluation_utils.py               # Evaluation utilities
│   ├── preprocess.py / postprocess.py    # Reaction preprocessing / postprocessing
│   ├── rule_based.py                     # Rule-based imputation
│   ├── rsmi_utils.py                     # Reaction SMILES utilities
│   ├── llm/                              # LLM client and prompt modules
│   │   ├── client.py                     #   Moonshot/Kimi API client
│   │   ├── bridge_strategy_client.py     #   Bridge strategy API client
│   │   ├── bridge_strategy_prompts.py    #   Bridge strategy prompts
│   │   ├── fallback_client.py            #   Fallback API client
│   │   ├── fallback_prompts.py           #   Fallback prompts (v3, final version)
│   │   ├── species_client.py             #   Species bridge API client
│   │   ├── species_prompts.py            #   Species bridge prompts
│   │   ├── models.py                     #   Data models
│   │   └── prompts.py                    #   General prompt utilities
│   ├── SynMCSImputer/                    # MCS-based imputation (merge, rules, structure)
│   ├── SynRuleImputer/                   # Rule-based imputation
│   ├── SynChemImputer/                   # Chemical post-processing
│   ├── SynProcessor/                     # Reaction SMILES preprocessing
│   ├── SynAnalysis/                      # Analysis and scoring (XGBoost model)
│   ├── SynUtils/                         # General utilities
│   ├── SynVis/                           # Visualization
│   └── SynCmd/                           # CLI commands
├── prompt/                               # Prompt reference files
│   └── Bridge_LLM_now.py                #   Current Bridge LLM prompt
├── llm/                                  # Ablation experiment (Appendix A, Section A2)
│   ├── ablation_fallback_llm.py          #   Direct LLM invocation ablation script
│   ├── ablation_results.csv              #   Per-reaction ablation results (322 reactions)
│   ├── ablation_summary.txt              #   Ablation summary
│   └── ablation_audit_report.txt         #   Ablation audit report
├── Data/
│   ├── Raw_data/                         # Original source data (Golden, Jaworski, USPTO)
│   ├── Rules/                            # Automated rules
│   ├── Testcase/                         # Test cases
│   └── Validation_set/                   # Validation data and workflow outputs
│       ├── validation_set.csv            #   Original test set (from SynRBL)
│       ├── validation_set_fixed.csv      #   Corrected test set (updated expected reactions)
│       ├── validation_set_fixed_LLM.csv  #   LLM subset (322 reactions)
│       └── rows-1-5032/                  #   Full workflow output (published validation artifact)
│           ├── pipeline_status.csv       #     92-column per-reaction processing trace
│           ├── accuracy_comparison_detail.csv  # Per-reaction accuracy comparison
│           ├── balanced_reactions.csv    #     Successfully balanced reactions
│           ├── failed_reactions.csv      #     Failed reactions
│           └── workflow_statistics.csv   #     Aggregate statistics
├── Pipeline/Validation/Analysis/         # Validation analysis notebooks
├── Scripts/                              # Utility scripts
├── Docs/                                 # Documentation and paper figures
├── Test/                                 # Unit tests
├── pyproject.toml                        # Python package metadata
├── requirements.txt                      # Runtime dependencies
├── LICENSE                               # MIT License (inherited from SynRBL)
├── CITATION.cff                          # Citation metadata
└── README.md
```

The validation artifact folder `Data/Validation_set/rows-1-5032/` is intentionally retained. Other generated `rows-*` output folders are ignored by `.gitignore`.

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

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Core dependencies:

- `synkit` >= 1.1.2
- `rdkit` >= 2026.3.1
- `pandas` >= 3.0.2
- `xgboost` >= 2.0.3
- `imbalanced_learn` >= 0.14.0
- `scikit_learn` == 1.7.2
- `reportlab` >= 4.1.0
- `fgutils` >= 0.1.3

The development environment used Python 3.11.15 with RDKit 2026.3.1, SynKit 1.3, pandas 3.0.2, scikit-learn 1.7.2, imbalanced-learn 0.14.1, fgutils 0.2.3, reportlab 4.4.10, and xgboost 3.0.0.

## LLM Configuration

ChemBridge uses a Moonshot/Kimi-compatible chat-completions endpoint.

Set your API key before running LLM-enabled workflows:

```powershell
$env:MOONSHOT_API_KEY="your_api_key_here"
```

Default LLM parameters:

- API key environment variable: `MOONSHOT_API_KEY`
- Base URL: `https://api.moonshot.cn/v1/chat/completions`
- Model: `kimi-k2.5`
- Temperature: `0.6`

Override from the command line:

```bash
python run_rebalancer_with_llm.py data.csv \
  --llm-api-key-env MOONSHOT_API_KEY \
  --llm-base-url https://api.moonshot.cn/v1/chat/completions \
  --llm-generate-model kimi-k2.5
```

Do not commit API keys or `.env` files to GitHub.

## Usage

### Input format

Input files may be `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, or `.json`.

Default column names:

- reaction column: `reactions`
- ID column: `R-id`
- expected/reference reaction column: `expected_reaction`

Common fallback column names (`reaction`, `rxn`, `rsmi`, `id`, `ID`) are resolved automatically.

### Run on the validation set

```bash
python run_rebalancer_with_llm.py Data/Validation_set/validation_set.csv \
  --reaction-col reactions \
  --id-col R-id \
  --expected-col expected_reaction
```

### Full LLM-enabled validation run

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

### Process a subset

```bash
python run_rebalancer_with_llm.py Data/Validation_set/validation_set.csv --head 100
```

### Useful options

```text
--output-dir                          Output directory
--synrbl-confidence-threshold         Confidence threshold for SynRBL
--score-threshold                     Score threshold for LLM candidate policy
--retry-confidence-threshold          Threshold for fallback retry decisions
--species-bridge-confidence-threshold Threshold for bridge acceptance
--enable-llm-thinking                 Enable Kimi thinking mode
--sep                                 Custom separator for CSV/TSV/TXT input
```

## Outputs

For each run, ChemBridge creates a run folder (e.g., `rows-1-5032`) under the output directory.

Main output files:

- `pipeline_status.csv / .json`: Complete per-reaction processing trace (92 columns), including workflow route, confidence scores, bridge/fallback diagnostics, and MCS details.
- `accuracy_comparison_detail.csv`: Per-reaction comparison against the expected reaction column.
- `balanced_reactions.csv / .json`: Successfully balanced reactions.
- `failed_reactions.csv / .json`: Unresolved reactions.
- `workflow_statistics.csv`: Aggregate route/status statistics.

Key result columns in `pipeline_status.csv`:

- `formal_output_reaction`: Final accepted reaction SMILES.
- `workflow_source`: Final source route (`SynRBL`, `Template`, `Bridge`, `Fallback`, `Prebalance`).
- `workflow_confidence`: Workflow-level confidence score.
- `success` / `output_status`: Whether a valid balanced output was produced.
- `stage1_case` / `stage2_case`: Branch labels for debugging.
- `mcs_vote_method`: MCS selection method used (`weighted_ranking`).
- `bridge_selected_strategy`: Bridge LLM decision (A/B/C).
- `fallback_case`: Fallback trigger reason.

## Benchmark and Analysis

On the 5,032-reaction test set (derived from the SynRBL validation dataset), the ChemBridge workflow achieves:

- **Atom conservation (Success)**: 4,965 / 5,032 (98.7%)
- **Exact topological matching (Accuracy)**: 4,850 / 5,032 (96.4%)
- **Cumulative actual matches** (including equivalent balancing): 4,914 / 5,032 (97.7%)

The original SynRBL algorithm achieves 4,617 / 5,032 (91.8%) Success and 4,601 / 5,032 (91.4%) Accuracy under the same confidence threshold (0.8).

The ablation experiment script (`llm/ablation_fallback_llm.py`) and its results (`llm/ablation_results.csv`) are included for reproducibility.

## Attribution, Citation, and License

ChemBridge-specific workflow modifications are authored by Peixin Liu. SynRBL provides the deterministic rule-based and MCS-based rebalancing components. ChemBridge provides the five-level cascade architecture, progressive voting MCS selection, global exhaustive allocation, multi-fragment merging, template matching, Bridge LLM strategy selection, Fallback LLM generative completion with error-correction retry, and confidence-driven hierarchical routing.

Please cite the original SynRBL authors when using the SynRBL-derived deterministic components:

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

This project retains the MIT License from SynRBL. If you publish ChemBridge as a derivative work, keep the original copyright/license notice and add your own copyright notice for your modifications.
