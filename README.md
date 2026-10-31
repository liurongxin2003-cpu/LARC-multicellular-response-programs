# LARC multicellular response programs

Reproducibility code and aggregate source data for CBAC-D-26-05807, a retrospective study of response-associated programs in locally advanced rectal cancer. This is the 2026-10-31 revision archive (version 1.1.0). Frozen definitions and original discovery findings are retained alongside selection-aware, incremental-model and technical-sensitivity results.

## Actual release status

GitHub repository: https://github.com/liurongxin2003-cpu/LARC-multicellular-response-programs. This archive is version 1.1.0. A Zenodo record and DOI have not yet been assigned; no DOI is claimed.

## Included evidence

- `frozen_programs/`: six fixed bidirectional gene lists and aggregate selection statistics.
- `results/R1_R12/`: selection-aware summaries/null statistics, correlations, overlap, scaling and four-section spatial checks.
- `results/M4_M5/`: restrained incremental models and observed-confounding sensitivity.
- `results/R15_R16/`: regulator stability, imposed-score diagnostics, LR source/count sensitivity and complete-source spatial co-detection.
- `results/R17/`: aggregate LR and fixed-reference program contrasts before/after source-library Scrublet prediction exclusion.
- `code/`: original and revision scripts with an explicit source-path mapping; C# utilities only stream integer data.

## Source data and access

Obtain original inputs from CNGBdb CNP0004138 / CSE0000154 (https://db.cngb.org/data_resources/project/CNP0004138/), GEO GSE209746 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209746) and GEO GSE87211 (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87211), under the originating repositories' terms. The locally saved CNGB project response marks select_control as Public; project-specific derivative redistribution terms were not established. This is not a claim that the source project is controlled-access.

Raw third-party matrices and CNP patient-, sample- and cell-level derivatives are not redistributed here. Source data must be obtained separately; this aggregate archive alone cannot rerun every individual-data analysis. Exclusions do not imply zero values or negative findings. Patient units govern response comparisons; the four spatial sections are not four verified independent donors. Table R17 uses source barcode library labels without an independently verified capture manifest; predictions are not doublet ground truth.

## Reproduction

Use Python 3.14.6 for the recorded revision environment. Install `requirements.txt` into an isolated environment. `python tools/smoke_test.py` checks syntax, all public tables, missing-value parsing and frozen aggregate identities without requiring original matrices; it does not claim full pipeline reproduction.

To reconstruct a project layout, run `python tools/restore_project_layout.py /path/to/local_project`. The tool copies only supplied scripts/results into their original relative locations. Acquire source inputs under `06_研究资料` using the source locations in the metadata manifest. Legacy Phase1–3 scripts retain original configuration constants and require setting ROOT/SOURCE to those locally acquired inputs before execution. Modern revision scripts use their source project-relative paths. Do not run a reporting/document builder against the original manuscript as part of reproduction.

Order: original Phase1–3 generation → selection-aware scripts in 07_返修_20261028/review_code → M4_M5_revision common/02/03/04 → CBC_revision_next extraction/03/04/04c → CBC_final_minor_revision 01/02/03/04. The exact M1 conditional null uses all 462 CR/NR assignments with PR identities fixed. M4's primary endpoint is held-out log loss; GSE87211 secondary leave-pair-out ranking improvement is retained. R17 preserves original lists/reference scaling and does not repeat M1–M4. Seed 20261101 is an arbitrary deterministic identifier, not a registration date.

## Interpretation and units

Scores are unitless; NR-minus-CR contrasts compare patient-level means. `g` denotes Hedges g only where named, not raw score differences. Bootstrap bounds are conditional/descriptive where specified in the original source tables. A zero prior-only perturbation with no retained edges is absence of graph support, not no biological activity. LR co-detection and modeled perturbations do not demonstrate physical communication or therapeutic benefit. Weak results, zeros and missing values are preserved.

## Licenses and attribution

Original author-written analysis code is MIT-licensed (`LICENSE`). Author-created aggregate results and documentation are CC BY 4.0 (`DATA_LICENSE.md`). Neither license grants redistribution of third-party raw data, external software or patient/cell derivatives. Cite the originating studies and this versioned archive; no DOI is claimed until assigned.
