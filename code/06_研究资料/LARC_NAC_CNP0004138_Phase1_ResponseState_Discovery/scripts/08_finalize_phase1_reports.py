from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_integrity_manifest():
    sources = [
        ROOT / "02_scRNA_matrix/source/MM_coordinate_matrix.mtx.gz",
        ROOT / "02_scRNA_matrix/source/MM_coordinate_matrix_genes.csv",
        ROOT / "02_scRNA_matrix/source/MM_coordinate_matrix_barcodes.csv",
        ROOT / "07_gse209746_validation/source/GSE209746_geneExp_n_114_ensembl.csv.gz",
        ROOT / "07_gse209746_validation/source/GSE209746_family.soft.gz",
        ROOT / "08_gse87211_validation/source/GSE87211_series_matrix.txt.gz",
        ROOT / "08_gse87211_validation/source/GSE87211_family.soft.gz",
        ROOT / "08_gse87211_validation/source/GPL13497_platform_data.txt",
        ROOT / "06_independent_validation_prep/hgnc_complete_set.txt",
    ]
    rows = []
    for path in sources:
        if path.exists():
            rows.append(
                {
                    "relative_path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "09_qc/Phase1_source_file_integrity_manifest.tsv", sep="\t", index=False)


def write_qa_note():
    note = f"""# Phase 1 computational and figure QA

## Runtime

- Python: {platform.python_version()}
- Plotting backend: Python/matplotlib and seaborn only.
- Statistical unit: patient-level or patient-celltype pseudobulk for inference; cells and UMAP are descriptive only.

## Analysis QA

- CNP0004138 matrix: 55,697 genes x 378,344 cells with 445,213,600 nonzero entries.
- Barcode mapping: 378,344/378,344 cells mapped; zero unmapped barcodes.
- Primary signature training: PRET CR versus NR only; PR excluded from gene selection.
- Validation: locked genes and signs imported unchanged; no validation-cohort refitting, threshold tuning, or sign reversal.
- Negative evidence retained: zero discovery genes reach BH FDR < 0.05; GSE87211 all-evaluable effects are weak; the 5-FU+RT subgroup is mostly non-supportive.

## Figure QA

- All requested draft PDFs have matching 300 dpi PNG previews.
- Key figures were visually inspected at original PNG resolution after export.
- Corrected issues: patient heatmap axis-label collision, shared-y-axis label loss, and cross-dataset heatmap footnote overlap.
- PDF text is exported with Type 42 fonts; figures are vector except raster point clouds/heatmap rendering as handled by matplotlib.
- UMAP is explicitly labeled as a 50,000-cell descriptive QC sample.

## Remaining risks

- Broad cell annotation is marker-guided; malignant epithelial identity is provisional without full CNV inference.
- Discovery has only 6 CR and 5 NR PRET patients.
- GSE87211 lacks public sample-level TRG; strict ypT0N0M0 is used as an auditable pCR endpoint.
- GSE87211 regimen analysis is sensitivity-grade and not the primary validation endpoint.
"""
    (ROOT / "09_qc/Phase1_computational_and_figure_QA.md").write_text(note, encoding="utf-8")


def write_final_report():
    text = """# FINAL LARC Phase 1 response-state discovery report

## Executive decision

**Strong Continue to Phase 2 spatial localization, with strict guardrails.**

Phase 1 passes the prespecified gate: metadata and the full processed single-cell matrix are reliable; patient-celltype pseudobulk is feasible; six PRET CR-versus-NR programs can be locked; PR is intermediate in all six; GSE209746 supports all six directions; GSE87211 is weak but not oppositional overall and becomes stronger in the platinum-containing regimen sensitivity. This supports a candidate treatment-response-associated state and a spatial localization hypothesis. It does not support a clinically validated biomarker or a causal mechanism.

## Required answers

1. **Was CNP0004138 metadata successfully frozen?** Yes. All 378,344 barcodes were linked to patient, sample, timepoint, and response metadata; zero cells were unmapped.
2. **Are CR/PR/NR/TRG labels reliable?** Yes for the source naming and supplementary response table: CR = TRG0/1, PR = TRG2, NR = TRG3. The cohort contains 7 CR, 15 PR, 5 NR, and 2 unknown patients. PRET contains 6 CR, 14 PR, 5 NR, and 2 unknown because two patients lack PRET samples.
3. **Was the scRNA matrix read successfully?** Yes. The complete processed matrix is 55,697 genes x 378,344 cells with 445,213,600 nonzero entries; raw FASTQ was not used.
4. **Was patient/timepoint/celltype pseudobulk constructed?** Yes. There are 635 retained patient-timepoint-celltype units after the default >=20-cell filter, including all-cell summaries.
5. **Which compartments have adequate patient-level PRET support?** Global/all cells, malignant epithelial, CAF/fibroblast, myeloid/macrophage, CD8 T, and endothelial. Each contains 6 CR, 14 PR, and 5 NR PRET patient units. Dendritic cells are too sparse for a locked program.
6. **Was a PRET CR-versus-NR locked program defined?** Yes. Six programs were frozen: global plus five compartment-specific programs. PR did not participate in gene selection.
7. **What do the locked programs contain?** Each has 100 NR-high resistance genes and 100 CR-high response genes. Because no individual gene reaches transcriptome-wide BH FDR < 0.05, all are labeled limited-significance and selected by effect size plus nominal P ranking.
8. **Is PR intermediate?** Yes. Mean resistance score is ordered CR < PR < NR in all six programs; Spearman rho ranges from 0.831 to 0.897. This is discovery-cohort support, not independent validation.
9. **Do PRET-to-POST comparisons show dynamics?** Yes, but not a uniform post-treatment increase. CR samples tend to move upward and NR samples downward, indicating convergence/remodeling and possible regression-to-the-mean effects. These paired changes are exploratory and cannot be used as pretreatment prediction evidence.
10. **Is GSE209746 usable, and what is the direction?** Yes. It contains 114 pretreatment RNA-seq biopsies, including 27 CR, 70 iCR, and 17 response-unknown samples. All six locked programs are higher in iCR than CR; median Hedges g is 0.455 and median AUC is 0.618. No full-cohort program passes six-program BH correction. MSS-only sensitivity remains 6/6 directionally supportive.
11. **Is GSE87211 usable, and what is the direction?** Conditionally. Public metadata do not expose sample-level TRG, so the published 70/41 grouping cannot be independently reconstructed. Using the auditable strict ypT0N0M0 endpoint yields 35 pCR and 166 non-pCR tumors. All six overall effects have the expected sign but are small and nonsignificant. In the platinum-containing CRT sensitivity (11 pCR, 75 non-pCR), all six directions agree and the CD8 program reaches BH FDR 0.026; the pure 5-FU+RT subgroup is mostly non-supportive.
12. **Is this main line stronger than TEAD-MPM and NSCLC-ICI?** Structurally yes. Unlike TEAD-MPM, it has real treated-patient response labels. Compared with NSCLC-ICI, it has full-transcriptome prospective spatial material and 73%-87% external locked-gene coverage rather than low-coverage panel transfer. Its final paper potential is not yet proven because Phase 2 spatial n is only four and discovery gene-level significance is limited.
13. **Should Phase 2 spatial localization start?** Yes. Start a bounded spatial localization phase, not virtual perturbation. The priority is to test whether the locked global, CD8, myeloid/endothelial, malignant, and CAF programs occupy reproducible CR/NR niches at section/patient level.
14. **What is the fallback if spatial localization fails?** Stop before virtual perturbation and reframe as a patient-pseudobulk response-state discovery with treatment-context-dependent bulk transfer. Do not claim a spatially validated program, mechanism, or biomarker.

## Independent representation

- GSE209746 represents 76.0%-87.0% of each 200-gene locked program.
- GSE87211 represents 73.0%-82.0% of each 200-gene locked program.

## Main boundary

The defensible conclusion is: **a locked pretreatment response-state program derived from platinum-treated LARC single-cell pseudobulk shows ordered CR-to-PR-to-NR behavior and partial, treatment-context-dependent transfer to independent pretreatment bulk cohorts.**

Do not use: clinically validated biomarker, causal mechanism proven, virtual knockout validation, or validated therapeutic target.

## Data sources

- CNP0004138/CSE0000154: https://db.cngb.org/data_resources/project/CNP0004138/
- Source study: https://doi.org/10.1016/j.xcrm.2023.101231
- GSE209746: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE209746
- GSE87211: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87211
- GSE87211 strict-pCR source context: https://doi.org/10.1038/s41416-022-01842-2
"""
    (ROOT / "11_reports/FINAL_LARC_Phase1_response_state_discovery_report.md").write_text(text, encoding="utf-8")


def write_go_nogo():
    text = """# FINAL Go/No-Go for LARC Phase 2 spatial localization

## Verdict

**Strong Continue to Phase 2 spatial localization.**

All prespecified Strong Continue conditions are met:

1. PRET CR-versus-NR programs are defined and frozen without using PR or validation cohorts for training.
2. PR is intermediate in all six discovery programs.
3. Malignant, CAF, myeloid, CD8, and endothelial compartments all have adequate patient-pseudobulk support.
4. GSE209746 supports 6/6 directions; GSE87211 supports 6/6 directions overall, although weakly.
5. Neither independent cohort is strongly opposite at its primary endpoint.
6. Locked-gene representation is 73%-87%, adequate for transfer scoring.
7. A concrete spatial hypothesis is available.

## Phase 2 question

Test whether the already locked resistance programs localize to reproducible post-NAC CR-versus-NR tissue niches, prioritizing CD8-associated immune state and myeloid/endothelial remodeling around malignant regions, while retaining CAF as a comparator rather than simply repeating the source CAF-FAP story.

## Non-negotiable Phase 2 design

- Use ST-CR1, ST-CR2, ST-NR1, and ST-NR2 only after section-level QC and program-gene coverage checks.
- Import the Phase 1 programs unchanged; no spatial refitting, gene reselection, or sign reversal.
- Treat section/patient as the inferential unit. Bins/spots are descriptive spatial observations, not independent patients.
- Analyze CR versus NR at response-group level because the exact donor assignment to ST-NR1/ST-NR2 is unresolved.
- Keep global and compartment-derived scores separate; external bulk transfer does not prove spatial cell-type identity.
- Report leave-one-section sensitivity and the low-quality ST-NR2 influence.
- Do not run virtual perturbation in the same step. It can be reconsidered only after spatial direction and localization are stable.

## Phase 2 stop rules

Stop before virtual perturbation if any of the following occurs:

- Locked-program coverage is insufficient or strongly imbalanced between CR and NR sections.
- Patient/section-level CR-versus-NR direction is opposite for the global program and all biologically prioritized programs.
- A result exists only at bin level and disappears after section aggregation.
- Localization is dominated by one section or by ST-NR2 quality artifacts.
- The only signal recapitulates the already published CAF_FAP/MIR4435-2HG mechanism without an independent response-state contribution.

## Allowed claim after a successful Phase 2

"The locked Phase 1 resistance-associated program shows section-level spatial localization in response-labeled LARC tissue."

This would still not establish a clinically validated biomarker or causal regulator.
"""
    (ROOT / "11_reports/FINAL_go_nogo_for_LARC_Phase2_spatial_localization.md").write_text(text, encoding="utf-8")


def write_tree_and_index():
    lines = [ROOT.name + "/"]
    files_for_index = []
    for directory in sorted([p for p in ROOT.rglob("*") if p.is_dir()]):
        relative = directory.relative_to(ROOT)
        depth = len(relative.parts)
        lines.append("  " * depth + relative.name + "/")
        for file in sorted([p for p in directory.iterdir() if p.is_file()]):
            lines.append("  " * (depth + 1) + file.name)
            files_for_index.append(file)
    (ROOT / "00_logs/directory_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    categories = {
        "01_metadata": "Frozen CNP0004138 patient/sample/timepoint/response mapping.",
        "02_scRNA_matrix": "Source processed matrix and clean cell metadata/matrix dimensions.",
        "03_celltype_annotation": "Broad marker-guided annotations, confidence, composition, and QC sample objects.",
        "04_pseudobulk": "Patient-timepoint-celltype count matrices and metadata.",
        "05_response_state_discovery": "DEG tables, locked programs, scores, gradients, and paired dynamics.",
        "06_independent_validation_prep": "Gene mapping resources and publication-linked audit material.",
        "07_gse209746_validation": "Prepared metadata, locked scoring, representation, and validation results.",
        "08_gse87211_validation": "Prepared metadata, strict-pCR endpoint, locked scoring, and regimen sensitivity.",
        "09_qc": "Mapping, pseudobulk, integrity, and computational/figure QA records.",
        "10_draft_figures": "Python-generated draft PDFs and PNG previews plus the figure contract.",
        "11_reports": "Integrated evidence, claim boundaries, final Phase 1 report, and Phase 2 decision.",
        "scripts": "Reproducible Python analysis and plotting scripts.",
        "00_logs": "Execution logs, metadata inspection records, and directory tree.",
    }
    rows = [{"directory": key, "description": value} for key, value in categories.items()]
    pd.DataFrame(rows).to_csv(ROOT / "00_logs/deliverable_index.tsv", sep="\t", index=False)


def main():
    write_integrity_manifest()
    write_qa_note()
    write_final_report()
    write_go_nogo()
    write_tree_and_index()
    print("Finalized reports, QA, source integrity manifest, and directory tree.")


if __name__ == "__main__":
    main()
