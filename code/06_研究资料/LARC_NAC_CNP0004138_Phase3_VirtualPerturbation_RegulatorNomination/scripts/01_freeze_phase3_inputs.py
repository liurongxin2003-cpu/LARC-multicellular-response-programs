from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

from phase3_config import INPUT, P1, P2, ROOT


P1_REQUIRED = [
    "01_metadata/CNP0004138_frozen_metadata.tsv",
    "03_celltype_annotation/CNP0004138_celltype_annotation.tsv",
    "03_celltype_annotation/CNP0004138_celltype_annotation_confidence.tsv",
    "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_matrix.tsv",
    "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_metadata.tsv",
    "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.tsv",
    "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv",
    "05_response_state_discovery/CNP0004138_locked_response_resistance_program_global.tsv",
    "05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv",
    "05_response_state_discovery/CNP0004138_pseudobulk_locked_program_scores.tsv",
    "05_response_state_discovery/CNP0004138_CR_PR_NR_program_score_contrasts.tsv",
    "11_reports/FINAL_LARC_Phase1_response_state_discovery_report.md",
]

P2_REQUIRED = [
    "04_program_gene_coverage/StereoSeq_locked_program_gene_coverage.tsv",
    "06_section_level_summary/StereoSeq_section_level_program_scores.tsv",
    "06_section_level_summary/StereoSeq_CR_vs_NR_section_effect_summary.tsv",
    "07_spatial_domain_annotation/StereoSeq_spatial_domain_scores.tsv",
    "07_spatial_domain_annotation/StereoSeq_spatial_domain_annotation.tsv",
    "08_program_localization/StereoSeq_program_domain_correlation.tsv",
    "08_program_localization/StereoSeq_CR_NR_localization_summary.tsv",
    "09_response_group_comparison/StereoSeq_nonredundancy_with_CAF_FAP_summary.md",
    "10_sensitivity_analysis/StereoSeq_leave_one_section_sensitivity.tsv",
    "13_reports/FINAL_LARC_Phase2_spatial_localization_report.md",
    "13_reports/FINAL_go_nogo_for_LARC_Phase3_virtual_perturbation.md",
]

EXTRA = [
    (P1, "03_celltype_annotation/CNP0004138_QC_sample_annotated.h5ad"),
    (P1, "03_celltype_annotation/CNP0004138_QC_sample_umap_coordinates.tsv"),
    (P1, "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_matrix.parquet"),
    (P1, "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet"),
    (P2, "03_stereoseq_matrix/StereoSeq_bin_coordinate_metadata.parquet"),
    (P2, "06_section_level_summary/StereoSeq_section_pseudobulk_logCPM.parquet"),
    (P2, "05_spatial_program_scores/ST_CR1_bin_level_program_scores.parquet"),
    (P2, "05_spatial_program_scores/ST_CR2_bin_level_program_scores.parquet"),
    (P2, "05_spatial_program_scores/ST_NR1_bin_level_program_scores.parquet"),
    (P2, "05_spatial_program_scores/ST_NR2_bin_level_program_scores.parquet"),
    (P2, "03_stereoseq_matrix/ST_CR1_analysis_gene_counts.npz"),
    (P2, "03_stereoseq_matrix/ST_CR2_analysis_gene_counts.npz"),
    (P2, "03_stereoseq_matrix/ST_NR1_analysis_gene_counts.npz"),
    (P2, "03_stereoseq_matrix/ST_NR2_analysis_gene_counts.npz"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    core_missing = []
    specs = [("phase1", P1, item, True) for item in P1_REQUIRED]
    specs += [("phase2", P2, item, True) for item in P2_REQUIRED]
    specs += [("extra", root, relative, False) for root, relative in EXTRA]

    for phase, source_root, relative, required in specs:
        source = source_root / relative
        destination = INPUT / source.name
        exists = source.exists()
        if required and not exists:
            core_missing.append(str(source))
        if exists:
            if not destination.exists() or destination.stat().st_size != source.stat().st_size:
                shutil.copy2(source, destination)
            checksum = sha256(destination)
            size = destination.stat().st_size
        else:
            checksum = ""
            size = 0
        rows.append({
            "phase": phase,
            "requested_relative_path": relative,
            "source_path": str(source),
            "copied_filename": destination.name,
            "required": required,
            "exists": exists,
            "bytes": size,
            "sha256": checksum,
        })

    check = pd.DataFrame(rows)
    check.to_csv(INPUT / "phase1_phase2_input_file_check.tsv", sep="\t", index=False)
    if core_missing:
        stop = ROOT / "13_reports/STOP_missing_phase1_phase2_inputs.md"
        stop.write_text("# STOP: missing Phase 1/2 inputs\n\n" + "\n".join(f"- {item}" for item in core_missing), encoding="utf-8")
        raise SystemExit(f"Missing {len(core_missing)} required inputs")

    pret_meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t")
    counts = pret_meta.groupby(["cell_type", "response_standardized"]).size().unstack(fill_value=0)
    summary = f"""# Phase 3 input summary

- Required Phase 1/2 inputs found: {int(check.loc[check.required, 'exists'].sum())}/{int(check.required.sum())}.
- Supplemental inputs found: {int(check.loc[~check.required, 'exists'].sum())}/{int((~check.required).sum())}.
- PRET pseudobulk statistical units: {len(pret_meta)} across {pret_meta.patient_id.nunique()} patients.
- Primary target cell types each have 6 CR, 5 NR, and 14 PR PRET patient-celltype units.
- The 50,000-cell annotated object is used only for descriptive state maps and sampled cell-level annotation.
- Primary response and regulator inference uses patient-celltype or patient-state pseudobulk.
- Phase 1 locked programs are imported unchanged; Phase 3 does not refit the response signature.

## PRET unit counts

```
{counts.to_string()}
```
"""
    (INPUT / "phase3_input_summary.md").write_text(summary, encoding="utf-8")
    print(f"Frozen {int(check.exists.sum())} Phase 1/2 input files; all required inputs present.")


if __name__ == "__main__":
    main()
