from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization")
PHASE1 = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")


INPUTS = {
    "CNP0004138_locked_response_resistance_program_global.tsv": PHASE1 / "05_response_state_discovery" / "CNP0004138_locked_response_resistance_program_global.tsv",
    "CNP0004138_locked_programs_by_celltype.tsv": PHASE1 / "05_response_state_discovery" / "CNP0004138_locked_programs_by_celltype.tsv",
    "CNP0004138_pseudobulk_locked_program_scores.tsv": PHASE1 / "05_response_state_discovery" / "CNP0004138_pseudobulk_locked_program_scores.tsv",
    "CNP0004138_CR_PR_NR_program_score_contrasts.tsv": PHASE1 / "05_response_state_discovery" / "CNP0004138_CR_PR_NR_program_score_contrasts.tsv",
    "CNP0004138_program_gene_representation.tsv": PHASE1 / "05_response_state_discovery" / "CNP0004138_program_gene_representation.tsv",
    "Phase1_cross_dataset_evidence_summary.tsv": PHASE1 / "11_reports" / "Phase1_cross_dataset_evidence_summary.tsv",
    "Phase1_claim_evidence_boundary_table.tsv": PHASE1 / "11_reports" / "Phase1_claim_evidence_boundary_table.tsv",
    "FINAL_LARC_Phase1_response_state_discovery_report.md": PHASE1 / "11_reports" / "FINAL_LARC_Phase1_response_state_discovery_report.md",
    "FINAL_go_nogo_for_LARC_Phase2_spatial_localization.md": PHASE1 / "11_reports" / "FINAL_go_nogo_for_LARC_Phase2_spatial_localization.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    input_dir = ROOT / "01_inputs_from_phase1"
    records = []
    for output_name, source in INPUTS.items():
        exists = source.exists()
        destination = input_dir / output_name
        if exists:
            shutil.copy2(source, destination)
        records.append(
            {
                "requested_input": output_name,
                "source_path": str(source),
                "resolved_path": str(destination) if exists else "",
                "exists": exists,
                "size_bytes": source.stat().st_size if exists else 0,
                "sha256": sha256(source) if exists else "",
                "import_status": "locked_copy_created" if exists else "missing",
            }
        )
    check = pd.DataFrame(records)
    check.to_csv(input_dir / "phase1_input_file_check.tsv", sep="\t", index=False)

    core = input_dir / "CNP0004138_locked_programs_by_celltype.tsv"
    if not core.exists():
        (ROOT / "13_reports" / "STOP_missing_locked_programs.md").write_text(
            "# STOP: missing locked programs\n\nThe Phase 1 locked program table could not be resolved. No spatial scoring was performed.\n",
            encoding="utf-8",
        )
        raise SystemExit("Missing core locked program table")

    programs = pd.read_csv(core, sep="\t")
    summary = (
        programs.groupby(["source_cell_type", "program_direction"], as_index=False)
        .agg(
            n_locked_genes=("gene_symbol", "nunique"),
            n_CR=("n_CR", "max"),
            n_PR=("n_PR", "max"),
            n_NR=("n_NR", "max"),
            selection_basis=("selection_basis", lambda x: "; ".join(sorted(set(x.astype(str))))),
        )
    )
    summary["spatial_refitting_allowed"] = False
    summary["import_rule"] = "gene membership and direction frozen from Phase 1"
    summary.to_csv(input_dir / "locked_program_import_summary.tsv", sep="\t", index=False)

    source_dir = ROOT / "03_stereoseq_matrix" / "source"
    import os
    spatial = pd.read_csv(Path(os.environ["LARC_SPATIAL_METADATA_TSV"]), sep="\t")
    required = {"section_id", "response_group", "sample_id", "patient_or_donor_id", "exact_donor_identity_status", "tissue_region_annotation", "bin_size", "expected_bin_number", "expected_gene_number", "QC_status", "notes"}
    if not required.issubset(spatial.columns):
        raise ValueError("Authorized source spatial manifest lacks required fields")
    spatial["coordinate_available"] = spatial["section_id"].map(
        lambda x: (source_dir / f"{x}.meta.info.tsv.txt").exists()
    )
    spatial["matrix_available"] = spatial["section_id"].map(
        lambda x: (source_dir / f"{x}.exp.count.tsv").exists()
        or (source_dir / f"{x}.exp.count.tsv.gz").exists()
    )
    spatial["primary_statistical_unit"] = "section"
    spatial.to_csv(ROOT / "02_spatial_metadata" / "StereoSeq_frozen_spatial_metadata.tsv", sep="\t", index=False)

    (ROOT / "02_spatial_metadata" / "StereoSeq_response_group_mapping_note.md").write_text(
        "# Stereo-seq response-group mapping\n\n"
        "- ST-CR1 and ST-CR2 are CR sections; source donor linkage must be obtained separately.\n"
        "- ST-NR1 and ST-NR2 are independently labeled NR, but the source supplementary table duplicates a section label for two donors. Their exact one-to-one donor assignment is unresolved.\n"
        "- Primary spatial inference is therefore CR versus NR at the section level. No donor-specific clinical covariate interpretation is permitted for the NR sections.\n"
        "- Bins are descriptive spatial observations and are never counted as independent patients.\n"
        "- All tissues are post-NAC; spatial localization does not constitute pretreatment response prediction.\n",
        encoding="utf-8",
    )
    print("Imported Phase 1 inputs and froze spatial response metadata.")


if __name__ == "__main__":
    main()
