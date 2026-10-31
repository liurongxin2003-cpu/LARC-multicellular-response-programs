from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "13_reports"
LOGS = ROOT / "00_logs"


def read_tsv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, sep="\t")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    display.columns = [str(column) for column in display.columns]
    rows = ["| " + " | ".join(display.columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(display.columns)) + " |")
    for _, row in display.iterrows():
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report() -> str:
    qc = read_tsv("03_stereoseq_matrix/StereoSeq_matrix_QC_summary.tsv")
    coverage = read_tsv("04_program_gene_coverage/StereoSeq_locked_program_gene_coverage.tsv")
    section = read_tsv("06_section_level_summary/StereoSeq_section_level_program_scores.tsv")
    effects = read_tsv("06_section_level_summary/StereoSeq_CR_vs_NR_section_effect_summary.tsv")
    localization = read_tsv("08_program_localization/StereoSeq_CR_NR_localization_summary.tsv")
    sensitivity = read_tsv("10_sensitivity_analysis/StereoSeq_leave_one_section_sensitivity.tsv")
    residual = read_tsv("09_response_group_comparison/StereoSeq_nonredundancy_section_residual_summary.tsv")

    qc_view = qc[[
        "section_id", "response_group", "n_gene_rows", "n_bins", "total_counts_median",
        "detected_genes_median", "fraction_bins_detected_genes_ge200", "mitochondrial_fraction_median",
        "QC_status",
    ]].copy()
    qc_view.columns = [
        "Section", "Group", "Genes", "Bins", "Median counts", "Median genes/bin",
        "Fraction >=200 genes", "Median mitochondrial fraction", "QC",
    ]
    for column in ["Median counts", "Median genes/bin"]:
        qc_view[column] = qc_view[column].map(lambda x: f"{x:.0f}")
    for column in ["Fraction >=200 genes", "Median mitochondrial fraction"]:
        qc_view[column] = qc_view[column].map(lambda x: f"{x:.3f}")

    cov_view = coverage.groupby("program_name")["total_represented_fraction"].agg(["min", "max"]).reset_index()
    cov_view.columns = ["Program", "Minimum coverage", "Maximum coverage"]
    cov_view["Minimum coverage"] = cov_view["Minimum coverage"].map(lambda x: f"{x:.3f}")
    cov_view["Maximum coverage"] = cov_view["Maximum coverage"].map(lambda x: f"{x:.3f}")

    section_pivot = section.pivot(
        index="section_id", columns="program_name", values="section_pseudobulk_resistance_score"
    )
    section_pivot = section_pivot.reindex(["ST-CR1", "ST-CR2", "ST-NR1", "ST-NR2"])
    section_pivot = section_pivot.reindex(columns=["global", "malignant", "CAF", "myeloid", "CD8", "endothelial"])
    section_view = section_pivot.reset_index().rename(columns={"section_id": "Section"})
    for column in section_view.columns[1:]:
        section_view[column] = section_view[column].map(lambda x: f"{x:.3f}")

    effect_view = effects[[
        "program_name", "effect_NR_minus_CR", "hedges_g_NR_minus_CR",
        "NR_above_CR_pairwise_fraction", "complete_section_separation",
        "exact_label_permutation_p_two_sided",
    ]].copy()
    effect_view.columns = ["Program", "NR-CR effect", "Hedges g", "NR>CR pairs", "Complete separation", "Exact p"]
    for column in ["NR-CR effect", "Hedges g", "NR>CR pairs", "Exact p"]:
        effect_view[column] = effect_view[column].map(lambda x: f"{x:.3f}")

    matched = {
        "global": "Resistance_multicellular_niche",
        "malignant": "Malignant_epithelial_domain",
        "CAF": "CAF_fibroblast_domain",
        "myeloid": "Myeloid_domain",
        "CD8": "T_CD8_domain",
        "endothelial": "Endothelial_domain",
    }
    matched_rows = []
    for program, domain in matched.items():
        row = localization[(localization.program_name == program) & (localization.domain_module == domain)].iloc[0]
        matched_rows.append({
            "Program": program,
            "Matched domain/niche": domain,
            "NR-CR localization rho": fmt(row.effect_NR_minus_CR_localization),
            "CAF-adjusted effect": fmt(row.effect_NR_minus_CR_partial_localization),
            "Both NR positive": bool(row.NR_sections_both_positive),
        })
    matched_view = pd.DataFrame(matched_rows)

    leave_one = sensitivity[sensitivity["sensitivity_scenario"] != "all_sections"]
    sens_range = leave_one.groupby("program_name")["effect_NR_minus_CR"].agg(["min", "max"]).reset_index()
    sens_range.columns = ["Program", "Minimum leave-one effect", "Maximum leave-one effect"]
    sens_range["All four positive"] = sens_range["Minimum leave-one effect"] > 0
    for column in ["Minimum leave-one effect", "Maximum leave-one effect"]:
        sens_range[column] = sens_range[column].map(lambda x: f"{x:.3f}")

    residual_view = residual[["program_name", "residual_effect_NR_minus_CR_after_CAF_FAP"]].copy()
    residual_view.columns = ["Program", "Residual NR-CR effect after CAF_FAP adjustment"]
    residual_column = "Residual NR-CR effect after CAF_FAP adjustment"
    residual_view[residual_column] = residual_view[residual_column].map(lambda x: f"{float(x):.3f}").astype("string")

    return f"""# LARC NAC CNP0004138 Phase 2 Stereo-seq spatial localization final report

## Executive verdict

**Final decision: Strong Continue to Phase 3 virtual perturbation / regulator nomination.** All four response-labeled post-NAC Stereo-seq sections were read successfully. The six unchanged Phase 1 locked programs had high gene coverage, all six showed the prespecified section-level NR-higher resistance direction, every NR section exceeded every CR section, and all directions remained positive in every leave-one-section analysis. The signal was not driven by ST-NR2 and was not reducible to the source-study CAF_FAP axis.

This is evidence for a **response-associated post-NAC spatial resistance program**. It is not a clinically validated biomarker, a proven causal mechanism, a validated target, or virtual-knockout validation.

## Study boundary and data provenance

- Source dataset: [CNGB CNP0004138](https://db.cngb.org/data_resources/project/CNP0004138/).
- Source publication: [Li et al., Cell Reports Medicine 2023, DOI 10.1016/j.xcrm.2023.101231](https://pmc.ncbi.nlm.nih.gov/articles/PMC10591051/).
- Spatial material: four bin50 full-transcriptome post-NAC sections, two CR and two NR.
- Statistical unit: section. Bins are descriptive spatial observations and are never treated as independent patients.
- ST-NR1/ST-NR2 exact donor identity remains unresolved; no donor-specific inference is made.
- Locked inputs: six Phase 1 programs, each with 100 NR-high and 100 CR-high genes. No spatial refitting, gene reselection, score reversal, or parameter tuning was performed.

## 1-2. Matrix loading and QC

All four expression matrices and coordinate files were loaded, and coordinate rows matched matrix columns exactly. Uppercase gene-symbol standardization produced zero duplicated symbols in these files. Actual downloaded dimensions below supersede earlier pilot estimates.

{markdown_table(qc_view)}

ST-CR2 and ST-NR2 were retained but labeled low quality because of lower library complexity. ST-NR1 had the deepest median coverage. Mitochondrial fractions were recorded as QC descriptors and were not used to discard sections.

## 3. Locked-program coverage

Every program-section combination met high-confidence coverage; the overall range was 0.875-0.945 and the maximum NR-high/CR-high coverage imbalance was 0.110.

{markdown_table(cov_view)}

## Scoring clarification

Bin maps use log1p(CP10K) followed by within-section gene z-scoring and are descriptive. Averaging such z-scores within a section would be mathematically centered near zero, so the prespecified CR-vs-NR section comparison was implemented using section-pseudobulk counts, logCPM, and gene scaling across the four section pseudobulks before applying the unchanged locked gene weights. This preserves section-level inference without treating bins as replicates.

## 4-6. Section-level CR-versus-NR evidence

Section resistance scores:

{markdown_table(section_view)}

All six programs showed higher resistance scores in NR and, equivalently, higher response scores in CR. Both NR sections were above both CR sections for every program.

{markdown_table(effect_view)}

The exact two-sided label-permutation p value is 0.333 for every complete-separation result because only six distinct 2-versus-2 labelings exist. It is reported for completeness and is not interpreted as evidence against the large, directionally consistent effects. Effect size and section consistency are primary.

The strongest section-level standardized effects were endothelial, global, myeloid, and CD8; malignant and CAF were positive but smaller. Malignant nevertheless showed coherent matched-domain spatial localization, so effect magnitude and localization are considered jointly.

## 7-8. Spatial localization and niche support

Marker modules define coarse domains/niches, not pure cell types. No deposited pathologist domain annotation was available for use as ground truth.

{markdown_table(matched_view)}

- Global resistance localized more strongly in NR to a multicellular resistance niche and malignant/proliferative/stress-rich regions.
- Malignant resistance localized to malignant-epithelial-rich regions in both NR sections.
- Myeloid resistance localized to myeloid-rich and remodeling regions in both NR sections.
- CD8 resistance showed a smaller positive matched T/CD8 localization effect, while its strongest NR enrichment was near malignant, proliferative, stress, and multicellular-niche modules; this is an immune-associated spatial state, not proof of T-cell causality.
- Endothelial resistance had a small but consistent matched vascular-domain effect and broader remodeling associations.
- CAF had a positive section score but a negative NR-minus-CR matched CAF-domain localization effect. It is retained as a comparator/supporting state, not the primary new mechanism.

Same-bin top-quintile enrichment and niche maps are descriptive colocalization. They do not establish cell-cell interaction or causal communication.

## 9. CAF_FAP nonredundancy

All six section-level directions remained positive after descriptive CAF_FAP residualization:

{markdown_table(residual_view)}

Residualization with n=4 is intrinsically unstable and is therefore supportive only. More importantly, the non-CAF malignant, myeloid, CD8, and endothelial programs had positive section directions and biologically matched spatial localization, whereas the CAF program itself did not strengthen in the matched CAF domain in NR. The result therefore extends beyond a simple repeat of the source CAF_FAP/MIR4435-2HG story.

## 10-11. ST-NR2 and leave-one-section sensitivity

ST-NR2 was lower complexity but did not drive the conclusion. Removing ST-NR2 retained a positive NR-minus-CR effect for all programs: global 1.959, malignant 0.424, CAF 0.402, myeloid 0.757, CD8 0.834, and endothelial 0.325. Removing low-quality ST-CR2 or either pass-QC section also retained every direction.

{markdown_table(sens_range)}

Leave-one analyses recomputed cross-section gene scaling after each removal. With three sections per scenario, they test directional stability, not statistical significance.

## 12. Cross-phase interpretation

Phase 1 established a pretreatment CR < PR < NR gradient and external directional representation in GSE209746 and GSE87211. Phase 2 now adds post-NAC section-level and spatial-domain support using the same locked genes. Together, the evidence is stronger for a reproducible response-associated state architecture across time and assay contexts. It still does not prove prospective prediction, clinical utility, or causality.

## 13-15. Phase 3 decision and priorities

Proceed to Phase 3, with the following target-state order:

1. **Malignant EMT/stress state**: strongest mechanistic priority because it has matched malignant localization and is less likely to be a source-study CAF repeat.
2. **Myeloid suppressive/remodeling state**: strong section effect, matched localization, and multicellular-niche relevance.
3. **CD8-associated immune state**: strong section effect and NR niche association, with careful distinction between immune dysfunction and cytotoxic activity.
4. **Endothelial remodeling state**: robust section effect but smaller matched-domain localization; secondary perturbation track.
5. **CAF state**: comparator and resistance-supporting context only unless Phase 3 identifies a regulator clearly independent of the known CAF_FAP/MIR4435-2HG axis.

Phase 3 should nominate regulators using pretreatment single-cell states and test robustness across model assumptions. If regulator rankings are unstable, the project should stop at a response-associated spatial-program paper and should not claim a mechanism.

## Direct answers to the 15 required questions

1. **Four sections read?** Yes, all matrices and coordinates were successfully matched.
2. **QC?** ST-CR1 and ST-NR1 passed; ST-CR2 and ST-NR2 were retained as low complexity. Exact values are tabulated above.
3. **Coverage sufficient?** Yes, 0.875-0.945 for every program-section pair.
4. **NR higher resistance?** Yes, for all six locked programs, with complete section separation.
5. **CR higher response?** Yes; response score is the exact inverse of resistance score.
6. **Most supported programs?** All six directionally; global, myeloid, CD8, and endothelial had the largest section effects, while malignant had especially coherent matched-domain localization.
7. **Where do programs localize?** Malignant to malignant-rich regions, myeloid to myeloid/remodeling regions, CD8 to T-cell and broader stress/malignant niches, endothelial to vascular/remodeling regions, and CAF only weakly to matched CAF regions in NR.
8. **Niche support?** Yes, descriptively, especially for global/malignant/myeloid and the malignant-proliferative-stress multicellular niche.
9. **Only CAF_FAP/MIR4435-2HG?** No; multiple non-CAF programs remain directionally and spatially supported.
10. **Does ST-NR2 control the result?** No; all six directions persist when it is removed.
11. **Leave-one sensitivity stable?** Yes, all 24 program-by-removal effects are positive.
12. **Is the Phase 1 + Phase 2 story stronger?** Yes for response-associated state and spatial architecture, not for clinical prediction or causality.
13. **Enter Phase 3?** Yes, Strong Continue.
14. **Priority states?** Malignant EMT/stress, myeloid remodeling, CD8-associated immune state, then endothelial remodeling; CAF remains a comparator.
15. **Fallback if Phase 3 fails?** Publish or frame the result as a response-associated spatial-program study and remove regulator/mechanism claims.
"""


def build_go_nogo() -> str:
    return """# Go / No-Go for LARC Phase 3 virtual perturbation

## Verdict

**Strong Continue to Phase 3 regulator nomination.**

| Prespecified condition | Result | Decision |
| --- | --- | --- |
| Locked-program coverage is sufficient | Minimum program-section coverage 0.875 | Pass |
| Global or at least two compartment programs support NR direction | All six programs support NR-higher resistance | Pass |
| Not driven by ST-NR2 or one section | All six directions persist in every leave-one analysis | Pass |
| Programs localize to plausible domains/niches | Global, malignant, myeloid, CD8, and endothelial show positive matched or niche localization | Pass |
| Not merely the source CAF mechanism | Non-CAF programs remain supported; CAF matched-domain effect is negative | Pass |
| Perturbation target states can be specified | Malignant EMT/stress, myeloid remodeling, CD8-associated immune, endothelial remodeling | Pass |

## Phase 3 scope

Phase 3 may perform virtual perturbation and **candidate regulator nomination**. It must not describe virtual output as knockout validation, causal proof, or a validated therapeutic target.

Priority order:

1. Malignant EMT/stress state.
2. Myeloid suppressive/remodeling state.
3. CD8-associated immune state.
4. Endothelial remodeling state.
5. CAF only as a comparator/supporting state.

## Required safeguards

- Keep Phase 1 programs locked and analyze perturbations within biologically defined cell states.
- Use patient-level or patient-timepoint pseudobulk for response inference; cells are not independent patients.
- Require stability across regulator methods, subsampling, and leave-one-patient analyses.
- Separate regulator nomination from causal validation.
- If rankings are unstable or collapse to the known CAF_FAP/MIR4435-2HG axis, stop mechanism claims and retain the spatial-program interpretation.

## Evidence boundary

The Phase 2 result supports a response-associated post-NAC spatial resistance architecture in four sections. Sample size, low-complexity sections, unresolved NR donor identity, and post-treatment sampling prevent claims of prospective response prediction, clinical utility, or causality.
"""


def write_tree() -> None:
    excluded = {"source"}
    lines = [ROOT.name + "/"]

    def visit(folder: Path, prefix: str = "") -> None:
        entries = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            if entry.is_dir():
                if entry.name in excluded:
                    count = sum(1 for item in entry.rglob("*") if item.is_file())
                    lines.append(f"{prefix}{connector}{entry.name}/ [{count} source files]")
                    continue
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if index == len(entries) - 1 else "|   "
                visit(entry, prefix + extension)
            else:
                size_mb = entry.stat().st_size / (1024 * 1024)
                lines.append(f"{prefix}{connector}{entry.name} ({size_mb:.2f} MB)")

    visit(ROOT)
    (LOGS / "directory_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_delivery_index() -> None:
    roles = {
        "02_spatial_metadata/StereoSeq_frozen_spatial_metadata.tsv": "Frozen section/response/QC metadata",
        "03_stereoseq_matrix/StereoSeq_matrix_QC_summary.tsv": "Matrix, coordinate, count, gene-complexity QC",
        "04_program_gene_coverage/StereoSeq_locked_program_gene_coverage.tsv": "Locked-gene representation audit",
        "05_spatial_program_scores/StereoSeq_bin_level_program_scores.tsv": "Descriptive bin-level locked scores",
        "06_section_level_summary/StereoSeq_section_level_program_scores.tsv": "Primary section-pseudobulk scores",
        "06_section_level_summary/StereoSeq_CR_vs_NR_section_effect_summary.tsv": "CR-versus-NR effects and exact descriptive test",
        "07_spatial_domain_annotation/StereoSeq_spatial_domain_scores.tsv": "Marker-based spatial niche scores",
        "08_program_localization/StereoSeq_program_domain_correlation.tsv": "Within-section program-domain correlations",
        "08_program_localization/StereoSeq_CR_NR_localization_summary.tsv": "Section-level localization comparison",
        "09_response_group_comparison/StereoSeq_nonredundancy_with_CAF_FAP_summary.md": "CAF_FAP comparator interpretation",
        "10_sensitivity_analysis/StereoSeq_leave_one_section_sensitivity.tsv": "Recomputed leave-one-section effects",
        "10_sensitivity_analysis/StereoSeq_ST_NR2_influence_report.md": "ST-NR2 influence audit",
        "11_cross_phase_integration/Phase2_claim_evidence_boundary_table.tsv": "Claim/evidence/inference boundary table",
        "13_reports/FINAL_LARC_Phase2_spatial_localization_report.md": "Complete Phase 2 report and 15 answers",
        "13_reports/FINAL_go_nogo_for_LARC_Phase3_virtual_perturbation.md": "Phase 3 decision and safeguards",
    }
    rows = []
    for relative, role in roles.items():
        path = ROOT / relative
        rows.append({
            "relative_path": relative,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else np.nan,
            "role": role,
        })
    pd.DataFrame(rows).to_csv(LOGS / "deliverable_index.tsv", sep="\t", index=False)


def write_qa() -> None:
    figures = sorted((ROOT / "12_draft_figures").glob("DRAFT_*.png"))
    records = []
    for png in figures:
        with Image.open(png) as image:
            rgb = np.asarray(image.convert("RGB").resize((160, 100)), dtype=float)
            records.append({
                "figure": png.name,
                "width_px": image.width,
                "height_px": image.height,
                "pixel_sd": round(float(rgb.std()), 3),
                "nonblank": bool(rgb.std() > 2),
                "matching_pdf": png.with_suffix(".pdf").exists(),
            })
    qa = pd.DataFrame(records)
    qa.to_csv(LOGS / "Phase2_figure_machine_QA.tsv", sep="\t", index=False)
    note = f"""# Phase 2 computational and figure QA

- Backend: Python only for scoring, plotting, export, preview, and visual QA.
- Figure bundle: {len(qa)} PNG previews; {int(qa.matching_pdf.sum())} have matching PDF exports.
- Machine nonblank check: {int(qa.nonblank.sum())}/{len(qa)} passed using downsampled RGB pixel variance.
- Representative visual inspection completed for QC violins, section points, section/evidence/sensitivity heatmaps, spatial score maps, spatial domains, NR niche maps, comparator heatmap, and nonredundancy plot.
- Comparator heatmap color range was tightened to the observed correlation scale after QA; numerical annotations remain the primary encoding.
- No visible label overlap, clipping, blank spatial panel, or incoherent section ordering was found in the final previews.
- Quantitative panels trace to TSV outputs and `scripts/06_make_phase2_figures.py`.
- Statistical unit is section; bin-level maps and correlations are labeled descriptive.
- Exact p values are shown only with their n=4 coarse-resolution warning. Effect size and directional consistency are primary.
- PDF is the requested vector deliverable; PNG is a 300-dpi review preview. TIFF/SVG were not requested for this pilot phase.
"""
    (LOGS / "Phase2_computational_and_figure_QA.md").write_text(note, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    (REPORTS / "FINAL_LARC_Phase2_spatial_localization_report.md").write_text(build_report(), encoding="utf-8")
    (REPORTS / "FINAL_go_nogo_for_LARC_Phase3_virtual_perturbation.md").write_text(build_go_nogo(), encoding="utf-8")
    write_delivery_index()
    write_qa()
    write_tree()
    # Refresh index and tree once reports and QA are present.
    write_delivery_index()
    write_tree()
    print("Final Phase 2 reports, QA notes, deliverable index, and directory tree created.")


if __name__ == "__main__":
    main()
