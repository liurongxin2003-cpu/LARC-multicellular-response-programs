from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spatial_config import PROGRAM_LABELS, PROGRAM_ORDER


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization")
PROGRAMS = [PROGRAM_LABELS[x] for x in PROGRAM_ORDER]
SOURCE_TO_LABEL = {source: PROGRAM_LABELS[source] for source in PROGRAM_ORDER}
MATCHED_DOMAIN = {
    "global": "Resistance_multicellular_niche",
    "malignant": "Malignant_epithelial_domain",
    "CAF": "CAF_fibroblast_domain",
    "myeloid": "Myeloid_domain",
    "CD8": "T_CD8_domain",
    "endothelial": "Endothelial_domain",
}


def normalized(values):
    values = pd.Series(values, dtype=float)
    scale = np.nanmax(np.abs(values.to_numpy()))
    return values / scale if scale and np.isfinite(scale) else values * 0


def main():
    phase1 = pd.read_csv(ROOT / "01_inputs_from_phase1" / "Phase1_cross_dataset_evidence_summary.tsv", sep="\t")
    phase2 = pd.read_csv(ROOT / "06_section_level_summary" / "StereoSeq_CR_vs_NR_section_effect_summary.tsv", sep="\t")
    coverage = pd.read_csv(ROOT / "04_program_gene_coverage" / "StereoSeq_locked_program_gene_coverage.tsv", sep="\t")
    sensitivity = pd.read_csv(
        ROOT / "10_sensitivity_analysis" / "StereoSeq_leave_one_section_program_stability_summary.tsv", sep="\t"
    )
    localization = pd.read_csv(
        ROOT / "08_program_localization" / "StereoSeq_CR_NR_localization_summary.tsv", sep="\t"
    )
    residual = pd.read_csv(
        ROOT / "09_response_group_comparison" / "StereoSeq_nonredundancy_section_residual_summary.tsv", sep="\t"
    )

    rows = []
    discovery = phase1[(phase1["dataset"] == "CNP0004138") & (phase1["cohort_subset"] == "PRET discovery")]
    for _, row in discovery.iterrows():
        label = SOURCE_TO_LABEL[row["program_source_cell_type"]]
        rows.append(
            {
                "program_name": label,
                "evidence_branch": "Phase1 PRET gradient",
                "direction_score": 1.0 if bool(row["direction_support"]) and bool(row["PR_intermediate"]) else -1.0,
                "raw_metric": row["effect_poor_minus_good"],
                "metric_definition": "NR-minus-CR discovery score; PR intermediate required",
                "statistical_unit": "patient-celltype pseudobulk",
                "independence_status": "discovery-trained",
            }
        )
    for dataset in ["GSE209746", "GSE87211"]:
        frame = phase1[(phase1["dataset"] == dataset) & (phase1["cohort_subset"] == "all_evaluable")]
        for _, row in frame.iterrows():
            label = SOURCE_TO_LABEL[row["program_source_cell_type"]]
            direction_score = float(np.clip((row["AUC"] - 0.5) * 2, -1, 1))
            rows.append(
                {
                    "program_name": label,
                    "evidence_branch": f"Phase1 {dataset}",
                    "direction_score": direction_score,
                    "raw_metric": row["AUC"],
                    "metric_definition": "AUC for poor response; 0.5 is null",
                    "statistical_unit": "patient bulk biopsy",
                    "independence_status": "locked external validation",
                }
            )

    p2_scaled = normalized(phase2.set_index("program_name")["effect_NR_minus_CR"])
    for _, row in phase2.iterrows():
        rows.append(
            {
                "program_name": row["program_name"],
                "evidence_branch": "Phase2 section direction",
                "direction_score": p2_scaled[row["program_name"]],
                "raw_metric": row["effect_NR_minus_CR"],
                "metric_definition": "section-pseudobulk NR-minus-CR effect",
                "statistical_unit": "section",
                "independence_status": "post-NAC spatial localization",
            }
        )

    sensitivity = sensitivity.set_index("program_name")
    for label in PROGRAMS:
        support = sensitivity.loc[label, "supported_scenarios"]
        rows.append(
            {
                "program_name": label,
                "evidence_branch": "Phase2 leave-one stability",
                "direction_score": support / 2 - 1,
                "raw_metric": support,
                "metric_definition": "positive direction scenarios out of four, rescaled to -1..1",
                "statistical_unit": "section sensitivity scenario",
                "independence_status": "robustness",
            }
        )

    matched = localization[
        localization.apply(lambda row: MATCHED_DOMAIN.get(row["program_name"]) == row["domain_module"], axis=1)
    ].set_index("program_name")
    loc_scaled = normalized(matched["effect_NR_minus_CR_partial_localization"])
    for label in PROGRAMS:
        rows.append(
            {
                "program_name": label,
                "evidence_branch": "Phase2 matched-domain localization",
                "direction_score": loc_scaled[label],
                "raw_metric": matched.loc[label, "effect_NR_minus_CR_partial_localization"],
                "metric_definition": "NR-minus-CR matched-domain partial localization after CAF_FAP control",
                "statistical_unit": "section correlation",
                "independence_status": "spatial localization",
            }
        )

    residual = residual.set_index("program_name")
    residual_scaled = normalized(residual["residual_effect_NR_minus_CR_after_CAF_FAP"])
    for label in PROGRAMS:
        rows.append(
            {
                "program_name": label,
                "evidence_branch": "Phase2 CAF-adjusted direction",
                "direction_score": residual_scaled[label],
                "raw_metric": residual.loc[label, "residual_effect_NR_minus_CR_after_CAF_FAP"],
                "metric_definition": "four-section NR-minus-CR residual after CAF_FAP adjustment",
                "statistical_unit": "section",
                "independence_status": "descriptive nonredundancy",
            }
        )

    evidence = pd.DataFrame(rows)
    evidence.to_csv(
        ROOT / "11_cross_phase_integration" / "Phase1_Phase2_cross_evidence_summary.tsv", sep="\t", index=False
    )

    coverage_min = coverage.groupby("program_name")["total_represented_fraction"].min()
    support_programs = phase2.loc[phase2["direction_support"], "program_name"].tolist()
    stable_programs = sensitivity.loc[sensitivity["supported_scenarios"] == 4].index.tolist()
    noncaf_programs = [p for p in support_programs if p in ["malignant", "myeloid", "CD8", "endothelial"]]
    matched_positive = matched.loc[matched["effect_NR_minus_CR_partial_localization"] > 0].index.tolist()

    claims = [
        {
            "claim": "Phase 1 locked programs are technically represented in all Stereo-seq sections.",
            "supporting_dataset": "CNP0004138 Stereo-seq",
            "evidence_type": "locked-gene coverage",
            "statistical_unit": "program-section",
            "support_strength": "high" if coverage_min.min() >= 0.70 else "moderate" if coverage_min.min() >= 0.50 else "low",
            "what_it_cannot_prove": "Response association, spatial localization, or biological activity.",
            "limitation": f"Minimum program-section coverage is {coverage_min.min():.3f}; coverage does not ensure expression variance.",
            "allowed_wording": "Locked programs have sufficient Stereo-seq gene representation for spatial scoring.",
        },
        {
            "claim": "A subset of locked programs retains the prespecified NR-higher section direction.",
            "supporting_dataset": "ST-CR1, ST-CR2, ST-NR1, ST-NR2",
            "evidence_type": "section-pseudobulk locked scoring",
            "statistical_unit": "section",
            "support_strength": "moderate" if len(support_programs) >= 3 else "limited",
            "what_it_cannot_prove": "Clinical utility or pretreatment prediction; all sections are post-NAC.",
            "limitation": f"Only 2 CR and 2 NR sections; supportive programs: {', '.join(support_programs) if support_programs else 'none'}.",
            "allowed_wording": "Section-level spatial direction supports selected resistance-associated programs.",
        },
        {
            "claim": "Spatial localization can be assigned to broad biological niches.",
            "supporting_dataset": "four Stereo-seq sections",
            "evidence_type": "within-section program-domain correlation and same-bin enrichment",
            "statistical_unit": "section correlation",
            "support_strength": "moderate" if len(matched_positive) >= 2 else "limited",
            "what_it_cannot_prove": "Pure cell identity, physical interaction, or causal signaling.",
            "limitation": f"Marker-domain annotation is computational; positive matched-domain effects: {', '.join(matched_positive) if matched_positive else 'none'}.",
            "allowed_wording": "The locked program shows response-associated spatial localization to marker-defined niches.",
        },
        {
            "claim": "The spatial result is not solely a reproduction of CAF_FAP/MIR4435-2HG.",
            "supporting_dataset": "four Stereo-seq sections plus source comparator modules",
            "evidence_type": "CAF_FAP correlation, partial localization, and section residual screen",
            "statistical_unit": "section and within-section correlation",
            "support_strength": "moderate" if len(noncaf_programs) >= 2 else "limited",
            "what_it_cannot_prove": "Independent mechanism or causal regulator.",
            "limitation": "CAF adjustment has n=4 and is unstable; novelty requires non-CAF section direction and localization.",
            "allowed_wording": "Non-CAF programs provide spatial information beyond a CAF-only comparator pattern.",
        },
        {
            "claim": "The section direction is robust to individual-section removal.",
            "supporting_dataset": "leave-one-section recomputation",
            "evidence_type": "normalization and scoring recomputed across remaining sections",
            "statistical_unit": "section sensitivity scenario",
            "support_strength": "moderate" if len(stable_programs) >= 2 else "limited",
            "what_it_cannot_prove": "Population-level reproducibility.",
            "limitation": f"Each scenario contains only three sections; stable programs: {', '.join(stable_programs) if stable_programs else 'none'}.",
            "allowed_wording": "Direction stability was assessed by leave-one-section recomputation.",
        },
        {
            "claim": "Spatial evidence can nominate a cell/state for later virtual perturbation.",
            "supporting_dataset": "Phase 1 plus Phase 2 integration",
            "evidence_type": "cross-phase direction, localization, nonredundancy, and sensitivity",
            "statistical_unit": "patient/section evidence chain",
            "support_strength": "conditional",
            "what_it_cannot_prove": "Virtual knockout validation or a validated therapeutic target.",
            "limitation": "Any Phase 3 regulator remains a computational nomination and requires independent experimental validation.",
            "allowed_wording": "A candidate perturbation target state may be nominated in Phase 3 if the spatial gates are met.",
        },
    ]
    pd.DataFrame(claims).to_csv(
        ROOT / "11_cross_phase_integration" / "Phase2_claim_evidence_boundary_table.tsv", sep="\t", index=False
    )

    (ROOT / "11_cross_phase_integration" / "Phase2_integrated_interpretation.md").write_text(
        "# Phase 2 integrated interpretation\n\n"
        f"Locked-program coverage is sufficient for scoring (minimum program-section fraction {coverage_min.min():.3f}).\n\n"
        f"Section-level programs with the prespecified NR-higher direction: {', '.join(support_programs) if support_programs else 'none'}.\n\n"
        f"Programs retaining a positive direction in all leave-one-section scenarios: {', '.join(stable_programs) if stable_programs else 'none'}.\n\n"
        f"Non-CAF supportive programs: {', '.join(noncaf_programs) if noncaf_programs else 'none'}. Positive matched-domain partial-localization effects: {', '.join(matched_positive) if matched_positive else 'none'}.\n\n"
        "The integrated interpretation is a response-associated post-NAC spatial program with section-level limitations. It is not a clinical biomarker or causal mechanism.\n",
        encoding="utf-8",
    )
    print("Integrated Phase 1 and Phase 2 evidence and claim boundaries.")


if __name__ == "__main__":
    main()
