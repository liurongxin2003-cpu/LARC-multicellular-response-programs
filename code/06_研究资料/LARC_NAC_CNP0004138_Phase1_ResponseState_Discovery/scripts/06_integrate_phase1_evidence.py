from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")
RNG = np.random.default_rng(87211)


def bootstrap_mean(values: np.ndarray, iterations: int = 10000):
    values = np.asarray(values, float)
    draws = np.empty(iterations)
    for i in range(iterations):
        draws[i] = RNG.choice(values, len(values), replace=True).mean()
    return np.quantile(draws, [0.025, 0.975])


def summarize_dynamics():
    path = ROOT / "05_response_state_discovery" / "CNP0004138_pre_post_program_dynamics.tsv"
    dynamics = pd.read_csv(path, sep="\t")
    rows = []
    for keys, frame in dynamics.groupby(
        ["program_name", "source_cell_type", "comparison", "response_standardized"], dropna=False
    ):
        values = frame["paired_change"].dropna().to_numpy()
        if len(values) == 0:
            continue
        ci = bootstrap_mean(values)
        if len(values) >= 2 and not np.allclose(values, 0):
            test = wilcoxon(values, alternative="two-sided", method="auto")
            statistic, p_value = test.statistic, test.pvalue
        else:
            statistic, p_value = np.nan, np.nan
        rows.append(
            {
                "program_name": keys[0],
                "source_cell_type": keys[1],
                "comparison": keys[2],
                "response_standardized": keys[3],
                "n_paired_patients": len(values),
                "mean_paired_change": values.mean(),
                "median_paired_change": np.median(values),
                "bootstrap_CI95_low": ci[0],
                "bootstrap_CI95_high": ci[1],
                "paired_standardized_effect_dz": values.mean() / values.std(ddof=1) if len(values) > 1 and values.std(ddof=1) else np.nan,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
                "statistical_unit": "paired patient-celltype pseudobulk",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        ROOT / "05_response_state_discovery" / "CNP0004138_pre_post_program_dynamics_summary.tsv",
        sep="\t",
        index=False,
    )
    return summary


def integrate_program_evidence():
    discovery = pd.read_csv(
        ROOT / "05_response_state_discovery" / "CNP0004138_PR_gradient_sensitivity_summary.tsv", sep="\t"
    )
    discovery = discovery[discovery["score"] == "resistance_score"].copy()
    discovery["program_source_cell_type"] = discovery["source_cell_type"]
    g209 = pd.read_csv(ROOT / "07_gse209746_validation" / "GSE209746_validation_results.tsv", sep="\t")
    g872 = pd.read_csv(ROOT / "08_gse87211_validation" / "GSE87211_validation_results.tsv", sep="\t")

    rows = []
    for _, row in discovery.iterrows():
        rows.append(
            {
                "dataset": "CNP0004138",
                "cohort_subset": "PRET discovery",
                "program_source_cell_type": row["source_cell_type"],
                "endpoint": "NR_vs_CR with PR gradient",
                "n_poor": row["n_NR"],
                "n_intermediate": row["n_PR"],
                "n_good": row["n_CR"],
                "effect_poor_minus_good": row["effect_NR_minus_CR"],
                "CI95_low": row["bootstrap_CI_low"],
                "CI95_high": row["bootstrap_CI_high"],
                "p_value": row["p_value_CR_vs_NR"],
                "FDR_BH": np.nan,
                "AUC": np.nan,
                "direction_support": row["expected_CR_to_NR_direction"],
                "PR_intermediate": row["PR_intermediate_by_mean"],
                "evidence_role": "discovery; score is trained in this cohort",
            }
        )
    for dataset, frame in [("GSE209746", g209), ("GSE87211", g872)]:
        for _, row in frame.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "cohort_subset": row["cohort_subset"],
                    "program_source_cell_type": row["program_source_cell_type"],
                    "endpoint": row["endpoint"],
                    "n_poor": row["n_poor"],
                    "n_intermediate": np.nan,
                    "n_good": row["n_good"],
                    "effect_poor_minus_good": row["mean_difference_poor_minus_good"],
                    "CI95_low": row["bootstrap_CI95_low"],
                    "CI95_high": row["bootstrap_CI95_high"],
                    "p_value": row["mannwhitney_p"],
                    "FDR_BH": row["mannwhitney_FDR_BH"],
                    "AUC": row["ROC_AUC_poor_response"],
                    "direction_support": row["direction_supports_locked_hypothesis"],
                    "PR_intermediate": np.nan,
                    "evidence_role": "locked independent validation" if row["cohort_subset"] == "all_evaluable" else "prespecified biological sensitivity",
                }
            )
    integrated = pd.DataFrame(rows)
    integrated.to_csv(ROOT / "11_reports" / "Phase1_cross_dataset_evidence_summary.tsv", sep="\t", index=False)
    return integrated


def build_claim_boundary_table(dynamics_summary: pd.DataFrame):
    postt = dynamics_summary[dynamics_summary["comparison"] == "POSTT_minus_PRET"]
    nr_global = postt[(postt["program_name"] == "global") & (postt["response_standardized"] == "NR")]
    nr_change = nr_global["mean_paired_change"].iloc[0] if len(nr_global) else np.nan
    claims = [
        {
            "claim": "CNP0004138 metadata and response groups were reconstructed and frozen.",
            "dataset": "CNP0004138",
            "evidence_type": "barcode-level metadata mapping",
            "supports_what": "Reliable patient, PRET/POSTT/POSTN, and CR/PR/NR linkage for 378,344 cells.",
            "cannot_prove_what": "Clinical generalizability beyond the source cohort.",
            "patient_or_sample_number": "29 patients; PRET evaluable: 6 CR, 14 PR, 5 NR",
            "statistical_unit": "patient/sample/cell mapping",
            "confidence_level": "high for deposited naming scheme",
            "main_limitation": "Two patients have unknown response; one CR and one PR lack PRET samples.",
        },
        {
            "claim": "A pretreatment response/resistance program can be locked from patient-level pseudobulk.",
            "dataset": "CNP0004138",
            "evidence_type": "PRET CR-versus-NR patient-celltype pseudobulk discovery",
            "supports_what": "Six 200-gene candidate programs with coherent CR-to-PR-to-NR score ordering.",
            "cannot_prove_what": "A clinically validated biomarker or transcriptome-wide significant individual genes.",
            "patient_or_sample_number": "6 CR, 14 PR, 5 NR in each supported priority compartment",
            "statistical_unit": "patient-celltype pseudobulk",
            "confidence_level": "moderate for program-level state; low for individual genes",
            "main_limitation": "Only 11 primary CR/NR patients; zero genes reach BH FDR < 0.05; gene selection is effect-ranked.",
        },
        {
            "claim": "The locked program transfers directionally to an independent pretreatment RNA-seq cohort.",
            "dataset": "GSE209746",
            "evidence_type": "locked external bulk scoring",
            "supports_what": "All six programs are higher in incomplete than complete responders; the pattern remains in MSS-only sensitivity.",
            "cannot_prove_what": "Cell-type-specific localization or prospective prediction.",
            "patient_or_sample_number": "27 CR and 70 iCR; 17 response-unknown excluded",
            "statistical_unit": "patient bulk biopsy",
            "confidence_level": "moderate directional support",
            "main_limitation": "No program passes six-program BH FDR in the full cohort; multimodal neoadjuvant regimens are heterogeneous and not identical to chemotherapy-only NAC.",
        },
        {
            "claim": "GSE87211 provides regimen-dependent external support.",
            "dataset": "GSE87211",
            "evidence_type": "locked external microarray scoring with strict pCR endpoint",
            "supports_what": "All six directions agree overall; platinum-containing CRT sensitivity shows stronger effects, including CD8 FDR support.",
            "cannot_prove_what": "The published 70/41 TRG grouping or a universal radiochemotherapy response marker.",
            "patient_or_sample_number": "35 pCR and 166 non-pCR overall; platinum subset 11 pCR and 75 non-pCR",
            "statistical_unit": "patient bulk biopsy",
            "confidence_level": "low-to-moderate; sensitivity-grade",
            "main_limitation": "Sample-level TRG is unavailable; strict ypT0N0M0 was reconstructed; pure 5-FU+RT subgroup is mostly non-supportive.",
        },
        {
            "claim": "The program has cell-compartment hypotheses suitable for later spatial localization.",
            "dataset": "CNP0004138 plus external bulk cohorts",
            "evidence_type": "broad marker reconstruction and compartment-specific pseudobulk programs",
            "supports_what": "Malignant, CAF, myeloid, CD8, and endothelial hypotheses can be tested spatially.",
            "cannot_prove_what": "True spatial colocalization, ligand-receptor causality, or regulator causality.",
            "patient_or_sample_number": "25 PRET response-evaluable patients; 378,344 total cells",
            "statistical_unit": "patient-celltype pseudobulk",
            "confidence_level": "moderate for broad compartments",
            "main_limitation": "Malignant epithelial annotation is provisional and has no full CNV confirmation.",
        },
        {
            "claim": "Post-treatment score dynamics are heterogeneous rather than uniformly resistance-increasing.",
            "dataset": "CNP0004138",
            "evidence_type": "paired PRET-to-POSTT pseudobulk scoring",
            "supports_what": f"Treatment-associated state remodeling; the global NR mean paired change is {nr_change:.3f}.",
            "cannot_prove_what": "Pretreatment prediction, clonal selection, or treatment-induced causal resistance.",
            "patient_or_sample_number": "paired counts vary by response and compartment",
            "statistical_unit": "paired patient-celltype pseudobulk",
            "confidence_level": "exploratory",
            "main_limitation": "Small response-stratified paired groups and compositional changes after therapy.",
        },
    ]
    table = pd.DataFrame(claims)
    table.to_csv(ROOT / "11_reports" / "Phase1_claim_evidence_boundary_table.tsv", sep="\t", index=False)
    return table


def write_interpretation(integrated: pd.DataFrame, dynamics: pd.DataFrame):
    g209 = integrated[(integrated["dataset"] == "GSE209746") & (integrated["cohort_subset"] == "all_evaluable")]
    g872 = integrated[(integrated["dataset"] == "GSE87211") & (integrated["cohort_subset"] == "all_evaluable")]
    ox = integrated[(integrated["dataset"] == "GSE87211") & (integrated["cohort_subset"] == "5-FU + Oxaliplatin + RT")]
    text = f"""# Phase 1 integrated interpretation

## Decision-level finding

CNP0004138 supports a **candidate pretreatment treatment-response-associated state**, not a clinically validated biomarker. The discovery cohort has genuine patient response labels, complete processed single-cell coverage, patient-level pseudobulk, and a consistent CR-to-PR-to-NR gradient across five biologically relevant compartments plus the all-cell summary. However, only 6 CR and 5 NR PRET patients train the contrast, and no individual gene reaches transcriptome-wide BH FDR < 0.05. The locked programs are therefore explicitly effect-ranked and limited-significance.

## Independent validation

- GSE209746: {int(g209['direction_support'].sum())}/{len(g209)} locked programs agree in direction in 27 CR versus 70 iCR pretreatment biopsies. Effect sizes are modest; no full-cohort program passes six-program BH correction.
- GSE87211 strict endpoint: {int(g872['direction_support'].sum())}/{len(g872)} programs agree in direction across 35 strict ypT0N0M0 pCR and 166 non-pCR tumors, but overall effects are small and nonsignificant.
- GSE87211 platinum-containing CRT sensitivity: {int(ox['direction_support'].sum())}/{len(ox)} programs agree. The CD8-derived program has the strongest support, but this is a regimen-stratified sensitivity analysis with only 11 pCR cases and cannot replace the primary all-evaluable result.
- The pure 5-FU+RT subgroup is mostly non-supportive, indicating treatment-context dependence rather than a universal response state.

## Treatment dynamics

Paired PRET-to-POSTT changes are heterogeneous across response groups and compartments. They are useful for describing treatment-state remodeling but do not show a uniform post-treatment rise of the resistance score and are not evidence of pretreatment prediction or acquired causal resistance.

## Biological localization

Malignant epithelial, CAF, myeloid/macrophage, CD8 T-cell, and endothelial programs all have sufficient PRET patient-level support to define a spatial hypothesis. The current annotation is broad and marker-guided; malignant status remains provisional without full CNV inference. Phase 2 should test where the **already locked** programs localize in four Stereo-seq sections, with patient/section-level rather than spot-level claims.

## Boundary

The strongest defensible claim is: **a locked pretreatment response-state program derived from platinum-treated LARC single-cell pseudobulk shows ordered response-group behavior and partial, treatment-context-dependent transfer to independent pretreatment bulk cohorts.** It cannot yet be called a clinically validated biomarker, a causal mechanism, or a spatially validated program.
"""
    (ROOT / "11_reports" / "Phase1_integrated_interpretation.md").write_text(text, encoding="utf-8")


def main():
    dynamics = summarize_dynamics()
    integrated = integrate_program_evidence()
    build_claim_boundary_table(dynamics)
    write_interpretation(integrated, dynamics)
    print("Integrated Phase 1 evidence and boundaries.")


if __name__ == "__main__":
    main()
