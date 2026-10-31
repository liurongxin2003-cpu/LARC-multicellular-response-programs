from __future__ import annotations

import pandas as pd

from phase3_config import INPUT, ROOT


def main() -> None:
    out = ROOT / "11_cross_phase_integration"
    targets = pd.read_csv(ROOT / "02_target_state_definition/Phase3_target_state_table.tsv", sep="\t")
    states = pd.read_csv(ROOT / "03_refined_cell_state_annotation/refined_target_state_response_effects.tsv", sep="\t")
    ranking = pd.read_csv(ROOT / "09_candidate_prioritization/final_candidate_regulator_ranking.tsv", sep="\t")
    lr = pd.read_csv(ROOT / "07_ligand_receptor_niche/spatially_informed_LR_priority_table.tsv", sep="\t")
    state_to_receiver = {
        "Malignant_EMT_stress": "Malignant_epithelial",
        "Myeloid_suppressive_remodeling": "Myeloid_macrophage",
        "CD8_dysfunctional_stress": "CD8_T",
        "Endothelial_remodeling": "Endothelial",
        "CAF_supporting_comparator": "CAF_Fibroblast",
    }
    rows = []
    for target in targets.itertuples():
        state = states[states.target_state == target.target_state].iloc[0]
        candidates = ranking[ranking.target_state == target.target_state].sort_values("final_integrated_score", ascending=False)
        top = candidates.iloc[0]
        receiver = state_to_receiver[target.target_state]
        axes = lr[lr.receiver_cell_type == receiver].sort_values("LR_priority_score", ascending=False)
        axis = axes.iloc[0] if len(axes) else None
        rows.append({
            "target_state": target.target_state,
            "phase1_locked_program_NR_minus_CR": target.phase1_NR_minus_CR_resistance_effect,
            "phase1_gradient_rho": target.phase1_CR_PR_NR_spearman_rho,
            "phase2_matched_localization_effect": target.phase2_NR_minus_CR_localization_effect,
            "phase3_canonical_marker_state_NR_minus_CR": state.effect_NR_minus_CR,
            "phase3_canonical_marker_direction_support": state.direction_support,
            "top_nominated_regulator": top.regulator,
            "top_regulator_confidence": top.final_confidence,
            "top_regulator_integrated_score": top.final_integrated_score,
            "top_regulator_evidence_categories": top.n_positive_evidence_categories,
            "top_regulator_stability": top.stability_score,
            "top_receiver_LR_axis": f"{axis.ligand}-{axis.receptor}" if axis is not None else "unknown",
            "top_receiver_LR_sender": axis.sender_cell_type if axis is not None else "unknown",
            "top_receiver_LR_priority": axis.LR_priority_score if axis is not None else float("nan"),
            "integrated_interpretation": "locked-program regulator nomination; canonical state mechanism not established",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "Phase1_Phase2_Phase3_integrated_evidence_summary.tsv", sep="\t", index=False)

    claims = [
        {
            "claim": "Pretreatment locked programs follow a CR-PR-NR response gradient",
            "supporting_phase": "Phase 1",
            "dataset": "CNP0004138 scRNA pseudobulk; GSE209746; GSE87211",
            "evidence_type": "patient-celltype pseudobulk association and external program projection",
            "statistical_unit": "patient or patient-celltype pseudobulk",
            "support_strength": "strong directional program support",
            "limitations": "not prospective and not a treatment-response biomarker validation cohort",
            "allowed_wording": "candidate treatment-response-associated program",
            "forbidden_wording": "clinical biomarker",
        },
        {
            "claim": "Locked resistance programs are higher in post-NAC NR spatial sections",
            "supporting_phase": "Phase 2",
            "dataset": "CNP0004138 Stereo-seq",
            "evidence_type": "section-pseudobulk score and spatial localization",
            "statistical_unit": "section; n=4",
            "support_strength": "consistent but small-n",
            "limitations": "two CR and two NR sections; two low-complexity sections; unresolved NR donor identity",
            "allowed_wording": "section-level spatial support",
            "forbidden_wording": "spatial validation proves clinical utility",
        },
        {
            "claim": "Canonical EMT/stress, myeloid remodeling, CD8 dysfunction, endothelial remodeling, and CAF marker modules are NR-higher",
            "supporting_phase": "Phase 3",
            "dataset": "CNP0004138 PRET patient-celltype pseudobulk",
            "evidence_type": "prespecified marker-module scoring",
            "statistical_unit": "patient-celltype pseudobulk",
            "support_strength": "not supported",
            "limitations": "marker modules may not capture the locked program biology",
            "allowed_wording": "canonical-state alignment was not observed",
            "forbidden_wording": "canonical EMT/remodeling/exhaustion mechanism established",
        },
        {
            "claim": "Multiple regulators are associated with the locked resistance program",
            "supporting_phase": "Phase 3",
            "dataset": "CNP0004138 PRET patient-celltype pseudobulk",
            "evidence_type": "DoRothEA/CollecTRI activity, prior/coexpression/tree GRN",
            "statistical_unit": "patient-celltype pseudobulk",
            "support_strength": "moderate computational nomination",
            "limitations": "VIPER unavailable; regulator-specific external validation absent",
            "allowed_wording": "multi-evidence candidate regulator",
            "forbidden_wording": "validated regulator",
        },
        {
            "claim": "In silico regulator perturbation can lower the locked resistance score",
            "supporting_phase": "Phase 3",
            "dataset": "CNP0004138 PRET pseudobulk and inferred GRN",
            "evidence_type": "two network stress-test simulations",
            "statistical_unit": "patient-celltype pseudobulk",
            "support_strength": "directional computational support",
            "limitations": "effects are constrained by inferred edges and are not experimental perturbations",
            "allowed_wording": "virtual perturbation suggests score reduction",
            "forbidden_wording": "knockout validation or causal proof",
        },
        {
            "claim": "A malignant-myeloid-CD8-endothelial communication niche is associated with resistance",
            "supporting_phase": "Phase 2 and Phase 3",
            "dataset": "CNP0004138 PRET pseudobulk and post-NAC Stereo-seq",
            "evidence_type": "LIANA consensus LR expression, receiver-score association, spatial proxy",
            "statistical_unit": "paired patient-celltype pseudobulk and section",
            "support_strength": "moderate hypothesis-generating",
            "limitations": "no physical interaction assay; spatial support is compartment-level proxy",
            "allowed_wording": "multicellular resistance-associated niche",
            "forbidden_wording": "ligand-receptor mechanism proven",
        },
        {
            "claim": "Candidate regulator expression localizes to Phase 2 resistance niches",
            "supporting_phase": "Phase 3",
            "dataset": "CNP0004138 Stereo-seq",
            "evidence_type": "within-section expression-resistance correlation and section expression direction",
            "statistical_unit": "section for response direction; bins descriptive",
            "support_strength": "candidate-specific mixed support",
            "limitations": "n=4 and TF expression is only an activity proxy",
            "allowed_wording": "spatially localized candidate",
            "forbidden_wording": "spatial data proves causality",
        },
        {
            "claim": "Phase 3 establishes a causal resistance mechanism",
            "supporting_phase": "Phase 3",
            "dataset": "all computational outputs",
            "evidence_type": "integrated computational nomination",
            "statistical_unit": "mixed, explicitly bounded",
            "support_strength": "not supported",
            "limitations": "zero high-confidence regulators and canonical-state direction mismatch",
            "allowed_wording": "requires experimental validation",
            "forbidden_wording": "causal mechanism proven or therapeutic target validated",
        },
    ]
    pd.DataFrame(claims).to_csv(out / "Phase3_claim_evidence_boundary_table.tsv", sep="\t", index=False)

    moderate = ranking[ranking.final_confidence == "moderate"].sort_values("final_integrated_score", ascending=False)
    top_text = ", ".join(f"{row.target_state}:{row.regulator}" for row in moderate.groupby("target_state").head(2).itertuples())
    model = f"""# Integrated mechanism model

## Supported architecture

Phase 1 identifies pretreatment, response-associated locked programs. Phase 2 places the same programs in post-NAC NR spatial sections and multicellular niches. Phase 3 nominates stable regulators and communication axes that may organize those locked programs.

Leading moderate computational nominations are: {top_text}.

The strongest non-CAF communication hypotheses include malignant LTA-TNFRSF1A/B/14 signaling toward myeloid/endothelial compartments and myeloid/endothelial HLA-I-KIR2DL3 associations with CD8 states.

## Critical contradiction

The five prespecified canonical marker-state modules did not show the expected NR-higher patient-pseudobulk direction. Therefore the data do not establish a canonical EMT, SPP1/APOE macrophage, exhausted-CD8, endothelial-remodeling, or CAF mechanism. Candidate regulators are tied to the locked resistance program, not validated state identities.

## Manuscript position

The defensible manuscript is a response-associated spatial-program study with computational regulator and niche nominations as secondary, hypothesis-generating evidence. A main causal mechanism figure is not justified without wet-lab perturbation and orthogonal state validation.
"""
    (out / "integrated_mechanism_model.md").write_text(model, encoding="utf-8")
    print("Integrated Phase 1-3 evidence and claim boundaries.")


if __name__ == "__main__":
    main()
