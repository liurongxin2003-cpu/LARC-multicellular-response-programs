from __future__ import annotations

import numpy as np
import pandas as pd

from phase3_config import ROOT


RECEIVER_STATE = {
    "Malignant_epithelial": "Malignant_EMT_stress",
    "Myeloid_macrophage": "Myeloid_suppressive_remodeling",
    "CD8_T": "CD8_dysfunctional_stress",
    "Endothelial": "Endothelial_remodeling",
    "CAF_Fibroblast": "CAF_supporting_comparator",
}

KNOWN_CAF_REGULATORS = {"FOSL1", "JUN", "FOS", "SMAD3", "STAT3", "TEAD1", "TEAD4", "YAP1", "WWTR1"}


def main() -> None:
    out = ROOT / "09_candidate_prioritization"
    activity = pd.read_csv(ROOT / "08_spatial_regulator_projection/pre_spatial_candidate_shortlist.tsv", sep="\t")
    overlap = pd.read_csv(ROOT / "05_grn_inference/regulator_NR_high_target_overlap.tsv", sep="\t")
    virtual = pd.read_csv(ROOT / "06_virtual_perturbation/virtual_perturbation_ranked_regulators.tsv", sep="\t")
    spatial = pd.read_csv(ROOT / "08_spatial_regulator_projection/regulator_section_level_summary.tsv", sep="\t")
    stability = pd.read_csv(ROOT / "10_robustness_sensitivity/regulator_robustness_summary.tsv", sep="\t")
    state_effect = pd.read_csv(ROOT / "03_refined_cell_state_annotation/refined_target_state_response_effects.tsv", sep="\t")
    lr = pd.read_csv(ROOT / "07_ligand_receptor_niche/spatially_informed_LR_priority_table.tsv", sep="\t")
    lr["target_state"] = lr.receiver_cell_type.map(RECEIVER_STATE)
    lr_support = lr.groupby(["target_state", "best_receiver_regulator"], as_index=False).agg(
        LR_axis_count=("ligand", "size"),
        maximum_LR_priority=("LR_priority_score", "max"),
        median_ligand_regulator_rho=("ligand_receiver_regulator_rho", "median"),
    ).rename(columns={"best_receiver_regulator": "regulator"})

    data = activity.merge(overlap, on=["target_state", "regulator"], how="left").merge(
        virtual, on=["target_state", "source_cell_type", "regulator"], how="left", suffixes=("", "_virtual")
    ).merge(spatial, on=["target_state", "regulator"], how="left").merge(
        stability, on=["target_state", "regulator"], how="left"
    ).merge(state_effect[["target_state", "direction_support", "effect_NR_minus_CR"]].rename(columns={
        "direction_support": "canonical_marker_state_NR_direction_support",
        "effect_NR_minus_CR": "canonical_marker_state_NR_minus_CR_effect",
    }), on="target_state", how="left").merge(lr_support, on=["target_state", "regulator"], how="left")
    data[["LR_axis_count", "maximum_LR_priority", "median_ligand_regulator_rho"]] = data[["LR_axis_count", "maximum_LR_priority", "median_ligand_regulator_rho"]].fillna(0)

    data["evidence_from_regulator_activity"] = (data.n_activity_methods >= 2) & (data.positive_method_fraction >= 2 / 3) & (data.median_resistance_correlation > 0)
    data["evidence_from_GRN"] = (data.two_or_more_method_NR_edges >= 2) | (data.prior_NR_high_overlap >= 2)
    data["evidence_from_virtual_perturbation"] = (data.perturbation_method_support_fraction >= 0.5) & (data.median_delta_resistance_NR < 0)
    data["evidence_from_ligand_receptor"] = data.LR_axis_count > 0
    data["evidence_from_spatial_projection"] = data.spatial_support.fillna(False)
    data["response_gradient_support"] = data.median_gradient_rho > 0
    data["independent_validation_support"] = "program-level GSE209746/GSE87211 support only; no regulator-specific validation"
    data["overlap_with_known_source_CAF_FAP_axis"] = (data.target_state == "CAF_supporting_comparator") | data.regulator.isin(KNOWN_CAF_REGULATORS)
    data["novelty_flag"] = np.where(~data.overlap_with_known_source_CAF_FAP_axis & (data.target_state != "CAF_supporting_comparator"), "non_CAF_candidate", "known_CAF_axis_overlap_or_context")
    evidence_columns = [
        "evidence_from_regulator_activity", "evidence_from_GRN", "evidence_from_virtual_perturbation",
        "evidence_from_ligand_receptor", "evidence_from_spatial_projection", "response_gradient_support",
    ]
    data["n_positive_evidence_categories"] = data[evidence_columns].sum(axis=1)
    data["stability_score"] = data.overall_stability_score
    data["final_integrated_score"] = (
        data.activity_evidence_score.fillna(0)
        + data.virtual_perturbation_score.fillna(0)
        + data.groupby("target_state")["two_or_more_method_NR_edges"].rank(pct=True).fillna(0)
        + data.groupby("target_state")["section_effect_NR_minus_CR"].rank(pct=True).fillna(0)
        + data.stability_score.fillna(0)
        + data.maximum_LR_priority.fillna(0)
    ) / 6

    high = (
        (data.n_positive_evidence_categories >= 5)
        & (data.stability_score >= 0.75)
        & data.evidence_from_virtual_perturbation
        & data.evidence_from_spatial_projection
        & data.canonical_marker_state_NR_direction_support
        & (data.target_state != "CAF_supporting_comparator")
    )
    moderate = (
        (data.n_positive_evidence_categories >= 4)
        & (data.stability_score >= 0.60)
        & data.evidence_from_virtual_perturbation
        & (data.target_state != "CAF_supporting_comparator")
    )
    data["final_confidence"] = np.select([high, moderate], ["high", "moderate"], default="low")
    data["allowed_claim"] = np.where(
        data.final_confidence == "high",
        "high-confidence computationally nominated regulator; requires experimental validation",
        np.where(data.final_confidence == "moderate", "multi-evidence candidate regulator of the locked resistance program", "exploratory regulator nomination only"),
    )
    data["forbidden_claim"] = "validated regulator; causal mechanism proven; knockout validation; therapeutic target validated; clinically actionable"
    data = data.sort_values(["final_confidence", "final_integrated_score"], ascending=[True, False], key=lambda s: s.map({"high": 2, "moderate": 1, "low": 0}) if s.name == "final_confidence" else s)

    required = [
        "regulator", "target_state", "evidence_from_regulator_activity", "evidence_from_GRN",
        "evidence_from_virtual_perturbation", "evidence_from_ligand_receptor",
        "evidence_from_spatial_projection", "response_gradient_support", "independent_validation_support",
        "overlap_with_known_source_CAF_FAP_axis", "novelty_flag", "stability_score", "final_confidence",
        "allowed_claim", "forbidden_claim",
    ]
    remaining = [column for column in data.columns if column not in required]
    data[required + remaining].to_csv(out / "final_candidate_regulator_ranking.tsv", sep="\t", index=False)

    caf_check = data[["regulator", "target_state", "overlap_with_known_source_CAF_FAP_axis", "novelty_flag", "remaining_edge_fraction", "CAF_axis_exclusion_support", "final_confidence"]]
    caf_check.to_csv(out / "known_CAF_axis_overlap_check.tsv", sep="\t", index=False)

    high_data = data[data.final_confidence == "high"]
    high_lines = ["# High-confidence candidate regulators", ""]
    if high_data.empty:
        high_lines.extend([
            "No regulator met the prespecified high-confidence threshold.",
            "",
            "The decisive unmet criterion was canonical target-state direction: all five predefined marker-based target-state modules failed to show the expected NR-higher patient-pseudobulk direction. Multi-evidence regulators can therefore be nominated for the locked resistance program, but not elevated to a validated EMT/remodeling/exhaustion mechanism.",
        ])
    else:
        for row in high_data.itertuples():
            high_lines.append(f"- {row.target_state}: {row.regulator}, integrated score {row.final_integrated_score:.3f}.")
    (out / "high_confidence_candidate_regulators.md").write_text("\n".join(high_lines) + "\n", encoding="utf-8")

    moderate_data = data[data.final_confidence == "moderate"].groupby("target_state").head(10)
    low_data = data[data.final_confidence == "low"].groupby("target_state").head(5)
    lines = ["# Moderate- and low-confidence candidates", "", "## Moderate candidates", ""]
    for row in moderate_data.itertuples():
        lines.append(f"- {row.target_state}: {row.regulator}; score {row.final_integrated_score:.3f}, evidence categories {row.n_positive_evidence_categories}/6, stability {row.stability_score:.3f}.")
    lines.extend(["", "## Leading exploratory candidates", ""])
    for row in low_data.itertuples():
        lines.append(f"- {row.target_state}: {row.regulator}; score {row.final_integrated_score:.3f}, evidence categories {row.n_positive_evidence_categories}/6.")
    lines.extend(["", "Moderate and low classifications are computational nominations and require experimental validation."])
    (out / "moderate_low_confidence_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Ranked {len(data)} regulator-state candidates: {len(high_data)} high, {len(moderate_data)} moderate shown, {int((data.final_confidence == 'low').sum())} low.")


if __name__ == "__main__":
    main()
