from __future__ import annotations

import pandas as pd

from phase3_config import INPUT, ROOT, TARGET_STATES


PROGRAM_MAP = {
    "Malignant_EMT_stress": ("Malignant_epithelial_specific", "malignant", "Malignant_epithelial_domain"),
    "Myeloid_suppressive_remodeling": ("Myeloid_macrophage_specific", "myeloid", "Myeloid_domain"),
    "CD8_dysfunctional_stress": ("CD8_T_specific", "CD8", "T_CD8_domain"),
    "Endothelial_remodeling": ("Endothelial_specific", "endothelial", "Endothelial_domain"),
    "CAF_supporting_comparator": ("CAF_Fibroblast_specific", "CAF", "CAF_fibroblast_domain"),
}


def main() -> None:
    contrasts = pd.read_csv(INPUT / "CNP0004138_CR_PR_NR_program_score_contrasts.tsv", sep="\t")
    localization = pd.read_csv(INPUT / "StereoSeq_CR_NR_localization_summary.tsv", sep="\t")
    rows = []
    for state, definition in TARGET_STATES.items():
        phase1_program, phase2_program, domain = PROGRAM_MAP[state]
        p1 = contrasts[(contrasts.program_name == phase1_program) & (contrasts.score == "resistance_score")].iloc[0]
        p2 = localization[(localization.program_name == phase2_program) & (localization.domain_module == domain)].iloc[0]
        rows.append({
            "target_state": state,
            "source_cell_type": definition["cell_type"],
            "priority_level": definition["priority"],
            "suitable_for_virtual_perturbation": definition["virtual_perturbation"],
            "phase1_NR_minus_CR_resistance_effect": p1.effect_NR_minus_CR,
            "phase1_CR_PR_NR_spearman_rho": p1.spearman_rho_CR_PR_NR,
            "phase1_primary_statistical_unit": p1.statistical_unit,
            "phase2_matched_domain": domain,
            "phase2_NR_minus_CR_localization_effect": p2.effect_NR_minus_CR_localization,
            "phase2_CAF_adjusted_localization_effect": p2.effect_NR_minus_CR_partial_localization,
            "phase2_both_NR_positive": p2.NR_sections_both_positive,
            "main_biological_hypothesis": definition["hypothesis"],
            "main_risks": definition["risk"],
        })
    table = pd.DataFrame(rows).sort_values("priority_level")
    out = ROOT / "02_target_state_definition"
    table.to_csv(out / "Phase3_target_state_table.tsv", sep="\t", index=False)
    rationale = """# Phase 3 target-state rationale

The five states are fixed before regulator analysis. Their order follows Phase 2 Go/No-Go and is not re-ranked using Phase 3 outcomes.

1. Malignant EMT/stress is the primary mechanism-nomination state because it combines the largest Phase 1 compartment effect with matched malignant spatial localization.
2. Myeloid suppressive/remodeling is the second priority because it has section-level, matched-domain, and multicellular-niche support.
3. CD8 dysfunctional/stress is retained as an immune-state hypothesis; cytotoxic activation and dysfunction are scored separately to prevent semantic collapse.
4. Endothelial remodeling has a strong section effect but smaller matched-domain localization and is therefore secondary.
5. CAF is a comparator/supporting state only. It cannot become the central mechanism unless evidence is clearly independent of the published CAF_FAP/MIR4435-2HG axis.

All states are response-associated candidate states. None is a validated biological subtype or causal mechanism.
"""
    (out / "Phase3_target_state_rationale.md").write_text(rationale, encoding="utf-8")
    print(f"Defined {len(table)} locked Phase 3 target states.")


if __name__ == "__main__":
    main()

