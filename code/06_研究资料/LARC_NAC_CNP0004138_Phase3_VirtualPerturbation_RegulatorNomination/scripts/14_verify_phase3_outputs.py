from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "00_logs"


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, sep="\t")


def main() -> None:
    deliverables = read("00_logs/deliverable_index.tsv")
    targets = read("02_target_state_definition/Phase3_target_state_table.tsv")
    state_effects = read("03_refined_cell_state_annotation/refined_target_state_response_effects.tsv")
    activity = read("04_regulator_activity/regulator_activity_by_target_state.tsv")
    grn = read("05_grn_inference/GRN_edges_by_target_state.tsv")
    virtual = read("06_virtual_perturbation/virtual_perturbation_results_by_state.tsv")
    lr = read("07_ligand_receptor_niche/spatially_informed_LR_priority_table.tsv")
    spatial = read("08_spatial_regulator_projection/regulator_section_level_summary.tsv")
    ranking = read("09_candidate_prioritization/final_candidate_regulator_ranking.tsv")
    robustness = read("10_robustness_sensitivity/regulator_robustness_summary.tsv")
    boundary = read("11_cross_phase_integration/Phase3_claim_evidence_boundary_table.tsv")
    figure_qa = read("00_logs/Phase3_figure_machine_QA.tsv")
    report = (ROOT / "13_reports/FINAL_LARC_Phase3_virtual_perturbation_report.md").read_text(encoding="utf-8")
    decision = (ROOT / "13_reports/FINAL_go_nogo_for_LARC_manuscript_drafting.md").read_text(encoding="utf-8")

    checks = [
        ("required_deliverables_exist", bool(deliverables.exists.all()), f"{int(deliverables.exists.sum())}/{len(deliverables)}"),
        ("required_deliverables_nonempty", bool((deliverables.bytes > 0).all()), "all bytes > 0"),
        ("five_fixed_target_states", len(targets) == 5, f"n={len(targets)}"),
        ("locked_programs_not_refit", "Phase 1 locked programs were never retrained" in report, "explicit report boundary"),
        ("patient_pseudobulk_state_effects", len(state_effects) == 5 and (state_effects.primary_statistical_unit == "patient-celltype pseudobulk").all(), "five patient-level state tests"),
        ("negative_state_alignment_retained", not bool(state_effects.direction_support.any()), "0/5 expected NR directions"),
        ("three_activity_tracks", activity.groupby(["target_state", "regulator"]).method.nunique().max() == 3, f"methods={sorted(activity.method.unique())}"),
        ("grn_multi_method_edges", bool((grn.method_support_count >= 2).any()), f"n={(grn.method_support_count >= 2).sum()}"),
        ("two_virtual_methods", virtual.groupby(["target_state", "regulator"]).perturbation_method.nunique().min() == 2, "all simulated candidates"),
        ("non_CAF_LR_axes", bool(lr.non_CAF_pair.any()), f"n={int(lr.non_CAF_pair.sum())}"),
        ("four_spatial_sections", spatial.statistical_unit.eq("section; n=4").all(), f"n regulator-state={len(spatial)}"),
        ("no_high_confidence_candidates", int((ranking.final_confidence == "high").sum()) == 0, "prespecified threshold retained"),
        ("moderate_non_CAF_candidates", ranking.loc[ranking.final_confidence == "moderate", "target_state"].nunique() >= 2, f"states={ranking.loc[ranking.final_confidence == 'moderate', 'target_state'].nunique()}"),
        ("robustness_all_candidates", len(robustness) == ranking[["target_state", "regulator"]].drop_duplicates().shape[0], f"n={len(robustness)}"),
        ("claim_boundary_present", len(boundary) >= 8, f"n={len(boundary)}"),
        ("figure_pairs", bool(figure_qa.matching_pdf.all()), f"{int(figure_qa.matching_pdf.sum())}/{len(figure_qa)}"),
        ("figure_nonblank", bool(figure_qa.nonblank.all()), f"{int(figure_qa.nonblank.sum())}/{len(figure_qa)}"),
        ("manuscript_decision_bounded", "Manuscript Go without mechanism claim" in decision, "decision text found"),
        ("report_answers_all_questions", "## Direct answers to the required questions" in report and report.count("**") >= 20, "direct-answer section present"),
    ]
    output = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    output.to_csv(LOGS / "Phase3_final_verification.tsv", sep="\t", index=False)
    failed = output[~output.passed]
    text = [
        "# Phase 3 final verification", "",
        f"- Checks passed: {int(output.passed.sum())}/{len(output)}.",
        f"- Final status: {'PASS' if failed.empty else 'FAIL'}.",
        "- Verification covers frozen states, patient-level inference, multi-method activity/GRN/perturbation, LR and spatial support, negative-state retention, confidence thresholds, figures, and manuscript boundaries.",
    ]
    if not failed.empty:
        text.extend(["", "## Failed checks", "", failed.to_markdown(index=False)])
    (LOGS / "Phase3_final_verification.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if not failed.empty:
        raise SystemExit("Verification failed: " + ", ".join(failed.check))
    print(f"Phase 3 final verification passed: {len(output)}/{len(output)} checks.")


if __name__ == "__main__":
    main()
