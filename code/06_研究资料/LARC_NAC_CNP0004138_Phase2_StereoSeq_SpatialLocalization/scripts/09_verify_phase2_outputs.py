from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "00_logs"


def read_tsv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, sep="\t")


def main() -> None:
    required = read_tsv("00_logs/deliverable_index.tsv")
    coverage = read_tsv("04_program_gene_coverage/StereoSeq_locked_program_gene_coverage.tsv")
    effects = read_tsv("06_section_level_summary/StereoSeq_CR_vs_NR_section_effect_summary.tsv")
    sensitivity = read_tsv("10_sensitivity_analysis/StereoSeq_leave_one_section_sensitivity.tsv")
    metadata = read_tsv("02_spatial_metadata/StereoSeq_frozen_spatial_metadata.tsv")
    figure_qa = read_tsv("00_logs/Phase2_figure_machine_QA.tsv")
    report = (ROOT / "13_reports/FINAL_LARC_Phase2_spatial_localization_report.md").read_text(encoding="utf-8")

    leave_one = sensitivity[sensitivity["sensitivity_scenario"] != "all_sections"]
    checks = [
        ("minimum_deliverables_exist", bool(required["exists"].all()), f"{int(required.exists.sum())}/{len(required)}"),
        ("minimum_deliverables_nonempty", bool((required["bytes"] > 0).all()), "all bytes > 0"),
        ("coverage_rows", len(coverage) == 24, f"n={len(coverage)}; expected 6 programs x 4 sections"),
        ("coverage_high", bool((coverage["total_represented_fraction"] >= 0.70).all()), f"min={coverage.total_represented_fraction.min():.3f}"),
        ("six_program_effects", len(effects) == 6, f"n={len(effects)}"),
        ("all_primary_effects_positive", bool((effects["effect_NR_minus_CR"] > 0).all()), f"min={effects.effect_NR_minus_CR.min():.3f}"),
        ("complete_section_separation", bool(effects["complete_section_separation"].all()), f"{int(effects.complete_section_separation.sum())}/6"),
        ("leave_one_rows", len(leave_one) == 24, f"n={len(leave_one)}; expected 6 programs x 4 removals"),
        ("all_leave_one_effects_positive", bool((leave_one["effect_NR_minus_CR"] > 0).all()), f"min={leave_one.effect_NR_minus_CR.min():.3f}"),
        ("remove_ST_NR2_positive", bool((leave_one.loc[leave_one.removed_section == "ST-NR2", "effect_NR_minus_CR"] > 0).all()), "all six programs"),
        ("four_frozen_sections", len(metadata) == 4, f"n={len(metadata)}"),
        ("NR_donor_identity_flagged", metadata.loc[metadata.response_group == "NR", "exact_donor_identity_status"].astype(str).str.contains("unresolved", case=False).all(), "ST-NR1/ST-NR2"),
        ("figure_pairs", bool(figure_qa["matching_pdf"].all()), f"{int(figure_qa.matching_pdf.sum())}/{len(figure_qa)}"),
        ("figure_nonblank", bool(figure_qa["nonblank"].all()), f"{int(figure_qa.nonblank.sum())}/{len(figure_qa)}"),
        ("report_uses_primary_scores", all(token in report for token in ["-0.661", "-0.988", "1.339", "0.311"]), "expected global section values found"),
        ("report_has_evidence_boundary", "not a clinically validated biomarker" in report and "not prove prospective prediction" in report, "explicit boundary present"),
    ]
    output = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    output.to_csv(LOGS / "Phase2_final_verification.tsv", sep="\t", index=False)
    failed = output.loc[~output.passed]
    summary = [
        "# Phase 2 final verification",
        "",
        f"- Checks passed: {int(output.passed.sum())}/{len(output)}.",
        f"- Final status: {'PASS' if failed.empty else 'FAIL'}.",
        "- Verification covers required deliverables, locked-program coverage, section effects, leave-one stability, NR donor flagging, figure integrity, and report score provenance.",
    ]
    if not failed.empty:
        summary.extend(["", "## Failed checks", "", failed.to_markdown(index=False)])
    (LOGS / "Phase2_final_verification.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if not failed.empty:
        raise SystemExit(f"Verification failed: {', '.join(failed.check)}")
    print(f"Phase 2 final verification passed: {len(output)}/{len(output)} checks.")


if __name__ == "__main__":
    main()
