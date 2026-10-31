from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spatial_config import PROGRAM_LABELS, PROGRAM_ORDER, RESPONSE, SECTIONS, normalize_gene


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization")


def load_programs():
    table = pd.read_csv(ROOT / "01_inputs_from_phase1" / "CNP0004138_locked_programs_by_celltype.tsv", sep="\t")
    table["gene_symbol"] = table["gene_symbol"].map(normalize_gene)
    programs = {}
    for source in PROGRAM_ORDER:
        frame = table[table["source_cell_type"] == source]
        programs[PROGRAM_LABELS[source]] = {
            "NR_high": frame.loc[frame["program_direction"] == "NR_high", "gene_symbol"].drop_duplicates().tolist(),
            "CR_high": frame.loc[frame["program_direction"] == "CR_high", "gene_symbol"].drop_duplicates().tolist(),
        }
    return programs


def score_subset(expression: pd.DataFrame, sections: list[str], programs: dict):
    subset = expression[sections]
    sd = subset.std(axis=1, ddof=0).replace(0, np.nan)
    z = subset.sub(subset.mean(axis=1), axis=0).div(sd, axis=0).fillna(0)
    rows = []
    for label, genes in programs.items():
        nr = [g for g in genes["NR_high"] if g in z.index]
        cr = [g for g in genes["CR_high"] if g in z.index]
        score = z.loc[nr].mean(axis=0) - z.loc[cr].mean(axis=0)
        nr_sections = [section for section in sections if RESPONSE[section] == "NR"]
        cr_sections = [section for section in sections if RESPONSE[section] == "CR"]
        effect = score[nr_sections].mean() - score[cr_sections].mean()
        rows.append(
            {
                "program_name": label,
                "included_sections": ";".join(sections),
                "n_CR_sections": len(cr_sections),
                "n_NR_sections": len(nr_sections),
                "mean_CR_score": score[cr_sections].mean(),
                "mean_NR_score": score[nr_sections].mean(),
                "effect_NR_minus_CR": effect,
                "direction_support": effect > 0,
                "complete_separation": bool(score[nr_sections].min() > score[cr_sections].max()),
                "minimum_NR_score": score[nr_sections].min(),
                "maximum_CR_score": score[cr_sections].max(),
            }
        )
    return pd.DataFrame(rows)


def main():
    expression = pd.read_parquet(
        ROOT / "06_section_level_summary" / "StereoSeq_section_pseudobulk_logCPM.parquet"
    )
    programs = load_programs()
    frames = []
    full = score_subset(expression, SECTIONS, programs)
    full["sensitivity_scenario"] = "all_sections"
    full["removed_section"] = "none"
    frames.append(full)
    for removed in SECTIONS:
        included = [section for section in SECTIONS if section != removed]
        result = score_subset(expression, included, programs)
        result["sensitivity_scenario"] = f"leave_out_{removed}"
        result["removed_section"] = removed
        frames.append(result)
    sensitivity = pd.concat(frames, ignore_index=True)
    full_effect = full.set_index("program_name")["effect_NR_minus_CR"]
    sensitivity["full_effect_NR_minus_CR"] = sensitivity["program_name"].map(full_effect)
    sensitivity["effect_change_from_full"] = sensitivity["effect_NR_minus_CR"] - sensitivity["full_effect_NR_minus_CR"]
    sensitivity["sign_preserved_vs_full"] = np.sign(sensitivity["effect_NR_minus_CR"]) == np.sign(sensitivity["full_effect_NR_minus_CR"])
    sensitivity["primary_statistical_unit"] = "section"
    sensitivity.to_csv(
        ROOT / "10_sensitivity_analysis" / "StereoSeq_leave_one_section_sensitivity.tsv", sep="\t", index=False
    )

    nr2 = sensitivity[sensitivity["removed_section"] == "ST-NR2"].copy()
    qc = pd.read_csv(ROOT / "03_stereoseq_matrix" / "StereoSeq_matrix_QC_summary.tsv", sep="\t")
    nr2_qc = qc[qc["section_id"] == "ST-NR2"].iloc[0]
    preserved = int(nr2["sign_preserved_vs_full"].sum())
    positive = int(nr2["direction_support"].sum())
    (ROOT / "10_sensitivity_analysis" / "StereoSeq_ST_NR2_influence_report.md").write_text(
        "# ST-NR2 influence report\n\n"
        f"ST-NR2 contains {int(nr2_qc['n_bins']):,} bins with a median of {nr2_qc['detected_genes_median']:.0f} detected genes and is retained as low quality.\n\n"
        f"After removing ST-NR2 and recomputing gene standardization across the remaining sections, {preserved}/6 programs preserve the sign of the full effect and {positive}/6 retain the prespecified positive NR-minus-CR direction.\n\n"
        "The section is considered conclusion-driving only when removal reverses the global effect or most compartment programs. Bin counts are not used as replication.\n",
        encoding="utf-8",
    )

    leave_support = sensitivity[sensitivity["sensitivity_scenario"] != "all_sections"].groupby("program_name").agg(
        supported_scenarios=("direction_support", "sum"),
        sign_preserved_scenarios=("sign_preserved_vs_full", "sum"),
        min_effect=("effect_NR_minus_CR", "min"),
        max_effect=("effect_NR_minus_CR", "max"),
    )
    stable = leave_support[(leave_support["supported_scenarios"] == 4) & (leave_support["sign_preserved_scenarios"] == 4)].index.tolist()
    (ROOT / "10_sensitivity_analysis" / "StereoSeq_sensitivity_interpretation.md").write_text(
        "# Leave-one-section sensitivity interpretation\n\n"
        "All scores are recomputed after each section removal; the analysis does not merely delete a plotted point from the four-section normalization.\n\n"
        f"Programs retaining a positive direction in all four leave-one-section scenarios: {', '.join(stable) if stable else 'none'}.\n\n"
        "With only three sections per sensitivity scenario, these results measure direction stability rather than statistical significance.\n",
        encoding="utf-8",
    )
    leave_support.reset_index().to_csv(
        ROOT / "10_sensitivity_analysis" / "StereoSeq_leave_one_section_program_stability_summary.tsv", sep="\t", index=False
    )
    print("Completed leave-one-section and ST-NR2 sensitivity analyses.")


if __name__ == "__main__":
    main()
