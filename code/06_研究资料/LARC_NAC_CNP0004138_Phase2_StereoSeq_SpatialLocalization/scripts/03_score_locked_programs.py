from __future__ import annotations

import itertools
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
        programs[source] = {
            "NR_high": frame.loc[frame["program_direction"] == "NR_high", "gene_symbol"].drop_duplicates().tolist(),
            "CR_high": frame.loc[frame["program_direction"] == "CR_high", "gene_symbol"].drop_duplicates().tolist(),
        }
    return programs


def load_section(section: str):
    archive = np.load(
        ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_analysis_gene_counts.npz",
        allow_pickle=False,
    )
    genes = archive["genes"].astype(str)
    bins = archive["bins"].astype(str)
    counts = archive["counts"].astype(np.float32)
    total_counts = archive["total_counts"].astype(np.float64)
    return genes, bins, counts, total_counts


def gene_zscores(counts: np.ndarray, total_counts: np.ndarray):
    scale = np.divide(10_000.0, total_counts, out=np.zeros_like(total_counts), where=total_counts > 0)
    lognorm = np.log1p(counts * scale[None, :]).astype(np.float32)
    means = lognorm.mean(axis=1, keepdims=True)
    sd = lognorm.std(axis=1, keepdims=True)
    z = np.divide(lognorm - means, sd, out=np.zeros_like(lognorm), where=sd > 0)
    return lognorm, z


def module_mean(z: np.ndarray, index: dict[str, int], genes: list[str]):
    represented = [index[gene] for gene in genes if gene in index]
    if not represented:
        return np.full(z.shape[1], np.nan, dtype=np.float32)
    return z[represented].mean(axis=0)


def exact_label_permutation(values: dict[str, float]):
    sections = list(values)
    observed_nr = [s for s in sections if RESPONSE[s] == "NR"]
    observed_cr = [s for s in sections if RESPONSE[s] == "CR"]
    observed = np.mean([values[s] for s in observed_nr]) - np.mean([values[s] for s in observed_cr])
    permutations = []
    for nr_set in itertools.combinations(sections, 2):
        cr_set = [s for s in sections if s not in nr_set]
        permutations.append(np.mean([values[s] for s in nr_set]) - np.mean([values[s] for s in cr_set]))
    p_two_sided = np.mean(np.abs(permutations) >= abs(observed) - 1e-12)
    return observed, p_two_sided, permutations


def hedges_g(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    correction = 1 - 3 / (4 * (len(x) + len(y)) - 9)
    return correction * (x.mean() - y.mean()) / pooled


def main():
    programs = load_programs()
    coverage_rows = []
    missing_rows = []
    section_map_summaries = []
    pseudobulk_logcpm = {}
    output_scores = ROOT / "05_spatial_program_scores" / "StereoSeq_bin_level_program_scores.tsv"
    output_scores.unlink(missing_ok=True)

    for section_i, section in enumerate(SECTIONS):
        print(f"Scoring {section}...", flush=True)
        genes, bins, counts, total_counts = load_section(section)
        index = {gene: i for i, gene in enumerate(genes)}
        _, z = gene_zscores(counts, total_counts)
        coordinates = pd.read_parquet(
            ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_bin_qc.parquet"
        ).reindex(bins)
        score_frame = pd.DataFrame(
            {
                "section_id": section,
                "response_group": RESPONSE[section],
                "bin_id": bins,
                "x": coordinates["x"].to_numpy(),
                "y": coordinates["y"].to_numpy(),
                "total_counts": total_counts,
                "detected_genes": coordinates["detected_genes"].to_numpy(),
            }
        )
        for source in PROGRAM_ORDER:
            label = PROGRAM_LABELS[source]
            nr_all = programs[source]["NR_high"]
            cr_all = programs[source]["CR_high"]
            nr_present = [gene for gene in nr_all if gene in index]
            cr_present = [gene for gene in cr_all if gene in index]
            nr = module_mean(z, index, nr_all)
            cr = module_mean(z, index, cr_all)
            resistance = nr - cr
            score_frame[f"{label}_NR_high_score"] = nr
            score_frame[f"{label}_CR_high_score"] = cr
            score_frame[f"{label}_resistance_score"] = resistance
            score_frame[f"{label}_response_score"] = -resistance
            coverage_fraction = (len(nr_present) + len(cr_present)) / (len(nr_all) + len(cr_all))
            nr_fraction = len(nr_present) / max(len(nr_all), 1)
            cr_fraction = len(cr_present) / max(len(cr_all), 1)
            coverage_rows.append(
                {
                    "program_name": label,
                    "source_cell_type": source,
                    "section_id": section,
                    "response_group": RESPONSE[section],
                    "n_NR_high_total": len(nr_all),
                    "n_NR_high_represented": len(nr_present),
                    "n_CR_high_total": len(cr_all),
                    "n_CR_high_represented": len(cr_present),
                    "total_represented_fraction": coverage_fraction,
                    "NR_high_represented_fraction": nr_fraction,
                    "CR_high_represented_fraction": cr_fraction,
                    "coverage_imbalance_absolute": abs(nr_fraction - cr_fraction),
                    "confidence": "high" if coverage_fraction >= 0.70 else "medium" if coverage_fraction >= 0.50 else "low",
                    "strong_conclusion_eligible": coverage_fraction >= 0.50,
                }
            )
            for direction, all_genes, present in [("NR_high", nr_all, nr_present), ("CR_high", cr_all, cr_present)]:
                for gene in sorted(set(all_genes) - set(present)):
                    missing_rows.append(
                        {
                            "section_id": section,
                            "program_name": label,
                            "source_cell_type": source,
                            "program_direction": direction,
                            "missing_gene": gene,
                        }
                    )
            section_map_summaries.append(
                {
                    "section_id": section,
                    "response_group": RESPONSE[section],
                    "program_name": label,
                    "source_cell_type": source,
                    "mean_within_section_NR_high_score": float(np.nanmean(nr)),
                    "mean_within_section_CR_high_score": float(np.nanmean(cr)),
                    "mean_within_section_resistance_score": float(np.nanmean(resistance)),
                    "median_within_section_resistance_score": float(np.nanmedian(resistance)),
                    "within_section_resistance_score_sd": float(np.nanstd(resistance, ddof=1)),
                    "within_section_resistance_q10": float(np.nanquantile(resistance, 0.10)),
                    "within_section_resistance_q90": float(np.nanquantile(resistance, 0.90)),
                    "n_bins": len(bins),
                    "n_genes_represented": len(set(nr_present + cr_present)),
                    "within_section_score_role": "descriptive map only; mean is constrained near zero by within-section gene z-scoring",
                }
            )

        score_frame.to_csv(output_scores, sep="\t", index=False, mode="a", header=section_i == 0)
        score_frame.to_parquet(
            ROOT / "05_spatial_program_scores" / f"{section.replace('-', '_')}_bin_level_program_scores.parquet",
            index=False,
        )
        section_sum = counts.sum(axis=1, dtype=np.float64)
        section_library = float(total_counts.sum())
        pseudobulk_logcpm[section] = pd.Series(
            np.log2(section_sum / section_library * 1_000_000 + 1), index=genes, name=section
        )

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(
        ROOT / "04_program_gene_coverage" / "StereoSeq_locked_program_gene_coverage.tsv", sep="\t", index=False
    )
    pd.DataFrame(missing_rows).to_csv(
        ROOT / "04_program_gene_coverage" / "StereoSeq_missing_program_genes.tsv", sep="\t", index=False
    )
    warnings = []
    for _, row in coverage.iterrows():
        if row["confidence"] != "high" or row["coverage_imbalance_absolute"] > 0.15:
            warnings.append(
                f"- {row['section_id']} / {row['program_name']}: coverage={row['total_represented_fraction']:.3f}, imbalance={row['coverage_imbalance_absolute']:.3f}, confidence={row['confidence']}."
            )
    warning_text = "\n".join(warnings) if warnings else "- No program-section pair fell below high coverage or exceeded 0.15 directional imbalance."
    (ROOT / "04_program_gene_coverage" / "StereoSeq_coverage_warning_report.md").write_text(
        "# Stereo-seq locked-program coverage warning report\n\n"
        "Programs were imported unchanged from Phase 1. High coverage is >=0.70, medium is 0.50-<0.70, and low is <0.50.\n\n"
        + warning_text
        + "\n",
        encoding="utf-8",
    )

    # Primary section comparison: aggregate counts first, then standardize each gene across the four sections.
    expression = pd.concat(pseudobulk_logcpm.values(), axis=1).fillna(0)
    expression.to_parquet(ROOT / "06_section_level_summary" / "StereoSeq_section_pseudobulk_logCPM.parquet")
    gene_sd = expression.std(axis=1, ddof=0).replace(0, np.nan)
    section_gene_z = expression.sub(expression.mean(axis=1), axis=0).div(gene_sd, axis=0).fillna(0)
    section_gene_z.to_parquet(ROOT / "06_section_level_summary" / "StereoSeq_section_pseudobulk_gene_zscores.parquet")
    section_rows = []
    map_summary = pd.DataFrame(section_map_summaries)
    for source in PROGRAM_ORDER:
        label = PROGRAM_LABELS[source]
        nr_genes = [g for g in programs[source]["NR_high"] if g in section_gene_z.index]
        cr_genes = [g for g in programs[source]["CR_high"] if g in section_gene_z.index]
        nr_score = section_gene_z.loc[nr_genes].mean(axis=0)
        cr_score = section_gene_z.loc[cr_genes].mean(axis=0)
        for section in SECTIONS:
            mapped = map_summary[(map_summary["section_id"] == section) & (map_summary["program_name"] == label)].iloc[0]
            section_rows.append(
                {
                    "section_id": section,
                    "response_group": RESPONSE[section],
                    "program_name": label,
                    "source_cell_type": source,
                    "section_pseudobulk_CR_high_score": cr_score[section],
                    "section_pseudobulk_NR_high_score": nr_score[section],
                    "section_pseudobulk_resistance_score": nr_score[section] - cr_score[section],
                    "section_pseudobulk_response_score": cr_score[section] - nr_score[section],
                    "mean_CR_high_score": mapped["mean_within_section_CR_high_score"],
                    "mean_NR_high_score": mapped["mean_within_section_NR_high_score"],
                    "mean_resistance_score": mapped["mean_within_section_resistance_score"],
                    "mean_response_score": -mapped["mean_within_section_resistance_score"],
                    "median_resistance_score": mapped["median_within_section_resistance_score"],
                    "score_sd": mapped["within_section_resistance_score_sd"],
                    "n_bins": mapped["n_bins"],
                    "n_genes_represented": mapped["n_genes_represented"],
                    "QC_status": "low_quality" if section == "ST-NR2" else "pass",
                    "primary_section_score": "section_pseudobulk_resistance_score",
                }
            )
    section_table = pd.DataFrame(section_rows)
    section_table.to_csv(
        ROOT / "06_section_level_summary" / "StereoSeq_section_level_program_scores.tsv", sep="\t", index=False
    )

    effects = []
    for label, frame in section_table.groupby("program_name", sort=False):
        values = frame.set_index("section_id")["section_pseudobulk_resistance_score"].to_dict()
        difference, permutation_p, permutations = exact_label_permutation(values)
        nr = frame.loc[frame["response_group"] == "NR", "section_pseudobulk_resistance_score"].to_numpy()
        cr = frame.loc[frame["response_group"] == "CR", "section_pseudobulk_resistance_score"].to_numpy()
        pairwise = np.array([n > c for n in nr for c in cr])
        effects.append(
            {
                "program_name": label,
                "n_CR_sections": len(cr),
                "n_NR_sections": len(nr),
                "mean_CR_resistance_score": cr.mean(),
                "mean_NR_resistance_score": nr.mean(),
                "effect_NR_minus_CR": difference,
                "hedges_g_NR_minus_CR": hedges_g(nr, cr),
                "exact_label_permutation_p_two_sided": permutation_p,
                "NR_above_CR_pairwise_fraction": pairwise.mean(),
                "complete_section_separation": bool(nr.min() > cr.max()),
                "direction_support": difference > 0,
                "inference_boundary": "four sections; effect size and consistency are primary; permutation p has coarse resolution",
            }
        )
    pd.DataFrame(effects).to_csv(
        ROOT / "06_section_level_summary" / "StereoSeq_CR_vs_NR_section_effect_summary.tsv", sep="\t", index=False
    )

    (ROOT / "05_spatial_program_scores" / "StereoSeq_program_score_method_note.md").write_text(
        "# Locked-program scoring method\n\n"
        "- Gene membership and NR-high/CR-high direction were imported unchanged from Phase 1.\n"
        "- Bin maps use log1p(CP10K), followed by gene-wise z-scoring within each section. These scores are descriptive spatial observations only.\n"
        "- A within-section gene z-score has mean zero by construction, so its section mean cannot test CR versus NR.\n"
        "- Primary section comparisons therefore aggregate counts within each section, calculate section logCPM, standardize each gene across the four sections, and then compute locked module means.\n"
        "- The inferential unit is the section. No bin-level P value is used as response evidence.\n",
        encoding="utf-8",
    )
    print("Completed locked-program coverage, bin scoring, and section-level effects.")


if __name__ == "__main__":
    main()
