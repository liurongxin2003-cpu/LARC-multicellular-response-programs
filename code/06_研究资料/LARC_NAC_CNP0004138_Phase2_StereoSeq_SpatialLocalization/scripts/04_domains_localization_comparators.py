from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.cluster import KMeans

from spatial_config import (
    COMPARATOR_MODULES,
    DOMAIN_MODULES,
    PROGRAM_LABELS,
    PROGRAM_ORDER,
    RESPONSE,
    SECTIONS,
    normalize_gene,
)


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization")


MATCHED_DOMAIN = {
    "global": "Resistance_multicellular_niche",
    "malignant": "Malignant_epithelial_domain",
    "CAF": "CAF_fibroblast_domain",
    "myeloid": "Myeloid_domain",
    "CD8": "T_CD8_domain",
    "endothelial": "Endothelial_domain",
}


def load_section(section: str):
    archive = np.load(
        ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_analysis_gene_counts.npz",
        allow_pickle=False,
    )
    genes = archive["genes"].astype(str)
    bins = archive["bins"].astype(str)
    counts = archive["counts"].astype(np.float32)
    total_counts = archive["total_counts"].astype(np.float64)
    scale = np.divide(10_000.0, total_counts, out=np.zeros_like(total_counts), where=total_counts > 0)
    lognorm = np.log1p(counts * scale[None, :]).astype(np.float32)
    means = lognorm.mean(axis=1, keepdims=True)
    sd = lognorm.std(axis=1, keepdims=True)
    z = np.divide(lognorm - means, sd, out=np.zeros_like(lognorm), where=sd > 0)
    return genes, bins, z


def module_score(z, index, gene_list):
    represented = [index[normalize_gene(gene)] for gene in gene_list if normalize_gene(gene) in index]
    if not represented:
        return np.full(z.shape[1], np.nan, dtype=np.float32), 0
    return z[represented].mean(axis=0), len(represented)


def safe_spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan, np.nan
    result = spearmanr(x[valid], y[valid])
    return result.statistic, result.pvalue


def partial_spearman(x, y, control):
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if valid.sum() < 4:
        return np.nan
    xr, yr, cr = rankdata(x[valid]), rankdata(y[valid]), rankdata(control[valid])
    design = np.column_stack([np.ones(valid.sum()), cr])
    x_resid = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_resid = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    if x_resid.std() == 0 or y_resid.std() == 0:
        return np.nan
    return pearsonr(x_resid, y_resid).statistic


def main():
    domain_output = ROOT / "07_spatial_domain_annotation" / "StereoSeq_spatial_domain_scores.tsv"
    annotation_output = ROOT / "07_spatial_domain_annotation" / "StereoSeq_spatial_domain_annotation.tsv"
    comparator_output = ROOT / "09_response_group_comparison" / "StereoSeq_comparator_module_scores.tsv"
    for path in [domain_output, annotation_output, comparator_output]:
        path.unlink(missing_ok=True)

    correlation_rows = []
    localization_rows = []
    comparator_corr_rows = []
    niche_rows = []
    section_comparator_means = []

    for section_i, section in enumerate(SECTIONS):
        print(f"Annotating domains and localization in {section}...", flush=True)
        genes, bins, z = load_section(section)
        index = {gene: i for i, gene in enumerate(genes)}
        program_scores = pd.read_parquet(
            ROOT / "05_spatial_program_scores" / f"{section.replace('-', '_')}_bin_level_program_scores.parquet"
        ).set_index("bin_id").reindex(bins)

        domain = pd.DataFrame({"section_id": section, "response_group": RESPONSE[section], "bin_id": bins})
        domain_representation = {}
        for name, module in DOMAIN_MODULES.items():
            domain[name], domain_representation[name] = module_score(z, index, module)
        domain["Resistance_multicellular_niche"] = domain[
            ["Malignant_epithelial_domain", "CAF_fibroblast_domain", "Myeloid_domain", "Endothelial_domain"]
        ].mean(axis=1)

        cluster_features = domain[list(DOMAIN_MODULES)].fillna(0).to_numpy()
        n_clusters = 6 if len(domain) >= 100 else 3
        cluster = KMeans(n_clusters=n_clusters, random_state=4138, n_init=10).fit_predict(cluster_features)
        domain["unsupervised_domain_cluster"] = cluster
        marker_columns = list(DOMAIN_MODULES)
        dominant = domain[marker_columns].idxmax(axis=1)
        maximum = domain[marker_columns].max(axis=1)
        domain["marker_domain_annotation"] = dominant.where(maximum > 0, "Mixed_or_low_signal")
        domain.to_csv(domain_output, sep="\t", index=False, mode="a", header=section_i == 0)
        domain.to_parquet(
            ROOT / "07_spatial_domain_annotation" / f"{section.replace('-', '_')}_spatial_domain_scores.parquet",
            index=False,
        )

        annotation = domain[
            ["section_id", "response_group", "bin_id", "unsupervised_domain_cluster", "marker_domain_annotation"]
        ].copy()
        annotation.to_csv(annotation_output, sep="\t", index=False, mode="a", header=section_i == 0)

        comparator = pd.DataFrame({"section_id": section, "response_group": RESPONSE[section], "bin_id": bins})
        comparator_representation = {}
        for name, module in COMPARATOR_MODULES.items():
            comparator[name], comparator_representation[name] = module_score(z, index, module)
        comparator.to_csv(comparator_output, sep="\t", index=False, mode="a", header=section_i == 0)
        comparator.to_parquet(
            ROOT / "09_response_group_comparison" / f"{section.replace('-', '_')}_comparator_module_scores.parquet",
            index=False,
        )
        for name in COMPARATOR_MODULES:
            section_comparator_means.append(
                {
                    "section_id": section,
                    "response_group": RESPONSE[section],
                    "module_name": name,
                    "mean_within_section_score": comparator[name].mean(),
                    "median_within_section_score": comparator[name].median(),
                    "n_genes_represented": comparator_representation[name],
                    "role": "within-section descriptive map summary",
                }
            )

        for source in PROGRAM_ORDER:
            label = PROGRAM_LABELS[source]
            resistance = program_scores[f"{label}_resistance_score"].to_numpy()
            caf_control = comparator["CAF_FAP_source"].to_numpy()
            for domain_name in list(DOMAIN_MODULES) + ["Resistance_multicellular_niche"]:
                domain_values = domain[domain_name].to_numpy()
                rho, bin_p = safe_spearman(resistance, domain_values)
                partial = partial_spearman(resistance, domain_values, caf_control)
                correlation_rows.append(
                    {
                        "section_id": section,
                        "response_group": RESPONSE[section],
                        "program_name": label,
                        "source_cell_type": source,
                        "domain_module": domain_name,
                        "spearman_rho": rho,
                        "bin_level_p_descriptive_only": bin_p,
                        "partial_spearman_controlling_CAF_FAP": partial,
                        "n_bins": len(bins),
                        "domain_genes_represented": domain_representation.get(domain_name, sum(domain_representation.values())),
                        "statistical_boundary": "bin correlation is descriptive; section correlation is the response-level observation",
                    }
                )
                localization_rows.append(
                    {
                        "section_id": section,
                        "response_group": RESPONSE[section],
                        "program_name": label,
                        "domain_module": domain_name,
                        "localization_strength_spearman": rho,
                        "partial_localization_controlling_CAF_FAP": partial,
                        "n_bins": len(bins),
                    }
                )

            for comparator_name in COMPARATOR_MODULES:
                rho, bin_p = safe_spearman(resistance, comparator[comparator_name].to_numpy())
                comparator_corr_rows.append(
                    {
                        "section_id": section,
                        "response_group": RESPONSE[section],
                        "program_name": label,
                        "comparator_module": comparator_name,
                        "spearman_rho": rho,
                        "bin_level_p_descriptive_only": bin_p,
                        "n_bins": len(bins),
                    }
                )

            matched = MATCHED_DOMAIN[label]
            matched_values = domain[matched].to_numpy()
            program_high = resistance >= np.nanquantile(resistance, 0.80)
            domain_high = matched_values >= np.nanquantile(matched_values, 0.80)
            observed = np.mean(program_high & domain_high)
            expected = np.mean(program_high) * np.mean(domain_high)
            niche_rows.append(
                {
                    "section_id": section,
                    "response_group": RESPONSE[section],
                    "program_name": label,
                    "matched_domain": matched,
                    "high_score_quantile": 0.80,
                    "observed_joint_fraction": observed,
                    "expected_joint_fraction_independence": expected,
                    "same_bin_colocalization_enrichment": observed / expected if expected > 0 else np.nan,
                    "n_joint_high_bins": int((program_high & domain_high).sum()),
                    "n_bins": len(bins),
                    "role": "descriptive same-bin spatial co-enrichment",
                }
            )

    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(
        ROOT / "08_program_localization" / "StereoSeq_program_domain_correlation.tsv", sep="\t", index=False
    )
    localization = pd.DataFrame(localization_rows)
    localization.to_csv(
        ROOT / "08_program_localization" / "StereoSeq_program_localization_by_section.tsv", sep="\t", index=False
    )
    niches = pd.DataFrame(niche_rows)
    niches.to_csv(ROOT / "08_program_localization" / "StereoSeq_niche_colocalization.tsv", sep="\t", index=False)

    response_summary = []
    for keys, frame in localization.groupby(["program_name", "domain_module"]):
        cr = frame.loc[frame["response_group"] == "CR", "localization_strength_spearman"].to_numpy()
        nr = frame.loc[frame["response_group"] == "NR", "localization_strength_spearman"].to_numpy()
        cr_partial = frame.loc[frame["response_group"] == "CR", "partial_localization_controlling_CAF_FAP"].to_numpy()
        nr_partial = frame.loc[frame["response_group"] == "NR", "partial_localization_controlling_CAF_FAP"].to_numpy()
        response_summary.append(
            {
                "program_name": keys[0],
                "domain_module": keys[1],
                "mean_CR_localization": np.nanmean(cr),
                "mean_NR_localization": np.nanmean(nr),
                "effect_NR_minus_CR_localization": np.nanmean(nr) - np.nanmean(cr),
                "mean_CR_partial_localization_controlling_CAF_FAP": np.nanmean(cr_partial),
                "mean_NR_partial_localization_controlling_CAF_FAP": np.nanmean(nr_partial),
                "effect_NR_minus_CR_partial_localization": np.nanmean(nr_partial) - np.nanmean(cr_partial),
                "NR_sections_both_positive": bool(np.all(nr > 0)),
                "section_level_n": 4,
                "inference_boundary": "two CR and two NR sections; no bin-level p value is response evidence",
            }
        )
    pd.DataFrame(response_summary).to_csv(
        ROOT / "08_program_localization" / "StereoSeq_CR_NR_localization_summary.tsv", sep="\t", index=False
    )
    comparator_corr = pd.DataFrame(comparator_corr_rows)
    comparator_corr.to_csv(
        ROOT / "09_response_group_comparison" / "StereoSeq_comparator_correlation_summary.tsv", sep="\t", index=False
    )
    pd.DataFrame(section_comparator_means).to_csv(
        ROOT / "09_response_group_comparison" / "StereoSeq_comparator_section_descriptive_summary.tsv", sep="\t", index=False
    )

    # Comparator scores on section pseudobulk gene z-scores are cross-section comparable.
    section_gene_z = pd.read_parquet(
        ROOT / "06_section_level_summary" / "StereoSeq_section_pseudobulk_gene_zscores.parquet"
    )
    section_modules = []
    for name, module in COMPARATOR_MODULES.items():
        represented = [normalize_gene(g) for g in module if normalize_gene(g) in section_gene_z.index]
        values = section_gene_z.loc[represented].mean(axis=0) if represented else pd.Series(np.nan, index=SECTIONS)
        for section in SECTIONS:
            section_modules.append(
                {
                    "section_id": section,
                    "response_group": RESPONSE[section],
                    "module_name": name,
                    "section_pseudobulk_score": values[section],
                    "n_genes_represented": len(represented),
                }
            )
    pd.DataFrame(section_modules).to_csv(
        ROOT / "09_response_group_comparison" / "StereoSeq_comparator_section_scores.tsv", sep="\t", index=False
    )

    # Section-level residual direction after CAF_FAP adjustment; descriptive with n=4.
    section_program = pd.read_csv(
        ROOT / "06_section_level_summary" / "StereoSeq_section_level_program_scores.tsv", sep="\t"
    )
    section_comp = pd.DataFrame(section_modules)
    caf = section_comp[section_comp["module_name"] == "CAF_FAP_source"].set_index("section_id")["section_pseudobulk_score"]
    residual_rows = []
    for label, frame in section_program.groupby("program_name"):
        values = frame.set_index("section_id")["section_pseudobulk_resistance_score"].reindex(SECTIONS)
        design = np.column_stack([np.ones(4), caf.reindex(SECTIONS).to_numpy()])
        residual = values.to_numpy() - design @ np.linalg.lstsq(design, values.to_numpy(), rcond=None)[0]
        nr = residual[[RESPONSE[s] == "NR" for s in SECTIONS]]
        cr = residual[[RESPONSE[s] == "CR" for s in SECTIONS]]
        residual_rows.append(
            {
                "program_name": label,
                "raw_correlation_with_section_CAF_FAP": safe_spearman(values.to_numpy(), caf.reindex(SECTIONS).to_numpy())[0],
                "residual_effect_NR_minus_CR_after_CAF_FAP": nr.mean() - cr.mean(),
                "residual_direction_support": nr.mean() - cr.mean() > 0,
                "n_sections": 4,
                "interpretation": "descriptive only; n=4 residualization is unstable",
            }
        )
    residual_table = pd.DataFrame(residual_rows)
    residual_table.to_csv(
        ROOT / "09_response_group_comparison" / "StereoSeq_nonredundancy_section_residual_summary.tsv", sep="\t", index=False
    )

    matched_summary = pd.DataFrame(response_summary)
    matched_summary = matched_summary[
        matched_summary.apply(lambda row: MATCHED_DOMAIN.get(row["program_name"]) == row["domain_module"], axis=1)
    ]
    strongest = matched_summary.sort_values("effect_NR_minus_CR_partial_localization", ascending=False).head(3)
    residual_support = residual_table[residual_table["residual_direction_support"]]["program_name"].tolist()
    (ROOT / "08_program_localization" / "StereoSeq_niche_interpretation.md").write_text(
        "# Stereo-seq resistance-niche interpretation\n\n"
        "Spatial localization is evaluated through within-section program-domain correlations and top-quintile same-bin co-enrichment. Bins are descriptive observations; response evidence remains section-level.\n\n"
        "The largest NR-minus-CR matched-domain partial-localization effects are:\n"
        + "\n".join(
            f"- {row.program_name} -> {row.domain_module}: {row.effect_NR_minus_CR_partial_localization:.3f}"
            for row in strongest.itertuples()
        )
        + "\n\nThese patterns nominate spatial niches for later review but do not establish cell-cell causality.\n",
        encoding="utf-8",
    )
    (ROOT / "09_response_group_comparison" / "StereoSeq_nonredundancy_with_CAF_FAP_summary.md").write_text(
        "# Nonredundancy with the source CAF_FAP story\n\n"
        "The locked programs were compared with CAF_FAP, MIR4435-2HG/AREG/EMT, generic CAF, immune, myeloid, endothelial, and source pCAF/nCAF modules.\n\n"
        f"Programs retaining a positive descriptive NR-minus-CR section residual after CAF_FAP adjustment: {', '.join(residual_support) if residual_support else 'none'}.\n\n"
        "Because there are only four sections, residualization is unstable and is used only as a nonredundancy screen. Added value requires coherent malignant, CD8, myeloid, or endothelial localization rather than a CAF-only signal.\n",
        encoding="utf-8",
    )
    (ROOT / "07_spatial_domain_annotation" / "StereoSeq_domain_annotation_note.md").write_text(
        "# Spatial-domain annotation note\n\n"
        "Domains are marker-module niches, not purified cell types. Each bin receives seven within-section marker scores, a dominant positive module label, and an unsupervised K-means cluster based on those scores. No histopathologist annotation was publicly deposited with these matrices.\n",
        encoding="utf-8",
    )
    print("Completed spatial domains, localization, niche, and comparator analyses.")


if __name__ == "__main__":
    main()
