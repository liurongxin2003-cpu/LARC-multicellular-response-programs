from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from phase3_config import P2, ROOT, TARGET_STATES


STATE_PROGRAM_DOMAIN = {
    "Malignant_EMT_stress": ("malignant", "Malignant_epithelial_domain"),
    "Myeloid_suppressive_remodeling": ("myeloid", "Myeloid_domain"),
    "CD8_dysfunctional_stress": ("CD8", "T_CD8_domain"),
    "Endothelial_remodeling": ("endothelial", "Endothelial_domain"),
    "CAF_supporting_comparator": ("CAF", "CAF_fibroblast_domain"),
}

SECTIONS = ["ST-CR1", "ST-CR2", "ST-NR1", "ST-NR2"]


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return np.nan
    return float(stats.spearmanr(a[keep], b[keep]).statistic)


def build_shortlist() -> pd.DataFrame:
    activity = pd.read_csv(ROOT / "04_regulator_activity/top_regulators_by_state.tsv", sep="\t")
    grn = pd.read_csv(ROOT / "05_grn_inference/GRN_method_concordance_summary.tsv", sep="\t")
    virtual = pd.read_csv(ROOT / "06_virtual_perturbation/virtual_perturbation_ranked_regulators.tsv", sep="\t")
    merged = activity.merge(grn, on=["target_state", "regulator"], how="inner").merge(
        virtual, on=["target_state", "source_cell_type", "regulator"], how="inner"
    )
    merged = merged[(merged.n_activity_methods >= 2) & (merged.n_perturbation_methods >= 2)].copy()
    merged["pre_spatial_score"] = (
        merged.activity_evidence_score
        + merged.virtual_perturbation_score
        + merged.groupby("target_state")["two_or_more_method_NR_edges"].rank(pct=True)
    ) / 3
    merged = merged.sort_values(["target_state", "pre_spatial_score"], ascending=[True, False])
    merged.to_csv(ROOT / "08_spatial_regulator_projection/pre_spatial_candidate_shortlist.tsv", sep="\t", index=False)
    return merged


def extract_rows(path: Path, genes: set[str]) -> tuple[list[str], dict[str, np.ndarray]]:
    extracted = {}
    with path.open("r", encoding="utf-8", errors="replace", buffering=16 * 1024 * 1024) as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        bins = header[1:]
        for line in stream:
            split = line.find("\t")
            if split < 0:
                continue
            gene = line[:split].upper()
            if gene not in genes:
                continue
            values = np.fromstring(line[split + 1 :], sep="\t", dtype=np.float32)
            if len(values) == len(bins):
                extracted[gene] = values
            if len(extracted) == len(genes):
                break
    return bins, extracted


def main() -> None:
    out = ROOT / "08_spatial_regulator_projection"
    shortlist = build_shortlist()
    top_for_maps = shortlist.groupby("target_state").head(3)
    map_keys = set(zip(top_for_maps.target_state, top_for_maps.regulator))
    regulators = set(shortlist.regulator.str.upper())
    state_lookup = shortlist.groupby("regulator").target_state.apply(list).to_dict()
    domains = pd.read_csv(ROOT / "01_inputs_from_phase1_phase2/StereoSeq_spatial_domain_scores.tsv", sep="\t")

    correlation_rows = []
    projection_rows = []
    section_rows = []
    missing_rows = []
    section_expression = []

    for section in SECTIONS:
        stem = section.replace("-", "_")
        clean_path = P2 / "03_stereoseq_matrix" / f"{stem}_expression_clean.tsv"
        bins, extracted = extract_rows(clean_path, regulators)
        score = pd.read_parquet(ROOT / "01_inputs_from_phase1_phase2" / f"{stem}_bin_level_program_scores.parquet").set_index("bin_id").reindex(bins)
        domain = domains[domains.section_id == section].set_index("bin_id").reindex(bins)
        total_counts = score.total_counts.to_numpy(dtype=float)
        for regulator in sorted(regulators):
            if regulator not in extracted:
                missing_rows.append({"section_id": section, "regulator": regulator, "reason": "not represented in clean spatial matrix"})
                continue
            counts = extracted[regulator].astype(float)
            logcp10k = np.log1p(counts / np.maximum(total_counts, 1) * 1e4)
            sd = np.std(logcp10k, ddof=1)
            z = (logcp10k - np.mean(logcp10k)) / (sd if sd > 0 else 1.0)
            section_expression.append({
                "section_id": section,
                "response_group": "CR" if "-CR" in section else "NR",
                "regulator": regulator,
                "pseudobulk_count": counts.sum(),
                "total_section_counts": total_counts.sum(),
                "section_logCPM": np.log1p(counts.sum() / max(total_counts.sum(), 1) * 1e6),
            })
            for target_state in state_lookup[regulator]:
                program, domain_name = STATE_PROGRAM_DOMAIN[target_state]
                resistance = score[f"{program}_resistance_score"].to_numpy(dtype=float)
                domain_score = domain[domain_name].to_numpy(dtype=float)
                correlation_rows.append({
                    "target_state": target_state,
                    "regulator": regulator,
                    "section_id": section,
                    "response_group": "CR" if "-CR" in section else "NR",
                    "regulator_expression_resistance_rho": safe_spearman(z, resistance),
                    "regulator_expression_matched_domain_rho": safe_spearman(z, domain_score),
                    "n_bins": len(z),
                    "p_value_reported": False,
                    "inference_boundary": "bin correlation is descriptive; section is the response unit",
                })
                if (target_state, regulator) in map_keys:
                    projection_rows.append(pd.DataFrame({
                        "target_state": target_state,
                        "regulator": regulator,
                        "section_id": section,
                        "response_group": "CR" if "-CR" in section else "NR",
                        "bin_id": bins,
                        "regulator_expression_logCP10K": logcp10k,
                        "regulator_expression_z_within_section": z,
                        "locked_resistance_score": resistance,
                        "matched_domain_score": domain_score,
                        "descriptive_only": True,
                    }))

    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(out / "regulator_spatial_correlation_with_resistance.tsv", sep="\t", index=False)
    pd.concat(projection_rows, ignore_index=True).to_csv(out / "spatial_regulator_projection_scores.tsv", sep="\t", index=False)
    expression = pd.DataFrame(section_expression)
    expression.to_csv(out / "regulator_section_expression.tsv", sep="\t", index=False)
    pd.DataFrame(missing_rows).to_csv(out / "spatial_regulator_missingness.tsv", sep="\t", index=False)

    for (state, regulator), frame in expression.merge(shortlist[["target_state", "regulator"]], on="regulator").groupby(["target_state", "regulator"]):
        frame = frame.drop_duplicates("section_id").set_index("section_id").reindex(SECTIONS)
        values = frame.section_logCPM.to_numpy(dtype=float)
        z = (values - values.mean()) / (values.std(ddof=1) or 1.0)
        cr, nr = z[:2], z[2:]
        corr = correlations[(correlations.target_state == state) & (correlations.regulator == regulator)]
        section_rows.append({
            "target_state": state,
            "regulator": regulator,
            "ST_CR1_section_expression_z": z[0],
            "ST_CR2_section_expression_z": z[1],
            "ST_NR1_section_expression_z": z[2],
            "ST_NR2_section_expression_z": z[3],
            "section_effect_NR_minus_CR": nr.mean() - cr.mean(),
            "NR_above_CR_pairwise_fraction": np.mean([n > c for n in nr for c in cr]),
            "mean_within_section_resistance_rho": corr.regulator_expression_resistance_rho.mean(),
            "mean_NR_resistance_rho": corr.loc[corr.response_group == "NR", "regulator_expression_resistance_rho"].mean(),
            "mean_matched_domain_rho": corr.regulator_expression_matched_domain_rho.mean(),
            "remove_ST_NR2_direction": bool(z[2] > cr.mean()),
            "spatial_support": bool((nr.mean() > cr.mean()) and (corr.regulator_expression_resistance_rho.mean() > 0)),
            "statistical_unit": "section; n=4",
        })
    summary = pd.DataFrame(section_rows)
    summary.to_csv(out / "regulator_section_level_summary.tsv", sep="\t", index=False)
    print(f"Projected {summary.regulator.nunique()} candidate regulators; saved bin maps for {len(map_keys)} state-regulator candidates.")


if __name__ == "__main__":
    main()
