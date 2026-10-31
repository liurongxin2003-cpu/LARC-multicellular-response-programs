from __future__ import annotations

import liana as li
import numpy as np
import pandas as pd
from scipy import stats

from phase3_config import INPUT, ROOT


PAIR_SPECS = [
    ("CAF_Fibroblast", "Malignant_epithelial"),
    ("CAF_Fibroblast", "Myeloid_macrophage"),
    ("Myeloid_macrophage", "CD8_T"),
    ("Malignant_epithelial", "Myeloid_macrophage"),
    ("Endothelial", "CD8_T"),
    ("Endothelial", "Myeloid_macrophage"),
    ("Malignant_epithelial", "Endothelial"),
    ("CAF_Fibroblast", "Endothelial"),
]

CELL_TO_STATE = {
    "Malignant_epithelial": "Malignant_EMT_stress",
    "Myeloid_macrophage": "Myeloid_suppressive_remodeling",
    "CD8_T": "CD8_dysfunctional_stress",
    "Endothelial": "Endothelial_remodeling",
    "CAF_Fibroblast": "CAF_supporting_comparator",
}

CELL_TO_PROGRAM = {
    "Malignant_epithelial": ("malignant", "Malignant_epithelial_domain"),
    "Myeloid_macrophage": ("myeloid", "Myeloid_domain"),
    "CD8_T": ("CD8", "T_CD8_domain"),
    "Endothelial": ("endothelial", "Endothelial_domain"),
    "CAF_Fibroblast": ("CAF", "CAF_fibroblast_domain"),
}


def safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return np.nan, np.nan
    result = stats.spearmanr(a[keep], b[keep])
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    out = ROOT / "07_ligand_receptor_niche"
    resource = li.rs.select_resource("consensus").drop_duplicates().copy()
    resource["ligand"] = resource.ligand.astype(str).str.upper()
    resource["receptor"] = resource.receptor.astype(str).str.upper()
    resource.to_csv(out / "LIANA_consensus_resource_snapshot.tsv", sep="\t", index=False)

    pb = pd.read_parquet(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet")
    pb.index = pb.index.astype(str).str.upper()
    meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t")
    locked_scores = pd.read_csv(INPUT / "CNP0004138_pseudobulk_locked_program_scores.tsv", sep="\t")
    localization = pd.read_csv(INPUT / "StereoSeq_CR_NR_localization_summary.tsv", sep="\t")
    activity_rank = pd.read_csv(ROOT / "04_regulator_activity/top_regulators_by_state.tsv", sep="\t")
    activity_values = pd.read_parquet(ROOT / "04_regulator_activity/regulator_activity_pseudobulk_scores.parquet")

    expression = {}
    for cell_type in sorted(set(sum(([a, b] for a, b in PAIR_SPECS), []))):
        ids = meta.loc[(meta.cell_type == cell_type) & meta.response_standardized.isin(["CR", "PR", "NR"]), "pseudobulk_id"].tolist()
        counts = pb[ids]
        library = counts.sum(axis=0)
        logcpm = np.log1p(counts.div(np.maximum(library, 1), axis=1) * 1e6)
        info = meta.set_index("pseudobulk_id").loc[ids, ["patient_id", "response_standardized"]].copy()
        info["pseudobulk_id"] = ids
        expression[cell_type] = (logcpm, info.reset_index(drop=True))

    localization_effect = {}
    for cell_type, (program, domain) in CELL_TO_PROGRAM.items():
        row = localization[(localization.program_name == program) & (localization.domain_module == domain)].iloc[0]
        localization_effect[cell_type] = float(row.effect_NR_minus_CR_partial_localization)

    interaction_rows = []
    ligand_rows = []
    for sender, receiver in PAIR_SPECS:
        sender_expr, sender_meta = expression[sender]
        receiver_expr, receiver_meta = expression[receiver]
        sender_ids = sender_meta.set_index("patient_id").pseudobulk_id
        receiver_ids = receiver_meta.set_index("patient_id").pseudobulk_id
        patients = sorted(set(sender_ids.index).intersection(receiver_ids.index))
        responses = meta.drop_duplicates("patient_id").set_index("patient_id").loc[patients, "response_standardized"].to_numpy()
        trend = pd.Series(responses).map({"CR": 0, "PR": 1, "NR": 2}).to_numpy()
        valid = resource[resource.ligand.isin(sender_expr.index) & resource.receptor.isin(receiver_expr.index)]

        receiver_state = CELL_TO_STATE[receiver]
        receiver_program = {
            "Malignant_epithelial": "Malignant_epithelial_specific",
            "Myeloid_macrophage": "Myeloid_macrophage_specific",
            "CD8_T": "CD8_T_specific",
            "Endothelial": "Endothelial_specific",
            "CAF_Fibroblast": "CAF_Fibroblast_specific",
        }[receiver]
        receiver_resistance = locked_scores[
            (locked_scores.program_name == receiver_program) & (locked_scores.timepoint == "PRET")
        ].set_index("patient_id").reindex(patients).resistance_score.to_numpy()
        top_regs = activity_rank[(activity_rank.target_state == receiver_state) & (activity_rank.n_activity_methods >= 2)].head(12).regulator.tolist()
        receiver_activity = activity_values[(activity_values.target_state == receiver_state) & activity_values.regulator.isin(top_regs)]
        receiver_activity = receiver_activity.groupby(["pseudobulk_id", "regulator"]).activity_score.mean().unstack()

        for ligand, ligand_frame in valid.groupby("ligand"):
            ligand_values = sender_expr.loc[ligand, sender_ids.loc[patients]].to_numpy(dtype=float)
            rho_ligand_resistance, p_ligand_resistance = safe_spearman(ligand_values, receiver_resistance)
            best_regulator = ""
            best_regulator_rho = np.nan
            if not receiver_activity.empty:
                receiver_pbids = receiver_ids.loc[patients].tolist()
                available = receiver_activity.reindex(receiver_pbids)
                correlations = {reg: safe_spearman(ligand_values, available[reg].to_numpy())[0] for reg in available.columns}
                finite = {reg: rho for reg, rho in correlations.items() if np.isfinite(rho)}
                if finite:
                    best_regulator = max(finite, key=finite.get)
                    best_regulator_rho = finite[best_regulator]
            ligand_rows.append({
                "sender_cell_type": sender,
                "receiver_cell_type": receiver,
                "ligand": ligand,
                "n_candidate_receptors": ligand_frame.receptor.nunique(),
                "ligand_receiver_resistance_spearman_rho": rho_ligand_resistance,
                "ligand_receiver_resistance_p": p_ligand_resistance,
                "best_receiver_regulator": best_regulator,
                "ligand_receiver_regulator_rho": best_regulator_rho,
                "method": "paired-patient ligand expression association; NicheNet-style prioritization, not NicheNet causal inference",
            })

        for pair in valid.itertuples(index=False):
            ligand_values = sender_expr.loc[pair.ligand, sender_ids.loc[patients]].to_numpy(dtype=float)
            receptor_values = receiver_expr.loc[pair.receptor, receiver_ids.loc[patients]].to_numpy(dtype=float)
            score = np.sqrt(np.maximum(ligand_values, 0) * np.maximum(receptor_values, 0))
            cr = score[responses == "CR"]
            nr = score[responses == "NR"]
            rho_gradient, p_gradient = safe_spearman(trend, score)
            rho_resistance, p_resistance = safe_spearman(score, receiver_resistance)
            interaction_rows.append({
                "sender_cell_type": sender,
                "receiver_cell_type": receiver,
                "ligand": pair.ligand,
                "receptor": pair.receptor,
                "n_paired_patients": len(patients),
                "mean_CR_interaction_score": np.mean(cr),
                "mean_PR_interaction_score": np.mean(score[responses == "PR"]),
                "mean_NR_interaction_score": np.mean(nr),
                "effect_NR_minus_CR": np.mean(nr) - np.mean(cr),
                "mannwhitney_p_CR_vs_NR": stats.mannwhitneyu(cr, nr, alternative="two-sided").pvalue,
                "gradient_CR_PR_NR_rho": rho_gradient,
                "gradient_p": p_gradient,
                "receiver_resistance_correlation": rho_resistance,
                "receiver_resistance_correlation_p": p_resistance,
                "sender_phase2_matched_localization_effect": localization_effect[sender],
                "receiver_phase2_matched_localization_effect": localization_effect[receiver],
                "pair_spatial_support_proxy": np.mean([localization_effect[sender], localization_effect[receiver]]),
                "non_CAF_pair": sender != "CAF_Fibroblast" and receiver != "CAF_Fibroblast",
                "statistical_unit": "paired patient-celltype pseudobulk",
            })

    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(out / "LR_interactions_by_response_state.tsv", sep="\t", index=False)
    ligands = pd.DataFrame(ligand_rows).drop_duplicates(["sender_cell_type", "receiver_cell_type", "ligand"])
    ligands.to_csv(out / "NicheNet_ligand_activity_summary.tsv", sep="\t", index=False)

    eligible = interactions[(interactions.effect_NR_minus_CR > 0) & (interactions.receiver_resistance_correlation > 0)].copy()
    for column in ["effect_NR_minus_CR", "gradient_CR_PR_NR_rho", "receiver_resistance_correlation", "pair_spatial_support_proxy"]:
        eligible[f"pct_{column}"] = eligible.groupby(["sender_cell_type", "receiver_cell_type"])[column].rank(pct=True)
    eligible["LR_priority_score"] = (
        eligible.pct_effect_NR_minus_CR
        + eligible.pct_gradient_CR_PR_NR_rho
        + eligible.pct_receiver_resistance_correlation
        + eligible.pct_pair_spatial_support_proxy
    ) / 4
    priority = eligible.sort_values(["sender_cell_type", "receiver_cell_type", "LR_priority_score"], ascending=[True, True, False]).groupby(["sender_cell_type", "receiver_cell_type"]).head(25)
    ligand_lookup = ligands.set_index(["sender_cell_type", "receiver_cell_type", "ligand"])
    priority["best_receiver_regulator"] = [ligand_lookup.loc[(row.sender_cell_type, row.receiver_cell_type, row.ligand), "best_receiver_regulator"] for row in priority.itertuples()]
    priority["ligand_receiver_regulator_rho"] = [ligand_lookup.loc[(row.sender_cell_type, row.receiver_cell_type, row.ligand), "ligand_receiver_regulator_rho"] for row in priority.itertuples()]
    priority.to_csv(out / "spatially_informed_LR_priority_table.tsv", sep="\t", index=False)

    non_caf = priority[priority.non_CAF_pair]
    top_axes = non_caf.sort_values("LR_priority_score", ascending=False).head(20)
    lines = [
        "# Resistance-niche ligand-receptor interpretation",
        "",
        f"LIANA consensus resource pairs evaluated: {len(resource):,}.",
        f"Patient-paired interaction rows: {len(interactions):,}; spatially informed positive-direction priorities: {len(priority):,}.",
        f"Non-CAF prioritized axes: {len(non_caf):,}.",
        "",
        "Priority is based on PRET paired-patient expression direction, receiver locked-score association, and Phase 2 domain-localization proxies. Spatial support is a proxy at the compartment level and does not prove physical ligand-receptor engagement.",
        "",
        "## Top non-CAF axes",
        "",
    ]
    for row in top_axes.itertuples():
        lines.append(f"- {row.sender_cell_type} -> {row.receiver_cell_type}: {row.ligand}-{row.receptor}; priority {row.LR_priority_score:.3f}, receiver-resistance rho {row.receiver_resistance_correlation:.3f}.")
    (out / "niche_axis_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Prioritized {len(priority)} spatially informed LR axes from {len(interactions):,} paired-patient interaction rows.")


if __name__ == "__main__":
    main()
