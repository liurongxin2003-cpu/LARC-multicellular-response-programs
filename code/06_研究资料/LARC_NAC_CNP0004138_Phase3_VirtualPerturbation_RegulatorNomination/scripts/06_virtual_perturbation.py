from __future__ import annotations

import numpy as np
import pandas as pd

from phase3_config import INPUT, ROOT, TARGET_STATES


PROGRAM_MAP = {
    "Malignant_EMT_stress": "Malignant_epithelial_specific",
    "Myeloid_suppressive_remodeling": "Myeloid_macrophage_specific",
    "CD8_dysfunctional_stress": "CD8_T_specific",
    "Endothelial_remodeling": "Endothelial_specific",
    "CAF_supporting_comparator": "CAF_Fibroblast_specific",
}


def zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    sd = values.std(axis=1, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (values - mean) / sd


def main() -> None:
    out = ROOT / "06_virtual_perturbation"
    edges = pd.read_csv(ROOT / "05_grn_inference/GRN_edges_by_target_state.tsv", sep="\t")
    activity_values = pd.read_parquet(ROOT / "04_regulator_activity/regulator_activity_pseudobulk_scores.parquet")
    pb = pd.read_parquet(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet")
    pb.index = pb.index.astype(str).str.upper()
    meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t").set_index("pseudobulk_id")
    locked_genes = pd.read_csv(INPUT / "CNP0004138_locked_programs_by_celltype.tsv", sep="\t")

    unit_rows = []
    summary_rows = []
    failures = []
    for target_state, definition in TARGET_STATES.items():
        cell_type = definition["cell_type"]
        unit_ids = meta.index[(meta.cell_type == cell_type) & meta.response_standardized.isin(["CR", "PR", "NR"])].tolist()
        counts = pb[unit_ids].to_numpy(dtype=float)
        library = counts.sum(axis=0)
        logcpm = np.log1p(counts / np.maximum(library, 1) * 1e6)
        scaled = zscore_rows(logcpm)
        gene_index = {gene: index for index, gene in enumerate(pb.index)}
        program = locked_genes[locked_genes.source_cell_type == cell_type]
        nr_genes = [gene for gene in program.loc[program.program_direction == "NR_high", "gene_symbol"].astype(str).str.upper().head(100) if gene in gene_index]
        cr_genes = [gene for gene in program.loc[program.program_direction == "CR_high", "gene_symbol"].astype(str).str.upper().head(100) if gene in gene_index]
        baseline_nr = scaled[[gene_index[g] for g in nr_genes]].mean(axis=0)
        baseline_cr = scaled[[gene_index[g] for g in cr_genes]].mean(axis=0)
        baseline_resistance = baseline_nr - baseline_cr

        state_edges = edges[(edges.target_state == target_state) & (edges.method_support_count >= 2)].copy()
        for regulator in sorted(state_edges.regulator.unique()):
            reg_edges = state_edges[state_edges.regulator == regulator].copy()
            reg_edges = reg_edges[reg_edges.target_gene.isin(nr_genes + cr_genes)]
            if reg_edges.empty:
                failures.append({"target_state": target_state, "regulator": regulator, "reason": "no >=2-method edge to represented locked genes"})
                continue
            activity = activity_values[(activity_values.target_state == target_state) & (activity_values.regulator == regulator)]
            if activity.empty:
                failures.append({"target_state": target_state, "regulator": regulator, "reason": "no regulator activity estimate"})
                continue
            activity_mean = activity.groupby("pseudobulk_id").activity_score.mean().reindex(unit_ids)
            if activity_mean.isna().any():
                failures.append({"target_state": target_state, "regulator": regulator, "reason": "incomplete activity estimates"})
                continue
            activity_z = (activity_mean.to_numpy() - activity_mean.mean()) / (activity_mean.std(ddof=1) or 1.0)
            activity_excess = np.maximum(activity_z, 0)

            for method in ["fixed_target_downweight", "activity_scaled_network_propagation"]:
                perturbed = scaled.copy()
                for edge in reg_edges.itertuples():
                    index = gene_index[edge.target_gene]
                    sign = edge.prior_weight if np.isfinite(edge.prior_weight) and edge.prior_weight != 0 else np.sign(edge.coexpression_spearman_rho)
                    confidence = edge.method_support_count / 3
                    base_step = 0.50 * confidence
                    if method == "fixed_target_downweight":
                        delta = -base_step * sign
                    else:
                        delta = -base_step * sign * activity_excess
                    perturbed[index, :] = perturbed[index, :] + delta
                perturbed_nr = perturbed[[gene_index[g] for g in nr_genes]].mean(axis=0)
                perturbed_cr = perturbed[[gene_index[g] for g in cr_genes]].mean(axis=0)
                perturbed_resistance = perturbed_nr - perturbed_cr
                delta_resistance = perturbed_resistance - baseline_resistance
                delta_nr = perturbed_nr - baseline_nr
                delta_cr = perturbed_cr - baseline_cr
                for index, unit_id in enumerate(unit_ids):
                    unit_rows.append({
                        "target_state": target_state,
                        "source_cell_type": cell_type,
                        "regulator": regulator,
                        "perturbation_method": method,
                        "pseudobulk_id": unit_id,
                        "patient_id": meta.loc[unit_id, "patient_id"],
                        "response_standardized": meta.loc[unit_id, "response_standardized"],
                        "baseline_resistance_score": baseline_resistance[index],
                        "perturbed_resistance_score": perturbed_resistance[index],
                        "delta_resistance_score": delta_resistance[index],
                        "delta_NR_high_module": delta_nr[index],
                        "delta_CR_high_module": delta_cr[index],
                        "delta_response_score": -delta_resistance[index],
                        "n_supported_locked_edges": len(reg_edges),
                        "simulation_boundary": "network perturbation suggestion; not knockout validation",
                    })
                responses = meta.loc[unit_ids, "response_standardized"].to_numpy()
                patients = meta.loc[unit_ids, "patient_id"].to_numpy()
                nr_mask = responses == "NR"
                leave_one = []
                for patient in patients[nr_mask]:
                    keep = nr_mask & (patients != patient)
                    if keep.any():
                        leave_one.append(np.mean(delta_resistance[keep]) < 0)
                summary_rows.append({
                    "target_state": target_state,
                    "source_cell_type": cell_type,
                    "regulator": regulator,
                    "perturbation_method": method,
                    "mean_delta_resistance_all": np.mean(delta_resistance),
                    "mean_delta_resistance_CR": np.mean(delta_resistance[responses == "CR"]),
                    "mean_delta_resistance_PR": np.mean(delta_resistance[responses == "PR"]),
                    "mean_delta_resistance_NR": np.mean(delta_resistance[nr_mask]),
                    "mean_delta_NR_high_module_NR": np.mean(delta_nr[nr_mask]),
                    "mean_delta_CR_high_module_NR": np.mean(delta_cr[nr_mask]),
                    "fraction_all_units_resistance_reduced": np.mean(delta_resistance < 0),
                    "fraction_NR_units_resistance_reduced": np.mean(delta_resistance[nr_mask] < 0),
                    "leave_one_NR_patient_direction_stability": np.mean(leave_one),
                    "n_supported_locked_edges": len(reg_edges),
                    "virtual_effect_support": bool(np.mean(delta_resistance[nr_mask]) < 0),
                })

    unit = pd.DataFrame(unit_rows)
    summary = pd.DataFrame(summary_rows)
    unit.to_csv(out / "virtual_perturbation_pseudobulk_results.tsv", sep="\t", index=False)
    summary.to_csv(out / "virtual_perturbation_results_by_state.tsv", sep="\t", index=False)
    ranked = summary.groupby(["target_state", "source_cell_type", "regulator"], as_index=False).agg(
        n_perturbation_methods=("perturbation_method", "nunique"),
        median_delta_resistance_NR=("mean_delta_resistance_NR", "median"),
        perturbation_method_support_fraction=("virtual_effect_support", "mean"),
        mean_NR_unit_reduction_fraction=("fraction_NR_units_resistance_reduced", "mean"),
        mean_leave_one_stability=("leave_one_NR_patient_direction_stability", "mean"),
        max_supported_locked_edges=("n_supported_locked_edges", "max"),
    )
    ranked["virtual_perturbation_score"] = (
        ranked.groupby("target_state")["median_delta_resistance_NR"].rank(pct=True, ascending=False)
        + ranked.perturbation_method_support_fraction
        + ranked.mean_NR_unit_reduction_fraction
        + ranked.mean_leave_one_stability
    ) / 4
    ranked = ranked.sort_values(["target_state", "virtual_perturbation_score"], ascending=[True, False])
    ranked.to_csv(out / "virtual_perturbation_ranked_regulators.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(out / "virtual_perturbation_failures.tsv", sep="\t", index=False)

    notes = """# Virtual perturbation method notes

Two executable network stress tests were applied within each fixed target cell type: (1) fixed signed downweighting of >=2-method GRN targets and (2) activity-scaled signed network propagation. Both recompute the unchanged Phase 1 NR-high minus CR-high score at patient-celltype pseudobulk level.

CellOracle and scTenifoldKnk were not used because the available processed object lacks a validated target-state trajectory, motif-accessibility context, and full per-state raw network required for defensible use. The implemented simulations test whether nominated regulator-target edges are directionally capable of reducing the locked score. They are not real knockouts, causal validation, or validated therapeutic interventions.
"""
    (out / "virtual_perturbation_method_notes.md").write_text(notes, encoding="utf-8")
    limitations = f"""# Virtual perturbation failures and limitations

- Regulators without at least one two-method GRN edge to a represented locked gene were not simulated.
- Failed regulator-state combinations: {len(failures)}; see `virtual_perturbation_failures.tsv`.
- Fixed downweighting is a deterministic stress test; activity-scaled propagation depends on prior/activity estimates.
- The score response is mathematically constrained by the inferred network and cannot establish biological causality.
- Stability is evaluated across patient pseudobulks and leave-one NR patients, not independent experimental perturbations.
"""
    (out / "virtual_perturbation_failure_or_limitations.md").write_text(limitations, encoding="utf-8")
    print(f"Simulated {len(ranked)} regulator-state combinations with two network perturbation methods.")


if __name__ == "__main__":
    main()
