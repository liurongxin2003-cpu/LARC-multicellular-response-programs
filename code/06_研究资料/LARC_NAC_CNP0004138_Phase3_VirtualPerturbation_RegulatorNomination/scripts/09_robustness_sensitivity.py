from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

from phase3_config import CAF_AXIS_GENES, INPUT, ROOT, SEED, TARGET_STATES


def response_effect(values: np.ndarray, labels: np.ndarray, left: set[str], right: set[str]) -> float:
    a = values[np.isin(labels, list(left))]
    b = values[np.isin(labels, list(right))]
    return float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan


def main() -> None:
    out = ROOT / "10_robustness_sensitivity"
    rng = np.random.default_rng(SEED)
    shortlist = pd.read_csv(ROOT / "08_spatial_regulator_projection/pre_spatial_candidate_shortlist.tsv", sep="\t")
    activity = pd.read_parquet(ROOT / "04_regulator_activity/regulator_activity_pseudobulk_scores.parquet")
    pret_meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t").set_index("pseudobulk_id")

    activity = activity.merge(shortlist[["target_state", "regulator"]].drop_duplicates(), on=["target_state", "regulator"])
    mean_activity = activity.groupby(["target_state", "regulator", "pseudobulk_id"], as_index=False).activity_score.mean()
    mean_activity = mean_activity.join(pret_meta[["patient_id", "response_standardized"]], on="pseudobulk_id")

    leave_rows = []
    label_rows = []
    for (state, regulator), frame in mean_activity.groupby(["target_state", "regulator"]):
        labels = frame.response_standardized.to_numpy()
        values = frame.activity_score.to_numpy(dtype=float)
        full = response_effect(values, labels, {"CR"}, {"NR"})
        primary_patients = frame.loc[frame.response_standardized.isin(["CR", "NR"]), "patient_id"].unique()
        for patient in primary_patients:
            keep = frame.patient_id != patient
            effect = response_effect(values[keep], labels[keep], {"CR"}, {"NR"})
            leave_rows.append({
                "target_state": state,
                "regulator": regulator,
                "removed_patient": patient,
                "full_effect_NR_minus_CR": full,
                "leave_one_effect_NR_minus_CR": effect,
                "sign_preserved": bool(np.sign(effect) == np.sign(full)),
                "positive_direction": bool(effect > 0),
            })
        for scenario, left, right in [
            ("CR_vs_NR", {"CR"}, {"NR"}),
            ("CR_plus_PR_vs_NR", {"CR", "PR"}, {"NR"}),
            ("CR_vs_PR_plus_NR", {"CR"}, {"PR", "NR"}),
        ]:
            effect = response_effect(values, labels, left, right)
            label_rows.append({
                "target_state": state,
                "regulator": regulator,
                "label_scenario": scenario,
                "effect_right_minus_left": effect,
                "positive_direction": bool(effect > 0),
            })
    leave = pd.DataFrame(leave_rows)
    leave.to_csv(out / "leave_one_patient_regulator_sensitivity.tsv", sep="\t", index=False)
    labels = pd.DataFrame(label_rows)
    labels.to_csv(out / "response_label_sensitivity.tsv", sep="\t", index=False)

    full_pb = pd.read_parquet(INPUT / "CNP0004138_patient_timepoint_celltype_pseudobulk_matrix.parquet")
    full_pb.index = full_pb.index.astype(str).str.upper()
    full_meta = pd.read_csv(INPUT / "CNP0004138_patient_timepoint_celltype_pseudobulk_metadata.tsv", sep="\t").set_index("pseudobulk_id")
    time_rows = []
    for state, definition in TARGET_STATES.items():
        regulators = shortlist.loc[shortlist.target_state == state, "regulator"].drop_duplicates().tolist()
        for scope, allowed_timepoints in [("PRET_only", {"PRET"}), ("PRET_plus_POST_exploratory", {"PRET", "POSTT"})]:
            unit_ids = full_meta.index[(full_meta.cell_type == definition["cell_type"]) & full_meta.timepoint.isin(allowed_timepoints) & full_meta.response_standardized.isin(["CR", "NR"])].tolist()
            if not unit_ids:
                continue
            counts = full_pb[unit_ids]
            logcpm = np.log1p(counts.div(np.maximum(counts.sum(axis=0), 1), axis=1) * 1e6)
            response = full_meta.loc[unit_ids, "response_standardized"].to_numpy()
            for regulator in regulators:
                if regulator not in logcpm.index:
                    continue
                values = logcpm.loc[regulator].to_numpy(dtype=float)
                time_rows.append({
                    "target_state": state,
                    "regulator": regulator,
                    "timepoint_scope": scope,
                    "expression_proxy_effect_NR_minus_CR": response_effect(values, response, {"CR"}, {"NR"}),
                    "n_units": len(unit_ids),
                    "boundary": "TF expression proxy; not regulator activity",
                })
    pd.DataFrame(time_rows).to_csv(out / "timepoint_scope_sensitivity.tsv", sep="\t", index=False)

    adata = sc.read_h5ad(INPUT / "CNP0004138_QC_sample_annotated.h5ad")
    annotations = pd.read_csv(ROOT / "03_refined_cell_state_annotation/refined_target_state_annotation.tsv", sep="\t").set_index("cell_index")
    gene_index = {str(g).upper(): i for i, g in enumerate(adata.var_names)}
    subsample_rows = []
    for state, definition in TARGET_STATES.items():
        regulators = [reg for reg in shortlist.loc[shortlist.target_state == state, "regulator"].drop_duplicates() if reg in gene_index]
        cell_indices = annotations.index[(annotations.target_state == state) & (annotations.timepoint == "PRET") & annotations.response_standardized.isin(["CR", "NR"])].to_numpy(dtype=int)
        if len(cell_indices) == 0 or not regulators:
            continue
        matrix = adata[cell_indices, [gene_index[reg] for reg in regulators]].X
        if sparse.issparse(matrix):
            matrix = matrix.toarray()
        obs = adata.obs.iloc[cell_indices]
        patient_groups = {patient: np.flatnonzero(obs.patient_id.to_numpy() == patient) for patient in obs.patient_id.unique()}
        baseline_effects = {}
        for j, regulator in enumerate(regulators):
            patient_means = np.array([matrix[idx, j].mean() for idx in patient_groups.values()])
            patient_labels = np.array([obs.iloc[idx[0]].response_standardized for idx in patient_groups.values()])
            baseline_effects[regulator] = response_effect(patient_means, patient_labels, {"CR"}, {"NR"})
        bootstrap_effects = {reg: [] for reg in regulators}
        for _ in range(50):
            sampled = {}
            for patient, idx in patient_groups.items():
                size = max(1, int(np.ceil(len(idx) * 0.5)))
                sampled[patient] = rng.choice(idx, size=size, replace=False)
            patient_labels = np.array([obs.iloc[idx[0]].response_standardized for idx in patient_groups.values()])
            for j, regulator in enumerate(regulators):
                patient_means = np.array([matrix[sampled[patient], j].mean() for patient in patient_groups])
                bootstrap_effects[regulator].append(response_effect(patient_means, patient_labels, {"CR"}, {"NR"}))
        for regulator in regulators:
            effects = np.asarray(bootstrap_effects[regulator])
            subsample_rows.append({
                "target_state": state,
                "regulator": regulator,
                "baseline_sampled_cell_expression_effect": baseline_effects[regulator],
                "median_half_cell_subsample_effect": np.median(effects),
                "subsample_sign_stability": np.mean(np.sign(effects) == np.sign(baseline_effects[regulator])),
                "positive_subsample_fraction": np.mean(effects > 0),
                "n_subsamples": len(effects),
                "boundary": "50k sampled-cell expression proxy; patient means remain the statistical units",
            })
    pd.DataFrame(subsample_rows).to_csv(out / "cell_subsampling_sensitivity.tsv", sep="\t", index=False)

    edges = pd.read_csv(ROOT / "05_grn_inference/GRN_edges_by_target_state.tsv", sep="\t")
    virtual = pd.read_csv(ROOT / "06_virtual_perturbation/virtual_perturbation_ranked_regulators.tsv", sep="\t")
    caf_rows = []
    for (state, regulator), frame in edges.groupby(["target_state", "regulator"]):
        supported = frame[frame.method_support_count >= 2]
        remaining = supported[~supported.target_gene.isin(CAF_AXIS_GENES)]
        original = virtual[(virtual.target_state == state) & (virtual.regulator == regulator)]
        caf_rows.append({
            "target_state": state,
            "regulator": regulator,
            "n_supported_edges_before": len(supported),
            "n_supported_edges_after_CAF_axis_exclusion": len(remaining),
            "remaining_edge_fraction": len(remaining) / len(supported) if len(supported) else np.nan,
            "original_virtual_support_fraction": original.perturbation_method_support_fraction.iloc[0] if len(original) else np.nan,
            "CAF_axis_exclusion_support": bool(len(remaining) > 0 and (len(remaining) / len(supported) >= 0.75 if len(supported) else False)),
        })
    caf = pd.DataFrame(caf_rows)
    caf.to_csv(out / "CAF_axis_exclusion_sensitivity.tsv", sep="\t", index=False)

    stability = shortlist[["target_state", "regulator"]].drop_duplicates()
    leave_summary = leave.groupby(["target_state", "regulator"], as_index=False).agg(
        leave_one_positive_fraction=("positive_direction", "mean"),
        leave_one_sign_preservation=("sign_preserved", "mean"),
    )
    label_summary = labels.groupby(["target_state", "regulator"], as_index=False).agg(
        response_label_positive_fraction=("positive_direction", "mean")
    )
    sub_summary = pd.DataFrame(subsample_rows)[["target_state", "regulator", "subsample_sign_stability", "positive_subsample_fraction"]]
    stability = stability.merge(leave_summary, on=["target_state", "regulator"], how="left").merge(label_summary, on=["target_state", "regulator"], how="left").merge(sub_summary, on=["target_state", "regulator"], how="left").merge(caf[["target_state", "regulator", "remaining_edge_fraction", "CAF_axis_exclusion_support"]], on=["target_state", "regulator"], how="left")
    stability["overall_stability_score"] = stability[["leave_one_positive_fraction", "response_label_positive_fraction", "positive_subsample_fraction", "remaining_edge_fraction"]].mean(axis=1)
    stability.to_csv(out / "regulator_robustness_summary.tsv", sep="\t", index=False)
    strong = stability[stability.overall_stability_score >= 0.75]
    note = f"""# Regulator ranking stability summary

- Candidate regulator-state combinations tested: {len(stability)}.
- Overall stability score >=0.75: {len(strong)}.
- Leave-one analysis uses patient-celltype pseudobulk activity, not cells.
- Response-label scenarios: CR vs NR, CR+PR vs NR, and CR vs PR+NR.
- PRET+POST is exploratory and uses TF expression as a proxy because full-timepoint activity was not refit.
- Cell subsampling uses 50 iterations at 50% of cells within each patient and retains patient means as statistical units.
- CAF-axis exclusion removes FAP/MIR4435-2HG/source-marker genes from supported GRN edges; it does not remove the entire fibroblast lineage.
- Negative and unstable candidates remain in the output and cannot be promoted to a core mechanism.
"""
    (out / "regulator_ranking_stability_summary.md").write_text(note, encoding="utf-8")
    print(f"Completed robustness testing for {len(stability)} candidate regulator-state combinations.")


if __name__ == "__main__":
    main()
