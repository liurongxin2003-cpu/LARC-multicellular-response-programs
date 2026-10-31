from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

from phase3_config import INPUT, RESPONSE_ORDER, ROOT, SEED, STATE_MODULES, TARGET_STATE_COMPONENTS, TARGET_STATES


def zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    sd = values.std(axis=1, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (values - mean) / sd


def module_score(matrix: np.ndarray, genes: list[str], gene_index: dict[str, int]) -> tuple[np.ndarray, int]:
    represented = [gene_index[g] for g in genes if g in gene_index]
    if not represented:
        return np.full(matrix.shape[1], np.nan), 0
    return np.nanmean(matrix[represented, :], axis=0), len(represented)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    d = (np.mean(b) - np.mean(a)) / pooled
    return d * (1 - 3 / (4 * (n1 + n2) - 9))


def bootstrap_effect(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    values = []
    for _ in range(n_boot):
        values.append(np.mean(rng.choice(b, len(b), replace=True)) - np.mean(rng.choice(a, len(a), replace=True)))
    return tuple(np.quantile(values, [0.025, 0.975]))


def main() -> None:
    out = ROOT / "03_refined_cell_state_annotation"
    rng = np.random.default_rng(SEED)

    pb = pd.read_parquet(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet")
    pb.index = pb.index.astype(str).str.upper()
    meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t").set_index("pseudobulk_id")
    pb_rows = []
    effect_rows = []

    for target_state, definition in TARGET_STATES.items():
        cell_type = definition["cell_type"]
        unit_ids = meta.index[(meta.cell_type == cell_type) & meta.response_standardized.isin(RESPONSE_ORDER)].tolist()
        counts = pb[unit_ids].to_numpy(dtype=np.float64)
        library = counts.sum(axis=0)
        logcpm = np.log1p(counts / np.maximum(library, 1) * 1e6)
        scaled = zscore_rows(logcpm)
        gene_index = {gene: index for index, gene in enumerate(pb.index)}
        module_values = {}
        module_coverage = {}
        for module in TARGET_STATE_COMPONENTS[target_state]:
            values, represented = module_score(scaled, STATE_MODULES[module], gene_index)
            module_values[module] = values
            module_coverage[module] = represented / len(STATE_MODULES[module])
        target_values = np.nanmean(np.vstack(list(module_values.values())), axis=0)
        for index, unit_id in enumerate(unit_ids):
            row = meta.loc[unit_id].to_dict()
            row.update({
                "pseudobulk_id": unit_id,
                "target_state": target_state,
                "target_state_score": target_values[index],
                "mean_component_gene_coverage": float(np.mean(list(module_coverage.values()))),
                "primary_statistical_unit": "patient-celltype pseudobulk",
            })
            for module, values in module_values.items():
                row[f"module_score__{module}"] = values[index]
            pb_rows.append(row)

        frame = pd.DataFrame({
            "score": target_values,
            "response": meta.loc[unit_ids, "response_standardized"].values,
        })
        cr = frame.loc[frame.response == "CR", "score"].to_numpy()
        nr = frame.loc[frame.response == "NR", "score"].to_numpy()
        labels = frame.response.map({"CR": 0, "PR": 1, "NR": 2}).to_numpy()
        ci_low, ci_high = bootstrap_effect(cr, nr, rng)
        effect_rows.append({
            "target_state": target_state,
            "source_cell_type": cell_type,
            "n_CR": len(cr),
            "n_PR": int((frame.response == "PR").sum()),
            "n_NR": len(nr),
            "mean_CR": np.mean(cr),
            "mean_PR": frame.loc[frame.response == "PR", "score"].mean(),
            "mean_NR": np.mean(nr),
            "effect_NR_minus_CR": np.mean(nr) - np.mean(cr),
            "hedges_g_NR_minus_CR": hedges_g(cr, nr),
            "bootstrap_CI_low": ci_low,
            "bootstrap_CI_high": ci_high,
            "mannwhitney_p": stats.mannwhitneyu(cr, nr, alternative="two-sided").pvalue,
            "spearman_CR_PR_NR": stats.spearmanr(labels, frame.score).statistic,
            "spearman_p": stats.spearmanr(labels, frame.score).pvalue,
            "direction_support": bool(np.mean(nr) > np.mean(cr)),
            "primary_statistical_unit": "patient-celltype pseudobulk",
        })

    pb_scores = pd.DataFrame(pb_rows)
    pb_scores.to_csv(out / "refined_target_state_pseudobulk_scores.tsv", sep="\t", index=False)
    pd.DataFrame(effect_rows).to_csv(out / "refined_target_state_response_effects.tsv", sep="\t", index=False)

    adata = sc.read_h5ad(INPUT / "CNP0004138_QC_sample_annotated.h5ad")
    gene_index = {str(gene).upper(): index for index, gene in enumerate(adata.var_names)}
    score_rows = []
    annotation_rows = []
    confidence_rows = []
    for target_state, definition in TARGET_STATES.items():
        mask = np.asarray(adata.obs.broad_cell_type == definition["cell_type"])
        cell_indices = np.flatnonzero(mask)
        if len(cell_indices) == 0:
            continue
        components = TARGET_STATE_COMPONENTS[target_state]
        component_scores = []
        coverage = []
        for module in components:
            represented = [gene_index[g] for g in STATE_MODULES[module] if g in gene_index]
            coverage.append(len(represented) / len(STATE_MODULES[module]))
            if represented:
                values = adata[cell_indices, represented].X
                if sparse.issparse(values):
                    values = np.asarray(values.mean(axis=1)).ravel()
                else:
                    values = np.asarray(values).mean(axis=1)
                values = (values - np.mean(values)) / (np.std(values, ddof=1) or 1.0)
            else:
                values = np.full(len(cell_indices), np.nan)
            component_scores.append(values)
            for cell_index, value in zip(cell_indices, values):
                score_rows.append({
                    "cell_index": int(cell_index),
                    "barcode": adata.obs_names[cell_index],
                    "patient_id": adata.obs.iloc[cell_index].patient_id,
                    "timepoint": adata.obs.iloc[cell_index].timepoint,
                    "response_standardized": adata.obs.iloc[cell_index].response_standardized,
                    "broad_cell_type": definition["cell_type"],
                    "target_state": target_state,
                    "module": module,
                    "module_score": value,
                    "descriptive_only": True,
                })
        matrix = np.vstack(component_scores).T
        target_score = np.nanmean(matrix, axis=1)
        order = np.argsort(np.nan_to_num(matrix, nan=-np.inf), axis=1)
        top = order[:, -1]
        second = order[:, -2] if matrix.shape[1] > 1 else order[:, -1]
        top_values = matrix[np.arange(len(matrix)), top]
        second_values = matrix[np.arange(len(matrix)), second]
        margin = top_values - second_values if matrix.shape[1] > 1 else np.abs(top_values)
        for local, cell_index in enumerate(cell_indices):
            broad_conf = str(adata.obs.iloc[cell_index].annotation_confidence)
            if coverage[top[local]] >= 0.6 and broad_conf == "high" and margin[local] >= 0.25:
                confidence = "high"
            elif coverage[top[local]] >= 0.4 and margin[local] >= 0.10:
                confidence = "medium"
            else:
                confidence = "low"
            annotation_rows.append({
                "cell_index": int(cell_index),
                "barcode": adata.obs_names[cell_index],
                "patient_id": adata.obs.iloc[cell_index].patient_id,
                "sample_id": adata.obs.iloc[cell_index].sample_id,
                "timepoint": adata.obs.iloc[cell_index].timepoint,
                "response_standardized": adata.obs.iloc[cell_index].response_standardized,
                "broad_cell_type": definition["cell_type"],
                "target_state": target_state,
                "refined_substate": components[top[local]],
                "target_state_score": target_score[local],
                "top_component_score": top_values[local],
                "component_margin": margin[local],
                "annotation_confidence": confidence,
                "sampled_cell_descriptive_only": True,
            })
        confidence_rows.append({
            "target_state": target_state,
            "source_cell_type": definition["cell_type"],
            "n_sampled_cells": len(cell_indices),
            "mean_component_gene_coverage": float(np.mean(coverage)),
            "minimum_component_gene_coverage": float(np.min(coverage)),
            "high_confidence_fraction": float(np.mean([row["annotation_confidence"] == "high" for row in annotation_rows if row["target_state"] == target_state])),
            "inference_boundary": "sampled cell-level annotation is descriptive; patient-celltype pseudobulk is inferential",
        })

    pd.DataFrame(score_rows).to_csv(out / "refined_target_state_scores.tsv", sep="\t", index=False)
    pd.DataFrame(annotation_rows).to_csv(out / "refined_target_state_annotation.tsv", sep="\t", index=False)
    pd.DataFrame(confidence_rows).to_csv(out / "refined_target_state_confidence.tsv", sep="\t", index=False)
    print(f"Scored {len(pb_scores)} target-state pseudobulks and annotated {len(annotation_rows)} sampled cells.")


if __name__ == "__main__":
    main()
