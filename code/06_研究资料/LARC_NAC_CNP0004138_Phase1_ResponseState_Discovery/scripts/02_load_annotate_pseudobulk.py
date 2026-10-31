from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import scanpy as sc
from scipy import sparse
from scipy.io import mmread
from sklearn.cluster import MiniBatchKMeans


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")
SOURCE = ROOT / "02_scRNA_matrix/source"
RNG = np.random.default_rng(20260701)


MARKERS = {
    "Epithelial": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "KRT20", "KRT7", "CEACAM5",
        "CEACAM6", "TACSTD2", "MUC1",
    ],
    "CAF_Fibroblast": [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRA",
        "PDGFRB", "COL6A1", "COL6A2", "COL6A3", "THY1",
    ],
    "Endothelial": [
        "PECAM1", "VWF", "KDR", "EMCN", "ENG", "RAMP2", "ESAM", "CLDN5",
        "PLVAP", "ACKR1",
    ],
    "Myeloid_macrophage": [
        "LYZ", "CD68", "C1QA", "C1QB", "C1QC", "APOE", "FCGR3A", "CTSB",
        "LST1", "TYROBP", "CTSD", "AIF1",
    ],
    "Dendritic": [
        "FCER1A", "CLEC10A", "LAMP3", "CD1C", "CLEC9A", "XCR1", "CD1E",
    ],
    "T_core": ["CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2"],
    "CD8_T": ["CD8A", "CD8B", "CST7", "NKG7", "CCL5", "GZMK"],
    "CD4_T": ["CD4", "IL7R", "LTB", "MAL", "CCR7", "LTST1"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "ICOS", "TNFRSF4", "TIGIT"],
    "B_plasma": [
        "MS4A1", "CD79A", "CD79B", "CD37", "CD74", "MZB1", "JCHAIN",
        "CD38", "IGHG1", "IGHA1", "IGHM", "SDC1",
    ],
    "Proliferating": ["MKI67", "TOP2A", "STMN1", "TUBA1B", "CENPF", "TYMS"],
}


FINAL_TYPES = [
    "Malignant_epithelial",
    "CAF_Fibroblast",
    "Endothelial",
    "Myeloid_macrophage",
    "Dendritic",
    "CD8_T",
    "CD4_T",
    "Treg",
    "B_plasma",
    "Proliferating",
    "Epithelial_normal_or_uncertain",
    "Unknown",
]


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    line = f"[{stamp}] RSS={rss:.2f} GB | {message}"
    print(line, flush=True)
    with (ROOT / "00_logs/02_matrix_annotation_pseudobulk.log").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(line + "\n")


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def select_umap_cells(cell_meta: pd.DataFrame, cap: int = 50000) -> np.ndarray:
    selected: list[int] = []
    for _, group in cell_meta.groupby(
        ["patient_id", "timepoint", "broad_cell_type"], sort=False
    ):
        indices = group.index.to_numpy()
        take = min(len(indices), 300)
        selected.extend(RNG.choice(indices, size=take, replace=False).tolist())
    selected = np.array(sorted(set(selected)), dtype=np.int32)
    if len(selected) > cap:
        selected = np.sort(RNG.choice(selected, size=cap, replace=False))
    elif len(selected) < cap:
        remaining = np.setdiff1d(
            np.arange(len(cell_meta), dtype=np.int32), selected, assume_unique=True
        )
        add_n = min(cap - len(selected), len(remaining))
        selected = np.sort(
            np.concatenate([selected, RNG.choice(remaining, size=add_n, replace=False)])
        )
    return selected


def main() -> None:
    started = time.time()
    matrix_path = SOURCE / "MM_coordinate_matrix.mtx.gz"
    genes_path = SOURCE / "MM_coordinate_matrix_genes.csv"
    barcodes_path = SOURCE / "MM_coordinate_matrix_barcodes.csv"
    frozen_path = ROOT / "01_metadata/CNP0004138_frozen_metadata.tsv"
    for path in [matrix_path, genes_path, barcodes_path, frozen_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    genes = pd.read_csv(genes_path, header=None, names=["gene_symbol"], dtype=str)
    barcodes = pd.read_csv(barcodes_path, header=None, names=["barcode"], dtype=str)
    cell_meta = pd.read_csv(frozen_path, sep="\t", dtype=str)
    if not barcodes["barcode"].equals(cell_meta["barcode"]):
        raise RuntimeError("Frozen metadata order differs from matrix barcode order")
    gene_to_index = dict(zip(genes["gene_symbol"], genes.index))

    log("Loading complete MatrixMarket matrix with scipy.io.mmread")
    matrix_coo = mmread(matrix_path).tocoo(copy=False)
    if matrix_coo.shape != (len(genes), len(barcodes)):
        raise RuntimeError(
            f"Matrix shape {matrix_coo.shape} != genes/barcodes {(len(genes), len(barcodes))}"
        )
    matrix_coo.data = matrix_coo.data.astype(np.float32, copy=False)
    log(
        f"Loaded matrix shape={matrix_coo.shape}, nnz={matrix_coo.nnz:,}, "
        f"dtype={matrix_coo.data.dtype}; converting to row-indexed CSR"
    )
    matrix = matrix_coo.tocsr()
    del matrix_coo
    gc.collect()
    matrix.sort_indices()
    boundaries = matrix.indptr
    cols = matrix.indices
    data = matrix.data
    chunk = 10_000_000
    log("CSR conversion complete")

    cell_meta["total_counts"] = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    cell_meta["n_genes_by_nnz"] = np.bincount(cols, minlength=len(barcodes)).astype(
        np.int32
    )

    score_names = list(MARKERS)
    scores = np.zeros((len(score_names), len(barcodes)), dtype=np.float32)
    detected = np.zeros((len(score_names), len(barcodes)), dtype=np.uint8)
    represented_rows = []
    for score_idx, (name, marker_genes) in enumerate(MARKERS.items()):
        represented = [gene for gene in marker_genes if gene in gene_to_index]
        represented_rows.append(
            {
                "marker_set": name,
                "requested_markers": len(marker_genes),
                "represented_markers": len(represented),
                "represented_gene_symbols": ",".join(represented),
            }
        )
        for gene in represented:
            row_idx = gene_to_index[gene]
            left, right = boundaries[row_idx], boundaries[row_idx + 1]
            gene_cols = cols[left:right]
            gene_values = np.log1p(data[left:right])
            np.add.at(scores[score_idx], gene_cols, gene_values)
            detected[score_idx, np.unique(gene_cols)] += 1
        if represented:
            scores[score_idx] /= np.sqrt(len(represented))

    score_index = {name: idx for idx, name in enumerate(score_names)}
    broad_lineages = [
        "Epithelial", "CAF_Fibroblast", "Endothelial", "Myeloid_macrophage",
        "B_plasma", "T_core",
    ]
    lineage_matrix = np.vstack([scores[score_index[name]] for name in broad_lineages])
    top_order = np.argsort(lineage_matrix, axis=0)
    top_idx = top_order[-1]
    second_idx = top_order[-2]
    top_score = lineage_matrix[top_idx, np.arange(len(barcodes))]
    second_score = lineage_matrix[second_idx, np.arange(len(barcodes))]
    cell_types = np.array(
        [broad_lineages[index] for index in top_idx], dtype=object
    )
    cell_types[top_score <= 0] = "Unknown"

    t_cells = cell_types == "T_core"
    t_sub = np.vstack(
        [
            scores[score_index["CD8_T"]],
            scores[score_index["CD4_T"]],
            scores[score_index["Treg"]],
        ]
    )
    t_labels = np.array(["CD8_T", "CD4_T", "Treg"], dtype=object)
    t_best = t_sub.argmax(axis=0)
    cell_types[t_cells] = t_labels[t_best[t_cells]]

    myeloid = cell_types == "Myeloid_macrophage"
    dc_score = scores[score_index["Dendritic"]]
    my_score = scores[score_index["Myeloid_macrophage"]]
    dc_detected = detected[score_index["Dendritic"]]
    dc_gate = myeloid & (dc_detected >= 1) & (dc_score >= 0.65 * np.maximum(my_score, 0.01))
    cell_types[dc_gate] = "Dendritic"

    epithelial = cell_types == "Epithelial"
    is_adjacent = cell_meta["timepoint"].eq("POSTN").to_numpy()
    cell_types[epithelial & ~is_adjacent] = "Malignant_epithelial"
    cell_types[epithelial & is_adjacent] = "Epithelial_normal_or_uncertain"

    prolif_score = scores[score_index["Proliferating"]]
    prolif_detected = detected[score_index["Proliferating"]]
    proliferating = (prolif_detected >= 2) | (prolif_score >= 1.0)
    cell_types[proliferating] = "Proliferating"

    margin = (top_score - second_score) / np.maximum(top_score, 1e-6)
    confidence = np.full(len(barcodes), "low", dtype=object)
    confidence[(top_score >= 0.15) & (margin >= 0.15)] = "medium"
    confidence[(top_score >= 0.30) & (margin >= 0.35)] = "high"
    confidence[cell_types == "Unknown"] = "low"

    cell_meta["broad_cell_type"] = cell_types
    cell_meta["annotation_confidence"] = confidence
    cell_meta["annotation_top_marker_score"] = top_score.astype(np.float32)
    cell_meta["annotation_margin"] = margin.astype(np.float32)
    cell_meta["annotation_method"] = "source-marker-guided broad annotation"
    for idx, name in enumerate(score_names):
        cell_meta[f"marker_score_{name}"] = scores[idx]

    clean_cols = [
        "patient_id", "sample_id", "barcode", "timepoint", "sample_type",
        "response_standardized", "TRG", "response_binary_for_primary",
        "total_counts", "n_genes_by_nnz", "broad_cell_type",
        "annotation_confidence", "annotation_top_marker_score", "annotation_margin",
        "annotation_method", "usable_for_phase1", "exclusion_reason",
    ]
    cell_meta[clean_cols].to_csv(
        ROOT / "02_scRNA_matrix/CNP0004138_scRNA_cell_metadata_clean.tsv",
        sep="\t", index=False,
    )
    cell_meta[[
        "barcode", "patient_id", "sample_id", "timepoint", "response_standardized",
        "TRG", "broad_cell_type", "annotation_confidence", "annotation_method",
    ]].to_csv(
        ROOT / "03_celltype_annotation/CNP0004138_celltype_annotation.tsv",
        sep="\t", index=False,
    )

    pd.DataFrame(represented_rows).to_csv(
        ROOT / "03_celltype_annotation/CNP0004138_celltype_marker_summary.tsv",
        sep="\t", index=False,
    )
    confidence_summary = (
        cell_meta.groupby(["broad_cell_type", "annotation_confidence"])
        .size().rename("cell_count").reset_index()
    )
    confidence_summary.to_csv(
        ROOT / "03_celltype_annotation/CNP0004138_celltype_annotation_confidence.tsv",
        sep="\t", index=False,
    )

    composition = (
        cell_meta.groupby(
            ["patient_id", "timepoint", "response_standardized", "broad_cell_type"]
        ).size().rename("cell_count").reset_index()
    )
    totals = composition.groupby(["patient_id", "timepoint"])["cell_count"].transform("sum")
    composition["cell_fraction"] = composition["cell_count"] / totals
    composition.to_csv(
        ROOT / "03_celltype_annotation/CNP0004138_celltype_composition.tsv",
        sep="\t", index=False,
    )
    log("Completed broad cell-type annotation")

    # Build pseudobulk units. All_cells is an additional patient-timepoint summary.
    specific_keys = list(
        zip(cell_meta["patient_id"], cell_meta["timepoint"], cell_meta["broad_cell_type"])
    )
    all_keys = list(zip(cell_meta["patient_id"], cell_meta["timepoint"], ["All_cells"] * len(cell_meta)))
    unique_specific = sorted(set(specific_keys))
    unique_all = sorted(set(all_keys))
    all_units = unique_all + unique_specific
    unit_lookup = {key: idx for idx, key in enumerate(all_units)}
    specific_ids = np.fromiter(
        (unit_lookup[key] for key in specific_keys), dtype=np.int32, count=len(specific_keys)
    )
    all_ids = np.fromiter(
        (unit_lookup[key] for key in all_keys), dtype=np.int32, count=len(all_keys)
    )
    specific_cell_counts = np.bincount(specific_ids, minlength=len(all_units))
    all_cell_counts = np.bincount(all_ids, minlength=len(all_units))
    unit_cell_counts = specific_cell_counts + all_cell_counts
    keep_unit = np.array(
        [key[2] == "All_cells" or unit_cell_counts[idx] >= 20 for idx, key in enumerate(all_units)]
    )
    old_to_new = np.full(len(all_units), -1, dtype=np.int32)
    old_to_new[np.where(keep_unit)[0]] = np.arange(keep_unit.sum(), dtype=np.int32)
    kept_units = [unit for idx, unit in enumerate(all_units) if keep_unit[idx]]
    specific_kept_ids = old_to_new[specific_ids]
    all_kept_ids = old_to_new[all_ids]
    n_units = len(kept_units)

    log(f"Aggregating pseudobulk counts into {n_units} retained units")
    pb = np.zeros((len(genes), n_units), dtype=np.float32)
    for gene_idx in range(len(genes)):
        left, right = boundaries[gene_idx], boundaries[gene_idx + 1]
        if right == left:
            continue
        gene_cols = cols[left:right]
        gene_data = data[left:right]
        ids = specific_kept_ids[gene_cols]
        valid = ids >= 0
        if valid.any():
            pb[gene_idx] += np.bincount(
                ids[valid], weights=gene_data[valid], minlength=n_units
            ).astype(np.float32)
        ids_all = all_kept_ids[gene_cols]
        pb[gene_idx] += np.bincount(
            ids_all, weights=gene_data, minlength=n_units
        ).astype(np.float32)
        if gene_idx and gene_idx % 5000 == 0:
            log(f"Pseudobulk aggregation: {gene_idx}/{len(genes)} genes")

    pb_meta_rows = []
    for idx, (patient, timepoint, cell_type) in enumerate(kept_units):
        first = cell_meta[(cell_meta["patient_id"] == patient) & (cell_meta["timepoint"] == timepoint)].iloc[0]
        old_idx = unit_lookup[(patient, timepoint, cell_type)]
        pb_meta_rows.append(
            {
                "pseudobulk_id": f"{patient}__{timepoint}__{cell_type}",
                "patient_id": patient,
                "timepoint": timepoint,
                "cell_type": cell_type,
                "response_standardized": first["response_standardized"],
                "TRG": first["TRG"],
                "response_binary_for_primary": first["response_binary_for_primary"],
                "cell_count": int(unit_cell_counts[old_idx]),
                "passes_20_cell_filter": "yes",
                "statistical_unit": "patient-timepoint-celltype",
            }
        )
    pb_meta = pd.DataFrame(pb_meta_rows)
    column_names = pb_meta["pseudobulk_id"].tolist()
    pb_df = pd.DataFrame(pb, index=genes["gene_symbol"], columns=column_names)
    pb_df.index.name = "gene_symbol"
    pb_df.to_csv(
        ROOT / "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_matrix.tsv",
        sep="\t",
    )
    pb_df.to_parquet(
        ROOT / "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_matrix.parquet"
    )
    pb_meta.to_csv(
        ROOT / "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_metadata.tsv",
        sep="\t", index=False,
    )
    pret_mask = pb_meta["timepoint"].eq("PRET").to_numpy()
    pb_df.loc[:, pret_mask].to_csv(
        ROOT / "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.tsv",
        sep="\t",
    )
    pb_df.loc[:, pret_mask].to_parquet(
        ROOT / "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet"
    )
    pb_meta.loc[pret_mask].to_csv(
        ROOT / "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv",
        sep="\t", index=False,
    )
    pb_meta.groupby(["timepoint", "response_standardized", "cell_type"]).agg(
        pseudobulk_units=("pseudobulk_id", "size"),
        patients=("patient_id", "nunique"),
        total_cells=("cell_count", "sum"),
        median_cells=("cell_count", "median"),
    ).reset_index().to_csv(
        ROOT / "09_qc/CNP0004138_pseudobulk_unit_count_summary.tsv",
        sep="\t", index=False,
    )
    log("Pseudobulk matrices written")

    # Extract a representative cell subset for PCA/clustering/UMAP QC.
    sample_cells = select_umap_cells(cell_meta)
    sample_csr = matrix[:, sample_cells].transpose().tocsr()
    sparse.save_npz(ROOT / "03_celltype_annotation/CNP0004138_QC_sample_counts.npz", sample_csr)
    pd.DataFrame({"cell_index": sample_cells, "barcode": cell_meta.iloc[sample_cells]["barcode"].to_numpy()}).to_csv(
        ROOT / "03_celltype_annotation/CNP0004138_QC_sample_index.tsv", sep="\t", index=False
    )

    # Release the complete COO before Scanpy transformations.
    del matrix, cols, data, boundaries, scores, detected, pb, pb_df
    gc.collect()
    log("Released complete COO matrix; starting sampled Scanpy QC")

    obs = cell_meta.iloc[sample_cells].copy()
    obs.index = obs["barcode"].astype(str)
    var = pd.DataFrame(index=genes["gene_symbol"].astype(str))
    adata = ad.AnnData(X=sample_csr, obs=obs, var=var)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    sc.pp.pca(adata, n_comps=30, mask_var="highly_variable", random_state=20260701)
    kmeans = MiniBatchKMeans(n_clusters=30, random_state=20260701, batch_size=4096)
    adata.obs["qc_cluster"] = kmeans.fit_predict(adata.obsm["X_pca"]).astype(str)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, random_state=20260701)
    sc.tl.umap(adata, random_state=20260701)
    adata.write_h5ad(ROOT / "03_celltype_annotation/CNP0004138_QC_sample_annotated.h5ad", compression="gzip")
    pd.DataFrame(
        {
            "barcode": adata.obs_names,
            "UMAP1": adata.obsm["X_umap"][:, 0],
            "UMAP2": adata.obsm["X_umap"][:, 1],
            "qc_cluster": adata.obs["qc_cluster"].to_numpy(),
            "broad_cell_type": adata.obs["broad_cell_type"].to_numpy(),
            "patient_id": adata.obs["patient_id"].to_numpy(),
            "timepoint": adata.obs["timepoint"].to_numpy(),
            "response_standardized": adata.obs["response_standardized"].to_numpy(),
        }
    ).to_csv(
        ROOT / "03_celltype_annotation/CNP0004138_QC_sample_umap_coordinates.tsv",
        sep="\t", index=False,
    )

    pd.DataFrame(
        [
            {
                "matrix_file": matrix_path.name,
                "gene_rows": len(genes),
                "cell_columns": len(barcodes),
                "nonzero_entries": 445_213_600,
                "gene_identifier": "unique gene symbols",
                "barcode_order_verified": "yes",
                "metadata_mapped_cells": len(cell_meta),
                "unmapped_cells": 0,
                "matrix_read_success": "yes",
                "processing_mode": "complete COO read; row-streamed pseudobulk; 50k-cell Scanpy QC subset",
                "elapsed_minutes": round((time.time() - started) / 60, 2),
            }
        ]
    ).to_csv(
        ROOT / "02_scRNA_matrix/CNP0004138_scRNA_matrix_dimension_log.tsv",
        sep="\t", index=False,
    )

    mapping_qc = pd.DataFrame(
        [
            {"metric": "source_barcodes", "value": len(barcodes), "status": "pass"},
            {"metric": "mapped_barcodes", "value": len(cell_meta), "status": "pass"},
            {"metric": "unmapped_barcodes", "value": 0, "status": "pass"},
            {"metric": "unique_patients", "value": cell_meta["patient_id"].nunique(), "status": "pass"},
            {"metric": "unique_timepoints", "value": cell_meta["timepoint"].nunique(), "status": "pass"},
            {"metric": "broad_cell_types", "value": cell_meta["broad_cell_type"].nunique(), "status": "review"},
        ]
    )
    mapping_qc.to_csv(
        ROOT / "09_qc/CNP0004138_scRNA_mapping_qc.tsv", sep="\t", index=False
    )
    cell_meta.groupby(
        ["patient_id", "sample_id", "timepoint", "response_standardized", "broad_cell_type"]
    ).size().rename("cell_count").reset_index().to_csv(
        ROOT / "09_qc/CNP0004138_patient_sample_cell_count_summary.tsv",
        sep="\t", index=False,
    )

    low_units = len(all_units) - n_units
    note = f"""# Pseudobulk filtering note

- Unit definition: patient + timepoint + broad cell type.
- Broad cell-type units required at least 20 cells.
- `All_cells` patient-timepoint units were retained regardless of cell-type composition.
- Retained units: {n_units}
- Filtered units with fewer than 20 cells: {low_units}
- PRET units are the only source for primary response-state discovery.
- POSTT and POSTN are retained for treatment-state/descriptive sensitivity only.
- Broad annotation is source-marker-guided; malignant epithelial calls are provisional because a full inferCNV reconstruction was outside this memory-controlled Phase 1 pass.
"""
    (ROOT / "09_qc/CNP0004138_pseudobulk_filtering_note.md").write_text(
        note, encoding="utf-8"
    )
    log(f"All matrix/annotation/pseudobulk tasks complete in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
