from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_config import COMPARATOR_MODULES, DOMAIN_MODULES, RESPONSE, SECTIONS, normalize_gene


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization")
SOURCE = ROOT / "03_stereoseq_matrix" / "source"


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("rt", encoding="utf-8", errors="replace", newline="")


def matrix_path(section: str) -> Path:
    gz = SOURCE / f"{section}.exp.count.tsv.gz"
    return gz if gz.exists() else SOURCE / f"{section}.exp.count.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_gene_set() -> set[str]:
    programs = pd.read_csv(ROOT / "01_inputs_from_phase1" / "CNP0004138_locked_programs_by_celltype.tsv", sep="\t")
    genes = {normalize_gene(x) for x in programs["gene_symbol"]}
    for module in list(DOMAIN_MODULES.values()) + list(COMPARATOR_MODULES.values()):
        genes.update(normalize_gene(x) for x in module)
    return genes


def process_section(section: str, selected: set[str]):
    source = matrix_path(section)
    coordinate_path = SOURCE / f"{section}.meta.info.tsv.txt"
    output_clean = ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_expression_clean.tsv"
    coordinates = pd.read_csv(coordinate_path, sep="\t", index_col=0)
    coordinates.index = coordinates.index.astype(str)

    seen = Counter()
    selected_counts: dict[str, np.ndarray] = {}
    malformed_rows = 0
    gene_count = 0
    mitochondrial_gene_count = 0
    with open_text(source) as handle, output_clean.open("wt", encoding="utf-8", newline="") as clean:
        header = handle.readline().rstrip("\n\r")
        header_parts = header.split("\t")
        bins = header_parts[1:]
        clean.write(header + "\n")
        n_bins = len(bins)
        total_counts = np.zeros(n_bins, dtype=np.float64)
        detected_genes = np.zeros(n_bins, dtype=np.uint32)
        mitochondrial_counts = np.zeros(n_bins, dtype=np.float64)

        for line in handle:
            line = line.rstrip("\n\r")
            if not line:
                continue
            gene, sep, payload = line.partition("\t")
            if not sep:
                malformed_rows += 1
                continue
            gene = normalize_gene(gene)
            values = np.fromstring(payload, sep="\t", dtype=np.float32)
            if values.size != n_bins:
                malformed_rows += 1
                continue
            clean.write(gene + "\t" + payload + "\n")
            gene_count += 1
            seen[gene] += 1
            total_counts += values
            detected_genes += values > 0
            if gene.startswith("MT-"):
                mitochondrial_counts += values
                mitochondrial_gene_count += 1
            if gene in selected:
                if gene in selected_counts:
                    selected_counts[gene] += values
                else:
                    selected_counts[gene] = values.copy()

    bins_array = np.asarray(bins, dtype=str)
    if len(coordinates) != len(bins_array) or set(coordinates.index) != set(bins_array):
        raise ValueError(
            f"{section}: coordinate/matrix mismatch: coordinates={len(coordinates)}, matrix={len(bins_array)}, overlap={len(set(coordinates.index) & set(bins_array))}"
        )
    coordinates = coordinates.reindex(bins_array)
    coordinates["section_id"] = section
    coordinates["response_group"] = RESPONSE[section]
    coordinates["total_counts"] = total_counts
    coordinates["detected_genes"] = detected_genes
    coordinates["mitochondrial_counts"] = mitochondrial_counts
    coordinates["mitochondrial_fraction"] = np.divide(
        mitochondrial_counts,
        total_counts,
        out=np.zeros_like(total_counts),
        where=total_counts > 0,
    )
    coordinates["nonzero_bin"] = total_counts > 0
    coordinates["low_complexity_lt200_genes"] = detected_genes < 200
    coordinates.to_parquet(ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_bin_qc.parquet")

    genes = sorted(selected_counts)
    selected_matrix = np.vstack([selected_counts[gene] for gene in genes]).astype(np.float32)
    np.savez_compressed(
        ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_analysis_gene_counts.npz",
        genes=np.asarray(genes, dtype=str),
        bins=bins_array,
        counts=selected_matrix,
        total_counts=total_counts,
        detected_genes=detected_genes,
        mitochondrial_fraction=coordinates["mitochondrial_fraction"].to_numpy(),
    )

    q = np.quantile(total_counts, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    f = np.quantile(detected_genes, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    duplicate_symbols = {gene: n for gene, n in seen.items() if n > 1}
    qc_status = "low_quality" if section == "ST-NR2" or np.median(detected_genes) < 300 else "pass"
    summary = {
        "section_id": section,
        "response_group": RESPONSE[section],
        "matrix_source_file": source.name,
        "matrix_sha256": sha256(source),
        "coordinate_source_file": coordinate_path.name,
        "coordinate_sha256": sha256(coordinate_path),
        "n_bins": n_bins,
        "n_coordinate_rows": len(coordinates),
        "n_gene_rows": gene_count,
        "n_unique_gene_symbols": len(seen),
        "n_duplicate_gene_symbols": len(duplicate_symbols),
        "duplicate_aggregation_for_analysis": "sum counts by uppercase gene symbol",
        "n_analysis_selected_genes": len(genes),
        "n_mitochondrial_gene_rows": mitochondrial_gene_count,
        "n_malformed_rows": malformed_rows,
        "n_zero_count_bins": int((total_counts == 0).sum()),
        "n_bins_detected_genes_ge200": int((detected_genes >= 200).sum()),
        "fraction_bins_detected_genes_ge200": float((detected_genes >= 200).mean()),
        "total_counts_min": q[0],
        "total_counts_p10": q[1],
        "total_counts_p25": q[2],
        "total_counts_median": q[3],
        "total_counts_p75": q[4],
        "total_counts_p90": q[5],
        "total_counts_max": q[6],
        "detected_genes_min": f[0],
        "detected_genes_p10": f[1],
        "detected_genes_p25": f[2],
        "detected_genes_median": f[3],
        "detected_genes_p75": f[4],
        "detected_genes_p90": f[5],
        "detected_genes_max": f[6],
        "mitochondrial_fraction_median": float(np.median(coordinates["mitochondrial_fraction"])),
        "QC_status": qc_status,
        "analysis_retention": "retained; low quality is handled by sensitivity analysis" if qc_status == "low_quality" else "retained",
    }
    duplicate_table = pd.DataFrame(
        [{"section_id": section, "gene_symbol": gene, "source_rows": n} for gene, n in duplicate_symbols.items()]
    )
    return summary, coordinates.reset_index(names="matrix_bin_id"), duplicate_table


def main():
    selected = selected_gene_set()
    summaries = []
    coordinate_frames = []
    duplicate_frames = []
    for section in SECTIONS:
        print(f"Processing {section}...", flush=True)
        summary, coordinates, duplicates = process_section(section, selected)
        summaries.append(summary)
        coordinate_frames.append(coordinates)
        if len(duplicates):
            duplicate_frames.append(duplicates)
        print(
            f"{section}: {summary['n_gene_rows']} genes x {summary['n_bins']} bins; median detected={summary['detected_genes_median']}; QC={summary['QC_status']}",
            flush=True,
        )

    qc = pd.DataFrame(summaries)
    qc.to_csv(ROOT / "03_stereoseq_matrix" / "StereoSeq_matrix_QC_summary.tsv", sep="\t", index=False)
    qc[["section_id", "matrix_source_file", "n_gene_rows", "n_unique_gene_symbols", "n_bins", "n_coordinate_rows", "n_malformed_rows"]].to_csv(
        ROOT / "03_stereoseq_matrix" / "StereoSeq_matrix_dimension_log.tsv", sep="\t", index=False
    )
    pd.concat(coordinate_frames, ignore_index=True).to_parquet(
        ROOT / "03_stereoseq_matrix" / "StereoSeq_bin_coordinate_metadata.parquet"
    )
    # A compact TSV keeps the requested coordinate deliverable practical; per-bin QC remains in Parquet.
    pd.concat(coordinate_frames, ignore_index=True)[
        ["matrix_bin_id", "section_id", "response_group", "bin_id", "bin_size", "x", "y", "total_counts", "detected_genes", "mitochondrial_fraction", "nonzero_bin", "low_complexity_lt200_genes"]
    ].to_csv(ROOT / "03_stereoseq_matrix" / "StereoSeq_bin_coordinate_metadata.tsv", sep="\t", index=False)
    if duplicate_frames:
        pd.concat(duplicate_frames, ignore_index=True).to_csv(
            ROOT / "03_stereoseq_matrix" / "StereoSeq_duplicate_gene_symbols.tsv", sep="\t", index=False
        )
    else:
        pd.DataFrame(columns=["section_id", "gene_symbol", "source_rows"]).to_csv(
            ROOT / "03_stereoseq_matrix" / "StereoSeq_duplicate_gene_symbols.tsv", sep="\t", index=False
        )

    spatial_path = ROOT / "02_spatial_metadata" / "StereoSeq_frozen_spatial_metadata.tsv"
    spatial = pd.read_csv(spatial_path, sep="\t")
    spatial = spatial.drop(columns=[column for column in ["actual_bin_number", "actual_gene_number"] if column in spatial])
    spatial = spatial.merge(
        qc[["section_id", "n_bins", "n_unique_gene_symbols", "QC_status"]].rename(
            columns={"n_bins": "actual_bin_number", "n_unique_gene_symbols": "actual_gene_number", "QC_status": "QC_status_measured"}
        ),
        on="section_id",
        how="left",
    )
    spatial["QC_status"] = spatial.pop("QC_status_measured")
    spatial["coordinate_available"] = True
    spatial["matrix_available"] = True
    spatial.to_csv(spatial_path, sep="\t", index=False)
    qc.to_csv(ROOT / "02_spatial_metadata" / "StereoSeq_section_qc_summary.tsv", sep="\t", index=False)
    print("Completed streaming matrix QC and selected-gene extraction.")


if __name__ == "__main__":
    main()
