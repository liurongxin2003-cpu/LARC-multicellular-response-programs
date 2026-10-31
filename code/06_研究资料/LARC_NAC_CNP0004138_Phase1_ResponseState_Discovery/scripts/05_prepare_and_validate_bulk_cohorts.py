from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")
RNG = np.random.default_rng(4138)


def clean(value: str) -> str:
    return value.strip().strip('"')


def parse_family_soft(path: Path) -> pd.DataFrame:
    records: list[dict] = []
    current: dict | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current:
                    records.append(current)
                current = {"geo_accession": clean(line.split("=", 1)[1])}
            elif current is not None and line.startswith("!Sample_"):
                key, value = line.split("=", 1)
                key = key.replace("!Sample_", "").strip()
                value = clean(value)
                if key == "characteristics_ch1" and ":" in value:
                    item_key, item_value = value.split(":", 1)
                    current[item_key.strip().lower()] = item_value.strip()
                elif key not in current:
                    current[key] = value
    if current:
        records.append(current)
    return pd.DataFrame(records)


def parse_series_matrix_metadata(path: Path) -> pd.DataFrame:
    fields: dict[str, list[list[str]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            row = next(csv.reader([line.rstrip("\n")], delimiter="\t"))
            fields[row[0]].append([clean(x) for x in row[1:]])

    accessions = fields["!Sample_geo_accession"][0]
    metadata = pd.DataFrame({"geo_accession": accessions})
    simple = {
        "!Sample_title": "title",
        "!Sample_source_name_ch1": "source_name",
        "!Sample_description": "description",
    }
    for field, output_name in simple.items():
        if field in fields and len(fields[field][0]) == len(metadata):
            metadata[output_name] = fields[field][0]
    for row in fields.get("!Sample_characteristics_ch1", []):
        parsed = [value.split(":", 1) if ":" in value else ("unkeyed", value) for value in row]
        keys = {key.strip().lower() for key, _ in parsed}
        if len(keys) == 1:
            key = next(iter(keys))
            metadata[key] = [value.strip() for _, value in parsed]
    return metadata


def read_series_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t", comment="!", index_col=0, compression="gzip")
    matrix.index = matrix.index.astype(str).str.strip('"')
    matrix.columns = matrix.columns.astype(str).str.strip('"')
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix


def read_platform(path: Path) -> pd.DataFrame:
    rows = []
    in_table = False
    header = None
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if not in_table:
                continue
            if header is None:
                header = line.split("\t")
                continue
            values = line.split("\t")
            values += [""] * (len(header) - len(values))
            rows.append(dict(zip(header, values)))
    return pd.DataFrame(rows)


def aggregate_to_symbols(matrix: pd.DataFrame, mapping: pd.Series, how: str) -> pd.DataFrame:
    shared = matrix.index.intersection(mapping.index)
    mapped = matrix.loc[shared].copy()
    mapped.insert(0, "gene_symbol", mapping.loc[shared].values)
    mapped = mapped[mapped["gene_symbol"].notna() & (mapped["gene_symbol"] != "")]
    numeric = mapped.columns.drop("gene_symbol")
    if how == "sum":
        return mapped.groupby("gene_symbol", sort=False)[numeric].sum()
    return mapped.groupby("gene_symbol", sort=False)[numeric].mean()


def log_cpm(counts: pd.DataFrame) -> pd.DataFrame:
    library = counts.sum(axis=0).replace(0, np.nan)
    return np.log2(counts.divide(library, axis=1) * 1_000_000 + 1)


def load_programs() -> pd.DataFrame:
    path = ROOT / "05_response_state_discovery" / "CNP0004138_locked_programs_by_celltype.tsv"
    programs = pd.read_csv(path, sep="\t")
    programs["gene_symbol"] = programs["gene_symbol"].astype(str).str.upper()
    return programs


def score_programs(expression: pd.DataFrame, programs: pd.DataFrame, dataset: str):
    expression = expression.copy()
    expression.index = expression.index.astype(str).str.upper()
    expression = expression.groupby(level=0).mean()
    gene_sd = expression.std(axis=1, ddof=0).replace(0, np.nan)
    z = expression.sub(expression.mean(axis=1), axis=0).div(gene_sd, axis=0)
    scores = []
    representation = []
    for source, program in programs.groupby("source_cell_type", sort=False):
        nr_all = program.loc[program["program_direction"] == "NR_high", "gene_symbol"].drop_duplicates().tolist()
        cr_all = program.loc[program["program_direction"] == "CR_high", "gene_symbol"].drop_duplicates().tolist()
        nr = [gene for gene in nr_all if gene in z.index and z.loc[gene].notna().any()]
        cr = [gene for gene in cr_all if gene in z.index and z.loc[gene].notna().any()]
        nr_score = z.loc[nr].mean(axis=0) if nr else pd.Series(np.nan, index=z.columns)
        cr_score = z.loc[cr].mean(axis=0) if cr else pd.Series(np.nan, index=z.columns)
        for sample in z.columns:
            scores.append(
                {
                    "dataset": dataset,
                    "sample_id": sample,
                    "program_source_cell_type": source,
                    "NR_high_score": nr_score[sample],
                    "CR_high_score": cr_score[sample],
                    "resistance_score": nr_score[sample] - cr_score[sample],
                    "response_score": cr_score[sample] - nr_score[sample],
                }
            )
        representation.append(
            {
                "dataset": dataset,
                "program_source_cell_type": source,
                "NR_high_locked_n": len(nr_all),
                "NR_high_represented_n": len(nr),
                "CR_high_locked_n": len(cr_all),
                "CR_high_represented_n": len(cr),
                "total_locked_n": len(set(nr_all + cr_all)),
                "total_represented_n": len(set(nr + cr)),
                "represented_fraction": len(set(nr + cr)) / max(len(set(nr_all + cr_all)), 1),
                "scoring_method": "per-gene cohort z-score; mean(NR-high) minus mean(CR-high)",
            }
        )
    return pd.DataFrame(scores), pd.DataFrame(representation)


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return correction * (x.mean() - y.mean()) / pooled


def bootstrap_difference(x: np.ndarray, y: np.ndarray, iterations: int = 10000):
    differences = np.empty(iterations)
    for i in range(iterations):
        differences[i] = RNG.choice(x, len(x), replace=True).mean() - RNG.choice(y, len(y), replace=True).mean()
    return np.quantile(differences, [0.025, 0.975])


def validate(
    scores: pd.DataFrame,
    group_col: str,
    poor_label: str,
    good_label: str,
    cohort_subset: str = "all_evaluable",
) -> pd.DataFrame:
    results = []
    for source, frame in scores.groupby("program_source_cell_type", sort=False):
        poor = frame.loc[frame[group_col] == poor_label, "resistance_score"].dropna().to_numpy()
        good = frame.loc[frame[group_col] == good_label, "resistance_score"].dropna().to_numpy()
        if len(poor) < 2 or len(good) < 2:
            continue
        mw = mannwhitneyu(poor, good, alternative="two-sided")
        welch = ttest_ind(poor, good, equal_var=False)
        ci = bootstrap_difference(poor, good)
        labels = (frame[group_col] == poor_label).astype(int)
        auc = roc_auc_score(labels, frame["resistance_score"])
        results.append(
            {
                "program_source_cell_type": source,
                "cohort_subset": cohort_subset,
                "endpoint": f"{poor_label}_vs_{good_label}",
                "poor_response_label": poor_label,
                "good_response_label": good_label,
                "n_poor": len(poor),
                "n_good": len(good),
                "mean_resistance_poor": poor.mean(),
                "mean_resistance_good": good.mean(),
                "mean_difference_poor_minus_good": poor.mean() - good.mean(),
                "bootstrap_CI95_low": ci[0],
                "bootstrap_CI95_high": ci[1],
                "hedges_g_poor_minus_good": hedges_g(poor, good),
                "mannwhitney_U": mw.statistic,
                "mannwhitney_p": mw.pvalue,
                "welch_t": welch.statistic,
                "welch_p": welch.pvalue,
                "ROC_AUC_poor_response": auc,
                "direction_supports_locked_hypothesis": (poor.mean() - good.mean()) > 0,
            }
        )
    result = pd.DataFrame(results)
    if len(result):
        result["mannwhitney_FDR_BH"] = multipletests(result["mannwhitney_p"], method="fdr_bh")[1]
    return result


def write_readiness(path: Path, rows: list[tuple[str, object]]):
    pd.DataFrame(rows, columns=["field", "value"]).to_csv(path, sep="\t", index=False)


def prepare_gse209746(programs: pd.DataFrame):
    out = ROOT / "07_gse209746_validation"
    source = out / "source"
    metadata = parse_family_soft(source / "GSE209746_family.soft.gz")
    metadata = metadata.rename(columns={"title": "sample_id", "msi status": "msi_status", "response": "response_raw"})
    metadata["response_standardized"] = metadata["response_raw"].map(
        {"CR": "complete_response", "iCR": "incomplete_response", "NA": "unknown"}
    ).fillna("unknown")
    metadata["pretreatment_verified"] = metadata["tissue"].eq("pre-treatment endoscopic biopsies")

    counts = pd.read_csv(source / "GSE209746_geneExp_n_114_ensembl.csv.gz", index_col=0)
    counts.index = counts.index.astype(str).str.split(".").str[0]
    hgnc = pd.read_csv(ROOT / "06_independent_validation_prep" / "hgnc_complete_set.txt", sep="\t", low_memory=False)
    hgnc = hgnc.dropna(subset=["ensembl_gene_id", "symbol"]).drop_duplicates("ensembl_gene_id")
    mapping = hgnc.set_index("ensembl_gene_id")["symbol"]
    expression = log_cpm(aggregate_to_symbols(counts, mapping, "sum"))

    metadata["matrix_available"] = metadata["sample_id"].isin(expression.columns)
    metadata["usable_for_validation"] = (
        metadata["matrix_available"]
        & metadata["pretreatment_verified"]
        & metadata["response_standardized"].isin(["complete_response", "incomplete_response"])
    )
    metadata.to_csv(out / "GSE209746_metadata_prepared.tsv", sep="\t", index=False)

    scores, representation = score_programs(expression, programs, "GSE209746")
    scores = scores.merge(
        metadata[["sample_id", "geo_accession", "response_raw", "response_standardized", "msi_status", "usable_for_validation"]],
        on="sample_id",
        how="left",
    )
    scores = scores[scores["usable_for_validation"].fillna(False)].copy()
    results_all = validate(scores, "response_standardized", "incomplete_response", "complete_response")
    results_mss = validate(
        scores[scores["msi_status"] == "MSS"],
        "response_standardized",
        "incomplete_response",
        "complete_response",
        cohort_subset="MSS_only",
    )
    results = pd.concat([results_all, results_mss], ignore_index=True)
    representation.to_csv(out / "GSE209746_program_gene_representation.tsv", sep="\t", index=False)
    scores.to_csv(out / "GSE209746_locked_program_scores.tsv", sep="\t", index=False)
    results.to_csv(out / "GSE209746_validation_results.tsv", sep="\t", index=False)

    write_readiness(
        out / "GSE209746_matrix_readiness_summary.tsv",
        [
            ("disease", "locally advanced rectal cancer"),
            ("treatment", "multimodal neoadjuvant therapy with radiation and chemotherapy; regimens heterogeneous"),
            ("assay", "pretreatment bulk RNA-seq"),
            ("matrix_dimensions_ensembl_by_sample", f"{counts.shape[0]} x {counts.shape[1]}"),
            ("gene_symbols_after_mapping", expression.shape[0]),
            ("metadata_samples", len(metadata)),
            ("evaluable_complete_response", (metadata["response_standardized"] == "complete_response").sum()),
            ("evaluable_incomplete_response", (metadata["response_standardized"] == "incomplete_response").sum()),
            ("unknown_response", (metadata["response_standardized"] == "unknown").sum()),
            ("validation_status", "READY; locked programs only; no refitting"),
        ],
    )
    (out / "GSE209746_label_verification_note.md").write_text(
        "# GSE209746 label verification\n\n"
        "- GEO records identify all 114 specimens as pretreatment endoscopic biopsies from rectal cancer.\n"
        "- Public labels contain 27 CR, 70 iCR, and 17 NA specimens. CR is treated as complete response; iCR is retained as incomplete response and is not relabeled NR.\n"
        "- The primary external contrast is iCR versus CR among 97 response-evaluable patients.\n"
        "- The source study includes heterogeneous multimodal neoadjuvant regimens combining radiation and chemotherapy; it is not a chemotherapy-only match to CNP0004138.\n"
        "- Five MSI and 104 MSS labels are available; five are NA. This cohort is not identical to the pMMR/MSS-only CNP0004138 discovery cohort.\n"
        "- The locked CNP0004138 programs were applied without gene selection, refitting, threshold tuning, or cohort-specific sign reversal.\n",
        encoding="utf-8",
    )


def prepare_gse87211(programs: pd.DataFrame):
    out = ROOT / "08_gse87211_validation"
    source = out / "source"
    metadata = parse_series_matrix_metadata(source / "GSE87211_series_matrix.txt.gz")
    matrix = read_series_matrix(source / "GSE87211_series_matrix.txt.gz")
    platform = read_platform(source / "GPL13497_platform_data.txt")
    platform = platform.loc[platform["GENE_SYMBOL"].fillna("") != ""].drop_duplicates("ID")
    mapping = platform.set_index("ID")["GENE_SYMBOL"].str.split("///").str[0].str.strip()
    expression = aggregate_to_symbols(matrix, mapping, "mean")

    numeric_cols = [
        "depth of invasion after rct",
        "lymph node metastasis after rct",
        "metastasis after rct",
        "depth of invasion before rct",
        "lymph node metastasis before rct",
        "metastasis before rct",
    ]
    for column in numeric_cols:
        metadata[column] = pd.to_numeric(metadata[column].replace("NA", np.nan), errors="coerce")
    metadata["sample_id"] = metadata["geo_accession"]
    metadata["pretreatment_verified"] = True
    metadata["matrix_available"] = metadata["geo_accession"].isin(expression.columns)
    metadata["response_endpoint"] = "not_evaluable"
    tumor = metadata["tissue"].eq("rectal tumor")
    complete_stage = metadata[["depth of invasion after rct", "lymph node metastasis after rct", "metastasis after rct"]].notna().all(axis=1)
    pcr = (
        metadata["depth of invasion after rct"].eq(0)
        & metadata["lymph node metastasis after rct"].eq(0)
        & metadata["metastasis after rct"].eq(0)
    )
    metadata.loc[tumor & complete_stage, "response_endpoint"] = "non_pCR"
    metadata.loc[tumor & complete_stage & pcr, "response_endpoint"] = "pCR"
    metadata["usable_for_validation"] = tumor & complete_stage & metadata["matrix_available"]
    metadata.to_csv(out / "GSE87211_metadata_prepared.tsv", sep="\t", index=False)

    scores, representation = score_programs(expression, programs, "GSE87211")
    scores = scores.merge(
        metadata[["sample_id", "response_endpoint", "tissue", "preoperative radiochemotherapy (rct)", "usable_for_validation"]],
        on="sample_id",
        how="left",
    )
    scores = scores[scores["usable_for_validation"].fillna(False)].copy()
    results_frames = [validate(scores, "response_endpoint", "non_pCR", "pCR")]
    for regimen in ["5-FU + RT", "5-FU + Oxaliplatin + RT"]:
        results_frames.append(
            validate(
                scores[scores["preoperative radiochemotherapy (rct)"] == regimen],
                "response_endpoint",
                "non_pCR",
                "pCR",
                cohort_subset=regimen,
            )
        )
    results = pd.concat(results_frames, ignore_index=True)
    representation.to_csv(out / "GSE87211_program_gene_representation.tsv", sep="\t", index=False)
    scores.to_csv(out / "GSE87211_locked_program_scores.tsv", sep="\t", index=False)
    results.to_csv(out / "GSE87211_validation_results.tsv", sep="\t", index=False)

    write_readiness(
        out / "GSE87211_matrix_readiness_summary.tsv",
        [
            ("disease", "locally advanced rectal cancer"),
            ("treatment", "preoperative fluoropyrimidine-based radiochemotherapy, with regimen heterogeneity"),
            ("assay", "pretreatment Agilent whole-genome expression microarray"),
            ("matrix_dimensions_probe_by_sample", f"{matrix.shape[0]} x {matrix.shape[1]}"),
            ("gene_symbols_after_mapping", expression.shape[0]),
            ("metadata_samples", len(metadata)),
            ("rectal_tumor_samples", tumor.sum()),
            ("strict_pCR_ypT0N0M0", ((metadata["response_endpoint"] == "pCR") & tumor).sum()),
            ("non_pCR_complete_ypTNM", ((metadata["response_endpoint"] == "non_pCR") & tumor).sum()),
            ("public_sample_level_TRG_percent_available", "NO"),
            ("validation_status", "CONDITIONALLY READY for strict pCR endpoint; locked programs only; no refitting"),
        ],
    )
    (out / "GSE87211_label_verification_note.md").write_text(
        "# GSE87211 label verification\n\n"
        "- GEO identifies 203 pretreatment rectal-tumor biopsies and 160 mucosal specimens. Only tumor biopsies enter response validation.\n"
        "- Public sample metadata do not contain sample-level AJCC TRG category or TRG percentage. Therefore, the 70 responder/41 non-responder grouping reported by a later reuse paper cannot be independently reconstructed from the deposited fields and is not used here.\n"
        "- A strict, auditable pCR endpoint is reconstructed as ypT0, ypN0, and ypM0 using the deposited post-radiochemotherapy pathological variables. All other tumors with complete ypTNM fields are non-pCR.\n"
        "- This strict endpoint differs from broad responder/non-responder groupings and must be described as a sensitivity-grade independent validation.\n"
        "- The locked CNP0004138 programs were applied without gene selection, refitting, threshold tuning, or sign reversal.\n",
        encoding="utf-8",
    )


def main():
    programs = load_programs()
    prepare_gse209746(programs)
    prepare_gse87211(programs)
    print("Prepared and validated GSE209746 and GSE87211.")


if __name__ == "__main__":
    main()
