from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")
RNG = np.random.default_rng(20260701)
PRIORITY_TYPES = [
    "All_cells",
    "Malignant_epithelial",
    "CAF_Fibroblast",
    "Myeloid_macrophage",
    "CD8_T",
    "Endothelial",
]


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def logcpm(counts: pd.DataFrame) -> pd.DataFrame:
    libraries = counts.sum(axis=0).replace(0, np.nan)
    return np.log2(counts.divide(libraries, axis=1) * 1e6 + 0.5)


def bootstrap_difference(
    values_a: np.ndarray, values_b: np.ndarray, iterations: int = 5000
) -> tuple[float, float]:
    if len(values_a) < 2 or len(values_b) < 2:
        return np.nan, np.nan
    diffs = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        diffs[idx] = (
            RNG.choice(values_a, len(values_a), replace=True).mean()
            - RNG.choice(values_b, len(values_b), replace=True).mean()
        )
    return tuple(np.quantile(diffs, [0.025, 0.975]))


def differential_table(
    counts: pd.DataFrame, meta: pd.DataFrame, cell_type: str
) -> pd.DataFrame:
    selected_meta = meta[
        meta["cell_type"].eq(cell_type)
        & meta["response_standardized"].isin(["CR", "NR"])
    ].copy()
    selected_counts = counts[selected_meta["pseudobulk_id"]]
    cpm = selected_counts.divide(selected_counts.sum(axis=0), axis=1) * 1e6
    keep = (cpm >= 1).sum(axis=1).ge(3) & selected_counts.sum(axis=1).ge(10)
    selected_counts = selected_counts.loc[keep]
    expr = logcpm(selected_counts)
    cr_ids = selected_meta.loc[selected_meta["response_standardized"].eq("CR"), "pseudobulk_id"]
    nr_ids = selected_meta.loc[selected_meta["response_standardized"].eq("NR"), "pseudobulk_id"]
    cr = expr[cr_ids].to_numpy()
    nr = expr[nr_ids].to_numpy()
    test = stats.ttest_ind(nr, cr, axis=1, equal_var=False, nan_policy="omit")
    effect = np.nanmean(nr, axis=1) - np.nanmean(cr, axis=1)
    pooled_sd = np.sqrt(
        ((nr.shape[1] - 1) * np.nanvar(nr, axis=1, ddof=1)
         + (cr.shape[1] - 1) * np.nanvar(cr, axis=1, ddof=1))
        / max(nr.shape[1] + cr.shape[1] - 2, 1)
    )
    hedges = effect / np.where(pooled_sd > 0, pooled_sd, np.nan)
    correction = 1 - 3 / max(4 * (nr.shape[1] + cr.shape[1]) - 9, 1)
    hedges *= correction
    result = pd.DataFrame(
        {
            "gene_symbol": expr.index,
            "cell_type": cell_type,
            "n_CR": len(cr_ids),
            "n_PR": int(
                meta[
                    meta["cell_type"].eq(cell_type)
                    & meta["response_standardized"].eq("PR")
                ]["patient_id"].nunique()
            ),
            "n_NR": len(nr_ids),
            "mean_logCPM_CR": np.nanmean(cr, axis=1),
            "mean_logCPM_NR": np.nanmean(nr, axis=1),
            "log2FC_NR_minus_CR": effect,
            "hedges_g_NR_minus_CR": hedges,
            "t_statistic": test.statistic,
            "p_value": test.pvalue,
        }
    )
    result["FDR_BH"] = bh_adjust(result["p_value"].fillna(1).to_numpy())
    result["direction"] = np.where(
        result["log2FC_NR_minus_CR"] > 0, "NR_high", "CR_high"
    )
    result["method"] = "patient-pseudobulk logCPM Welch t-test"
    return result.sort_values(["p_value", "log2FC_NR_minus_CR"], ascending=[True, False])


def choose_program(deg: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    selected_rows = []
    for direction, sign in [("NR_high", 1), ("CR_high", -1)]:
        candidates = deg[deg["direction"].eq(direction)].copy()
        candidates["signed_effect"] = sign * candidates["log2FC_NR_minus_CR"]
        candidates = candidates[candidates["signed_effect"] > 0]
        candidates = candidates.sort_values(
            ["p_value", "signed_effect"], ascending=[True, False]
        ).head(100)
        significant = int((candidates["FDR_BH"] < 0.05).sum())
        basis = (
            "FDR/effect-ranked"
            if significant >= 50
            else "limited-significance: nominal-P/effect-ranked"
        )
        candidates = candidates.assign(
            program_direction=direction,
            program_rank=np.arange(1, len(candidates) + 1),
            source_cell_type=cell_type,
            selection_basis=basis,
        )
        selected_rows.append(candidates)
    if not selected_rows:
        return pd.DataFrame()
    return pd.concat(selected_rows, ignore_index=True)


def score_program(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    program: pd.DataFrame,
    cell_type: str,
    program_name: str,
) -> tuple[pd.DataFrame, dict]:
    sample_meta = meta[meta["cell_type"].eq(cell_type)].copy()
    sample_counts = counts[sample_meta["pseudobulk_id"]]
    expr = logcpm(sample_counts)
    nr_genes = program.loc[
        program["program_direction"].eq("NR_high"), "gene_symbol"
    ].tolist()
    cr_genes = program.loc[
        program["program_direction"].eq("CR_high"), "gene_symbol"
    ].tolist()
    nr_represented = [gene for gene in nr_genes if gene in expr.index]
    cr_represented = [gene for gene in cr_genes if gene in expr.index]
    all_genes = nr_represented + cr_represented
    sub = expr.loc[all_genes]
    means = sub.mean(axis=1)
    stds = sub.std(axis=1, ddof=1).replace(0, np.nan)
    z = sub.sub(means, axis=0).div(stds, axis=0).fillna(0)
    nr_score = z.loc[nr_represented].mean(axis=0) if nr_represented else pd.Series(0, index=z.columns)
    cr_score = z.loc[cr_represented].mean(axis=0) if cr_represented else pd.Series(0, index=z.columns)
    output = sample_meta.set_index("pseudobulk_id").copy()
    output["program_name"] = program_name
    output["source_cell_type"] = cell_type
    output["NR_high_score"] = nr_score
    output["CR_high_score"] = cr_score
    output["resistance_score"] = output["NR_high_score"] - output["CR_high_score"]
    output["response_score"] = -output["resistance_score"]
    output["represented_NR_high_genes"] = len(nr_represented)
    output["represented_CR_high_genes"] = len(cr_represented)
    representation = {
        "program_name": program_name,
        "source_cell_type": cell_type,
        "requested_NR_high_genes": len(nr_genes),
        "represented_NR_high_genes": len(nr_represented),
        "requested_CR_high_genes": len(cr_genes),
        "represented_CR_high_genes": len(cr_represented),
        "representation_fraction": len(all_genes) / max(len(nr_genes) + len(cr_genes), 1),
    }
    return output.reset_index(), representation


def contrast_rows(scores: pd.DataFrame) -> list[dict]:
    rows = []
    for (program_name, cell_type), frame in scores.groupby(
        ["program_name", "source_cell_type"]
    ):
        for score_name in ["resistance_score", "response_score"]:
            groups = {
                group: frame.loc[frame["response_standardized"].eq(group), score_name].dropna().to_numpy()
                for group in ["CR", "PR", "NR"]
            }
            cr, nr = groups["CR"], groups["NR"]
            if len(cr) and len(nr):
                test = stats.mannwhitneyu(nr, cr, alternative="two-sided")
                effect = nr.mean() - cr.mean()
                low, high = bootstrap_difference(nr, cr)
            else:
                test = type("T", (), {"statistic": np.nan, "pvalue": np.nan})()
                effect, low, high = np.nan, np.nan, np.nan
            ordered = frame[frame["response_standardized"].isin(["CR", "PR", "NR"])].copy()
            ordered["response_order"] = ordered["response_standardized"].map(
                {"CR": 0, "PR": 1, "NR": 2}
            )
            rho, rho_p = stats.spearmanr(ordered["response_order"], ordered[score_name])
            rows.append(
                {
                    "program_name": program_name,
                    "source_cell_type": cell_type,
                    "score": score_name,
                    "n_CR": len(cr),
                    "n_PR": len(groups["PR"]),
                    "n_NR": len(nr),
                    "mean_CR": np.mean(cr) if len(cr) else np.nan,
                    "mean_PR": np.mean(groups["PR"]) if len(groups["PR"]) else np.nan,
                    "mean_NR": np.mean(nr) if len(nr) else np.nan,
                    "effect_NR_minus_CR": effect,
                    "bootstrap_CI_low": low,
                    "bootstrap_CI_high": high,
                    "mannwhitney_U": test.statistic,
                    "p_value_CR_vs_NR": test.pvalue,
                    "spearman_rho_CR_PR_NR": rho,
                    "spearman_p": rho_p,
                    "statistical_unit": "patient-celltype pseudobulk",
                }
            )
    return rows


def main() -> None:
    started = time.time()
    counts = pd.read_parquet(
        ROOT / "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet"
    )
    meta = pd.read_csv(
        ROOT / "04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv",
        sep="\t",
    )
    counts.index = counts.index.astype(str)

    full_tables = []
    programs = []
    support_rows = []
    for cell_type in PRIORITY_TYPES:
        ct_meta = meta[meta["cell_type"].eq(cell_type)]
        n_cr = ct_meta.loc[ct_meta["response_standardized"].eq("CR"), "patient_id"].nunique()
        n_pr = ct_meta.loc[ct_meta["response_standardized"].eq("PR"), "patient_id"].nunique()
        n_nr = ct_meta.loc[ct_meta["response_standardized"].eq("NR"), "patient_id"].nunique()
        supported = n_cr >= 3 and n_nr >= 3
        support_rows.append(
            {"cell_type": cell_type, "n_CR": n_cr, "n_PR": n_pr, "n_NR": n_nr, "program_eligible": "yes" if supported else "no"}
        )
        if not supported:
            continue
        deg = differential_table(counts, meta, cell_type)
        full_tables.append(deg)
        program = choose_program(deg, cell_type)
        if not program.empty:
            programs.append(program)

    if not programs:
        (ROOT / "11_reports/STOP_primary_program_not_defined.md").write_text(
            "# STOP\n\nNo priority cell type had sufficient CR/NR pseudobulk support.",
            encoding="utf-8",
        )
        raise RuntimeError("No locked program could be defined")

    full_deg = pd.concat(full_tables, ignore_index=True)
    full_deg.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_PRET_CR_vs_NR_DEG_by_celltype.tsv",
        sep="\t", index=False,
    )
    locked = pd.concat(programs, ignore_index=True)
    locked.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv",
        sep="\t", index=False,
    )
    global_program = locked[locked["source_cell_type"].eq("All_cells")].copy()
    global_program.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_locked_response_resistance_program_global.tsv",
        sep="\t", index=False,
    )

    all_scores = []
    representation_rows = []
    for cell_type, program in locked.groupby("source_cell_type"):
        program_name = "global" if cell_type == "All_cells" else f"{cell_type}_specific"
        score_frame, representation = score_program(
            counts, meta, program, cell_type, program_name
        )
        all_scores.append(score_frame)
        representation_rows.append(representation)
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_pseudobulk_locked_program_scores.tsv",
        sep="\t", index=False,
    )
    contrasts = pd.DataFrame(contrast_rows(scores))
    contrasts.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_CR_PR_NR_program_score_contrasts.tsv",
        sep="\t", index=False,
    )
    pd.DataFrame(representation_rows).to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_program_gene_representation.tsv",
        sep="\t", index=False,
    )
    gradient = contrasts[contrasts["score"].eq("resistance_score")].copy()
    gradient["PR_intermediate_by_mean"] = (
        (gradient["mean_PR"] >= np.minimum(gradient["mean_CR"], gradient["mean_NR"]))
        & (gradient["mean_PR"] <= np.maximum(gradient["mean_CR"], gradient["mean_NR"]))
    )
    gradient["expected_CR_to_NR_direction"] = gradient["effect_NR_minus_CR"] > 0
    gradient.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_PR_gradient_sensitivity_summary.tsv",
        sep="\t", index=False,
    )

    # Paired treatment dynamics for every locked cell-type program.
    full_counts = pd.read_parquet(
        ROOT / "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_matrix.parquet"
    )
    full_meta = pd.read_csv(
        ROOT / "04_pseudobulk/CNP0004138_patient_timepoint_celltype_pseudobulk_metadata.tsv",
        sep="\t",
    )
    dynamics_rows = []
    for cell_type, program in locked.groupby("source_cell_type"):
        program_name = "global" if cell_type == "All_cells" else f"{cell_type}_specific"
        scored, _ = score_program(full_counts, full_meta, program, cell_type, program_name)
        for patient, patient_frame in scored.groupby("patient_id"):
            pre = patient_frame[patient_frame["timepoint"].eq("PRET")]
            if pre.empty:
                continue
            for post_time in ["POSTT", "POSTN"]:
                post = patient_frame[patient_frame["timepoint"].eq(post_time)]
                if post.empty:
                    continue
                dynamics_rows.append(
                    {
                        "patient_id": patient,
                        "response_standardized": pre.iloc[0]["response_standardized"],
                        "program_name": program_name,
                        "source_cell_type": cell_type,
                        "comparison": f"{post_time}_minus_PRET",
                        "PRET_resistance_score": pre.iloc[0]["resistance_score"],
                        "POST_resistance_score": post.iloc[0]["resistance_score"],
                        "paired_change": post.iloc[0]["resistance_score"] - pre.iloc[0]["resistance_score"],
                        "statistical_unit": "paired patient-celltype pseudobulk",
                    }
                )
    dynamics = pd.DataFrame(dynamics_rows)
    dynamics.to_csv(
        ROOT / "05_response_state_discovery/CNP0004138_pre_post_program_dynamics.tsv",
        sep="\t", index=False,
    )

    support = pd.DataFrame(support_rows)
    note = f"""# Response-program definition note

## Primary contrast

- PRET NR versus PRET CR only.
- PR was excluded from gene selection and used only after locking for ordered-gradient sensitivity.
- Statistical unit: patient-celltype pseudobulk.

## Differential method

- Library-size normalized log2(CPM + 0.5).
- Welch patient-level test, BH correction, and Hedges g effect size.
- Given 6 PRET-CR and 5 PRET-NR patients at most, each module contains the top 100 genes in each direction ranked by nominal P and effect size.
- Modules with fewer than 50 FDR-significant selected genes are explicitly labelled `limited-significance`.

## Supported compartments

{support.to_markdown(index=False)}

## Locked outputs

- Global program: {len(global_program)} genes.
- Cell-type programs: {len(locked)} rows across {locked['source_cell_type'].nunique()} supported compartments.
- Validation cohorts had no role in selection, ranking, or tuning.

Elapsed time: {(time.time() - started) / 60:.2f} minutes.
"""
    (ROOT / "05_response_state_discovery/response_program_definition_note.md").write_text(
        note, encoding="utf-8"
    )
    print(support.to_string(index=False))
    print(gradient[["program_name", "source_cell_type", "n_CR", "n_PR", "n_NR", "effect_NR_minus_CR", "spearman_rho_CR_PR_NR", "PR_intermediate_by_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
