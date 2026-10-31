from __future__ import annotations

import json
from importlib.metadata import version

import decoupler as dc
import numpy as np
import pandas as pd
from scipy import stats

from phase3_config import INPUT, RESPONSE_ORDER, ROOT, SEED, TARGET_STATES, TF_CANDIDATES


PROGRAM_MAP = {
    "Malignant_EMT_stress": "Malignant_epithelial_specific",
    "Myeloid_suppressive_remodeling": "Myeloid_macrophage_specific",
    "CD8_dysfunctional_stress": "CD8_T_specific",
    "Endothelial_remodeling": "Endothelial_specific",
    "CAF_supporting_comparator": "CAF_Fibroblast_specific",
}


def hedges_g(cr: np.ndarray, nr: np.ndarray) -> float:
    pooled = np.sqrt(((len(cr) - 1) * np.var(cr, ddof=1) + (len(nr) - 1) * np.var(nr, ddof=1)) / (len(cr) + len(nr) - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    return ((np.mean(nr) - np.mean(cr)) / pooled) * (1 - 3 / (4 * (len(cr) + len(nr)) - 9))


def bootstrap_ci(cr: np.ndarray, nr: np.ndarray, rng: np.random.Generator, n: int = 1000) -> tuple[float, float]:
    effects = [np.mean(rng.choice(nr, len(nr), replace=True)) - np.mean(rng.choice(cr, len(cr), replace=True)) for _ in range(n)]
    return tuple(np.quantile(effects, [0.025, 0.975]))


def leave_one_sign(values: np.ndarray, responses: np.ndarray, patient_ids: np.ndarray) -> float:
    primary = np.isin(responses, ["CR", "NR"])
    signs = []
    for patient in np.unique(patient_ids[primary]):
        keep = primary & (patient_ids != patient)
        cr = values[keep & (responses == "CR")]
        nr = values[keep & (responses == "NR")]
        if len(cr) and len(nr):
            signs.append(np.mean(nr) > np.mean(cr))
    return float(np.mean(signs)) if signs else np.nan


def main() -> None:
    out = ROOT / "04_regulator_activity"
    rng = np.random.default_rng(SEED)
    pb = pd.read_parquet(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet")
    pb.index = pb.index.astype(str).str.upper()
    meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t").set_index("pseudobulk_id")
    locked = pd.read_csv(INPUT / "CNP0004138_pseudobulk_locked_program_scores.tsv", sep="\t")

    dorothea = dc.op.dorothea(levels=["A", "B", "C"])
    collectri_full = dc.op.collectri()
    prespecified_union = set().union(*[set(values) for values in TF_CANDIDATES.values()])
    collectri = collectri_full[collectri_full["source"].isin(prespecified_union)].copy()
    dorothea.to_csv(out / "DoRothEA_ABC_resource_snapshot.tsv", sep="\t", index=False)
    collectri.to_csv(out / "CollecTRI_prespecified_TF_resource_snapshot.tsv", sep="\t", index=False)

    method_specs = [
        ("DoRothEA_ULM", dc.mt.ulm, dorothea),
        ("DoRothEA_MLM", dc.mt.mlm, dorothea),
        ("CollecTRI_ULM", dc.mt.ulm, collectri),
    ]
    rows = []
    activity_frames = []
    for target_state, definition in TARGET_STATES.items():
        cell_type = definition["cell_type"]
        unit_ids = meta.index[(meta.cell_type == cell_type) & meta.response_standardized.isin(RESPONSE_ORDER)].tolist()
        counts = pb[unit_ids].T
        library = counts.sum(axis=1)
        logcpm = np.log1p(counts.div(np.maximum(library, 1), axis=0) * 1e6)
        program_name = PROGRAM_MAP[target_state]
        resistance = locked[(locked.program_name == program_name) & locked.pseudobulk_id.isin(unit_ids)].set_index("pseudobulk_id")["resistance_score"].reindex(unit_ids)
        responses = meta.loc[unit_ids, "response_standardized"].to_numpy()
        patients = meta.loc[unit_ids, "patient_id"].to_numpy()
        trend = pd.Series(responses).map({"CR": 0, "PR": 1, "NR": 2}).to_numpy()

        for method_name, method, network in method_specs:
            estimates, pvalues = method(logcpm, network, tmin=5, verbose=False)
            estimates.index = unit_ids
            pvalues.index = unit_ids
            activity_frames.append(estimates.assign(target_state=target_state, method=method_name, pseudobulk_id=unit_ids))
            target_counts = network.groupby("source").target.nunique()
            for regulator in estimates.columns:
                values = estimates[regulator].to_numpy(dtype=float)
                cr = values[responses == "CR"]
                nr = values[responses == "NR"]
                ci_low, ci_high = bootstrap_ci(cr, nr, rng)
                rho_gradient, p_gradient = stats.spearmanr(trend, values)
                rho_resistance, p_resistance = stats.spearmanr(resistance.to_numpy(), values)
                rows.append({
                    "regulator": regulator,
                    "target_state": target_state,
                    "source_cell_type": cell_type,
                    "activity_difference_NR_vs_CR": np.mean(nr) - np.mean(cr),
                    "gradient_CR_PR_NR": rho_gradient,
                    "gradient_p": p_gradient,
                    "association_with_resistance_score": rho_resistance,
                    "resistance_association_p": p_resistance,
                    "response_group_effect_size": hedges_g(cr, nr),
                    "p_value_CR_vs_NR": stats.mannwhitneyu(cr, nr, alternative="two-sided").pvalue,
                    "bootstrap_CI_low": ci_low,
                    "bootstrap_CI_high": ci_high,
                    "method": method_name,
                    "stability_score": leave_one_sign(values, responses, patients),
                    "n_regulon_targets": int(target_counts.get(regulator, 0)),
                    "prespecified_candidate": regulator in TF_CANDIDATES[target_state],
                    "notes": "activity is prior-based inference from patient-celltype pseudobulk; not TF expression or causal proof",
                })

    activity = pd.DataFrame(rows)
    activity.to_csv(out / "regulator_activity_by_target_state.tsv", sep="\t", index=False)
    long_values = []
    for frame in activity_frames:
        id_columns = ["target_state", "method", "pseudobulk_id"]
        long_values.append(frame.melt(id_vars=id_columns, var_name="regulator", value_name="activity_score"))
    pd.concat(long_values, ignore_index=True).to_parquet(out / "regulator_activity_pseudobulk_scores.parquet", index=False)

    aggregate = activity.groupby(["target_state", "source_cell_type", "regulator"], as_index=False).agg(
        n_activity_methods=("method", "nunique"),
        median_activity_NR_minus_CR=("activity_difference_NR_vs_CR", "median"),
        median_gradient_rho=("gradient_CR_PR_NR", "median"),
        median_resistance_correlation=("association_with_resistance_score", "median"),
        positive_method_fraction=("activity_difference_NR_vs_CR", lambda x: float(np.mean(np.asarray(x) > 0))),
        mean_stability_score=("stability_score", "mean"),
        minimum_p_value=("p_value_CR_vs_NR", "min"),
        prespecified_candidate=("prespecified_candidate", "max"),
    )
    aggregate["activity_evidence_score"] = (
        aggregate.groupby("target_state")["median_activity_NR_minus_CR"].rank(pct=True)
        + aggregate.groupby("target_state")["median_resistance_correlation"].rank(pct=True)
        + aggregate["positive_method_fraction"]
        + aggregate["mean_stability_score"]
    ) / 4
    top = aggregate.sort_values(["target_state", "activity_evidence_score"], ascending=[True, False]).groupby("target_state").head(30)
    top.to_csv(out / "top_regulators_by_state.tsv", sep="\t", index=False)

    method_note = f"""# Regulator activity method summary

- Python packages: decoupler {version('decoupler')}; LIANA {version('liana')}.
- Priors: DoRothEA confidence A-C and CollecTRI snapshots saved with this analysis.
- CollecTRI was restricted before scoring to the union of prespecified state-relevant TFs because the full 42,990-edge network caused a native process termination in this runtime; DoRothEA remained genome-wide.
- Methods: DoRothEA ULM, DoRothEA MLM, and CollecTRI ULM.
- VIPER was attempted but caused a native process termination in the current Windows/Python 3.12 runtime; no VIPER values are reported or imputed.
- Input: PRET patient-celltype pseudobulk logCPM for the fixed target cell type; n=6 CR, 14 PR, 5 NR per target state.
- Response inference: patient-celltype pseudobulk. No cell-level p values are used.
- Stability: fraction of leave-one-primary-patient scenarios retaining an NR-higher activity direction.
- Ranking requires concordance across methods and association with the unchanged Phase 1 locked resistance score.
- Inferred activity is not TF expression, binding validation, knockout validation, or causal proof.
"""
    (out / "regulator_activity_method_summary.md").write_text(method_note, encoding="utf-8")
    (ROOT / "00_logs/regulator_resource_versions.json").write_text(json.dumps({
        "decoupler": version("decoupler"),
        "liana": version("liana"),
        "DoRothEA_edges": len(dorothea),
        "CollecTRI_edges": len(collectri),
    }, indent=2), encoding="utf-8")
    print(f"Computed {len(activity):,} regulator-state-method summaries; retained {len(top)} top activity candidates.")


if __name__ == "__main__":
    main()
