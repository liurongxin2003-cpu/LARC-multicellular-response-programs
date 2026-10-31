from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor

from phase3_config import INPUT, ROOT, SEED, TARGET_STATES, TF_CANDIDATES


def main() -> None:
    out = ROOT / "05_grn_inference"
    activity = pd.read_csv(ROOT / "04_regulator_activity/regulator_activity_by_target_state.tsv", sep="\t")
    top = pd.read_csv(ROOT / "04_regulator_activity/top_regulators_by_state.tsv", sep="\t")
    dorothea = pd.read_csv(ROOT / "04_regulator_activity/DoRothEA_ABC_resource_snapshot.tsv", sep="\t")
    collectri = pd.read_csv(ROOT / "04_regulator_activity/CollecTRI_prespecified_TF_resource_snapshot.tsv", sep="\t")
    prior = pd.concat([
        dorothea.assign(prior_source="DoRothEA_ABC"),
        collectri.assign(prior_source="CollecTRI_prespecified"),
    ], ignore_index=True)
    prior["source"] = prior.source.astype(str).str.upper()
    prior["target"] = prior.target.astype(str).str.upper()

    pb = pd.read_parquet(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet")
    pb.index = pb.index.astype(str).str.upper()
    meta = pd.read_csv(INPUT / "CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv", sep="\t").set_index("pseudobulk_id")
    locked = pd.read_csv(INPUT / "CNP0004138_locked_programs_by_celltype.tsv", sep="\t")

    edge_rows = []
    overlap_rows = []
    concordance_rows = []
    selected_rows = []
    for target_state, definition in TARGET_STATES.items():
        cell_type = definition["cell_type"]
        state_top = top[(top.target_state == target_state) & (top.n_activity_methods >= 2)].head(18)
        prescribed = activity[(activity.target_state == target_state) & activity.prespecified_candidate].groupby("regulator").agg(
            n_methods=("method", "nunique"),
            median_effect=("activity_difference_NR_vs_CR", "median"),
            positive_fraction=("activity_difference_NR_vs_CR", lambda x: float(np.mean(np.asarray(x) > 0))),
        ).reset_index()
        prescribed = prescribed[(prescribed.n_methods >= 1) & (prescribed.positive_fraction >= 2 / 3)].sort_values("median_effect", ascending=False).head(10)
        regulators = list(dict.fromkeys(state_top.regulator.tolist() + prescribed.regulator.tolist()))
        regulators = [reg.upper() for reg in regulators if reg.upper() in pb.index]
        selected_rows.extend({"target_state": target_state, "regulator": reg, "selection": "activity top/multi-method or prespecified positive"} for reg in regulators)

        program = locked[locked.source_cell_type == cell_type]
        nr_genes = program.loc[program.program_direction == "NR_high", "gene_symbol"].astype(str).str.upper().head(100).tolist()
        cr_genes = program.loc[program.program_direction == "CR_high", "gene_symbol"].astype(str).str.upper().head(100).tolist()
        target_genes = [gene for gene in nr_genes + cr_genes if gene in pb.index]
        unit_ids = meta.index[(meta.cell_type == cell_type) & meta.response_standardized.isin(["CR", "PR", "NR"])].tolist()
        counts = pb[unit_ids]
        library = counts.sum(axis=0)
        expr = np.log1p(counts.div(np.maximum(library, 1), axis=1) * 1e6).T
        x = expr[regulators].to_numpy(dtype=float)

        prior_sub = prior[prior.source.isin(regulators) & prior.target.isin(target_genes)]
        prior_group = prior_sub.groupby(["source", "target"]).agg(
            prior_weight=("weight", "mean"),
            prior_sources=("prior_source", lambda x: ";".join(sorted(set(x)))),
            prior_source_count=("prior_source", "nunique"),
        ).reset_index()
        prior_lookup = prior_group.set_index(["source", "target"]).to_dict("index")

        rf_scores = {}
        for target in target_genes:
            y = expr[target].to_numpy(dtype=float)
            model = ExtraTreesRegressor(n_estimators=150, max_features="sqrt", min_samples_leaf=3, random_state=SEED, n_jobs=1)
            model.fit(x, y)
            for regulator, importance in zip(regulators, model.feature_importances_):
                rf_scores[(regulator, target)] = float(importance)

        state_edges = []
        for regulator in regulators:
            for target in target_genes:
                rho, p_value = stats.spearmanr(expr[regulator], expr[target])
                prior_info = prior_lookup.get((regulator, target), {})
                prior_support = bool(prior_info)
                rf_importance = rf_scores[(regulator, target)]
                rf_rank = sum(rf_scores[(other, target)] > rf_importance for other in regulators) + 1
                coexpression_support = bool(np.isfinite(rho) and abs(rho) >= 0.30)
                rf_support = rf_rank <= min(5, len(regulators))
                method_count = int(prior_support) + int(coexpression_support) + int(rf_support)
                if method_count == 0:
                    continue
                direction_consistent = bool((prior_info.get("prior_weight", np.sign(rho) if np.isfinite(rho) else 0) * rho) > 0) if np.isfinite(rho) else False
                row = {
                    "target_state": target_state,
                    "source_cell_type": cell_type,
                    "regulator": regulator,
                    "target_gene": target,
                    "target_locked_direction": "NR_high" if target in nr_genes else "CR_high",
                    "prior_support": prior_support,
                    "prior_weight": prior_info.get("prior_weight", np.nan),
                    "prior_sources": prior_info.get("prior_sources", ""),
                    "coexpression_spearman_rho": rho,
                    "coexpression_p": p_value,
                    "coexpression_support_abs_rho_ge_0_30": coexpression_support,
                    "genie3_style_extra_trees_importance": rf_importance,
                    "extra_trees_rank_for_target": rf_rank,
                    "extra_trees_top5_support": rf_support,
                    "method_support_count": method_count,
                    "direction_consistent_with_prior": direction_consistent,
                    "network_confidence": "high" if method_count == 3 and direction_consistent else "moderate" if method_count >= 2 else "low",
                    "inference_boundary": "GRN support is prior/coexpression/tree-based nomination, not causal direction proof",
                }
                state_edges.append(row)
                edge_rows.append(row)

        edges = pd.DataFrame(state_edges)
        for regulator in regulators:
            subset = edges[edges.regulator == regulator]
            nr_supported = subset[(subset.target_locked_direction == "NR_high") & (subset.method_support_count >= 2)].target_gene.nunique()
            cr_supported = subset[(subset.target_locked_direction == "CR_high") & (subset.method_support_count >= 2)].target_gene.nunique()
            prior_targets = set(prior.loc[prior.source == regulator, "target"])
            a = len(prior_targets.intersection(nr_genes))
            b = len(set(nr_genes) - prior_targets)
            c = len(prior_targets.intersection(cr_genes))
            d = len(set(cr_genes) - prior_targets)
            odds, fisher_p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            overlap_rows.append({
                "target_state": target_state,
                "regulator": regulator,
                "n_locked_NR_high_targets": len(nr_genes),
                "n_locked_CR_high_targets": len(cr_genes),
                "prior_NR_high_overlap": a,
                "prior_CR_high_overlap": c,
                "multi_method_NR_high_targets": nr_supported,
                "multi_method_CR_high_targets": cr_supported,
                "fisher_odds_NR_vs_CR_prior_overlap": odds,
                "fisher_p_one_sided": fisher_p,
            })
            reg_edges = subset[subset.target_locked_direction == "NR_high"]
            concordance_rows.append({
                "target_state": target_state,
                "regulator": regulator,
                "prior_supported_NR_edges": int(reg_edges.prior_support.sum()),
                "coexpression_supported_NR_edges": int(reg_edges.coexpression_support_abs_rho_ge_0_30.sum()),
                "tree_supported_NR_edges": int(reg_edges.extra_trees_top5_support.sum()),
                "two_or_more_method_NR_edges": int((reg_edges.method_support_count >= 2).sum()),
                "three_method_NR_edges": int((reg_edges.method_support_count == 3).sum()),
                "mean_method_support_NR_edges": reg_edges.method_support_count.mean(),
            })

    pd.DataFrame(edge_rows).to_csv(out / "GRN_edges_by_target_state.tsv", sep="\t", index=False)
    pd.DataFrame(overlap_rows).to_csv(out / "regulator_NR_high_target_overlap.tsv", sep="\t", index=False)
    pd.DataFrame(concordance_rows).to_csv(out / "GRN_method_concordance_summary.tsv", sep="\t", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "GRN_candidate_regulator_input.tsv", sep="\t", index=False)
    note = """# GRN inference method note

Three evidence types were combined within each fixed target cell type: DoRothEA/CollecTRI prior edges, patient-pseudobulk Spearman coexpression, and an ExtraTrees feature-importance approximation to GENIE3-style tree inference. The tree model used 150 estimators, one execution thread, and a minimum leaf size of three across 25 PRET patient-celltype units.

Edges are retained when at least one method supports them; confidence is high only with three-method and sign-consistent support, moderate with at least two methods, and otherwise low. Locked NR-high and CR-high genes are unchanged Phase 1 genes. This network nominates regulatory relationships and does not prove direction, binding, or causality.
"""
    (out / "GRN_method_note.md").write_text(note, encoding="utf-8")
    print(f"Generated {len(edge_rows):,} GRN edges across {len(TARGET_STATES)} target states.")


if __name__ == "__main__":
    main()
