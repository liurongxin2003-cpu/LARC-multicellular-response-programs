from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from phase3_config import INPUT, ROOT, SEED


OUT = ROOT / "12_draft_figures"
STATE_ORDER = [
    "Malignant_EMT_stress", "Myeloid_suppressive_remodeling", "CD8_dysfunctional_stress",
    "Endothelial_remodeling", "CAF_supporting_comparator",
]
STATE_LABEL = {
    "Malignant_EMT_stress": "Malignant\nlocked state",
    "Myeloid_suppressive_remodeling": "Myeloid\nlocked state",
    "CD8_dysfunctional_stress": "CD8-associated\nlocked state",
    "Endothelial_remodeling": "Endothelial\nlocked state",
    "CAF_supporting_comparator": "CAF\ncomparator",
}
STATE_COLOR = {
    "Malignant_EMT_stress": "#B55252",
    "Myeloid_suppressive_remodeling": "#C9873A",
    "CD8_dysfunctional_stress": "#3D78A6",
    "Endothelial_remodeling": "#4D927C",
    "CAF_supporting_comparator": "#8A6F9E",
}
RESPONSE_COLOR = {"CR": "#2F78A8", "PR": "#999999", "NR": "#C75B39"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})
sns.set_theme(style="white", context="paper", font_scale=1.0)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def top_candidates(n: int = 3) -> pd.DataFrame:
    rank = pd.read_csv(ROOT / "09_candidate_prioritization/final_candidate_regulator_ranking.tsv", sep="\t")
    moderate = rank[rank.final_confidence == "moderate"].sort_values("final_integrated_score", ascending=False)
    return moderate.groupby("target_state").head(n)


def refined_umap() -> None:
    adata = sc.read_h5ad(INPUT / "CNP0004138_QC_sample_annotated.h5ad")
    ann = pd.read_csv(ROOT / "03_refined_cell_state_annotation/refined_target_state_annotation.tsv", sep="\t")
    umap = adata.obsm["X_umap"]
    fig, axes = plt.subplots(1, 5, figsize=(12.0, 2.6))
    for ax, state in zip(axes, STATE_ORDER):
        frame = ann[ann.target_state == state]
        idx = frame.cell_index.to_numpy(dtype=int)
        values = frame.target_state_score.to_numpy(dtype=float)
        limit = np.quantile(np.abs(values), 0.98) or 1.0
        ax.scatter(umap[idx, 0], umap[idx, 1], c=values, cmap="vlag", vmin=-limit, vmax=limit, s=1.8, linewidths=0, rasterized=True)
        ax.set_title(STATE_LABEL[state], color=STATE_COLOR[state], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"n={len(frame):,}")
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.suptitle("Refined target-state scores in the 50,000-cell descriptive UMAP sample", x=0.06, ha="left", fontweight="bold")
    fig.text(0.99, 0.02, "Cell-level visualization is descriptive; response inference uses patient pseudobulk.", ha="right", color="#555555")
    save(fig, "DRAFT_refined_state_umap_descriptive")


def state_response_plot() -> None:
    data = pd.read_csv(ROOT / "03_refined_cell_state_annotation/refined_target_state_pseudobulk_scores.tsv", sep="\t")
    fig, axes = plt.subplots(1, 5, figsize=(12.0, 3.3), sharey=True)
    for ax, state in zip(axes, STATE_ORDER):
        frame = data[data.target_state == state]
        sns.boxplot(data=frame, x="response_standardized", y="target_state_score", order=["CR", "PR", "NR"], color="white", fliersize=0, linewidth=0.9, ax=ax)
        sns.stripplot(data=frame, x="response_standardized", y="target_state_score", order=["CR", "PR", "NR"], palette=RESPONSE_COLOR, hue="response_standardized", legend=False, size=4, jitter=0.14, ax=ax)
        ax.axhline(0, color="#BBBBBB", lw=0.7)
        ax.set_title(STATE_LABEL[state], color=STATE_COLOR[state], fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("State score" if ax is axes[0] else "")
    fig.suptitle("Canonical marker modules do not reproduce the NR-higher locked-program direction", x=0.06, y=0.99, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.72, bottom=0.18, wspace=0.20)
    fig.text(0.99, 0.02, "Each point is one PRET patient-celltype pseudobulk (6 CR, 14 PR, 5 NR).", ha="right", color="#555555")
    save(fig, "DRAFT_refined_state_score_by_response_descriptive")


def activity_figures() -> None:
    candidates = top_candidates(4)
    activity = pd.read_csv(ROOT / "04_regulator_activity/regulator_activity_by_target_state.tsv", sep="\t")
    chosen = set(zip(candidates.target_state, candidates.regulator))
    data = activity[[((s, r) in chosen) for s, r in zip(activity.target_state, activity.regulator)]].copy()
    data["label"] = data.target_state.map(lambda x: STATE_LABEL[x].replace("\n", " ")) + " | " + data.regulator
    matrix = data.pivot_table(index="label", columns="method", values="activity_difference_NR_vs_CR", aggfunc="median")
    fig, ax = plt.subplots(figsize=(6.8, max(3.2, 0.24 * len(matrix))))
    limit = max(0.5, np.nanquantile(np.abs(matrix.to_numpy()), 0.98))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "NR - CR inferred activity"}, ax=ax)
    ax.set_title("Regulator activity support across priors and inference methods", loc="left", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "DRAFT_regulator_activity_heatmap")

    agg = candidates.copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(agg))
    colors = [STATE_COLOR[s] for s in agg.target_state]
    size = 30 + 35 * agg.n_activity_methods
    scatter = ax.scatter(agg.median_resistance_correlation, y, s=size, c=agg.activity_evidence_score, cmap="YlOrRd", vmin=0.5, vmax=1, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_yticks(y, [f"{r} | {STATE_LABEL[s].replace(chr(10), ' ')}" for r, s in zip(agg.regulator, agg.target_state)])
    ax.invert_yaxis(); ax.set_xlabel("Correlation with locked resistance score")
    ax.set_title("Top multi-method regulator-activity nominations", loc="left", fontweight="bold")
    fig.colorbar(scatter, ax=ax, label="Activity evidence score", shrink=0.75)
    save(fig, "DRAFT_top_regulators_by_state_dotplot")


def grn_figures() -> None:
    candidates = top_candidates(2)
    keys = set(zip(candidates.target_state, candidates.regulator))
    edges = pd.read_csv(ROOT / "05_grn_inference/GRN_edges_by_target_state.tsv", sep="\t")
    edges = edges[[((s, r) in keys) for s, r in zip(edges.target_state, edges.regulator)]]
    edges = edges[(edges.target_locked_direction == "NR_high") & (edges.method_support_count >= 2)]
    selected = edges.sort_values(["method_support_count", "genie3_style_extra_trees_importance"], ascending=False).groupby(["target_state", "regulator"]).head(6)
    graph = nx.DiGraph()
    node_colors = []
    for row in selected.itertuples():
        reg_node = f"{row.target_state}|{row.regulator}"
        target_node = f"{row.target_state}|{row.target_gene}"
        graph.add_node(reg_node, kind="regulator", state=row.target_state, label=row.regulator)
        graph.add_node(target_node, kind="target", state=row.target_state, label=row.target_gene)
        graph.add_edge(reg_node, target_node, weight=row.method_support_count)
    states = list(dict.fromkeys(candidates.target_state.tolist()))
    pos = {}
    for i, state in enumerate(states):
        regs = [n for n, d in graph.nodes(data=True) if d["state"] == state and d["kind"] == "regulator"]
        targets = [n for n, d in graph.nodes(data=True) if d["state"] == state and d["kind"] == "target"]
        for j, node in enumerate(regs): pos[node] = (0, i * 3 + j * 0.6)
        for j, node in enumerate(targets): pos[node] = (2.4, i * 3 + (j - len(targets) / 2) * 0.22)
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    for node, attrs in graph.nodes(data=True):
        node_colors.append(STATE_COLOR[attrs["state"]] if attrs["kind"] == "regulator" else "#D9D9D9")
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.35, width=[0.5 * graph[u][v]["weight"] for u, v in graph.edges()], arrowsize=7, edge_color="#777777")
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=[460 if graph.nodes[n]["kind"] == "regulator" else 190 for n in graph.nodes()], edgecolors="white", linewidths=0.6)
    nx.draw_networkx_labels(graph, pos, labels={n: d["label"] for n, d in graph.nodes(data=True)}, font_size=6, ax=ax)
    ax.set_axis_off(); ax.set_title("Multi-method GRN links nominated regulators to locked NR-high genes", loc="left", fontweight="bold")
    save(fig, "DRAFT_GRN_target_overlap_network")

    conc = pd.read_csv(ROOT / "05_grn_inference/GRN_method_concordance_summary.tsv", sep="\t")
    conc = conc[[((s, r) in keys) for s, r in zip(conc.target_state, conc.regulator)]].copy()
    conc["label"] = conc.target_state.map(lambda x: STATE_LABEL[x].replace("\n", " ")) + " | " + conc.regulator
    matrix = conc.set_index("label")[["prior_supported_NR_edges", "coexpression_supported_NR_edges", "tree_supported_NR_edges", "two_or_more_method_NR_edges"]]
    fig, ax = plt.subplots(figsize=(6.8, max(3.2, 0.28 * len(matrix))))
    sns.heatmap(matrix, cmap="Blues", annot=True, fmt=".0f", linewidths=0.4, cbar_kws={"label": "Edge count"}, ax=ax)
    ax.set_title("GRN support is strongest when prior, correlation, and tree evidence agree", loc="left", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "DRAFT_GRN_method_concordance_heatmap")


def perturbation_figures() -> None:
    candidates = top_candidates(3)
    keys = set(zip(candidates.target_state, candidates.regulator))
    data = pd.read_csv(ROOT / "06_virtual_perturbation/virtual_perturbation_results_by_state.tsv", sep="\t")
    data = data[[((s, r) in keys) for s, r in zip(data.target_state, data.regulator)]].copy()
    data["label"] = data.target_state.map(lambda x: STATE_LABEL[x].replace("\n", " ")) + " | " + data.regulator
    matrix = data.pivot(index="label", columns="perturbation_method", values="mean_delta_resistance_NR")
    fig, ax = plt.subplots(figsize=(6.5, max(3.2, 0.26 * len(matrix))))
    limit = max(0.02, np.nanmax(np.abs(matrix.to_numpy())))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".3f", linewidths=0.4, cbar_kws={"label": "Delta resistance score in NR"}, ax=ax)
    ax.set_title("Network stress tests suggest candidate-specific score reductions", loc="left", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "DRAFT_virtual_perturbation_effect_heatmap")

    rank = pd.read_csv(ROOT / "06_virtual_perturbation/virtual_perturbation_ranked_regulators.tsv", sep="\t")
    rank = rank[[((s, r) in keys) for s, r in zip(rank.target_state, rank.regulator)]].sort_values("median_delta_resistance_NR")
    fig, ax = plt.subplots(figsize=(7.0, max(3.4, 0.25 * len(rank))))
    labels = [f"{r} | {STATE_LABEL[s].replace(chr(10), ' ')}" for r, s in zip(rank.regulator, rank.target_state)]
    ax.barh(np.arange(len(rank)), -rank.median_delta_resistance_NR, color=[STATE_COLOR[s] for s in rank.target_state])
    ax.set_yticks(np.arange(len(rank)), labels); ax.invert_yaxis()
    ax.axvline(0, color="#888888", lw=0.8); ax.set_xlabel("Predicted NR resistance reduction")
    ax.set_title("Leading virtual-perturbation candidates", loc="left", fontweight="bold")
    save(fig, "DRAFT_virtual_perturbation_top_candidates")


def lr_figures() -> None:
    lr = pd.read_csv(ROOT / "07_ligand_receptor_niche/spatially_informed_LR_priority_table.tsv", sep="\t")
    top = lr.sort_values("LR_priority_score", ascending=False).groupby(["sender_cell_type", "receiver_cell_type"]).head(1)
    node_state = {"Malignant_epithelial": "Malignant_EMT_stress", "Myeloid_macrophage": "Myeloid_suppressive_remodeling", "CD8_T": "CD8_dysfunctional_stress", "Endothelial": "Endothelial_remodeling", "CAF_Fibroblast": "CAF_supporting_comparator"}
    top = top.sort_values(["sender_cell_type", "receiver_cell_type"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for index, row in top.iterrows():
        y = len(top) - index
        sender = row.sender_cell_type
        receiver = row.receiver_cell_type
        ax.add_patch(plt.Rectangle((0.2, y - 0.30), 2.0, 0.60, facecolor=STATE_COLOR[node_state[sender]], edgecolor="white"))
        ax.add_patch(plt.Rectangle((5.8, y - 0.30), 2.0, 0.60, facecolor=STATE_COLOR[node_state[receiver]], edgecolor="white"))
        ax.text(1.2, y, sender.replace("_", " "), ha="center", va="center", color="white", fontweight="bold", fontsize=7)
        ax.text(6.8, y, receiver.replace("_", " "), ha="center", va="center", color="white", fontweight="bold", fontsize=7)
        ax.annotate("", xy=(5.65, y), xytext=(2.35, y), arrowprops=dict(arrowstyle="-|>", lw=1 + 2 * row.LR_priority_score, color="#777777", shrinkA=0, shrinkB=0))
        ax.text(4.0, y + 0.12, f"{row.ligand}-{row.receptor}", ha="center", va="bottom", fontsize=7)
        ax.text(4.0, y - 0.14, f"priority {row.LR_priority_score:.2f}", ha="center", va="top", fontsize=6, color="#666666")
    ax.set_xlim(0, 8); ax.set_ylim(0.3, len(top) + 0.8); ax.set_axis_off()
    ax.set_title("Patient-paired, spatially informed ligand-receptor niche hypotheses", loc="left", fontweight="bold")
    save(fig, "DRAFT_ligand_receptor_network")

    top2 = lr.sort_values("LR_priority_score", ascending=False).groupby(["sender_cell_type", "receiver_cell_type"]).head(2).copy()
    top2["axis"] = top2.ligand + "-" + top2.receptor
    top2["pair"] = top2.sender_cell_type.str.replace("_", " ") + " -> " + top2.receiver_cell_type.str.replace("_", " ")
    matrix = top2.pivot_table(index="axis", columns="pair", values="LR_priority_score", aggfunc="max")
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.22 * len(matrix))))
    sns.heatmap(matrix, cmap="YlOrRd", vmin=0.5, vmax=1, linewidths=0.4, cbar_kws={"label": "LR priority score"}, ax=ax)
    ax.set_title("Leading non-CAF and CAF-context niche axes", loc="left", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "DRAFT_niche_axis_heatmap")


def spatial_figures() -> None:
    chosen = {
        "Malignant_EMT_stress": "IRF3",
        "Myeloid_suppressive_remodeling": "E2F6",
        "CD8_dysfunctional_stress": "ZBTB7A",
        "Endothelial_remodeling": "RELB",
    }
    chunks = []
    path = ROOT / "08_spatial_regulator_projection/spatial_regulator_projection_scores.tsv"
    for chunk in pd.read_csv(path, sep="\t", chunksize=250000):
        keep = np.array([chosen.get(state) == regulator for state, regulator in zip(chunk.target_state, chunk.regulator)])
        if keep.any(): chunks.append(chunk.loc[keep])
    data = pd.concat(chunks, ignore_index=True)
    fig, axes = plt.subplots(4, 4, figsize=(10.0, 8.2))
    sections = ["ST-CR1", "ST-CR2", "ST-NR1", "ST-NR2"]
    for r, (state, regulator) in enumerate(chosen.items()):
        state_data = data[(data.target_state == state) & (data.regulator == regulator)]
        limit = np.quantile(np.abs(state_data.regulator_expression_z_within_section), 0.98)
        for c, section in enumerate(sections):
            ax = axes[r, c]
            frame = state_data[state_data.section_id == section]
            if len(frame) > 20000: frame = frame.sample(20000, random_state=SEED)
            bins = frame.bin_id.str.split(":", n=1).str[1].str.split("_", expand=True).astype(float)
            ax.scatter(bins[0], -bins[1], c=frame.regulator_expression_z_within_section, cmap="vlag", vmin=-limit, vmax=limit, s=1.2, linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for spine in ax.spines.values(): spine.set_visible(False)
            if r == 0: ax.set_title(section, color=RESPONSE_COLOR["CR" if "CR" in section else "NR"], fontweight="bold")
            if c == 0:
                short_state = STATE_LABEL[state].replace("\nlocked state", "").replace("\n", " ")
                ax.set_ylabel(f"{regulator}\n{short_state}", color=STATE_COLOR[state], fontweight="bold", rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("Candidate-regulator expression proxies in Phase 2 Stereo-seq sections", x=0.05, ha="left", fontweight="bold")
    fig.subplots_adjust(left=0.13, top=0.91, bottom=0.07, hspace=0.18, wspace=0.10)
    fig.text(0.99, 0.01, "Within-section expression z-score; descriptive bins; shared scale within each regulator row.", ha="right", color="#555555")
    save(fig, "DRAFT_candidate_regulator_spatial_maps")

    summary = pd.read_csv(ROOT / "08_spatial_regulator_projection/regulator_section_level_summary.tsv", sep="\t")
    summary = pd.concat([summary[(summary.target_state == state) & (summary.regulator == regulator)] for state, regulator in chosen.items()])
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    x = np.arange(len(summary)); width = 0.36
    ax.bar(x - width / 2, summary.section_effect_NR_minus_CR, width, label="Section expression NR-CR", color=[STATE_COLOR[s] for s in summary.target_state])
    ax.bar(x + width / 2, summary.mean_within_section_resistance_rho, width, label="Mean bin-level rho", color="#BFC5CA")
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(x, summary.regulator); ax.set_ylabel("Effect / correlation")
    ax.legend(loc="upper right"); ax.set_title("Spatial projection requires both section direction and descriptive bin correlation", loc="left", fontweight="bold")
    save(fig, "DRAFT_regulator_resistance_spatial_correlation")


def ranking_stability_figures() -> None:
    candidates = top_candidates(3)
    fig, ax = plt.subplots(figsize=(7.2, max(3.5, 0.26 * len(candidates))))
    ordered = candidates.sort_values("final_integrated_score")
    labels = [f"{r} | {STATE_LABEL[s].replace(chr(10), ' ')}" for r, s in zip(ordered.regulator, ordered.target_state)]
    ax.barh(np.arange(len(ordered)), ordered.final_integrated_score, color=[STATE_COLOR[s] for s in ordered.target_state])
    ax.set_yticks(np.arange(len(ordered)), labels); ax.set_xlim(0, 1)
    ax.set_xlabel("Integrated nomination score"); ax.set_title("Final regulator nominations remain moderate because canonical-state direction failed", loc="left", fontweight="bold")
    for i, row in enumerate(ordered.itertuples()): ax.text(row.final_integrated_score + 0.01, i, row.final_confidence, va="center", fontsize=6)
    save(fig, "DRAFT_final_candidate_regulator_ranking")

    stability = pd.read_csv(ROOT / "10_robustness_sensitivity/regulator_robustness_summary.tsv", sep="\t")
    keys = set(zip(candidates.target_state, candidates.regulator))
    stability = stability[[((s, r) in keys) for s, r in zip(stability.target_state, stability.regulator)]].copy()
    stability["label"] = stability.target_state.map(lambda x: STATE_LABEL[x].replace("\n", " ")) + " | " + stability.regulator
    matrix = stability.set_index("label")[["leave_one_positive_fraction", "response_label_positive_fraction", "positive_subsample_fraction", "remaining_edge_fraction"]]
    matrix.columns = ["Leave-one patient", "Response labels", "Cell subsampling", "CAF-axis exclusion"]
    fig, ax = plt.subplots(figsize=(6.8, max(3.5, 0.26 * len(matrix))))
    sns.heatmap(matrix, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "Stability"}, ax=ax)
    ax.set_title("Regulator nomination stability across prespecified sensitivity analyses", loc="left", fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    save(fig, "DRAFT_regulator_stability_heatmap")


def integrated_schematic() -> None:
    evidence = pd.read_csv(ROOT / "11_cross_phase_integration/Phase1_Phase2_Phase3_integrated_evidence_summary.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    boxes = [
        (0.3, 3.4, 2.2, 1.0, "Phase 1", "PRET locked programs\nCR < PR < NR", "#D9E8F2"),
        (3.0, 3.4, 2.2, 1.0, "Phase 2", "Post-NAC spatial\nNR > CR in 6/6 programs", "#F3DEC9"),
        (5.7, 3.4, 3.8, 1.0, "Phase 3", "Moderate regulators + LR niches\n0 high-confidence regulators", "#E1DFEA"),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#666666", linewidth=0.8))
        ax.text(x + 0.12, y + h - 0.22, title, fontweight="bold", va="top")
        ax.text(x + 0.12, y + 0.18, body, va="bottom", fontsize=8)
    ax.annotate("", xy=(3.0, 3.9), xytext=(2.5, 3.9), arrowprops=dict(arrowstyle="-|>", color="#555555"))
    ax.annotate("", xy=(5.7, 3.9), xytext=(5.2, 3.9), arrowprops=dict(arrowstyle="-|>", color="#555555"))
    x_positions = [0.5, 2.4, 4.3, 6.2, 8.1]
    for x, row in zip(x_positions, evidence.itertuples()):
        state = row.target_state
        ax.add_patch(plt.Rectangle((x, 1.3), 1.5, 1.35, facecolor="white", edgecolor=STATE_COLOR[state], linewidth=1.5))
        ax.text(x + 0.75, 2.43, STATE_LABEL[state], ha="center", va="top", color=STATE_COLOR[state], fontweight="bold", fontsize=7)
        ax.text(x + 0.75, 1.84, f"{row.top_nominated_regulator}\n{row.top_regulator_confidence}", ha="center", va="center", fontsize=8)
        ax.text(x + 0.75, 1.42, f"marker NR effect {row.phase3_canonical_marker_state_NR_minus_CR:.2f}", ha="center", fontsize=6, color="#8B2E2E")
    ax.add_patch(plt.Rectangle((0.5, 0.25), 9.0, 0.55, facecolor="#F4F4F4", edgecolor="#888888", linewidth=0.8))
    ax.text(5.0, 0.53, "Defensible manuscript claim: response-associated spatial programs with candidate regulators and niche axes\nExperimental validation is required", ha="center", va="center", fontweight="bold", fontsize=8)
    ax.set_title("Integrated model: strong program architecture, hypothesis-generating regulation", loc="left", fontweight="bold")
    save(fig, "DRAFT_integrated_mechanism_schematic")


def main() -> None:
    refined_umap()
    state_response_plot()
    activity_figures()
    grn_figures()
    perturbation_figures()
    lr_figures()
    spatial_figures()
    ranking_stability_figures()
    integrated_schematic()
    print("Created 15 Phase 3 draft figures with Python.")


if __name__ == "__main__":
    main()
