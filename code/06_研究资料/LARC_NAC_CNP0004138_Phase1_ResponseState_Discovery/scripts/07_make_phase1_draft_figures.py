from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")
OUT = ROOT / "10_draft_figures"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
        "figure.dpi": 150,
    }
)
sns.set_style("ticks")

RESPONSE_COLORS = {"CR": "#0072B2", "PR": "#E69F00", "NR": "#D55E00", "unknown": "#8C8C8C"}
PROGRAM_LABELS = {
    "All_cells": "Global",
    "Malignant_epithelial": "Malignant",
    "CAF_Fibroblast": "CAF",
    "Myeloid_macrophage": "Myeloid",
    "CD8_T": "CD8 T",
    "Endothelial": "Endothelial",
}
PROGRAM_ORDER = ["All_cells", "Malignant_epithelial", "CAF_Fibroblast", "Myeloid_macrophage", "CD8_T", "Endothelial"]


def save(fig: plt.Figure, name: str):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def clean_axes(ax):
    sns.despine(ax=ax)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def plot_umap():
    data = pd.read_csv(ROOT / "03_celltype_annotation" / "CNP0004138_QC_sample_umap_coordinates.tsv", sep="\t")
    celltypes = sorted(data["broad_cell_type"].unique())
    palette = dict(zip(celltypes, sns.color_palette("tab20", n_colors=len(celltypes))))
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    for celltype in celltypes:
        frame = data[data["broad_cell_type"] == celltype]
        ax.scatter(frame["UMAP1"], frame["UMAP2"], s=1.2, alpha=0.5, linewidths=0, color=palette[celltype], label=celltype)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Broad marker-guided annotation (QC sample)", loc="left", fontweight="bold")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", markerscale=4, ncol=1)
    ax.text(0.01, -0.12, f"n={len(data):,} sampled cells; descriptive QC only", transform=ax.transAxes, color="#555555")
    clean_axes(ax)
    save(fig, "DRAFT_celltype_umap_qc")


def plot_composition():
    data = pd.read_csv(ROOT / "03_celltype_annotation" / "CNP0004138_celltype_composition.tsv", sep="\t")
    data = data[data["response_standardized"].isin(["CR", "PR", "NR"])]
    summary = data.groupby(["timepoint", "response_standardized", "broad_cell_type"], as_index=False)["cell_fraction"].mean()
    summary["group"] = summary["timepoint"] + "\n" + summary["response_standardized"]
    group_order = [g for g in ["PRET\nCR", "PRET\nPR", "PRET\nNR", "POSTT\nCR", "POSTT\nPR", "POSTT\nNR", "POSTN\nCR", "POSTN\nPR", "POSTN\nNR"] if g in summary["group"].unique()]
    pivot = summary.pivot(index="group", columns="broad_cell_type", values="cell_fraction").fillna(0).reindex(group_order)
    colors = sns.color_palette("tab20", n_colors=pivot.shape[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bottom = np.zeros(len(pivot))
    for color, column in zip(colors, pivot.columns):
        ax.bar(np.arange(len(pivot)), pivot[column], bottom=bottom, color=color, width=0.72, label=column, linewidth=0)
        bottom += pivot[column].to_numpy()
    ax.set_xticks(np.arange(len(pivot)), pivot.index)
    ax.set_ylabel("Mean cell fraction per patient")
    ax.set_ylim(0, 1)
    ax.set_title("Broad cell composition by response and timepoint", loc="left", fontweight="bold")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)
    ax.text(0.01, -0.18, "Composition is descriptive and not a cell-level response test.", transform=ax.transAxes, color="#555555")
    clean_axes(ax)
    save(fig, "DRAFT_celltype_composition_by_response_timepoint")


def load_discovery_scores():
    data = pd.read_csv(ROOT / "05_response_state_discovery" / "CNP0004138_pseudobulk_locked_program_scores.tsv", sep="\t")
    data = data[data["response_standardized"].isin(["CR", "PR", "NR"])]
    return data


def plot_gradient():
    data = load_discovery_scores()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3), sharey=False)
    for i, (source, ax) in enumerate(zip(PROGRAM_ORDER, axes.flat)):
        frame = data[data["source_cell_type"] == source]
        sns.pointplot(
            data=frame,
            x="response_standardized",
            y="resistance_score",
            order=["CR", "PR", "NR"],
            hue="response_standardized",
            palette=RESPONSE_COLORS,
            errorbar=("ci", 95),
            dodge=False,
            markers="o",
            linestyles="-",
            linewidth=1.2,
            ax=ax,
            legend=False,
        )
        sns.stripplot(data=frame, x="response_standardized", y="resistance_score", order=["CR", "PR", "NR"], hue="response_standardized", palette=RESPONSE_COLORS, jitter=0.12, size=3, alpha=0.75, ax=ax, legend=False)
        ax.axhline(0, color="#888888", lw=0.6, ls="--")
        ax.set_title(PROGRAM_LABELS[source], loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Resistance score" if i % 3 == 0 else "")
        clean_axes(ax)
    fig.suptitle("Pretreatment patient-level score follows CR to PR to NR ordering", x=0.08, ha="left", fontsize=9, fontweight="bold")
    fig.text(0.08, 0.015, "Points are patient-celltype pseudobulk units; error bars are bootstrap-style 95% confidence intervals from seaborn resampling.", color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    save(fig, "DRAFT_PRET_CR_PR_NR_program_gradient")


def plot_deg_summary():
    data = pd.read_csv(ROOT / "05_response_state_discovery" / "CNP0004138_PRET_CR_vs_NR_DEG_by_celltype.tsv", sep="\t")
    selected = []
    for celltype, frame in data.groupby("cell_type"):
        top_nr = frame.sort_values(["p_value", "log2FC_NR_minus_CR"], ascending=[True, False]).query("log2FC_NR_minus_CR > 0").head(3)
        top_cr = frame.sort_values(["p_value", "log2FC_NR_minus_CR"], ascending=[True, True]).query("log2FC_NR_minus_CR < 0").head(3)
        selected.append(pd.concat([top_cr, top_nr]))
    plot = pd.concat(selected, ignore_index=True)
    plot["label"] = plot["gene_symbol"] + "  "
    plot["cell_type"] = pd.Categorical(plot["cell_type"], PROGRAM_ORDER, ordered=True)
    plot = plot.sort_values(["cell_type", "log2FC_NR_minus_CR"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.1))
    for i, (source, ax) in enumerate(zip(PROGRAM_ORDER, axes.flat)):
        frame = plot[plot["cell_type"] == source].reset_index(drop=True)
        colors = np.where(frame["log2FC_NR_minus_CR"] > 0, RESPONSE_COLORS["NR"], RESPONSE_COLORS["CR"])
        ax.barh(np.arange(len(frame)), frame["log2FC_NR_minus_CR"], color=colors, height=0.68)
        ax.set_yticks(np.arange(len(frame)), frame["gene_symbol"])
        ax.axvline(0, color="#555555", lw=0.7)
        ax.set_title(PROGRAM_LABELS[source], loc="left", fontweight="bold")
        ax.set_xlabel("log2FC (NR - CR)")
        ax.text(0.99, 0.02, "FDR<0.05: 0", transform=ax.transAxes, ha="right", color="#555555")
        sns.despine(ax=ax, left=True)
    fig.suptitle("Effect-ranked discovery genes are nominal, not transcriptome-wide significant", x=0.08, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "DRAFT_celltype_DEG_effect_summary")


def plot_score_by_celltype():
    data = load_discovery_scores()
    data["program"] = data["source_cell_type"].map(PROGRAM_LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sns.boxplot(data=data, x="program", y="resistance_score", hue="response_standardized", hue_order=["CR", "PR", "NR"], palette=RESPONSE_COLORS, showfliers=False, linewidth=0.8, ax=ax)
    sns.stripplot(data=data, x="program", y="resistance_score", hue="response_standardized", hue_order=["CR", "PR", "NR"], palette=RESPONSE_COLORS, dodge=True, jitter=0.12, size=2.5, alpha=0.75, ax=ax, legend=False)
    ax.axhline(0, color="#888888", lw=0.6, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("Resistance score")
    ax.set_title("Locked program scores by broad compartment and response", loc="left", fontweight="bold")
    ax.legend(title="PRET response", frameon=False, ncol=3, loc="upper left")
    ax.tick_params(axis="x", rotation=20)
    clean_axes(ax)
    save(fig, "DRAFT_CNP0004138_program_score_by_celltype_response")


def plot_score_heatmap():
    data = load_discovery_scores()
    pivot = data.pivot(index="patient_id", columns="source_cell_type", values="resistance_score")
    response = data.drop_duplicates("patient_id").set_index("patient_id")["response_standardized"]
    order = response.map({"CR": 0, "PR": 1, "NR": 2}).sort_values().index
    pivot = pivot.reindex(order).reindex(columns=PROGRAM_ORDER)
    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.22, 5], wspace=0.03)
    ax0 = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    side = np.array([mpl.colors.to_rgba(RESPONSE_COLORS[response.loc[p]]) for p in pivot.index]).reshape(-1, 1, 4)
    ax0.imshow(side, aspect="auto")
    ax0.set_xticks([])
    ax0.set_yticks(np.arange(len(pivot)), pivot.index, fontsize=5.5)
    ax0.set_ylabel("")
    sns.heatmap(pivot, cmap="vlag", center=0, linewidths=0.2, linecolor="white", cbar_kws={"label": "Resistance score", "shrink": 0.65}, ax=ax)
    ax.set_xticklabels([PROGRAM_LABELS[x] for x in pivot.columns], rotation=30, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Pretreatment locked scores across response-ordered patients", loc="left", fontweight="bold")
    fig.text(0.025, 0.50, "Patient", rotation=90, va="center")
    fig.text(0.07, 0.02, "Side color: CR blue, PR amber, NR vermillion. Missing compartment units remain blank.", color="#555555")
    save(fig, "DRAFT_CNP0004138_program_score_heatmap")


def plot_dynamics():
    data = pd.read_csv(ROOT / "05_response_state_discovery" / "CNP0004138_pre_post_program_dynamics_summary.tsv", sep="\t")
    data = data[(data["comparison"] == "POSTT_minus_PRET") & data["response_standardized"].isin(["CR", "PR", "NR"])]
    data["program"] = data["source_cell_type"].map(PROGRAM_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    for i, (response, ax) in enumerate(zip(["CR", "PR", "NR"], axes)):
        frame = data[data["response_standardized"] == response].set_index("source_cell_type").reindex(PROGRAM_ORDER).reset_index()
        y = np.arange(len(frame))
        ax.errorbar(
            frame["mean_paired_change"],
            y,
            xerr=[frame["mean_paired_change"] - frame["bootstrap_CI95_low"], frame["bootstrap_CI95_high"] - frame["mean_paired_change"]],
            fmt="o",
            color=RESPONSE_COLORS[response],
            ecolor="#666666",
            capsize=2,
            markersize=4,
        )
        ax.axvline(0, color="#777777", lw=0.7, ls="--")
        ax.set_yticks(y)
        ax.set_title(response, color=RESPONSE_COLORS[response], fontweight="bold")
        ax.set_xlabel("Mean paired change\nPOSTT - PRET")
        ax.invert_yaxis()
        clean_axes(ax)
    axes[0].set_yticks(y, [PROGRAM_LABELS[x] for x in frame["source_cell_type"]])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Post-treatment remodeling differs by pretreatment response group", x=0.08, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "DRAFT_PRE_POST_program_dynamics_by_response")


def plot_validation_scores(dataset: str, folder: str, good: str, poor: str, output: str):
    data = pd.read_csv(ROOT / folder / f"{dataset}_locked_program_scores.tsv", sep="\t")
    group_col = "response_standardized" if dataset == "GSE209746" else "response_endpoint"
    label_map = {good: "Good", poor: "Poor"}
    data = data[data[group_col].isin([good, poor])].copy()
    data["response_group"] = data[group_col].map(label_map)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.3), sharey=False)
    for source, ax in zip(PROGRAM_ORDER, axes.flat):
        frame = data[data["program_source_cell_type"] == source]
        sns.boxplot(data=frame, x="response_group", y="resistance_score", order=["Good", "Poor"], palette={"Good": RESPONSE_COLORS["CR"], "Poor": RESPONSE_COLORS["NR"]}, hue="response_group", legend=False, showfliers=False, linewidth=0.8, ax=ax)
        sns.stripplot(data=frame, x="response_group", y="resistance_score", order=["Good", "Poor"], palette={"Good": RESPONSE_COLORS["CR"], "Poor": RESPONSE_COLORS["NR"]}, hue="response_group", legend=False, jitter=0.16, size=2.1, alpha=0.55, ax=ax)
        ax.set_title(PROGRAM_LABELS[source], loc="left", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Resistance score" if source in ["All_cells", "Myeloid_macrophage"] else "")
        clean_axes(ax)
    subtitle = "27 CR vs 70 iCR" if dataset == "GSE209746" else "35 strict pCR vs 166 non-pCR"
    fig.suptitle(f"{dataset} locked external validation ({subtitle})", x=0.08, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, output)


def plot_gse872_effects():
    data = pd.read_csv(ROOT / "08_gse87211_validation" / "GSE87211_validation_results.tsv", sep="\t")
    subsets = ["all_evaluable", "5-FU + Oxaliplatin + RT", "5-FU + RT"]
    labels = ["All evaluable", "Oxaliplatin + RT", "5-FU + RT"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), sharey=True)
    for i, (subset, label, ax) in enumerate(zip(subsets, labels, axes)):
        frame = data[data["cohort_subset"] == subset].set_index("program_source_cell_type").reindex(PROGRAM_ORDER).reset_index()
        y = np.arange(len(frame))
        ax.errorbar(frame["hedges_g_poor_minus_good"], y, fmt="o", color="#333333", markersize=4)
        for yy, (_, row) in enumerate(frame.iterrows()):
            if row["mannwhitney_FDR_BH"] < 0.05:
                ax.text(row["hedges_g_poor_minus_good"] + 0.04, yy, "*", va="center", color=RESPONSE_COLORS["NR"], fontweight="bold")
        ax.axvline(0, color="#777777", lw=0.7, ls="--")
        ax.set_yticks(y)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Hedges g\n(non-pCR - pCR)")
        ax.invert_yaxis()
        clean_axes(ax)
    axes[0].set_yticks(y, [PROGRAM_LABELS[x] for x in frame["program_source_cell_type"]])
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("GSE87211 validation is treatment-regimen dependent", x=0.08, ha="left", fontsize=9, fontweight="bold")
    fig.text(0.08, 0.01, "* BH FDR < 0.05 within six locked programs. Regimen panels are sensitivity analyses.", color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    save(fig, "DRAFT_GSE87211_validation_score_plot")


def calculate_discovery_g(data):
    rows = []
    for source, frame in data.groupby("source_cell_type"):
        poor = frame.loc[frame["response_standardized"] == "NR", "resistance_score"].to_numpy()
        good = frame.loc[frame["response_standardized"] == "CR", "resistance_score"].to_numpy()
        nx, ny = len(poor), len(good)
        pooled = np.sqrt(((nx - 1) * poor.var(ddof=1) + (ny - 1) * good.var(ddof=1)) / (nx + ny - 2))
        correction = 1 - 3 / (4 * (nx + ny) - 9)
        rows.append((source, correction * (poor.mean() - good.mean()) / pooled))
    return dict(rows)


def plot_cross_dataset_heatmap():
    discovery = load_discovery_scores()
    g_discovery = calculate_discovery_g(discovery)
    g209 = pd.read_csv(ROOT / "07_gse209746_validation" / "GSE209746_validation_results.tsv", sep="\t")
    g872 = pd.read_csv(ROOT / "08_gse87211_validation" / "GSE87211_validation_results.tsv", sep="\t")
    columns = ["CNP discovery", "GSE209 all", "GSE209 MSS", "GSE872 all", "GSE872 oxaliplatin", "GSE872 5-FU"]
    matrix = pd.DataFrame(index=PROGRAM_ORDER, columns=columns, dtype=float)
    for source in PROGRAM_ORDER:
        matrix.loc[source, "CNP discovery"] = g_discovery[source]
    mappings = [
        (g209, "all_evaluable", "GSE209 all"),
        (g209, "MSS_only", "GSE209 MSS"),
        (g872, "all_evaluable", "GSE872 all"),
        (g872, "5-FU + Oxaliplatin + RT", "GSE872 oxaliplatin"),
        (g872, "5-FU + RT", "GSE872 5-FU"),
    ]
    for frame, subset, column in mappings:
        values = frame[frame["cohort_subset"] == subset].set_index("program_source_cell_type")["hedges_g_poor_minus_good"]
        matrix.loc[values.index, column] = values
    matrix.index = [PROGRAM_LABELS[x] for x in matrix.index]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", cbar_kws={"label": "Hedges g (poor - good)", "shrink": 0.72}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Cross-dataset effect direction is consistent but context dependent", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.08, 0.025, "CNP values are discovery-trained and should not be compared as independent validation.", color="#555555")
    save(fig, "DRAFT_Phase1_cross_dataset_evidence_heatmap")


def main():
    plot_umap()
    plot_composition()
    plot_gradient()
    plot_deg_summary()
    plot_score_by_celltype()
    plot_score_heatmap()
    plot_dynamics()
    plot_validation_scores(
        "GSE209746",
        "07_gse209746_validation",
        "complete_response",
        "incomplete_response",
        "DRAFT_GSE209746_validation_score_plot",
    )
    plot_gse872_effects()
    plot_cross_dataset_heatmap()
    print("Created Phase 1 draft figures with Python.")


if __name__ == "__main__":
    main()
