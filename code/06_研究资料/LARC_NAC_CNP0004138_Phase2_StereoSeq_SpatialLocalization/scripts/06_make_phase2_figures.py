from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spatial_config import DOMAIN_MODULES, PROGRAM_LABELS, PROGRAM_ORDER, RESPONSE, SECTIONS


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization")
OUT = ROOT / "12_draft_figures"

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
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
    }
)
sns.set_style("ticks")

CR_COLOR = "#0072B2"
NR_COLOR = "#D55E00"
NEUTRAL = "#5B5B5B"
PROGRAMS = [PROGRAM_LABELS[x] for x in PROGRAM_ORDER]
PROGRAM_PRETTY = {"global": "Global", "malignant": "Malignant", "CAF": "CAF", "myeloid": "Myeloid", "CD8": "CD8 T", "endothelial": "Endothelial"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean(ax, grid=True):
    sns.despine(ax=ax)
    if grid:
        ax.grid(axis="y", color="#E6E6E6", lw=0.5)
        ax.set_axisbelow(True)


def section_title(ax, section):
    color = CR_COLOR if RESPONSE[section] == "CR" else NR_COLOR
    qc = "low complexity" if section in ["ST-CR2", "ST-NR2"] else "pass"
    ax.set_title(f"{section} | {RESPONSE[section]} | {qc}", color=color, fontweight="bold", loc="left")


def load_program(section):
    return pd.read_parquet(
        ROOT / "05_spatial_program_scores" / f"{section.replace('-', '_')}_bin_level_program_scores.parquet"
    )


def load_domain(section):
    return pd.read_parquet(
        ROOT / "07_spatial_domain_annotation" / f"{section.replace('-', '_')}_spatial_domain_scores.parquet"
    )


def plot_qc(metric, ylabel, name, log=False):
    frames = []
    rng = np.random.default_rng(4138)
    for section in SECTIONS:
        data = pd.read_parquet(ROOT / "03_stereoseq_matrix" / f"{section.replace('-', '_')}_bin_qc.parquet")
        if len(data) > 15000:
            data = data.iloc[rng.choice(len(data), 15000, replace=False)]
        frames.append(pd.DataFrame({"section_id": section, metric: data[metric].to_numpy()}))
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    sns.violinplot(data=data, x="section_id", y=metric, order=SECTIONS, color="#B8C7D9", inner="quartile", cut=0, linewidth=0.8, ax=ax)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " by Stereo-seq section", loc="left", fontweight="bold")
    ax.text(0.01, -0.18, "Violin shapes use at most 15,000 bins per section for rendering; section-level inference is not based on bins.", transform=ax.transAxes, color="#555555")
    clean(ax)
    save(fig, name)


def plot_spatial_program(label):
    values = []
    data_by_section = {}
    for section in SECTIONS:
        data = load_program(section)
        data_by_section[section] = data
        values.append(data[f"{label}_resistance_score"].to_numpy())
    pooled = np.concatenate(values)
    limit = np.nanquantile(np.abs(pooled), 0.98)
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), constrained_layout=True)
    scatter = None
    for ax, section in zip(axes, SECTIONS):
        data = data_by_section[section]
        size = float(np.clip(90000 / len(data), 0.35, 5.0))
        scatter = ax.scatter(
            data["x"],
            data["y"],
            c=data[f"{label}_resistance_score"],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            s=size,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        section_title(ax, section)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.70, pad=0.01)
    cbar.set_label("Within-section resistance score")
    fig.suptitle(f"{PROGRAM_PRETTY[label]} locked resistance program | descriptive bin maps", x=0.02, ha="left", fontsize=9, fontweight="bold")
    save(fig, f"DRAFT_spatial_maps_{label}_resistance_score")


def plot_section_scores():
    data = pd.read_csv(ROOT / "06_section_level_summary" / "StereoSeq_section_level_program_scores.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = {program: i for i, program in enumerate(PROGRAMS)}
    offsets = {"ST-CR1": -0.24, "ST-CR2": -0.08, "ST-NR1": 0.08, "ST-NR2": 0.24}
    markers = {"ST-CR1": "o", "ST-CR2": "s", "ST-NR1": "o", "ST-NR2": "s"}
    for section in SECTIONS:
        frame = data[data["section_id"] == section]
        xs = [positions[p] + offsets[section] for p in frame["program_name"]]
        ax.scatter(
            xs,
            frame["section_pseudobulk_resistance_score"],
            color=CR_COLOR if RESPONSE[section] == "CR" else NR_COLOR,
            marker=markers[section],
            s=35,
            edgecolor="white",
            linewidth=0.5,
            label=section,
            zorder=3,
        )
    ax.axhline(0, color="#777777", ls="--", lw=0.7)
    ax.set_xticks(range(len(PROGRAMS)), [PROGRAM_PRETTY[p] for p in PROGRAMS], rotation=20, ha="right")
    ax.set_ylabel("Section-pseudobulk resistance score")
    ax.set_xlabel("")
    ax.set_title("Locked program direction uses one value per section", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, title="Section", loc="upper left")
    clean(ax)
    save(fig, "DRAFT_section_level_CR_vs_NR_program_scores")


def plot_section_heatmap():
    data = pd.read_csv(ROOT / "06_section_level_summary" / "StereoSeq_section_level_program_scores.tsv", sep="\t")
    matrix = data.pivot(index="section_id", columns="program_name", values="section_pseudobulk_resistance_score").reindex(index=SECTIONS, columns=PROGRAMS)
    matrix.columns = [PROGRAM_PRETTY[p] for p in matrix.columns]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    sns.heatmap(matrix, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", cbar_kws={"label": "Resistance score", "shrink": 0.72}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Section-pseudobulk locked scores", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "DRAFT_section_level_program_effect_heatmap")


def plot_domain_maps():
    domain_names = list(DOMAIN_MODULES) + ["Mixed_or_low_signal"]
    palette = dict(zip(domain_names, sns.color_palette("tab10", n_colors=len(domain_names))))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.8), constrained_layout=True)
    for ax, section in zip(axes, SECTIONS):
        programs = load_program(section)
        domain = load_domain(section)
        for category in domain_names:
            mask = domain["marker_domain_annotation"] == category
            ax.scatter(
                programs.loc[mask, "x"],
                programs.loc[mask, "y"],
                s=float(np.clip(90000 / len(domain), 0.35, 5.0)),
                color=palette[category],
                linewidths=0,
                rasterized=True,
                label=category,
            )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        section_title(ax, section)
        for spine in ax.spines.values():
            spine.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [x.replace("_domain", "").replace("_", " ") for x in labels], frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Marker-module spatial domains | niche labels, not purified cell types", x=0.02, ha="left", fontsize=9, fontweight="bold")
    save(fig, "DRAFT_spatial_domain_maps")


def plot_localization_heatmap():
    data = pd.read_csv(ROOT / "08_program_localization" / "StereoSeq_CR_NR_localization_summary.tsv", sep="\t")
    matrix = data.pivot(index="program_name", columns="domain_module", values="effect_NR_minus_CR_localization").reindex(PROGRAMS)
    matrix.index = [PROGRAM_PRETTY[p] for p in matrix.index]
    matrix.columns = [x.replace("_domain", "").replace("_", " ") for x in matrix.columns]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    limit = np.nanmax(np.abs(matrix.to_numpy()))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".2f", linewidths=0.4, linecolor="white", cbar_kws={"label": "NR - CR localization rho", "shrink": 0.72}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Response-group difference in within-section program-domain localization", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "DRAFT_program_domain_correlation_heatmap")


def plot_nr_niches():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    for row, section in enumerate(["ST-NR1", "ST-NR2"]):
        program = load_program(section)
        domain = load_domain(section)
        resistance = program["global_resistance_score"].to_numpy()
        niche = domain["Resistance_multicellular_niche"].to_numpy()
        joint = (resistance >= np.nanquantile(resistance, 0.8)) & (niche >= np.nanquantile(niche, 0.8))
        panels = [(resistance, "Global resistance", "RdBu_r"), (niche, "Multicellular niche", "viridis"), (joint.astype(int), "Top-quintile overlap", "magma")]
        for col, (values, title, cmap) in enumerate(panels):
            ax = axes[row, col]
            ax.scatter(program["x"], program["y"], c=values, cmap=cmap, s=float(np.clip(90000 / len(program), 0.35, 5.0)), linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{section} | {title}", loc="left", fontweight="bold")
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("NR resistance-niche maps | descriptive same-bin co-enrichment", x=0.02, ha="left", fontsize=9, fontweight="bold")
    save(fig, "DRAFT_NR_resistance_niche_maps")


def plot_comparator_heatmap():
    data = pd.read_csv(ROOT / "09_response_group_comparison" / "StereoSeq_comparator_correlation_summary.tsv", sep="\t")
    matrix = data.groupby(["program_name", "comparator_module"])["spearman_rho"].mean().unstack().reindex(PROGRAMS)
    matrix.index = [PROGRAM_PRETTY[p] for p in matrix.index]
    matrix.columns = [x.replace("_", " ") for x in matrix.columns]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    limit = max(0.15, float(np.nanmax(np.abs(matrix.to_numpy()))))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".2f", linewidths=0.4, linecolor="white", cbar_kws={"label": "Mean within-section rho", "shrink": 0.70}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Locked-program correlation with source and biological comparator modules", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "DRAFT_comparator_module_correlation_heatmap")


def plot_nonredundancy():
    data = pd.read_csv(ROOT / "09_response_group_comparison" / "StereoSeq_nonredundancy_section_residual_summary.tsv", sep="\t").set_index("program_name").reindex(PROGRAMS).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    y = np.arange(len(data))
    axes[0].barh(y, data["raw_correlation_with_section_CAF_FAP"], color="#A8A8A8")
    axes[0].axvline(0, color="#777777", ls="--", lw=0.7)
    axes[0].set_yticks(y, [PROGRAM_PRETTY[p] for p in data["program_name"]])
    axes[0].set_xlabel("Section-level rho with CAF_FAP")
    axes[0].set_title("CAF dependence", fontweight="bold")
    colors = [NR_COLOR if value > 0 else CR_COLOR for value in data["residual_effect_NR_minus_CR_after_CAF_FAP"]]
    axes[1].barh(y, data["residual_effect_NR_minus_CR_after_CAF_FAP"], color=colors)
    axes[1].axvline(0, color="#777777", ls="--", lw=0.7)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_xlabel("Residual NR - CR effect")
    axes[1].set_title("After CAF_FAP adjustment", fontweight="bold")
    axes[0].invert_yaxis()
    clean(axes[0], grid=False)
    clean(axes[1], grid=False)
    fig.suptitle("Nonredundancy screen with four-section CAF_FAP adjustment", x=0.08, ha="left", fontsize=9, fontweight="bold")
    fig.text(0.08, 0.01, "Descriptive only: residualization with n=4 is unstable.", color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    save(fig, "DRAFT_nonredundancy_summary_plot")


def plot_sensitivity():
    data = pd.read_csv(ROOT / "10_sensitivity_analysis" / "StereoSeq_leave_one_section_sensitivity.tsv", sep="\t")
    data = data[data["sensitivity_scenario"] != "all_sections"]
    scenario_order = [f"leave_out_{s}" for s in SECTIONS]
    matrix = data.pivot(index="program_name", columns="sensitivity_scenario", values="effect_NR_minus_CR").reindex(index=PROGRAMS, columns=scenario_order)
    matrix.index = [PROGRAM_PRETTY[p] for p in matrix.index]
    matrix.columns = [x.replace("leave_out_", "Remove ") for x in matrix.columns]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    limit = np.nanmax(np.abs(matrix.to_numpy()))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-limit, vmax=limit, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", cbar_kws={"label": "Recomputed NR - CR effect", "shrink": 0.72}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Leave-one-section sensitivity recomputes gene scaling", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "DRAFT_leave_one_section_sensitivity_heatmap")


def plot_cross_phase():
    path = ROOT / "11_cross_phase_integration" / "Phase1_Phase2_cross_evidence_summary.tsv"
    if not path.exists():
        return
    data = pd.read_csv(path, sep="\t")
    matrix = data.pivot(index="program_name", columns="evidence_branch", values="direction_score").reindex(PROGRAMS)
    matrix.index = [PROGRAM_PRETTY[p] for p in matrix.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.5, linecolor="white", cbar_kws={"label": "Direction score", "shrink": 0.72}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Phase 1 response evidence and Phase 2 spatial direction", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    save(fig, "DRAFT_Phase1_Phase2_evidence_heatmap")


def main():
    plot_qc("detected_genes", "Detected genes per bin", "DRAFT_StereoSeq_QC_detected_genes_by_section")
    plot_qc("total_counts", "Total counts per bin", "DRAFT_StereoSeq_QC_total_counts_by_section", log=True)
    for label in PROGRAMS:
        plot_spatial_program(label)
    plot_section_scores()
    plot_section_heatmap()
    plot_domain_maps()
    plot_localization_heatmap()
    plot_nr_niches()
    plot_comparator_heatmap()
    plot_nonredundancy()
    plot_sensitivity()
    plot_cross_phase()
    print("Created Phase 2 draft figures with Python.")


if __name__ == "__main__":
    main()
