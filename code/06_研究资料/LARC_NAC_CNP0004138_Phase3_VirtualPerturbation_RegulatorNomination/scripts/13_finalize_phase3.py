from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from phase3_config import ROOT


REPORTS = ROOT / "13_reports"
LOGS = ROOT / "00_logs"


def read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, sep="\t")


def md_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    rows = ["| " + " | ".join(map(str, display.columns)) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(rows)


def build_report() -> str:
    states = read("03_refined_cell_state_annotation/refined_target_state_response_effects.tsv")
    ranking = read("09_candidate_prioritization/final_candidate_regulator_ranking.tsv")
    grn = read("05_grn_inference/GRN_method_concordance_summary.tsv")
    virtual = read("06_virtual_perturbation/virtual_perturbation_ranked_regulators.tsv")
    spatial = read("08_spatial_regulator_projection/regulator_section_level_summary.tsv")
    robustness = read("10_robustness_sensitivity/regulator_robustness_summary.tsv")
    lr = read("07_ligand_receptor_niche/spatially_informed_LR_priority_table.tsv")

    state_view = states[["target_state", "n_CR", "n_PR", "n_NR", "effect_NR_minus_CR", "hedges_g_NR_minus_CR", "spearman_CR_PR_NR", "direction_support"]].copy()
    state_view.columns = ["Target state", "CR", "PR", "NR", "Marker NR-CR", "Hedges g", "Gradient rho", "Expected direction"]
    for column in ["Marker NR-CR", "Hedges g", "Gradient rho"]:
        state_view[column] = state_view[column].map(lambda x: f"{x:.3f}")

    top = ranking[ranking.final_confidence == "moderate"].sort_values("final_integrated_score", ascending=False).groupby("target_state").head(3)
    top_view = top[["target_state", "regulator", "final_integrated_score", "n_positive_evidence_categories", "stability_score", "evidence_from_spatial_projection", "evidence_from_ligand_receptor", "median_delta_resistance_NR"]].copy()
    top_view.columns = ["Target state", "Regulator", "Score", "Evidence/6", "Stability", "Spatial", "LR", "Virtual delta NR"]
    for column in ["Score", "Stability", "Virtual delta NR"]:
        top_view[column] = top_view[column].map(lambda x: f"{x:.3f}")

    confidence_counts = ranking.groupby(["target_state", "final_confidence"]).size().unstack(fill_value=0).reset_index()
    return f"""# LARC NAC CNP0004138 Phase 3 virtual perturbation and regulator nomination report

## Executive verdict

**Final manuscript decision: Manuscript Go without mechanism claim.** Phase 3 successfully nominated multi-evidence regulators and ligand-receptor niche axes for the unchanged Phase 1 locked resistance programs. However, no regulator met the prespecified high-confidence threshold, and all five canonical marker-based target-state modules failed to show the expected NR-higher patient-pseudobulk direction. The project can proceed to manuscript drafting as a response-associated spatial-program study with hypothesis-generating regulator/niche analyses, not as a causal mechanism paper.

Permitted interpretation: candidate regulator, virtual perturbation suggests, spatially localized candidate axis, and requires experimental validation. Forbidden interpretation: validated regulator, knockout validation, causal mechanism proven, therapeutic target validated, clinically actionable, or clinical biomarker.

## Data and method boundary

- Dataset: [CNGB CNP0004138](https://db.cngb.org/data_resources/project/CNP0004138/), with source study [Li et al., Cell Reports Medicine 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10591051/).
- Regulator priors/activity: DoRothEA A-C and CollecTRI through [decoupler](https://decoupler.readthedocs.io/en/stable/).
- Ligand-receptor prior: 4,624-pair [LIANA consensus resource](https://liana-py.readthedocs.io/en/stable/notebooks/prior_knowledge.html).
- Primary regulator inference: PRET patient-celltype pseudobulk, with 6 CR, 14 PR, and 5 NR units per target cell type.
- Cell-level UMAP/state annotation uses a 50,000-cell sample descriptively. Cells are not independent response units.
- Spatial response inference uses four sections. Bins are descriptive observations, not patients.
- Phase 1 locked programs were never retrained, reselected, or direction-flipped.

## 1. Target states entering Phase 3

All five prespecified states entered Phase 3: malignant, myeloid, CD8-associated, endothelial, and CAF comparator. Four non-CAF states were eligible for primary regulator nomination; CAF remained comparator/supporting context.

{md_table(state_view)}

The critical negative result is explicit: canonical EMT/stress, suppressive-remodeling myeloid, CD8 dysfunction, endothelial remodeling, and CAF modules were not NR-higher. Malignant, CD8, and endothelial marker effects were instead appreciably negative. Therefore the locked programs cannot be renamed as validated canonical states.

## 2. Regulator activity

Three executable tracks were used: DoRothEA ULM, DoRothEA MLM, and prespecified-TF CollecTRI ULM. DoRothEA was genome-wide; CollecTRI was restricted to prespecified TFs after the full network caused a native runtime termination. DoRothEA-VIPER also terminated natively and produced no estimates; this failure is retained in the log.

Leading multi-method activity nominations included malignant NR2F6/MYC/E2F3 and later integrated SRF/IRF3; myeloid MAZ/CEBPB/JUN and integrated E2F6/SRF; CD8 SREBF1/PPARA with integrated ZBTB7A/KLF5/CREB1; endothelial PPARG/RELB/RFX5. Activity is prior-based inference rather than TF expression or binding validation.

## 3. GRN support

GRNs combined DoRothEA/CollecTRI prior edges, patient-pseudobulk Spearman coexpression, and 150-tree ExtraTrees feature importance as a GENIE3-style approximation. Candidate regulators had variable but nonzero two- and three-method links to locked NR-high genes. This supports network nomination but cannot prove regulatory direction or binding.

## 4. Virtual perturbation

Two network stress tests were completed for 103 regulator-state combinations: fixed signed target downweighting and activity-scaled network propagation. Thirty-nine combinations reduced the NR resistance score in both methods. Leading median NR score changes included SRF in malignant (-0.143), E2F6 in myeloid (-0.130), ZBTB7A in CD8 (-0.117), and RELB in endothelial (-0.010). CAF TEAD4 also reduced the simulated score (-0.126) but was downgraded because it overlaps the known CAF/TEAD context.

These effects are constrained by the inferred network. They are not virtual-knockout validation or causal evidence.

## 5-6. Candidate confidence

{md_table(confidence_counts)}

No candidate was high-confidence because high confidence required the corresponding canonical target-state marker module to support the NR direction. Twenty-four candidates were moderate and 63 were low/exploratory.

Leading moderate candidates:

{md_table(top_view)}

Exploratory candidates include candidates supported by only one activity prior/method, weak or positive virtual delta, absent spatial section direction, unstable sampled-cell expression, or known CAF-axis overlap. They are preserved in the full ranking and must not be promoted selectively.

## 7. Ligand-receptor niche

The LIANA consensus screen produced 27,984 patient-paired interaction rows. Two hundred positive-direction axes entered the spatially informed priority table, including 125 non-CAF axes. The most coherent hypotheses included malignant LTA-TNFRSF1A/B/14 toward myeloid/endothelial compartments and myeloid/endothelial HLA-I-KIR2DL3 associations with CD8 states.

This supports a multicellular resistance-associated niche hypothesis. It does not prove receptor engagement, signaling flux, or physical communication.

## 8. Spatial projection

Among pre-spatial candidates, spatial support was observed for 6/17 malignant-state, 8/19 myeloid-state, 8/17 CD8-state, 5/16 endothelial-state, and 6/18 CAF-context regulator-state combinations. Examples with both positive section direction and descriptive resistance correlation included malignant IRF3, myeloid E2F6/MAZ/KLF5, and CD8 ZBTB7A/KLF5. Several top integrated candidates, including malignant SRF and endothelial RELB, lacked the full spatial-support criterion and remain moderate.

TF expression in Stereo-seq is an activity proxy. Bin-level correlations are descriptive; n=4 sections define the response direction.

## 9. Independence from the CAF source story

The main regulator nominations and 125 prioritized LR axes were non-CAF. CAF-axis gene exclusion retained at least 75% of supported edges for the tested shortlist. Nevertheless, TEAD4, JUN/FOS, STAT3, and related CAF-context candidates overlap known biology and were not treated as novel high-confidence mechanisms.

## 10-11. Robustness and patient dependence

All 87 shortlisted regulator-state combinations had an integrated stability score of at least 0.75 across leave-one-patient activity, response-label sensitivity, sampled-cell subsampling, and CAF-axis exclusion. Leading candidates retained positive activity direction in leave-one-patient analyses, arguing against a single primary patient driver.

This does not eliminate model dependence. Sampled-cell expression was unstable for some candidates, such as malignant MYCN, and PRET+POST analyses used TF expression rather than refitted activity. No cell cluster or bin was used as a patient replicate.

## 12-14. Integrated main line and manuscript position

The integrated main line is valid at the **program architecture** level:

1. Phase 1: PRET CR < PR < NR locked response-state gradients with program-level external projection.
2. Phase 2: post-NAC NR-higher section scores and spatial multicellular niches.
3. Phase 3: moderate regulator and LR-axis nominations that can directionally alter the locked score in network simulations.

The recommended manuscript focus is: **pretreatment response-associated programs persist as post-treatment spatial resistance architectures and nominate multicellular regulatory niches**. Phase 3 candidates should be secondary or hypothesis-generating panels. A main EMT/remodeling/exhaustion causal mechanism figure is not justified.

## 15. Claims that remain prohibited

- Validated regulator or validated therapeutic target.
- Causal mechanism proven.
- Knockout or virtual-knockout validation.
- Clinically actionable target or clinical biomarker.
- Spatial data proves causality.
- Canonical EMT, suppressive macrophage, exhausted-CD8, or endothelial-remodeling mechanism established.

## Direct answers to the required questions

1. **States entering Phase 3:** all five; CAF only as comparator.
2. **Regulator activity:** multiple positive activity nominations in four non-CAF states, with method-specific variation.
3. **GRN support:** yes for moderate candidates through prior, coexpression, and tree evidence; not causal proof.
4. **Does virtual perturbation lower resistance?** yes for 39/103 combinations in both simulations, but the result is model-constrained.
5. **High-confidence regulators:** none under the prespecified threshold.
6. **Exploratory regulators:** 63 low-confidence regulator-state combinations, plus known CAF-axis overlaps.
7. **Multicellular LR niche:** supported as a hypothesis, especially malignant-myeloid/endothelial and myeloid/endothelial-CD8 axes.
8. **Spatial projection:** mixed candidate-specific support; strongest examples include IRF3, E2F6/MAZ/KLF5, and ZBTB7A/KLF5.
9. **Independent of CAF_FAP/MIR4435-2HG?** substantially, but not completely; non-CAF candidates and axes dominate the recommended story.
10. **Robustness:** directionally supportive within the computational system.
11. **Single-patient/cluster driver?** no leading patient-level evidence of a single driver; some sampled-cell proxies remain unstable.
12. **Integrated Phase 1-3 line:** established for response-associated program architecture, not for a causal mechanism.
13. **Proceed to manuscript drafting?** yes, without a mechanism claim.
14. **Manuscript mechanism focus:** multicellular response-associated spatial-program architecture with candidate regulator/niche nominations.
15. **Cannot claim:** validation, causality, clinical actionability, or canonical-state mechanism proof.
"""


def build_decision() -> str:
    return """# Go / No-Go for LARC manuscript drafting

## Final verdict

**Manuscript Go without mechanism claim.**

| Decision criterion | Result | Assessment |
| --- | --- | --- |
| At least two states have strong regulator nominations | Four non-CAF states have high-scoring moderate candidates | Partial pass |
| At least one non-CAF regulator nomination | Yes | Pass |
| Virtual perturbation lowers locked resistance score | 39/103 combinations supported in both network simulations | Pass with model-dependence warning |
| Regulator rankings are stable | Leading candidates pass leave-one and most sensitivity tests | Pass within computational system |
| LR analysis supports a multicellular niche | 200 prioritized axes; 125 non-CAF | Pass as hypothesis |
| Spatial projection supports candidates | Mixed, candidate-specific support | Partial pass |
| Result is not only the source CAF story | Non-CAF candidates and axes dominate | Pass |
| Canonical target-state identity is independently supported | All five marker modules fail the expected NR direction | Fail |
| High-confidence regulators exist | 0 high-confidence candidates | Fail |

## Recommended manuscript

Draft a **response-associated spatial-program paper**. Phase 1 and Phase 2 remain the primary evidence. Phase 3 should provide computational regulator and ligand-receptor nominations as secondary evidence and a roadmap for experimental validation.

Do not title or frame the article as a proven malignant EMT, suppressive-myeloid, exhausted-CD8, endothelial-remodeling, or CAF mechanism. Do not use virtual perturbation as knockout validation.

## Experimental gate for a future mechanism upgrade

Mechanism-level upgrading requires perturbing at least one nominated non-CAF regulator in the relevant cell model, showing reproducible reduction of the locked NR-high program and treatment-resistance phenotype, and validating the associated ligand-receptor/state change with an orthogonal assay.
"""


def write_qa() -> None:
    records = []
    for png in sorted((ROOT / "12_draft_figures").glob("DRAFT_*.png")):
        with Image.open(png) as image:
            array = np.asarray(image.convert("RGB").resize((160, 100)), dtype=float)
            records.append({
                "figure": png.name,
                "width_px": image.width,
                "height_px": image.height,
                "pixel_sd": round(float(array.std()), 3),
                "nonblank": bool(array.std() > 2),
                "matching_pdf": png.with_suffix(".pdf").exists(),
            })
    qa = pd.DataFrame(records)
    qa.to_csv(LOGS / "Phase3_figure_machine_QA.tsv", sep="\t", index=False)
    note = f"""# Phase 3 computational and figure QA

- Python-only backend for analysis, plotting, export, preview, and visual QA.
- Figure bundle: {len(qa)} PNG previews and {int(qa.matching_pdf.sum())} matching PDFs.
- Nonblank machine check: {int(qa.nonblank.sum())}/{len(qa)} passed.
- Visual inspection covered marker-state points, regulator activity, GRN network, virtual perturbation, LR pathways, spatial maps, ranking, stability, and integrated schematic.
- Overlapping state titles, spatial row labels, integrated-claim text, and LR-network labels were corrected after visual review.
- Quantitative panels trace to TSV/parquet outputs and `scripts/12_make_phase3_figures.py`.
- Patient or patient-celltype pseudobulk is the response unit; spatial sections are the spatial response unit; cells and bins remain descriptive.
- PDF is the requested vector deliverable; PNG is a 300-dpi review preview. TIFF/SVG were not requested for this analysis phase.
- No microscopy, contrast adjustment, stitching, or image reuse was performed.
"""
    (LOGS / "Phase3_computational_and_figure_QA.md").write_text(note, encoding="utf-8")


def write_index() -> None:
    roles = {
        "02_target_state_definition/Phase3_target_state_table.tsv": "Fixed target-state definitions and Phase 1/2 support",
        "03_refined_cell_state_annotation/refined_target_state_annotation.tsv": "Descriptive sampled-cell refined annotation",
        "04_regulator_activity/top_regulators_by_state.tsv": "Multi-method regulator-activity shortlist",
        "05_grn_inference/GRN_edges_by_target_state.tsv": "Prior/coexpression/tree GRN edges",
        "06_virtual_perturbation/virtual_perturbation_ranked_regulators.tsv": "Two-method network perturbation ranking",
        "07_ligand_receptor_niche/spatially_informed_LR_priority_table.tsv": "Patient-paired, spatially informed LR priorities",
        "08_spatial_regulator_projection/regulator_spatial_correlation_with_resistance.tsv": "Descriptive regulator-resistance spatial correlations",
        "09_candidate_prioritization/final_candidate_regulator_ranking.tsv": "Integrated final confidence ranking",
        "09_candidate_prioritization/high_confidence_candidate_regulators.md": "High-confidence threshold result",
        "10_robustness_sensitivity/regulator_ranking_stability_summary.md": "Robustness interpretation",
        "11_cross_phase_integration/Phase3_claim_evidence_boundary_table.tsv": "Claim/evidence/inference boundary",
        "11_cross_phase_integration/integrated_mechanism_model.md": "Cross-phase model and contradiction",
        "13_reports/FINAL_LARC_Phase3_virtual_perturbation_report.md": "Complete Phase 3 report and 15 answers",
        "13_reports/FINAL_go_nogo_for_LARC_manuscript_drafting.md": "Manuscript decision",
    }
    rows = []
    for relative, role in roles.items():
        path = ROOT / relative
        rows.append({"relative_path": relative, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else np.nan, "role": role})
    pd.DataFrame(rows).to_csv(LOGS / "deliverable_index.tsv", sep="\t", index=False)


def write_tree() -> None:
    lines = [ROOT.name + "/"]
    excluded = {"__pycache__"}

    def visit(folder: Path, prefix: str = "") -> None:
        entries = [entry for entry in sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower())) if entry.name not in excluded]
        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            connector = "`-- " if last else "|-- "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                visit(entry, prefix + ("    " if last else "|   "))
            else:
                lines.append(f"{prefix}{connector}{entry.name} ({entry.stat().st_size / 1024 / 1024:.2f} MB)")
    visit(ROOT)
    (LOGS / "directory_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "FINAL_LARC_Phase3_virtual_perturbation_report.md").write_text(build_report(), encoding="utf-8")
    (REPORTS / "FINAL_go_nogo_for_LARC_manuscript_drafting.md").write_text(build_decision(), encoding="utf-8")
    write_qa(); write_index(); write_tree(); write_index(); write_tree()
    print("Created Phase 3 final reports, QA notes, deliverable index, and directory tree.")


if __name__ == "__main__":
    main()
