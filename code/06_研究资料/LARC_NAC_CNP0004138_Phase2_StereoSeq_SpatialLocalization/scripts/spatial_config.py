from __future__ import annotations


SECTIONS = ["ST-CR1", "ST-CR2", "ST-NR1", "ST-NR2"]
RESPONSE = {"ST-CR1": "CR", "ST-CR2": "CR", "ST-NR1": "NR", "ST-NR2": "NR"}
PROGRAM_ORDER = [
    "All_cells",
    "Malignant_epithelial",
    "CAF_Fibroblast",
    "Myeloid_macrophage",
    "CD8_T",
    "Endothelial",
]
PROGRAM_LABELS = {
    "All_cells": "global",
    "Malignant_epithelial": "malignant",
    "CAF_Fibroblast": "CAF",
    "Myeloid_macrophage": "myeloid",
    "CD8_T": "CD8",
    "Endothelial": "endothelial",
}


DOMAIN_MODULES = {
    "Malignant_epithelial_domain": ["EPCAM", "KRT8", "KRT18", "KRT19", "CEACAM5", "TACSTD2"],
    "CAF_fibroblast_domain": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "ACTA2", "TAGLN"],
    "Endothelial_domain": ["PECAM1", "VWF", "KDR", "EMCN", "PLVAP"],
    "Myeloid_domain": ["LYZ", "CD68", "C1QA", "C1QB", "APOE", "SPP1"],
    "T_CD8_domain": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB", "NKG7"],
    "Proliferation_domain": ["MKI67", "TOP2A", "STMN1"],
    "Stress_hypoxia_domain": ["HIF1A", "VEGFA", "CA9", "SLC2A1"],
}


COMPARATOR_MODULES = {
    "CAF_FAP_source": [
        "FAP", "COL1A1", "COL1A2", "COL3A1", "FN1", "POSTN", "THY1", "PDGFRA", "PDGFRB", "TGFB1", "MMP2", "MMP11", "SPARC"
    ],
    "MIR4435_2HG_AREG_EMT_axis": [
        "MIR4435-2HG", "AREG", "YAP1", "CTGF", "VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "FN1", "ITGA5"
    ],
    "Generic_CAF": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "ACTA2", "TAGLN", "COL3A1", "PDGFRA"],
    "EMT_malignant_stress": [
        "VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "FN1", "ITGA5", "HIF1A", "VEGFA", "CA9", "SLC2A1", "DDIT3", "ATF4"
    ],
    "CD8_cytotoxic_exhaustion": [
        "CD3D", "CD3E", "CD8A", "CD8B", "NKG7", "GNLY", "GZMB", "GZMK", "PRF1", "IFNG", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "TOX"
    ],
    "Myeloid_inflammatory_suppressive": [
        "LYZ", "CD68", "C1QA", "C1QB", "C1QC", "APOE", "SPP1", "IL1B", "FCGR3A", "CSF1R", "TREM2", "MRC1", "MSR1", "TGFB1"
    ],
    "Endothelial_remodeling_angiogenesis": [
        "PECAM1", "VWF", "KDR", "EMCN", "PLVAP", "ESM1", "ENG", "ANGPT2", "FLT1", "TEK", "VEGFA", "RGCC", "KLF2"
    ],
    "Source_pCAF": ["PI16", "SLIT2", "ADAM28", "DCN", "CXCL12", "C3", "C7", "CFD", "CD34"],
    "Source_nCAF": ["FAP", "BMP4", "MMP1", "MIR4435-2HG", "TGFB1", "COL1A1", "COL1A2", "FN1", "POSTN"],
}


def normalize_gene(gene: str) -> str:
    return str(gene).strip().upper()
