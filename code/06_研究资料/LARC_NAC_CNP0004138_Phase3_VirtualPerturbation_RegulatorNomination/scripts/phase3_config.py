from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT.parent / "LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery"
P2 = ROOT.parent / "LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization"
INPUT = ROOT / "01_inputs_from_phase1_phase2"

SEED = 20260701
RESPONSE_ORDER = ["CR", "PR", "NR"]

TARGET_STATES = {
    "Malignant_EMT_stress": {
        "cell_type": "Malignant_epithelial",
        "priority": 1,
        "virtual_perturbation": True,
        "hypothesis": "A stress-adapted mesenchymal malignant state sustains the locked NR-high program.",
        "risk": "Immediate-early stress can reflect dissociation or treatment response rather than stable resistance.",
    },
    "Myeloid_suppressive_remodeling": {
        "cell_type": "Myeloid_macrophage",
        "priority": 2,
        "virtual_perturbation": True,
        "hypothesis": "SPP1/APOE-like macrophage remodeling supports the multicellular resistance niche.",
        "risk": "Macrophage state continua and patient composition can confound regulator ranking.",
    },
    "CD8_dysfunctional_stress": {
        "cell_type": "CD8_T",
        "priority": 3,
        "virtual_perturbation": True,
        "hypothesis": "A dysfunctional or stress-associated CD8 state accompanies NR-high spatial niches.",
        "risk": "Cytotoxic activation and dysfunction share genes and cannot be separated by a single module.",
    },
    "Endothelial_remodeling": {
        "cell_type": "Endothelial",
        "priority": 4,
        "virtual_perturbation": True,
        "hypothesis": "Inflammatory and angiogenic endothelial remodeling supports resistant niches.",
        "risk": "Low endothelial abundance and small matched-domain localization can reduce stability.",
    },
    "CAF_supporting_comparator": {
        "cell_type": "CAF_Fibroblast",
        "priority": 5,
        "virtual_perturbation": False,
        "hypothesis": "CAF signaling provides supporting context but should not dominate the new mechanism.",
        "risk": "Results may duplicate the published CAF_FAP/MIR4435-2HG axis.",
    },
}

STATE_MODULES = {
    "Malignant_EMT": ["VIM", "FN1", "ZEB1", "SNAI2", "TWIST1", "SERPINE1", "TGFB1"],
    "Malignant_hypoxia": ["HIF1A", "VEGFA", "CA9", "SLC2A1"],
    "Malignant_epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "CEACAM5"],
    "Malignant_proliferation": ["MKI67", "TOP2A", "STMN1"],
    "Malignant_apoptosis_stress": ["DDIT3", "ATF3", "JUN", "FOS"],
    "Myeloid_macrophage": ["LYZ", "CD68", "CD14", "FCGR3A"],
    "Myeloid_suppressive_remodeling": ["SPP1", "APOE", "C1QA", "C1QB", "MRC1", "MARCO"],
    "Myeloid_inflammatory": ["IL1B", "TNF", "CXCL8", "NFKBIA"],
    "Myeloid_antigen_presentation": ["HLA-DRA", "HLA-DRB1", "CD74"],
    "CD8_cytotoxic": ["GZMB", "PRF1", "NKG7", "GNLY"],
    "CD8_exhaustion": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX"],
    "CD8_progenitor_like": ["TCF7", "IL7R", "CCR7"],
    "CD8_activation": ["IFNG", "CXCL13", "CD69"],
    "Endothelial_identity": ["PECAM1", "VWF", "KDR"],
    "Endothelial_angiogenic": ["VEGFA", "PLVAP", "ANGPT2"],
    "Endothelial_inflammatory": ["SELE", "ICAM1", "VCAM1"],
    "Endothelial_remodeling": ["COL4A1", "COL4A2", "FN1"],
    "CAF_identity": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "CAF_FAP_myCAF": ["FAP", "POSTN", "ACTA2"],
    "CAF_inflammatory": ["CXCL12", "IL6", "CCL2"],
    "CAF_BMP4_like": ["BMP4"],
    "CAF_source_axis": ["FAP", "MIR4435-2HG", "AREG", "BMP4", "POSTN"],
}

TARGET_STATE_COMPONENTS = {
    "Malignant_EMT_stress": ["Malignant_EMT", "Malignant_hypoxia", "Malignant_apoptosis_stress"],
    "Myeloid_suppressive_remodeling": ["Myeloid_suppressive_remodeling"],
    "CD8_dysfunctional_stress": ["CD8_exhaustion", "CD8_activation"],
    "Endothelial_remodeling": ["Endothelial_angiogenic", "Endothelial_inflammatory", "Endothelial_remodeling"],
    "CAF_supporting_comparator": ["CAF_FAP_myCAF", "CAF_inflammatory", "CAF_source_axis"],
}

TF_CANDIDATES = {
    "Malignant_EMT_stress": ["JUN", "FOS", "FOSL1", "ATF3", "HIF1A", "TEAD1", "TEAD4", "SMAD2", "SMAD3", "SMAD4", "SNAI2", "ZEB1", "RUNX1", "RUNX2", "STAT3"],
    "Myeloid_suppressive_remodeling": ["NFKB1", "RELA", "STAT3", "SPI1", "CEBPB", "IRF1", "IRF8", "HIF1A", "JUN", "FOS", "FOSL1"],
    "CD8_dysfunctional_stress": ["TOX", "TCF7", "TBX21", "EOMES", "NFATC1", "NFATC2", "NFATC3", "IRF1", "STAT1", "BATF", "NR4A1", "NR4A2", "NR4A3", "JUN", "FOS"],
    "Endothelial_remodeling": ["KLF2", "KLF4", "ETS1", "ERG", "HIF1A", "RELA", "STAT3", "JUN", "FOS"],
    "CAF_supporting_comparator": ["FOSL1", "JUN", "FOS", "SMAD3", "STAT3", "TEAD1", "TEAD4", "RUNX1"],
}

CAF_AXIS_GENES = {
    "FAP", "MIR4435-2HG", "AREG", "BMP4", "POSTN", "ACTA2", "COL1A1", "COL1A2",
    "DCN", "LUM", "CXCL12", "IL6", "CCL2",
}

