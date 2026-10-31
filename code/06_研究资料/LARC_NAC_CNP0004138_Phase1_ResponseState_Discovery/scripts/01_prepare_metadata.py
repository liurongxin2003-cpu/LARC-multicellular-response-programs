from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\codex\LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery")
SOURCE = Path(
    r"D:\codex\HighQuality_scSpatial_VirtualPerturbation_ProjectSwitch"
    r"\06_strong_go_pilot\LARC_CNP0004138"
)


import os, json
PATIENTS = json.loads(Path(os.environ["LARC_PATIENT_RESPONSE_JSON"]).read_text(encoding="utf8"))
# Required authorized source mapping: patient key -> [response_group, TRG].


def normalize_patient(raw: str) -> str:
    number = int(re.sub(r"\D", "", raw))
    return f"XFZ{number:02d}"


def parse_barcode(barcode: str) -> dict[str, str]:
    match = re.match(r"^(XFZ-?\d+)-(PRET|POSTT|POSTN)-(.+)$", barcode)
    if not match:
        return {
            "patient_id": "unknown",
            "sample_id": "unknown",
            "timepoint": "unknown",
            "sample_type": "unknown",
        }
    patient = normalize_patient(match.group(1))
    timepoint = match.group(2)
    sample_type = {
        "PRET": "pretreatment_tumor",
        "POSTT": "posttreatment_tumor",
        "POSTN": "posttreatment_adjacent_normal",
    }[timepoint]
    return {
        "patient_id": patient,
        "sample_id": f"{patient}_{timepoint}",
        "timepoint": timepoint,
        "sample_type": sample_type,
    }


def main() -> None:
    barcode_path = SOURCE / "MM_coordinate_matrix_barcodes.csv"
    if not barcode_path.exists():
        raise FileNotFoundError(barcode_path)

    barcodes = pd.read_csv(barcode_path, header=None, names=["barcode"], dtype=str)
    parsed = pd.DataFrame([parse_barcode(value) for value in barcodes["barcode"]])
    frozen = pd.concat([barcodes, parsed], axis=1)

    frozen["response_raw"] = frozen["patient_id"].map(
        {patient: response for patient, (response, _) in PATIENTS.items()}
    ).fillna("unknown")
    frozen["response_standardized"] = frozen["response_raw"]
    frozen["TRG"] = frozen["patient_id"].map(
        {patient: trg for patient, (_, trg) in PATIENTS.items()}
    ).fillna("unknown")
    frozen["response_binary_for_primary"] = frozen["response_standardized"].map(
        {"CR": "CR_like", "NR": "NR_like"}
    ).fillna("exclude_from_primary")
    frozen["matrix_available"] = "yes"
    frozen["usable_for_phase1"] = frozen["patient_id"].ne("unknown").map(
        {True: "yes", False: "no"}
    )
    frozen["exclusion_reason"] = ""
    frozen.loc[frozen["patient_id"].eq("unknown"), "exclusion_reason"] = (
        "barcode_not_parsed"
    )
    frozen.loc[frozen["timepoint"].eq("POSTN"), "exclusion_reason"] = (
        "adjacent_normal; descriptive/sensitivity only"
    )
    frozen.loc[
        frozen["response_standardized"].eq("unknown")
        & frozen["exclusion_reason"].eq(""),
        "exclusion_reason",
    ] = "response_unknown; excluded from response contrasts"
    frozen["notes"] = "source-level processed matrix barcode"

    order = [
        "patient_id",
        "sample_id",
        "barcode",
        "timepoint",
        "response_raw",
        "response_standardized",
        "TRG",
        "response_binary_for_primary",
        "sample_type",
        "matrix_available",
        "usable_for_phase1",
        "exclusion_reason",
        "notes",
    ]
    frozen = frozen[order]
    frozen.to_csv(
        ROOT / "01_metadata/CNP0004138_frozen_metadata.tsv",
        sep="\t",
        index=False,
    )

    patient_rows = []
    for patient, (response, trg) in PATIENTS.items():
        subset = frozen[frozen["patient_id"].eq(patient)]
        timepoints = sorted(subset["timepoint"].unique())
        patient_rows.append(
            {
                "patient_id": patient,
                "response_raw": response,
                "response_standardized": response,
                "TRG": trg,
                "response_binary_for_primary": {
                    "CR": "CR_like",
                    "NR": "NR_like",
                }.get(response, "exclude_from_primary"),
                "timepoints_present": ",".join(timepoints),
                "PRET_available": "yes" if "PRET" in timepoints else "no",
                "POSTT_available": "yes" if "POSTT" in timepoints else "no",
                "POSTN_available": "yes" if "POSTN" in timepoints else "no",
                "total_cells_source_matrix": len(subset),
                "usable_for_primary_training": (
                    "yes" if response in {"CR", "NR"} and "PRET" in timepoints else "no"
                ),
            }
        )
    patient_summary = pd.DataFrame(patient_rows)
    patient_summary.to_csv(
        ROOT / "01_metadata/CNP0004138_patient_response_summary.tsv",
        sep="\t",
        index=False,
    )

    sample_summary = (
        frozen.groupby(
            [
                "patient_id",
                "sample_id",
                "timepoint",
                "sample_type",
                "response_standardized",
                "TRG",
                "response_binary_for_primary",
            ],
            dropna=False,
        )
        .size()
        .rename("cell_count_source_matrix")
        .reset_index()
        .sort_values(["patient_id", "timepoint"])
    )
    sample_summary.to_csv(
        ROOT / "01_metadata/CNP0004138_sample_timepoint_summary.tsv",
        sep="\t",
        index=False,
    )

    response_counts = patient_summary["response_standardized"].value_counts().to_dict()
    unparsed = int(frozen["patient_id"].eq("unknown").sum())
    log = f"""# CNP0004138 metadata freeze log

Freeze date: 2026-07-01

## Sources

- CNGBdb CNP0004138 processed barcode list.
- Qin et al. Supplementary Table S1 retained in the preceding feasibility audit.
- Response definitions fixed before expression analysis: CR=TRG0/1, PR=TRG2, NR=TRG3.

## Freeze result

- Source-matrix barcodes: {len(frozen):,}
- Parsed barcodes: {len(frozen) - unparsed:,}
- Unparsed barcodes: {unparsed:,}
- Patients: {len(patient_summary)}
- Patient response counts: {response_counts}
- Primary training: PRET CR versus PRET NR only.
- PR: excluded from training; retained for ordered-gradient sensitivity.
- POSTT/POSTN: excluded from pretreatment training.
- POSTN: adjacent normal, descriptive/sensitivity only.

The response and timepoint mappings passed the metadata stop gate.
"""
    (ROOT / "01_metadata/metadata_freeze_log.md").write_text(log, encoding="utf-8")

    print(f"barcodes={len(frozen)} unparsed={unparsed} patients={len(patient_summary)}")
    print(patient_summary.groupby(["response_standardized", "PRET_available"]).size())


if __name__ == "__main__":
    main()
