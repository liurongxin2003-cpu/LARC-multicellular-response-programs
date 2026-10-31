# Separately acquired metadata inputs

The distributed analysis logic does not include individual source patient-response or donor-linkage constants. Original source scripts are unchanged in the author project. The public mapping manifest records both original-source and distributed-file hashes and the narrow adaptations.

For Phase1 `01_prepare_metadata.py`, set `LARC_PATIENT_RESPONSE_JSON` to an authorized local JSON object mapping the originating patient keys to two-element arrays `[response_group, TRG]`. Obtain the mapping from the originating source study. Missing metadata must remain explicit; do not infer values from this aggregate archive.

For Phase2 `01_import_phase1_and_freeze_metadata.py`, set `LARC_SPATIAL_METADATA_TSV` to the separately acquired source manifest with fields `section_id,response_group,sample_id,patient_or_donor_id,exact_donor_identity_status,tissue_region_annotation,bin_size,expected_bin_number,expected_gene_number,QC_status,notes` (tab-delimited). Preserve the unresolved NR donor linkage, and report sections rather than assuming four independent donors.

Scripts fail if required source inputs are unavailable. No artificial example patient or donor rows are supplied. These configuration adaptations change how source metadata are supplied, not the analytical definitions or statistical results.
