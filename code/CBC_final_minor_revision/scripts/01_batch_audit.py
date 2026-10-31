from pathlib import Path
import pandas as pd,numpy as np,json,re
R=Path(__file__).resolve().parents[1];ROOT=R.parent;P1=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery'
m=pd.read_csv(ROOT/'CBC_revision_next/source_data/cell_metadata_linked.tsv',sep='\t')
m['source_library_label']=m.barcode.str.extract(r'^(.*?-lib-\d+)(?:_|$)')[0]
assert m.source_library_label.notna().all() and m.barcode.is_unique
inv=m.groupby(['source_library_label','patient_id','sample_id','timepoint','response_standardized']).size().rename('n_cells').reset_index()
assert inv.source_library_label.is_unique
inv.to_csv(R/'analysis/source_library_label_inventory.tsv',sep='\t',index=False)
keep=m.timepoint.eq('PRET')&m.response_standardized.isin(['CR','PR','NR']);labels=sorted(m.loc[keep,'source_library_label'].unique());labelmap={v:i for i,v in enumerate(labels)}
group=np.full(len(m),-1,int);local=np.full(len(m),-1,int);rr=[]
for key,g in labelmap.items():
 ix=np.flatnonzero(keep&m.source_library_label.eq(key));group[ix]=g;local[ix]=np.arange(len(ix));rr.append(dict(group_index=g,source_library_label=key,n_cells=len(ix),patient_id=m.iloc[ix[0]].patient_id))
np.savetxt(R/'source_data/library_lookup.tsv',np.column_stack([group,local]),fmt='%d',delimiter='\t');pd.DataFrame(rr).to_csv(R/'source_data/selected_library_groups.tsv',sep='\t',index=False)
m.to_csv(R/'source_data/cell_metadata_with_source_library.tsv',sep='\t',index=False)
fields=[]
for p in [P1/'01_metadata/CNP0004138_frozen_metadata.tsv',P1/'02_scRNA_matrix/CNP0004138_scRNA_cell_metadata_clean.tsv',P1/'03_celltype_annotation/CNP0004138_celltype_annotation.tsv']:
 cols=pd.read_csv(p,sep='\t',nrows=0).columns.tolist();fields.append(dict(path=str(p.resolve()),columns=cols,explicit_capture_fields=[c for c in cols if re.search(r'capture|batch|library|lane|chip|run',c,re.I)]))
audit=dict(all_cells=len(m),all_source_library_labels=m.source_library_label.nunique(),PRET_cells=int(m.timepoint.eq('PRET').sum()),PRET_source_library_labels=m.loc[m.timepoint.eq('PRET'),'source_library_label'].nunique(),response_labeled_PRET_cells=int(keep.sum()),response_labeled_PRET_patients=m.loc[keep,'patient_id'].nunique(),analyzed_source_library_groups=len(labels),all_barcodes_parsed=True,explicit_capture_manifest_unavailable=True,shared_cross_patient_batch_unavailable=True,source_paper_platform='DNBelab C4 droplet system',source_paper_prior_doublet_filter='DoubletFinder, marker and cluster filtering reported; individual source calls unavailable',source_paper='https://pmc.ncbi.nlm.nih.gov/articles/PMC10591051/',interpretation='source barcode lib labels are usable computational partitions, not independently verified capture/chip IDs; Scrublet will be residual-doublet sensitivity, not validated doublet correction',fields=fields)
(R/'qa/batch_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
protocol=dict(seed=20261101,scanpy='1.12.2',partition='source barcode library label, preserving all response-labeled PRET cells of every lineage',sim_doublet_ratio=2,expected_doublet_rate=.05,expected_rate_status='fixed software-default prior, not known DNBelab loading rate',stdev_doublet_rate=.02,n_prin_comps=30,use_approx_neighbors=False,threshold='automatic simulated-score valley, no outcome-based tuning',downstream='frozen LR axes, same patient aggregation/library denominators/minimum20cells; no M1-M4 reselection rerun',no_ambient_correction=True,no_assumption_of_verified_capture_independence=True)
(R/'source_data/residual_doublet_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in audit.items() if k!='fields'},ensure_ascii=False,indent=2))
