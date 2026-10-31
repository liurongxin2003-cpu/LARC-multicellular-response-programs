from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parents[1];ROOT=R.parent;P=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization/03_stereoseq_matrix';rows=[];coverage=[];checks=[]
pairs=[('LTA','TNFRSF1A'),('LTA','TNFRSF14'),('VWF','STAB2'),('BMP8A','TGFBR1'),('HLA-B','KIR2DL3')]
for section in ['CR1','CR2','NR1','NR2']:
 target=R/f'source_data/ST_{section}_LR_gene_rows.tsv';source=P/f'ST_{section}_expression_clean.tsv'
 d=pd.read_csv(target,sep='\t',index_col=0);d.index=d.index.str.upper();assert d.index.is_unique
 values=d.to_numpy();assert np.isfinite(values).all() and (values>=0).all() and np.equal(values,np.floor(values)).all()
 z=np.load(P/f'ST_{section}_analysis_gene_counts.npz');assert np.array_equal(d.columns.to_numpy(),z['bins']);represented=list(z['genes'])
 for g in d.index:
  if g in represented:np.testing.assert_array_equal(d.loc[g].to_numpy(),z['counts'][represented.index(g)])
 xy=np.array([tuple(map(float,b.split(':')[1].split('_'))) for b in d.columns]);assert len(np.unique(xy,axis=0))==len(xy)
 for scope,mask in [('all_bins',np.ones(len(d.columns),bool)),('genes_ge300',z['detected_genes']>=300)]:
  ids=np.flatnonzero(mask);tree=cKDTree(xy[ids]);near=tree.query(xy[ids],k=9)[1];near=np.array([v[v!=j][:8] for j,v in enumerate(near)])
  for ligand,receptor in pairs:
   available=ligand in d.index and receptor in d.index
   if available:
    a=d.loc[ligand].to_numpy()[ids]>0;b=d.loc[receptor].to_numpy()[ids]>0;nb=b[near].any(axis=1);nlig=int(a.sum());nr=int(b.sum());both=int((a&b).sum());neighbor=int((a&nb).sum())
   else:nlig=nr=both=neighbor=None
   rows.append(dict(section='ST-'+section,response='CR' if section.startswith('CR') else 'NR',condition=scope,ligand=ligand,receptor=receptor,retained_bins=len(ids),ligand_present_in_full_source=ligand in d.index,receptor_present_in_full_source=receptor in d.index,ligand_positive_bins=nlig,receptor_positive_bins=nr,both_positive_same_bin=both,ligand_bins_with_receptor_in_eight_other_neighbors=neighbor,fraction_ligand_bins_with_neighbor_receptor=neighbor/nlig if available and nlig else np.nan,scope='section descriptive co-detection; mixed bins and dependent neighbors, no patient or cell-interaction inference'))
 for g in sorted(set(a for pair in pairs for a in pair)):
  exists=g in d.index;v=d.loc[g].to_numpy() if exists else None;coverage.append(dict(section='ST-'+section,gene=g,in_full_source=exists,n_bins=len(d.columns),positive_bins=int((v>0).sum()) if exists else None,total_gene_counts=float(v.sum()) if exists else None,within_frozen_1095_panel=g in represented))
 checks.append(dict(section=section,n_target_rows=len(d),integer_nonnegative_finite_counts=True,bin_order_matches=True,overlapping_original_panel_counts_match=True,n_bins=len(d.columns),full_source_absolute_path=str(source.resolve()),full_source_bytes=source.stat().st_size,extracted_target_absolute_path=str(target.resolve()),extracted_target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scope='targeted descriptive source completion; no new spatial response test, patient replication or communication inference'))
pd.DataFrame(rows).to_csv(R/'analysis/Revision_Supplementary_Table_R16i_full_spatial_codetection.tsv',sep='\t',index=False);pd.DataFrame(coverage).to_csv(R/'analysis/Revision_Supplementary_Table_R16j_full_spatial_gene_coverage.tsv',sep='\t',index=False);(R/'qa/full_spatial_target_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
(R/'source_data/spatial_source_completion_scope.json').write_text(json.dumps(dict(date='2026-10-31',reason='Complete spatial matrices located during final source audit beyond the frozen 1095-gene panel; targeted descriptive completion, not preregistered analysis',axes=pairs,conditions=['all_bins','genes_ge300'],neighbor_definition='8 nearest other retained bins, Euclidean original coordinates; self removed',outputs=['R16i','R16j'],not_claimed=['response enrichment','independent donor replication','cell-specific sender/receiver assignment','physical contact or communication','biological binding'],no_new_response_test=True),indent=2),encoding='utf8')
print(pd.DataFrame(rows).to_string(index=False))
