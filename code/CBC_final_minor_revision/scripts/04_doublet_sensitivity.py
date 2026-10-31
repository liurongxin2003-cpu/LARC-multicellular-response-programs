from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.sparse import coo_matrix
R=Path(__file__).resolve().parents[1];ROOT=R.parent;OLD=ROOT/'CBC_revision_next';P1=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery'
m=pd.read_csv(R/'source_data/cell_metadata_with_source_library.tsv',sep='\t');p=pd.read_csv(R/'analysis/Revision_Supplementary_Table_R17a_scrublet_cell_scores.tsv',sep='\t');assert len(p)==147851 and p.barcode.is_unique and p.doublet_score.notna().all(),'Incomplete scoring: do not interpret missing calls as singlets'
p=p.sort_values('source_cell_index');idx=p.source_cell_index.to_numpy();mm=m.iloc[idx].reset_index(drop=True);np.testing.assert_array_equal(mm.barcode,p.barcode)
flag=p.predicted_doublet.to_numpy(dtype=bool);masks={'original':np.ones(len(p),bool),'exclude_scrublet_flagged':~flag};rng=np.random.default_rng(20261101)
z=np.load(OLD/'analysis/raw_target_gene_counts.npz');np.testing.assert_array_equal(z['barcodes'],m.barcode);X=z['counts'][:,idx];genes=z['genes'].tolist()
pairs=[('Malignant_epithelial','Myeloid_macrophage','LTA','TNFRSF1A'),('Malignant_epithelial','Myeloid_macrophage','LTA','TNFRSF14'),('CAF_Fibroblast','Malignant_epithelial','VWF','STAB2'),('Endothelial','Myeloid_macrophage','BMP8A','TGFBR1'),('Endothelial','CD8_T','HLA-B','KIR2DL3')]
patients=sorted(mm.patient_id.unique());scores=[];summary=[];ret=[]
def contrast(d):
 cr=d.loc[d.response.eq('CR'),'score'].to_numpy();nr=d.loc[d.response.eq('NR'),'score'].to_numpy()
 if min(len(cr),len(nr))<2:return dict(n_CR=len(cr),n_NR=len(nr),effect=np.nan,low=np.nan,high=np.nan)
 b=nr[rng.integers(len(nr),size=(10000,len(nr)))].mean(1)-cr[rng.integers(len(cr),size=(10000,len(cr)))].mean(1);lo,hi=np.quantile(b,[.025,.975]);return dict(n_CR=len(cr),n_NR=len(nr),effect=float(nr.mean()-cr.mean()),low=lo,high=hi)
for condition,keep in masks.items():
 for sender,receiver,l,r in pairs:
  rows=[]
  for patient in patients:
   a=np.flatnonzero(keep&mm.patient_id.eq(patient).to_numpy()&mm.broad_cell_type.eq(sender).to_numpy());b=np.flatnonzero(keep&mm.patient_id.eq(patient).to_numpy()&mm.broad_cell_type.eq(receiver).to_numpy());lc=X[genes.index(l),a].sum();rc=X[genes.index(r),b].sum();ok=len(a)>=20 and len(b)>=20;score=np.sqrt(np.log1p(lc/max(mm.iloc[a].total_counts.sum(),1)*1e6)*np.log1p(rc/max(mm.iloc[b].total_counts.sum(),1)*1e6)) if ok else np.nan
   rows.append(dict(condition=condition,patient_id=patient,response=mm.loc[mm.patient_id.eq(patient),'response_standardized'].iloc[0],ligand=l,receptor=r,score=score,sender_cells=len(a),receiver_cells=len(b),both_genes_detected=lc>0 and rc>0));scores.append(rows[-1])
  d=pd.DataFrame(rows).dropna(subset=['score']);summary.append(dict(condition=condition,ligand=l,receptor=r,n_patients=len(d),patients_both_genes_detected=int(d.both_genes_detected.sum()),**contrast(d)))
 for (patient,typ),ii in mm.groupby(['patient_id','broad_cell_type']).groups.items():
  ii=np.array(ii);ret.append(dict(condition=condition,patient_id=patient,cell_type=typ,original_cells=len(ii),retained_cells=int(keep[ii].sum())))
pd.DataFrame(scores).to_csv(R/'analysis/Revision_Supplementary_Table_R17c_LR_patient_scores.tsv',sep='\t',index=False);lr=pd.DataFrame(summary);lr.to_csv(R/'analysis/Revision_Supplementary_Table_R17d_LR_sensitivity.tsv',sep='\t',index=False);pd.DataFrame(ret).to_csv(R/'analysis/Revision_Supplementary_Table_R17e_cell_retention.tsv',sep='\t',index=False)
original=pd.read_csv(OLD/'analysis/Revision_Supplementary_Table_R16c_sensitivity.tsv',sep='\t').query("condition=='all_PRET'")
for row in lr.query("condition=='original'").itertuples():assert abs(original.loc[original.ligand.eq(row.ligand)&original.receptor.eq(row.receptor),'effect_NR_minus_CR'].item()-row.effect)<1e-12
# Fixed-list, fixed-original-reference program sensitivity; no feature reselection or M1-M4 rerun.
programs=pd.read_csv(P1/'05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv',sep='\t');types=programs.source_cell_type.unique().tolist();fullgenes=pd.read_csv(P1/'02_scRNA_matrix/source/MM_coordinate_matrix_genes.csv',header=None).iloc[:,0].str.upper().to_numpy();assert len(fullgenes)==55697 and len(set(fullgenes))==len(fullgenes)
aggregate=np.zeros((len(patients),len(types),len(fullgenes)),np.float64);ncell=np.zeros((len(patients),len(types)),int);groups=pd.read_csv(R/'source_data/selected_library_groups.tsv',sep='\t');calls=p.set_index('source_cell_index').predicted_doublet
for row in groups.itertuples():
 ix=np.flatnonzero(m.source_library_label.eq(row.source_library_label));local=m.iloc[ix];keep=~calls.loc[ix].to_numpy(dtype=bool);pi=patients.index(row.patient_id);buf=np.fromfile(R/f'source_data/library_counts/library_{row.group_index:03d}.bin',dtype='<i4').reshape(-1,3);mat=coo_matrix((buf[:,2],(buf[:,0],buf[:,1])),shape=(len(ix),55697)).tocsr()
 for ti,typ in enumerate(types):
  mask=keep if typ=='All_cells' else keep&local.broad_cell_type.eq(typ).to_numpy();aggregate[pi,ti]+=np.asarray(mat[mask].sum(0)).ravel();ncell[pi,ti]+=mask.sum()
pb=pd.read_parquet(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet');pb.index=pb.index.str.upper();pm=pd.read_csv(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv',sep='\t');frozen=pd.read_csv(P1/'05_response_state_discovery/CNP0004138_pseudobulk_locked_program_scores.tsv',sep='\t');prs=[];means=[]
for ti,typ in enumerate(types):
 cols=pm.loc[pm.cell_type.eq(typ),'pseudobulk_id'].tolist();expr=np.log2(pb[cols].divide(pb[cols].sum(0),axis=1)*1e6+.5);gs=programs.loc[programs.source_cell_type.eq(typ)];nr=gs.loc[gs.program_direction.eq('NR_high'),'gene_symbol'].str.upper().tolist();cr=gs.loc[gs.program_direction.eq('CR_high'),'gene_symbol'].str.upper().tolist();gg=nr+cr;mu=expr.loc[gg].mean(1);sd=expr.loc[gg].std(axis=1,ddof=1).replace(0,np.nan);post=aggregate[:,ti];postexpr=np.log2(post/np.maximum(post.sum(1,keepdims=True),1)*1e6+.5);newz=(postexpr[:,[list(fullgenes).index(g) for g in gg]]-mu.to_numpy())/sd.to_numpy();newz=np.nan_to_num(newz);postscore=newz[:,:100].mean(1)-newz[:,100:].mean(1)
 for pi,patient in enumerate(patients):
  f=frozen[frozen.patient_id.eq(patient)&frozen.source_cell_type.eq(typ)].iloc[0];basez=(expr.loc[gg,f.pseudobulk_id]-mu)/sd;repro=basez.iloc[:100].fillna(0).mean()-basez.iloc[100:].fillna(0).mean();assert abs(repro-f.resistance_score)<2e-5
  for cond,sc in [('original',float(f.resistance_score)),('exclude_scrublet_flagged',postscore[pi] if ncell[pi,ti]>=20 else np.nan)]:prs.append(dict(patient_id=patient,program=typ,condition=cond,response=f.response_standardized,score=sc,retained_cells=int(f.cell_count) if cond=='original' else int(ncell[pi,ti])))
for (typ,cond),g in pd.DataFrame(prs).groupby(['program','condition']):means.append(dict(program=typ,condition=cond,n_patients=int(g.score.notna().sum()),**contrast(g.dropna(subset=['score']))))
pd.DataFrame(prs).to_csv(R/'analysis/Revision_Supplementary_Table_R17f_fixed_program_patient_scores.tsv',sep='\t',index=False);pd.DataFrame(means).to_csv(R/'analysis/Revision_Supplementary_Table_R17g_fixed_program_sensitivity.tsv',sep='\t',index=False)
overall=dict(n_cells=len(p),n_predicted_doublets=int(flag.sum()),fraction_flagged=float(flag.mean()),n_retained=int((~flag).sum()),patients=25,libraries=len(groups),thresholds='automatic per source library label; prior 0.05 not empirical loading rate',counts_and_barcodes_exact=True,LR_baseline_reproduced=True,program_reference='original 27 PRET pseudobulks per compartment; original locked 200 genes; fixed mean and sample SD; descriptive association only')
(R/'qa/doublet_sensitivity_summary.json').write_text(json.dumps(overall,indent=2),encoding='utf8');print(json.dumps(overall,indent=2));print(lr.to_string(index=False));print(pd.DataFrame(means).to_string(index=False))
