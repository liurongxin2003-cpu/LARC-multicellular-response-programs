"""Targeted source-cell audit; marker/QC flags are not validated doublet or soup calls."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];ROOT=R.parent;A=R/'analysis';P1=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery';P3=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase3_VirtualPerturbation_RegulatorNomination';P2=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization';rng=np.random.default_rng(20261031)
def save(x,n):pd.DataFrame(x).to_csv(A/n,sep='\t',index=False)
z=np.load(A/'raw_target_gene_counts.npz');X=z['counts'];genes=z['genes'].tolist();meta=pd.read_csv(R/'source_data/cell_metadata_linked.tsv',sep='\t');assert np.array_equal(z['barcodes'],meta.barcode.to_numpy())
eligible=meta.timepoint.eq('PRET')&meta.response_standardized.isin(['CR','PR','NR']);idx=np.flatnonzero(eligible);m=meta.loc[eligible].copy().reset_index(drop=True);X=X[:,idx];patients=sorted(m.patient_id.unique());assert len(patients)==25
panels={'Malignant_epithelial':['EPCAM','KRT8','KRT18','KRT19'],'CAF_Fibroblast':['COL1A1','COL1A2','DCN','LUM','PDGFRA'],'Myeloid_macrophage':['LST1','TYROBP','FCER1G','LYZ','CD14'],'CD8_T':['CD3D','CD3E','TRAC','CD8A','CD8B'],'Endothelial':['PECAM1','CDH5','EMCN','KDR']}
hits={k:(X[[genes.index(g) for g in gs]]>0).sum(0) for k,gs in panels.items()};lineage=np.zeros(len(m),bool);mixed=np.zeros(len(m),bool)
for typ in panels:
 target=m.broad_cell_type.eq(typ).to_numpy();own=hits[typ]>= (2 if typ=='CAF_Fibroblast' else 1)
 if typ=='CD8_T':own=((X[[genes.index(g) for g in ['CD3D','CD3E','TRAC']]]>0).sum(0)>=1)&((X[[genes.index(g) for g in ['CD8A','CD8B','NKG7']]]>0).sum(0)>=1)
 lineage[target]=own[target];other=np.stack([v>=2 for k,v in hits.items() if k!=typ]).any(0);mixed[target]=other[target]
qhigh=np.zeros(len(m),bool)
for _,ii in m.groupby(['patient_id','broad_cell_type']).groups.items():
 ii=np.asarray(ii);qhigh[ii]=(m.loc[ii,'total_counts']>m.loc[ii,'total_counts'].quantile(.95))&(m.loc[ii,'n_genes_by_nnz']>m.loc[ii,'n_genes_by_nnz'].quantile(.95))
m['lineage_compatible']=lineage;m['mixed_lineage_flag']=mixed;m['joint_high_QC_flag']=qhigh;m.to_csv(A/'R16_PRET_cell_QC_flags.tsv',sep='\t',index=False)
pairs=[('Malignant_epithelial','Myeloid_macrophage','LTA','TNFRSF1A'),('Malignant_epithelial','Myeloid_macrophage','LTA','TNFRSF14'),('CAF_Fibroblast','Malignant_epithelial','VWF','STAB2'),('Endothelial','Myeloid_macrophage','BMP8A','TGFBR1'),('Endothelial','CD8_T','HLA-B','KIR2DL3')]
conditions={'all_PRET':np.ones(len(m),bool),'high_annotation_confidence':m.annotation_confidence.eq('high').to_numpy(),'lineage_compatible':lineage,'exclude_mixed_lineage':~mixed,'exclude_joint_high_QC':~qhigh,'lineage_and_low_mixing':lineage&~mixed,'remove_single_counts':np.ones(len(m),bool),'target_counts_ge3':np.ones(len(m),bool)}
exprrows=[];retention=[];cov=[];sensitivity=[];pairpoints=[];pb=pd.read_parquet(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet');pb.index=pb.index.astype(str).str.upper();pbmeta=pd.read_csv(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv',sep='\t');error=0
targets=list(dict.fromkeys((a,g) for a,b,l,r in pairs for a,g in [(a,l),(b,r)]))
cache={}
for condition,mask in conditions.items():
 threshold=2 if condition=='remove_single_counts' else 3 if condition=='target_counts_ge3' else 1
 for typ,gene in targets:
  gi=genes.index(gene)
  for patient in patients:
   jj=np.flatnonzero(m.patient_id.eq(patient).to_numpy()&m.broad_cell_type.eq(typ).to_numpy()&mask);count=X[gi,jj];lib=m.iloc[jj].total_counts.sum();effective=np.where(count>=threshold,count,0).sum();response=m.loc[m.patient_id.eq(patient),'response_standardized'].iloc[0];score=float(np.log1p(effective/max(lib,1)*1e6));row=dict(condition=condition,patient_id=patient,cell_type=typ,gene=gene,response=response,n_cells=len(jj),gene_positive_cells=int((count>=threshold).sum()),gene_counts=float(effective),library_counts=float(lib),logCPM=score,positive_fraction=float((count>=threshold).mean()) if len(jj) else np.nan,estimable=len(jj)>=20,threshold=threshold)
   exprrows.append(row);cache[(condition,typ,gene,patient)]=row
   if condition=='all_PRET':
    pid=pbmeta.loc[pbmeta.patient_id.eq(patient)&pbmeta.cell_type.eq(typ),'pseudobulk_id'].item();old=float(pb.loc[gene,pid]);error=max(error,abs(effective-old));assert abs(effective-old)<1e-8,(patient,typ,gene,effective,old)
 for typ in panels:
  for patient in patients:
   jj=np.flatnonzero(m.patient_id.eq(patient).to_numpy()&m.broad_cell_type.eq(typ).to_numpy());kept=jj[mask[jj]];retention.append(dict(condition=condition,patient_id=patient,cell_type=typ,original_cells=len(jj),retained_cells=len(kept),retained_fraction=len(kept)/len(jj) if len(jj) else np.nan,estimable=len(kept)>=20))
 for sender,receiver,ligand,receptor in pairs:
  rr=[]
  for patient in patients:
   left=cache[(condition,sender,ligand,patient)];right=cache[(condition,receiver,receptor,patient)];ok=left['estimable']&right['estimable'];value=np.sqrt(left['logCPM']*right['logCPM']) if ok else np.nan
   rr.append(dict(condition=condition,sender=sender,receiver=receiver,ligand=ligand,receptor=receptor,patient_id=patient,response=left['response'],score=value,sender_cells=left['n_cells'],receiver_cells=right['n_cells'],ligand_positive_cells=left['gene_positive_cells'],receptor_positive_cells=right['gene_positive_cells'],estimable=ok));pairpoints.append(rr[-1])
  d=pd.DataFrame(rr);good=d.dropna(subset=['score']);cr=good[good.response.eq('CR')].score.to_numpy();nr=good[good.response.eq('NR')].score.to_numpy();ok=min(len(cr),len(nr))>=2
  if ok:
   boots=nr[rng.integers(len(nr),size=(10000,len(nr)))].mean(1)-cr[rng.integers(len(cr),size=(10000,len(cr)))].mean(1);lo,hi=np.quantile(boots,[.025,.975]);effect=nr.mean()-cr.mean()
  else:effect=lo=hi=np.nan
  sensitivity.append(dict(condition=condition,sender=sender,receiver=receiver,ligand=ligand,receptor=receptor,n_patients=len(good),n_CR=len(cr),n_NR=len(nr),n_PR=int(good.response.eq('PR').sum()),effect_NR_minus_CR=effect,bootstrap_low=lo,bootstrap_high=hi,patients_both_genes_detected=int(((good.ligand_positive_cells>0)&(good.receptor_positive_cells>0)).sum()),estimable=ok,scope='patient bootstrap conditional on fixed nominated axis; QC/count masks are sensitivity, not ambient/doublet correction or independent validation'))
# Cell-level expression across all original broad types is descriptive; patients remain units.
for gene in sorted(set([l for _,_,l,_ in pairs]+[r for _,_,_,r in pairs])):
 c=X[genes.index(gene)]
 for typ,ii in m.groupby('broad_cell_type').groups.items():
  ii=np.asarray(ii);pos=c[ii]>0
  cov.append(dict(gene=gene,cell_type=typ,n_cells=len(ii),positive_cells=int(pos.sum()),positive_fraction=float(pos.mean()),single_count_fraction_among_positive=float((c[ii][pos]==1).mean()) if pos.any() else np.nan,total_gene_counts=float(c[ii].sum()),patients_with_positive=int(m.iloc[ii[pos]].patient_id.nunique()),median_counts_positive=float(np.median(c[ii][pos])) if pos.any() else np.nan))
flagrows=[]
for typ,gene in targets:
 c=X[genes.index(gene)];jj=np.flatnonzero(m.broad_cell_type.eq(typ).to_numpy());pos=c[jj]>0
 flagrows.append(dict(cell_type=typ,gene=gene,n_cells=len(jj),gene_positive=int(pos.sum()),fraction_gene_positive_with_own_lineage=float(lineage[jj[pos]].mean()) if pos.any() else np.nan,mixed_fraction_positive=float(mixed[jj[pos]].mean()) if pos.any() else np.nan,mixed_fraction_negative=float(mixed[jj[~pos]].mean()) if (~pos).any() else np.nan,joint_high_QC_fraction_positive=float(qhigh[jj[pos]].mean()) if pos.any() else np.nan,single_count_fraction_positive=float((c[jj[pos]]==1).mean()) if pos.any() else np.nan))
original=pd.read_csv(P3/'07_ligand_receptor_niche/LR_interactions_by_response_state.tsv',sep='\t');sens=pd.DataFrame(sensitivity);baseline=sens[sens.condition.eq('all_PRET')];checks=[]
for row in baseline.itertuples():
 old=original[original.sender_cell_type.eq(row.sender)&original.receiver_cell_type.eq(row.receiver)&original.ligand.eq(row.ligand)&original.receptor.eq(row.receptor)].iloc[0];checks.append(dict(ligand=row.ligand,receptor=row.receptor,new_effect=row.effect_NR_minus_CR,original_effect=old.effect_NR_minus_CR,difference=row.effect_NR_minus_CR-old.effect_NR_minus_CR,spatial_proxy=old.pair_spatial_support_proxy,proxy_unique_values_within_sender_receiver=int(original[original.sender_cell_type.eq(row.sender)&original.receiver_cell_type.eq(row.receiver)].pair_spatial_support_proxy.nunique())))
assert max(abs(x['difference']) for x in checks)<1e-5
sp=[]
for section in ['CR1','CR2','NR1','NR2']:
 z=np.load(P2/f'03_stereoseq_matrix/ST_{section}_analysis_gene_counts.npz');gs=list(z['genes'])
 for sender,receiver,l,r in pairs:
  sp.append(dict(section='ST-'+section,sender=sender,receiver=receiver,ligand=l,receptor=r,ligand_in_frozen_spatial_panel=l in gs,receptor_in_frozen_spatial_panel=r in gs,both_available=l in gs and r in gs,n_bins=len(z['bins']),boundary='missing from selected analysis panel is not biological absence; original spatial LR score is compartment proxy, not pair-specific colocalization'))
for data,name in [(exprrows,'R16a_patient_gene_expression'),(pairpoints,'R16b_patient_axis_scores'),(sensitivity,'R16c_sensitivity'),(retention,'R16d_retention'),(cov,'R16e_broad_lineage_expression'),(flagrows,'R16f_marker_and_QC_flags'),(sp,'R16g_spatial_gene_coverage'),(checks,'R16h_original_reproduction')]:save(data,'Revision_Supplementary_Table_'+name+'.tsv')
(R/'qa/m28_validation.json').write_text(json.dumps(dict(PRET_patients=len(patients),PRET_cells=len(m),source_gene_count_max_error=error,original_LR_effect_max_error=max(abs(x['difference']) for x in checks),patient_conditions=len(conditions),axes=len(pairs),true_ambient_fraction_unavailable=True,validated_doublet_labels_unavailable=True,spatial_gene_pair_colocalization_not_claimed=True),indent=2),encoding='utf8')
print(sens.to_string(index=False));print(pd.DataFrame(flagrows).to_string(index=False))
