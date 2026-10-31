from pathlib import Path
import os,sys
R=Path(__file__).resolve().parents[1];ROOT=R.parent
sys.path.insert(0,str(R/'runtime'))
os.environ['NUMBA_CACHE_DIR']=str(R/'qa/numba_cache');os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd,json,time,warnings
import scanpy as sc,anndata as ad
from scipy.sparse import coo_matrix
groups=pd.read_csv(R/'source_data/selected_library_groups.tsv',sep='\t');m=pd.read_csv(R/'source_data/cell_metadata_with_source_library.tsv',sep='\t');v=pd.read_csv(R/'source_data/library_counts/selected_cell_count_validation.tsv',sep='\t');sel=m.iloc[v.source_cell_index]
np.testing.assert_array_equal(v.total_counts,sel.total_counts);np.testing.assert_array_equal(v.nnz,sel.n_genes_by_nnz)
summ=[];points=[]
for row in groups.itertuples():
 start=time.time();out=R/f'analysis/scrublet_library_{row.group_index:03d}.tsv';js=R/f'analysis/scrublet_library_{row.group_index:03d}.json'
 if out.exists() and js.exists() and json.loads(js.read_text())['status']=='computed':summ.append(json.loads(js.read_text()));points.append(pd.read_csv(out,sep='\t'));continue
 ix=np.flatnonzero(m.source_library_label.eq(row.source_library_label));assert len(ix)==row.n_cells
 buf=np.fromfile(R/f'source_data/library_counts/library_{row.group_index:03d}.bin',dtype='<i4').reshape(-1,3)
 X=coo_matrix((buf[:,2],(buf[:,0],buf[:,1])),shape=(len(ix),55697)).tocsr();assert X.nnz==len(buf);np.testing.assert_array_equal(np.asarray(X.sum(1)).ravel(),m.iloc[ix].total_counts)
 a=ad.AnnData(X.astype(np.float64));a.obs_names=m.iloc[ix].barcode.to_numpy();captured=[]
 try:
  with warnings.catch_warnings(record=True) as ww:
   warnings.simplefilter('always');sc.pp.scrublet(a,sim_doublet_ratio=2,expected_doublet_rate=.05,stdev_doublet_rate=.02,n_prin_comps=30,use_approx_neighbors=False,threshold=None,random_state=20261101,verbose=False);captured=[str(w.message) for w in ww]
  scores=a.obs.doublet_score.to_numpy();calls=a.obs.predicted_doublet.to_numpy();threshold=float(a.uns['scrublet']['threshold']);status='computed';error=''
  simulated=a.uns['scrublet']['doublet_scores_sim'];np.savez_compressed(R/f'source_data/scrublet_simulated_{row.group_index:03d}.npz',scores=simulated)
 except Exception as e:
  scores=np.full(len(ix),np.nan);calls=np.full(len(ix),None,object);threshold=np.nan;status='not_estimable';error=repr(e);simulated=np.array([])
 d=pd.DataFrame(dict(source_cell_index=ix,barcode=m.iloc[ix].barcode.to_numpy(),patient_id=m.iloc[ix].patient_id.to_numpy(),source_library_label=row.source_library_label,doublet_score=scores,predicted_doublet=calls));d.to_csv(out,sep='\t',index=False);points.append(d)
 rr=dict(source_library_label=row.source_library_label,n_cells=len(ix),patient_id=row.patient_id,n_predicted_doublets=int(np.asarray(calls,bool).sum()) if status=='computed' else None,threshold=threshold,simulated_count=len(simulated),score_min=float(np.nanmin(scores)) if status=='computed' else None,score_median=float(np.nanmedian(scores)) if status=='computed' else None,score_max=float(np.nanmax(scores)) if status=='computed' else None,status=status,error=error,warnings=captured,seconds=time.time()-start)
 js.write_text(json.dumps(rr,ensure_ascii=False,indent=2),encoding='utf8');summ.append(rr);print(row.source_library_label,status,rr['n_predicted_doublets'],'/',len(ix),'threshold',threshold,flush=True)
pd.concat(points,ignore_index=True).to_csv(R/'analysis/Revision_Supplementary_Table_R17a_scrublet_cell_scores.tsv',sep='\t',index=False);pd.DataFrame(summ).to_csv(R/'analysis/Revision_Supplementary_Table_R17b_scrublet_library_summary.tsv',sep='\t',index=False)
(R/'qa/scrublet_input_validation.json').write_text(json.dumps(dict(analyzed_cells=len(sel),total_count_max_error=0,nnz_max_error=0,n_library_groups=len(groups),computed=sum(r['status']=='computed' for r in summ),unavailable=sum(r['status']!='computed' for r in summ)),indent=2),encoding='utf8')
