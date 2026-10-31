"""Revision-specific analysis. Frozen inputs are read only; all selection is repeated inside outer folds.

Exact null: enumerate the 462 assignments of 6 CR and 5 NR labels among the
11 defining patients; PR membership is fixed. This is a conditional null, not
an unrestricted 25-patient permutation test. No model tuning is performed.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import itertools,json,time,hashlib
import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
threadpool_limits(1)

R=Path(__file__).resolve().parents[1]; P=R.parent/'06_研究资料'
P1=P/'LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery'
OUT=R/'analysis';OUT.mkdir(exist_ok=True)
TYPES=['All_cells','Malignant_epithelial','CAF_Fibroblast','Myeloid_macrophage','CD8_T','Endothelial']
LABELS=['Global','Malignant','CAF','Myeloid','CD8 T','Endothelial']
SEED=20261028

def logcpm(a):
    return np.log2(a/np.maximum(a.sum(axis=0),1)[None,:]*1e6+.5)

def bh(p):
    p=np.asarray(p);ix=np.argsort(p);q=np.empty_like(p);q[ix]=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];return np.minimum(q,1)

def hedges(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a)+len(b);sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(n-2));g=(1-3/(4*n-9))*(a.mean()-b.mean())/sd
    se=np.sqrt(n/(len(a)*len(b))+g*g/(2*(n-2)))
    return g,g-1.96*se,g+1.96*se

def jt(scores,groups):
    # Mann-Whitney contributions for all three ordered group pairs, including ties.
    ans=np.zeros(scores.shape[0]);denom=0
    for lo,hi in [(0,1),(0,2),(1,2)]:
        for i in np.flatnonzero(groups==lo):
            for j in np.flatnonzero(groups==hi):
                d=scores[:,j]-scores[:,i];ans+=(d>0)+.5*(d==0);denom+=1
    return ans/denom

def fit_all_assignments(raw,train,test,masks):
    """Vectorize Welch tests over all assignments, preserving original selection rule."""
    tr=raw[:,train]
    cpm=tr/np.maximum(tr.sum(0),1)[None,:]*1e6
    keep=((cpm>=1).sum(1)>=3)&(tr.sum(1)>=10)
    selection_expr=logcpm(tr[keep])
    nr_mask=masks[:,train].T.astype(float);cr_mask=1-nr_mask
    nn=nr_mask.sum(0);nc=cr_mask.sum(0)
    smn=selection_expr@nr_mask;smc=selection_expr@cr_mask
    mean_n=smn/nn;mean_c=smc/nc;eff=mean_n-mean_c
    varn=np.maximum((selection_expr**2@nr_mask-smn**2/nn)/(nn-1),0)
    varc=np.maximum((selection_expr**2@cr_mask-smc**2/nc)/(nc-1),0)
    vn=varn/nn;vc=varc/nc
    with np.errstate(divide='ignore',invalid='ignore'):
        t=eff/np.sqrt(vn+vc);df=(vn+vc)**2/(vn**2/(nn-1)+vc**2/(nc-1))
        p=2*stats.t.sf(np.abs(t),df)
    p=np.nan_to_num(p,nan=1)
    score_expr=logcpm(raw)[keep]
    mu=score_expr[:,train].mean(1);sd=score_expr[:,train].std(1,ddof=1)
    z=np.divide(score_expr[:,test]-mu[:,None],sd[:,None],out=np.zeros((keep.sum(),len(test))),where=sd[:,None]>0)
    result=np.zeros((len(masks),len(test)));gene_indices=np.flatnonzero(keep);chosen=[]
    for k in range(len(masks)):
        sets=[]
        for sign in [1,-1]:
            inds=np.flatnonzero(sign*eff[:,k]>0)
            order=np.lexsort((-sign*eff[inds,k],p[inds,k]))
            sets.append(inds[order[:100]])
        result[k]=z[sets[0]].mean(0)-z[sets[1]].mean(0)
        chosen.append((gene_indices[sets[0]],gene_indices[sets[1]]))
    return result,chosen

def main():
    started=time.time()
    counts=pd.read_parquet(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet')
    meta=pd.read_csv(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv',sep='\t')
    summary=[];predictions=[];null_rows=[];selection_rows=[];covariates=[]
    for typ,label in zip(TYPES,LABELS):
        m=meta[(meta.cell_type==typ)&meta.response_standardized.isin(['CR','PR','NR'])].copy()
        # CR/NR first, PR last. Stable patient ordering is independent of outcomes within each subset.
        extreme=m[m.response_standardized.isin(['CR','NR'])].sort_values('patient_id')
        pr=m[m.response_standardized=='PR'].sort_values('patient_id')
        m=pd.concat([extreme,pr]).reset_index(drop=True)
        raw=counts[m.pseudobulk_id].to_numpy(float)
        masks=[]
        for cr in itertools.combinations(range(11),6):
            mask=np.ones(11,bool);mask[list(cr)]=False;masks.append(mask)
        masks=np.asarray(masks)
        obs_mask=(extreme.response_standardized=='NR').to_numpy()
        observed=int(np.flatnonzero((masks==obs_mask).all(1))[0])
        # Cross-fitting uses only the ten remaining CR/NR patients for ranking and scaling.
        oof=np.zeros((462,25));pr_folds=[]
        for held in range(11):
            train=np.array([i for i in range(11) if i!=held]);test=np.r_[held,np.arange(11,25)]
            scores,chosen=fit_all_assignments(raw,train,test,masks)
            oof[:,held]=scores[:,0];pr_folds.append(scores[:,1:])
            nr,cr=chosen[observed]
            for direction,indices in [('NR_high',nr),('CR_high',cr)]:
                for rank,idx in enumerate(indices,1):selection_rows.append(dict(program=label,omitted_patient=m.patient_id.iloc[held],direction=direction,rank=rank,gene=str(counts.index[idx])))
            print(label,'fold',held+1,'/11',flush=True)
        oof[:,11:]=np.mean(pr_folds,axis=0)
        # Each PR patient is excluded from every selection/scaling fit; averaged over all eleven fits.
        labels=np.where(masks,2,0);groups_observed=np.r_[np.where(obs_mask,2,0),np.ones(14,int)]
        delta=(oof[:,:11]*masks).sum(1)/5-(oof[:,:11]*(~masks)).sum(1)/6
        trend=np.zeros(462);rho=np.zeros(462)
        for k in range(462):
            groups=np.r_[labels[k],np.ones(14,int)]
            trend[k]=jt(oof[k:k+1],groups)[0]
            rho[k]=stats.spearmanr(groups,oof[k]).statistic
        nr_scores=oof[observed,:11][obs_mask];cr_scores=oof[observed,:11][~obs_mask]
        g,gl,gh=hedges(nr_scores,cr_scores)
        sm=dict(program=label,n_CR=6,n_PR=14,n_NR=5,conditional_assignments=462,
                mean_CR=cr_scores.mean(),mean_PR=oof[observed,11:].mean(),mean_NR=nr_scores.mean(),
                oof_delta=delta[observed],oof_g=g,oof_g_CI_low_descriptive=gl,oof_g_CI_high_descriptive=gh,
                oof_delta_exact_p=float(np.mean(delta>=delta[observed]-1e-12)),
                jt_pairwise_concordance=trend[observed],jt_conditional_exact_p=float(np.mean(trend>=trend[observed]-1e-12)),
                spearman_rho=rho[observed],PR_intermediate=bool(cr_scores.mean()<oof[observed,11:].mean()<nr_scores.mean()))
        summary.append(sm)
        for i,row in m.iterrows():
            predictions.append(dict(program=label,patient_id=row.patient_id,response=row.response_standardized,
                score=oof[observed,i],prediction_role='outer_LOO_CR_NR' if i<11 else 'PR_mean_11_training_fits',cell_count=row.cell_count))
        for k in range(462):null_rows.append(dict(program=label,assignment=k,observed=k==observed,CR_patients=';'.join(m.patient_id.iloc[np.flatnonzero(~masks[k])]),delta=delta[k],JT=trend[k],rho=rho[k]))
        full,fullgenes=fit_all_assignments(raw,np.arange(11),np.arange(25),masks[observed:observed+1])
        for direction,indices in [('NR_high',fullgenes[0][0]),('CR_high',fullgenes[0][1])]:
            for rank,idx in enumerate(indices,1):selection_rows.append(dict(program=label,omitted_patient='NONE_FULL_TRAINING_REFERENCE',direction=direction,rank=rank,gene=str(counts.index[idx])))
        print('SUMMARY',json.dumps(sm),flush=True)
        pd.DataFrame(summary).to_csv(OUT/'crossfit_summary.tsv',sep='\t',index=False)
        pd.DataFrame(predictions).to_csv(OUT/'crossfit_patient_scores.tsv',sep='\t',index=False)
        pd.DataFrame(null_rows).to_csv(OUT/'exact_reselection_null.tsv',sep='\t',index=False)
        pd.DataFrame(selection_rows).to_csv(OUT/'crossfit_training_gene_lists.tsv',sep='\t',index=False)
    d=pd.DataFrame(summary);d['oof_delta_BH_q']=bh(d.oof_delta_exact_p);d['JT_BH_q']=bh(d.jt_conditional_exact_p)
    d.to_csv(OUT/'crossfit_summary.tsv',sep='\t',index=False)
    versions={}
    import importlib.metadata as im
    for pkg in ['numpy','pandas','scipy','pyarrow','scikit-learn','statsmodels','matplotlib','SciencePlots','marsilea']:
        try:versions[pkg]=im.version(pkg)
        except im.PackageNotFoundError:pass
    (OUT/'revision_analysis_protocol.json').write_text(json.dumps(dict(seed=SEED,versions=versions,elapsed_seconds=time.time()-started,
        selection='Welch nominal P then signed mean effect, top100/direction, training-only CPM>=1 in >=3 samples and total counts>=10',
        scaling='log2(CPM+0.5) over full gene-library; each outer fold uses only training CR/NR gene means/SD',
        null='exact conditional CR/NR relabeling across all 462 assignments; full selection and outer LOO repeated; PR identities fixed',
        inference='one-sided conditional exact P; BH over six programs separately for delta and JT; nominal g CIs descriptive because crossfit estimates share training data',
        timing='Revision analysis designed after peer review; no preregistration or retrospective prespecification claimed'),indent=2),encoding='utf8')
    print(d.to_string(index=False),flush=True)

if __name__=='__main__':main()
