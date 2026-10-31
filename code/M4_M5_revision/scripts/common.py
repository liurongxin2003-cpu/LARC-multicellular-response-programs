import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,itertools,hashlib,importlib.util
import numpy as np,pandas as pd
from scipy import stats,special
from threadpoolctl import threadpool_limits
threadpool_limits(1)
R=Path(__file__).resolve().parents[1];ROOT=R.parent;P=ROOT/'06_研究资料';P1=P/'LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery';LATEST=ROOT/'09_最小必要返修_原稿母版'
TYPES=['All_cells','Malignant_epithelial','CAF_Fibroblast','Myeloid_macrophage','CD8_T','Endothelial'];NAMES=['Global','Malignant','CAF','Myeloid','CD8 T','Endothelial'];SEED=20261029
MODELS=['Intercept','Global','Global+Malignant','Global+CAF','Global+Myeloid','Global+CD8 T','Global+Endothelial','Global+Mean5']
EXTERNAL_MODELS=MODELS+['Global+All5'];LAMBDAS=[.1,1.,10.]
def read(p):return pd.read_csv(p,sep='\t')
def save(d,name):pd.DataFrame(d).to_csv(R/'analysis'/name,sep='\t',index=False)
def writej(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def logcpm(a):return np.log2(a/np.maximum(a.sum(0),1)[None,:]*1e6+.5)
def cnp_data():
    counts=pd.read_parquet(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet')
    meta=read(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv')
    m=meta[meta.cell_type.eq('All_cells')&meta.response_standardized.isin(['CR','NR'])].sort_values('patient_id').reset_index(drop=True)
    ids=m.patient_id.tolist();labels=m.response_standardized.eq('NR').to_numpy(int)
    raw=[]
    for typ in TYPES:
        mm=meta[meta.cell_type.eq(typ)].set_index('patient_id').loc[ids]
        assert not mm.pseudobulk_id.duplicated().any()
        raw.append(counts[mm.pseudobulk_id].to_numpy(float))
    return counts,meta,ids,labels,raw
def select_score(raw,train,test,masks,return_genes=False):
    """Unchanged M1 training-only feature rule, now needed to fit M4 models.
    Only selected training indices determine filtering, Welch ranking and z parameters.
    No M1 P values, M1 null summaries, or original frozen outputs are overwritten.
    """
    tr=raw[:,train];cpm=tr/np.maximum(tr.sum(0),1)[None,:]*1e6
    keep=((cpm>=1).sum(1)>=3)&(tr.sum(1)>=10);expr=logcpm(tr[keep])
    nr=masks[:,train].T.astype(float);cr=1-nr;nn=nr.sum(0);nc=cr.sum(0)
    assert nn.min()>=2 and nc.min()>=2
    sn=expr@nr;sc=expr@cr;effect=sn/nn-sc/nc
    vn=np.maximum((expr**2@nr-sn**2/nn)/(nn-1),0)/nn;vc=np.maximum((expr**2@cr-sc**2/nc)/(nc-1),0)/nc
    with np.errstate(divide='ignore',invalid='ignore'):
        t=effect/np.sqrt(vn+vc);df=(vn+vc)**2/(vn**2/(nn-1)+vc**2/(nc-1));p=2*stats.t.sf(abs(t),df)
    p=np.nan_to_num(p,nan=1)
    sx=logcpm(raw)[keep];mu=sx[:,train].mean(1);sd=sx[:,train].std(1,ddof=1)
    z=np.divide(sx[:,test]-mu[:,None],sd[:,None],out=np.zeros((keep.sum(),len(test))),where=sd[:,None]>0)
    out=np.zeros((len(masks),len(test)));chosen=[]
    for k in range(len(masks)):
        sets=[]
        for sign in [1,-1]:
            ix=np.flatnonzero(sign*effect[:,k]>0);order=np.lexsort((-sign*effect[ix,k],p[ix,k]));sets.append(ix[order[:100]])
        assert min(map(len,sets))==100
        out[k]=z[sets[0]].mean(0)-z[sets[1]].mean(0)
        if return_genes:chosen.append(tuple(np.flatnonzero(keep)[s] for s in sets))
    return (out,chosen) if return_genes else out
def design(scores,train,model):
    """scores: batch x all cases x six programs, with fold-training gene z-scales."""
    mu=scores[:,train].mean(1,keepdims=True);sd=scores[:,train].std(1,keepdims=True);sd=np.where(sd>1e-12,sd,1)
    z=(scores-mu)/sd
    if model=='Intercept':f=z[:,:,:0]
    elif model=='Global':f=z[:,:,:1]
    elif model=='Global+Mean5':f=np.stack([z[:,:,0],z[:,:,1:].mean(2)],2)
    elif model=='Global+All5':f=z
    else:f=z[:,:,[0,NAMES.index(model.split('+',1)[1])]]
    fmu=f[:,train].mean(1,keepdims=True);fsd=f[:,train].std(1,keepdims=True);fsd=np.where(fsd>1e-12,fsd,1)
    f=(f-fmu)/fsd
    return np.concatenate([np.ones((*f.shape[:2],1)),f],2),dict(program_mean=mu[:,0],program_SD=sd[:,0],feature_mean=fmu[:,0],feature_SD=fsd[:,0])
def ridge_logistic(X,y,lam):
    """Batch Newton optimization: SUM binomial NLL + lam/2 ||slopes||².
    Intercept unpenalized; no tuning. Backtracking prevents numerical divergence.
    """
    b,n,p=X.shape;beta=np.zeros((b,p));beta[:,0]=special.logit(np.clip(y.mean(1),1e-6,1-1e-6));pen=np.eye(p)*lam;pen[0,0]=0
    def obj(v):
        eta=np.einsum('bnp,bp->bn',X,v);return (np.logaddexp(0,eta)-y*eta).sum(1)+.5*np.einsum('bp,pq,bq->b',v,pen,v)
    for iteration in range(60):
        eta=np.einsum('bnp,bp->bn',X,beta);pr=special.expit(eta);grad=np.einsum('bnp,bn->bp',X,pr-y)+beta@pen
        if np.max(abs(grad))<1e-9:break
        w=np.maximum(pr*(1-pr),1e-10);h=np.einsum('bni,bn,bnj->bij',X,w,X)+pen[None,:,:]
        step=np.linalg.solve(h,grad[...,None])[...,0];before=obj(beta);alpha=np.ones(b)
        for line in range(18):
            candidate=beta-alpha[:,None]*step;ok=obj(candidate)<=before+1e-12
            if ok.all():break
            alpha[~ok]*=.5
        beta=candidate
    else:raise RuntimeError('ridge optimization failed to converge')
    return beta,iteration+1
def predict_fold(scores,y,train,held,model,lam):
    X,scale=design(scores,train,model);beta,it=ridge_logistic(X[:,train],y[:,train],lam)
    p=special.expit(np.einsum('bp,bp->b',X[:,held],beta));return p,beta,scale,it
def losses(y,p):
    p=np.clip(p,1e-12,1-1e-12);return -(y*np.log(p)+(1-y)*np.log1p(-p))
def auc(y,p):
    a=p[np.asarray(y)==1];b=p[np.asarray(y)==0];return float(((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])).mean())
def metrics(y,p):return dict(log_loss=float(losses(y,p).mean()),brier=float(((p-y)**2).mean()),auc=auc(y,p))
def g_effect(nr,cr):
    nr=np.asarray(nr);cr=np.asarray(cr);n=len(nr)+len(cr)
    if min(len(nr),len(cr))<2:return np.nan
    sd=np.sqrt(((len(nr)-1)*nr.var(ddof=1)+(len(cr)-1)*cr.var(ddof=1))/(n-2))
    return (1-3/(4*n-9))*(nr.mean()-cr.mean())/sd if sd>0 else np.nan
def load_external():
    # Read original local parsing code without calling its writer/main.
    path=P1/'scripts/05_prepare_and_validate_bulk_cohorts.py'
    if not path.exists():path=ROOT/'07_返修_20261028/legacy_scripts/Phase1/05_prepare_and_validate_bulk_cohorts.py'
    spec=importlib.util.spec_from_file_location('bulk_readonly',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=P1
    hgnc=pd.read_csv(P1/'06_independent_validation_prep/hgnc_complete_set.txt',sep='\t',low_memory=False)
    mapping=hgnc.dropna(subset=['ensembl_gene_id','symbol']).drop_duplicates('ensembl_gene_id').set_index('ensembl_gene_id').symbol
    raw=pd.read_csv(P1/'07_gse209746_validation/source/GSE209746_geneExp_n_114_ensembl.csv.gz',index_col=0);raw.index=raw.index.astype(str).str.split('.').str[0]
    ex209=m.log_cpm(m.aggregate_to_symbols(raw,mapping,'sum'))
    pl=m.read_platform(P1/'08_gse87211_validation/source/GPL13497_platform_data.txt');pl['GENE_SYMBOL']=pl.GENE_SYMBOL.astype(str).str.split(' /// ').str[0].str.upper();probe=pl.drop_duplicates('ID').set_index('ID').GENE_SYMBOL
    ex872=m.aggregate_to_symbols(m.read_series_matrix(P1/'08_gse87211_validation/source/GSE87211_series_matrix.txt.gz'),probe,'mean')
    genes=read(P1/'05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv');genes.gene_symbol=genes.gene_symbol.astype(str).str.upper()
    datasets={}
    for ds,expr,di,group,poor in [('GSE209746',ex209,'07_gse209746_validation','response_standardized','incomplete_response'),('GSE87211',ex872,'08_gse87211_validation','response_endpoint','non_pCR')]:
        meta=read(P1/di/f'{ds}_metadata_prepared.tsv');meta=meta[meta.usable_for_validation].set_index('sample_id').sort_index();expr.index=expr.index.astype(str).str.upper();expr=expr.loc[:,meta.index]
        assert meta.index.is_unique and expr.columns.is_unique
        if ds=='GSE87211':assert meta.patient.nunique()==len(meta)
        union=sorted(set(genes.gene_symbol)&set(expr.index));expr=expr.loc[union];W=np.zeros((len(union),6));lookup={g:i for i,g in enumerate(union)}
        cov=[]
        for k,typ in enumerate(TYPES):
            for direction,sign in [('NR_high',1),('CR_high',-1)]:
                ss=genes[genes.source_cell_type.eq(typ)&genes.program_direction.eq(direction)].gene_symbol;represented=sorted(set(ss)&set(union));assert len(represented)>20
                for g in represented:W[lookup[g],k]+=sign/len(represented)
                cov.append(dict(dataset=ds,program=NAMES[k],direction=direction,locked=len(ss),represented=len(represented)))
        y=meta[group].eq(poor).to_numpy(int);datasets[ds]=dict(expr=expr.to_numpy(float),genes=np.array(union),weights=W,meta=meta,y=y,coverage=cov)
    return datasets
def external_fold_scores(expr,weights,train):
    mu=expr[:,train].mean(1);sd=expr[:,train].std(1,ddof=0);z=np.divide(expr-mu[:,None],sd[:,None],out=np.zeros_like(expr),where=sd[:,None]>0)
    return z.T@weights
