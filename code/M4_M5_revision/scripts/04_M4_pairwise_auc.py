"""Secondary discrimination diagnostic, added after pooled-LOO calibration check.
Same models and primary lambda. Both members of every NR/CR pair are excluded
from all training steps. Pairs are not treated as independent patients.
"""
from common import *
import time,datetime
start=time.time();A=R/'analysis'
writej(R/'qa/M4_secondary_metric_amendment.json',dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Pooled-LOO AUC of intercept-only model was 0 in every cohort because class-dependent training prevalence changes; it is not a calibrated discrimination reference',remedy='Report leave-pair-out AUC from a common fitted model for each held poor/good pair; keep all original pooled-LOO AUC values labelled diagnostic',unchanged=['all model formulas','lambda=1 primary','primary mean held-out log loss','all original predictions and loss comparisons'],not_model_selection=True))
counts,meta,ids,y,raw=cnp_data();rows=[];summary=[]
def run(ds,patients,yy,models,score_generator):
    cases0=np.flatnonzero(yy==0);cases1=np.flatnonzero(yy==1);concordance={m:np.zeros((len(cases1),len(cases0))) for m in models}
    total=len(cases0)*len(cases1)
    for i,poor in enumerate(cases1):
        for j,good in enumerate(cases0):
            tr=np.array([k for k in range(len(yy)) if k not in [poor,good]])
            score=score_generator(tr)[None,:,:]
            for model in models:
                if model=='Intercept':pr=np.repeat(yy[tr].mean(),2)
                else:
                    X,_=design(score,tr,model);beta,_=ridge_logistic(X[:,tr],yy[None,tr].astype(float),1.);pr=special.expit(X[0,[poor,good]]@beta[0])
                value=float(pr[0]>pr[1])+.5*float(pr[0]==pr[1]);concordance[model][i,j]=value
                rows.append(dict(dataset=ds,model=model,held_poor_case=patients[poor],held_good_case=patients[good],n_train=len(tr),p_poor=float(pr[0]),p_good=float(pr[1]),concordance=value))
        if (i+1)%10==0:print(ds,'LPO poor-case row',i+1,'/',len(cases1),flush=True)
    rng=np.random.default_rng(SEED+len(yy)+77);w0=rng.multinomial(len(cases0),np.ones(len(cases0))/len(cases0),size=20000);w1=rng.multinomial(len(cases1),np.ones(len(cases1))/len(cases1),size=20000)
    boot={m:np.sum((w1@v)*w0,axis=1)/total for m,v in concordance.items()}
    for model,c in concordance.items():
        gain=boot[model]-boot['Global'];summary.append(dict(dataset=ds,model=model,n_patients_or_source_cases=len(yy),held_pairs=total,leave_pair_out_AUC=float(c.mean()),AUC_bootstrap_low=float(np.quantile(boot[model],.025)),AUC_bootstrap_high=float(np.quantile(boot[model],.975)),delta_AUC_vs_Global=float(c.mean()-concordance['Global'].mean()),delta_AUC_bootstrap_low=float(np.quantile(gain,.025)),delta_AUC_bootstrap_high=float(np.quantile(gain,.975)),scope='patient/case bootstrap of fixed pair predictions; pairs share patients and are not independent replicates'))
    print(ds,'LPO COMPLETE',total,'pairs',flush=True)
run('CNP',ids,y,MODELS,lambda tr:np.stack([select_score(r,tr,np.arange(len(y)),y[None,:].astype(bool))[0] for r in raw],axis=1))
for ds in ['GSE209746','GSE87211']:
    z=np.load(A/f'M4_{ds}_expression.npz');pp=np.load(A/f'M4_{ds}_features_predictions.npz');expr=z['expression'];weights=z['weights'];yy=pp['labels'];patients=pp['patients']
    run(ds,patients,yy,EXTERNAL_MODELS,lambda tr:external_fold_scores(expr,weights,tr))
save(rows,'M4_leave_pair_out_predictions.tsv');save(summary,'M4_leave_pair_out_AUC.tsv')
writej(R/'qa/M4_LPO_execution.json',dict(seconds=time.time()-start,models_unchanged=True,primary_metric_unchanged=True,secondary=True,cohort_rows=len(summary),prediction_pair_rows=len(rows),unit='patient/case; not independent pairs'))
print(pd.DataFrame(summary)[['dataset','model','leave_pair_out_AUC','delta_AUC_vs_Global','delta_AUC_bootstrap_low','delta_AUC_bootstrap_high']].to_string(index=False),flush=True)
