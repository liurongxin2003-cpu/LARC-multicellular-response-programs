"""Fixed, patient-level incremental comparison; no model/penalty selection."""
from common import *
import time,datetime
start=time.time();A=R/'analysis';Q=R/'qa'
protocol=dict(seed=SEED,primary_penalty=1.0,penalty_sensitivity=[.1,10.0],objective='sum binomial negative log likelihood + lambda/2 sum squared slopes; intercept unpenalized',primary_comparisons=MODELS[2:],primary_endpoint='Global-only minus augmented-model mean outer-held-out log loss; positive means improvement',secondary_metrics=['Brier loss','poor-response AUC'],discovery='11 patients, 6 CR-like/5 NR; training-only full gene selection, gene z scaling, program scaling and fitting within every outer fold',joint_discovery='Global plus equal-weight mean of five training-standardized compartment scores; two penalized slopes; no unrestricted six-score fit',external='fixed discovery gene lists; fold-training-only gene and program scales and logistic fit in each external cohort separately',uncertainty='20000 paired stratified patient/case bootstrap draws with fitted predictions fixed; descriptive ranges, not full-training uncertainty',permutation='all 462 extreme-label assignments repeat feature selection and model fitting; diagnostic complete no-label-association null, NOT conditional null of no increment given an associated Global score',stability='delete each discovery patient; rerun gene selection and complete leave-one-out model comparison on remaining ten; primary lambda only',created_before_execution=datetime.datetime.now(datetime.timezone.utc).isoformat())
prot=R/'M4_M5_analysis_protocol.json'
if not prot.exists():writej(prot,protocol)
counts,meta,ids,y,raw=cnp_data();n=len(ids)
masks=[]
for cr in itertools.combinations(range(n),6):
    m=np.ones(n,bool);m[list(cr)]=False;masks.append(m)
masks=np.array(masks);obs=int(np.flatnonzero((masks==y).all(1))[0]);B=len(masks)
print('M4 BEGIN',n,'patients;',B,'label assignments;',len(MODELS),'models;',LAMBDAS,flush=True)
cache=A/'M4_discovery_full_pipeline_features.npz'
if cache.exists():
    z=np.load(cache);scorecache=z['scores'];assert np.array_equal(z['patient_id'],ids) and np.array_equal(z['labels'],y)
else:
    scorecache=np.empty((n,B,n,6))
    for held in range(n):
        train=np.delete(np.arange(n),held)
        for k in range(6):scorecache[held,:,:,k]=select_score(raw[k],train,np.arange(n),masks)
        print('Selected/scaled discovery outer fold',held+1,'/11',flush=True)
    np.savez_compressed(cache,scores=scorecache,patient_id=np.array(ids),labels=y,assignments=masks)
# The newly required training-feature matrices retain the completed M1 rule.
old=read(LATEST/'analysis/crossfit_patient_scores.tsv');maxerror=0
for h,patient in enumerate(ids):
    for k,name in enumerate(NAMES):
        value=old[old.patient_id.eq(patient)&old.program.eq(name)].score.iloc[0];maxerror=max(maxerror,abs(scorecache[h,obs,h,k]-value))
assert maxerror<1e-8,maxerror
inputrows=[];paramrows=[];pred=np.empty((len(LAMBDAS),len(MODELS),B,n))
for h,patient in enumerate(ids):
    tr=np.delete(np.arange(n),h);sc=scorecache[h]
    X,scale=design(sc,tr,'Global+All5')
    for j,who in enumerate(ids):inputrows.append(dict(dataset='CNP',outer_held=patient,patient_id=who,role='test' if j==h else 'train',response='NR' if y[j] else 'CR',**{name:float(sc[obs,j,k]) for k,name in enumerate(NAMES)},**{'z_'+name:float(X[obs,j,k+1]) for k,name in enumerate(NAMES)}))
    for li,lam in enumerate(LAMBDAS):
        for mi,model in enumerate(MODELS):
            pp,beta,scale,it=predict_fold(sc,masks.astype(float),tr,h,model,lam);pred[li,mi,:,h]=pp
            if lam==1:
                paramrows.append(dict(dataset='CNP',outer_held=patient,model=model,n_train=len(tr),lambda_=lam,coefficients=json.dumps(beta[obs].tolist()),program_means=json.dumps(scale['program_mean'][obs].tolist()),program_SDs=json.dumps(scale['program_SD'][obs].tolist()),feature_mean=json.dumps(scale['feature_mean'][obs].tolist()),feature_SD=json.dumps(scale['feature_SD'][obs].tolist()),Newton_iterations=it))
save(inputrows,'M4_patient_model_inputs_discovery.tsv');save(paramrows,'M4_model_parameters.tsv')
np.savez_compressed(A/'M4_discovery_full_pipeline_predictions.npz',probability=pred,models=np.array(MODELS),lambdas=LAMBDAS,assignments=masks,observed_assignment=obs,patients=np.array(ids))
# A direct poisoning check: changing held-out expression/label cannot change training lists/scales.
leak=[];h=0;tr=np.delete(np.arange(n),h)
for k in range(6):
    one=masks[obs:obs+1].copy();sc,genes=select_score(raw[k],tr,np.arange(n),one,True)
    altered=raw[k].copy();altered[:,h]=altered[::-1,h]*17+1;changed=one.copy();changed[0,h]=~changed[0,h]
    sc2,genes2=select_score(altered,tr,np.arange(n),changed,True)
    err=float(np.max(abs(sc[:,tr]-sc2[:,tr])));same=all(np.array_equal(a,b) for a,b in zip(genes[0],genes2[0]));assert err<1e-10 and same
    leak.append(dict(program=NAMES[k],held_patient=ids[h],training_score_max_error=err,training_gene_lists_unchanged=same))
writej(Q/'M4_leakage_audit.json',dict(observed_held_score_matches_previous_M1_max_error=maxerror,poisoning=leak,not_used='previous cross-fitted score vectors were not used as a global fixed training matrix; each outer fold has its own full training-feature matrix'))
print('Discovery model fits complete; feature audit max error',maxerror,flush=True)
allpred=[];allmetrics=[];allcomparisons=[];nullrows=[]
def summarize(ds,patients,yy,models,pp):
    """pp lambda x model x case. All model comparisons use identical held-out cases."""
    rng=np.random.default_rng(SEED+len(patients));nr=np.flatnonzero(yy);cr=np.flatnonzero(1-yy);DRAWS=20000
    w0=rng.multinomial(len(cr),np.ones(len(cr))/len(cr),size=DRAWS);w1=rng.multinomial(len(nr),np.ones(len(nr))/len(nr),size=DRAWS)
    w=np.zeros((DRAWS,len(yy)));w[:,cr]=w0;w[:,nr]=w1;gb=models.index('Global');boots={}
    for li,lam in enumerate(LAMBDAS):
        for mi,model in enumerate(models):
            p=pp[li,mi];mm=metrics(yy,p)
            for i,patient in enumerate(patients):allpred.append(dict(dataset=ds,patient_id=patient,poor_response=int(yy[i]),model=model,lambda_=lam,held_out_probability=float(p[i]),log_loss=float(losses(yy[i],p[i])),brier_loss=float((p[i]-yy[i])**2)))
            extra={}
            if lam==1:
                bl=w@losses(yy,p)/len(yy);bb=w@((p-yy)**2)/len(yy)
                c=(p[nr,None]>p[None,cr])+.5*(p[nr,None]==p[None,cr]);ba=np.sum((w1@c)*w0,axis=1)/(len(nr)*len(cr));boots[model]=(bl,bb,ba)
                for key,vals in [('log_loss',bl),('brier',bb),('auc',ba)]:extra[key+'_bootstrap_low'],extra[key+'_bootstrap_high']=np.quantile(vals,[.025,.975])
            allmetrics.append(dict(dataset=ds,model=model,lambda_=lam,n=len(yy),n_good=len(cr),n_poor=len(nr),penalized_slopes=0 if model=='Intercept' else (1 if model=='Global' else (6 if model=='Global+All5' else 2)),**mm,**extra))
    primary=pp[LAMBDAS.index(1.)];base=primary[gb];base_loss=losses(yy,base)
    for mi,model in enumerate(models):
        if model in ['Intercept','Global']:continue
        d=base_loss-losses(yy,primary[mi]);bootg=boots['Global'][0]-boots[model][0];bootb=boots['Global'][1]-boots[model][1];boota=boots[model][2]-boots['Global'][2]
        allcomparisons.append(dict(dataset=ds,model=model,n=len(yy),lambda_=1.,delta_log_loss=float(d.mean()),delta_log_loss_bootstrap_low=float(np.quantile(bootg,.025)),delta_log_loss_bootstrap_high=float(np.quantile(bootg,.975)),delta_brier=float(np.mean((base-yy)**2-(primary[mi]-yy)**2)),delta_brier_bootstrap_low=float(np.quantile(bootb,.025)),delta_brier_bootstrap_high=float(np.quantile(bootb,.975)),delta_auc=auc(yy,primary[mi])-auc(yy,base),delta_auc_bootstrap_low=float(np.quantile(boota,.025)),delta_auc_bootstrap_high=float(np.quantile(boota,.975)),patients_with_lower_log_loss=int((d>0).sum()),patients_with_higher_log_loss=int((d<0).sum()),case_omission_gain_min=float(min((d.sum()-d)/(len(d)-1))),case_omission_gain_max=float(max((d.sum()-d)/(len(d)-1))),uncertainty='paired stratified bootstrap of fixed held-out predictions; descriptive, not full-training uncertainty'))
summarize('CNP',ids,y,MODELS,pred[:,:,obs,:])
li=LAMBDAS.index(1.);baseidx=MODELS.index('Global');null_loss=losses(masks[None,:,:],pred[li]).mean(2);gains=null_loss[baseidx][None,:]-null_loss
maxgain=gains[[MODELS.index(m) for m in MODELS[2:]]].max(0)
for model in MODELS[2:]:
    idx=MODELS.index(model);obsval=gains[idx,obs]
    for a in range(B):nullrows.append(dict(model=model,assignment=a,observed=a==obs,delta_log_loss=float(gains[idx,a]),conditional_global_null_p=float(np.mean(gains[idx]>=obsval-1e-12)) if a==obs else np.nan,max_six_global_null_p=float(np.mean(maxgain>=obsval-1e-12)) if a==obs else np.nan,null_scope='complete no-label-association null; not a conditional incremental-value test'))
save(nullrows,'M4_full_pipeline_permutation.tsv')
# Delete-one patient refitting: unlike a fixed-prediction bootstrap, retrains all steps.
paircache={};deletion=[]
for dropped in range(n):
    keep=np.delete(np.arange(n),dropped);prob=np.zeros((len(MODELS),len(keep)))
    for ih,held in enumerate(keep):
        key=tuple(sorted([dropped,held]));train=np.array([i for i in range(n) if i not in key])
        if key not in paircache:paircache[key]=np.stack([select_score(rr,train,np.arange(n),y[None,:].astype(bool))[0] for rr in raw],axis=1)
        sc=paircache[key][None,:,:]
        for mi,model in enumerate(MODELS):prob[mi,ih]=predict_fold(sc,y[None,:].astype(float),train,held,model,1.)[0][0]
    base=metrics(y[keep],prob[MODELS.index('Global')])
    for mi,model in enumerate(MODELS):
        mm=metrics(y[keep],prob[mi]);deletion.append(dict(dropped_patient=ids[dropped],dropped_response='NR' if y[dropped] else 'CR',model=model,n=10,**mm,delta_log_loss=base['log_loss']-mm['log_loss'],delta_auc=mm['auc']-base['auc']))
    print('Delete-one full refit',ids[dropped],flush=True)
save(deletion,'M4_delete_one_full_refit.tsv')
# Cohort-internal external comparisons do not pool discovery or constitute independent clinical validation.
ext=load_external();coverage=[]
for ds,d in ext.items():
    yy=d['y'];patients=d['meta'].index.to_numpy(str);nn=len(yy);cc=np.empty((nn,nn,6));prob=np.empty((len(LAMBDAS),len(EXTERNAL_MODELS),nn));coverage.extend(d['coverage'])
    for held in range(nn):
        tr=np.delete(np.arange(nn),held);cc[held]=external_fold_scores(d['expr'],d['weights'],tr)
        for li,lam in enumerate(LAMBDAS):
            for mi,model in enumerate(EXTERNAL_MODELS):prob[li,mi,held]=predict_fold(cc[held:held+1],yy[None,:].astype(float),tr,held,model,lam)[0][0]
    np.savez_compressed(A/f'M4_{ds}_features_predictions.npz',scores=cc,probabilities=prob,patients=patients,labels=yy,models=np.array(EXTERNAL_MODELS),lambdas=LAMBDAS)
    np.savez_compressed(A/f'M4_{ds}_expression.npz',expression=d['expr'],genes=d['genes'],weights=d['weights'],patients=patients)
    d['meta'].to_csv(R/'source_data'/f'{ds}_eligible_metadata.tsv',sep='\t')
    heldinputs=pd.DataFrame([dict(patient_id=patients[i],poor_response=int(yy[i]),**{name:cc[i,i,k] for k,name in enumerate(NAMES)}) for i in range(nn)])
    heldinputs.to_csv(A/f'M4_{ds}_held_feature_inputs.tsv',sep='\t',index=False)
    summarize(ds,patients,yy,EXTERNAL_MODELS,prob)
    print('External M4',ds,nn,'unique case IDs; poor',yy.sum(),flush=True)
save(coverage,'M4_external_gene_coverage.tsv');save(allpred,'M4_patient_predictions.tsv');save(allmetrics,'M4_model_performance.tsv');save(allcomparisons,'M4_model_comparisons.tsv')
res=pd.DataFrame(allcomparisons);dd=pd.DataFrame(deletion)
for i,row in res.iterrows():
    if row.dataset=='CNP':
        vals=dd[dd.model.eq(row.model)].delta_log_loss;res.loc[i,'delete_one_positive_count']=int((vals>0).sum());res.loc[i,'delete_one_gain_min']=vals.min();res.loc[i,'delete_one_gain_max']=vals.max()
        nul=pd.DataFrame(nullrows);r=nul[nul.model.eq(row.model)&nul.observed].iloc[0];res.loc[i,'complete_null_P']=r.conditional_global_null_p;res.loc[i,'complete_null_max_six_P']=r.max_six_global_null_p
res.to_csv(A/'M4_model_comparisons.tsv',sep='\t',index=False)
writej(Q/'M4_execution.json',dict(elapsed_seconds=time.time()-start,patients=ids,n_patients=11,assignments=B,primary_penalty=1,models=MODELS,external_models=EXTERNAL_MODELS,observed_assignment=obs,held_feature_match_error=maxerror,full_pipeline_model_fits=len(LAMBDAS)*len(MODELS)*B*11,delete_one_full_refits=11,source_matrices='real CNP patient pseudobulk counts and local processed external expression',status='completed'))
print('M4 RESULTS\n',res[['dataset','model','delta_log_loss','delta_log_loss_bootstrap_low','delta_log_loss_bootstrap_high','delta_auc','delete_one_positive_count']].to_string(index=False),flush=True)
print('M4 COMPLETE seconds',time.time()-start,flush=True)
