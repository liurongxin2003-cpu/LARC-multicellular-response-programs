"""Observed metadata, single-covariate sensitivity, read-depth stress test, spatial QC.
No absent clinical field is imputed. No bin/cell is a response replicate.
"""
from common import *
import re,time
import statsmodels.api as sm
start=time.time();A=R/'analysis';availability=[];results=[]
def available(ds,name,series,n,unit,source,use,reason=''):
    if series is None:missing=n;unique=0;examples=''
    else:
        s=pd.Series(series).replace(['NA','N/A','unknown','Unknown','not available',''],np.nan);missing=int(s.isna().sum());unique=int(s.nunique());examples=' | '.join(s.dropna().astype(str).unique()[:7])
    availability.append(dict(dataset=ds,variable=name,analysis_unit=unit,eligible_units=n,missing_n=missing,missing_fraction=missing/n if n else np.nan,unique_values=unique,use_status=use,reason=reason,examples=examples,absolute_source_path=str(source)))
def contrast(d,scope='score association'):
    cr=d[d.response.eq('CR')].score.to_numpy();nr=d[d.response.eq('NR')].score.to_numpy()
    return dict(n=len(d),n_CR=len(cr),n_PR=int(d.response.eq('PR').sum()),n_NR=len(nr),effect=float(nr.mean()-cr.mean()) if len(cr) and len(nr) else np.nan,hedges_g=g_effect(nr,cr),estimable=min(len(cr),len(nr))>=2,unit='raw response-associated score difference',scope=scope)
def adjusted(d,variables,tag):
    needed=['score','response']+variables;q=d.dropna(subset=needed).copy()
    if q.response.nunique()<2 or min(q.response.eq('CR').sum(),q.response.eq('NR').sum())<2:return None
    out=contrast(q);out.update(analysis='covariate_adjusted',condition=tag,variable='+'.join(variables),complete_case=True)
    # Categorical response: PR has its own coefficient; no equal-spacing assumption.
    design=pd.DataFrame({'intercept':np.ones(len(q)),'NR':q.response.eq('NR').astype(float)},index=q.index)
    if q.response.eq('PR').any():design['PR']=q.response.eq('PR').astype(float)
    baseline=sm.OLS(q.score,design).fit(cov_type='HC3',use_t=True);out['unadjusted_beta_same_cases']=float(baseline.params['NR'])
    for v in variables:
        x=q[v]
        if x.nunique()<2:
            out.update(adjusted_beta=np.nan,CI_low=np.nan,CI_high=np.nan,estimable=False,reason='covariate constant in complete-case subset');return out
        if pd.api.types.is_numeric_dtype(x):design[v]=(x-x.mean())/x.std(ddof=0)
        else:
            z=pd.get_dummies(x.astype(str),prefix=v,drop_first=True,dtype=float);design=pd.concat([design,z],axis=1)
    if len(q)<=design.shape[1]+2 or np.linalg.matrix_rank(design.to_numpy())<design.shape[1]:
        out.update(adjusted_beta=np.nan,CI_low=np.nan,CI_high=np.nan,estimable=False,reason='insufficient residual degrees of freedom or rank deficiency');return out
    fit=sm.OLS(q.score,design).fit(cov_type='HC3',use_t=True);ci=fit.conf_int().loc['NR']
    out.update(adjusted_beta=float(fit.params['NR']),CI_low=float(ci.iloc[0]),CI_high=float(ci.iloc[1]),parameters=design.shape[1],residual_df=float(fit.df_resid),effect_change=float(fit.params['NR']-baseline.params['NR']),estimable=True,uncertainty='HC3 t interval conditional on generated scores; does not include feature selection/fitting uncertainty; no adjusted significance claim')
    return out
counts,meta,extreme_ids,y,raw_extreme=cnp_data();covpath=P1/'02_scRNA_matrix/CNP0004138_scRNA_cell_metadata_clean.tsv'
cells=read(covpath);cells=cells[cells.timepoint.eq('PRET')&cells.response_standardized.isin(['CR','PR','NR'])].copy()
cells['library_number_within_patient']=cells.barcode.str.extract(r'-lib-(\d+)')[0]
base=meta[meta.cell_type.eq('All_cells')&meta.response_standardized.isin(['CR','PR','NR'])].set_index('patient_id').sort_index();ids=base.index
scores=read(LATEST/'analysis/crossfit_patient_scores.tsv');pbdepth=pd.Series(counts.to_numpy(dtype=np.float64).sum(0),index=counts.columns);pbgenes=(counts>0).sum(0)
comp=meta[meta.patient_id.isin(ids)].pivot(index='patient_id',columns='cell_type',values='cell_count').reindex(ids)
patient=base[['response_standardized','cell_count']].rename(columns={'response_standardized':'response','cell_count':'all_cell_count'}).copy()
for typ,name in zip(TYPES[1:],NAMES[1:]):patient['fraction_'+name]=comp[typ]/comp.All_cells
patient['source_samples']=cells.groupby('patient_id').sample_id.nunique();patient['source_libraries_within_patient']=cells.groupby('patient_id').library_number_within_patient.nunique()
cnp=[]
for typ,name in zip(TYPES,NAMES):
    m=meta[meta.cell_type.eq(typ)&meta.patient_id.isin(ids)].copy();m=m.merge(patient,left_on='patient_id',right_index=True)
    m['score']=m.patient_id.map(scores[scores.program.eq(name)].set_index('patient_id').score)
    m['counts']=m.pseudobulk_id.map(pbdepth);m['detected_genes']=m.pseudobulk_id.map(pbgenes);m['counts_per_cell']=m.counts/m.cell_count;m['own_fraction']=m.cell_count/m.all_cell_count
    cc=cells if typ=='All_cells' else cells[cells.broad_cell_type.eq(typ)]
    group=cc.groupby('patient_id');assert (m.set_index('patient_id').cell_count==group.size().reindex(m.patient_id).to_numpy()).all()
    check=m.set_index('patient_id').counts-group.total_counts.sum().reindex(m.patient_id).to_numpy();assert abs(check).max()<1e-6
    m['mean_cell_detected_genes']=m.patient_id.map(group.n_genes_by_nnz.mean());m['median_cell_detected_genes']=m.patient_id.map(group.n_genes_by_nnz.median());m['program']=name
    for v in ['cell_count','counts','counts_per_cell','detected_genes','mean_cell_detected_genes']:m['log_'+v]=np.log1p(m[v])
    cnp.append(m)
cnp=pd.concat(cnp,ignore_index=True);save(cnp,'M5_patient_covariates.tsv');patient.to_csv(A/'M5_patient_composition.tsv',sep='\t')
for v in ['age','sex','MSI','AJCC stage','individual treatment regimen','sequencing batch']:
    available('CNP',v,None,25,'patient',covpath,'unavailable','absent from the actual frozen patient/cell/pseudobulk metadata; not imputed')
available('CNP','patient-specific technical platform ID',None,25,'patient',covpath,'unavailable','scRNA-seq data type is known, but a per-patient platform/run field is not supplied')
available('CNP','source sample identifier',cells.drop_duplicates('patient_id').set_index('patient_id').sample_id.reindex(ids),25,'patient',covpath,'identifier only','source IDs are retained, not fitted as patient fixed effects')
for v in ['patient_id','all_cell_count','source_samples','source_libraries_within_patient',*['fraction_'+n for n in NAMES[1:]]]:
    available('CNP',v,patient.reset_index()[v] if v=='patient_id' else patient[v],25,'patient',covpath,'identifier only' if v=='patient_id' else ('constant/not batch' if patient[v].nunique()<2 else 'tested'), 'Within-patient library numbers are not shared sequencing-batch IDs' if 'librar' in v else '')
for v in ['cell_count','counts','counts_per_cell','detected_genes','mean_cell_detected_genes']:
    available('CNP',v,cnp[cnp.program.eq('Global')][v],25,'patient within program',covpath,'tested','Each program analysis has 25 patients, not 150 independent pseudobulks')
covariates=['log_cell_count','log_counts','log_counts_per_cell','log_detected_genes','log_mean_cell_detected_genes',*['fraction_'+n for n in NAMES[1:]]]
correlations=[]
for name,d in cnp.groupby('program',sort=False):
    results.append(dict(dataset='CNP',program=name,analysis='unadjusted',condition='all labeled',**contrast(d)))
    for v in covariates:
        rho=stats.spearmanr(d.score,d[v]).statistic if d[v].nunique()>1 else np.nan
        rankscore=stats.rankdata(d.score);rankcov=stats.rankdata(d[v]);Z=np.column_stack([np.ones(len(d)),d.response.eq('PR'),d.response.eq('NR')]).astype(float)
        a=rankscore-Z@np.linalg.lstsq(Z,rankscore,rcond=None)[0];b=rankcov-Z@np.linalg.lstsq(Z,rankcov,rcond=None)[0];within=np.corrcoef(a,b)[0,1] if b.std()>1e-9 else np.nan
        correlations.append(dict(program=name,covariate=v,n=len(d),overall_spearman_rho=rho,response_adjusted_rank_rho=within,scope='descriptive generated-score correlation; no independent P value asserted'))
        for subset,tag in [(d,'all25_categorical_response'),(d[d.response.ne('PR')],'extremes11')]:
            z=adjusted(subset,[v],tag)
            if z:results.append(dict(dataset='CNP',program=name,**z))
    # Two technical variables with distinct interpretation, rather than many correlated nuisances.
    z=adjusted(d,['log_cell_count','log_counts_per_cell'],'two_technical_covariates')
    if z:results.append(dict(dataset='CNP',program=name,**z))
    for floor in [20,50,100,200,500]:
        q=d[d.cell_count>=floor];results.append(dict(dataset='CNP',program=name,analysis='minimum_cell_filter',condition=f'cells>={floor}',threshold=floor,**contrast(q),reason='fixed cross-fitted scores retained; subset assessment, no claim of a refitted M1 null'))
    for v in ['counts','mean_cell_detected_genes','fraction_Malignant']:
        cutoff=float(d[v].median())
        for high in [False,True]:
            q=d[d[v].ge(cutoff) if high else d[v].lt(cutoff)]
            results.append(dict(dataset='CNP',program=name,analysis='quality_or_composition_stratum',condition=('higher' if high else 'lower')+' '+v,cutoff=cutoff,**contrast(q),reason='outcome-blind median split; NA if fewer than two CR or NR patients; scores not refitted'))
save(correlations,'M5_score_covariate_relationships.tsv')
print('CNP covariate adjustment and stratification completed',flush=True)
# Existing external feature cache is independent of external outcome in gene selection/scaling.
for ds in ['GSE209746','GSE87211']:
    mm=read(R/'source_data'/f'{ds}_eligible_metadata.tsv').set_index('sample_id');inp=read(A/f'M4_{ds}_held_feature_inputs.tsv').set_index('patient_id')
    mm=mm.reindex(inp.index);mm['response']=np.where(inp.poor_response.eq(1),'NR','CR')
    clinical=pd.DataFrame(index=mm.index)
    if ds=='GSE209746':
        clinical['MSI_status']=mm.msi_status.where(mm.msi_status.isin(['MSS','MSI']))
        for v in ['age','sex','AJCC stage','individual treatment regimen','sequencing batch']:available(ds,v,None,len(mm),'source pretreatment case ID',P1/'07_gse209746_validation/GSE209746_metadata_prepared.tsv','unavailable','not in the supplied prepared metadata; source cohort treatment setting is not an individual regimen field')
        available(ds,'MSI_status',clinical.MSI_status,len(mm),'source pretreatment case ID',P1/'07_gse209746_validation/GSE209746_metadata_prepared.tsv','tested','Missing MSI remains missing; sparse MSI group limits precision')
        for v in ['platform_id','instrument_model']:available(ds,v,mm[v],len(mm),'source pretreatment case ID',P1/'07_gse209746_validation/GSE209746_metadata_prepared.tsv','constant; no within-cohort adjustment','one observed platform/instrument in eligible cohort; not a sequencing-batch label')
        clinicalvars=['MSI_status'];joint=[]
    else:
        clinical['age']=pd.to_numeric(mm.age,errors='coerce');clinical['sex_source_gender']=mm.gender.where(mm.gender.isin(['F','M']))
        for new,old in [('baseline_T','depth of invasion before rct'),('baseline_N','lymph node metastasis before rct'),('baseline_M','metastasis before rct')]:clinical[new]=mm[old].where(mm[old].notna()).astype('string')
        clinical['regimen']=mm['preoperative radiochemotherapy (rct)'].where(mm['preoperative radiochemotherapy (rct)'].notna())
        clinical['oxaliplatin_exposure']=clinical.regimen.map(lambda s:float('Oxaliplatin' in s) if isinstance(s,str) else np.nan)
        clinical['reported_KRAS_nonWT']=mm['kras mutation'].map(lambda s:np.nan if pd.isna(s) else ('WT' if str(s)=='WT' else 'reported non-WT'))
        clinicalvars=list(clinical.columns);joint=['age','sex_source_gender','oxaliplatin_exposure']
        src=P1/'08_gse87211_validation/GSE87211_metadata_prepared.tsv'
        available(ds,'patient identifier',mm.patient,len(mm),'patient',src,'identifier only','201 unique patients among 201 eligible tumor samples')
        available(ds,'source platform GPL13497',pd.Series(['GPL13497']*len(mm)),len(mm),'patient',P1/'08_gse87211_validation/source/GPL13497_platform_data.txt','constant; no within-cohort adjustment','platform is documented by the original probe annotation; not invented from expression totals')
        for v in clinicalvars:available(ds,v,clinical[v],len(mm),'patient with one eligible tumor sample',src,'tested','Source baseline T/N/M components are tested separately; no AJCC stage was invented')
        for v in ['MSI','AJCC stage','sequencing batch']:available(ds,v,None,len(mm),'patient',src,'unavailable','not supplied as a field; baseline components do not uniquely establish a versioned AJCC stage')
        for v in ['depth of invasion after rct','lymph node metastasis after rct','metastasis after rct']:available(ds,v,mm[v],len(mm),'patient',src,'excluded from adjustment','post-treatment fields define/overlap the response endpoint; adjusting for them would leak outcome information')
    for name in NAMES:
        d=pd.DataFrame({'score':inp[name],'response':mm.response}).join(clinical)
        results.append(dict(dataset=ds,program=name,analysis='unadjusted',condition='all eligible cases',**contrast(d)))
        for v in clinicalvars:
            q=adjusted(d,[v],'clinical_single_covariate')
            if q:results.append(dict(dataset=ds,program=name,**q))
        if joint:
            q=adjusted(d,joint,'clinical_age_sex_oxaliplatin_joint')
            if q:results.append(dict(dataset=ds,program=name,**q))
        if ds=='GSE209746':
            q=d[d.MSI_status.eq('MSS')];results.append(dict(dataset=ds,program=name,analysis='clinical_restriction',condition='MSS_only',**contrast(q)))
        else:
            for tag,mask in [('baseline_M0_only',clinical.baseline_M.eq('0.0')),('exclude_cetuximab',~clinical.regimen.fillna('').str.contains('Cetuximab'))]:results.append(dict(dataset=ds,program=name,analysis='clinical_restriction',condition=tag,**contrast(d[mask.fillna(False)])))
    clinical.assign(response=mm.response).to_csv(A/f'M5_{ds}_clinical_covariates.tsv',sep='\t')
    print('External clinical adjustments',ds,'completed',flush=True)
# New depth-matching stress test: full training-only selection/scaling, no M1 null rerun.
depthrows=[];depthsettings=[];depthsummary=[];rng=np.random.default_rng(SEED)
for k,(typ,name) in enumerate(zip(TYPES,NAMES)):
    m=meta[meta.cell_type.eq(typ)&meta.response_standardized.isin(['CR','PR','NR'])].copy()
    m=pd.concat([m[m.response_standardized.isin(['CR','NR'])].sort_values('patient_id'),m[m.response_standardized.eq('PR')].sort_values('patient_id')]).reset_index(drop=True)
    raw=counts[m.pseudobulk_id].to_numpy(float);assert np.max(abs(raw-np.rint(raw)))<1e-7
    raw=np.rint(raw).astype(np.int64);libs=raw.sum(0);yy=m.response_standardized.eq('NR').to_numpy(bool)[None,:]
    for rep in range(5):
        oof=np.zeros(25);pr=[]
        for held in range(11):
            train=np.delete(np.arange(11),held);target=int(libs[train].min());thinned=np.empty_like(raw)
            for patient in range(25):
                # Independent streams ensure held-out count changes cannot alter training draws.
                gen=np.random.default_rng(SEED+100000*k+1000*held+100*rep+patient);thinned[:,patient]=gen.binomial(raw[:,patient],min(1.,target/max(libs[patient],1)))
            ss=select_score(thinned,train,np.arange(25),yy)[0];oof[held]=ss[held];pr.append(ss[11:])
            depthsettings.append(dict(program=name,replicate=rep,omitted_patient=m.patient_id.iloc[held],training_minimum_count_target=target,held_original_count=int(libs[held]),held_thinned_count=int(thinned[:,held].sum()),held_below_training_target=bool(libs[held]<target)))
        oof[11:]=np.mean(pr,axis=0)
        for i,row in m.iterrows():depthrows.append(dict(program=name,replicate=rep,patient_id=row.patient_id,response=row.response_standardized,score=oof[i]))
        q=pd.DataFrame({'response':m.response_standardized,'score':oof});z=contrast(q);depthsummary.append(dict(program=name,replicate=rep,**z,uncertainty='five independent Monte Carlo thinning replicates; their range is not a confidence interval'))
    print('Depth-thinning full-fold sensitivity',name,'5 replicates completed',flush=True)
save(depthrows,'M5_depth_thinning_patient_scores.tsv');save(depthsettings,'M5_depth_thinning_settings.tsv');save(depthsummary,'M5_depth_thinning_summary.tsv')
for r in depthsummary:results.append(dict(dataset='CNP',analysis='training_only_depth_thinning',condition='expected depth capped at minimum training library',**r))
# Four source-study sections stay four sections, not four verified independent patients.
P2=P/'LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization';P5=P/'LARC_NAC_CNP0004138_Phase5_TargetedStrengthening'
sets=read(P1/'05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv');sets.gene_symbol=sets.gene_symbol.astype(str).str.upper()
represented_union=set()
for sec in ['ST_CR1','ST_CR2','ST_NR1','ST_NR2']:
    with np.load(P2/f'03_stereoseq_matrix/{sec}_analysis_gene_counts.npz') as z:represented_union.update(z['genes'].astype(str))
spgenes=[];spqc=[]
for section in ['ST-CR1','ST-CR2','ST-NR1','ST-NR2']:
    key=section.replace('-','_');d=pd.read_parquet(P5/f'03_scrna_spatial_reference_mapping/{key}_reference_mapping.parquet')
    malignant=d.mapped_cell_type.eq('Malignant_epithelial').to_numpy();ng=d.detected_genes.to_numpy()
    masks={'all_bins':np.ones(len(d),bool),'malignant_only':malignant,'genes>=100':ng>=100,'genes>=300':ng>=300,'genes>=500':ng>=500,'malignant_and_genes>=300':malignant&(ng>=300)}
    with np.load(P2/f'03_stereoseq_matrix/{key}_analysis_gene_counts.npz') as z:
        assert np.array_equal(z['bins'].astype(str),d.index.astype(str).to_numpy());genes=z['genes'].astype(str);a=z['counts'];total=z['total_counts'];lookup={g:i for i,g in enumerate(genes)}
        for scope,mask in masks.items():
            bins=int(mask.sum());lib=float(total[mask].sum());valid=bins>=20 and lib>0
            spqc.append(dict(section=section,scope=scope,total_bins=len(d),retained_bins=bins,retained_fraction=bins/len(d),total_counts=lib,median_detected_genes=float(np.median(ng[mask])) if bins else np.nan,malignant_assigned_fraction=float(malignant[mask].mean()) if bins else np.nan,estimable=valid,unit='section; bins are descriptive observations'))
            expr=np.log2(a[:,mask].sum(1,dtype=np.float64)/max(lib,1)*1e6+1) if valid else np.full(len(genes),np.nan)
            for typ,name in zip(TYPES,NAMES):
                for direction in ['NR_high','CR_high']:
                    gl=sets[sets.source_cell_type.eq(typ)&sets.program_direction.eq(direction)].gene_symbol
                    gl=gl[gl.isin(represented_union)]
                    for gene in gl:spgenes.append(dict(section=section,scope=scope,program=name,direction=direction,gene=gene,logCPM=expr[lookup[gene]] if gene in lookup else (0. if valid else np.nan),valid=valid))
    print('Spatial QC/count reaggregation',section,flush=True)
sp=pd.DataFrame(spgenes);out=[]
for scope,q in sp.groupby('scope'):
    if not q.valid.all():
        for name in NAMES:results.append(dict(dataset='Stereo-seq',program=name,analysis='spatial_QC_subset',condition=scope,n=4,effect=np.nan,estimable=False,reason='one or more sections retained fewer than 20 bins; no zero imputation'))
        continue
    q=q.copy();q['z']=q.groupby(['program','direction','gene']).logCPM.transform(lambda x:(x-x.mean())/x.std(ddof=0) if x.std(ddof=0)>0 else x*0)
    sc=q.groupby(['section','program','direction']).z.mean().unstack('direction');sc['score']=sc.NR_high-sc.CR_high;sc=sc.reset_index();sc['scope']=scope
    out.append(sc)
    for name,g in sc.groupby('program'):
        g=g.copy();g['response']=np.where(g.section.str.contains('CR'),'CR','NR');v=contrast(g);v['unit']='raw section-score difference; four sections, not verified independent donors'
        results.append(dict(dataset='Stereo-seq',program=name,analysis='spatial_QC_subset',condition=scope,**v,uncertainty='descriptive section sensitivity only; no bin-based P or patient-level CI'))
save(pd.concat(out,ignore_index=True),'M5_spatial_subset_scores.tsv');save(spqc,'M5_spatial_QC_retention.tsv')
newbase=pd.concat(out).query("scope=='all_bins'").set_index(['section','program']).score
oldbase=read(LATEST/'analysis/malignant_subset_section_scores.tsv').query("scope=='all_bins'").set_index(['section','program']).response_associated_score
sp_error=float(np.max(abs(newbase.reindex(oldbase.index)-oldbase)));assert sp_error<1e-8,sp_error
writej(R/'qa/M5_spatial_baseline_reproduction.json',dict(max_error=sp_error,representation='same source-represented union as original; genes absent from every source section are excluded rather than zero-averaged',unchanged_primary_section_scores=True))
for field in ['retained_bins','total_counts','median_detected_genes','malignant_assigned_fraction']:available('Stereo-seq',field,pd.DataFrame(spqc).query("scope=='all_bins'")[field],4,'section',P5/'03_scrna_spatial_reference_mapping','descriptive section sensitivity','four source-study sections; donor independence incompletely documented; not a multivariable patient model')
for field in ['age','sex','individual regimen','independent donor ID for NR1/NR2']:available('Stereo-seq',field,None,4,'section',P2,'unavailable/partly unresolved','no invented mapping; retain section-level reporting')
availability=pd.DataFrame(availability);result=pd.DataFrame(results)
availability.to_csv(R/'M5_metadata_availability.tsv',sep='\t',index=False);availability.to_csv(A/'M5_metadata_availability.tsv',sep='\t',index=False)
result.to_csv(R/'M5_sensitivity_results.tsv',sep='\t',index=False);result.to_csv(A/'M5_sensitivity_results.tsv',sep='\t',index=False)
writej(R/'qa/M5_execution.json',dict(elapsed_seconds=time.time()-start,unique_CNP_patients=25,defining_CNP_patients=11,patient_program_rows=len(cnp),single_covariates=covariates,depth_replicates=5,section_QC_scopes=list(masks),rows=len(result),metadata_rows=len(availability),source_cell_count_and_count_sum_aggregation_verified=True,clinical_adjustments='GSE209746 MSI; GSE87211 age, gender, baseline T/N/M, regimen, oxaliplatin, reported KRAS plus age/sex/oxaliplatin joint',post_treatment_TNM_excluded=True,no_clinical_imputation=True,status='completed'))
print('M5 COMPLETE',len(result),'sensitivity rows;',len(availability),'availability rows; seconds',time.time()-start,flush=True)
