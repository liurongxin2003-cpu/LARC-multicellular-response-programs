import os
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

R=Path(__file__).resolve().parents[1];P=R.parent/'06_研究资料';O=R/'analysis'
P1=P/'LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery';P2=P/'LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization';P4=P/'LARC_NAC_CNP0004138_Phase4_Robustness_ManuscriptPreparation';P5=P/'LARC_NAC_CNP0004138_Phase5_TargetedStrengthening'
names={'All_cells':'Global','Malignant_epithelial':'Malignant','CAF_Fibroblast':'CAF','Myeloid_macrophage':'Myeloid','CD8_T':'CD8 T','Endothelial':'Endothelial'}
def read(p):return pd.read_csv(p,sep='\t')
def save(d,n):d.to_csv(O/n,sep='\t',index=False)
def main():
    sets=read(P1/'05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv')
    sets['gene_symbol']=sets.gene_symbol.astype(str).str.upper()
    overlaps=[]
    for a,b in itertools.combinations(names,2):
        for direction in ['NR_high','CR_high','both']:
            aa=sets[sets.source_cell_type.eq(a)];bb=sets[sets.source_cell_type.eq(b)]
            if direction!='both':aa=aa[aa.program_direction.eq(direction)];bb=bb[bb.program_direction.eq(direction)]
            sa=set(aa.gene_symbol);sb=set(bb.gene_symbol)
            overlaps.append(dict(program_A=names[a],program_B=names[b],direction=direction,intersection=len(sa&sb),union=len(sa|sb),jaccard=len(sa&sb)/len(sa|sb)))
    save(pd.DataFrame(overlaps),'program_gene_overlap.tsv')
    corrs=[]
    specs=[('CNP apparent',P1/'05_response_state_discovery/CNP0004138_pseudobulk_locked_program_scores.tsv','patient_id','source_cell_type','response_standardized'),('GSE209746',P1/'07_gse209746_validation/GSE209746_locked_program_scores.tsv','sample_id','program_source_cell_type','response_standardized'),('GSE87211',P1/'08_gse87211_validation/GSE87211_locked_program_scores.tsv','sample_id','program_source_cell_type','response_endpoint')]
    for dataset,path,sample,program,group in specs:
        d=read(path);d=d[d[program].isin(names)]
        if dataset=='CNP apparent':d=d[d[group].isin(['CR','PR','NR'])]
        scores=d.pivot(index=sample,columns=program,values='resistance_score').rename(columns=names)
        metadata=d.drop_duplicates(sample).set_index(sample)[group].reindex(scores.index)
        for gr in ['ALL',*metadata.dropna().unique()]:
            sub=scores if gr=='ALL' else scores.loc[metadata.eq(gr)]
            for a,b in itertools.combinations(sub.columns,2):
                vals=sub[[a,b]].dropna();rho=stats.spearmanr(vals[a],vals[b]).statistic if len(vals)>2 else np.nan
                corrs.append(dict(dataset=dataset,group=gr,program_A=a,program_B=b,n=len(vals),rho=rho))
    cross=read(O/'crossfit_patient_scores.tsv');wide=cross.pivot(index='patient_id',columns='program',values='score');group=cross.drop_duplicates('patient_id').set_index('patient_id').response.reindex(wide.index)
    for gr in ['ALL','CR','PR','NR']:
        sub=wide if gr=='ALL' else wide.loc[group.eq(gr)]
        for a,b in itertools.combinations(sub.columns,2):corrs.append(dict(dataset='CNP crossfit',group=gr,program_A=a,program_B=b,n=len(sub),rho=stats.spearmanr(sub[a],sub[b]).statistic))
    save(pd.DataFrame(corrs),'program_correlations.tsv')
    meta=read(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv')
    counts=pd.read_parquet(P1/'04_pseudobulk/CNP0004138_PRET_patient_celltype_pseudobulk_matrix.parquet')
    depths=counts.sum(0);detected=(counts>0).sum(0)
    comp=read(P1/'03_celltype_annotation/CNP0004138_celltype_composition.tsv');comp=comp[comp.timepoint.eq('PRET')]
    technical=[];covrows=[]
    total=meta[meta.cell_type.eq('All_cells')].set_index('patient_id').cell_count
    for typ,label in names.items():
        d=meta[meta.cell_type.eq(typ)&meta.response_standardized.isin(['CR','PR','NR'])].copy()
        d['counts']=d.pseudobulk_id.map(depths);d['detected_genes']=d.pseudobulk_id.map(detected)
        d['compartment_fraction']=d.cell_count/d.patient_id.map(total)
        d=d.merge(cross[cross.program.eq(label)][['patient_id','score']],on='patient_id')
        d['program']=label
        covrows.append(d)
        for gr in ['CR','PR','NR']:
            f=d[d.response_standardized.eq(gr)]
            for var in ['cell_count','counts','detected_genes','compartment_fraction']:
                technical.append(dict(program=label,response=gr,variable=var,n=len(f),median=f[var].median(),min=f[var].min(),max=f[var].max()))
    cov=pd.concat(covrows);save(cov,'CNP_patient_technical_covariates.tsv');save(pd.DataFrame(technical),'CNP_covariate_group_summary.tsv')
    sensitivities=[]
    for program,d in cov.groupby('program'):
        y=stats.rankdata(d.score);x=stats.rankdata(d.response_standardized.map({'CR':0,'PR':1,'NR':2}))
        for var in ['cell_count','counts','detected_genes','compartment_fraction']:
            z=stats.rankdata(d[var]);design=np.c_[np.ones(len(d)),z]
            ry=y-design@np.linalg.lstsq(design,y,rcond=None)[0];rx=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
            corr=np.corrcoef(rx,ry)[0,1] if ry.std()>0 and rx.std()>0 else np.nan
            sensitivities.append(dict(program=program,covariate=var,partial_rank_correlation=corr,n=len(d),inference='descriptive; no independent crossfit CI/P asserted'))
    save(pd.DataFrame(sensitivities),'single_covariate_partial_rank_sensitivity.tsv')
    missing=[]
    for v in ['age','sex','stage','MSI','treatment regimen per patient','library batch','raw-droplet ambient RNA estimates','doublet labels']:
        missing.append(dict(dataset='CNP discovery',variable=v,status='not available in the frozen analysis metadata; not imputed'))
    save(pd.DataFrame(missing),'unavailable_covariates.tsv')
    old=read(P4/'05_leave_one_patient_stability/leave_one_patient_transfer_results.tsv')
    save(old[old.direction_support_count<4],'eight_failed_redefinitions.tsv')
    for fn,src in [('signature_size_sensitivity_results.tsv',P4/'04_signature_size_sensitivity/signature_size_sensitivity_results.tsv'),('comparator_removal_all_scenarios.tsv',P5/'02_known_program_nonredundancy/known_program_removal_sensitivity.tsv'),('comparator_module_provenance.tsv',P5/'02_known_program_nonredundancy/comparator_module_provenance.tsv'),('GSE209746_gene_coverage.tsv',P1/'07_gse209746_validation/GSE209746_program_gene_representation.tsv'),('GSE87211_gene_coverage.tsv',P1/'08_gse87211_validation/GSE87211_program_gene_representation.tsv')]:save(read(src),fn)
    print('Supporting audits complete',flush=True)
    spatial=[];subset=[];nullrows=[];spatialsummary=[]
    rng=np.random.default_rng(20261028)
    for section in ['ST-CR1','ST-CR2','ST-NR1','ST-NR2']:
        key=section.replace('-','_');d=pd.read_parquet(P5/f'03_scrna_spatial_reference_mapping/{key}_reference_mapping.parquet')
        subsetmask=d.mapped_cell_type.eq('Malignant_epithelial').to_numpy()
        spatial.append(dict(section=section,response='CR' if 'CR' in section else 'NR',n_bins=len(d),malignant_assigned_bins=int(subsetmask.sum()),malignant_assigned_fraction=float(subsetmask.mean()),mean_malignant_weight=d.weight_Malignant_epithelial.mean(),median_detected_genes=d.detected_genes.median()))
        with np.load(P2/f'03_stereoseq_matrix/{key}_analysis_gene_counts.npz') as z:
            assert np.array_equal(z['bins'].astype(str),d.index.astype(str).to_numpy())
            genes=z['genes'].astype(str);a=z['counts'];all_library=z['total_counts']
            for scope,mask in [('all_bins',np.ones(len(d),bool)),('malignant_assigned_bins',subsetmask)]:
                totals=a[:,mask].sum(1,dtype=np.float64);lib=all_library[mask].sum();log=np.log2(totals/max(lib,1)*1e6+1)
                for typ,label in names.items():
                    gs=sets[sets.source_cell_type.eq(typ)]
                    for direction in ['NR_high','CR_high']:
                        for gene in gs[gs.program_direction.eq(direction)].gene_symbol:
                            ix=np.flatnonzero(genes==gene)
                            if len(ix):subset.append(dict(section=section,scope=scope,program=label,direction=direction,gene=gene,logCPM=log[ix[0]],n_bins=int(mask.sum())))
        xy=d[['x','y']].to_numpy(int);tree=NearestNeighbors(n_neighbors=9,algorithm='kd_tree',n_jobs=1).fit(xy);neigh=tree.kneighbors(xy,return_distance=False)[:,1:]
        aggregate=d[['weight_Malignant_epithelial','weight_Myeloid_macrophage','weight_CD8_T','weight_Endothelial']].to_numpy().mean(1)[neigh].mean(1)
        malignant=d.weight_Malignant_epithelial.to_numpy()[neigh].mean(1)
        high=d.global_resistance_score.to_numpy()>=np.quantile(d.global_resistance_score,.8)
        minx,miny=xy.min(0);maxx,maxy=xy.max(0);stride=int((maxy-miny)/50)+1000
        keys=((xy[:,0]-minx)//50)*stride+(xy[:,1]-miny)//50;order=np.argsort(keys);sortedkeys=keys[order]
        effects={name:[] for name in ['aggregate','malignant']};naccept=0
        # Rigid translations retain each reference field's local geometry. No wrap, interpolation, or bin shuffling.
        shifts=set();attempts=0
        while naccept<199 and attempts<4000:
            attempts+=1
            dx=int(rng.integers(-max(2,(maxx-minx)//200),max(3,(maxx-minx)//200)+1));dy=int(rng.integers(-max(2,(maxy-miny)//200),max(3,(maxy-miny)//200)+1))
            if (dx,dy)==(0,0) or (dx,dy) in shifts:continue
            shifts.add((dx,dy));target=keys+dx*stride+dy;pos=np.searchsorted(sortedkeys,target);pos=np.minimum(pos,len(keys)-1);good=sortedkeys[pos]==target
            # Check coordinates separately to exclude packed-key boundary aliases.
            idx=order[pos];good&=(xy[idx,0]==xy[:,0]+50*dx)&(xy[idx,1]==xy[:,1]+50*dy)
            if good.mean()<.5 or (good&high).sum()<.5*high.sum() or (good&~high).sum()<.5*(~high).sum():continue
            naccept+=1
            for name,field in [('aggregate',aggregate),('malignant',malignant)]:
                val=field[idx[good]];h=high[good];effect=val[h].mean()-val[~h].mean();effects[name].append(effect)
                nullrows.append(dict(section=section,metric=name,iteration=naccept,dx=50*dx,dy=50*dy,overlap_fraction=good.mean(),enrichment=effect))
        for name,field in [('aggregate',aggregate),('malignant',malignant)]:
            obs=field[high].mean()-field[~high].mean();nul=np.asarray(effects[name]);p=(1+(nul>=obs).sum())/(len(nul)+1)
            spatialsummary.append(dict(section=section,metric=name,observed_enrichment=obs,n_translations=len(nul),translation_p_one_sided=p,null_median=np.median(nul),null_q025=np.quantile(nul,.025),null_q975=np.quantile(nul,.975),boundary='within-section spatial compatibility; overlap-conditioned rigid-shift null, no patient-level inference'))
        print('Spatial',section,'translations',naccept,flush=True)
    save(pd.DataFrame(spatial),'spatial_composition.tsv');save(pd.DataFrame(nullrows),'spatial_translation_null.tsv');save(pd.DataFrame(spatialsummary),'spatial_translation_summary.tsv')
    sd=pd.DataFrame(subset)
    # Match the frozen primary spatial convention: log2(CPM+1), union gene
    # representation with absent source genes represented by zero, ddof=0.
    filled=[]
    for (scope,program,direction,gene),g in sd.groupby(['scope','program','direction','gene']):
        f=g.set_index('section').reindex(['ST-CR1','ST-CR2','ST-NR1','ST-NR2'])
        f['logCPM']=f.logCPM.fillna(0);f['scope']=scope;f['program']=program;f['direction']=direction;f['gene']=gene
        filled.append(f.reset_index())
    sd=pd.concat(filled,ignore_index=True);save(sd,'malignant_subset_gene_logCPM.tsv')
    sd['z']=sd.groupby(['scope','program','direction','gene']).logCPM.transform(lambda s:(s-s.mean())/s.std(ddof=0) if s.std(ddof=0)>0 else s*0)
    sc=sd.groupby(['section','scope','program','direction']).z.mean().unstack('direction');sc['response_associated_score']=sc.NR_high-sc.CR_high
    save(sc.reset_index(),'malignant_subset_section_scores.tsv')
    effects=[]
    for (scope,program),f in sc.reset_index().groupby(['scope','program']):
        cr=f[f.section.str.contains('CR')].response_associated_score;nr=f[f.section.str.contains('NR')].response_associated_score
        effects.append(dict(scope=scope,program=program,NR_minus_CR=nr.mean()-cr.mean(),both_NR_above_both_CR=bool(nr.min()>cr.max())))
    save(pd.DataFrame(effects),'malignant_subset_effects.tsv')
    print('DONE',flush=True)
if __name__=='__main__':main()
