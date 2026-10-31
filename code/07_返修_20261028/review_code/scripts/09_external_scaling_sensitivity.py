"""Outcome-labeled tumor-only scaling sensitivity; no gene reselection or tuning."""
from pathlib import Path
import importlib.util,json
import pandas as pd,numpy as np
from scipy import stats
R=Path(__file__).resolve().parents[1];P=R.parent/'06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery';A=R/'analysis'
helper=P/'scripts/05_prepare_and_validate_bulk_cohorts.py'
if not helper.exists():helper=R/'legacy_scripts/Phase1/05_prepare_and_validate_bulk_cohorts.py'
spec=importlib.util.spec_from_file_location('frozen_bulk_helpers',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=P
programs=m.load_programs();hgnc=pd.read_csv(P/'06_independent_validation_prep/hgnc_complete_set.txt',sep='\t',low_memory=False)
mapping=hgnc.dropna(subset=['ensembl_gene_id','symbol']).drop_duplicates('ensembl_gene_id').set_index('ensembl_gene_id').symbol
raw=pd.read_csv(P/'07_gse209746_validation/source/GSE209746_geneExp_n_114_ensembl.csv.gz',index_col=0);raw.index=raw.index.astype(str).str.split('.').str[0];expr209=m.log_cpm(m.aggregate_to_symbols(raw,mapping,'sum'))
platform=m.read_platform(P/'08_gse87211_validation/source/GPL13497_platform_data.txt');platform['GENE_SYMBOL']=platform.GENE_SYMBOL.astype(str).str.split(' /// ').str[0].str.upper();probe=platform.drop_duplicates('ID').set_index('ID').GENE_SYMBOL
expr872=m.aggregate_to_symbols(m.read_series_matrix(P/'08_gse87211_validation/source/GSE87211_series_matrix.txt.gz'),probe,'mean')
out=[];audit=[]
for dataset,expr,directory,group,poor in [('GSE209746',expr209,'07_gse209746_validation','response_standardized','incomplete_response'),('GSE87211',expr872,'08_gse87211_validation','response_endpoint','non_pCR')]:
    meta=pd.read_csv(P/directory/(dataset+'_metadata_prepared.tsv'),sep='\t');valid=meta[meta.usable_for_validation].set_index('sample_id');target=expr[expr.columns.intersection(valid.index)]
    scores,_=m.score_programs(target,programs,dataset)
    audit.append(dict(dataset=dataset,archived_reference_columns=expr.shape[1],target_only_reference_columns=target.shape[1],reference_restriction='labeled eligible pretreatment tumor samples only'))
    for typ,f in scores.groupby('program_source_cell_type'):
        f=f.merge(valid[[group]],left_on='sample_id',right_index=True);a=f[f[group].eq(poor)].resistance_score.to_numpy();b=f[~f[group].eq(poor)].resistance_score.to_numpy();g=m.hedges_g(a,b);variance=(len(a)+len(b))/(len(a)*len(b))+g*g/(2*(len(a)+len(b)));se=np.sqrt(variance)
        out.append(dict(dataset=dataset,program_source_cell_type=typ,n_poor=len(a),n_good=len(b),target_only_g=g,CI_low=g-1.96*se,CI_high=g+1.96*se,scaling='eligible outcome-labeled tumor columns only; gene population SD; no reselection'))
pd.DataFrame(out).to_csv(A/'external_target_only_scaling.tsv',sep='\t',index=False)
pd.DataFrame(audit).to_csv(A/'external_scaling_reference_counts.tsv',sep='\t',index=False)
print(pd.DataFrame(out).to_string(index=False));print(pd.DataFrame(audit).to_string(index=False))
