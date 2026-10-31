"""Frozen evidence audit, patient-conditional ranges and algorithmic controls."""
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];ROOT=R.parent;P3=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase3_VirtualPerturbation_RegulatorNomination';V=ROOT/'06_研究资料/LARC_NAC_FINAL_FIGURE_REDESIGN_v11/source_data';P1=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery';A=R/'analysis';rng=np.random.default_rng(20261030)
def read(p):return pd.read_csv(p,sep='\t')
def save(d,n):pd.DataFrame(d).to_csv(A/n,sep='\t',index=False)
rank=read(P3/'09_candidate_prioritization/final_candidate_regulator_ranking.tsv');sel=read(V/'Figure6_selected_candidate_evidence.tsv');sel.to_csv(A/'Figure6_original_selected_candidates.tsv',sep='\t',index=False)
components=pd.DataFrame(dict(activity=rank.activity_evidence_score.fillna(0),perturbation=rank.virtual_perturbation_score.fillna(0),GRN=rank.groupby('target_state').two_or_more_method_NR_edges.rank(pct=True).fillna(0),spatial=rank.groupby('target_state').section_effect_NR_minus_CR.rank(pct=True).fillna(0),stability=rank.stability_score.fillna(0),LR=rank.maximum_LR_priority.fillna(0)))
error=float(np.max(np.abs(components.mean(axis=1)-rank.final_integrated_score)));assert error<1e-12,error
ablation=[]
for name in components:
 v=components.drop(columns=name).mean(axis=1);ranks=v.groupby(rank.target_state).rank(ascending=False,method='min');full=rank.groupby('target_state').final_integrated_score.rank(ascending=False,method='min')
 for s in sel.itertuples():
  j=rank.index[rank.target_state.eq(s.target_state)&rank.regulator.eq(s.regulator)].item()
  ablation.append(dict(regulator=s.regulator,target_state=s.target_state,removed_component=name,full_integrated_score=rank.loc[j,'final_integrated_score'],full_within_state_rank=full[j],ablated_score=v[j],ablated_within_state_rank=ranks[j],n_candidates_in_state=int(rank.target_state.eq(s.target_state).sum()),scope='evidence-component ablation; not sampling uncertainty or recalibrated confidence'))
save(ablation,'Revision_Supplementary_Table_R15a_component_ablation.tsv')
activity=pd.read_parquet(P3/'04_regulator_activity/regulator_activity_pseudobulk_scores.parquet');meta=read(P3/'01_inputs_from_phase1_phase2/CNP0004138_PRET_patient_celltype_pseudobulk_metadata.tsv').set_index('pseudobulk_id');unit=read(P3/'06_virtual_perturbation/virtual_perturbation_pseudobulk_results.tsv');edges=read(P3/'05_grn_inference/GRN_edges_by_target_state.tsv');genes=read(P1/'05_response_state_discovery/CNP0004138_locked_programs_by_celltype.tsv');summary=[];loo=[];points=[];nullrows=[];edgerows=[]
for s in sel.itertuples():
 a=activity[activity.target_state.eq(s.target_state)&activity.regulator.eq(s.regulator)].pivot(index='pseudobulk_id',columns='method',values='activity_score').dropna();m=meta.loc[a.index];responses=m.response_standardized.to_numpy();cr=np.flatnonzero(responses=='CR');nr=np.flatnonzero(responses=='NR');assert len(cr)==6 and len(nr)==5
 x=a.to_numpy();boots=np.median(x[rng.choice(nr,(10000,5))].mean(1)-x[rng.choice(cr,(10000,6))].mean(1),axis=1);observed=np.median(x[nr].mean(0)-x[cr].mean(0));assert abs(observed-s.median_activity_NR_minus_CR)<1e-9
 summary.append(dict(regulator=s.regulator,target_state=s.target_state,metric='activity_NR_minus_CR_median_methods',estimate=observed,low=np.quantile(boots,.025),high=np.quantile(boots,.975),n_CR=6,n_NR=5,n_methods=x.shape[1],scope='patient bootstrap conditional on frozen selected regulator/activity estimates; no GRN or selection uncertainty'))
 for j in np.r_[cr,nr]:
  keep=np.arange(len(x))!=j;value=np.median(x[(responses=='NR')&keep].mean(0)-x[(responses=='CR')&keep].mean(0));loo.append(dict(regulator=s.regulator,target_state=s.target_state,patient_removed=m.iloc[j].patient_id,metric='activity_NR_minus_CR',estimate=value,full=observed))
 e=edges[edges.target_state.eq(s.target_state)&edges.regulator.eq(s.regulator)&edges.method_support_count.ge(2)].copy();program=genes[genes.source_cell_type.eq(s.source_cell_type)];directions=dict(zip(program.gene_symbol.str.upper(),program.program_direction));e=e[e.target_gene.isin(directions)].copy()
 e['signed_step']=-.5*(e.method_support_count/3)*np.where(e.prior_weight.notna()&e.prior_weight.ne(0),e.prior_weight,np.sign(e.coexpression_spearman_rho));e['score_weight']=e.target_gene.map(lambda g:1/100 if directions[g]=='NR_high' else -1/100);q=float((e.signed_step*e.score_weight).sum())
 for variant,ee in [('all_supported_edges',e),('exclude_self_targets',e[e.target_gene.ne(s.regulator)]),('prior_supported_only',e[e.prior_support]),('prior_direction_consistent_only',e[e.prior_support&e.direction_consistent_with_prior])]:
  edgerows.append(dict(regulator=s.regulator,target_state=s.target_state,condition=variant,n_edges=len(ee),unit_step_score_delta=float((ee.signed_step*ee.score_weight).sum()),n_prior=int(ee.prior_support.sum()),n_self=int(ee.target_gene.eq(s.regulator).sum()),scope='fixed frozen graph edge restriction; no network refit or hypothesis test'))
 # Conditional target-slot randomization preserves signed shifts and edge count.
 # It deliberately destroys NR/CR module alignment; selection makes it diagnostic.
 weights=np.r_[np.ones(100)/100,-np.ones(100)/100];steps=e.signed_step.to_numpy();draw=np.array([np.sum(rng.permutation(weights)[:len(steps)]*steps) for _ in range(5000)])
 for method in ['fixed_target_downweight','activity_scaled_network_propagation']:
  d=unit[unit.target_state.eq(s.target_state)&unit.regulator.eq(s.regulator)&unit.perturbation_method.eq(method)];dn=d[d.response_standardized.eq('NR')];vals=dn.delta_resistance_score.to_numpy();actual=vals.mean()
  factor=1. if method=='fixed_target_downweight' else float(actual/q) if abs(q)>1e-12 else 0.
  np.testing.assert_allclose(d.delta_resistance_score,q if method=='fixed_target_downweight' else q*np.maximum((a.mean(axis=1).reindex(d.pseudobulk_id).to_numpy()-a.mean(axis=1).mean())/a.mean(axis=1).std(ddof=1),0),atol=1e-12)
  b=vals[rng.integers(0,5,size=(10000,5))].mean(1);summary.append(dict(regulator=s.regulator,target_state=s.target_state,metric=method,estimate=actual,low=np.quantile(b,.025),high=np.quantile(b,.975),n_CR=0,n_NR=5,n_methods=1,scope='fixed-network NR-patient bootstrap; fixed shifts are degenerate by construction, not biological-effect CI'))
  for v in dn.itertuples():points.append(dict(regulator=s.regulator,target_state=s.target_state,method=method,patient_id=v.patient_id,response=v.response_standardized,delta=v.delta_resistance_score))
  for patient in dn.patient_id:loo.append(dict(regulator=s.regulator,target_state=s.target_state,patient_removed=patient,metric=method,estimate=dn.loc[dn.patient_id.ne(patient),'delta_resistance_score'].mean(),full=actual))
  nullrows.append(dict(regulator=s.regulator,target_state=s.target_state,method=method,observed=actual,target_slot_null_low=np.quantile(draw*factor,.025),target_slot_null_high=np.quantile(draw*factor,.975),target_slot_null_median=np.median(draw*factor),within_module_reassignment_delta=actual,n_null=5000,n_NR=5,n_edges=len(e),null_scope='random fixed target-slot assignment breaks score-module alignment; diagnostic only; no calibrated biological P value',within_module_scope='within-NR/CR module target permutation is algebraically invariant'))
save(summary,'Revision_Supplementary_Table_R15b_patient_ranges.tsv');save(loo,'Revision_Supplementary_Table_R15c_leave_one.tsv');save(points,'Revision_Supplementary_Table_R15d_NR_patient_perturbations.tsv');save(nullrows,'Revision_Supplementary_Table_R15e_null_diagnostics.tsv');save(edgerows,'Revision_Supplementary_Table_R15f_graph_restrictions.tsv');save(rank,'R15_all_original_87_candidates.tsv')
(R/'qa/m9_validation.json').write_text(json.dumps(dict(integrated_reproduction_max_error=error,selected_candidates=len(sel),original_candidates=len(rank),perturbation_shift_reproduction=True,integrated_rank_sampling_CI_not_estimated=True,biological_perturbation_null_not_available=True),indent=2),encoding='utf8')
print(pd.DataFrame(summary).to_string(index=False));print('ablations',len(ablation),'null diagnostics',len(nullrows))
