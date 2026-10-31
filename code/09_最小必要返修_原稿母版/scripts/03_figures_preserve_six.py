"""Retain all six main figures, changing only implicated panels/labels/data."""
from pathlib import Path
import importlib.util,inspect,json,shutil,os
R=Path(__file__).resolve().parents[1];ROOT=R.parent
old=ROOT/'07_返修_20261028/scripts/04_revision_figures.py'
spec=importlib.util.spec_from_file_location('figure_helpers',old);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.R=R;m.A=R/'analysis';m.F=R/'figures'
for s in ['PDF','SVG','TIFF','PREVIEWS','panels','source_data','scripts','qa']:(m.F/s).mkdir(parents=True,exist_ok=True)
baseline_panel=m.panel
def safe_panel(ax,label,title=None):
    if label!='c' and ax.get_position().width<.65:return baseline_panel(ax,label,title)
    ax.text(-.04,1.04,label,transform=ax.transAxes,fontweight='bold',fontsize=9,va='bottom')
    if title:ax.set_title(title,loc='left',pad=7)
m.panel=safe_panel
flowcode=inspect.getsource(m.figure1).replace('Pretreatment single-cell landscape','Single-cell QC landscape')
exec(compile(flowcode,str(__file__),'exec'),m.__dict__)
maps=m.load_maps()
resume=os.environ.get('CBC_RESUME_F6')=='1'
only5=os.environ.get('CBC_ONLY_F5')=='1'
if os.environ.get('CBC_ONLY_F1')=='1':
    m.figure1(maps)
    shutil.copy2(__file__,m.F/'scripts'/Path(__file__).name)
    raise SystemExit(0)
if not resume:m.figure1(maps)
# Reuse the reviewed M1 Figure 2 layout with explicit group and cell legends.
src=(ROOT/'08_逐条精修/01_大意见1/scripts/02_M1_figures.py').read_text(encoding='utf8')
start=src.index('code=inspect.getsource');end=src.index('m.figure2()',start)
ns={'m':m,'inspect':inspect,'__file__':str(__file__)}
# Execute only the function adaptation, not its old export/data-directory setup.
exec(compile(src[start:end],str(__file__),'exec'),ns)
# Keep group counts above the narrow color strip; white letters on a 3-pt
# line lost their ascenders/descenders against the white background.
adapted=ns['code'].replace("idx.mean(),1.03,f'{group}","idx.mean(),1.07,f'{group}").replace("color='white',fontsize=6.3","color='#333333',fontsize=6.5")
exec(compile(adapted,str(__file__),'exec'),m.__dict__)
if os.environ.get('CBC_ONLY_F2')=='1':
    m.figure2();shutil.copy2(__file__,m.F/'scripts'/Path(__file__).name)
    raise SystemExit(0)
if not resume:m.figure2();m.figure3();m.figure4()
code=inspect.getsource(m.figure5)
a_start=code.index("    ax=fig.add_subplot(g[0,0])")
a_end=code.index("    sub=g[0,1]",a_start)
replacement="""    aa=g[0,0].subgridspec(2,1,height_ratios=[.8,1.2],hspace=.92)
    ax=fig.add_subplot(aa[0]);panel(ax,'a','Reference mapping accuracy')
    pred=pd.read_csv(P5/'03_scrna_spatial_reference_mapping/scRNA_reference_leave_one_patient_predictions.tsv',sep='\\t')
    accuracy=[pred.correct.mean(),pred[pred.observed_cell_type.isin(TYPES[1:])].correct.mean()]
    ax.scatter(accuracy,[0,1],s=25,color='#386C8E');ax.set_yticks([0,1],['Overall','Focal']);ax.set_xlim(0,1);ax.set_ylim(1.5,-.5);ax.set_xlabel('Leave-one-patient accuracy');clean(ax)
    pd.DataFrame({'group':['Overall','Focal'],'accuracy':accuracy}).to_csv(F/'source_data/Figure5_mapping_accuracy.tsv',sep='\\t',index=False)
    ax=fig.add_subplot(aa[1]);ax.set_title('Represented reference markers',loc='left',pad=7)
    qc=pd.read_csv(P5/'03_scrna_spatial_reference_mapping/StereoSeq_reference_mapping_QC.tsv',sep='\\t').set_index('section_id').reindex(SECTIONS)
    for k,(section,row) in enumerate(qc.iterrows()):ax.scatter(row.represented_fraction,k,s=25,color=CR if 'CR' in section else NR)
    ax.set_xlim(0,1);ax.set_ylim(3.5,-.5);ax.set_yticks(range(4),['CR1','CR2','NR1','NR2']);ax.set_xlabel('Marker representation fraction');clean(ax)
"""
code=code[:a_start]+replacement+code[a_end:]
exec(compile(code,str(__file__),'exec'),m.__dict__)
if not resume or only5:m.figure5(maps)
if only5:
    shutil.copy2(__file__,m.F/'scripts'/Path(__file__).name)
    raise SystemExit(0)
# Keep the original Figure 6 a/b/c modules and their frozen numerical sources.
previous_spec=m.importlib.util.spec_from_file_location
m.importlib.util.spec_from_file_location=m.relocated_spec
try:
    six=m.load_legacy('retained_figure6',m.P/'figure6_v18/scripts/build_figure6_v18.py')
    # The old PDF is a provenance dependency of load_data, not a plotting input.
    # It is absent from the archive; load the three audited frozen source tables.
    rank=m.pd.read_csv(six.RANKING_SOURCE,sep='\t').sort_values(['final_integrated_score','regulator'],ascending=[False,True]).reset_index(drop=True)
    pert=m.pd.read_csv(six.PERTURB_SOURCE,sep='\t')
    lr=m.pd.read_csv(six.LR_SOURCE,sep='\t').nlargest(4,'LR_priority_score').sort_values('LR_priority_score',ascending=False).reset_index(drop=True)
    assert rank.regulator.tolist()==six.EXPECTED_ORDER and lr.axis.tolist()==six.EXPECTED_LR
    assert (len(rank),len(pert),len(lr))==(8,16,4)
    oldpert=six.plot_perturbation
    def perturb(ax,ranking,data):
        oldpert(ax,ranking,data);ax.set_xlabel('Δ response-associated score')
    six.plot_perturbation=perturb
    network=inspect.getsource(six.draw_network)
    network=network.replace('width = 1.0 + 2.2 * (float(row.LR_priority_score) - smin) / max(smax - smin, 1e-9)','width = 1.4')
    exec(compile(network,str(__file__),'exec'),six.__dict__)
    fig,metrics=six.build_figure(rank,pert,lr)
    m.export(fig,'Figure_6')
finally:m.importlib.util.spec_from_file_location=previous_spec
m.supplement()
for p in (R/'analysis').glob('*.tsv'):shutil.copy2(p,m.F/'source_data'/p.name)
for p in (R/'source_data').glob('Figure*.tsv'):shutil.copy2(p,m.F/'source_data'/p.name)
for p in [old,Path(__file__)]:shutil.copy2(p,m.F/'scripts'/p.name)
note=json.loads((ROOT/'07_返修_20261028/figures/qa/reference_alignment.json').read_text(encoding='utf8'))
note.update(current_instruction='Keep all six original main figures; targeted changes only',retained_main_figures=list(range(1,7)),figure6='All 8 candidates, 16 simulations and 4 LR axes retained; uncertainty/null/ambient checks remain pending; LR linewidth held constant to avoid exaggerating tiny ranking differences',figure5='Original mapping accuracy, marker coverage, reference maps and marker concordance retained; accuracy/coverage separated; corrected neighborhood null replaces flawed query',actual_libraries=['Matplotlib','SciencePlots','Marsilea'],reference_reused=True)
(m.F/'qa/reference_alignment.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf8')
(m.F/'qa/export_manifest.json').write_text(json.dumps(m.manifest,indent=2),encoding='utf8')
print('Exported six main figures and one additional supplement; no main module deleted.',flush=True)
