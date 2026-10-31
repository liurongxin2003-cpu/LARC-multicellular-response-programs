from pathlib import Path
import json,shutil,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots,marsilea as ma
from PIL import Image
from matplotlib.transforms import Bbox
R=Path(__file__).resolve().parents[1];ROOT=R.parent;A=R/'analysis';F=R/'figures'
for p in ['panels','source_data','scripts','qa','main_unchanged']:(F/p).mkdir(exist_ok=True)
plt.style.use(['science','no-latex']);plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'axes.linewidth':.6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'axes.spines.top':False,'axes.spines.right':False,'xtick.top':False,'ytick.right':False,'xtick.minor.visible':False,'ytick.minor.visible':False})
W=183/25.4;order=['ZBTB7A','E2F6','KLF5','RELB','SRF','MAZ','IRF3','RXRA'];colors=['#7570B3','#4C9F70','#7570B3','#29A39D','#E66552','#4C9F70','#E66552','#29A39D'];qa=[]
def read(n):return pd.read_csv(A/n,sep='\t')
def tag(ax,l,t):ax.text(-.13,1.04,l,transform=ax.transAxes,fontweight='bold',fontsize=9);ax.set_title(t,loc='left',pad=8)
def export(fig,key,panels):
 fig.savefig(F/(key+'.pdf'));fig.savefig(F/(key+'.svg'));fig.savefig(F/(key+'_preview.png'),dpi=180);tmp=F/'qa'/(key+'.png');fig.savefig(tmp,dpi=600)
 with Image.open(tmp) as im:im.convert('RGB').save(F/(key+'.tiff'),compression='tiff_lzw',dpi=(600,600))
 tmp.unlink();fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for k,axs in panels.items():
  bb=Bbox.union([a.get_tightbbox(renderer) for a in axs]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.04,1.08)
  for ext in ['pdf','svg']:fig.savefig(F/'panels'/f'{key}_{k}.{ext}',bbox_inches=bb)
 qa.append(dict(figure=key,width_mm=183,height_mm=fig.get_figheight()*25.4,sha256={e:hashlib.sha256((F/(key+'.'+e)).read_bytes()).hexdigest() for e in ['pdf','svg','tiff']}));plt.close(fig)
def heat(fig,spec,data,rows,cols,lo,hi,cmap,l,title,label):
 bb=spec.get_position(fig);h=ma.Heatmap(data,width=bb.width*W*.84,height=bb.height*fig.get_figheight(),vmin=lo,vmax=hi,cmap=cmap,linewidth=.4,legend=False);h.layout.is_composite=True;h.layout.set_figsize((W,fig.get_figheight()));h.layout.set_anchor((bb.x0*W,bb.y0*fig.get_figheight()));h.render(figure=fig);ax=h.get_main_ax();ax.set_axis_on();ax.set(xticks=np.arange(len(cols))+.5,xticklabels=cols,yticks=np.arange(len(rows))+.5,yticklabels=rows);ax.tick_params(length=0);plt.setp(ax.get_xticklabels(),rotation=35,ha='right',rotation_mode='anchor')
 for sp in ax.spines.values():sp.set_visible(False)
 cbax=fig.add_axes([bb.x0+bb.width*.88,bb.y0+.04*bb.height,.01,bb.height*.92]);cb=fig.colorbar(ax.collections[0],cax=cbax);cb.solids.set_rasterized(False);cb.set_label(label);tag(ax,l,title);return [ax,cbax]
summary=read('Revision_Supplementary_Table_R15b_patient_ranges.tsv');points=read('Revision_Supplementary_Table_R15d_NR_patient_perturbations.tsv');ab=read('Revision_Supplementary_Table_R15a_component_ablation.tsv');nul=read('Revision_Supplementary_Table_R15e_null_diagnostics.tsv')
fig=plt.figure(figsize=(W,175/25.4));gs=fig.add_gridspec(2,2,left=.13,right=.93,bottom=.16,top=.92,wspace=.46,hspace=.70);ax=fig.add_subplot(gs[0,0]);z=summary[summary.metric.eq('activity_NR_minus_CR_median_methods')].set_index('regulator').loc[order]
for i,(r,c) in enumerate(zip(z.itertuples(),colors)):ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.low],[r.high-r.estimate]],fmt='o',ms=3,color=c,lw=.8,capsize=2)
ax.axvline(0,ls='--',color='.6',lw=.6);ax.set(yticks=np.arange(8),yticklabels=order,ylim=(7.6,-.6),xlabel='NR − CR inferred activity');tag(ax,'a','Frozen activity: patient ranges');aa=[ax]
ax=fig.add_subplot(gs[0,1])
for i,(reg,c) in enumerate(zip(order,colors)):
 for j,method in enumerate(['fixed_target_downweight','activity_scaled_network_propagation']):
  v=points[points.regulator.eq(reg)&points.method.eq(method)].delta.to_numpy();offset=-.12 if j==0 else .12;ax.scatter(v,i+offset+np.linspace(-.05,.05,5),s=9,marker='o' if j==0 else 's',facecolors=c if j==0 else 'none',edgecolors=c,lw=.6)
ax.axvline(0,color='.6',ls='--',lw=.6);ax.set(yticks=np.arange(8),yticklabels=order,ylim=(7.6,-.6),xlabel='Δ response-associated score');tag(ax,'b','All five NR patients per method');ax.legend(handles=[plt.Line2D([0],[0],marker='o',ls='',color='.4',ms=3,label='Fixed shift'),plt.Line2D([0],[0],marker='s',ls='',markerfacecolor='none',color='.4',ms=3,label='Activity scaled')],frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.1,-.32));bb=[ax]
components=['activity','perturbation','GRN','spatial','stability','LR'];mat=ab.pivot(index='regulator',columns='removed_component',values='ablated_within_state_rank').loc[order,components];cc=heat(fig,gs[1,0],mat.to_numpy(),order,['Activity','Perturb.','GRN','Spatial','Stability','LR'],1,9,'YlGnBu','c','Rank after component omission','Within-state rank')
ax=fig.add_subplot(gs[1,1]);z=nul[nul.method.eq('fixed_target_downweight')].set_index('regulator').loc[order]
for i,(r,c) in enumerate(zip(z.itertuples(),colors)):
 ax.plot([r.target_slot_null_low,r.target_slot_null_high],[i,i],color='.65',lw=2);ax.plot(r.target_slot_null_median,i,'|',color='.3');ax.scatter(r.observed,i,s=15,marker='D',color=c,zorder=4)
ax.axvline(0,ls='--',color='.6',lw=.6);ax.set(yticks=np.arange(8),yticklabels=order,ylim=(7.6,-.6),xlabel='Fixed-shift score change');tag(ax,'d','Target-slot randomization');ax.legend(handles=[plt.Line2D([0],[0],color='.65',lw=2,label='Random target slots'),plt.Line2D([0],[0],marker='D',ls='',color='.3',ms=3,label='Observed / within-module')],frameon=False,loc='lower left',bbox_to_anchor=(-.12,-.34));dd=[ax]
export(fig,'Supplementary_Figure_R4',{'a':aa,'b':bb,'c':cc,'d':dd})

flag=read('Revision_Supplementary_Table_R16f_marker_and_QC_flags.tsv');expr=read('Revision_Supplementary_Table_R16e_broad_lineage_expression.tsv');s=read('Revision_Supplementary_Table_R16c_sensitivity.tsv');endpoints=[('Malignant_epithelial','LTA'),('Myeloid_macrophage','TNFRSF1A'),('Myeloid_macrophage','TNFRSF14'),('CAF_Fibroblast','VWF'),('Malignant_epithelial','STAB2'),('Endothelial','BMP8A'),('Myeloid_macrophage','TGFBR1'),('Endothelial','HLA-B'),('CD8_T','KIR2DL3')];abbr={'Malignant_epithelial':'Malig.','Myeloid_macrophage':'Myel.','CAF_Fibroblast':'CAF','Endothelial':'Endo.','CD8_T':'CD8 T'};rows=[abbr[t]+': '+g for t,g in endpoints];f=flag.set_index(['cell_type','gene']).loc[endpoints];v=expr.set_index(['cell_type','gene']).loc[endpoints]
fig=plt.figure(figsize=(W,181/25.4));gs=fig.add_gridspec(2,2,left=.18,right=.92,bottom=.19,top=.92,wspace=.55,hspace=.70);ax=fig.add_subplot(gs[0,0]);ax.scatter(v.positive_fraction*100,np.arange(9),color='#3C5488',s=15);ax.set(xscale='log',yticks=np.arange(9),yticklabels=rows,ylim=(8.6,-.6),xlabel='Cells with detected gene (%)',xlim=(.06,110));tag(ax,'a','Assigned-lineage detection');aa=[ax]
bb=heat(fig,gs[0,1],f[['single_count_fraction_positive','mixed_fraction_positive']].to_numpy(),rows,['Single count','Mixed markers'],0,1,'YlOrBr','b','Flags among gene-positive cells','Fraction')
ax=fig.add_subplot(gs[1,0]);pairs=[('LTA','TNFRSF1A'),('LTA','TNFRSF14'),('VWF','STAB2'),('BMP8A','TGFBR1'),('HLA-B','KIR2DL3')];plabel=[l+'–'+r for l,r in pairs]
for j,(cond,col,mk) in enumerate([('all_PRET','.3','o'),('lineage_and_low_mixing','#00A087','s'),('remove_single_counts','#E64B35','D')]):
 q=s[s.condition.eq(cond)].set_index(['ligand','receptor']).loc[pairs]
 for i,r in enumerate(q.itertuples()):ax.errorbar(r.effect_NR_minus_CR,i+(j-1)*.16,xerr=[[r.effect_NR_minus_CR-r.bootstrap_low],[r.bootstrap_high-r.effect_NR_minus_CR]],fmt=mk,ms=2.7,lw=.65,color=col,capsize=1.4)
ax.axvline(0,color='.6',ls='--',lw=.6);ax.set(yticks=np.arange(5),yticklabels=plabel,ylim=(4.6,-.6),xlabel='NR − CR paired-expression score');tag(ax,'c','Patient-level axis sensitivity');ax.legend(handles=[plt.Line2D([0],[0],marker=mk,color=co,ls='',ms=3,label=la) for co,mk,la in [('.3','o','Original'),('#00A087','s','Lineage + low mixing'),('#E64B35','D','Remove single counts')]],frameon=False,loc='lower left',bbox_to_anchor=(-.14,-.42));cc=[ax]
conds=['all_PRET','high_annotation_confidence','lineage_compatible','exclude_mixed_lineage','exclude_joint_high_QC','lineage_and_low_mixing','remove_single_counts','target_counts_ge3'];mat=s.assign(axis=s.ligand+'–'+s.receptor).pivot(index='axis',columns='condition',values='patients_both_genes_detected').loc[plabel,conds];dd=heat(fig,gs[1,1],mat.to_numpy(),plabel,['Original','High conf.','Lineage','Low mixing','No high QC','Lineage + low','Counts ≥2','Counts ≥3'],0,25,'Blues','d','Patients with both genes detected','Source patients')
export(fig,'Supplementary_Figure_R5',{'a':aa,'b':bb,'c':cc,'d':dd})
for p in A.glob('Revision_Supplementary_Table_R1[56]*.tsv'):shutil.copy2(p,F/'source_data'/p.name)
shutil.copy2(__file__,F/'scripts'/Path(__file__).name)
for ext,folder in [('pdf','PDF'),('svg','SVG'),('tiff','TIFF')]:shutil.copy2(ROOT/f'09_最小必要返修_原稿母版/figures/{folder}/Figure_6.{ext}',F/f'main_unchanged/Figure_6.{ext}')
(F/'qa/export_manifest.json').write_text(json.dumps(qa,indent=2),encoding='utf8');print('R4/R5 exported; Figure 6 copied unchanged')
