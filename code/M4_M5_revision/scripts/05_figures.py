"""R2/R3: data-preserving additional figures; no refitting or hypothesis selection."""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
import marsilea as ma
import marsilea.plotter as mp
from PIL import Image
R=Path(__file__).resolve().parents[1];A=R/'analysis';F=R/'figures'
for p in [F/'panels',F/'source_data',F/'scripts',F/'qa']:p.mkdir(parents=True,exist_ok=True)
plt.style.use(['science','no-latex'])
plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'axes.titlesize':8,'axes.linewidth':.6,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,'axes.spines.top':False,'axes.spines.right':False,'xtick.top':False,'ytick.right':False,'xtick.minor.visible':False,'ytick.minor.visible':False})
W=183/25.4;names=['Global','Malignant','CAF','Myeloid','CD8 T','Endothelial'];models=['Global+'+p for p in names[1:]]+['Global+Mean5'];labels=['+ Malignant','+ CAF','+ Myeloid','+ CD8 T','+ Endothelial','+ Mean of five']
colors={'CNP':'#3C5488','GSE209746':'#00A087','GSE87211':'#E64B35'}
def read(n):return pd.read_csv(A/n,sep='\t')
def tag(ax,letter,title):
    ax.text(-.12,1.04,letter,transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom');ax.set_title(title,loc='left',pad=7)
qa=[]
def export(fig,key,panelaxes):
    fig.savefig(F/(key+'.pdf'));fig.savefig(F/(key+'.svg'));fig.savefig(F/(key+'_preview.png'),dpi=180)
    png=F/'qa'/(key+'_600.png');fig.savefig(png,dpi=600)
    with Image.open(png) as im:im.convert('RGB').save(F/(key+'.tiff'),compression='tiff_lzw',dpi=(600,600))
    png.unlink()
    # Native axes crops preserve actual vector objects, rather than embedding screenshots.
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for panel,axes in panelaxes.items():
        from matplotlib.transforms import Bbox
        bb=Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.dpi_scale_trans.inverted()).expanded(1.04,1.08)
        fig.savefig(F/'panels'/f'{key}_{panel}.pdf',bbox_inches=bb);fig.savefig(F/'panels'/f'{key}_{panel}.svg',bbox_inches=bb)
    with Image.open(F/(key+'.tiff')) as im:rec=dict(figure=key,width_mm=183,height_mm=fig.get_figheight()*25.4,pixels=im.size,mode=im.mode,dpi=[float(v) for v in im.info.get('dpi')],compression=im.info.get('compression'))
    for ext in ['pdf','svg','tiff']:rec[ext+'_sha256']=hashlib.sha256((F/(key+'.'+ext)).read_bytes()).hexdigest()
    qa.append(rec);plt.close(fig)

c=read('M4_model_comparisons.tsv');p=read('M4_patient_predictions.tsv');lpo=read('M4_leave_pair_out_AUC.tsv')
fig=plt.figure(figsize=(W,147/25.4));gs=fig.add_gridspec(2,3,left=.17,right=.98,top=.91,bottom=.11,hspace=.62,wspace=.3,height_ratios=[1.2,1]);aa=[]
for i,ds in enumerate(colors):
    ax=fig.add_subplot(gs[0,i]);z=c[c.dataset.eq(ds)].set_index('model').loc[models]
    x=z.delta_log_loss.to_numpy();lo=z.delta_log_loss_bootstrap_low.to_numpy();hi=z.delta_log_loss_bootstrap_high.to_numpy();yy=np.arange(6)
    ax.errorbar(x,yy,xerr=np.vstack([x-lo,hi-x]),fmt='o',color=colors[ds],ms=3.6,lw=.8,capsize=2)
    ax.axvline(0,color='.55',lw=.6,ls='--');ax.set(ylim=(5.6,-.6),xlim=(-.195,.08),xticks=[-.15,-.05,.05],yticks=yy,yticklabels=labels if i==0 else ['']*6,xlabel='Δ held-out log loss')
    tag(ax,'a' if i==0 else '',{'CNP':'Discovery (11 patients)','GSE209746':'GSE209746 (97 cases)','GSE87211':'GSE87211 (201 patients)'}[ds]);aa.append(ax)
ax=fig.add_subplot(gs[1,:2]);d=p[p.dataset.eq('CNP')&p.lambda_.eq(1)].pivot(index='patient_id',columns='model',values='log_loss');q=p[p.dataset.eq('CNP')&p.model.eq('Global')&p.lambda_.eq(1)].set_index('patient_id').loc[d.index]
gain=d.Global-d['Global+Mean5'];xx=np.arange(len(d))
for value,col,mk in [(0,'#00A087','o'),(1,'#E64B35','s')]:
    keep=q.poor_response.eq(value).to_numpy();ax.scatter(xx[keep],gain.to_numpy()[keep],color=col,marker=mk,s=15,label='CR-like' if value==0 else 'NR',zorder=3)
ax.axhline(0,color='.55',lw=.6,ls='--');ax.set(xticks=xx,xticklabels=[x.replace('XFZ','') for x in d.index],ylabel='Global − joint loss',xlabel='Patient (XFZ prefix)');ax.legend(frameon=False,ncol=2,loc='lower left');tag(ax,'b','All discovery patients: Global + mean of five')
ab=ax
ax=fig.add_subplot(gs[1,2]);yy=np.arange(3)
for j,model in enumerate(['Global','Global+Mean5']):
    z=lpo[lpo.model.eq(model)].set_index('dataset').loc[list(colors)];v=z.leave_pair_out_AUC.to_numpy();offset=(-.10 if j==0 else .10)
    ax.errorbar(v,yy+offset,xerr=[v-z.AUC_bootstrap_low,z.AUC_bootstrap_high-v],fmt='o' if j==0 else 's',ms=3,lw=.8,capsize=2,color='.35' if j==0 else '#3C5488',label='Global' if j==0 else '+ Mean5')
ax.axvline(.5,color='.55',lw=.6,ls='--');ax.set(xlim=(0,1.02),ylim=(2.5,-.5),yticks=yy,yticklabels=['CNP','G209746','G87211'],xticks=[0,.5,1],xlabel='Leave-pair-out AUC');ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,-.41),ncol=1);tag(ax,'c','Paired discrimination')
export(fig,'Supplementary_Figure_R2',{'a':aa,'b':[ab],'c':[ax]})

s=pd.read_csv(R/'M5_sensitivity_results.tsv',sep='\t');rho=read('M5_score_covariate_relationships.tsv');depth=read('M5_depth_thinning_summary.tsv')
fig=plt.figure(figsize=(W,170/25.4));gs=fig.add_gridspec(2,2,left=.14,right=.93,bottom=.17,top=.92,wspace=.48,hspace=.70)
def heat_at(spec,data,cols,vmin,vmax,cmap,letter,title,cbarlabel,groups=None):
    bb=spec.get_position(fig);h=ma.Heatmap(data,width=bb.width*W*.86,height=bb.height*fig.get_figheight(),vmin=vmin,vmax=vmax,cmap=cmap,linewidth=.4,legend=False)
    if groups is not None:h.add_top(mp.Colors(groups,palette={'Technical':'#8F9CB1','Composition':'#E8BA85'}),size=.07,pad=.02)
    h.layout.is_composite=True;h.layout.set_figsize((W,fig.get_figheight()));h.layout.set_anchor((bb.x0*W,bb.y0*fig.get_figheight()));h.render(figure=fig)
    ax=h.get_main_ax();ax.set_axis_on();ax.set(xticks=np.arange(len(cols))+.5,xticklabels=cols,yticks=np.arange(6)+.5,yticklabels=names);ax.tick_params(axis='both',length=0);plt.setp(ax.get_xticklabels(),rotation=35,ha='right',rotation_mode='anchor');tag(ax,letter,title)
    for sp in ax.spines.values():sp.set_visible(False)
    cbax=fig.add_axes([bb.x0+bb.width*.89,bb.y0+.05*bb.height,.01,bb.height*.90]);cb=fig.colorbar(ax.collections[0],cax=cbax,orientation='vertical');cb.set_label(cbarlabel);cb.ax.tick_params(labelsize=6)
    cb.solids.set_rasterized(False)
    return ax,cb.ax
keys=['log_cell_count','log_counts','log_counts_per_cell','log_detected_genes','fraction_Malignant']
mat=rho.pivot(index='program',columns='covariate',values='overall_spearman_rho').loc[names,keys].to_numpy()
ha=heat_at(gs[0,0],mat,['Cells','Counts','Counts/cell','Genes','Malignant %'],-1,1,'RdBu_r','a','Score–covariate associations','Spearman ρ', ['Technical']*4+['Composition'])
ax=fig.add_subplot(gs[0,1]);un=s[s.dataset.eq('CNP')&s.analysis.eq('unadjusted')].set_index('program').loc[names]
ax.scatter(un.effect,np.arange(6)-.18,s=13,marker='D',color='.25',label='Unadjusted',zorder=4)
for condition,variable,col,offset,label in [('two_technical_covariates','', '#3C5488',0,'Cells + counts/cell'),('all25_categorical_response','fraction_Malignant','#E64B35',.18,'Malignant fraction')]:
    z=s[s.dataset.eq('CNP')&s.analysis.eq('covariate_adjusted')&s.condition.eq(condition)]
    if variable:z=z[z.variable.eq(variable)]
    z=z.set_index('program').loc[names];v=z.adjusted_beta.to_numpy();ax.errorbar(v,np.arange(6)+offset,xerr=[v-z.CI_low,z.CI_high-v],fmt='o',ms=2.8,lw=.65,capsize=1.6,color=col,label=label)
ax.axvline(0,color='.55',ls='--',lw=.6);ax.set(yticks=np.arange(6),yticklabels=names,ylim=(5.6,-.6),xlabel='NR − CR score coefficient');tag(ax,'b','Conditional covariate adjustment');ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(-.1,-.39),ncol=1)
hb=ax
ax=fig.add_subplot(gs[1,0]);dy=np.linspace(-.16,.16,5)
for i,n in enumerate(names):
    zz=depth[depth.program.eq(n)].sort_values('replicate').hedges_g.to_numpy();ax.plot([zz.min(),zz.max()],[i,i],color='#3C5488',lw=1);ax.scatter(zz,i+dy,s=12,color='#3C5488',zorder=3)
ax.scatter(un.hedges_g,np.arange(6),color='.2',marker='D',s=18,label='Original held-out',zorder=4);ax.axvline(0,color='.55',ls='--',lw=.6);ax.set(yticks=np.arange(6),yticklabels=names,ylim=(5.6,-.6),xlabel='Hedges g (NR − CR)');ax.legend(handles=[Line2D([0],[0],marker='D',ls='',color='.2',ms=4,label='Original held-out'),Line2D([0],[0],marker='o',ls='-',color='#3C5488',ms=3,label='Five depth-thinning repeats')],frameon=False,loc='lower left',bbox_to_anchor=(-.08,-.40));tag(ax,'c','Training-only depth perturbation');hc=ax
scopes=['all_bins','genes>=100','genes>=300','genes>=500','malignant_only','malignant_and_genes>=300']
z=s[s.dataset.eq('Stereo-seq')].pivot(index='program',columns='condition',values='effect').loc[names,scopes]
hd=heat_at(gs[1,1],z.to_numpy(),['All bins','Genes ≥100','Genes ≥300','Genes ≥500','Malignant','Malig. + ≥300'],0,1.7,'YlGnBu','d','Spatial QC: four sections','NR − CR score')
export(fig,'Supplementary_Figure_R3',{'a':ha,'b':[hb],'c':[hc],'d':hd})
for n in ['M4_model_comparisons.tsv','M4_patient_predictions.tsv','M4_leave_pair_out_AUC.tsv','M5_score_covariate_relationships.tsv','M5_depth_thinning_summary.tsv']:
    shutil.copy2(A/n,F/'source_data'/n)
shutil.copy2(R/'M5_sensitivity_results.tsv',F/'source_data/M5_sensitivity_results.tsv');shutil.copy2(__file__,F/'scripts'/Path(__file__).name)
(F/'qa/export_metadata.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(qa,indent=2))
