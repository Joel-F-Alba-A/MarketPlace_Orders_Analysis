"""Regenerate the report charts from saved results; never train models."""
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--language', choices=['es', 'en'], default='es')
args = parser.parse_args()
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parents[1];r=p/'results';out=p/'output/pdf';figs=out/'figuras';figs = figs / 'en' if args.language == 'en' else figs;figs.mkdir(parents=True, exist_ok=True)
metrics=pd.read_csv(r/'04_recommendation_ica_matrix_factorization/fa_ica_pca_reconstruction_metrics.csv');v=metrics[metrics.Dataset=='Validation'].set_index('Algorithm')
svd=pd.read_csv(r/'03_svd_pca_probabilistic_pca_lsa/svd_k5_reconstruction_metrics.csv').set_index('Dataset')
ppca=pd.read_csv(r/'03_svd_pca_probabilistic_pca_lsa/ppca_k5_reconstruction_metrics.csv').set_index('Dataset')
ae=pd.read_csv(r/'07_autoencoder_vae/ae_best_per_dimension.csv')
ae5=ae[ae.latent_dim==5].iloc[0]
names=['PCA','PPCA','SVD','ICA','FA','AE']
vals=[v.loc['PCA','MSE'],ppca.loc['validation','Reconstruction MSE'],svd.loc['validation','Reconstruction MSE'],v.loc['ICA','MSE'],v.loc['FA','MSE'],ae5.val_mse]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#333333','text.color':'#222222','xtick.color':'#444444','ytick.color':'#444444','axes.edgecolor':'#999999'})
fig,ax=plt.subplots(figsize=(7.6,2.8),layout='constrained')
colors=['#B8BDC3']*len(vals);colors[int(np.argmax(vals))]='#173B56'
bars=ax.bar(names,vals,color=colors,width=.58)
ax.set(title=('Reconstrucción en validación con cinco dimensiones' if args.language == "es" else 'Validation reconstruction with five dimensions'),ylabel=('MSE (menor es mejor)' if args.language == "es" else 'MSE (lower is better)'),ylim=(0,.145))
ax.bar_label(bars,labels=[f'{x:.6f}' if x<.001 else f'{x:.4f}' for x in vals],padding=4,fontsize=9)
ax.yaxis.grid(True,color='#EEEEEE',linewidth=.6);ax.set_axisbelow(True)
fig.savefig(figs/'05_reconstruccion_todos_modelos.png',dpi=250);plt.close(fig)
pd.DataFrame({'Model':names,'Validation MSE':vals,'Split':['70/15/15']*5+['60/20/20'],'Centering':['Yes','Yes','No','Yes','Yes','Yes']}).to_csv(figs/'05_reconstruccion_todos_modelos.csv',index=False)
# Anomaly quantities, read from saved comparisons.
def count(path,alpha,column='Anomalies'):
 d=pd.read_csv(r/path);return int(d[(d.Dataset=='Validation')&np.isclose(d.Alpha,alpha)].iloc[0][column])
vae=pd.read_csv(r/'07_autoencoder_vae/vae_validation_scores.csv')
a=[]
for alpha,suffix in [(.05,'05'),(.01,'01')]:
 a.append([count('06_anomaly_detection/pca_anomaly_quantile_comparison.csv',alpha,'T2 Anomalies'),count('06_anomaly_detection/pca_anomaly_quantile_comparison.csv',alpha,'SPE Anomalies'),count('06_anomaly_detection/ica_anomaly_quantile_comparison.csv',alpha,'NLL Anomalies'),count('06_anomaly_detection/mahalanobis_quantile_comparison.csv',alpha),count('06_anomaly_detection/lof_k20_detection_summary.csv',alpha),count('06_anomaly_detection/if_400trees_1024samples_detection_summary.csv',alpha),int(vae['vae_anomaly_'+suffix].sum())])
fig,ax=plt.subplots(figsize=(7.6,2.95),layout='constrained');x=np.arange(7)
for offset,counts,color,label in [(-.18,a[0],'#173B56',r'$\alpha=0.05$'),(.18,a[1],'#B8BDC3',r'$\alpha=0.01$')]:
 bars=ax.bar(x+offset,counts,width=.34,color=color,label=label)
 ax.bar_label(bars,labels=[(f'{v:,}'.replace(',', '.') if args.language == 'es' else f'{v:,}') for v in counts],padding=3,fontsize=8)
ax.set(title=('Observaciones atípicas señaladas por método y umbral' if args.language == "es" else 'Flagged observations by method and threshold'),ylabel=('Número de observaciones' if args.language == "es" else 'Number of observations'),xticks=x,xticklabels=['PCA\nT²','PCA/ICA\nSPE','ICA\nNLL','Mahalanobis','LOF','Isolation\nForest','VAE'],ylim=(0,2300))
ax.legend(frameon=False,ncol=2,loc='upper center');ax.yaxis.grid(True,color='#EEEEEE',linewidth=.6);ax.set_axisbelow(True)
fig.savefig(figs/'06_conteo_anomalias_metodos.png',dpi=250);plt.close(fig)
pd.DataFrame({'Method':['PCA T2','PCA/ICA SPE','ICA NLL','Mahalanobis','LOF','Isolation Forest','VAE'],'alpha_005':a[0],'alpha_001':a[1]}).to_csv(figs/'06_conteo_anomalias_metodos.csv',index=False)
