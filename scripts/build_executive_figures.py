"""Build report figures from saved results, without fitting any models."""
from pathlib import Path
import hashlib
import json
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = ROOT / 'output/pdf'
FIGURES = OUT / 'figuras'
FIGURES.mkdir(parents=True, exist_ok=True)
SOURCES = {}

def record(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read(relative):
    return pd.read_csv(record(RESULTS / relative))

BLUE, TEAL, ORANGE, RED = '#22648A', '#208D88', '#C97929', '#B64C54'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 12,
    'axes.labelcolor': '#334155', 'text.color': '#173B56',
    'axes.edgecolor': '#B6C2CC', 'xtick.color': '#475569',
    'ytick.color': '#475569', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})

def page(title, subtitle):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.9))
    fig.subplots_adjust(left=.085, right=.97, bottom=.29, top=.76, wspace=.34)
    fig.text(.06, .94, title, fontsize=19, weight='bold')
    fig.text(.06, .875, subtitle, fontsize=11)
    return fig, axes

def finish(fig, name, interpretation, note, pdf):
    fig.text(.06, .175, textwrap.fill(interpretation, 138), fontsize=10.5,
             linespacing=1.5, va='top')
    fig.text(.06, .075, textwrap.fill(note, 160), fontsize=9, color='#64748B',
             linespacing=1.3, va='top')
    fig.savefig(FIGURES / f'{name}.png', dpi=300)
    pdf.savefig(fig)
    plt.close(fig)

def labels(ax, bars, fmt):
    ax.bar_label(bars, labels=[fmt(v.get_height()) for v in bars], padding=4, fontsize=10)

with PdfPages(OUT / 'figuras_comparativas.pdf') as pdf:
    # 1. Actual characterization of the chosen GMM, not grid-search cluster IDs.
    notebook_path = record(ROOT / 'notebooks/02_gmm_latent_variables_em.ipynb')
    nb = json.loads(notebook_path.read_text())
    profiles = []
    for cluster in range(4):
        cell = next(c for c in nb['cells'] if c['cell_type'] == 'code'
                    and f'analysis.analyze_cluster({cluster})' in ''.join(c['source']))
        output = '\n'.join(''.join(o.get('text', o.get('data', {}).get('text/plain', [])))
                           for o in cell.get('outputs', []))
        count = int(re.search(r'Number of observations: ([\d,]+)', output)[1].replace(',', ''))
        loss = float(re.search(r'Negative profit observations: ([\d.]+)%', output)[1])
        profiles.append((count, loss))
    counts, losses = np.array(profiles).T
    assert counts.sum() == 36104
    share = counts / counts.sum() * 100
    fig, (ax, bx) = page('01  Perfiles comerciales identificados por GMM',
                        'Cuatro componentes diagonales, n_init=10. Caracterización de 36.104 observaciones de validación.')
    names = ['C0\nUna unidad', 'C1\nDos unidades', 'C2\nPérdidas', 'C3\n3–5 unidades']
    bars = ax.bar(names, share, color=[BLUE, TEAL, RED, ORANGE], width=.62)
    labels(ax, bars, lambda x: f'{x:.2f}%')
    ax.set(title='Composición de los perfiles', ylabel='Porcentaje de validación', ylim=(0, 65))
    bars = bx.bar(names, losses, color=[BLUE, TEAL, RED, ORANGE], width=.62)
    labels(bx, bars, lambda x: f'{x:.2f}%')
    bx.set(title='Observaciones con beneficio negativo', ylabel='Porcentaje dentro de cada perfil', ylim=(0, 116))
    finish(fig, '01_perfiles_gmm',
           'El perfil de pérdidas reúne el 5,82% de validación y todas sus observaciones tienen beneficio negativo. '
           'Los perfiles restantes también incluyen pérdidas: cantidad y rentabilidad distinguen patrones, sin definir grupos completamente homogéneos.',
           'Fuente: notebook 02, salidas de caracterización del modelo elegido. Cada registro representa un ítem de pedido; no es un conteo de órdenes únicas.', pdf)

    # 2. Match five dimensions, scaling and validation split. Do not mix SVD or VAE.
    metrics = read('04_recommendation_ica_matrix_factorization/fa_ica_pca_reconstruction_metrics.csv')
    val = metrics[metrics.Dataset.eq('Validation')].set_index('Algorithm')
    ppca = read('03_svd_pca_probabilistic_pca_lsa/ppca_k5_reconstruction_metrics.csv').set_index('Dataset')
    models = ['PCA', 'ICA', 'PPCA', 'FA']
    errors = [val.loc['PCA', 'MSE'], val.loc['ICA', 'MSE'],
              ppca.loc['validation', 'Reconstruction MSE'], val.loc['FA', 'MSE']]
    pca = read('03_svd_pca_probabilistic_pca_lsa/pca_evaluation_by_k.csv')
    assert np.isclose(errors[0], errors[1])
    fig, (ax, bx) = page('02  Calidad de las representaciones latentes',
                        'Diez variables estandarizadas. Comparación con cinco dimensiones y la misma partición 70/15/15.')
    bars = ax.bar(models, errors, color=[BLUE, TEAL, ORANGE, '#8A729F'], width=.6)
    labels(ax, bars, lambda x: f'{x:.4f}')
    ax.set(title='Reconstrucción en validación', ylabel='MSE (menor es mejor)', ylim=(0, .145))
    bx.plot(pca.k, pca['Cumulative Explained Variance'] * 100, marker='o', color=BLUE, lw=2.3)
    selected = pca[pca.k.eq(5)].iloc[0]
    bx.scatter([5], [selected['Cumulative Explained Variance'] * 100], s=120, color=ORANGE, zorder=5)
    bx.annotate('5 componentes: 91,90%', (5, 91.897), xytext=(2.2, 67),
                arrowprops={'arrowstyle': '->', 'color': '#64748B'}, fontsize=11)
    bx.set(title='Información retenida por PCA', xlabel='Número de componentes',
           ylabel='Varianza acumulada de entrenamiento (%)', ylim=(0, 105), xticks=range(1, 10))
    finish(fig, '02_representaciones_latentes',
           'PCA e ICA obtienen el mismo MSE de reconstrucción con cinco dimensiones; sus coordenadas e interpretaciones pueden diferir. '
           'Cinco componentes PCA conservan el 91,90% de la varianza, reduciendo a la mitad la dimensionalidad.',
           'Fuentes: notebooks 03 y 04, tablas exportadas. Se excluyen SVD (escalamiento sin centrado) y VAE (otra partición y objetivo). Menor MSE no implica mayor interpretabilidad.', pdf)

    # 3. Same sample for original-space preservation and seed comparisons.
    folder = RESULTS / '05_manifolds_tsne_umap_embeddings'
    ts = read('05_manifolds_tsne_umap_embeddings/tsne_preservation_metrics_eval30000_ks2-5-10-30-50_seed42_9e0775b6d4bf.csv')
    um = read('05_manifolds_tsne_umap_embeddings/umap_preservation_metrics_eval30000_ks2-5-10-30-50_seed42_55d4f50e322d.csv')
    trow = ts[ts.Perplexity.eq(30)].iloc[0]
    urow = um[um.n_neighbors.eq(30) & np.isclose(um.min_dist, .1)].iloc[0]
    tstab = read('05_manifolds_tsne_umap_embeddings/tsne_seed_stability/summary.csv')
    latest = record(folder / 'umap_stability_latest_run.txt').read_text().strip()
    latest_path = Path(latest)
    if not latest_path.is_absolute():
        latest_path = folder / latest_path
    if not latest_path.exists():
        raise FileNotFoundError(latest_path)
    ustab = pd.read_csv(record(latest_path / 'summary.csv'))
    ks = [2, 5, 10, 30, 50]
    fig, (ax, bx) = page('03  t-SNE y UMAP: preservación y estabilidad',
                        'Muestra de 30.000 registros. t-SNE: perplexity=30. UMAP: n_neighbors=30 y min_dist=0,1.')
    for row, color, name in [(trow, BLUE, 't-SNE'), (urow, TEAL, 'UMAP')]:
        ax.plot(range(5), [row[f'Neighborhood Overlap@{k}'] for k in ks],
                marker='o', color=color, lw=2.2, label=name)
    ax.set(title='Preservación de vecinos del espacio original', xlabel='Tamaño del vecindario k',
           ylabel='Fracción de vecinos compartidos', xticks=range(5), xticklabels=ks, ylim=(0, 1))
    ax.legend(frameon=False)
    for frame, color, name in [(tstab, BLUE, 't-SNE'), (ustab, TEAL, 'UMAP')]:
        bx.errorbar(range(5), frame.mean_overlap, yerr=frame.pairwise_std, marker='o',
                    capsize=4, lw=2.2, color=color, label=name)
    bx.set(title='Acuerdo de vecinos entre diez semillas', xlabel='Tamaño del vecindario k',
           ylabel='Overlap medio entre embeddings', xticks=range(5), xticklabels=ks, ylim=(0, 1))
    bx.legend(frameon=False)
    finish(fig, '03_embeddings',
           f't-SNE conserva más vecinos exactos y muestra mayor acuerdo entre semillas. UMAP mejora su estabilidad al ampliar el vecindario. '
           f'El Spearman global es {trow["Global Spearman"]:.3f} para t-SNE y {urow["Global Spearman"]:.3f} para UMAP.',
           'Fuente: notebook 05, evaluación multik y última ejecución UMAP registrada. Barras: desviación estándar descriptiva de 45 pares; no son intervalos de confianza ni pares independientes.', pdf)

    # 4. Counts do not compare score magnitudes or claim accuracy.
    pca_q = read('06_anomaly_detection/pca_anomaly_quantile_comparison.csv')
    ica_q = read('06_anomaly_detection/ica_anomaly_quantile_comparison.csv')
    maha = read('06_anomaly_detection/mahalanobis_quantile_comparison.csv')
    lof = read('06_anomaly_detection/lof_k20_detection_summary.csv')
    forest = read('06_anomaly_detection/if_400trees_1024samples_detection_summary.csv')
    vae = read('07_autoencoder_vae/vae_validation_scores.csv')
    assert len(vae) == 36104
    rates = []
    def count(frame, alpha, col='Anomalies'):
        rows = frame[frame.Dataset.eq('Validation') & np.isclose(frame.Alpha, alpha)]
        assert len(rows) == 1
        return int(rows.iloc[0][col])
    for alpha, suffix in [(.05, '05'), (.01, '01')]:
        rates.append([count(pca_q, alpha, 'T2 Anomalies'), count(pca_q, alpha, 'SPE Anomalies'),
                      count(ica_q, alpha, 'NLL Anomalies'), count(maha, alpha), count(lof, alpha),
                      count(forest, alpha), int(vae[f'vae_anomaly_{suffix}'].sum())])
    if_pairs = read('06_anomaly_detection/if_seed_pairwise_stability.csv')
    vae_pairs = read('07_autoencoder_vae/vae_seed_pairwise_stability.csv')
    fig, (ax, bx) = page('04  Anomalías: amplitud de selección y estabilidad',
                        'Validación: 36.104 registros. Umbrales calculados en entrenamiento; scores altos indican mayor atipicidad.')
    names = ['PCA\nT²', 'PCA/ICA\nSPE', 'ICA\nNLL', 'Mahala-\nnobis', 'LOF', 'IF', 'VAE']
    x = np.arange(len(names))
    for off, values, color, name in [(-.19, rates[0], BLUE, 'α = 0,05'), (.19, rates[1], ORANGE, 'α = 0,01')]:
        bars = ax.bar(x + off, np.array(values) / 36104 * 100, width=.36, color=color, label=name)
        ax.bar_label(bars, labels=[f'{int(v):,}'.replace(',', '.') for v in values], padding=3, fontsize=8.5, rotation=90)
    ax.set(title='Tasa señalada por método y umbral', ylabel='Observaciones señaladas (%)',
           xticks=x, xticklabels=names, ylim=(0, 6.8))
    ax.tick_params(axis='x', labelsize=9)
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    series = [if_pairs.Spearman.to_numpy(), vae_pairs.spearman.to_numpy()]
    box = bx.boxplot(series, tick_labels=['Isolation Forest', 'VAE'], patch_artist=True,
                     widths=.42, medianprops={'color': '#173B56', 'linewidth': 2})
    for patch, color in zip(box['boxes'], [TEAL, ORANGE]):
        patch.set_facecolor(color);patch.set_alpha(.55)
    for i, vals in enumerate(series, 1):
        bx.scatter(np.full(len(vals), i) + np.linspace(-.085, .085, len(vals)), vals,
                   color=[TEAL, ORANGE][i-1], s=12, alpha=.45)
        bx.text(i, .08, f'Media: {vals.mean():.3f}', ha='center', fontsize=10)
    bx.set(title='Estabilidad del ranking entre semillas', ylabel='Correlación de Spearman', ylim=(0, 1.06))
    finish(fig, '04_anomalias',
           'El criterio del 1% reduce el número de registros señalados, sin demostrar mayor exactitud. IF mantiene rankings más consistentes '
           'que VAE en sus experimentos por semillas. Recuentos similares entre métodos no implican que detecten los mismos registros.',
           'Fuentes: notebooks 06 y 07. Etiquetas sobre barras: número de registros. SPE agrupa PCA e ICA por sus resultados coincidentes. Cada punto: un par de semillas; no se incluye LOF, evaluado por remuestreo.', pdf)

manifest = {'source_sha256': SOURCES,
            'notes': 'Read-only saved results; no fitting. GMM characterization from executed notebook outputs. '
                     'Figures compare matching tasks, not absolute anomaly scores. AE omitted: no completed grid result.',
            'figures': [p.name for p in sorted(FIGURES.glob('*.png'))]}
(OUT / 'figuras_fuentes.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print('Created 4 figures and figuras_comparativas.pdf; all source assertions passed.')
