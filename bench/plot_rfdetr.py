"""Plot the measured RF-DETR metric replay, including complete sample ranges."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    report = json.loads(Path('bench/results/rfdetr_nano.json').read_text())
    assert report['metric_all_keys_byte_identical']
    data = report['metric_backends']
    keys = ['faster-coco-eval', 'ultrafast']
    colors = ['#7261B5', '#087F8C']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.7))
    fig.subplots_adjust(left=.085, right=.97, bottom=.23, top=.69, wspace=.31)
    fig.text(.085, .94, 'RF-DETR: faster metric computation', fontsize=24, weight='bold')
    fig.text(.085, .877, 'Actual one-pass metric · Nano predictions · 5,000 COCO images · maxDets=500', color='#576879')
    for i, (ax, field, unit) in enumerate(zip(axes, ['median_seconds', 'median_peak_rss_MiB'], ['seconds', 'MiB'])):
        medians = [data[k][field] for k in keys]
        samples = [[r['total_seconds'] if i == 0 else r['peak_rss_MiB']
                    for r in data[k]['measurements']] for k in keys]
        errors = np.array([[m-min(v) for m,v in zip(medians,samples)],
                           [max(v)-m for m,v in zip(medians,samples)]])
        bars = ax.bar(['faster-coco-eval\n1.8.0', 'ultrafast\n0.1.4 + adapter'], medians,
                      color=colors, width=.55, yerr=errors, capsize=4)
        ax.bar_label(bars, labels=[f'{v:,.3f}' if i == 0 else f'{v:,.1f}' for v in medians], padding=12)
        ax.set_ylim(0, max(max(v) for v in samples)*1.27)
        ax.set_ylabel(unit)
        ax.set_title('Metric update + compute' if i == 0 else 'Peak process memory', loc='left', pad=12)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        headline = f'{medians[0]/medians[1]:.2f}× faster' if i == 0 else f'{100*(1-medians[1]/medians[0]):.1f}% less memory'
        fig.text(.085 if i == 0 else .588, .775, headline, fontsize=21, color=colors[1], weight='bold')
    fig.text(.085, .09, '3 fresh processes per backend; median and min–max. Two CPU threads on a shared EPYC 9554.', fontsize=10, color='#576879')
    fig.text(.085, .042, 'Aggregate and per-class tensors match exactly. Metric replay excludes inference and trace loading.', fontsize=10, color='#576879')
    for extension in ['png','svg','pdf']:
        fig.savefig(f'docs/assets/rfdetr-metric.{extension}', dpi=180)


if __name__ == '__main__':
    main()
