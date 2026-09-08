"""Plot measured file-input evaluation medians with the complete run ranges."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=Path('bench/results/efficiency_v013.json'))
    p.add_argument('--out', type=Path, default=Path('docs/assets/compact-yolo26n'))
    a = p.parse_args()
    report = json.loads(a.results.read_text())
    data = report['yolo26n_file_backends']
    keys = ['pycocotools', 'faster-coco-eval', 'ultrafast']
    labels = ['pycocotools\n2.0.11', 'faster-coco-eval\n1.8.0', 'ultrafast\n0.1.3']
    colors = ['#D46346', '#7261B5', '#087F8C']
    assert report['cases']['yolo26n']['all_samples_byte_identical']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.075, right=.97, bottom=.23, top=.69, wspace=.28)
    fig.text(.075, .94, 'YOLO26n: exact COCO evaluation from files', fontsize=22, weight='bold')
    fig.text(.075, .875, '5,000 images · 596,202 predictions · JSON loading included', color='#576879')
    fields = ['median_seconds', 'median_peak_rss_MiB']
    for i, (ax, field, unit) in enumerate(zip(axes, fields, ['seconds', 'MiB'])):
        medians = [data[k][field] for k in keys]
        samples = [[r['runs'][0]['total_scoring_seconds'] if i == 0 else r['peak_rss_MiB']
                    for r in data[k]['measurements']] for k in keys]
        errors = np.array([[m - min(s) for m, s in zip(medians, samples)],
                           [max(s) - m for m, s in zip(medians, samples)]])
        bars = ax.bar(labels, medians, color=colors, width=.62, yerr=errors,
                      capsize=4, error_kw={'elinewidth': 1})
        ax.bar_label(bars, labels=[f'{x:,.2f}' if i == 0 else f'{x:,.1f}' for x in medians],
                     padding=9, fontsize=11)
        ax.set_ylim(0, max(max(s) for s in samples) * 1.28)
        ax.set_ylabel(unit)
        ax.set_title('Evaluation time' if i == 0 else 'Peak process memory', loc='left', pad=12)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        headline = (f'{medians[0] / medians[2]:.1f}× faster' if i == 0 else
                    f'{100 * (1 - medians[2] / medians[0]):.1f}% less memory')
        fig.text(.075 if i == 0 else .574, .77, headline + ' vs pycocotools',
                 color=colors[2], fontsize=16, weight='bold')
    fig.text(.075, .085, 'Medians of 3 fresh processes; whiskers show min–max. Two CPU threads; shared EPYC 9554 host.',
             fontsize=10, color='#576879')
    fig.text(.075, .04, 'ultrafast arrays match pycocotools byte for byte. Source: bench/results/efficiency_v013.json',
             fontsize=10, color='#576879')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(a.out.with_suffix('.' + suffix), dpi=180)


if __name__ == '__main__':
    main()
