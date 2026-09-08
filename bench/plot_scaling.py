"""Render measured scaling results as a paper-ready PNG, SVG and PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('bench/results/scaling.json'))
    parser.add_argument('--out', type=Path, default=Path('docs/assets/scaling'))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    points = sorted(data['points'], key=lambda p: p['annotations'] + p['detections'])
    if len(points) < 2 or not all(all(p['byte_identical'].values()) for p in points):
        raise ValueError('Need at least two measured points with complete array parity')
    x = [(p['annotations'] + p['detections']) / 1e6 for p in points]
    backends = ('reference', 'faster_coco_eval', 'ultrafast') if all('faster_coco_eval' in p for p in points) else ('reference', 'ultrafast')
    comparator = 'faster_coco_eval' if 'faster_coco_eval' in backends else 'reference'
    names = {'reference': 'pycocotools', 'faster_coco_eval': 'faster-coco-eval', 'ultrafast': 'ultrafast-pycocotools'}
    times = {b: [p[b]['runs'][0]['total_scoring_seconds'] for p in points] for b in backends}
    memory = {b: [p[b]['peak_rss_MiB'] / 1024 for p in points] for b in backends}
    colors = {'reference': '#D46346', 'faster_coco_eval': '#7261B5', 'ultrafast': '#087F8C'}
    ink, muted = '#182D40', '#576879'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelcolor': ink, 'text.color': ink,
                         'xtick.color': muted, 'ytick.color': muted,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5))
    fig.subplots_adjust(left=.075, right=.965, bottom=.23, top=.69, wspace=.27)
    fig.text(.075, .933, 'COCO evaluation at scale', fontsize=26, weight='bold')
    fig.text(.075, .879, 'Objects365 v2  /  synthetic predictions  /  agreement checked against pycocotools', fontsize=11, color=muted)
    legend = [Line2D([0], [0], color=colors[b], marker='o', lw=2.6, markersize=5,
                     label=names[b]) for b in backends]
    fig.legend(handles=legend, loc='upper left', bbox_to_anchor=(.07, .835), frameon=False, ncol=len(backends), columnspacing=2.5)
    for index, (ax, values, title, unit) in enumerate(zip(axes, [times, memory],
                ['Evaluation time', 'Peak process memory'], ['seconds', 'GiB'])):
        high = max(v for b in backends for v in values[b])
        for backend in backends:
            ax.plot(x, values[backend], color=colors[backend], lw=2.8,
                    marker='o', markersize=5.8, markeredgecolor='white', markeredgewidth=1.2, zorder=3)
        ax.fill_between(x, values['ultrafast'], values['reference'], color=colors['reference'], alpha=.045)
        ax.set_xlim(0, max(x) * 1.11)
        ax.set_ylim(0, high * 1.18)
        ax.set_title(f'{chr(97 + index)}   {title}', loc='left', fontsize=12, weight='bold', pad=17)
        ax.set_ylabel(unit, fontsize=10, labelpad=10)
        ax.set_xlabel('GT + prediction boxes (millions)', fontsize=10, labelpad=11)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis='y', color='#E8EDF1', lw=.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for edge in ('left', 'bottom'):
            ax.spines[edge].set_color('#CED7DF')
        ax.tick_params(length=0, pad=7)
        ratio = values[comparator][-1] / values['ultrafast'][-1]
        benefit = 'faster' if index == 0 else 'lower peak memory'
        ax.text(.035, .93, f'{ratio:.1f}×', transform=ax.transAxes,
                fontsize=25, color=colors['ultrafast'], weight='bold', va='top')
        ax.text(.035, .78, f'{benefit} vs {names[comparator]} (largest input)', transform=ax.transAxes,
                fontsize=9, color=muted)
        for backend in backends:
            value = values[backend][-1]
            suffix = 's' if index == 0 else ' GiB'
            label = f'{value:.1f}{suffix}' if index == 0 else f'{value:.2f}{suffix}'
            ax.annotate(label, (x[-1], value), xytext=(-7, 11), textcoords='offset points',
                        ha='right', color=colors[backend], weight='bold', fontsize=10)
    last = points[-1]
    fig.text(.075, .102, f"{points[0]['images']:,} → {last['images']:,} images   ·   "
             f"up to {last['annotations']:,} GT + {last['detections']:,} prediction boxes   ·   365 categories", fontsize=10, color=muted)
    fig.text(.075, .062, '2 CPU cores per scorer · one run per size · linear axes · measured points connected; no fitted curves', fontsize=9, color=muted)
    footer = 'Time excludes JSON parsing and inference. Memory is whole-process peak RSS. Shared host.'
    if any('previously published' in p.get('measurement_source', '') for p in points):
        footer += ' Prior reference/ultrafast runs reused.'
    fig.text(.075, .029, footer, fontsize=8.5, color=muted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        path = args.out.with_suffix('.' + extension)
        fig.savefig(path, dpi=240)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
