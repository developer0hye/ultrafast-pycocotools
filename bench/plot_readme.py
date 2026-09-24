"""Plot the README headline chart from a multi-backend hotcoco_benchmark.py run.

Writes light and dark SVGs. ultrafast is the accent colour; the other backends
share one neutral colour, so every bar is identified by its axis label.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BACKENDS = [('pycocotools', 'pycocotools'), ('faster', 'faster-coco-eval'),
            ('hotcoco', 'hotcoco'), ('ufcoco', 'ultrafast')]
DISTRIBUTIONS = {'pycocotools': 'pycocotools', 'faster': 'faster-coco-eval',
                 'hotcoco': 'hotcoco', 'ufcoco': 'ultrafast-pycocotools'}
TASKS = [('bbox', 'Bounding boxes'), ('segm', 'Segmentation'), ('keypoints', 'Keypoints')]
METRICS = [('wall_total', 'Wall time (s)', '{:.2f}'), ('peak_rss_mb', 'Peak RSS (MB)', '{:,.0f}')]
THEMES = {
    'light': dict(surface='#fcfcfb', text='#0b0b0b', muted='#52514e', grid='#e4e3df',
                  accent='#2a78d6', neutral='#8d8c86'),
    'dark': dict(surface='#1a1a19', text='#ffffff', muted='#c3c2b7', grid='#383835',
                 accent='#3987e5', neutral='#6b6a64'),
}


def cpu_name(raw):
    return raw.replace('(R)', '').replace('(TM)', '').replace(' CPU', '').split(' @ ')[0]


def plot(result, theme, out):
    colors = THEMES[theme]
    summary, threads = result['summary'], result['threads'][0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'text.color': colors['text'], 'axes.labelcolor': colors['muted'],
                         'xtick.color': colors['muted'], 'ytick.color': colors['text']})
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4), facecolor=colors['surface'])
    fig.subplots_adjust(left=.12, right=.97, top=.8, bottom=.1, hspace=.6, wspace=.35)
    for row, (metric, metric_label, fmt) in enumerate(METRICS):
        for col, (task, task_label) in enumerate(TASKS):
            ax = axes[row, col]
            ax.set_facecolor(colors['surface'])
            values = [summary[f'{task}-files-t{threads}-{key}'][metric]['median'] for key, _ in BACKENDS]
            bar_colors = [colors['accent'] if key == 'ufcoco' else colors['neutral'] for key, _ in BACKENDS]
            y = range(len(BACKENDS))
            ax.barh(y, values, color=bar_colors, height=.62)
            for position, value in zip(y, values):
                ax.annotate(fmt.format(value), (value, position), xytext=(4, 0), textcoords='offset points',
                            va='center', fontsize=9, color=colors['text'])
            ax.set_yticks(list(y), [label for _, label in BACKENDS] if col == 0 else [])
            ax.invert_yaxis()
            ax.set_xlim(0, max(values) * 1.3)
            ax.tick_params(axis='both', length=0)
            ax.grid(axis='x', color=colors['grid'], linewidth=.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f'{task_label} · {metric_label}', loc='left', fontsize=10.5,
                         color=colors['text'], pad=6)
    versions = ', '.join(f'{DISTRIBUTIONS[key]} {result["packages"][DISTRIBUTIONS[key]]}'
                         for key, _ in BACKENDS)
    fig.text(.12, .94, 'COCO val2017 evaluation, 5,000 images (lower is better)',
             fontsize=13, weight='bold', color=colors['text'])
    fig.text(.12, .89, f'{cpu_name(result["cpu"])} · {threads} threads · median of {result["rounds"]} fresh processes'
             ' · JSON loading included', fontsize=9, color=colors['muted'])
    fig.text(.12, .02, versions, fontsize=8, color=colors['muted'])
    fig.savefig(out, facecolor=colors['surface'])
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='results.json from hotcoco_benchmark.py')
    parser.add_argument('--out', type=Path, default=Path('docs/assets/readme-benchmark'))
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        plot(result, theme, args.out.with_name(f'{args.out.name}-{theme}.svg'))


if __name__ == '__main__':
    main()
