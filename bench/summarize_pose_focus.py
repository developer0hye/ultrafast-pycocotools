"""Verify pose before/after measurements and the matched hotcoco comparisons."""
import argparse
import json
from pathlib import Path

from hotcoco_benchmark import digest, spread
from summarize_hotcoco import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for host in ('m2', 'server'):
        pair_dir = args.evidence / (host + '-pairs')
        runs = json.loads((pair_dir / 'execution.json').read_text())
        assert len(runs) == 24 and len({r['name'] for r in runs}) == 24
        hot = summarize(args.evidence / (host + '-hotcoco'))
        assert hot['tasks'] == ['keypoints'] and hot['threads'] == [1, 2] and hot['rounds'] == 6
        assert hot['process_count'] == 58 and hot['timed_process_count'] == 48
        candidate_runtime = hot['runtime']['ufcoco']['source_file_sha256']
        candidate_binaries = {v for k, v in candidate_runtime.items() if k.endswith(('.so', '.pyd'))}
        reference = hot['measurements'][0]['digests']
        pairs = []
        for mode in ('files', 'list'):
            row = dict(mode=mode)
            for variant in ('baseline', 'candidate'):
                selected = [r for r in runs if r['name'].startswith(mode + '-') and r['name'].endswith('-' + variant)]
                assert len(selected) == 6
                for run in selected:
                    result = json.loads((pair_dir / (run['name'] + '.json')).read_text())
                    assert json.dumps(result, sort_keys=True) == json.dumps(run['result'], sort_keys=True)
                    assert result['threads'] == '2' and result['iou_type'] == 'keypoints'
                    assert result['digests'] == reference and result['diagnostic_only'] is False
                    assert result['params']['maxDets'] == [20]
                    assert result['stats'] == runs[0]['result']['stats']
                    if variant == 'candidate':
                        binaries = {v for k, v in result['source_file_sha256'].items() if k.endswith(('.so', '.pyd'))}
                        assert binaries == candidate_binaries
                row[variant] = {key: spread([r['result'][key] for r in selected])
                                for key in ('wall_total', 'cpu_total', 'eval_total', 'peak_rss_mb')}
            row['wall_reduction_percent'] = 100 * (1 - row['candidate']['wall_total']['median'] / row['baseline']['wall_total']['median'])
            row['rss_reduction_percent'] = 100 * (1 - row['candidate']['peak_rss_mb']['median'] / row['baseline']['peak_rss_mb']['median'])
            pairs.append(row)
        report[host] = dict(before_after=pairs, hotcoco=hot,
                            pairs_sha256=digest(pair_dir / 'execution.json'))
    a, b = [report[h]['hotcoco'] for h in ('m2', 'server')]
    for key in ('packages', 'input_sha256', 'script_sha256'):
        assert a[key] == b[key], key
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['| Host | Input | Wall (s), before → after | CPU (s), before → after | Peak RSS (MB), before → after |',
             '| --- | --- | ---: | ---: | ---: |']
    for host, data in report.items():
        for row in data['before_after']:
            cells = [host, row['mode']]
            for key in ('wall_total', 'cpu_total', 'peak_rss_mb'):
                digits = 1 if key == 'peak_rss_mb' else 3
                cells.append(f'{row["baseline"][key]["median"]:.{digits}f} → {row["candidate"][key]["median"]:.{digits}f}')
            lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '| Host | Input | Pool size | Wall (s), hotcoco → candidate | Peak RSS (MB), hotcoco → candidate |',
              '| --- | --- | ---: | ---: | ---: |']
    for host, data in report.items():
        for row in data['hotcoco']['table']:
            cells = [host, row['mode'], str(row['threads'])]
            for key in ('wall_total', 'peak_rss_mb'):
                digits = 1 if key == 'peak_rss_mb' else 3
                cells.append(f'{row["hotcoco"][key]["median"]:.{digits}f} → {row["ufcoco"][key]["median"]:.{digits}f}')
            lines.append('| ' + ' | '.join(cells) + ' |')
    (args.out / 'tables.md').write_text('\n'.join(lines) + '\n')
    print('Verified 48 before/after runs and 116 comparison/oracle processes on two hosts')


if __name__ == '__main__':
    main()
