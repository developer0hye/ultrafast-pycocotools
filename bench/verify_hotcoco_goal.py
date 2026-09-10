"""Verify the complete two-host performance matrix without dropping any runs."""
import argparse
import json
from pathlib import Path

from summarize_hotcoco import summarize


def verify(folder):
    summary = summarize(folder)
    assert summary['tasks'] == ['bbox', 'segm', 'keypoints']
    assert summary['modes'] == ['files', 'list'] and summary['threads'] == [1, 2]
    assert summary['rounds'] == 6 and summary['process_count'] == 174
    assert summary['timed_process_count'] == 144 and len(summary['table']) == 12
    for checks in summary['parity'].values():
        assert all(check['stats_max_abs_diff'] <= 1e-12 for check in checks.values())
    raw = json.loads((folder / 'results.json').read_text())
    warm = {(r['task'], r['mode'], r['threads'], r['impl']): r['result']
            for r in raw['runs'] if r['label'] == 'warmup'}
    oracle = {(r['task'], r['mode']): r['result']
              for r in raw['runs'] if r['label'] == 'oracle'}
    for run in raw['runs']:
        result = json.loads((folder / (run['name'] + '.json')).read_text())
        assert json.dumps(result, sort_keys=True) == json.dumps(run['result'], sort_keys=True)
        if not run['diagnostic']:
            expected = warm[(run['task'], run['mode'], run['threads'], run['impl'])]
            for field in ('digests', 'params', 'package_version', 'source_file_sha256', 'runtime_file_sha256'):
                assert result[field] == expected[field], (run['name'], field)
            if run['impl'] == 'ufcoco':
                assert result['stats'] == oracle[(run['task'], run['mode'])]['stats']
    for row in summary['table']:
        assert row['wall_speed_ratio'] > 1, ('wall-time goal not met', row)
        assert row['rss_reduction_percent'] > 0, ('peak RSS goal not met', row)
        paired = []
        for iteration in range(6):
            values = {r['impl']: r['result']['wall_total'] for r in raw['runs']
                      if (r['task'], r['mode'], r['threads'], r['label']) ==
                      (row['task'], row['mode'], row['threads'], f'round{iteration}')}
            paired.append(values['hotcoco'] / values['ufcoco'])
        row['paired_wall_speed_ratios'] = paired
        row['faster_pairs'] = sum(value > 1 for value in paired)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m2', type=Path, required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {host: verify(folder) for host, folder in [('m2', args.m2), ('server', args.server)]}
    for field in ('packages', 'input_sha256', 'script_sha256'):
        assert result['m2'][field] == result['server'][field], field
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    text = ['| Host | Task | Input | Pool size | Wall (s), hotcoco → candidate | Peak RSS (MB), hotcoco → candidate | Speed ratio | Faster pairs |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for host, data in result.items():
        for row in data['table']:
            cells = [host, row['task'], row['mode'], str(row['threads'])]
            for metric, digits in [('wall_total', 3), ('peak_rss_mb', 1)]:
                cells.append(f'{row["hotcoco"][metric]["median"]:.{digits}f} → {row["ufcoco"][metric]["median"]:.{digits}f}')
            cells.extend([f'{row["wall_speed_ratio"]:.2f}×', f'{row["faster_pairs"]}/6'])
            text.append('| ' + ' | '.join(cells) + ' |')
    (args.out / 'tables.md').write_text('\n'.join(text) + '\n')
    print('PASS: all 24 configurations have lower median wall time and peak RSS; 348 processes verified')


if __name__ == '__main__':
    main()
