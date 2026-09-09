"""Verify finished hotcoco runs and regenerate their median comparison tables."""

import argparse
import json
from pathlib import Path

from hotcoco_benchmark import compare_arrays, digest, spread


def summarize(folder):
    data = json.loads((folder/'results.json').read_text())
    assert 'completed_utc' in data, 'benchmark still running'
    workloads = len(data['tasks']) * len(data['modes'])
    expected = workloads * (1 + 2*len(data['threads'])*(1+data['rounds']))
    assert len(data['runs']) == expected
    assert len({r['name'] for r in data['runs']}) == expected
    assert all(r['returncode'] == 0 for r in data['runs'])
    measured = [r for r in data['runs'] if not r['diagnostic']]
    assert all(r['stable_digests'] for r in measured)
    table = []
    for task in data['tasks']:
        for mode in data['modes']:
            oracle = folder/f'{task}-{mode}-t1-pycocotools-oracle.npz'
            for threads in data['threads']:
                pair = {}
                for impl in ('hotcoco', 'ufcoco'):
                    key = f'{task}-{mode}-t{threads}-{impl}'
                    check = data['parity'][task+'-'+mode][f'{impl}-t{threads}']
                    assert compare_arrays(oracle, folder/(key+'-warmup.npz')) == check['arrays']
                    assert check['same_params'], (key, 'different COCO parameters')
                    if impl == 'ufcoco':
                        assert all(a['byte_identical'] for a in check['arrays'].values())
                        assert check['stats_max_abs_diff'] == 0
                    rows = [r for r in measured if (r['task'],r['mode'],r['threads'],r['impl']) ==
                            (task,mode,threads,impl)]
                    assert len(rows) == data['rounds']
                    expected_summary = {k: spread([r['result'][k] for r in rows])
                                        for k in ('wall_total','cpu_total','eval_total','peak_rss_mb')}
                    assert expected_summary == data['summary'][key]
                    pair[impl] = expected_summary
                table.append(dict(task=task, mode=mode, threads=threads, **pair,
                                  wall_speed_ratio=pair['hotcoco']['wall_total']['median']/pair['ufcoco']['wall_total']['median'],
                                  rss_reduction_percent=100*(1-pair['ufcoco']['peak_rss_mb']['median']/pair['hotcoco']['peak_rss_mb']['median'])))
    samples = [s for r in measured for s in r['telemetry']]
    telemetry = dict(cpu_percent=spread([s['cpu_percent'] for s in samples]),
                     available_ram=spread([s['available_ram'] for s in samples]),
                     timed_runs_with_host_swap_out=sum(r['telemetry'][-1]['swap_out'] > r['telemetry'][0]['swap_out'] for r in measured),
                     total_host_swap_out_delta=data['runs'][-1]['telemetry'][-1]['swap_out']-data['baseline_load']['swap_out'],
                     total_host_swap_in_delta=data['runs'][-1]['telemetry'][-1]['swap_in']-data['baseline_load']['swap_in'])
    compact = {k:data[k] for k in ('started_utc','completed_utc','cpu','physical_cores','logical_cpus',
                                  'ram_bytes','platform','python','packages','input_sha256','script_sha256',
                                  'threads','rounds','modes','tasks','method','baseline_load','parity')}
    compact.update(table=table, telemetry=telemetry, process_count=expected, timed_process_count=len(measured),
                   raw_results_sha256=digest(folder/'results.json'))
    compact['measurements'] = [dict(name=r['name'], task=r['task'], mode=r['mode'], impl=r['impl'],
                                    threads=r['threads'], started_utc=r['telemetry'][0]['utc'],
                                    **{k:r['result'][k] for k in ('wall_total','cpu_total','eval_total','peak_rss_mb','digests')})
                               for r in measured]
    compact['runtime'] = {impl:next({k:r['result'][k] for k in ('package_version','python','numpy',
                                                              'runtime_file_sha256','source_file_sha256')}
                                   for r in data['runs'] if r['impl']==impl)
                          for impl in ('pycocotools','hotcoco','ufcoco')}
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m2', type=Path, required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {name:summarize(path) for name,path in [('m2',args.m2),('server',args.server)]}
    assert result['m2']['input_sha256'] == result['server']['input_sha256']
    assert result['m2']['packages'] == result['server']['packages']
    assert result['m2']['script_sha256'] == result['server']['script_sha256']
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    text = []
    for host,data in result.items():
        text.extend([f'## {host}: {data["cpu"]}', '',
                     'All values are medians. Each pair is hotcoco → ultrafast.', '',
                     '| Task | Input | Pool size | Wall (s) | CPU (s) | Peak RSS (MB) |',
                     '| --- | --- | ---: | ---: | ---: | ---: |'])
        for row in data['table']:
            cells = [row['task'],row['mode'],str(row['threads'])]
            for key in ('wall_total','cpu_total','peak_rss_mb'):
                digits = 1 if key=='peak_rss_mb' else 3
                cells.append(f'{row["hotcoco"][key]["median"]:.{digits}f} → {row["ufcoco"][key]["median"]:.{digits}f}')
            text.append('| '+' | '.join(cells)+' |')
        text.append('')
    (args.out/'tables.md').write_text('\n'.join(text)+'\n')
    print(args.out/'tables.md')


if __name__ == '__main__':
    main()
