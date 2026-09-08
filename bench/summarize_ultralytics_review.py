"""Fail on missing/different review evidence, then summarize saved raw measurements."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def span(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'samples': values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    execution = json.loads((a.results / 'execution.json').read_text())
    tasks = ('detect', 'segment', 'pose', 'lvis', 'lvis_segment')
    summary = {'absolute_tolerance': 1e-12, 'tasks': {}, 'raw_execution': execution}
    expected = 3 * execution['full_rounds'] * 2 + len(tasks) * (execution['replay_rounds'] + 1) * 2
    assert len(execution['execution']) == expected, (len(execution['execution']), expected)
    executions = {entry['name']: entry for entry in execution['execution']}
    assert len(executions) == expected
    assert all(entry['exit_code'] == 0 for entry in executions.values())
    for task in tasks:
        records = {}
        arrays = {}
        for revision in ('reference', 'replacement'):
            records[revision] = {}
            for mode, count in (('full', 0 if task.startswith('lvis') else execution['full_rounds']),
                                ('replay', execution['replay_rounds']), ('diagnostic', 1)):
                records[revision][mode] = [json.loads((a.results / f'{task}-{mode}-{i}-{revision}.json').read_text())
                                           for i in range(count)]
                for i, record in enumerate(records[revision][mode]):
                    assert f'{task}-{mode}-{i}-{revision}' in executions
                    assert len(record['calls']) == (1 if mode == 'full' else 3)
                    if mode != 'full':
                        assert all(call['metrics'] == record['calls'][0]['metrics'] for call in record['calls'])
                    assert record['diagnostic_only'] == (mode == 'diagnostic')
            arrays[revision] = np.load(a.results / f'{task}-diagnostic-0-{revision}.npz')
        result = {'arrays': {}, 'metrics': {}, 'performance': {}}
        assert set(arrays['reference'].files) == set(arrays['replacement'].files)
        for key in arrays['reference'].files:
            reference, replacement = arrays['reference'][key], arrays['replacement'][key]
            assert reference.shape == replacement.shape and reference.size
            assert np.isfinite(reference).all() and np.isfinite(replacement).all(), (task, key)
            np.testing.assert_allclose(replacement, reference, rtol=0, atol=1e-12)
            result['arrays'][key] = {'shape': list(reference.shape),
                                     'max_absolute_difference': float(np.max(np.abs(replacement - reference)))}
            if key.startswith('call0'):
                for values in arrays.values():
                    for i in (1, 2):
                        np.testing.assert_array_equal(values[key], values[key.replace('call0', f'call{i}')])
        for mode in ('full', 'replay', 'diagnostic'):
            metric_delta = 0.0
            for ref, actual in zip(records['reference'][mode], records['replacement'][mode]):
                assert ref['gt_sha256'] == actual['gt_sha256']
                assert ref['pred_sha256'] == actual['pred_sha256']
                for x, y in zip(ref['calls'], actual['calls']):
                    assert x['metrics'].keys() == y['metrics'].keys()
                    for key in x['metrics']:
                        delta = abs(x['metrics'][key] - y['metrics'][key])
                        assert delta <= 1e-12, (task, mode, key, delta)
                        metric_delta = max(metric_delta, delta)
                if mode == 'full':
                    assert ref['pre_evaluator_metrics'] == actual['pre_evaluator_metrics']
            if records['reference'][mode]:
                result['metrics'][mode] = {
                    'max_absolute_difference': metric_delta,
                    'reference': records['reference'][mode][0]['calls'][0]['metrics'],
                    'replacement': records['replacement'][mode][0]['calls'][0]['metrics'],
                }
        # Replays must reproduce the COCO metrics returned by real validation,
        # not merely agree with one another on a different set of images.
        for revision, modes in records.items():
            anchor = modes['full'][0] if modes['full'] else modes['replay'][0]
            for runs in modes.values():
                for record in runs:
                    assert record['gt_sha256'] == anchor['gt_sha256']
                    assert record['pred_sha256'] == anchor['pred_sha256']
                    for call in record['calls']:
                        for key, value in call['metrics'].items():
                            assert abs(value - anchor['calls'][0]['metrics'][key]) <= 1e-12, (task, revision, key)
        ref_stats = records['reference']['diagnostic'][0]['evaluator_statistics']
        new_stats = records['replacement']['diagnostic'][0]['evaluator_statistics']
        assert ref_stats.keys() == new_stats.keys()
        result['evaluator_statistics'] = {}
        for call, stats in ref_stats.items():
            # The libraries expose different summary aliases. Compare shared
            # names here; the full validator key set and arrays are gated above.
            shared = stats.keys() & new_stats[call].keys()
            maximum = 0.0
            for key in shared:
                value = stats[key]
                delta = abs(value - new_stats[call][key])
                assert delta <= 1e-12, (task, call, key, delta)
                maximum = max(maximum, delta)
            result['evaluator_statistics'][call] = {
                'shared_keys': sorted(shared), 'shared_max_difference': maximum,
                'reference_only_keys': sorted(stats.keys() - new_stats[call].keys()),
                'replacement_only_keys': sorted(new_stats[call].keys() - stats.keys()),
            }
        for revision, modes in records.items():
            perf = {'replay_cold_seconds': span([x['calls'][0]['seconds'] for x in modes['replay']]),
                    'replay_cached_seconds': span([c['seconds'] for x in modes['replay'] for c in x['calls'][1:]]),
                    'replay_peak_rss_mib': span([x['peak_rss_mib'] for x in modes['replay']])}
            if modes['full']:
                perf.update(whole_validation_seconds=span([x['calls'][0]['seconds'] for x in modes['full']]),
                            evaluator_within_validation_seconds=span([x['evaluator']['seconds'] for x in modes['full']]),
                            whole_validation_peak_rss_mib=span([x['peak_rss_mib'] for x in modes['full']]))
            result['performance'][revision] = perf
            for mode in ('full', 'replay'):
                entries = [executions[f'{task}-{mode}-{i}-{revision}'] for i in range(len(modes[mode]))]
                if entries:
                    perf[f'{mode}_fresh_process_seconds'] = span([entry['fresh_process_seconds'] for entry in entries])
                    samples = [sample for entry in entries for sample in entry['samples']]
                    perf[f'{mode}_host_cpu_percent'] = span([sample['host_cpu_percent'] for sample in samples])
                    perf[f'{mode}_available_ram_gib'] = span([sample['available_ram'] / 2**30 for sample in samples])
                    perf[f'{mode}_sampled_tree_peak_rss_mib'] = span([
                        max(sample['tree_rss'] for sample in entry['samples']) / 2**20 for entry in entries])
        result['raw_records'] = records
        summary['tasks'][task] = result
    a.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({task: {'metrics': value['metrics'], 'performance': value['performance']}
                      for task, value in summary['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
