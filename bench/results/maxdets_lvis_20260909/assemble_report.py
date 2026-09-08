"""Freeze measured records and verify exact results before computing medians."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics as st
import subprocess

import numpy as np

root = Path.cwd()
raw = root / 'bench/out/mask-cap-lvis'
out = root / 'bench/results/maxdets_lvis_20260909'
out.mkdir(exist_ok=True)
archive = root.parent / 'ultrafast-pycocotools/bench/out/ultralytics-pr26101-evidence-20260909'
baseline = '41d0cad777263329880c3a9c75b129d897546dde'
candidate = '0a0de58d11f8b7404f5f2885ee72f7385e61d2e7'
final_m2 = '26cec60da379422ab9c8e4b7e05f495dcc55f396'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def read(p):
    return json.loads(p.read_text())


def save(name, value):
    (out / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


report = dict(baseline_runtime_revision=baseline, candidate_runtime_revisions=dict(server=candidate, m2=final_m2),
              package_note='Server and earlier M2 source builds report 0.1.7; final M2 source build reports 0.1.9. Baseline runtime is packaged in 0.1.8. These are controlled source builds, not public 0.1.7 wheel measurements.',
              aggregation='Median of six fresh processes per variant/task; cached time first averages the two cached calls in each process. Diagnostics are excluded.',
              normalization='Unavailable nonfinite legacy CPU readings become JSON null; original-file hashes identify unmodified local raw files.',
              scopes={}, original_file_sha256={}, input_sha256={})
scopes = [('m2-cap', 'exploratory cap-only build, no frozen commit'),
          ('m2-final', 'superseded: some LVIS mask samples overlap D-FINE checks; runtime 9e22d89'),
          ('m2-counted-quiet', 'superseded memory protocol: peak RSS included array hash copies'),
          ('m2-final-v019', 'final: version 0.1.9, unaligned-array guard, peak RSS before verification copies'),
          ('server-preparation', 'exploratory preparation-only build, no frozen commit'),
          ('server-summary', 'combined preparation and summary runtime 9e22d89'),
          ('server-counted', 'final counted-group runtime')]
for folder, status in scopes:
    directory = raw / folder
    execution = read(directory/'execution.json')
    save(folder+'-execution.json', execution)
    scope = {'status': status, 'tasks': {}}
    tasks = sorted({re.match(r'(.+)-(\d+)-(baseline|candidate)\.json', p.name)[1]
                    for p in directory.glob('*.json') if re.match(r'(.+)-(\d+)-(baseline|candidate)\.json', p.name)})
    is_server = folder.startswith('server')
    samples = [s for e in execution if int(e['name'].rsplit('-', 2)[1]) < 6 for s in e['samples']]
    scope['host_telemetry'] = dict(median_cpu_percent=st.median(s['host_cpu_percent'] for s in samples),
                                 max_cpu_percent=max(s['host_cpu_percent'] for s in samples),
                                 min_available_ram_bytes=min(s['available_ram'] for s in samples))
    runtime_fingerprints = set()
    for task in tasks:
        records = {v: [read(directory/f'{task}-{i}-{v}.json') for i in range(6)]
                   for v in ('baseline', 'candidate')}
        reference = records['baseline'][0]
        medians = {}
        for variant, ds in records.items():
            for i, d in enumerate(ds):
                p = directory/f'{task}-{i}-{variant}.json'
                report['original_file_sha256'][str(p.relative_to(root))] = sha(p)
                if is_server:
                    assert d['gt_sha256'] == reference['gt_sha256'] and d['pred_sha256'] == reference['pred_sha256']
                    assert all(c['metrics'] == reference['calls'][0]['metrics'] for c in d['calls'])
                else:
                    assert d['stats'] == reference['stats'] and d['digests'] == reference['digests']
                if folder in ('m2-counted-quiet', 'm2-final-v019'):
                    runtime_fingerprints.add(json.dumps(d['runtime_file_sha256'], sort_keys=True))
                    revision = baseline if variant == 'baseline' else (final_m2 if folder == 'm2-final-v019' else candidate)
                    for name, digest in d['source_file_sha256'].items():
                        if name.endswith(('/coco.py', '/cocoeval.py', '/_lvis.py')):
                            contents = subprocess.check_output(['git', 'show', revision+':python/ultrafast_pycocotools/'+Path(name).name])
                            assert hashlib.sha256(contents).hexdigest() == digest
            if is_server:
                assert len({json.dumps(d['native_hashes'],sort_keys=True) for d in ds}) == 1
                medians[variant] = dict(cold_seconds=st.median(d['calls'][0]['seconds'] for d in ds),
                    cached_seconds=st.median(st.mean(c['seconds'] for c in d['calls'][1:]) for d in ds),
                    peak_rss_mib=st.median(d['peak_rss_mib'] for d in ds))
            else:
                medians[variant] = dict(load_and_evaluate_seconds=st.median(d['wall_total'] for d in ds),
                    summary_seconds=st.median(d['timings']['summarize'] for d in ds),
                    peak_rss_decimal_mb=st.median(d['peak_rss_mb'] for d in ds))
        item = dict(medians=medians, all_metrics_equal=True, records=records)
        diag = {v: read(directory/f'{task}-6-{v}.json') for v in records
                if (directory/f'{task}-6-{v}.json').exists()}
        if diag:
            item['diagnostics'] = diag
            if is_server:
                public = read(archive/'formal'/f'{task}-diagnostic-0-replacement.json')
                for d in diag.values():
                    assert d['evaluator_statistics'] == public['evaluator_statistics']
                    assert all(c['metrics'] == public['calls'][0]['metrics'] for c in d['calls'])
                item['all_evaluator_statistics_equal_public_archive'] = True
            else:
                a_path = directory/f'{task}-6-baseline.npz'
                b_path = directory/f'{task}-6-candidate.npz'
                with np.load(a_path) as a, np.load(b_path) as b:
                    assert a.files == b.files
                    item['complete_arrays'] = {}
                    for key in a.files:
                        assert a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes()
                        item['complete_arrays'][key] = dict(shape=list(a[key].shape), dtype=str(a[key].dtype),
                            sha256=hashlib.sha256(a[key].tobytes()).hexdigest(),max_absolute_difference=0)
                item['npz_sha256'] = {p.name: sha(p) for p in (a_path,b_path)}
        scope['tasks'][task] = item
    if runtime_fingerprints:
        assert len(runtime_fingerprints) == 1
        scope['all_36_timed_runtime_fingerprints_equal'] = True
        scope['runtime_file_sha256'] = json.loads(next(iter(runtime_fingerprints)))
    report['scopes'][folder] = scope

server = read(raw/'server-provenance.json')
for variant, revision in [('baseline', baseline), ('counted', candidate)]:
    for name, digest in server['builds'][variant]['installed_sha256'].items():
        if name.endswith('.py'):
            contents = subprocess.check_output(['git','show',revision+':python/ultrafast_pycocotools/'+name])
            assert hashlib.sha256(contents).hexdigest() == digest
for entry in server['npz'].values():
    p = archive/entry['public_archive_path']
    assert sha(p) == entry['sha256'] and entry['byte_identical']
report['public_arrays'] = dict(archive_url='https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz',
    archive_sha256='8c9dd4d1cf3a0a6e1b4e83a49a7b799023756294a2108fc7389efffc5aafa33e',diagnostics=server['npz'])
for name in ['instances_val2017.json','segment-predictions.json','lvis_gt_100.json','lvis_dt_100.json',
             'lvis_gt_93_annotated.json','lvis_yolo26n_seg_93_annotated.json']:
    report['input_sha256'][name] = sha(archive/'inputs'/name)
old = read(root/'bench/results/nonbbox_buffers_20260909/m2-native-alloc-g.json')['rust']
new = read(raw/'native-alloc.json')['rust']
report['native_allocations'] = dict(baseline_revision='506c8213985bafcbd46b88120aa28ec1354c7213',
    candidate_revision=candidate, predicted_removed_rle_payload=20557564,
    counters={k:dict(before=old[k],after=new[k],reduction=old[k]-new[k])
              for k in ['live_bytes','peak_bytes','total_allocated_bytes','allocations']},
    conditional_subtotal_bytes=766722083,performance_timings=False)
for name in ['local-provenance.json','server-provenance.json','native-alloc.json',
             'native-alloc.log','tests-counted-cap.log','rust-counted-cap.log','tests-summary.log',
             'tests-pre-summary.log','zero-cap-before.log','dfine-tests.log',
             'paired_cap.py','paired_m2_final.py','paired_m2_counted.py','paired_preparation.py',
             'paired_server_final.py','paired_server_counted.py','paired_m2_v019.py','tests-v019.log']:
    shutil.copy2(raw/name,out/name)
save('report.json',report)
print('PASS: complete arrays, every metric/statistic, source/runtime hashes and public NPZ identity')
for name in ['m2-final-v019','server-counted']:
    scope=report['scopes'][name]
    print(name,scope['host_telemetry'])
    for task,item in scope['tasks'].items():print(task,item['medians'])
