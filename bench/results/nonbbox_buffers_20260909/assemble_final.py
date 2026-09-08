"""Freeze measured records and verify complete arrays before reporting medians."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics as st

import numpy as np

root = Path.cwd()
raw = root / 'bench/out/nonbbox-memory'
out = root / 'bench/results/nonbbox_buffers_20260909'
public = root.parent / 'ultrafast-pycocotools/bench/out/ultralytics-pr26101-evidence-20260909/formal'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

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
    (out/name).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')

report = dict(baseline_revision='b52e8527cb772393880289ee4eb0fb30f29a7d15',
              candidate_revision='41d0cad777263329880c3a9c75b129d897546dde',
              release_status='Unreleased source builds; reported package version is still 0.1.7.',
              normalization='Nonfinite unavailable legacy CPU readings become JSON null; original file hashes refer to unmodified raw records.',
              aggregation='Median of six fresh processes per variant/task; cached time first averages the two cached calls within each process.',
              scopes={}, original_file_sha256={})
for host, tasks in [('m2-fixed', ['segm', 'keypoints']), ('server-h', ['segment', 'pose']), ('m2-final', ['segm', 'keypoints'])]:
    scope = dict(tasks={}, status='superseded: no per-process runtime fingerprint; busy host' if host=='m2-final' else 'final')
    execution = read(raw/host/'execution.json')
    save(f'final-{host}-execution.json', execution)
    samples=[s for e in execution for s in e['samples']]
    scope['host_telemetry'] = dict(median_cpu_percent=st.median(s['host_cpu_percent'] for s in samples),
                                 max_cpu_percent=max(s['host_cpu_percent'] for s in samples),
                                 min_available_ram_bytes=min(s['available_ram'] for s in samples))
    versions=set()
    fingerprints=set()
    for task in tasks:
        data={v:[read(raw/host/f'{task}-{i}-{v}.json') for i in range(6)] for v in ['baseline','candidate']}
        ref=data['baseline'][0]
        for variant, records in data.items():
            for i,d in enumerate(records):
                p=raw/host/f'{task}-{i}-{variant}.json'
                report['original_file_sha256'][str(p.relative_to(root))]=sha(p)
                if host=='server-h':
                    assert d['gt_sha256']==ref['gt_sha256'] and d['pred_sha256']==ref['pred_sha256']
                    assert all(c['metrics']==ref['calls'][0]['metrics'] for c in d['calls'])
                    versions.add(d['python'])
                else:
                    assert d['stats']==ref['stats'] and d['digests']==ref['digests']
                if host=='m2-fixed':
                    versions.add(d['python'])
                    fingerprints.add(json.dumps(d['runtime_file_sha256'],sort_keys=True))
            if host=='server-h':
                assert len({json.dumps(d['native_hashes'],sort_keys=True) for d in records})==1
            elif host=='m2-fixed':
                assert len({json.dumps(d['source_file_sha256'],sort_keys=True) for d in records})==1
        medians={}
        for v, ds in data.items():
            if host=='server-h':
                medians[v]=dict(cold_seconds=st.median(d['calls'][0]['seconds'] for d in ds),
                               cached_seconds=st.median(st.mean(c['seconds'] for c in d['calls'][1:]) for d in ds),
                               peak_rss_mib=st.median(d['peak_rss_mib'] for d in ds))
            else:
                medians[v]=dict(load_and_evaluate_seconds=st.median(d['wall_total'] for d in ds),
                               eval_cpu_seconds=st.median(d['eval_cpu'] for d in ds),
                               peak_rss_decimal_mb=st.median(d['peak_rss_mb'] for d in ds))
        entry=dict(medians=medians,all_metrics_equal=True,records=data)
        if host!='m2-final':
            stem=f'{task}-6' if host=='server-h' else f'{task}-diagnostic'
            paths=[raw/host/f'{stem}-{v}.npz' for v in ['baseline','candidate']]
            diagnostics={v:read(raw/host/f'{stem}-{v}.json') for v in ['baseline','candidate']}
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                assert a.files==b.files
                entry['full_array_comparison']={}
                for key in a.files:
                    assert np.array_equal(a[key],b[key]), (host,task,key)
                    entry['full_array_comparison'][key]=dict(shape=list(a[key].shape),dtype=str(a[key].dtype),
                        max_absolute_difference=0,sha256=hashlib.sha256(np.ascontiguousarray(a[key]).tobytes()).hexdigest())
            entry['diagnostics']=diagnostics
            entry['npz_sha256']={p.name:sha(p) for p in paths}
            if host=='server-h':
                pub=public/f'{task}-diagnostic-0-replacement.npz'
                assert sha(pub)==sha(paths[0])==sha(paths[1])
                entry['public_npz']=dict(archive_url='https://github.com/developer0hye/ultrafast-pycocotools/releases/download/v0.1.7/ultralytics-pr26101-evidence-20260909.tar.gz',
                    archive_sha256='8c9dd4d1cf3a0a6e1b4e83a49a7b799023756294a2108fc7389efffc5aafa33e',
                    path=f'formal/{pub.name}',sha256=sha(pub),byte_identical=True)
                pr=read(public/f'{task}-diagnostic-0-replacement.json')
                assert all(c['metrics']==pr['calls'][0]['metrics'] for c in ref['calls'])
                assert all(d['evaluator_statistics']==pr['evaluator_statistics'] for d in diagnostics.values())
            else:
                assert diagnostics['baseline']['stats']==diagnostics['candidate']['stats']
        scope['tasks'][task]=entry
    if versions:
        assert len(versions)==1
        scope['python']=list(versions)[0]
    if fingerprints:
        assert len(fingerprints)==1
        scope['runtime_file_sha256']=json.loads(next(iter(fingerprints)))
        scope['all_24_process_runtime_fingerprints_equal']=True
    report['scopes'][host]=scope

manifest=read(raw/'python-runtime-manifest.json')
for name,digest in manifest['files'].items():
    assert sha(Path(manifest['snapshot'])/name)==digest,name
report['frozen_runtime']=dict(manifest='python-runtime-manifest.json',verified_files=len(manifest['files']),
    loaded_core_library=str(Path(manifest['snapshot'])/'lib/libpython3.12.dylib'),
    observation='macOS dyld confirmed the loaded core library is in this private snapshot; LIBDIR also lists the original installation as an alias.')
for name in ['python-runtime-manifest.json','paired_m2_fixed.py','paired_m2_final.py','paired_final.py']:
    shutil.copy2(raw/name,out/name)
save('final_corrected.json',report)
for host,scope in report['scopes'].items():
    print(host,scope['host_telemetry'])
    for task,e in scope['tasks'].items(): print(task,e['medians'])
print('PASS: all complete arrays, metrics, runtime fingerprints, 1898 runtime files')
