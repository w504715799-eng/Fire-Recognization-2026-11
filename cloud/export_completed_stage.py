"""Export an explicit text-only experiment snapshot; never includes model/data/credentials."""
from pathlib import Path
import datetime,gzip,hashlib,io,json,tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=[]
    for folder in ('artifacts/evaluation-cpu-20261106',
                   'artifacts/qwen3vl-2b-preparation-20261106',
                   'artifacts/teacher-qwen3vl2b/base/cpu-pilot4-20261106',
                   'artifacts/w-probability-candidate-20261106'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.json','.jsonl'})
    files.extend(ROOT/p for p in (
        'artifacts/teacher/dfire-fit-pilot-v2_serial.json',
        'artifacts/teacher/dfire-fit-pilot-v2_batch4.json',
        'artifacts/teacher/dfire-fit-pilot-v2_expanded.json',
        'artifacts/teacher-comparison-20261106/gpu-handoff-snapshot.json',
        'artifacts/run-records/dfire_B_seed17_full_20261106.json',
        'runs/dfire_B_seed17_full_20261106/results.csv',
        'runs/dfire_M_seed17_full_20261106/results.csv',
        'artifacts/methods-20261106/after-M-ordinary-status.json',
        'weights/Qwen3-VL-2B-Instruct/89644892e4d85e24eaac8bacfd4f463576704203/download-receipt.json'))
    for pattern in ('teacher-v2_*20261106.log','*cpu*20261106.log','*calibration*20261106.log',
                    'qwen3vl2b-*20261106.log','w-probability-candidate*20261106.log','dfire-B-full-20261106.log'):
        files.extend((ROOT/'logs').glob(pattern))
    # Include the new regression only after a terminal report, never disguise partial work.
    current=ROOT/'artifacts/teacher-qwen3vl2b/base/cpu-regression64-20261106'
    report=current/'report.json'
    terminal=report.exists() and json.loads(report.read_text()).get('status') in {'completed','stopped','failed'}
    if terminal:files.extend(p for p in current.iterdir() if p.is_file() and p.suffix in {'.json','.jsonl'})
    files=[p for p in sorted(set(files)) if 'qwen3vl2b-cpu-regression64-' not in p.name or terminal]
    output=ROOT/'artifacts/staging/stage-20261106-results.tar'
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise RuntimeError('Keep prior exports; do not overwrite')
    manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'new_2b_regression_terminal':terminal,'M_csv_is_partial_snapshot':True,'files':[]}
    with tarfile.open(output,'w') as tar:
        for p in files:
            raw=p.read_bytes();relative=str(p.relative_to(ROOT));dest=relative
            if relative.startswith('weights/'):dest='model-metadata/qwen3vl2b-download-receipt.json'
            data=raw
            if len(raw)>100000 or p.suffix=='.log':data=gzip.compress(raw,mtime=0);dest+='.gz'
            entry=tarfile.TarInfo(dest);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
            manifest['files'].append({'source':relative,'export':dest,'original_bytes':len(raw),
                'original_sha256':hashlib.sha256(raw).hexdigest(),'export_bytes':len(data),
                'export_sha256':hashlib.sha256(data).hexdigest()})
        data=json.dumps(manifest,ensure_ascii=False,indent=2).encode();entry=tarfile.TarInfo('export-manifest.json')
        entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'files':len(manifest['files']),
                      'new_2b_regression_terminal':terminal}))


if __name__=='__main__':main()
