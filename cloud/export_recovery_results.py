"""Text-only recovery evidence export; excludes weights, optimizer tensors and private configs."""
from pathlib import Path
import datetime,gzip,hashlib,io,json,tarfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    files=[]
    for folder in ['artifacts/recovery-20261109','artifacts/teacher-evaluation-20261109','artifacts/evaluation-20261109']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv') and not p.name.endswith('.local.json'))
    files.append(ROOT/'artifacts/qwen3vl2b-lora-recovery-20261109/report.json')
    files.append(ROOT/'artifacts/recovery-20261109/executed_resume_research_detector.py')
    for name in ['dfire_M_seed17_full_20261106','dfire_M_ordinary_seed17_full_20261106']:
        files.extend([ROOT/'runs'/name/'results.csv',ROOT/'artifacts/run-records'/(name+'.json')])
        files.extend((ROOT/'runs'/name/'original-dev').glob('*.json'))
    for pattern in ['*20261109.log','dfire-M-full-20261106.log','dfire-M-ordinary-full-20261106.log','qwen3vl2b-lora-cpu-full-20261106.log','after-M-ordinary-20261106.log','after-ordinary-lora-eval-20261106.log']:
        files.extend((ROOT/'logs').glob(pattern))
    out=ROOT/'artifacts/staging/recovery-results-20261109.tar'
    manifest={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinary_fusion_may_still_be_running':True,'files':[]}
    with tarfile.open(out,'x') as tar:
        for p in sorted(set(files)):
            raw=p.read_bytes();dest=str(p.relative_to(ROOT));data=raw
            if p.suffix=='.log' or len(raw)>50000:dest+='.gz';data=gzip.compress(raw,mtime=0)
            info=tarfile.TarInfo(dest);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            manifest['files'].append({'source':str(p.relative_to(ROOT)),'export':dest,'original_bytes':len(raw),
                'original_sha256':hashlib.sha256(raw).hexdigest(),'export_bytes':len(data),'export_sha256':hashlib.sha256(data).hexdigest()})
        data=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('export-manifest.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    print(json.dumps({'files':len(manifest['files']),'archive_bytes':out.stat().st_size}))


if __name__=='__main__':main()
