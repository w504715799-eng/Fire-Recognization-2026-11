"""Download only on cloud; official ModelScope mirror, HF runtime content identity checked."""
from pathlib import Path
import hashlib,json,os,platform,time
import requests

ROOT=Path('/root/rivermind-data/FIRE_R2')

def verify(path,meta,hf=None):
    if path.stat().st_size!=meta['Size']:raise RuntimeError('Downloaded size differs: '+meta['Path'])
    sha=hashlib.sha256();git=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):sha.update(block);git.update(block)
    if sha.hexdigest()!=meta['Sha256']:raise RuntimeError('Official mirror SHA256 mismatch: '+meta['Path'])
    if hf:
        if hf.get('lfs_sha256') and sha.hexdigest()!=hf['lfs_sha256']:raise RuntimeError('HF LFS content mismatch')
        if not hf.get('lfs_sha256') and hf.get('git_blob_oid') and git.hexdigest()!=hf['git_blob_oid']:
            raise RuntimeError('HF processor/config content differs: '+meta['Path'])
    return sha.hexdigest()

def main():
    if platform.system()!='Linux' or not ROOT.is_dir():raise RuntimeError('Cloud-only model download')
    plan=json.loads((ROOT/'artifacts/cloud-session-2026-11-05/qwen-download-plan.json').read_text())
    manifest=json.loads((ROOT/'artifacts/cloud-session-2026-11-05/qwen-modelscope-manifest.json').read_text())
    mirror={f['Path']:f for f in manifest['Data']['Files'] if f['Type']=='blob'}
    target=Path(plan['destination']);target.mkdir(parents=True,exist_ok=True)
    report={'model_id':plan['model_id'],'hf_revision':plan['revision'],'source':'official Qwen ModelScope repository','runtime_content_verified_against_hf':False,'files':[],'model_loaded':False}
    for hf in plan['files']:
        name=hf['path']
        if name=='.gitattributes':continue
        meta=mirror[name];doc=name in ('LICENSE','README.md');dest=target/name;partial=dest.with_suffix(dest.suffix+'.partial')
        if not dest.exists():
            for attempt in range(5):
                offset=partial.stat().st_size if partial.exists() else 0
                if offset==meta['Size']:break
                headers={'Range':f'bytes={offset}-'} if offset else {}
                try:
                    with requests.get('https://modelscope.cn/api/v1/models/Qwen/Qwen3.5-0.8B/repo',
                         params={'Revision':meta['Revision'],'FilePath':name},headers=headers,stream=True,timeout=(15,60)) as response:
                        response.raise_for_status()
                        if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):raise RuntimeError('Resume range not honored')
                        last=time.monotonic()
                        with partial.open('ab' if offset else 'wb') as out:
                            for block in response.iter_content(1024*1024):
                                out.write(block);offset+=len(block)
                                if offset>meta['Size']:raise RuntimeError('Oversized model response')
                                if time.monotonic()-last>10:
                                    print(json.dumps({'file':name,'bytes':offset,'total':meta['Size']}),flush=True);last=time.monotonic()
                    if offset==meta['Size']:break
                except requests.RequestException as exc:
                    if getattr(exc,'response',None) is not None and exc.response.status_code in (401,403):raise
                    if attempt==4:raise
                    time.sleep(5*(attempt+1))
            digest=verify(partial,meta,None if doc else hf);os.replace(partial,dest)
        else:digest=verify(dest,meta,None if doc else hf)
        report['files'].append({'path':name,'bytes':dest.stat().st_size,'sha256':digest,
                                'mirror_file_revision':meta['Revision'],'hf_runtime_match':not doc,'documentation':doc})
        (target/'download-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['files'][-1]),flush=True)
    report['runtime_content_verified_against_hf']=True;report['total_bytes']=sum(f['bytes'] for f in report['files'])
    report['documentation_note']='Official mirror documentation/license retained; runtime files verified against fixed HF revision.'
    (target/'download-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    (target/'revision.json').write_text(json.dumps({'revision':plan['revision'],'model_id':plan['model_id'],'verified_runtime_content':True},indent=2)+'\n')
    print(json.dumps({'status':'download_verified','bytes':report['total_bytes'],'model_loaded':False}),flush=True)

if __name__=='__main__':main()
