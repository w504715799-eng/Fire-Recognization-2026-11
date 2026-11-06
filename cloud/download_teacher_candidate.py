"""Authorized cloud-only fixed-version download to a user-approved root-filesystem directory."""
from pathlib import Path
import argparse,hashlib,json,os,platform,shutil,sys,time
import requests
ROOT=Path('/root/rivermind-data/FIRE_R2');sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,file_lock


def hashes(path):
    h=hashlib.sha256();g=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block);g.update(block)
    return h.hexdigest(),g.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True)
    p.add_argument('--max-mib-per-second',type=float,default=12.);a=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=ROOT/'.envs/sci-env':raise ProjectError('Use authorized cloud sci-env only')
    plan=read_json(ROOT/a.plan);approval=read_json(ROOT/a.approval)
    if approval.get('approved') is not True or any(approval.get(k)!=plan[k] for k in ('model_id','revision','runtime_bytes','destination')):
        raise ProjectError('Explicit matching large-download approval required')
    if plan['model_id']!='Qwen/Qwen3-VL-2B-Instruct':raise ProjectError('This approved transfer is for the exact user-selected model')
    dest=Path(plan['destination']);alias=Path(plan['project_alias'])
    if dest!=Path('/FIRE_R2-models/Qwen3-VL-2B-Instruct')/plan['revision']:raise ProjectError('Unapproved physical destination')
    if not alias.is_relative_to(ROOT/'weights'):raise ProjectError('Project alias must remain under weights')
    manifest=read_json(ROOT/'artifacts/qwen3vl-2b-preparation-20261106/modelscope-files.json')
    mirror={f['Path']:f for f in manifest['Data']['Files'] if f['Type']=='blob'}
    dest.mkdir(parents=True,exist_ok=True)
    if dest.stat().st_dev!=Path('/').stat().st_dev:raise ProjectError('Requested destination is not on the root filesystem')
    missing=sum(f['bytes'] for f in plan['files'] if not (dest/f['path']).exists())
    if shutil.disk_usage(dest).free<missing+2*1024**3:raise ProjectError('Insufficient root space; do not delete unrelated files')
    report={'model_id':plan['model_id'],'hf_revision':plan['revision'],'physical_directory':str(dest),
            'source':'official Qwen ModelScope mirror, fixed HF content identity','files':[],
            'runtime_content_verified_against_hf':False,'model_loaded':False,'status':'downloading'}
    with file_lock(ROOT/'artifacts/.locks/qwen3vl2b-download.lock'):
        write_json(dest/'download-receipt.json',report)
        for item in plan['files']:
            name=item['path'];meta=mirror[name];target=dest/name;partial=target.with_suffix(target.suffix+'.partial')
            if meta['Size']!=item['bytes']:raise ProjectError('Mirror/HF size mismatch')
            if item.get('lfs_sha256') and item['lfs_sha256']!=meta['Sha256']:raise ProjectError('Mirror/HF weight identity mismatch')
            if not target.exists():
                for attempt in range(5):
                    offset=partial.stat().st_size if partial.exists() else 0
                    if offset==item['bytes']:break
                    if offset>item['bytes']:raise ProjectError('Oversized partial; retain for inspection')
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    try:
                        with requests.get('https://modelscope.cn/api/v1/models/'+plan['model_id']+'/repo',
                            params={'Revision':meta['Revision'],'FilePath':name},headers=headers,stream=True,timeout=(15,60)) as response:
                            response.raise_for_status()
                            if offset and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                                raise ProjectError('Server did not honor exact resume range')
                            begin=time.monotonic();initial=offset;last=begin
                            with partial.open('ab' if offset else 'wb') as stream:
                                for block in response.iter_content(1024*1024):
                                    stream.write(block);offset+=len(block)
                                    if offset>item['bytes']:raise ProjectError('Response exceeds published size')
                                    elapsed=time.monotonic()-begin;expected=(offset-initial)/(a.max_mib_per_second*1024**2)
                                    if expected>elapsed:time.sleep(min(expected-elapsed,1))
                                    if time.monotonic()-last>=10:
                                        report['progress']={'file':name,'bytes':offset,'total':item['bytes'],'MiB_s':(offset-initial)/max(time.monotonic()-begin,.001)/1024**2}
                                        write_json(dest/'download-receipt.json',report);print(json.dumps(report['progress']),flush=True);last=time.monotonic()
                                stream.flush();os.fsync(stream.fileno())
                        if offset==item['bytes']:break
                    except requests.RequestException as exc:
                        code=getattr(getattr(exc,'response',None),'status_code',None)
                        if code in (401,403) or attempt==4:raise
                        if code==429:
                            delay=int(exc.response.headers.get('Retry-After','60'));time.sleep(max(delay,1))
                        else:time.sleep(5*(attempt+1))
                actual,git=hashes(partial)
                if partial.stat().st_size!=item['bytes'] or actual!=meta['Sha256']:raise ProjectError('Size/SHA mismatch; retain partial')
                if item.get('lfs_sha256'):
                    if actual!=item['lfs_sha256']:raise ProjectError('HF weight SHA mismatch')
                elif git!=item['git_blob_oid']:raise ProjectError('HF tokenizer/config content identity mismatch')
                os.replace(partial,target)
            else:
                actual,git=hashes(target)
                if target.stat().st_size!=item['bytes'] or actual!=meta['Sha256']:raise ProjectError('Existing asset differs')
                if not item.get('lfs_sha256') and git!=item['git_blob_oid']:raise ProjectError('Existing runtime file differs from HF')
            report['files'].append({'path':name,'bytes':target.stat().st_size,'sha256':actual,'mirror_file_revision':meta['Revision']})
            write_json(dest/'download-receipt.json',report)
        report.update(status='verified',runtime_content_verified_against_hf=True,total_bytes=sum(f['bytes'] for f in report['files']))
        report['progress']={'phase':'complete','bytes':report['total_bytes'],'total':report['total_bytes'],'percent':100.}
        write_json(dest/'download-receipt.json',report)
        write_json(dest/'revision.json',{'model_id':plan['model_id'],'revision':plan['revision'],'verified_runtime_content':True})
        alias.parent.mkdir(parents=True,exist_ok=True)
        if alias.is_symlink():
            if alias.resolve()!=dest:raise ProjectError('Conflicting model alias')
        elif alias.exists():raise ProjectError('Alias location already owns content; no overwrite')
        else:alias.symlink_to(dest,target_is_directory=True)
        report['project_alias']=str(alias);write_json(dest/'download-receipt.json',report)
        print(json.dumps({'status':'verified','bytes':report['total_bytes'],'physical':str(dest),'alias':str(alias),'model_loaded':False}),flush=True)


if __name__=='__main__':main()
