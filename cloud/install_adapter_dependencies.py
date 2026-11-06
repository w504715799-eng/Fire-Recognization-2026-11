"""Install only the two pinned small cloud wheels into an isolated optional layer."""
from pathlib import Path
import argparse,json,os,subprocess,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,file_lock
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config));spec=read_json(ROOT/'configs/adapter_dependencies.json')
    target=Path(spec['target']);wheels=Path(spec['wheels']);report={'status':'planned','spec':spec,'files':[]}
    with file_lock(ROOT/'artifacts/.locks/adapter-dependencies.lock'):
        if target.exists():raise ProjectError('Inspect existing adapter layer; do not overwrite')
        wheels.mkdir(exist_ok=True);receipt=wheels/'install-receipt.json';write_json(receipt,report)
        try:
            paths=[]
            for row in spec['packages']:
                dest=wheels/row['url'].rsplit('/',1)[-1]
                if dest.exists():raise ProjectError('Preserve prior downloads; inspect before retry')
                partial=dest.with_suffix('.whl.partial')
                with urllib.request.urlopen(row['url'],timeout=30) as response,partial.open('xb') as stream:
                    stream.write(response.read(row['bytes']+1))
                if partial.stat().st_size!=row['bytes'] or sha256(partial)!=row['sha256']:raise ProjectError('Pinned wheel mismatch')
                partial.rename(dest);paths.append(str(dest));report['files'].append({'path':str(dest),'bytes':dest.stat().st_size,'sha256':sha256(dest)});write_json(receipt,report)
            subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--no-compile','--no-cache-dir','--target',str(target),*paths],check=True)
            report.update(status='installed',installed_bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()))
        except BaseException as exc:report.update(status='failed',error=str(exc));raise
        finally:write_json(receipt,report)
    print(json.dumps(report))


if __name__=='__main__':main()
