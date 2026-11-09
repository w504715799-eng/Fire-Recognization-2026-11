"""Read existing fit files once into OS page cache; never decode, copy or change source bytes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,json,resource,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,output_path,ProjectError,sha256
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config));out=ROOT/'artifacts/recovery-20261109/fit-page-cache.json'
    if out.exists():raise ProjectError('Do not repeat the cache warm-up')
    source=ROOT/'artifacts/data/dfire-kaggle-21527-sha65d81b60/fit.txt';paths=[Path(x) for x in source.read_text().splitlines()]
    total=sum(p.stat().st_size for p in paths)
    if total>3_000_000_000:raise ProjectError('This bounded warm-up was sized for the verified 2.07GB fit set')
    started=time.monotonic()
    def read(path):
        count=0
        with path.open('rb') as f:
            while time.monotonic()-started<300:
                chunk=f.read(1024*1024)
                if not chunk:return count,True
                count+=len(chunk)
        return count,False
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(read,paths))
    report={'status':'completed' if all(done for _,done in results) else 'budget_stopped','threads':4,
            'files':len(paths),'expected_bytes':total,'read_bytes':sum(n for n,_ in results),'seconds':time.monotonic()-started,
            'fit_list_sha256':sha256(source),'disk_copies_created':0,'source_files_modified':False,
            'cache_location':'Linux-managed OS page cache for existing fit image paths; no new disk cache',
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(out,report);print(json.dumps(report))


if __name__=='__main__':main()
