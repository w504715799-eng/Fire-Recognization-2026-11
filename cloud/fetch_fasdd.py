"""Public author archives: bounded parallel ranges, whole-stream hashes, streaming extraction.

No image/annotation changes. Complete original files are kept; compressed containers are not
duplicated on disk. API reference: https://stream-unzip.docs.trade.gov.uk/get-started/
"""
from pathlib import Path,PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,os,platform,shutil,socket,time,io,zipfile,tempfile
import requests
from stream_unzip import stream_unzip

SPECS={
 'cv':{'key':'fasdd-cv','file_id':'85bad3972ed60e0a20a790032c0d85fb','bytes':12326081298,
       'md5':'da5e1410e13113fc9d605567aea54508','expanded':12739273073,'images':95314,'top':'FASDD_CV'},
 'uav':{'key':'fasdd-uav','file_id':'9456106d26c5fc6b74143c3707115d39','bytes':14888184890,
       'md5':'c5ea9651ca672fc128c6f17d0797c2f2','expanded':15142026203,'images':25097,'top':'FASDD_UAV'}
}
PROJECT=Path('/root/rivermind-data/FIRE_R2')
SHARED=Path('/root/rivermind-data/shared/datasets')

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)

def fetch_range(url,start,end):
    last=None
    for attempt in range(6):
        try:
            with requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},
                              timeout=(12,45),stream=True) as r:
                if r.status_code==429:
                    delay=r.headers.get('Retry-After','15')
                    time.sleep(min(max(float(delay) if delay.isdigit() else 15.,1.),120.));continue
                r.raise_for_status()
                if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise ValueError('Server did not honor the exact public byte range')
                data=bytearray()
                for block in r.iter_content(1024*1024):
                    data.extend(block)
                    if len(data)>end-start+1:raise ValueError('Range response too large')
                if len(data)!=end-start+1:raise requests.exceptions.ChunkedEncodingError('Short byte range')
                return bytes(data)
        except requests.RequestException as exc:
            if getattr(exc,'response',None) is not None and exc.response.status_code in (401,403):raise
            last=exc;time.sleep(min(2**attempt,15))
    raise RuntimeError(f'Range failed after bounded retries: {start}-{end}: {last}')

class Download:
    def __init__(self,spec,report,workers=8):
        self.spec=spec;self.report=report;self.workers=workers;self.received=0;self.files=0;self.expanded=0
        self.md5=hashlib.md5();self.sha=hashlib.sha256();self.started=time.monotonic();self.last=0.;self.phase='download_extract'
    def progress(self,force=False,**extra):
        now=time.monotonic()
        if force or now-self.last>=5:
            elapsed=now-self.started
            result={'phase':self.phase,'dataset':self.spec['key'],'bytes_consumed_in_order':self.received,
                    'archive_bytes':self.spec['bytes'],'percent':round(100*self.received/self.spec['bytes'],2),
                    'files_extracted':self.files,'expanded_bytes':self.expanded,'elapsed_seconds':round(elapsed,1),
                    'end_to_end_Bps':round(self.received/max(elapsed,.001)),**extra}
            write_json(self.report,result);print(json.dumps(result),flush=True);self.last=now
    def chunks(self):
        url='https://china.scidb.cn/download?fileId='+self.spec['file_id'];step=8*1024*1024
        count=(self.spec['bytes']+step-1)//step
        # At most workers chunks in flight: no full compressed archive or unbounded RAM queue.
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending={}
            def submit(index):
                pending[index]=pool.submit(fetch_range,url,index*step,min((index+1)*step,self.spec['bytes'])-1)
            for index in range(min(self.workers,count)):submit(index)
            for index in range(count):
                block=pending.pop(index).result()
                if index+self.workers<count:submit(index+self.workers)
                self.received+=len(block);self.md5.update(block);self.sha.update(block);self.progress()
                yield block

def extract(chunks,destination,manifest,progress=None):
    names=set();total=0;files=0
    with manifest.open('x') as log:
        for raw_name,declared_size,contents in stream_unzip(chunks):
            name=raw_name.decode('utf-8');relative=PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name:raise ValueError('Unsafe archive path')
            target=destination/name
            if not target.resolve().is_relative_to(destination.resolve()):raise ValueError('Archive path escapes staging')
            if name.endswith('/'):
                target.mkdir(parents=True,exist_ok=True)
                for block in contents:
                    if block:raise ValueError('Directory has unexpected payload')
                continue
            if name in names:raise ValueError('Repeated archive path; no records were discarded')
            names.add(name);target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();size=0
            with target.open('xb') as output:
                for block in contents:
                    size+=len(block);total+=len(block)
                    if total>30_000_000_000:raise ValueError('Expanded dataset exceeds30GB')
                    output.write(block);h.update(block)
            if declared_size is not None and size!=declared_size:raise ValueError('Uncompressed member size mismatch')
            files+=1;log.write(json.dumps({'path':name,'bytes':size,'sha256':h.hexdigest()})+'\n')
            if progress:
                progress.files=files;progress.expanded=total;progress.progress()
                if files%1000==0 and shutil.disk_usage(destination).free<1024**3:raise RuntimeError('Less than1GiB free; preserve staging and stop')
    # A ZIP reader may stop at central directory; consume its remaining transport bytes for the full MD5.
    for _ in chunks:pass
    return files,total

def self_test():
    with tempfile.TemporaryDirectory(dir=PROJECT/'tmp') as temp:
        root=Path(temp);buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('dataset/',b'');z.writestr('dataset/empty.txt',b'');z.writestr('dataset/label.txt',b'0 0.5 0.5 0.1 0.1\n')
        original=buffer.getvalue();read=bytearray()
        def chunks():
            for pos in range(0,len(original),13):
                block=original[pos:pos+13];read.extend(block);yield block
        it=chunks();files,size=extract(it,root,root/'manifest.jsonl')
        assert bytes(read)==original and files==2 and (root/'dataset/label.txt').read_bytes()==b'0 0.5 0.5 0.1 0.1\n'
        bad=io.BytesIO()
        with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape',b'x')
        try:extract(iter([bad.getvalue()]),root,root/'bad.jsonl')
        except ValueError:pass
        else:raise AssertionError('Traversal accepted')
    print('Streaming CRC/full-byte-consumption/path tests passed',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=SPECS);p.add_argument('--workers',type=int,default=8);p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if platform.system()!='Linux' or not PROJECT.exists():raise RuntimeError('Authorized cloud workspace only')
    if args.self_test:return self_test()
    spec=SPECS[args.part];base=SHARED/spec['key']/'v9';base.mkdir(parents=True,exist_ok=True)
    if (base/'EXTRACTED.json').exists():
        receipt=json.loads((base/'EXTRACTED.json').read_text())
        if receipt.get('archive_md5')==spec['md5']:print('Already extracted and archive-verified; reuse');return
        raise RuntimeError('Existing extraction has another identity')
    lock=SHARED/'.locks'/f"{spec['key']}__v9.lock";lock.mkdir()
    write_json(lock/'owner.json',{'project':'FIRE_R2','host':socket.gethostname(),'pid':os.getpid(),'source_file_id':spec['file_id']})
    report=PROJECT/'artifacts/fasdd-full-20261105'/f"{args.part}-progress.json"
    staging=base/('.extracting-'+str(int(time.time())));staging.mkdir()
    manifest=base/(staging.name+'-files.jsonl');download=Download(spec,report,args.workers)
    try:
        chunks=download.chunks();files,expanded=extract(chunks,staging,manifest,download)
        if download.received!=spec['bytes'] or download.md5.hexdigest()!=spec['md5']:
            raise RuntimeError('Whole original archive length/MD5 verification failed')
        if expanded!=spec['expanded']:raise RuntimeError('Expanded size differs from original central-directory evidence')
        if (base/'extracted').exists():raise RuntimeError('Do not replace an existing extraction')
        os.rename(staging,base/'extracted');os.rename(manifest,base/'original-files.jsonl')
        annotation=base/'extracted'/spec['top']/'annotations'/('COCO_'+args.part.upper())/'Annotations'
        images=base/'extracted'/spec['top']/'images'
        if not annotation.is_dir() or not images.is_dir():raise RuntimeError('Inspect actual published layout before creating aliases')
        for split in ('train','val','test'):
            obj=json.loads((annotation/(split+'.json')).read_text())
            if {int(c['id']):c['name'].lower() for c in obj['categories']}!={0:'fire',1:'smoke'}:raise RuntimeError('Published class mapping differs from the protocol')
        (base/'annotations').symlink_to(os.path.relpath(annotation,base),target_is_directory=True)
        (base/'extracted/images').symlink_to(os.path.relpath(images,base/'extracted'),target_is_directory=True)
        receipt={'archive_md5':download.md5.hexdigest(),'archive_sha256':download.sha.hexdigest(),
                 'archive_bytes':download.received,'expanded_bytes':expanded,'files':files,
                 'source':'https://china.scidb.cn/download?fileId='+spec['file_id'],
                 'original_files_changed':False,'archive_container_retained':False,'per_member_crc_verified':True,
                 'status':'archive_verified_extracted_pending_pixel_audit'}
        write_json(base/'EXTRACTED.json',receipt);download.phase='extracted';download.progress(True,receipt=receipt)
    except BaseException as exc:
        download.phase='failed_staging_preserved';download.progress(True,error=str(exc),staging=str(staging));raise
    finally:
        (lock/'owner.json').unlink();lock.rmdir()

if __name__=='__main__':main()
