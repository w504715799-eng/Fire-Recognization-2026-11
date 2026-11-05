"""Few long ranged requests, durable sparse archive checkpoints, retained official ZIPs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
import hashlib,json,os,platform,socket,tempfile,threading,time
import requests

ROOT=Path('/root/rivermind-data/FIRE_R2')
DATA=Path('/root/rivermind-data/shared/datasets')
SPECS={
 'cv':('fasdd-cv','FASDD_CV.zip','85bad3972ed60e0a20a790032c0d85fb',12326081298,'da5e1410e13113fc9d605567aea54508'),
 'uav':('fasdd-uav','FASDD_UAV.zip','9456106d26c5fc6b74143c3707115d39',14888184890,'c5ea9651ca672fc128c6f17d0797c2f2')}

def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        json.dump(obj,f,indent=2);name=f.name
    os.replace(name,path)

def hashes(path):
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):md5.update(block);sha.update(block)
    return md5.hexdigest(),sha.hexdigest()

class RateGate:
    def __init__(self):self.lock=threading.Lock();self.until=0.;self.hits=0
    def wait(self,cancelled=lambda:False):
        while time.time()<self.until:
            if cancelled():return
            time.sleep(min(10,self.until-time.time()))
    def limited(self,header):
        with self.lock:
            if self.until<=time.time():self.hits+=1
            delay=min(300*2**(self.hits-1),1200)
            if header:
                try:delay=max(0,float(header))
                except ValueError:
                    try:delay=max(0,parsedate_to_datetime(header).timestamp()-time.time())
                    except (ValueError,TypeError):pass
            self.until=max(self.until,time.time()+delay)
            atomic_json(ROOT/'artifacts/fasdd-full-20261105/rate-gate.json',{
                'status':429,'hits':self.hits,'retry_after_header':header,
                'do_not_request_before_utc':datetime.fromtimestamp(self.until,timezone.utc).isoformat()})
            if delay>3600:raise RuntimeError('Long server-requested cooldown recorded; pause rather than keep a paid server waiting')

GATE=RateGate()

class Archive:
    def __init__(self,part):
        self.part=part;key,name,fid,total,md5=SPECS[part]
        self.base=DATA/key/'v9';self.raw=self.base/'raw';self.raw.mkdir(parents=True,exist_ok=True)
        self.final=self.raw/name;self.file=self.raw/(name+'.sparse.part');self.state_path=self.raw/(name+'.download.json')
        self.total=total;self.expected=md5;self.url='https://china.scidb.cn/download?fileId='+fid
        self.lock_dir=DATA/'.locks'/f'{key}__v9.lock';self.mutex=threading.Lock();self.started=time.monotonic();self.last=0
        self.failed=False;self.fd=None
    def checkpoint(self,phase='downloading',force=False,error=None):
        with self.mutex:
            if not force and time.monotonic()-self.last<5:return
            if self.fd is not None:os.fsync(self.fd)
            self.state['phase']=phase;self.state['error']=error
            self.state['bytes_received']=sum(s['written'] for s in self.state['segments'])
            self.state['percent']=100*self.state['bytes_received']/self.total
            self.state['elapsed_seconds']=time.monotonic()-self.started
            self.state['rate_limit_wait_until']=GATE.until
            atomic_json(self.state_path,self.state)
            atomic_json(ROOT/f'artifacts/fasdd-full-20261105/{self.part}-archive-progress.json',self.state)
            self.last=time.monotonic()
    def segment(self,index):
        segment=self.state['segments'][index];attempts=0
        with requests.Session() as session:
            while segment['written']<segment['length']:
                if self.failed:raise RuntimeError('Other segment stopped; partial retained')
                GATE.wait(lambda:self.failed)
                if self.failed:raise RuntimeError('Other segment stopped; partial retained')
                start=segment['start']+segment['written'];end=segment['start']+segment['length']-1
                try:
                    with session.get(self.url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(15,60)) as r:
                        if r.status_code==429:
                            GATE.limited(r.headers.get('Retry-After'));self.checkpoint('waiting_rate_limit',True)
                            attempts+=1
                            if attempts>=5:raise RuntimeError('Repeated official rate limiting; checkpoint retained for later recovery')
                            continue
                        r.raise_for_status()
                        if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                            raise RuntimeError('Range not honored; do not mix response bytes')
                        for block in r.iter_content(1024*1024):
                            if self.failed:raise RuntimeError('Other segment stopped')
                            if segment['written']+len(block)>segment['length']:raise RuntimeError('Oversized segment response')
                            offset=segment['start']+segment['written'];view=memoryview(block)
                            while view:
                                n=os.pwrite(self.fd,view,offset)
                                if n<=0:raise OSError('Short positional write')
                                view=view[n:];offset+=n
                            with self.mutex:segment['written']+=len(block)
                            self.checkpoint()
                        self.checkpoint(force=True)
                        if segment['written']<segment['length']:raise requests.exceptions.ChunkedEncodingError('Short segment')
                except requests.RequestException as exc:
                    if getattr(exc,'response',None) is not None and exc.response.status_code in (401,403):raise
                    attempts+=1;self.checkpoint('network_retry',True,str(exc))
                    if attempts>=8:raise
                    time.sleep(min(5*attempts,60))
    def run(self):
        if self.final.exists():
            if hashes(self.final)[0]!=self.expected:raise RuntimeError('Existing archive differs; do not overwrite')
            return {'part':self.part,'status':'reused_verified'}
        self.lock_dir.mkdir();atomic_json(self.lock_dir/'owner.json',{'project':'FIRE_R2','pid':os.getpid(),'hostname':socket.gethostname(),'mode':'retained_whole_zip'})
        try:
            if self.state_path.exists():
                self.state=json.loads(self.state_path.read_text())
                if self.state['archive_bytes']!=self.total or self.state['publisher_md5']!=self.expected:raise RuntimeError('Resume identity mismatch')
            else:
                if self.file.exists():raise RuntimeError('Sparse partial without state; preserve and inspect')
                old=self.raw/(self.final.name+'.part');prefix=old.stat().st_size if old.exists() else 0
                if prefix>self.total:raise RuntimeError('Old prefix is too large')
                middle=self.total//2
                self.state={'source':self.url,'archive_bytes':self.total,'publisher_md5':self.expected,
                            'format':'sparse_partial_use_checkpoint_not_file_length','segments':[
                    {'start':a,'length':b-a,'written':max(0,min(prefix-a,b-a))} for a,b in [(0,middle),(middle,self.total)]]}
                atomic_json(self.state_path,self.state)
            old=self.raw/(self.final.name+'.part')
            if not self.file.exists() and old.exists():os.rename(old,self.file)
            if not self.file.exists() and any(s['written'] for s in self.state['segments']):raise RuntimeError('Partial content missing; do not trust checkpoint')
            self.fd=os.open(self.file,os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(self.fd,self.total)
            self.checkpoint(force=True)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(self.segment,i) for i in range(2)]
                for future in as_completed(futures):
                    try:future.result()
                    except BaseException:self.failed=True;raise
            self.checkpoint('verifying_whole_archive',True)
            md5,sha=hashes(self.file)
            if md5!=self.expected:raise RuntimeError('Whole official MD5 mismatch; preserve partial and investigate')
            os.close(self.fd);self.fd=None;os.rename(self.file,self.final)
            self.state.update(archive_md5=md5,archive_sha256=sha,archive_path=str(self.final),archive_retained=True)
            self.checkpoint('archive_verified',True)
            return {'part':self.part,'status':'archive_verified','sha256':sha}
        except BaseException as exc:
            if hasattr(self,'state'):self.checkpoint('interrupted_partial_retained',True,str(exc))
            raise
        finally:
            if self.fd is not None:os.close(self.fd)
            (self.lock_dir/'owner.json').unlink();self.lock_dir.rmdir()

def main():
    if platform.system()!='Linux' or not ROOT.exists():raise RuntimeError('Cloud-only operation')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(Archive(part).run):part for part in ('cv','uav')}
        for future in as_completed(futures):
            try:print(json.dumps(future.result()),flush=True)
            except Exception as exc:print(json.dumps({'part':futures[future],'error':str(exc),'complete':False}),flush=True)

if __name__=='__main__':main()
