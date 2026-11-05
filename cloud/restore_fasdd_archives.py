"""Verify retained official archives and reuse interrupted extraction without changing GT."""
from pathlib import Path,PurePosixPath
import argparse,binascii,hashlib,json,os,platform,socket,tempfile,zipfile,time

PROJECT=Path('/root/rivermind-data/FIRE_R2')
DATA=Path('/root/rivermind-data/shared/datasets')
SPECS={
 'cv':{'key':'fasdd-cv','name':'FASDD_CV.zip','md5':'da5e1410e13113fc9d605567aea54508',
       'bytes':12326081298,'expanded':12739273073,'top':'FASDD_CV','images':95314},
 'uav':{'key':'fasdd-uav','name':'FASDD_UAV.zip','md5':'c5ea9651ca672fc128c6f17d0797c2f2',
        'bytes':14888184890,'expanded':15142026203,'top':'','images':25097}}

def file_hashes(path):
    md5=hashlib.md5();sha=hashlib.sha256();crc=0
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):md5.update(block);sha.update(block);crc=binascii.crc32(block,crc)
    return md5.hexdigest(),sha.hexdigest(),crc&0xffffffff

def save(path,obj):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(temp,path)

def restore(archive,stage,manifest,progress):
    restored=reused=0;total=0;seen=set();last=time.monotonic()
    with zipfile.ZipFile(archive) as z,manifest.open('w') as report:
        entries=z.infolist()
        if sum(i.file_size for i in entries)>30_000_000_000:raise RuntimeError('Unpacked archive exceeds30GB')
        for item in entries:
            relative=PurePosixPath(item.filename)
            if relative.is_absolute() or '..' in relative.parts or (item.external_attr>>16)&0o170000==0o120000:
                raise RuntimeError('Unsafe archive member')
            target=stage/item.filename
            if not target.resolve().is_relative_to(stage.resolve()):raise RuntimeError('Archive path escape')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            if item.filename in seen:raise RuntimeError('Repeated archive path; no member discarded')
            seen.add(item.filename);total+=item.file_size
            same=False;sha=None
            if target.is_file() and target.stat().st_size==item.file_size:
                _,sha,crc=file_hashes(target);same=crc==item.CRC
            if same:reused+=1
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256()
                with z.open(item) as source,tempfile.NamedTemporaryFile(dir=target.parent,prefix='.recover-',delete=False) as out:
                    temporary=Path(out.name)
                    for block in iter(lambda:source.read(4*1024*1024),b''):out.write(block);h.update(block)
                if temporary.stat().st_size!=item.file_size:raise RuntimeError('Extracted file size mismatch')
                os.replace(temporary,target);sha=h.hexdigest();restored+=1
            report.write(json.dumps({'path':item.filename,'bytes':item.file_size,'crc32':f'{item.CRC:08x}','sha256':sha,'reused':same})+'\n')
            if time.monotonic()-last>=10:
                progress({'phase':'restoring','files':len(seen),'reused_files':reused,'restored_files':restored,'expanded_bytes':total});last=time.monotonic()
    return {'files':len(seen),'reused_files':reused,'restored_files':restored,'expanded_bytes':total}

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=SPECS,required=True);args=p.parse_args()
    if platform.system()!='Linux' or not PROJECT.exists():raise RuntimeError('Authorized cloud project only')
    spec=SPECS[args.part];base=DATA/spec['key']/'v9';archive=base/'raw'/spec['name']
    lock=DATA/'.locks'/f"{spec['key']}__v9.lock";lock.mkdir()
    save(lock/'owner.json',{'project':'FIRE_R2','pid':os.getpid(),'hostname':socket.gethostname(),'mode':'restore_retained_zip'})
    progress_path=PROJECT/f'artifacts/fasdd-full-20261105/{args.part}-restore-progress.json'
    def progress(value):save(progress_path,value);print(json.dumps(value),flush=True)
    try:
        progress({'phase':'checking_archive','archive':str(archive)})
        if archive.stat().st_size!=spec['bytes']:raise RuntimeError('Incomplete original archive')
        md5,sha,_=file_hashes(archive)
        if md5!=spec['md5']:raise RuntimeError('Publisher MD5 mismatch; preserve archive and inspect')
        final=base/'extracted'
        stages=[x for x in base.glob('.extracting-*') if x.is_dir()]
        if final.exists():stage=final
        elif len(stages)==1:stage=stages[0]
        elif not stages:stage=base/'.restoring-original';stage.mkdir(exist_ok=True)
        else:raise RuntimeError('Multiple interrupted stages; select the verified original stage explicitly')
        summary=restore(archive,stage,base/'original-files-complete.jsonl',progress)
        if summary['expanded_bytes']!=spec['expanded']:raise RuntimeError('Expanded size differs from original metadata')
        if stage!=final:os.rename(stage,final)
        rawroot=final/spec['top'] if spec['top'] else final
        annotations=rawroot/'annotations'/f'COCO_{args.part.upper()}'/'Annotations'
        images=rawroot/'images'
        if not annotations.is_dir() or not images.is_dir():raise RuntimeError('Published layout needs a read-path adaptation')
        for split in ('train','val','test'):
            obj=json.loads((annotations/(split+'.json')).read_text())
            if {int(c['id']):c['name'].lower() for c in obj['categories']}!={0:'fire',1:'smoke'}:
                raise RuntimeError('Published category mapping differs; do not rewrite GT')
        alias=base/'annotations'
        if not alias.exists():alias.symlink_to(os.path.relpath(annotations,base),target_is_directory=True)
        else:
            for split in ('train','val','test'):
                if file_hashes(alias/(split+'.json'))[1]!=file_hashes(annotations/(split+'.json'))[1]:raise RuntimeError('Existing annotation alias differs')
        alias=final/'images'
        if not alias.exists():alias.symlink_to(os.path.relpath(images,final),target_is_directory=True)
        elif alias.resolve()!=images.resolve():raise RuntimeError('Existing image alias differs')
        receipt={**summary,'archive_md5':md5,'archive_sha256':sha,'archive_bytes':spec['bytes'],
                 'archive_retained':True,'original_labels_modified':False,'status':'extracted_pending_pixel_audit'}
        save(base/'EXTRACTED.json',receipt);progress({'phase':'extracted',**receipt})
    finally:
        (lock/'owner.json').unlink();lock.rmdir()

if __name__=='__main__':main()
