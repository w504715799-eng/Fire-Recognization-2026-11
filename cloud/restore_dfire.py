"""Restore this verified archive on cloud; preserve every original file and label."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile,os,platform

EXPECTED='65d81b602991261b47ddc71281e99108362a30336a448e1f8d21352c825c6d83'
ROOT=Path('/root/rivermind-data/shared/datasets/dfire/kaggle-21527-sha65d81b60')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    if platform.system()!='Linux' or not ROOT.is_dir():raise RuntimeError('Cloud dataset location required')
    archive=ROOT/'raw/dfire.zip';partial=ROOT/'raw/dfire.zip.part'
    source=archive if archive.exists() else partial
    if source.stat().st_size!=3049605157 or digest(source)!=EXPECTED:raise RuntimeError('Full archive verification failed; retain partial')
    if source==partial:os.replace(partial,archive)
    target=ROOT/'extracted';target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        if sum(m.file_size for m in members)>30_000_000_000:raise RuntimeError('Dataset exceeds30GB')
        for m in members:
            p=PurePosixPath(m.filename)
            if p.is_absolute() or '..' in p.parts or (m.external_attr>>16)&0o170000==0o120000:raise RuntimeError('Unsafe archive member')
            dest=target/m.filename
            if not dest.resolve().is_relative_to(target.resolve()):raise RuntimeError('Path escape')
            if m.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
            data=z.read(m)
            if dest.exists():
                if dest.read_bytes()!=data:raise RuntimeError(f'Existing original differs: {m.filename}')
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    record={'archive_sha256':EXPECTED,'archive_bytes':archive.stat().st_size,'unpacked_bytes':sum(m.file_size for m in members),'status':'extracted_awaiting_full_pixel_audit','original_files_changed':False}
    (ROOT/'RESTORE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
