"""Run on cloud after three verified uploads; no model and no image deletion."""
from pathlib import Path, PurePosixPath
import hashlib, json, os, shutil, stat, zipfile
from datetime import datetime, timezone

root=Path('/root/rivermind-data/shared/datasets/home-fire/v1')
project=Path('/root/rivermind-data/FIRE_R2')
counts={'train':3900,'val':1300,'test':1300}
records=[]
total_expanded=0
for split, count in counts.items():
    archive=root/'raw'/(split+'.zip')
    receipt=json.loads((project/'incoming'/('home-'+split+'-transfer.json')).read_text())
    assert receipt['status']=='verified_and_local_removed', split
    assert archive.stat().st_size==receipt['bytes'],split
    assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==receipt['sha256'],split
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0] in ('images','labels'),i.filename
            assert not stat.S_ISLNK(i.external_attr>>16),i.filename
        total_expanded+=sum(i.file_size for i in z.infolist())
        assert total_expanded<30_000_000_000
        assert sum(i.filename.startswith('images/') and not i.is_dir() for i in z.infolist())==count
        assert sum(i.filename.startswith('labels/') and i.filename.endswith('.txt') for i in z.infolist())==count
        # Full local CRC + identical remote SHA already verified the archive;
        # extractall also checks each member CRC, so avoid a redundant full pass.
    records.append({'split':split,'images':count,'archive_sha256':receipt['sha256']})
target=root/'extracted'
assert not target.exists(),'Already extracted; inspect/reuse, do not overwrite'
assert shutil.disk_usage(root).free>total_expanded+1024**3
staging=root/'.extracted-r2-staging'
assert not staging.exists(),'Old staging exists; inspect before retry'
staging.mkdir()
try:
    for split in counts:
        with zipfile.ZipFile(root/'raw'/(split+'.zip')) as z:z.extractall(staging/split)
    os.rename(staging,target)
except BaseException:
    shutil.rmtree(staging)
    raise
annotations=root/'annotations'
assert not annotations.exists() and not annotations.is_symlink()
annotations.symlink_to('extracted',target_is_directory=True)
report={'dataset':'homefire','version':'v1','images':sum(counts.values()),'official_splits':counts,'expanded_bytes':total_expanded,'archives':records,'image_root':str(target),'annotation_root':str(annotations),'finished_at':datetime.now(timezone.utc).isoformat(),'originals_modified':False,'deduplicated':False}
(project/'artifacts/homefire-extraction.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
