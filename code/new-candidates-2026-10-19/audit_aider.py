"""Download and audit untouched author AIDER archive. No prediction or relabeling."""
import collections,hashlib,io,json,sys,time,zipfile
from pathlib import Path
import requests,socket
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'references/new-candidates-2026-10-19';DATA=ROOT/'data/new-candidates-2026-10-19/aider'
assert Path(sys.prefix).name=='sci-env'
requests.packages.urllib3.util.connection.allowed_gai_family=lambda:socket.AF_INET
pre=json.loads((OUT/'aider-preflight.json').read_text());meta=json.loads((OUT/'aider-metadata.txt').read_text());f=next(x for x in meta['files'] if x['key']=='AIDER.zip')
assert pre['compressed_bytes']==f['size'] and max(pre['compressed_bytes'],pre['uncompressed_bytes'])<=30_000_000_000
DATA.mkdir(parents=True,exist_ok=True);dest=DATA/'original.zip'
if not dest.exists():
    part=DATA/'original.zip.partial';assert not part.exists(),'Existing partial: inspect rather than overwrite'
    total=0;last=time.monotonic()
    with requests.get(f['links']['self'],stream=True,timeout=(20,45)) as r,part.open('wb') as output:
        r.raise_for_status()
        for chunk in r.iter_content(1024*1024):
            total+=len(chunk);assert total<=f['size'];output.write(chunk)
            if time.monotonic()-last>20:print(f'downloaded {total}/{f["size"]} bytes',flush=True);last=time.monotonic()
    assert total==f['size'];part.rename(dest)
md5=hashlib.md5();sha=hashlib.sha256()
with dest.open('rb') as source:
    for chunk in iter(lambda:source.read(4*1024*1024),b''):md5.update(chunk);sha.update(chunk)
expected=f['checksum'].split(':')[-1];assert md5.hexdigest()==expected
counts=collections.Counter();failures=[];allnames=[]
with zipfile.ZipFile(dest) as z:
    bad_crc=z.testzip();assert bad_crc is None
    for i in z.infolist():
        if not i.filename.lower().endswith('.jpg'):continue
        cls=Path(i.filename).parent.name;counts[cls]+=1;allnames.append(i.filename)
        try:
            with Image.open(io.BytesIO(z.read(i))) as im:im.load()
        except Exception as e:failures.append({'path':i.filename,'error':str(e)})
    selected=sorted([n for n in allnames if '/normal/' in n],key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:24]
    selected+=sum([sorted([n for n in allnames if '/'+c+'/' in n],key=lambda n:hashlib.sha256(n.encode()).hexdigest())[:2] for c in ['fire','collapsed_building','flooded_areas','traffic_incident']],[])
    (OUT/'aider-fixed-sample.json').write_text(json.dumps({'rule':'24 normal, 2 each other original class, smallest SHA256(full member name), fixed before viewing; only audit samples, no new test split','members':selected},indent=2))
    canvas=Image.new('RGB',(1280,4*220),'#eeeeee');draw=ImageDraw.Draw(canvas);sampledir=DATA/'fixed-samples';sampledir.mkdir(exist_ok=True)
    for k,n in enumerate(selected):
        raw=z.read(n);name=n.replace('/','__');(sampledir/name).write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as im:tile=ImageOps.contain(im.convert('RGB'),(158,184))
        x=k%8*160;y=k//8*220;canvas.paste(tile,(x,y+35));draw.text((x+2,y+2),Path(n).parent.name+'\n'+Path(n).name,fill='black')
    canvas.save(OUT/'aider-fixed-contact-sheet.jpg',quality=93)
out={'source_record':3888300,'compressed_bytes':dest.stat().st_size,'uncompressed_bytes':pre['uncompressed_bytes'],'publisher_md5':expected,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'md5_matches':True,'all_members_crc_ok':True,'all_image_decode_failures':failures,'original_class_counts':dict(counts),'all_images_decoded':not failures,'annotation_type':'original mutually exclusive scene classes; no boxes; fire and smoke combined','physical_four_state_counts':None,'labels_changed':False,'images_deleted':False,'cross_dataset_deduplication':False}
(OUT/'aider-full-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False),flush=True)
