"""Fixed samples of original Home-fire data; numeric IDs preserved, no new GT."""
import collections,concurrent.futures,hashlib,importlib.util,io,json,struct,sys,zipfile,zlib,binascii
from pathlib import Path
import requests
from PIL import Image,ImageDraw,ImageOps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'references/new-candidates-2026-10-19';DEST=ROOT/'data/new-candidates-2026-10-19/homefire/fixed-samples'
assert Path(sys.prefix).name=='sci-env';DEST.mkdir(parents=True,exist_ok=True)
records=json.loads((OUT/'homefire-records.json').read_text());selected=[]
for state in ('0','1','0,1','empty'):
    selected+=sorted((r for r in records if r['raw_class_combination']==state),key=lambda r:hashlib.sha256(r['label'].encode()).hexdigest())[:6]
(OUT/'homefire-fixed-sample.json').write_text(json.dumps({'rule':'Six per raw class combination, minimum SHA256(full label member name), fixed before viewing. Not a cleaned subset.', 'samples':selected},indent=2))
spec=importlib.util.spec_from_file_location('zipinspect',ROOT/'code/dataset-verification-2026-10-16/inspect_archives.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
url='https://www.kaggle.com/api/v1/datasets/download/pengbo00/home-fire-dataset';size=1821136811;r=m.RangeFile(url,size)
with zipfile.ZipFile(r) as z:infos={x.filename:x for x in z.infolist()}
def fetch(a,b):
    with requests.get(url,headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'},stream=True,timeout=(15,40)) as response:
        assert response.status_code==206 and response.headers['Content-Range']==f'bytes {a}-{b}/{size}'
        raw=response.raw.read(b-a+2);assert len(raw)==b-a+1;return raw
def load(rec):
    name=rec['label'].replace('/labels/','/images/').replace('.txt','.jpg');i=infos[name];p=DEST/name.replace('/','__')
    if not p.exists():
        h=fetch(i.header_offset,i.header_offset+29);assert h[:4]==b'PK\x03\x04';fn,extra=struct.unpack_from('<HH',h,26);a=i.header_offset+30+fn+extra
        assert i.compress_size<10_000_000
        comp=fetch(a,a+i.compress_size-1);raw=zlib.decompress(comp,-15) if i.compress_type==8 else comp if i.compress_type==0 else None
        assert raw is not None and len(raw)==i.file_size and binascii.crc32(raw)&0xffffffff==i.CRC;p.write_bytes(raw)
    raw=p.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:im.load();dims=im.size
    print('decoded '+name,flush=True);return {'member':name,'saved':str(p.relative_to(ROOT)),'bytes':len(raw),'dimensions':dims,'sha256':hashlib.sha256(raw).hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:receipt=list(ex.map(load,selected))
can=Image.new('RGB',(1440,1120),'#eeeeee');d=ImageDraw.Draw(can)
for k,(rec,rr) in enumerate(zip(selected,receipt)):
    with Image.open(ROOT/rr['saved']) as im:
        im=im.convert('RGB');dd=ImageDraw.Draw(im);W,H=im.size
        for c,x,y,w,h in rec['boxes']:dd.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),outline='red' if c==0 else 'cyan',width=max(2,W//300))
        tile=ImageOps.contain(im,(238,245))
    x=k%6*240;y=k//6*280;can.paste(tile,(x,y+35));d.text((x+2,y+2),'IDs '+rec['raw_class_combination']+' '+Path(rec['label']).name,fill='black')
can.save(OUT/'homefire-fixed-contact-sheet.jpg',quality=93)
(OUT/'homefire-sample-receipt.json').write_text(json.dumps({'samples':receipt,'all_24_decoded':True,'class_mapping_is_visual_inference_only':True},indent=2)+'\n')
