"""Audit original COCO metadata and fixed small visual sample; never modify GT."""
import collections,hashlib,importlib.util,io,json,sys,zipfile,concurrent.futures,struct,zlib,binascii
from pathlib import Path
import requests
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'references/direct-data-search-2026-10-19'
assert Path(sys.prefix).name=='sci-env'
entries=json.loads((OUT/'fasdd-original-files.json').read_text())
images={Path(x['path']).name:x for x in entries if x['path'].endswith('.jpg')}
records=[];summary={};bad=[];orphans=[];totals=collections.Counter()
for split in ('train','val','test'):
    o=json.loads((OUT/f'original-coco-{split}.json').read_text());byid={i['id']:i for i in o['images']};anns=collections.defaultdict(list)
    assert o['categories']==[{'id':0,'name':'fire','supercategory':'null'},{'id':1,'name':'smoke','supercategory':'null'}]
    assert len(byid)==len(o['images'])
    assert len(set(a['id'] for a in o['annotations']))==len(o['annotations'])
    for a in o['annotations']:
        if a['image_id'] not in byid:orphans.append((split,a['id']));continue
        im=byid[a['image_id']];x,y,w,h=a['bbox']
        if a['category_id'] not in (0,1) or w<=0 or h<=0 or x<0 or y<0 or x+w>im['width'] or y+h>im['height']:bad.append({'split':split,'ann_id':a['id'],'image':im['file_name'],'bbox':a['bbox'],'dimensions':[im['width'],im['height']]})
        anns[a['image_id']].append(a);totals[a['category_id']]+=1
    counts=collections.Counter()
    for i in o['images']:
        aa=anns[i['id']];cl={a['category_id'] for a in aa};state=str(int(0 in cl))+str(int(1 in cl));counts[state]+=1
        records.append({'file':i['file_name'],'split':split,'state':state,'width':i['width'],'height':i['height'],'boxes':[{'class':a['category_id'],'bbox':a['bbox']} for a in aa]})
    summary[split]={'images':len(o['images']),'annotations':len(o['annotations']),'states':dict(counts)}
names=[Path(r['file']).name for r in records]
states=collections.Counter(r['state'] for r in records)
out={'source':'author FASDD_CV.zip V9, original COCO train/val/test JSON', 'image_count':len(records),'states':dict(states),'split_counts':summary,'box_counts':dict(totals),'invalid_or_out_of_bounds_boxes':len(bad),'orphan_annotations':orphans,'repeated_image_record_names':[k for k,v in collections.Counter(names).items() if v>1],'labels_without_archive_image':sorted(set(names)-set(images)),'archive_images_without_coco_record':sorted(set(images)-set(names)),'all_images_decoded':False,'whole_archive_md5_verified':False,'state_counts_are_annotation_based_not_physical_guarantee':True}
(OUT/'fasdd-original-label-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(OUT/'fasdd-original-box-anomalies.json').write_text(json.dumps(bad,indent=2));(OUT/'fasdd-original-records.json').write_text(json.dumps(records));print(json.dumps(out,ensure_ascii=False),flush=True)
selected=[]
for state in ('10','01','11','00'):
    selected+=sorted((r for r in records if r['state']==state),key=lambda r:hashlib.sha256(Path(r['file']).name.encode()).hexdigest())[:6]
(OUT/'fasdd-original-fixed-sample.json').write_text(json.dumps({'rule':'First six SHA256(filename) per annotation state, fixed before viewing. No evaluation split or new GT.', 'samples':selected},indent=2))
if '--sample' not in sys.argv:sys.exit()
spec=importlib.util.spec_from_file_location('inspectzip',ROOT/'code/dataset-verification-2026-10-16/inspect_archives.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class R(m.RangeFile):
    def read(self,n=-1):
        if n>15_000_000:
            if n>100_000_000:raise ValueError('metadata cap')
            return b''.join(super(R,self).read(min(8_000_000,n-k)) for k in range(0,n,8_000_000))
        return super().read(n)
url='https://china.scidb.cn/download?fileId=85bad3972ed60e0a20a790032c0d85fb';size=12326081298;remote=R(url,size)
with zipfile.ZipFile(remote) as z: infos={Path(i.filename).name:i for i in z.infolist() if i.filename.endswith('.jpg')}
dest=ROOT/'data/supplementary-verification-2026-10-19/fasdd-original-sample';dest.mkdir(parents=True,exist_ok=True)
def fetchrange(start,end):
    with requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(15,40)) as resp:
        assert resp.status_code==206 and resp.headers['Content-Range']==f'bytes {start}-{end}/{size}'
        raw=resp.raw.read(end-start+2);assert len(raw)==end-start+1;return raw
def sample(r):
    name=Path(r['file']).name;i=infos[name];p=dest/name
    if not p.exists():
        assert i.compress_size<10_000_000
        header=fetchrange(i.header_offset,i.header_offset+29);assert header[:4]==b'PK\x03\x04'
        fn,extra=struct.unpack_from('<HH',header,26);start=i.header_offset+30+fn+extra
        compressed=fetchrange(start,start+i.compress_size-1)
        raw=zlib.decompress(compressed,-15) if i.compress_type==8 else compressed if i.compress_type==0 else None
        assert raw is not None and len(raw)==i.file_size and binascii.crc32(raw)&0xffffffff==i.CRC
        p.write_bytes(raw)
    raw=p.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:im.load();dims=im.size
    assert dims==(r['width'],r['height'])
    print('decoded '+name,flush=True)
    return {'file':name,'bytes':len(raw),'dimensions':dims,'sha256':hashlib.sha256(raw).hexdigest(),'state':r['state']}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:receipt=list(ex.map(sample,selected))
canvas=Image.new('RGB',(1440,4*280),'#eeeeee');draw=ImageDraw.Draw(canvas)
for k,r in enumerate(selected):
    with Image.open(dest/Path(r['file']).name) as im:
        im=im.convert('RGB');d=ImageDraw.Draw(im)
        for a in r['boxes']:
            x,y,w,h=a['bbox'];d.rectangle((x,y,x+w,y+h),outline='red' if a['class']==0 else 'cyan',width=max(2,im.width//300))
        tile=ImageOps.contain(im,(238,245))
    x=k%6*240;y=k//6*280;canvas.paste(tile,(x,y+30));draw.text((x+2,y+2),r['state']+' '+Path(r['file']).name,fill='black')
canvas.save(OUT/'fasdd-original-contact-sheet.jpg',quality=93)
(OUT/'fasdd-original-sample-receipt.json').write_text(json.dumps({'sample_images':receipt,'all_sample_images_decoded':True,'full_archive_downloaded':False,'labels_changed':False},indent=2)+'\n')
