"""Read author ZIP metadata and original COCO labels over bounded HTTP ranges; no models."""
import collections, importlib.util, json, zipfile, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'references/direct-data-search-2026-10-19'
assert Path(sys.prefix).name=='sci-env'
spec=importlib.util.spec_from_file_location('inspectzip',ROOT/'code/dataset-verification-2026-10-16/inspect_archives.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Range(m.RangeFile):
    def read(self,n=-1):
        if n>15_000_000:
            if n>100_000_000:raise ValueError('metadata range cap exceeded')
            return b''.join(super(Range,self).read(min(8_000_000,n-k)) for k in range(0,n,8_000_000))
        return super().read(n)
URL='https://china.scidb.cn/download?fileId=85bad3972ed60e0a20a790032c0d85fb'
info=json.loads((OUT/'fasdd-original-fileinfo-front.json').read_text())['data']['content']
assert info['size']<=30_000_000_000
r=Range(URL,info['size'])
with zipfile.ZipFile(r) as z:
    es=[{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':f'{i.CRC:08x}'} for i in z.infolist()]
    size=sum(e['bytes'] for e in es)
    out={'source':URL,'compressed_bytes':info['size'],'uncompressed_bytes':size,'publisher_md5':info['md5'],'md5_verified':False,
         'full_archive_downloaded':False,'all_images_decoded':False,'entries':len(es),
         'extensions':dict(collections.Counter(Path(e['path']).suffix.lower() for e in es)),
         'size_gate_passed':size<=30_000_000_000,'range_bytes':r.fetched}
    (OUT/'fasdd-original-files.json').write_text(json.dumps(es))
    (OUT/'fasdd-original-preflight.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)
    assert out['size_gate_passed']
    js=[i for i in z.infolist() if i.filename.endswith('.json')]
    print('JSON entries '+str([(i.filename,i.file_size,i.compress_size) for i in js]),flush=True)
    for i in js:
        if '/COCO' in i.filename and i.file_size<=100_000_000:
            raw=z.read(i);(OUT/('original-coco-'+Path(i.filename).name)).write_bytes(raw)
            o=json.loads(raw);print(json.dumps({'file':i.filename,'categories':o.get('categories'),'images':len(o.get('images',[])),'annotations':len(o.get('annotations',[]))}),flush=True)
    out['range_bytes']=r.fetched
    (OUT/'fasdd-original-preflight.json').write_text(json.dumps(out,indent=2)+'\n')
