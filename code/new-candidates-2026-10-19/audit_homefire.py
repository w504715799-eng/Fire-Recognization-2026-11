"""Bounded original ZIP directory/label audit; no relabeling, deletion or models."""
import collections,importlib.util,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'references/new-candidates-2026-10-19'
assert Path(sys.prefix).name=='sci-env'
spec=importlib.util.spec_from_file_location('zipinspect',ROOT/'code/dataset-verification-2026-10-16/inspect_archives.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class CachedRange(m.RangeFile):
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        for (a,b),raw in self.cache.items():
            if a<=self.pos and self.pos+n<=b+1:
                out=raw[self.pos-a:self.pos-a+n];self.pos+=n;return out
        return super().read(n)
r=CachedRange('https://www.kaggle.com/api/v1/datasets/download/pengbo00/home-fire-dataset',1821136811)
with zipfile.ZipFile(r) as z:
    infos=z.infolist();es=[{'path':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'offset':i.header_offset} for i in infos]
    total=sum(i.file_size for i in infos);assert max(r.size,total)<=30_000_000_000
    pre={'compressed_bytes':r.size,'uncompressed_bytes':total,'entries':len(infos),'extensions':dict(collections.Counter(Path(i.filename).suffix for i in infos)),'full_archive_downloaded':False,'whole_checksum_verified':False}
    (OUT/'homefire-preflight.json').write_text(json.dumps(pre,indent=2));(OUT/'homefire-files.json').write_text(json.dumps(es));print(json.dumps(pre),flush=True)
    labels=[i for i in infos if '/labels/' in i.filename and i.filename.endswith('.txt')]
    # Coalesce adjacent original label byte ranges. All source labels are read, never selected by outcome.
    ranges=[]
    for i in sorted(labels,key=lambda x:x.header_offset):
        a=i.header_offset;b=min(r.size-1,a+30+len(i.filename.encode('utf8'))+len(i.extra)+i.compress_size+2048)
        if ranges and a-ranges[-1][1]<4096 and b-ranges[-1][0]<14_000_000:ranges[-1][1]=max(ranges[-1][1],b)
        else:ranges.append([a,b])
    print('Label read spans '+str(len(ranges)),flush=True)
    assert sum(b-a+1 for a,b in ranges)<100_000_000,'Unexpected scattered labels; stop large data reads'
    for a,b in ranges:r.seek(a);r.read(b-a+1)
    counts=collections.Counter();boxcounts=collections.Counter();by_split=collections.defaultdict(collections.Counter);records=[];bad=[]
    labels_out=OUT/'homefire-original-labels';labels_out.mkdir(exist_ok=True)
    for i in labels:
        raw=z.read(i);path=labels_out/i.filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        classes=set();boxes=[]
        for line in raw.decode('utf-8-sig').splitlines():
            if not line.strip():continue
            vs=line.split()
            try:
                assert len(vs)==5
                c=int(vs[0]);x,y,w,h=map(float,vs[1:]);classes.add(c);boxcounts[c]+=1;boxes.append([c,x,y,w,h])
                if w<=0 or h<=0 or min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:bad.append({'path':i.filename,'line':line,'reason':'geometry'})
            except Exception:bad.append({'path':i.filename,'line':line,'reason':'parse'})
        state=','.join(map(str,sorted(classes))) or 'empty';split=i.filename.split('/')[0];counts[state]+=1;by_split[split][state]+=1;records.append({'label':i.filename,'split':split,'raw_class_combination':state,'boxes':boxes})
    imagepaths={i.filename for i in infos if Path(i.filename).suffix.lower() in ['.jpg','.jpeg','.png']}
    imstems={str(Path(n).with_suffix('')) for n in imagepaths};labelstems={str(Path(i.filename.replace('/labels/','/images/')).with_suffix('')) for i in labels}
    extra={}
    for i in infos:
        if Path(i.filename).suffix.lower() in ['.yaml','.yml','.names','.md'] or Path(i.filename).name in ['classes.txt','README.txt']:
            if i.file_size<100_000:extra[i.filename]=z.read(i).decode(errors='replace')
    out={'image_entries':len(imagepaths),'label_entries':len(labels),'raw_class_combinations':dict(counts),'box_counts':dict(boxcounts),'by_split':{k:dict(v) for k,v in by_split.items()},'images_missing_label':sorted(imstems-labelstems),'labels_missing_image':sorted(labelstems-imstems),'anomaly_count':len(bad),'class_map_metadata':extra,'range_bytes':r.fetched,'all_images_decoded':False}
    (OUT/'homefire-label-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(OUT/'homefire-box-anomalies.json').write_text(json.dumps(bad,indent=2));(OUT/'homefire-records.json').write_text(json.dumps(records));print(json.dumps(out,ensure_ascii=False),flush=True)
