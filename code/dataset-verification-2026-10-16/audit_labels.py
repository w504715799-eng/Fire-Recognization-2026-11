"""Stream all ZIP entries, decode images, audit original YOLO labels. No deduplication."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import time
import zipfile
from PIL import Image
import yaml

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'references/dataset-verification-2026-10-16'
if Path(sys.prefix).name!='sci-env':raise RuntimeError('Use sci-env')
IMG_EXT={'.jpg','.jpeg','.png','.bmp','.webp'}

def ignored(name):
    p=PurePosixPath(name)
    return '__MACOSX' in p.parts or p.name.startswith('._')

def split(name):
    return next((x for x in PurePosixPath(name).parts if x in ('train','test','valid','val')), 'unspecified')

def audit(key):
    source=ROOT/'data/supplementary-verification-2026-10-16'/key/'original.zip'
    start=time.monotonic()
    with zipfile.ZipFile(source) as z:
        entries=z.infolist(); files=[x for x in entries if not x.is_dir()]
        images=[x for x in files if not ignored(x.filename) and PurePosixPath(x.filename).suffix.lower() in IMG_EXT]
        labels=[x for x in files if not ignored(x.filename) and '/labels/' in '/'+x.filename and x.filename.lower().endswith('.txt')]
        named={x.filename:x for x in files}
        metadata={}
        read=set()
        for x in files:
            if x.file_size<100_000 and not ignored(x.filename) and (x.filename.lower().endswith(('.yaml','.yml','.names')) or 'readme' in x.filename.lower() or PurePosixPath(x.filename).name=='classes.txt'):
                metadata[x.filename]=z.read(x).decode('utf-8',errors='replace');read.add(x.filename)
        configs=[yaml.safe_load(v) for k,v in metadata.items() if k.endswith(('.yaml','.yml'))]
        configs=[x for x in configs if isinstance(x,dict) and 'names' in x]
        if len(configs)!=1:raise RuntimeError(f'Need explicit class mapping, found {len(configs)} configs')
        names=configs[0]['names']
        mapping={int(k):str(v).lower() for k,v in names.items()} if isinstance(names,dict) else {i:str(v).lower() for i,v in enumerate(names)}
        semantic_mapping_explicit=set(mapping.values())=={'fire','smoke'}
        if not semantic_mapping_explicit:
            if mapping!={0:'0',1:'1'}:raise RuntimeError(f'Unexpected names: {mapping}')
            mapping={0:'class_0',1:'class_1'}
        positive0='fire' if semantic_mapping_explicit else 'class_0'
        positive1='smoke' if semantic_mapping_explicit else 'class_1'
        counters=defaultdict(Counter);dimensions=Counter();box_counts=Counter();issues=[];rows=[]
        matched_labels=set();label_basenames={x.filename for x in labels}
        for idx,im in enumerate(images):
            name=im.filename; sp=split(name)
            lp=str(PurePosixPath(name).with_suffix('.txt')).replace('/images/','/labels/')
            if lp.startswith('images/'):lp='labels/'+lp[len('images/'):]
            rec={'image':name,'split':sp,'label':lp,'issues':[],'boxes':[],'image_readable':False}
            present=set();unknown=False
            try:
                raw=z.read(im);read.add(name)
                with Image.open(io.BytesIO(raw)) as img:
                    rec['width'],rec['height']=img.size;rec['format']=img.format;img.load()
                rec['image_readable']=True;dimensions[f'{rec["width"]}x{rec["height"]}']+=1
            except Exception as e:rec['issues'].append('image_read_error');rec['image_error']=str(e)
            if lp not in label_basenames:
                rec['issues'].append('missing_label');unknown=True
            else:
                matched_labels.add(lp)
                try:
                    txt=z.read(named[lp]).decode('utf-8-sig');read.add(lp)
                    rec['label_text']=txt
                    for ln,line in enumerate(txt.splitlines(),1):
                        if not line.strip():continue
                        parts=line.split()
                        try:
                            cls_float=float(parts[0]);cls=int(cls_float)
                            if cls!=cls_float or cls not in mapping:raise ValueError('invalid class')
                            present.add(mapping[cls])
                        except Exception:
                            rec['issues'].append(f'line_{ln}:invalid_class');unknown=True;continue
                        if len(parts)!=5:
                            rec['issues'].append(f'line_{ln}:not_5_columns');continue
                        try: vals=list(map(float,parts[1:]))
                        except ValueError:rec['issues'].append(f'line_{ln}:non_numeric');continue
                        x,y,w,h=vals
                        flags=[]
                        if not all(map(math.isfinite,vals)):flags.append('nonfinite')
                        else:
                            if w<=0 or h<=0:flags.append('nonpositive_area')
                            if not all(0<=v<=1 for v in vals):flags.append('coordinate_outside_0_1')
                            if min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:flags.append('box_crosses_border')
                        rec['boxes'].append({'class':mapping[cls],'xywh':vals,'flags':flags,'line':ln})
                        box_counts[mapping[cls]]+=1
                        for fl in flags:rec['issues'].append(f'line_{ln}:{fl}')
                except Exception as e:
                    rec['issues'].append('label_read_error');rec['label_error']=str(e);unknown=True
            state='unknown' if unknown else 'both' if present=={positive0,positive1} else ('fire_only' if semantic_mapping_explicit else 'label0_only') if present=={positive0} else ('smoke_only' if semantic_mapping_explicit else 'label1_only') if present=={positive1} else 'background'
            rec['declared_state']=state
            stem=PurePosixPath(name).stem
            rec['filename_source_proxy']=re.sub(r'\.rf\.[^.]+$','',stem)
            rec['small_fire_boxes_640']=sum(b['class']=='fire' and not b['flags'] and b['xywh'][2]*rec.get('width',0)*b['xywh'][3]*rec.get('height',0)*(640/max(rec.get('width',640),rec.get('height',640)))**2<1024 for b in rec['boxes']) if semantic_mapping_explicit else None
            for scope in ('all',sp):
                counters[scope]['images']+=1;counters[scope][state]+=1
                counters[scope]['image_readable']+=int(rec['image_readable'])
                counters[scope]['label_present']+=int(lp in label_basenames)
                counters[scope]['structurally_clean_images']+=int(not rec['issues'])
                if semantic_mapping_explicit:
                    counters[scope]['small_fire_positive_images']+=int(rec['small_fire_boxes_640']>0)
                    counters[scope]['small_fire_boxes_640']+=rec['small_fire_boxes_640']
                for b in rec['boxes']:counters[scope][b['class']+'_boxes']+=1
                if not rec['issues']:counters[scope]['clean_'+state]+=1
            if rec['issues']:issues.append({'image':name,'issues':rec['issues']})
            rows.append(rec)
            if (idx+1)%1000==0:print(f'{key}: audited {idx+1}/{len(images)} images',flush=True)
        orphan=sorted(label_basenames-matched_labels)
        unread_errors=[]
        for x in files:
            if x.filename not in read:
                try:z.read(x)
                except Exception as e:unread_errors.append({'file':x.filename,'error':str(e)})
        proxy=defaultdict(list)
        for r in rows:proxy[r['filename_source_proxy']].append(r)
        proxy_summary={'description':'within-dataset filename before Roboflow suffix; not event IDs; no files removed',
                       'groups':len(proxy),'multi_file_groups':sum(len(v)>1 for v in proxy.values()),
                       'cross_split_proxy_groups':sum(len({x['split'] for x in v})>1 for v in proxy.values()),
                       'groups_by_declared_state':{s:len({r['filename_source_proxy'] for r in rows if r['declared_state']==s}) for s in sorted({r['declared_state'] for r in rows})}}
        summary={'key':key,'archive':str(source.relative_to(ROOT)),'files':len(files),'images':len(images),
                 'labels':len(labels),'class_mapping':mapping,'semantic_mapping_explicit':semantic_mapping_explicit,'by_split':dict(counters),'box_counts':dict(box_counts),
                 'dimensions':dict(dimensions),'orphan_labels':orphan,'image_or_label_issue_images':len(issues),
                 'issue_types':dict(Counter(re.sub(r'^line_\d+:','',e) for r in issues for e in r['issues'])),
                 'other_entry_crc_errors':unread_errors,'package_metadata':metadata,'filename_proxy':proxy_summary,
                 'uncompressed_bytes_all_entries':sum(x.file_size for x in entries),
                 'ignored_packaging_files':sum(ignored(x.filename) for x in files),
                 'seconds':time.monotonic()-start,'cross_dataset_comparison_performed':False,'files_removed':False}
        (OUT/f'{key}-audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        (OUT/f'{key}-issues.json').write_text(json.dumps(issues,ensure_ascii=False,indent=2)+'\n')
        with (OUT/f'{key}-image-manifest.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('key',choices=['cqu','indoor']);audit(p.parse_args().key)
