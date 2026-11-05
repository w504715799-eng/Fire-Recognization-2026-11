"""Deterministic within-dataset audit contact sheets. Does not change source labels."""
import argparse
from pathlib import Path
import hashlib
import io
import json
import sys
import zipfile
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
if Path(sys.prefix).name!='sci-env':raise RuntimeError('Use sci-env')
OUT=ROOT/'references/dataset-verification-2026-10-16'
p=argparse.ArgumentParser();p.add_argument('key',choices=['cqu','indoor']);key=p.parse_args().key
rows=[json.loads(x) for x in (OUT/f'{key}-image-manifest.jsonl').read_text().splitlines()]
selected=[]
with zipfile.ZipFile(ROOT/'data/supplementary-verification-2026-10-16'/key/'original.zip') as z:
    for state in sorted({r['declared_state'] for r in rows}):
        pool=sorted([r for r in rows if r['declared_state']==state], key=lambda r:hashlib.sha256(('visual-audit-20261016'+r['image']).encode()).hexdigest())
        pick=[];seen=set()
        for r in pool:
            if r['filename_source_proxy'] in seen:continue
            seen.add(r['filename_source_proxy']);pick.append(r)
            if len(pick)==12:break
        grid=Image.new('RGB',(1280,270*((len(pick)+3)//4)),(245,245,245));d=ImageDraw.Draw(grid)
        for i,r in enumerate(pick):
            im=Image.open(io.BytesIO(z.read(r['image']))).convert('RGB'); im.thumbnail((310,235))
            x=(i%4)*320+(320-im.width)//2;y=(i//4)*270+25
            grid.paste(im,(x,y));d.text(((i%4)*320+5,(i//4)*270+5),f'{key} {state} #{i+1} {r["split"]}',fill='black')
            for b in r['boxes']:
                cx,cy,w,h=b['xywh'];box=[x+(cx-w/2)*im.width,y+(cy-h/2)*im.height,x+(cx+w/2)*im.width,y+(cy+h/2)*im.height]
                d.rectangle(box,outline='red' if b['class'] in ('fire','class_0') else 'blue',width=2)
            selected.append({'key':key,'state':state,'number':i+1,'image':r['image'],'label':r['label']})
        grid.save(OUT/f'{key}-visual-{state}.jpg',quality=90)
(OUT/f'{key}-visual-sample.json').write_text(json.dumps(selected,indent=2)+'\n')
print(json.dumps({'key':key,'sample_images':len(selected),'method':'fixed SHA256 ranking, at most one per filename-prefix group, 12 per declared state','source_files_changed':False}))
