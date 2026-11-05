"""Original image-level CSV and fixed visual sample audit. No models or relabeling."""
import collections, csv, hashlib, importlib.util, io, json, sys, zipfile
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'references/data-gap-search-2026-10-19'
assert Path(sys.prefix).name == 'sci-env'
rows = list(csv.reader((OUT/'flickr-firesmoke-original-labels.csv').open()))
files = json.loads((OUT/'fismo-images-file-list.json').read_text())
prefix = 'FiSmo-Images/Flickr-FireSmoke/imgs/'
images = [x for x in files if x['path'].startswith(prefix) and x['path'].lower().endswith('.jpg')]
names = [x['path'][len(prefix):] for x in images]
bad = [r for r in rows if len(r) != 3 or r[1] not in ('0','1') or r[2] not in ('0','1')]
counts = collections.Counter(''.join(r[1:]) for r in rows)
audit = {'csv_rows': len(rows), 'image_entries': len(images), 'image_uncompressed_bytes': sum(x['bytes'] for x in images),
         'raw_column_state_counts': dict(counts), 'invalid_rows': bad,
         'repeated_label_filenames': [k for k,v in collections.Counter(r[0] for r in rows).items() if v>1],
         'labels_without_image_entry': sorted(set(r[0] for r in rows)-set(names)),
         'images_without_label': sorted(set(names)-set(r[0] for r in rows)),
         'csv_sha256': hashlib.sha256((OUT/'flickr-firesmoke-original-labels.csv').read_bytes()).hexdigest(),
         'annotation_level': 'two binary image-level EMERGENCY fire/smoke labels; no bounding boxes',
         'column_mapping_evidence': 'Original FiSmo paper section 2.1 and Table 2; first binary is emergency fire, second is emergency smoke',
         'physical_fire_only_count': None, 'physical_smoke_only_count': None, 'physical_background_count': None,
         'semantic_admission': 'failed: emergency-conditioned labels differ from physical visible-fire/smoke task',
         'full_archive_downloaded': False, 'all_images_decoded': False,
         'sampling': 'six smallest SHA256(filename) per raw state; fixed before viewing; no deletion or relabeling'}
assert not bad
selected = []
for state in ('10','01','11','00'):
    selected += sorted([r for r in rows if ''.join(r[1:]) == state], key=lambda r:hashlib.sha256(r[0].encode()).hexdigest())[:6]
audit['sample_rows'] = selected
(OUT/'flickr-label-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k!='sample_rows'},ensure_ascii=False),flush=True)
if '--sample' not in sys.argv: sys.exit()
spec=importlib.util.spec_from_file_location('zipinspect', ROOT/'code/dataset-verification-2026-10-16/inspect_archives.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
form=json.loads((OUT/'fismo-images-range-probe.json').read_text())['forms'][0]
url=requests.Request('GET',form['action'],params={k:v for k,v in form['inputs'] if k}).prepare().url
remote=mod.RangeFile(url,2603306346)
dest=ROOT/'data/supplementary-verification-2026-10-19/flickr-firesmoke-sample';dest.mkdir(parents=True,exist_ok=True)
receipt=[]
with zipfile.ZipFile(remote) as archive:
    canvas=Image.new('RGB',(1080,4*240),(240,240,240)); draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(selected):
        path=dest/row[0]
        if not path.exists(): path.write_bytes(archive.read(prefix+row[0]))
        raw=path.read_bytes()
        with Image.open(io.BytesIO(raw)) as im:
            im.load(); wh=im.size
            tile=ImageOps.contain(im.convert('RGB'),(178,205))
        x=(i%6)*180;y=(i//6)*240
        canvas.paste(tile,(x,y+35));draw.text((x+2,y+2),''.join(row[1:])+' '+row[0],fill='black')
        receipt.append({'filename':row[0],'raw_state':''.join(row[1:]),'dimensions':wh,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        print('decoded '+row[0],flush=True)
    canvas.save(OUT/'flickr-contact-sheet.jpg',quality=92)
(OUT/'flickr-sample-receipt.json').write_text(json.dumps({'images':receipt,'range_bytes_read':remote.fetched,'all_sample_images_decoded':True},indent=2)+'\n')
