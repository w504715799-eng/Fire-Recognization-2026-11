"""Small public metadata and existing-label audit; no weights/images/deduplication."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import re
import sys
import requests

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'references/research-reassessment-2026-11-04'
OUT.mkdir(parents=True, exist_ok=True)
assert Path(sys.prefix).name == 'sci-env'
LIMIT = 2_000_000


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def capture(item):
    name, url = item
    try:
        with requests.get(url, timeout=(10, 30), stream=True) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length', '0')) > LIMIT:
                raise ValueError('Metadata limit exceeded; body not downloaded')
            body = bytearray()
            for chunk in response.iter_content(65536):
                if len(body) + len(chunk) > LIMIT:
                    raise ValueError('Metadata limit exceeded; partial body discarded')
                body.extend(chunk)
            (OUT / name).write_bytes(body)
            return {'file': name, 'url': url, 'status': response.status_code,
                    'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    except Exception as exc:
        return {'file': name, 'url': url, 'error': str(exc)}


SOURCES = [
    ('dfire-readme.md', 'https://raw.githubusercontent.com/gaia-solutions-on-demand/DFireDataset/master/README.md'),
    ('homefire-repo.json', 'https://api.github.com/repos/PengBo0/Home-fire-dataset'),
    ('homefire-release.json', 'https://api.github.com/repos/PengBo0/Home-fire-dataset/releases/tags/v1.0.0'),
    ('fasdd-official.html', 'https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda'),
    ('fasdd-mirror-tree.json', 'https://huggingface.co/api/datasets/mountainmoon2000/wildfire_mm/tree/main?recursive=false'),
    ('qwen-model-metadata.json', 'https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B'),
]
with ThreadPoolExecutor(max_workers=4) as pool:
    captures = list(pool.map(capture, SOURCES))
repo_file = OUT / 'homefire-repo.json'
if repo_file.exists():
    branch = json.loads(repo_file.read_text()).get('default_branch', 'main')
    captures.append(capture(('homefire-readme.md', f'https://raw.githubusercontent.com/PengBo0/Home-fire-dataset/{branch}/README.md')))
save('capture-index.json', captures)

home_root = ROOT / 'references/new-candidates-2026-10-19/homefire-original-labels'
home = {'files': 0, 'boxes': Counter(), 'states': Counter(), 'splits': Counter(), 'malformed_lines': 0}
for p in home_root.rglob('*.txt'):
    counts = Counter()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            c, x, y, w, h = line.split()
            c = int(c)
            x, y, w, h = map(float, (x, y, w, h))
            assert c in (0, 1) and all(math.isfinite(v) for v in (x, y, w, h))
            assert w > 0 and h > 0 and min(x-w/2, y-h/2) >= -1e-6 and max(x+w/2, y+h/2) <= 1+1e-6
            counts[c] += 1
        except (ValueError, AssertionError):
            home['malformed_lines'] += 1
    home['files'] += 1
    home['boxes'].update(counts)
    home['splits'][p.relative_to(home_root).parts[0]] += 1
    home['states']['both' if counts[0] and counts[1] else 'fire_only' if counts[0] else 'smoke_only' if counts[1] else 'empty'] += 1

cv = {}
for split in ('train', 'val', 'test'):
    p = ROOT / f'references/direct-data-search-2026-10-19/original-coco-{split}.json'
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    obj = json.loads(p.read_text())
    images = {im['id']: im for im in obj['images']}
    boxes = Counter()
    outside = degenerate = 0
    excursion = 0.0
    for ann in obj['annotations']:
        boxes[ann['category_id']] += 1
        im = images[ann['image_id']]
        x, y, w, h = ann['bbox']
        degenerate += int(w <= 0 or h <= 0)
        ex = max(0, -x, -y, x+w-im['width'], y+h-im['height'])
        outside += int(ex > 1e-6)
        excursion = max(excursion, ex)
    assert before == hashlib.sha256(p.read_bytes()).hexdigest()
    cv[split] = {'images': len(obj['images']), 'annotations': len(obj['annotations']),
                 'categories': obj['categories'], 'boxes': boxes, 'outside': outside,
                 'degenerate': degenerate, 'max_excursion_px': excursion, 'source_sha256': before}

manifests = {}
for name in ('homefire', 'fasdd-cv'):
    p = ROOT / f'references/data-protocol-2026-10-23/{name}-split-manifest.jsonl'
    roles = Counter(json.loads(line)['role'] for line in p.read_text().splitlines() if line)
    manifests[name] = {'roles': roles, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

summary = {'date': '2026-11-04', 'homefire_existing_labels': home,
           'fasdd_cv_existing_coco': cv, 'existing_manifests': manifests,
           'cross_dataset_comparison': False, 'labels_modified': False,
           'images_or_weights_downloaded': False, 'full_image_packages_checked': False,
           'python': sys.executable, 'metadata_download_bytes': sum(x.get('bytes', 0) for x in captures)}
release = OUT / 'homefire-release.json'
if release.exists():
    r = json.loads(release.read_text())
    summary['homefire_public_release'] = {'tag': r.get('tag_name'), 'published_at': r.get('published_at'),
        'assets': [{k:a.get(k) for k in ('name','size','browser_download_url')} for a in r.get('assets',[])]}
html = OUT / 'fasdd-official.html'
if html.exists():
    blocks = re.findall(r'<script[^>]*type=[\"\x27]application/ld\+json[\"\x27][^>]*>(.*?)</script>', html.read_text(), re.S)
    for block in blocks:
        try:
            meta = json.loads(block)
            save('fasdd-official-structured.json', meta)
            summary['fasdd_official'] = {k:meta.get(k) for k in ('name','version','license','size','description')}
        except json.JSONDecodeError:
            pass
save('public-data-recheck.json', summary)
print(json.dumps({'homefire_labels': home['files'], 'homefire_errors': home['malformed_lines'],
    'homefire_states':home['states'], 'cv_images':sum(x['images'] for x in cv.values()),
    'cv_annotations':sum(x['annotations'] for x in cv.values()),
    'cv_degenerate':sum(x['degenerate'] for x in cv.values()),
    'cv_outside':sum(x['outside'] for x in cv.values()),
    'manifests':{k:v['roles'] for k,v in manifests.items()},
    'captures':captures}, ensure_ascii=False))
