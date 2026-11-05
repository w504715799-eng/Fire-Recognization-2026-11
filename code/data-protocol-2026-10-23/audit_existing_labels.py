"""CPU-only audit of existing annotations and deterministic split manifests.

No network, model libraries, image removal, label edits, or deduplication.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'references/data-protocol-2026-10-23'
SALT = 'fire-master-v1-20261022'
assert Path(sys.prefix).name == 'sci-env'
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def assign(rows, dataset):
    # A derived partition of the original validation set; author train/test unchanged.
    val = [r for r in rows if r['official_split'] == 'val']
    val.sort(key=lambda r: (hashlib.sha256(
        f'{SALT}|{dataset}|val|{r["file"]}'.encode()).hexdigest(), r['file']))
    dev = {r['file'] for r in val[:len(val)//2]}
    for r in rows:
        sp = r['official_split']
        r['role'] = 'fit' if sp == 'train' else 'test' if sp == 'test' else (
            'dev' if r['file'] in dev else 'cal')
    return rows


def manifest(name, rows):
    path = OUT / name
    with path.open('w') as f:
        for r in sorted(rows, key=lambda x: (x['official_split'], x['file'])):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    roles = defaultdict(Counter)
    for r in rows:
        roles[r['role']]['images'] += 1
        roles[r['role']][r['state']] += 1
        roles[r['role']]['fire_boxes'] += r['fire_boxes']
        roles[r['role']]['smoke_boxes'] += r['smoke_boxes']
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'by_role': {k: dict(v) for k, v in roles.items()}}


def audit_home():
    base = ROOT / 'references/new-candidates-2026-10-19/homefire-original-labels'
    rows, errors, digests = [], [], []
    for p in sorted(base.rglob('*.txt')):
        rel = p.relative_to(base).as_posix()
        counts = Counter()
        for line_no, line in enumerate(p.read_text().splitlines(), 1):
            if not line.strip():
                continue
            v = line.split()
            try:
                assert len(v) == 5
                cls = int(v[0]); x, y, w, h = map(float, v[1:])
                assert cls in (0, 1) and all(math.isfinite(z) for z in (x,y,w,h))
                assert w > 0 and h > 0
                assert min(x-w/2, y-h/2) >= -1e-6
                assert max(x+w/2, y+h/2) <= 1+1e-6
                counts[cls] += 1
            except Exception:
                errors.append({'file': rel, 'line': line_no, 'raw': line})
        state = 'both' if counts[0] and counts[1] else 'fire_only' if counts[0] else (
            'smoke_only' if counts[1] else 'empty')
        rows.append({'file': rel.replace('/labels/', '/images/').replace('.txt','.jpg'),
                     'label': rel, 'official_split': rel.split('/')[0],
                     'fire_boxes': counts[0], 'smoke_boxes': counts[1], 'state': state})
        digests.append((rel, sha(p)))
    assert len(rows) == 6500
    files_digest = hashlib.sha256(json.dumps(digests).encode()).hexdigest()
    result = {'source': str(base.relative_to(ROOT)), 'labels':len(rows),
              'mapping': {'0':'fire','1':'smoke'},
              'mapping_basis':'local raw-label/visual correspondence plus independent user implementation; no author numeric mapping file found',
              'label_digest':files_digest,'errors':errors,
              'manifest':manifest('homefire-split-manifest.jsonl', assign(rows,'homefire'))}
    save('homefire-review.json', result)
    return result


def audit_cv():
    base = ROOT / 'references/direct-data-search-2026-10-19'
    rows, issues, source_hashes, summaries = [], [], {}, {}
    max_excursion = 0.0
    for split in ('train','val','test'):
        p = base / f'original-coco-{split}.json'
        source_hashes[p.name] = sha(p)
        obj = json.loads(p.read_text())
        assert {c['id']:c['name'] for c in obj['categories']} == {0:'fire',1:'smoke'}
        images = {im['id']:im for im in obj['images']}
        anns = defaultdict(list)
        tally, affected = Counter(), set()
        for a in obj['annotations']:
            im = images[a['image_id']]
            anns[a['image_id']].append(a)
            x,y,w,h = a['bbox']; W,H=im['width'],im['height']
            assert all(math.isfinite(z) for z in (x,y,w,h))
            reasons=[]
            if w<=0 or h<=0:reasons.append('nonpositive')
            if x<0 or y<0:reasons.append('negative_origin')
            excursion=max(0.,x+w-W,y+h-H)
            max_excursion=max(max_excursion,excursion)
            if excursion>1e-6:reasons.append('outside')
            normalized=[(x+w/2)/W,(y+h/2)/H,w/W,h/H]
            # Numeric check only, not an import or execution of a detector.
            native_tolerance_ok=min(normalized)>=-.01 and max(normalized)<=1.01
            if not native_tolerance_ok:tally['outside_native_1percent_numeric_tolerance']+=1
            if reasons:
                affected.add(a['image_id'])
                tally.update(reasons)
                issues.append({'split':split,'file':im['file_name'],'ann_id':a['id'],
                    'category_id':a['category_id'],'bbox':a['bbox'],'area':a.get('area'),
                    'image_size':[W,H],'reasons':reasons,'excursion_px':excursion,
                    'normalized_view':normalized,'numeric_tolerance_ok':native_tolerance_ok})
        for im in obj['images']:
            aa=anns[im['id']];counts=Counter(a['category_id'] for a in aa)
            state='both' if counts[0] and counts[1] else 'fire_only' if counts[0] else (
                'smoke_only' if counts[1] else 'empty')
            rows.append({'file':im['file_name'],'image_id':im['id'],'official_split':split,
                'width':im['width'],'height':im['height'],'state':state,
                'fire_boxes':counts[0],'smoke_boxes':counts[1],
                'degenerate_ann_ids':[a['id'] for a in aa if a['bbox'][2]<=0 or a['bbox'][3]<=0]})
        annotation_ids=[a['id'] for a in obj['annotations']]
        summaries[split]={'images':len(images),'annotations':len(obj['annotations']),
                          'reasons':dict(tally),'affected_images':len(affected),
                          'annotation_id_zero':sum(x==0 for x in annotation_ids),
                          'duplicate_annotation_ids':len(annotation_ids)-len(set(annotation_ids))}
    assert len(rows)==95314
    rows=assign(rows,'fasdd_cv')
    lookup={(r['official_split'],r['file']):r['role'] for r in rows}
    for issue in issues:issue['derived_role']=lookup[(issue['split'],issue['file'])]
    result={'source_hashes':source_hashes,'by_official_split':summaries,
        'images':len(rows),'annotations':sum(x['annotations'] for x in summaries.values()),
        'anomalous_boxes':len(issues),'affected_images':len({(x['split'],x['file']) for x in issues}),
        'max_excursion_px':max_excursion,'degenerate':[x for x in issues if 'nonpositive' in x['reasons']],
        'manifest':manifest('fasdd-cv-split-manifest.jsonl',rows),
        'labels_edited':False,'images_removed':False,'models_imported':False,
        'whole_images_decoded_this_run':False}
    for name,digest in source_hashes.items():assert sha(base/name)==digest
    save('fasdd-cv-review.json',result)
    save('fasdd-cv-anomalies-reviewed.json',issues)
    return result


if __name__=='__main__':
    home=audit_home();cv=audit_cv()
    save('split-rule.json',{'salt':SALT,'rule':'sha256(salt|dataset|val|exact_original_file) sort; first floor(n/2) dev, remainder cal',
        'homefire_dataset_key':'homefire','fasdd_cv_dataset_key':'fasdd_cv',
        'original_train_and_test_unchanged':True,'group_ids_available':False,
        'cross_dataset_deduplication':False,'scope':'existing-label manifests, image restoration pending'})
    print(json.dumps({'home_labels':home['labels'],'home_errors':len(home['errors']),
        'home_roles':home['manifest']['by_role'],'cv':{k:cv[k] for k in ['images','annotations','anomalous_boxes','affected_images','max_excursion_px','by_official_split','degenerate']},
        'cv_roles':cv['manifest']['by_role']},ensure_ascii=False,indent=2))
