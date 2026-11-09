"""Offline W/S preparation. No teacher/model import, filtering or GT rewriting."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re
from pathlib import Path

from .common import ProjectError,sha256

SEMANTIC_FIELDS = ("fire_visible", "smoke_visible", "cloud_fog_like",
                   "bright_light_like", "reflection_like", "low_visibility")
SCENE_PROMPT = ('Describe only visible evidence in this image. Return one JSON object with exactly '
                + ', '.join(SEMANTIC_FIELDS) + '. Each value must be yes, no, or uncertain. '
                'Do not infer fire from smoke alone. Use uncertain when evidence is insufficient.')
REGION_PROMPT = ('Inspect the neutral outlined region 1 and its context. Return one JSON object '
                 'with exactly two fields: content and visibility. Choose one content value from '
                 'fire, smoke, both, other, uncertain. Choose one visibility value from clear, limited, '
                 'unjudgeable. Describe visible content; do not infer an unseen object. No other text.')


def digest_object(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


from .research_data import load_manifest


def fit_rows(rows):
    fit=[r for r in rows if r['role']=='fit']
    if not fit:raise ProjectError('This manifest has no fit images')
    if any(r.get('state') not in {'fire_only','smoke_only','both','empty'} for r in fit):
        raise ProjectError('Unknown original-label state in fit')
    return fit


def group_key(row):
    small=row.get('has_small_fire')
    if small is None:raise ProjectError('Image dimensions are required for original-state x small-fire groups')
    return f"{row['state']}|small_fire={int(small)}"


def parse_advice(raw, task):
    """Technical failures become explicit missing supervision, never new labels."""
    try:
        if isinstance(raw,str):
            fenced=re.fullmatch(r'\s*```(?:json)?\s*\n?(.*?)\s*```\s*',raw,re.DOTALL)
            if fenced:raw=fenced.group(1)
        obj=json.loads(raw) if isinstance(raw,str) else raw
        if task=='A':
            valid=isinstance(obj,dict) and set(obj)==set(SEMANTIC_FIELDS) and all(v in {'yes','no','uncertain'} for v in obj.values())
        elif task=='B':
            valid=isinstance(obj,dict) and set(obj)=={'content','visibility'} and obj['content'] in {'fire','smoke','both','other','uncertain'} and obj['visibility'] in {'clear','limited','unjudgeable'}
        else:raise ProjectError('Unknown teacher task')
        return (obj,'ok') if valid else (None,'invalid_schema')
    except (ValueError,TypeError):return None,'invalid_json'


def selected_regions(row):
    selected=[]
    for cls in (0,1):
        candidates=[b for b in row['boxes'] if b['class_id']==cls and b['bbox'][2]>0 and b['bbox'][3]>0]
        if candidates:
            selected.append(min(candidates,key=lambda b:hashlib.sha256(f"roi-v1|{row['sample_id']}|{b['original_annotation_id']}".encode()).hexdigest()))
    return selected


def teacher_jobs(rows, manifest_sha256):
    jobs=[]
    for row in fit_rows(rows):
        common={'sample_id':row['sample_id'],'image_relpath':row['image_relpath'],
                'manifest_sha256':manifest_sha256,'image_sha256':row.get('image_sha256'),'max_new_tokens':192,'do_sample':False}
        jobs.append({**common,'task':'A','view':'letterbox_640','prompt':SCENE_PROMPT,'prompt_sha256':hashlib.sha256(SCENE_PROMPT.encode()).hexdigest()})
        for box in selected_regions(row):
            # Geometry is preprocessing metadata, not injected as the class name in the prompt.
            jobs.append({**common,'task':'B','view':'context_crop_1.5_neutral_outline',
                         'original_annotation_id':box['original_annotation_id'],'bbox':box['bbox'],'bbox_format':box['format'],
                         'prompt':REGION_PROMPT,'prompt_sha256':hashlib.sha256(REGION_PROMPT.encode()).hexdigest()})
    return {'schema_version':1,'manifest_sha256':manifest_sha256,'scope':'fit_only','jobs':jobs,'runs_teacher':False}


def validate_cache(cache, rows, manifest_sha256,required_tasks=('A','B')):
    if cache.get('scope')=='pilot_fit_only':raise ProjectError('Pilot timing cache cannot replace full-fit supervision')
    if cache.get('teacher_id')!='Qwen/Qwen3.5-0.8B':raise ProjectError('Unexpected teacher model')
    expected={r['sample_id'] for r in fit_rows(rows)}
    if cache.get('manifest_sha256')!=manifest_sha256:raise ProjectError('Teacher cache belongs to a different frozen manifest')
    if set(cache.get('samples',{}))!=expected:raise ProjectError('Cache must explicitly cover every fit sample, including failures, and no dev/cal/test sample')
    for key in ('teacher_id','teacher_revision','prompt_hashes'):
        if not cache.get(key):raise ProjectError(f'Missing cache provenance: {key}')
    from .teacher_protocols import protocol
    spec=protocol(cache.get('protocol','v1'))
    tasks=set(cache.get('tasks',('A','B')))
    if not tasks<={'A','B'} or not set(required_tasks)<=tasks:raise ProjectError('Cache task scope does not support this experiment')
    prompts={task:hashlib.sha256(spec[task].encode()).hexdigest() for task in tasks}
    if cache['prompt_hashes']!=prompts:raise ProjectError('Cache prompt revision differs from the frozen prompts')
    for row in fit_rows(rows):
        item=cache['samples'][row['sample_id']]
        image_hash=item.get('image_sha256','')
        if len(image_hash)!=64 or any(c not in '0123456789abcdef' for c in image_hash):raise ProjectError('Every cache item needs its input image SHA-256')
        if row.get('image_sha256') is not None and row['image_sha256']!=image_hash:raise ProjectError('Teacher image hash differs from the verified original image')
        allowed={b['original_annotation_id'] for b in selected_regions(row)} if 'B' in tasks else set()
        seen=set()
        for region in item.get('regions',[]):
            ident=region['original_annotation_id']
            if ident not in allowed or ident in seen:raise ProjectError('Unexpected/repeated ROI advice; use the predefined ROI selection')
            seen.add(ident)
        if seen!=allowed:raise ProjectError('Every selected ROI needs an explicit response or failure record')
    return cache['samples']


def quality_score(row, item):
    terms=[]
    semantic,_=parse_advice(item.get('semantic'),'A')
    if not row['boxes']:
        if semantic is None:terms.append(1.)
        elif semantic['fire_visible']=='yes' or semantic['smoke_visible']=='yes':terms.append(.75)
        elif all(semantic[k]=='no' for k in ('fire_visible','smoke_visible')):terms.append(1.25)
        else:terms.append(1.)
    regions={r['original_annotation_id']:r for r in item.get('regions',[])}
    for box in selected_regions(row):
        region=regions.get(box['original_annotation_id'],{})
        advice,_=parse_advice(region.get('response'),'B')
        if advice is None or advice['visibility']!='clear' or advice['content']=='uncertain':terms.append(1.)
        else:
            supported=advice['content'] in ({'fire','both'} if box['class_id']==0 else {'smoke','both'})
            terms.append(1.25 if supported else .75)
    return sum(terms)/len(terms) if terms else 1.


def normalize_weights(raw, rows, mode='group'):
    if mode not in {'group','global'}:raise ProjectError('Normalization must be group or global')
    groups=defaultdict(list)
    for row in fit_rows(rows):
        value=float(raw[row['sample_id']])
        if not math.isfinite(value) or value<=0:raise ProjectError('Every fit image must have a finite positive raw score')
        groups[group_key(row) if mode=='group' else 'all'].append(row['sample_id'])
    result={}
    for members in groups.values():
        mean=sum(raw[k] for k in members)/len(members)
        result.update({k:raw[k]/mean for k in members})
    return result


def shuffled_weights(weights, rows, seed=17):
    groups=defaultdict(list);result={}
    for row in fit_rows(rows):groups[group_key(row)].append(row['sample_id'])
    for members in groups.values():
        members.sort();order=sorted(members,key=lambda k:hashlib.sha256(f'shuffle-W|{seed}|{k}'.encode()).hexdigest())
        result.update({dest:weights[src] for dest,src in zip(members,order)})
    return result


def semantic_targets(row,item=None):
    classes={b['class_id'] for b in row['boxes']}
    target=[float(0 in classes),float(1 in classes),0.,0.,0.,0.];mask=[True,True,False,False,False,False]
    advice,_=parse_advice((item or {}).get('semantic'),'A')
    if advice:
        for i,key in enumerate(SEMANTIC_FIELDS[2:],2):
            if advice[key]!='uncertain':target[i]=float(advice[key]=='yes');mask[i]=True
    return {'targets':target,'mask':mask}


def prepare_supervision(rows, manifest_sha256, cache=None, variant='qwen', normalization='group', seed=17, rule_scores=None, shuffle_semantic=False):
    fit=fit_rows(rows)
    needed=('A','B') if variant in ('qwen','shuffle') else ('A',)
    items=validate_cache(cache,rows,manifest_sha256,needed) if cache is not None else {}
    if variant in {'qwen','shuffle'} and not items:raise ProjectError('Qwen weighting needs a validated offline cache, including explicit failures')
    if variant=='rule':
        if not rule_scores or set(rule_scores)!=set(r['sample_id'] for r in fit):raise ProjectError('RuleW needs all fit rule scores')
        raw={k:float(v) for k,v in rule_scores.items()}
    elif variant in {'qwen','shuffle'}:raw={r['sample_id']:quality_score(r,items[r['sample_id']]) for r in fit}
    elif variant=='uniform':raw={r['sample_id']:1. for r in fit}
    else:raise ProjectError('Unknown weighting variant')
    weights=normalize_weights(raw,rows,normalization)
    if variant=='shuffle':weights=shuffled_weights(weights,rows,seed)
    samples={r['sample_id']:{'group':group_key(r),'raw_score':raw[r['sample_id']], 'weight':weights[r['sample_id']],**semantic_targets(r,items.get(r['sample_id']))} for r in fit}
    if shuffle_semantic:
        if not items:raise ProjectError('Semantic shuffle needs a validated cache')
        for field in range(2,6):
            groups=defaultdict(list)
            for row in fit:
                if samples[row['sample_id']]['mask'][field]:groups[row['state']].append(row['sample_id'])
            for members in groups.values():
                members.sort();order=sorted(members,key=lambda k:hashlib.sha256(f'S-shuffle|{seed}|{field}|{k}'.encode()).hexdigest())
                values=[samples[k]['targets'][field] for k in order]
                for ident,value in zip(members,values):samples[ident]['targets'][field]=value
    return {'schema_version':1,'manifest_sha256':manifest_sha256,'variant':variant,'normalization':normalization,
            'teacher_cache_sha256':digest_object(cache) if cache is not None else None,'shuffle_semantic':shuffle_semantic,
            'all_images_retained':True,'original_annotations_changed':False,'samples':samples}


def rule_image_score(path):
    """Predeclared simple baseline; blur is a cue, never a deletion rule."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im=im.convert('L');im.thumbnail((256,256));x=np.asarray(im,dtype=float)/255.
    exposure=float(((x<.02)|(x>.98)).mean())
    detail=float(np.abs(np.diff(x,axis=0)).mean()+np.abs(np.diff(x,axis=1)).mean()) if min(x.shape)>1 else 0.
    return .75+.25*(1-exposure)+.25*min(detail/.1,1.)


def consistent_batch(index, every=4):
    if index<0 or every<1:raise ProjectError('Invalid batch schedule')
    return index % every == 0


def validate_supervision(advice,manifest,factors):
    rows=load_manifest(manifest,roles={'fit'},enrich=False);expected={r['sample_id'] for r in rows}
    if advice.get('manifest_sha256')!=sha256(manifest) or set(advice.get('samples',{}))!=expected:
        raise ProjectError('Supervision must cover every original fit image and bind the exact frozen manifest')
    if advice.get('variant')!=factors['weighting'] or advice.get('normalization')!=factors['normalization']:
        raise ProjectError('Supervision weighting/normalization differs from the experiment control')
    if bool(advice.get('shuffle_semantic'))!=bool(factors.get('shuffle_semantic')):
        raise ProjectError('Semantic shuffle control differs')
    if factors.get('semantic')=='qwen' and not advice.get('teacher_cache_sha256'):
        raise ProjectError('S needs a complete validated teacher cache')
    if factors.get('semantic')=='joint_presence':
        from .teacher_transfer import validate_transfer
        validate_transfer(advice,rows,alpha_expected=factors['teacher_joint_mix'])
    elif advice.get('student_protocol')=='joint_presence_transfer_v1':
        raise ProjectError('Teacher targets cannot enter the no-teacher control')
    for item in advice['samples'].values():
        if not math.isfinite(item['weight']) or item['weight']<=0:raise ProjectError('All images require positive finite weights')
        if len(item['targets'])!=6 or len(item['mask'])!=6:raise ProjectError('Six semantic fields required')
    if factors['weighting'] in ('qwen','shuffle') and all(abs(x['weight']-1.)<1e-12 for x in advice['samples'].values()):
        raise ProjectError('W has no effective intervention: reuse the corresponding unweighted result rather than train a duplicate')
    if factors.get('semantic')=='qwen' and not any(any(x['mask'][2:]) for x in advice['samples'].values()):
        raise ProjectError('S has no known context targets: it would duplicate the GT-only P control')
    return advice
