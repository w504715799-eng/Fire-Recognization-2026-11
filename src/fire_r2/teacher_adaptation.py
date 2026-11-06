"""Traceable instruction views from source fit labels; no source edits or invented context GT."""
import hashlib,json
from collections import Counter
from .common import ProjectError
from .supervision import selected_regions

SCENE_TASK=('Identify the fire and smoke target categories in this image. Return one JSON object '
            'with fire_present and smoke_present. Each value is the string yes or no. Give a direct answer.')
REGION_TASK=('Classify the indicated target region as fire or smoke. Return one JSON object with '
             'target_class, whose value is fire or smoke. Give a direct answer.')
PARTITIONS=('teacher_holdout','teacher_dev','teacher_train')


def partition_fit(rows,excluded_ids,counts=None,salt='qwen3vl-task-adaptation-v1-20261106'):
    counts=counts or {'teacher_holdout':64,'teacher_dev':32,'teacher_train':250}
    if set(counts)!=set(PARTITIONS) or any(not isinstance(n,int) or n<1 for n in counts.values()):
        raise ProjectError('Specify positive per-state counts for train/dev/holdout')
    result={key:[] for key in PARTITIONS};result['development_regression']=[]
    fit=[r for r in rows if r['role']=='fit']
    if not set(excluded_ids)<={r['sample_id'] for r in fit}:raise ProjectError('Development exclusions must belong to source fit')
    result['development_regression']=[r for r in fit if r['sample_id'] in excluded_ids]
    for state in ('fire_only','smoke_only','both','empty'):
        pool=[r for r in fit if r['state']==state and r['sample_id'] not in excluded_ids]
        pool.sort(key=lambda r:hashlib.sha256((salt+'|'+r['sample_id']).encode()).hexdigest())
        if len(pool)<sum(counts.values()):raise ProjectError('Insufficient source-fit images; do not reuse holdout')
        offset=0
        for part in PARTITIONS:
            result[part].extend(pool[offset:offset+counts[part]]);offset+=counts[part]
    all_ids=[r['sample_id'] for rs in result.values() for r in rs]
    if len(all_ids)!=len(set(all_ids)):raise ProjectError('Teacher partitions overlap')
    return result


def instruction_records(row,partition,manifest_hash):
    if row['role']!='fit':raise ProjectError('Teacher adaptation uses source fit only')
    common={'sample_id':row['sample_id'],'image_path':row['image_relpath'],'image_sha256':row.get('image_sha256'),
            'teacher_partition':partition,'detector_role':'fit','source_manifest_sha256':manifest_hash,
            'supervision_basis':'original annotation categories, not independent physical/quality truth',
            'context_fields_supervised':False}
    classes={b['class_id'] for b in row['boxes']}
    target={'fire_present':'yes' if 0 in classes else 'no','smoke_present':'yes' if 1 in classes else 'no'}
    records=[{**common,'task':'scene_presence','view':'letterbox_640','instruction':SCENE_TASK,
              'target':target,'supervised_fields':list(target),'original_annotation_id':None}]
    for box in selected_regions(row):
        target={'target_class':'fire' if box['class_id']==0 else 'smoke'}
        records.append({**common,'task':'region_class','view':'context_crop_1.5_thin_outline_after_resize',
            'instruction':REGION_TASK,'target':target,'supervised_fields':['target_class'],
            'original_annotation_id':box['original_annotation_id'],'source_class_id':box.get('original_class_id'),
            'canonical_class_id':box['class_id'],'bbox':box['bbox'],'bbox_format':box['format']})
    for r in records:
        r['target_json']=json.dumps(r['target'],separators=(',',':'))
        r['instruction_sha256']=hashlib.sha256(r['instruction'].encode()).hexdigest()
    return records


def supervised_value_spans(target):
    """Character ranges for semantic values only, to mask format/prompt/image tokens in SFT."""
    text=json.dumps(target,separators=(',',':'));spans=[];offset=0
    for key,value in target.items():
        needle=json.dumps(key)+':'+json.dumps(value)
        start=text.index(needle,offset)+len(json.dumps(key))+2
        spans.append((start,start+len(value)));offset=start+len(value)
    return text,spans


def semantic_token_labels(plain_ids,offsets,expanded_ids,value_spans):
    """Align semantic values after processor image-token expansion without supervising formatting."""
    chosen=[i for i,(left,right) in enumerate(offsets)
            if right>left and any(left<end and right>start for start,end in value_spans)]
    if not chosen:raise ProjectError('No semantic target tokens')
    suffix=plain_ids[min(chosen):]
    if len(suffix)>len(expanded_ids) or expanded_ids[-len(suffix):]!=suffix:
        raise ProjectError('Processor changed the assistant suffix; do not guess token-label alignment')
    labels=[-100]*len(expanded_ids);shift=len(expanded_ids)-len(plain_ids)
    for i in chosen:
        position=i+shift
        if position<0 or expanded_ids[position]!=plain_ids[i]:raise ProjectError('Semantic token alignment failed')
        labels[position]=expanded_ids[position]
    return labels


def instruction_inputs(processor,record,max_length=768):
    """Real source pixels and teacher-forced values, with next-token positions explicit."""
    from pathlib import Path
    from .common import sha256
    from .teacher_runner import teacher_view
    if record['detector_role']!='fit':raise ProjectError('Teacher instructions must originate in fit')
    if sha256(Path(record['image_path']))!=record['image_sha256']:raise ProjectError('Instruction source image changed')
    box={'bbox':record['bbox'],'format':record['bbox_format']} if record['task']=='region_class' else None
    view=teacher_view(record['image_path'],box,region_marker='thin_outline_after_resize')
    message=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':record['instruction']}]}]
    prompt=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    target,spans=supervised_value_spans(record['target']);text=prompt+target+processor.tokenizer.eos_token
    plain=processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    inputs=processor(text=[text],images=[view],return_tensors='pt',add_special_tokens=False)
    ids=inputs['input_ids'][0].tolist()
    labels=semantic_token_labels(plain['input_ids'],plain['offset_mapping'],ids,[(len(prompt)+a,len(prompt)+b) for a,b in spans])
    positions=[i for i,x in enumerate(labels) if x!=-100]
    if len(ids)>max_length or not positions or min(positions)<1:raise ProjectError('Invalid sequence/semantic loss positions')
    return inputs,[i-1 for i in positions],[labels[i] for i in positions]
