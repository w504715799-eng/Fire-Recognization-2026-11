"""CPU diagnostics only: immutable raw responses, syntax repair separated from semantics."""
from collections import Counter,defaultdict
import copy,hashlib,json,math,re,statistics

from .common import ProjectError
from .supervision import (SEMANTIC_FIELDS,parse_advice,selected_regions,quality_score,
                          normalize_weights,group_key,semantic_targets,digest_object)


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate response key')
        result[key]=value
    return result


def normalize_syntax(raw,task):
    """No GT argument: only fences, exact bare A enums, singleton-object arrays.

    Audit output is never substituted into a training cache. No invalid category
    (including marker '1') is mapped to a valid answer, and no field is supplied.
    """
    if not isinstance(raw,str):return None,['non_string_response']
    text=raw.strip();changes=[]
    fenced=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\s*```',text,re.DOTALL)
    if fenced:text=fenced.group(1);changes.append('remove_code_fence')
    if task=='A':
        keys='|'.join(re.escape(k) for k in SEMANTIC_FIELDS)
        pattern=r'("(?:'+keys+r')"\s*:\s*)(yes|no|uncertain)(?=\s*[,}])'
        text,n=re.subn(pattern,lambda m:m[1]+json.dumps(m[2]),text)
        if n:changes.append('quote_exact_bare_enums')
    try:obj=json.loads(text,object_pairs_hook=_unique_object)
    except (ValueError,TypeError):return None,changes+['invalid_json']
    if isinstance(obj,list) and len(obj)==1 and isinstance(obj[0],dict):
        obj=obj[0];changes.append('unwrap_singleton_object_array')
    answer,status=parse_advice(obj,task)
    return answer,changes+([] if status=='ok' else ['invalid_schema'])


def _stats(calls,task):
    rows=[c for c in calls if c['task']==task]
    normalized=[normalize_syntax(c['raw_response'],task)[0] for c in rows]
    fields=SEMANTIC_FIELDS if task=='A' else ('content','visibility')
    return {'calls':len(rows),'recorded_schema_pass':sum(c.get('parse_status')=='ok' for c in rows),
            'answer_distributions_after_syntax_only':{key:dict(Counter(x[key] if x is not None else 'invalid' for x in normalized)) for key in fields},
            'strict_reparse_pass':sum(parse_advice(c['raw_response'],task)[1]=='ok' for c in rows),
            'syntax_normalized_pass':sum(x is not None for x in normalized),
            'output_tokens':sum(c.get('output_tokens') or 0 for c in rows),
            'median_query_seconds':statistics.median(c['seconds'] for c in rows) if rows else None,
            'batch_sizes':dict(Counter(str(c.get('batch_size',1)) for c in rows))}


def audit_cache(cache,rows,original_eight_ids):
    """Compare to original-label presence/support, not independent semantic truth."""
    before=digest_object(cache);by_id={r['sample_id']:r for r in rows};samples=cache['samples']
    if any(k not in by_id or by_id[k]['role']!='fit' for k in samples):raise ProjectError('Unexpected/non-fit audit image')
    per_image=[];calls=[]
    for ident,item in samples.items():
        row=by_id[ident];allowed={b['original_annotation_id']:b for b in selected_regions(row)}
        expected=[('A',None)]+[('B',key) for key in allowed]
        actual=[(c['task'],c.get('original_annotation_id')) for c in item['calls']]
        if Counter(actual)!=Counter(expected):raise ProjectError('Missing, duplicated or mismatched image/ROI calls')
        record={'sample_id':ident,'state':row['state'],'cohort':'original_8' if ident in original_eight_ids else 'new_fit',
                'presence':{},'regions':[],'raw_score':quality_score(row,item),
                'semantic_known_fields':sum(semantic_targets(row,item)['mask'][2:])}
        for call in item['calls']:
            normalized,changes=normalize_syntax(call['raw_response'],call['task'])
            calls.append(call)
            if call['task']=='A':
                for cls,field in enumerate(SEMANTIC_FIELDS[:2]):
                    prediction=normalized[field] if normalized else None
                    gt=any(b['class_id']==cls for b in row['boxes'])
                    record['presence'][field]={'original_label_present':gt,'answer_after_syntax_only':prediction,
                        'known':prediction in ('yes','no'),
                        'agreement':(prediction=='yes')==gt if prediction in ('yes','no') else None}
            else:
                box=allowed[call['original_annotation_id']];content=normalized['content'] if normalized else None
                supported=content in ({'fire','both'} if box['class_id']==0 else {'smoke','both'})
                record['regions'].append({'original_annotation_id':box['original_annotation_id'],
                    'canonical_class_id':box['class_id'],'source_class_id':box.get('original_class_id'),'content_after_syntax_only':content,
                    'known':content in ('fire','smoke','both','other'),
                    'original_class_supported':supported if content in ('fire','smoke','both','other') else None,
                    'visibility':normalized['visibility'] if normalized else None,
                    'normalizations':changes})
        per_image.append(record)
    def summarize(items):
        result={'images':len(items),'presence':{},'region_class_support':{}}
        for field in SEMANTIC_FIELDS[:2]:
            p=[r['presence'][field] for r in items];known=[x for x in p if x['known']]
            result['presence'][field]={'denominator':len(p),'known':len(known),'agree':sum(x['agreement'] for x in known),
                'agreement_among_known':sum(x['agreement'] for x in known)/len(known) if known else None}
        for cls,name in enumerate(('fire','smoke')):
            p=[x for r in items for x in r['regions'] if x['canonical_class_id']==cls];known=[x for x in p if x['known']]
            result['region_class_support'][name]={'denominator':len(p),'known':len(known),
                'supported':sum(x['original_class_supported'] for x in known),
                'content_counts':dict(Counter(str(x['content_after_syntax_only']) for x in p))}
        result['known_context_fields_from_original_cache']=sum(x['semantic_known_fields'] for x in items)
        return result
    fit=[by_id[k] for k in samples];raw={x['sample_id']:x['raw_score'] for x in per_image}
    weights=normalize_weights(raw,fit);groups=defaultdict(list)
    for row in fit:groups[group_key(row)].append(weights[row['sample_id']])
    grouped={k:{'images':len(v),'weight_sum':sum(v),'sum_error':abs(sum(v)-len(v))} for k,v in groups.items()}
    assert all(v>0 and math.isfinite(v) for v in weights.values())
    assert all(x['sum_error']<1e-10 for x in grouped.values())
    assert before==digest_object(cache)
    return {'protocol':cache.get('protocol','v1'),'cache_object_sha256':before,'raw_cache_unchanged':True,
        'teacher_revision':cache.get('teacher_revision'),'prompt_hashes':cache.get('prompt_hashes'),
        'runtime':cache.get('runtime'),'generation':cache.get('generation'),
        'format':{t:_stats(calls,t) for t in ('A','B')},'overall':summarize(per_image),
        'cohorts':{key:summarize([r for r in per_image if r['cohort']==key]) for key in ('original_8','new_fit')},
        'by_original_state':{key:summarize([r for r in per_image if r['state']==key]) for key in ('fire_only','smoke_only','both','empty')},
        'diagnostic_W':{'scope':'hypothetical_pilot_group_normalization_not_training_supervision',
                        'positive_weights':True,'groups':grouped},'per_image':per_image,
        'semantic_accuracy':'unmeasured_until_independent_review',
        'interpretation':'Original-label agreement/support is diagnostic; it is not independent visual truth, bbox quality, or student gain.'}


def compare_caches(caches,rows,original_eight_ids):
    reports={name:audit_cache(cache,rows,original_eight_ids) for name,cache in caches.items()}
    paired={}
    if 'v2_serial' in caches and 'v2_batch4' in caches:
        a=caches['v2_serial'];b=caches['v2_batch4']
        if set(a['samples'])!=set(b['samples']):raise ProjectError('Serial/batched selection differs')
        if a['prompt_hashes']!=b['prompt_hashes'] or a['teacher_revision']!=b['teacher_revision']:
            raise ProjectError('Serial/batched teacher/prompt differs')
        if a.get('protocol','v1')!=b.get('protocol','v1') or a.get('generation')!=b.get('generation'):
            raise ProjectError('Serial/batched protocol or generation settings differ')
        agree=0;total=0
        for ident,item in a['samples'].items():
            other=b['samples'][ident]
            if item['image_sha256']!=other['image_sha256']:raise ProjectError('Serial/batched pixels differ')
            left={(c['task'],c.get('original_annotation_id')):parse_advice(c['raw_response'],c['task'])[0] for c in item['calls']}
            right={(c['task'],c.get('original_annotation_id')):parse_advice(c['raw_response'],c['task'])[0] for c in other['calls']}
            if set(left)!=set(right):raise ProjectError('Serial/batched ROI mapping differs')
            left_views={(c['task'],c.get('original_annotation_id')):c.get('view') for c in item['calls']}
            right_views={(c['task'],c.get('original_annotation_id')):c.get('view') for c in other['calls']}
            if left_views!=right_views:raise ProjectError('Serial/batched input rendering differs')
            for key in left:
                total+=1;agree+=left[key] is not None and left[key]==right[key]
        paired={'calls':total,'identical_valid_answers':agree,'semantic_correctness_not_established':True,
                'same_reported_gpu':a.get('runtime',{}).get('gpu')==b.get('runtime',{}).get('gpu')}
    return {'reports':reports,'serial_batch_comparison':paired,
            'pending':[name for name in ('v2_serial','v2_batch4','v2_expanded') if name not in caches]}


def human_review_template(rows,selected_ids):
    """Blind review starts from pixels/regions; teacher answers are kept in a separate report."""
    records=[]
    for row in rows:
        if row['sample_id'] not in selected_ids:continue
        for task,box in [('A',None)]+[('B',b) for b in selected_regions(row)]:
            ident=row['sample_id']+'|'+task+'|'+str(box['original_annotation_id'] if box else None)
            records.append({'review_id':hashlib.sha256(ident.encode()).hexdigest()[:16],
                'sample_id':row['sample_id'],'task':task,'original_annotation_id':box['original_annotation_id'] if box else None,
                'bbox':box['bbox'] if box else None,'bbox_format':box['format'] if box else None,
                'reviewer_id':None,'reviewed_at':None,'visible_fire':None,'visible_smoke':None,
                'cloud_fog_like':None,'bright_light_like':None,'reflection_like':None,'low_visibility':None,
                'content':None,'visibility':None,'notes':None,'status':'pending_independent_review'})
    return {'independent_review_completed':False,'blind_to_teacher_answers':True,'records':records}


def cpu_diagnostic_cache(report):
    """Adapt completed CPU results for audits only; never mix into frozen GPU supervision."""
    if report.get('status')!='completed':raise ProjectError('Report partial CPU progress separately; do not silently drop unfinished queries')
    if report.get('execution_device')!='cpu' or report.get('cuda_initialized') is not False:
        raise ProjectError('Expected verified CPU-only results')
    samples={};prompts={};seen=set()
    for call in report['calls']:
        ident=call['sample_id'];key=(ident,call['task'],call.get('original_annotation_id'))
        if key in seen:raise ProjectError('Duplicate CPU query')
        seen.add(key)
        if call['task'] in prompts and prompts[call['task']]!=call['prompt_sha256']:
            raise ProjectError('Mixed CPU prompts')
        prompts[call['task']]=call['prompt_sha256']
        item=samples.setdefault(ident,{'image_sha256':call['image_sha256'],'semantic':None,'regions':[],'calls':[]})
        if item['image_sha256']!=call['image_sha256']:raise ProjectError('Mixed input bytes for CPU sample')
        answer,status=parse_advice(call['raw_response'],call['task'])
        if answer!=call['parsed'] or status!=call['parse_status']:raise ProjectError('CPU parsed result disagrees with raw response')
        item['calls'].append({**call,'batch_size':1,'view_sha256':call['view_sha256'],'execution_device':'cpu','dtype':report['dtype']})
        if call['task']=='A':item['semantic']=answer
        else:item['regions'].append({'original_annotation_id':call['original_annotation_id'],'response':answer})
    if set(samples)!=set(report['samples']):raise ProjectError('Incomplete CPU image coverage')
    return {'schema_version':1,'scope':'pilot_fit_only','teacher_id':'Qwen/Qwen3.5-0.8B',
        'teacher_revision':report['teacher_revision'],'protocol':report['protocol'],
        'manifest_sha256':report['source_manifest_sha256'],'prompt_hashes':prompts,'samples':samples,
        'runtime':{'execution_device':'cpu','dtype':report['dtype'],'threads':report['threads'],
            'current_segment_seconds':report['total_seconds'],'reused_calls':report.get('reused_calls',0),
            'total_inference_query_seconds':sum(c['seconds'] for c in report['calls']),
            'peak_rss_kib':report['peak_rss_kib']},'not_gpu_throughput':True}
