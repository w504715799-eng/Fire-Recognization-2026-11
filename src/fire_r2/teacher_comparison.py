"""Compare bound teacher queries without conflating syntax, semantics, devices or timing."""
import statistics
from .common import ProjectError
from .supervision import parse_advice


def query_records(cache,ids=None,syntax_only_normalization=False):
    records={}
    for ident,item in cache['samples'].items():
        if ids is not None and ident not in ids:continue
        for call in item['calls']:
            key=(ident,call['task'],call.get('original_annotation_id'))
            if key in records:raise ProjectError('Duplicate comparison query')
            answer=parse_advice(call['raw_response'],call['task'])[0]
            if syntax_only_normalization:
                from .teacher_audit import normalize_syntax
                answer=normalize_syntax(call['raw_response'],call['task'])[0]
            records[key]={'call':call,'image_sha256':item['image_sha256'],'answer':answer}
    return records


def query_cost(cache,ids=None):
    records=list(query_records(cache,ids).values());calls=[r['call'] for r in records]
    total=sum(c['seconds'] for c in calls)
    return {'queries':len(calls),'images':len({r[0] for r in query_records(cache,ids)}),
            'query_wall_seconds':total,'queries_per_second':len(calls)/total if total else None,
            'output_tokens':sum(c.get('output_tokens') or 0 for c in calls),
            'schema_pass':sum(r['answer'] is not None for r in records),
            'by_task':{task:{'queries':sum(c['task']==task for c in calls),
                'median_amortized_query_seconds':statistics.median([c['seconds'] for c in calls if c['task']==task])
                    if any(c['task']==task for c in calls) else None} for task in ('A','B')},
            'timing_scope':'sum of query wall times; batch times counted through per-query amortization; excludes model loading'}


def paired_answers(left,right,ids=None,allow_protocol_change=False,syntax_only_normalization=False):
    a=query_records(left,ids,syntax_only_normalization);b=query_records(right,ids,syntax_only_normalization)
    if set(a)!=set(b):raise ProjectError('Paired comparison requires the same image/ROI query keys')
    if left['teacher_revision']!=right['teacher_revision']:raise ProjectError('Teacher revision differs')
    if not allow_protocol_change and left['prompt_hashes']!=right['prompt_hashes']:
        raise ProjectError('Prompt mismatch in same-protocol comparison')
    exact=valid_pairs=0;fields={};differences=[]
    for key in a:
        if a[key]['image_sha256']!=b[key]['image_sha256']:raise ProjectError('Paired source image bytes differ')
        la,ra=a[key]['answer'],b[key]['answer']
        if la is not None and ra is not None:
            valid_pairs+=1;exact+=la==ra
            for field in la:
                name=key[1]+'.'+field;f=fields.setdefault(name,{'known_schema_pairs':0,'equal':0})
                f['known_schema_pairs']+=1;f['equal']+=la[field]==ra[field]
        if la is None or ra is None or la!=ra:
            differences.append({'sample_id':key[0],'task':key[1],'original_annotation_id':key[2],
                                'left':la,'right':ra,'both_schema_valid':la is not None and ra is not None})
    cost_a=query_cost(left,ids);cost_b=query_cost(right,ids)
    return {'queries':len(a),'both_schema_valid':valid_pairs,'exact_valid_answers':exact,
            'fields':fields,'differences':differences,'left_cost':cost_a,'right_cost':cost_b,
            'right_over_left_time':cost_b['query_wall_seconds']/cost_a['query_wall_seconds'] if cost_a['query_wall_seconds'] else None,
            'protocol_change_allowed':allow_protocol_change,
            'syntax_only_normalized':syntax_only_normalization,
            'interpretation':'Agreement is not semantic correctness; speed comparison must retain hardware, precision, batch and cohort labels.'}


def compare_teacher_models(left,right,ids=None):
    """Explicit cross-model regression, not a within-model precision or capacity ablation."""
    identities=[(c.get('teacher_id'),c.get('teacher_revision')) for c in (left,right)]
    if any(not all(x) for x in identities) or identities[0]==identities[1]:
        raise ProjectError('Cross-model comparison needs two distinct explicit model identities')
    if left.get('manifest_sha256')!=right.get('manifest_sha256') or left['prompt_hashes']!=right['prompt_hashes']:
        raise ProjectError('Cross-model regression requires matching source manifest and prompts')
    a=query_records(left,ids);b=query_records(right,ids)
    if set(a)!=set(b):raise ProjectError('Cross-model regression requires matching image/ROI queries')
    changes=[];equal=valid=0
    for key in a:
        lc,rc=a[key]['call'],b[key]['call']
        if a[key]['image_sha256']!=b[key]['image_sha256'] or not lc.get('view_sha256') or lc['view_sha256']!=rc.get('view_sha256'):
            raise ProjectError('Cross-model regression requires identical source and rendered view bytes')
        la,ra=a[key]['answer'],b[key]['answer']
        valid+=la is not None and ra is not None
        equal+=la is not None and ra is not None and la==ra
        if la!=ra or la is None:
            changes.append({'sample_id':key[0],'task':key[1],'original_annotation_id':key[2],'left':la,'right':ra})
    return {'left_model':identities[0],'right_model':identities[1],'queries':len(a),
            'both_schema_valid':valid,'identical_answers':equal,'differences':changes,
            'left_cost':query_cost(left,ids),'right_cost':query_cost(right,ids),
            'left_runtime':left.get('runtime'),'right_runtime':right.get('runtime'),
            'interpretation':'Different model families, processor and chat template; not an isolated parameter-count effect. Agreement is not correctness.'}
