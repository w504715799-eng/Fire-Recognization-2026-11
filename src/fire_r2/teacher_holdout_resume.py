"""Validate a terminal same-protocol CPU holdout prefix before reusing its answers."""
import json
from .common import ProjectError


def resume_answers(prior,current,identity,examples,allow_interrupted=False):
    error=prior.get('error','').lower()
    controlled=allow_interrupted and not error and 'KeyboardInterrupt' in prior.get('traceback','')
    if prior.get('status')!='failed' or not ('timed out' in error or 'stage budget' in error or controlled):
        raise ProjectError('Only an explicit time-limited prior segment may be continued')
    for key in ('device','dtype','threads','batch','instruction_sha256'):
        if prior.get(key)!=current.get(key):raise ProjectError('Resume binding differs: '+key)
    if prior.get('identity')!=identity:raise ProjectError('Resume model identity differs')
    expected={(r['sample_id'],r['task'],r['original_annotation_id']):r for r in examples};done={}
    for item in prior['responses']:
        key=(item['sample_id'],item['task'],item['annotation_id'])
        if key in done or key not in expected:raise ProjectError('Duplicate/unexpected resumed query')
        row=expected[key]
        if item['prompt_sha256']!=row['instruction_sha256'] or item['target_from_original_annotation']!=row['target']:
            raise ProjectError('Resumed prompt/target differs')
        if json.loads(item['raw_response'])!=item['answer']:raise ProjectError('Stored answer differs from raw output')
        done[key]=item
    for task in ('scene_presence','region_class'):
        rows=[r for r in examples if r['task']==task]
        for i in range(0,len(rows),current['batch']):
            keys=[(r['sample_id'],r['task'],r['original_annotation_id']) for r in rows[i:i+current['batch']]]
            if any(k in done for k in keys) and not all(k in done for k in keys):
                raise ProjectError('Do not alter batch composition when continuing')
    return done


def owns_batch(start,batch_size,shard_index,shard_count):
    if shard_count not in (1,2) or not 0<=shard_index<shard_count or start%batch_size:
        raise ProjectError('Invalid bounded whole-batch shard')
    return (start//batch_size)%shard_count==shard_index


def merge_remainders(prior,reports,expected_rows):
    expected={(r['sample_id'],r['task'],r['original_annotation_id']) for r in expected_rows};merged={}
    for report in reports:
        if report['status']!='completed_shard' or report['identity']!=prior['identity'] or report['instruction_sha256']!=prior['instruction_sha256']:
            raise ProjectError('Shard identity/status mismatch')
        for item in report['responses']:
            key=(item['sample_id'],item['task'],item['annotation_id'])
            if key in merged and merged[key]!=item:raise ProjectError('Overlapping answers differ; do not select one')
            merged[key]=item
    if set(merged)!=expected:raise ProjectError('Merged query set is incomplete or contains extra queries')
    return [merged[(r['sample_id'],r['task'],r['original_annotation_id'])] for r in expected_rows]
