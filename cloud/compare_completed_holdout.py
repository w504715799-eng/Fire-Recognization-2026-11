"""Complete matched CPU short-task comparison, preserving all four original states."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256


def main():
    base=ROOT/'artifacts/teacher-heldout-20261106';out=base/'complete-comparison.json'
    if out.exists():raise ProjectError('Keep existing completed comparison')
    paths={'0.8b':base/'base-0.8b/report.json','2b':base/'base-2b-parallel/report.json'}
    reports={k:read_json(p) for k,p in paths.items()};maps={}
    for name,r in reports.items():
        if r['status']!='completed' or len(r['responses'])!=512:raise ProjectError('Both teachers must complete all 512 queries')
        m={(x['sample_id'],x['task'],x['annotation_id']):x for x in r['responses']}
        if len(m)!=512:raise ProjectError('Duplicate queries')
        maps[name]=m
    if set(maps['0.8b'])!=set(maps['2b']) or reports['0.8b']['instruction_sha256']!=reports['2b']['instruction_sha256']:raise ProjectError('Heldout query sets/manifests differ')
    states={}
    for x in reports['0.8b']['responses']:
        if x['task']=='scene_presence':
            t=x['target_from_original_annotation'];f=t['fire_present']=='yes';s=t['smoke_present']=='yes'
            states[x['sample_id']]='both' if f and s else 'fire_only' if f else 'smoke_only' if s else 'empty'
    result={'scope':'256-image teacher-internal source-fit holdout; not detector test, human semantic truth or external generalization',
            'models':{},'paired':{},'instruction_sha256':reports['0.8b']['instruction_sha256']}
    for name,r in reports.items():
        model={'identity':r['identity'],'source_report_sha256':sha256(paths[name]),'tasks':r['tasks'],'states':{},
               'seconds':r['seconds'],'parallel_remainder':r.get('resumed_as_two_disjoint_workers',False),'confusion':{},
               'valid':sum(x['valid'] for x in r['responses'])}
        for state in ('fire_only','smoke_only','both','empty'):
            subset=[x for x in r['responses'] if states[x['sample_id']]==state]
            model['states'][state]={task:{'queries':sum(x['task']==task for x in subset),
                'exact_agree':sum(x['task']==task and x['agrees_original_annotation'] for x in subset)} for task in ('scene_presence','region_class')}
        for field in ('fire_present','smoke_present','target_class'):
            c={}
            for x in r['responses']:
                if field not in x['target_from_original_annotation']:continue
                gt=x['target_from_original_annotation'][field];answer=x['answer'].get(field,'invalid') if isinstance(x['answer'],dict) else 'invalid'
                key=gt+' -> '+answer;c[key]=c.get(key,0)+1
            model['confusion'][field]=c
        result['models'][name]=model
    for task in ('scene_presence','region_class'):
        p={'both_correct':0,'only_0_8b_correct':0,'only_2b_correct':0,'both_wrong':0}
        for key,left in maps['0.8b'].items():
            right=maps['2b'][key]
            for field in ('view_sha256','prompt_sha256','target_from_original_annotation'):
                if left[field]!=right[field]:raise ProjectError('Paired heldout inputs/targets differ')
            if key[1]!=task:continue
            a,b=left['agrees_original_annotation'],right['agrees_original_annotation']
            p['both_correct' if a and b else 'only_0_8b_correct' if a else 'only_2b_correct' if b else 'both_wrong']+=1
        result['paired'][task]=p
    write_json(out,result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
