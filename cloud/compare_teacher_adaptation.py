"""Full matched GPU base/adapter results, including per-state errors and v2 regressions."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--adapted',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    paths={k:output_path(ROOT,v) for k,v in [('base',a.base),('adapted',a.adapted)]};reports={k:read_json(p) for k,p in paths.items()}
    key=lambda r:(r['cohort'],r['sample_id'],r['task'],r['annotation_id'])
    maps={k:{key(x):x for x in r['responses']} for k,r in reports.items()}
    if any(r['status']!='completed' or len(r['responses'])!=640 for r in reports.values()) or any(len(m)!=640 for m in maps.values()):raise ProjectError('Require both complete 640-query GPU evaluations')
    if set(maps['base'])!=set(maps['adapted']):raise ProjectError('Evaluation query keys differ')
    for f in ('identity','device','dtype','batch','teacher_holdout_sha256','development_regression_sha256'):
        if reports['base'][f]!=reports['adapted'][f]:raise ProjectError('Evaluation protocol differs: '+f)
    for k in maps['base']:
        for f in ('view_sha256','prompt_sha256'):
            if maps['base'][k][f]!=maps['adapted'][k][f]:raise ProjectError('Paired input differs')
    data=ROOT/'artifacts/qwen3vl-2b-preparation-20261106/instructions';targets={};states={}
    for cohort,part in [('heldout','teacher_holdout'),('v2_regression','development_regression')]:
        path=data/(part+'.instructions.jsonl')
        if sha256(path)!=reports['base'][part+'_sha256']:raise ProjectError('Target metadata changed')
        for line in path.open():
            r=json.loads(line);k=(cohort,r['sample_id'],r['task'],r['original_annotation_id']);targets[k]=r['target']
            if r['task']=='scene_presence':
                f=r['target']['fire_present']=='yes';s=r['target']['smoke_present']=='yes';states[(cohort,r['sample_id'])]='both' if f and s else 'fire_only' if f else 'smoke_only' if s else 'empty'
    if set(targets)!=set(maps['base']):raise ProjectError('Target/evaluation query set differs')
    result={'scope':'complete matched GPU evaluation, original annotation task alignment; not independent human truth or detector AP',
            'source_report_sha256':{k:sha256(p) for k,p in paths.items()},'models':{},'paired':{},'context_truth_available':False}
    for name,r in reports.items():
        out={'heldout_tasks':r['heldout_tasks'],'confusion':{},'by_state':{},'v2_original_label_diagnostics':{},
             'schema_valid':sum(x['valid'] for x in r['responses']),'seconds':r['seconds'],'peak_gpu_bytes':r['peak_gpu_bytes']}
        for task in ('scene_presence','region_class'):
            for state in ('fire_only','smoke_only','both','empty'):
                items=[x for x in r['responses'] if x['cohort']=='heldout' and x['task']==task and states[('heldout',x['sample_id'])]==state]
                out['by_state'].setdefault(state,{})[task]={'queries':len(items),'agree':sum(x['agrees_original_annotation'] for x in items)}
        for field in ('fire_present','smoke_present','target_class'):
            counts={}
            for k,x in maps[name].items():
                if k[0]!='heldout' or field not in targets[k]:continue
                gt=targets[k][field];answer=x['answer'].get(field,'invalid') if isinstance(x['answer'],dict) else 'invalid';pair=gt+' -> '+answer;counts[pair]=counts.get(pair,0)+1
            out['confusion'][field]=counts
        for field,targetfield in [('fire_visible','fire_present'),('smoke_visible','smoke_present')]:
            items=[(k,x) for k,x in maps[name].items() if k[0]=='v2_regression' and k[2]=='scene_presence']
            out['v2_original_label_diagnostics'][field]={'queries':len(items),'agree':sum(x['answer'].get(field)==targets[k][targetfield] for k,x in items if isinstance(x['answer'],dict))}
        for cls in ('fire','smoke'):
            items=[(k,x) for k,x in maps[name].items() if k[0]=='v2_regression' and k[2]=='region_class' and targets[k]['target_class']==cls]
            out['v2_original_label_diagnostics'][cls+'_region']={'queries':len(items),'class_supported_including_both':sum(x['answer'].get('content') in (cls,'both') for k,x in items if isinstance(x['answer'],dict)),'exact_class_only':sum(x['answer'].get('content')==cls for k,x in items if isinstance(x['answer'],dict))}
        result['models'][name]=out
    for task in ('scene_presence','region_class'):
        p={'both_correct':0,'only_base_correct':0,'only_adapted_correct':0,'both_wrong':0}
        for k,x in maps['base'].items():
            if k[0]!='heldout' or k[2]!=task:continue
            left=x['agrees_original_annotation'];right=maps['adapted'][k]['agrees_original_annotation'];p['both_correct' if left and right else 'only_base_correct' if left else 'only_adapted_correct' if right else 'both_wrong']+=1
        result['paired'][task]=p
    result['v2_context_changes']={}
    for field in ('cloud_fog_like','bright_light_like','reflection_like','low_visibility'):
        pairs=[(x['answer'],maps['adapted'][k]['answer']) for k,x in maps['base'].items() if k[0]=='v2_regression' and k[2]=='scene_presence']
        result['v2_context_changes'][field]={'queries':len(pairs),'changed':sum(x.get(field)!=y.get(field) for x,y in pairs if isinstance(x,dict) and isinstance(y,dict)),'correctness_unmeasured':True}
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Keep prior comparisons')
    write_json(out,result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
