"""One explicitly documented native-task compatibility check; original six-choice evidence stays intact."""
from pathlib import Path
import argparse,datetime,json,math,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,sha256,ProjectError
from fire_r2.runtime import cloud_gpu
from fire_r2.teacher_models import checked_processor
from fire_r2.teacher_adaptation import SCENE_TASK
from fire_r2.teacher_transfer import score_presence_views
from cloud.region_diagnostic import region_view,metrics,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    if sys.platform!='linux' or ROOT!=Path('/root/rivermind-data/FIRE_R2'):raise ProjectError('Cloud only')
    cfg=read_json(ROOT/'configs/region_diagnostic_20261110.json');parent=ROOT/cfg['output'];out=parent/'native-compatibility'
    if out.exists():raise ProjectError('Keep previous attempt')
    out.mkdir();prior=read_json(parent/'inference.json');prep=read_json(parent/'preparation.json')
    if prior['status']!='completed' or prior['seconds']>=600:raise ProjectError('Insufficient cumulative diagnostic budget')
    if sha256(parent/'regions.jsonl')!=prep['regions_sha256']:raise ProjectError('Same frozen region selection required')
    if read_json(ROOT/cfg['wait_for_queue'])['status']!='completed':raise ProjectError('Never compete with seed29')
    teacher=read_json(ROOT/cfg['teacher_config']);spec=read_json(ROOT/teacher['model_spec']);cloud=read_json(ROOT/a.cloud_config)
    report={'status':'starting','scope':'exploratory native-task check after six-choice order bias; not a prespecified confirmation',
        'source_regions_sha256':prep['regions_sha256'],'original_six_choice_scores_sha256':prior['scores_sha256'],
        'prompt':SCENE_TASK,'prompt_sha256':digest(SCENE_TASK),'score_source_sha256':sha256(ROOT/'src/fire_r2/teacher_transfer.py'),
        'view':'context1.0 with same thin outline','teacher':teacher['teacher'],'completed_regions':0,'max_seconds':600,
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'batches':[]}
    write_json(out/'report.json',report);started=time.monotonic()
    def timeout(*args):raise TimeoutError('Native compatibility diagnostic budget reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        with cloud_gpu(ROOT,cloud):
            import torch
            from transformers import Qwen3VLForConditionalGeneration
            from peft import PeftModel
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            processor,identity=checked_processor(ROOT,spec,cloud);processor.tokenizer.padding_side='left'
            adapter=ROOT/teacher['adapter']
            if sha256(adapter/'adapter_model.safetensors')!=teacher['teacher']['adapter_sha256']:raise ProjectError('Adapter differs')
            provenance=read_json(adapter.parent/'report.json')
            if provenance['identity']!=identity or provenance['status']!='completed':raise ProjectError('Adapter provenance differs')
            model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/spec['model_dir'],local_files_only=True,trust_remote_code=False,dtype=torch.float16,attn_implementation='sdpa').to('cuda')
            model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False);model.eval();model.requires_grad_(False)
            report.update(status='running',identity=identity,load_seconds=time.monotonic()-started);torch.cuda.reset_peak_memory_stats()
            rows=[json.loads(x) for x in (parent/'regions.jsonl').read_text().splitlines()]
            with (out/'scores.jsonl').open('x') as f:
                for start in range(0,len(rows),4):
                    batch=rows[start:start+4];tick=time.monotonic();views=[]
                    for r in batch:
                        if sha256(Path(r['image_path']))!=r['image_sha256']:raise ProjectError('Image changed')
                        views.append(region_view(r,1.))
                    results=score_presence_views(model,processor,views,'cuda');torch.cuda.synchronize()
                    for r,view,s in zip(batch,views,results):
                        s.update(id=r['id'],view_sha256=__import__('hashlib').sha256(view.tobytes()).hexdigest());f.write(json.dumps(s)+'\n')
                    f.flush();report['completed_regions']+=len(batch);report['batches'].append({'regions':len(batch),'seconds':time.monotonic()-tick});write_json(out/'report.json',report)
                    print(json.dumps({'regions':report['completed_regions'],'seconds':time.monotonic()-started}),flush=True)
            report.update(status='completed',peak_gpu_bytes=torch.cuda.max_memory_allocated(),scores_sha256=sha256(out/'scores.jsonl'))
    except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-started;report['cumulative_teacher_seconds']=report['seconds']+prior['seconds'];write_json(out/'report.json',report)
    lookup={r['id']:r for r in [json.loads(x) for x in (out/'scores.jsonl').read_text().splitlines()]}
    previous={(r['id'],r['view']):r for r in [json.loads(x) for x in (parent/'scores.jsonl').read_text().splitlines()]}
    result={'status':'completed','scope':report['scope'],'ranking':{},'halves':{},'GT_groups':{},'all_regions_same_as_first_stage':True}
    import collections
    for cls in (0,1):
        pairs=[]
        for r in rows:
            if lookup[r['id']]['view_sha256']!=previous[(r['id'],'context1.0')]['view_sha256']:raise ProjectError('Paired tight crop changed')
            if r['predicted_class']!=cls or r['kind'] not in ('prediction_tp','prediction_background','prediction_cross_class'):continue
            support=lookup[r['id']]['presence_probabilities'][cls]
            pairs.append({**r,'positive':int(r['kind']=='prediction_tp'),'P0':r['P0_score'],'teacher':support,'combined':math.sqrt(support*r['P0_score'])})
        name=('fire','smoke')[cls];result['ranking'][name]={k:metrics(pairs,k) for k in ['P0','teacher','combined']}
        for half in (0,1):result['halves'][f'{name}_{half}']={k:metrics([r for r in pairs if r['diagnostic_half']==half],k) for k in ['P0','teacher','combined']}
    groups=collections.defaultdict(list)
    for r in rows:
        if not r['kind'].startswith(('GT','external')):continue
        cls=r['original_GT_class'];s=lookup[r['id']];support=s['presence_probabilities'][cls] if cls in (0,1) else s['joint_probabilities'][0]
        groups[str((r['source'],r['kind'],cls))].append(support)
    result['GT_groups']={k:{'regions':len(v),'mean_support':sum(v)/len(v),'support_ge_0.5':sum(x>=.5 for x in v)} for k,v in groups.items()}
    write_json(out/'analysis.json',result)


if __name__=='__main__':main()
