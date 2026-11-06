"""Explicit remote CPU-only teacher probe, alongside an unchanged GPU training run."""
from pathlib import Path
import argparse,csv,hashlib,json,os,resource,signal,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.runtime import check_remote_environment
from fire_r2.teacher_runner import teacher_view
from fire_r2.teacher_protocols import protocol,AnswerTrie
from fire_r2.supervision import selected_regions,parse_advice,digest_object


def training_snapshot():
    p=ROOT/'runs/dfire_B_seed17_full_20261106/results.csv';rows=list(csv.DictReader(p.open()))
    return {'epochs':len(rows),'recent_epoch_seconds':[round(float(b['time'])-float(a['time']),3) for a,b in zip(rows[-4:-1],rows[-3:])]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True)
    p.add_argument('--threads',type=int,default=6);p.add_argument('--per-state',type=int,default=1)
    p.add_argument('--protocol',choices=['v1','v2'],default='v2')
    p.add_argument('--model-config',help='Explicit separately verified candidate model; never reuse legacy assets')
    p.add_argument('--per-call-seconds',type=int,default=90);p.add_argument('--max-seconds',type=int,default=600)
    p.add_argument('--reuse-report',help='Reuse completed identical CPU queries; never use GPU/other precision answers')
    p.add_argument('--output',default='artifacts/teacher-cpu-probe-20261106');a=p.parse_args()
    if not 1<=a.threads<=8 or not 1<=a.per_state<=16:raise ProjectError('Bound CPU probe size/threads')
    cloud=read_json(ROOT/a.cloud_config);check_remote_environment(ROOT,cloud)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('CPU mode requires empty CUDA_VISIBLE_DEVICES before imports')
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve previous CPU probe; choose a fresh output')
    out.mkdir(parents=True)
    spec=protocol(a.protocol);metadata=read_json(ROOT/'artifacts/teacher-audit-20261106/selected-fit-metadata.json')
    rows=[]
    for state in ('fire_only','smoke_only','both','empty'):
        rows.extend([r for r in metadata['rows'] if r['state']==state][:a.per_state])
    if any(r['role']!='fit' for r in rows):raise ProjectError('CPU probe is source-fit only')
    candidate=read_json(ROOT/a.model_config) if a.model_config else None
    if candidate:
        if candidate.get('variant','base')!='base':raise ProjectError('Base candidate probe does not load adapters; do not label base outputs as adapted')
        from fire_r2.teacher_models import candidate_plan
        plan=candidate_plan(ROOT,candidate)
        if plan['issues']:raise ProjectError('Candidate model assets are not ready')
        model_dir=Path(plan['directory']);model_id=candidate['model_id']
    else:
        model_dir=ROOT/'weights/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17';model_id='Qwen/Qwen3.5-0.8B'
    receipt=read_json(model_dir/'download-receipt.json')
    if receipt.get('model_id')!=model_id or not receipt.get('runtime_content_verified_against_hf'):raise ProjectError('Verified matching teacher assets required')
    report={'status':'loading','model_id':model_id,'execution_device':'cpu','dtype':'float32','threads':a.threads,
            'interop_threads':1,'gpu_lock_acquired':False,'new_weights_downloaded':False,'protocol':a.protocol,
            'generation':{'do_sample':False,'enable_thinking':False,'constrained_answer_grammar':spec['constrained']},
            'teacher_revision':receipt['hf_revision'],'samples':[r['sample_id'] for r in rows],
            'source_manifest_sha256':metadata['source_manifest_sha256'],'calls':[],
            'training_before':training_snapshot(),'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            'memory_limit_bytes':int(Path('/sys/fs/cgroup/memory.max').read_text()),
            'interpretation':'CPU semantic diagnostics only; CPU timings are not GPU throughput or student gains.'}
    completed={}
    if a.reuse_report:
        prior_path=ROOT/a.reuse_report;prior=read_json(prior_path)
        if prior.get('model_id','Qwen/Qwen3.5-0.8B')!=model_id:raise ProjectError('CPU reuse model ID differs')
        for key in ('execution_device','dtype','threads','protocol','teacher_revision','source_manifest_sha256'):
            if prior.get(key)!=report[key]:raise ProjectError('CPU reuse binding differs: '+key)
        selected={r['sample_id'] for r in rows}
        for call in prior['calls']:
            if call['sample_id'] not in selected:continue
            key=(call['sample_id'],call['task'],call.get('original_annotation_id'))
            if key in completed:raise ProjectError('Duplicate CPU query in reused report')
            if call['prompt_sha256']!=hashlib.sha256(spec[call['task']].encode()).hexdigest():
                raise ProjectError('Reused CPU prompt differs')
            completed[key]=call
        report['calls']=list(completed.values())
        report.update(reused_report=str(prior_path),reused_report_sha256=sha256(prior_path),reused_calls=len(completed))
    started=time.monotonic();previous_handler=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):raise TimeoutError('Bounded CPU teacher call timed out')
    signal.signal(signal.SIGALRM,timeout)
    write_json(out/'report.json',report)
    try:
        # No CUDA selection, shared GPU lock, or GPU occupancy check in this mode.
        import torch
        torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
        if candidate:
            from fire_r2.teacher_models import checked_processor
            processor,identity=checked_processor(ROOT,candidate,cloud)
            from transformers import Qwen3VLForConditionalGeneration as Model
            report['model_identity']=identity
        else:
            from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration as Model
            processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
        model=Model.from_pretrained(model_dir,local_files_only=True,
                dtype=torch.float32,attn_implementation='sdpa').to('cpu').eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        if any(p.device.type!='cpu' for p in model.parameters()) or torch.cuda.is_initialized():
            raise ProjectError('CPU-only execution contract violated')
        report['load_seconds']=time.monotonic()-started
        eos=model.generation_config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
        tries={task:AnswerTrie(processor.tokenizer,task,eos) for task in ('A','B')} if spec['constrained'] else {}
        report.update(status='running',torch=torch.__version__,actual_torch_threads=torch.get_num_threads())
        with (out/'responses.jsonl').open('x') as journal:
            for row in rows:
                image=Path(row['image_relpath']);image_hash=sha256(image)
                for task,box in [('A',None)]+[('B',b) for b in selected_regions(row)]:
                    key=(row['sample_id'],task,box['original_annotation_id'] if box else None)
                    if key in completed:
                        if completed[key]['image_sha256']!=image_hash:raise ProjectError('Reused CPU image differs')
                        continue
                    remaining=a.max_seconds-(time.monotonic()-started)
                    if remaining<=0:raise TimeoutError('CPU stage budget reached')
                    tick=time.monotonic();signal.alarm(max(1,min(a.per_call_seconds,int(remaining))))
                    view=teacher_view(image,box,region_marker=spec['region_marker'])
                    messages=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':spec[task]}]}]
                    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    inputs=processor(text=[text],images=[view],return_tensors='pt')
                    extra={'max_new_tokens':192}
                    if tries:
                        trie=tries[task];extra={'max_new_tokens':trie.max_answer_tokens+1,'eos_token_id':eos,
                            'prefix_allowed_tokens_fn':trie.callback(inputs['input_ids'].shape[1])}
                    with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,**extra)
                    ids=generated[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(ids,skip_special_tokens=True)[0]
                    signal.alarm(0);answer,status=parse_advice(raw,task)
                    item={'sample_id':row['sample_id'],'task':task,'original_annotation_id':box['original_annotation_id'] if box else None,
                          'image_sha256':image_hash,'view_sha256':hashlib.sha256(view.tobytes()).hexdigest(),
                          'bbox':box['bbox'] if box else None,'bbox_format':box['format'] if box else None,
                          'prompt_sha256':hashlib.sha256(spec[task].encode()).hexdigest(),
                          'input_shape':list(inputs['input_ids'].shape),'image_grid_thw':inputs['image_grid_thw'].tolist(),
                          'raw_response':raw,'parsed':answer,'parse_status':status,'seconds':time.monotonic()-tick,
                          'output_tokens':int(ids.shape[1]),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    report['calls'].append(item);journal.write(json.dumps(item,ensure_ascii=False)+'\n');journal.flush()
                    write_json(out/'report.json',report)
                    print(json.dumps({'task':task,'sample_id':row['sample_id'],'seconds':item['seconds'],'status':status}),flush=True)
        report['status']='completed'
        report['cuda_initialized']=torch.cuda.is_initialized()
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)
        report.update(total_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      completed_new_calls=len(report['calls'])-report.get('reused_calls',0),
                      training_after=training_snapshot())
        write_json(out/'report.json',report)


if __name__=='__main__':main()
