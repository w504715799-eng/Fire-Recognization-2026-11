"""Matched short-task teacher holdout on cloud CPU; raw answers retained, no training."""
from pathlib import Path
import argparse,hashlib,itertools,json,os,resource,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.runtime import check_remote_environment
from fire_r2.teacher_models import checked_processor
from fire_r2.teacher_runner import teacher_view
from fire_r2.teacher_protocols import AnswerTrie


def main():
    signal.signal(signal.SIGINT,signal.default_int_handler)
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True)
    p.add_argument('--model',choices=['0.8b','2b'],required=True);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-seconds',type=int,default=5400)
    p.add_argument('--resume-report');p.add_argument('--output')
    p.add_argument('--allow-interrupted-resume',action='store_true')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args()
    from fire_r2.teacher_holdout_resume import owns_batch
    owns_batch(0,4,a.shard_index,a.shard_count)
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.threads not in (2,4,6):raise ProjectError('Explicit bounded cloud CPU')
    out=output_path(ROOT,a.output or ('artifacts/teacher-heldout-20261106/base-'+a.model))
    if out.exists():raise ProjectError('Keep previous holdout output')
    out.mkdir(parents=True);started=time.monotonic()
    source=ROOT/'artifacts/qwen3vl-2b-preparation-20261106/instructions/teacher_holdout.instructions.jsonl'
    receipt=read_json(source.parent/'receipt.json')
    if sha256(source)!=receipt['partitions']['teacher_holdout']['instruction_sha256']:raise ProjectError('Heldout manifest changed')
    examples=[json.loads(line) for line in source.open()]
    if len(examples)!=512 or len({r['sample_id'] for r in examples})!=256:raise ProjectError('Expected complete 256-image holdout')
    if any(r['detector_role']!='fit' or r['teacher_partition']!='teacher_holdout' for r in examples):raise ProjectError('Wrong holdout role')
    report={'status':'loading','device':'cpu','dtype':'float32','threads':a.threads,'batch':4,
        'instruction_sha256':sha256(source),'scope':receipt['heldout_scope'],'responses':[],
        'context_fields_evaluated':False,'gpu_lock_acquired':False,'max_seconds':a.max_seconds,
        'shard_index':a.shard_index,'shard_count':a.shard_count}
    def timeout(*args):raise TimeoutError('Bounded CPU holdout stage/batch timed out')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(a.max_seconds)
    try:
        import torch
        torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
        if a.model=='2b':
            spec=read_json(ROOT/'configs/teacher_qwen3vl2b_base.json');directory=ROOT/spec['model_dir']
            processor,identity=checked_processor(ROOT,spec,read_json(ROOT/a.cloud_config))
            from transformers import Qwen3VLForConditionalGeneration as Model
        else:
            directory=ROOT/'weights/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17'
            from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration as Model
            processor=AutoProcessor.from_pretrained(directory,local_files_only=True)
            asset=read_json(directory/'download-receipt.json')
            if asset['model_id']!='Qwen/Qwen3.5-0.8B' or not asset['runtime_content_verified_against_hf']:raise ProjectError('Wrong legacy assets')
            identity={'model_id':asset['model_id'],'revision':asset['hf_revision'],'tokenizer_sha256':sha256(directory/'tokenizer.json')}
        report['identity']=identity;processor.tokenizer.padding_side='left';completed={}
        if a.resume_report:
            from fire_r2.teacher_holdout_resume import resume_answers
            prior_path=output_path(ROOT,a.resume_report);prior=read_json(prior_path)
            completed=resume_answers(prior,report,identity,examples,a.allow_interrupted_resume)
            report.update(responses=list(prior['responses']),reused_queries=len(completed),
                          previous_report=str(prior_path),previous_report_sha256=sha256(prior_path),
                          previous_segment_seconds=prior['seconds'])
        model=Model.from_pretrained(directory,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to('cpu').eval()
        model.requires_grad_(False)
        eos=model.generation_config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
        answers={'scene_presence':[{'fire_present':x,'smoke_present':y} for x,y in itertools.product(('yes','no'),repeat=2)],
                 'region_class':[{'target_class':x} for x in ('fire','smoke')]}
        tries={task:AnswerTrie(processor.tokenizer,task,eos,[json.dumps(x,separators=(',',':')) for x in values]) for task,values in answers.items()}
        report.update(status='running',load_seconds=time.monotonic()-started)
        with (out/'responses.jsonl').open('x') as log:
            for task in answers:
                rows=[r for r in examples if r['task']==task]
                for start in range(0,len(rows),4):
                    batch=rows[start:start+4]
                    keys=[(r['sample_id'],r['task'],r['original_annotation_id']) for r in batch]
                    if not all(key in completed for key in keys) and not owns_batch(start,4,a.shard_index,a.shard_count):continue
                    remaining=a.max_seconds-(time.monotonic()-started)
                    if remaining<=0:raise TimeoutError('CPU heldout stage budget')
                    signal.alarm(max(1,min(180,int(remaining))));tick=time.monotonic();batch=rows[start:start+4];views=[];texts=[]
                    for r in batch:
                        if sha256(Path(r['image_path']))!=r['image_sha256']:raise ProjectError('Heldout source image changed')
                        box={'bbox':r['bbox'],'format':r['bbox_format']} if task=='region_class' else None
                        view=teacher_view(r['image_path'],box,region_marker='thin_outline_after_resize');views.append(view)
                        msg=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':r['instruction']}]}]
                        texts.append(processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False))
                    keys=[(r['sample_id'],r['task'],r['original_annotation_id']) for r in batch]
                    if all(key in completed for key in keys):
                        for key,view in zip(keys,views):
                            if hashlib.sha256(view.tobytes()).hexdigest()!=completed[key]['view_sha256']:
                                raise ProjectError('Resumed input view bytes differ')
                        signal.alarm(0)
                        continue
                    inputs=processor(text=texts,images=views,padding=True,return_tensors='pt');trie=tries[task]
                    with torch.inference_mode():outputs=model.generate(**inputs,do_sample=False,eos_token_id=eos,max_new_tokens=trie.max_answer_tokens+1,prefix_allowed_tokens_fn=trie.callback(inputs['input_ids'].shape[1]))
                    raw=processor.batch_decode(outputs[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
                    elapsed=time.monotonic()-tick;signal.alarm(0)
                    for r,view,text in zip(batch,views,raw):
                        try:answer=json.loads(text)
                        except ValueError:answer=None
                        item={'sample_id':r['sample_id'],'task':task,'annotation_id':r['original_annotation_id'],
                            'raw_response':text,'target_from_original_annotation':r['target'],'answer':answer,
                            'valid':answer in answers[task],'agrees_original_annotation':answer==r['target'],
                            'view_sha256':hashlib.sha256(view.tobytes()).hexdigest(),'prompt_sha256':r['instruction_sha256'],
                            'seconds_amortized':elapsed/len(batch)}
                        report['responses'].append(item);log.write(json.dumps(item)+'\n');log.flush()
                    report['completed_queries']=len(report['responses']);write_json(out/'report.json',report)
                    print(json.dumps({'model':a.model,'queries':len(report['responses']),'batch_seconds':elapsed}),flush=True)
        if torch.cuda.is_initialized():raise ProjectError('CPU contract violated')
        report.update(status='completed' if a.shard_count==1 else 'completed_shard',cuda_initialized=False)
        report['tasks']={task:{'queries':sum(r['task']==task for r in report['responses']),
                               'agree':sum(r['task']==task and r['agrees_original_annotation'] for r in report['responses'])} for task in answers}
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);report.update(seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(out/'report.json',report)


if __name__=='__main__':main()
