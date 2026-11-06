"""Bounded real CPU batch comparison using the production teacher batch builder."""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,output_path
from fire_r2.runtime import check_remote_environment
from fire_r2.teacher_protocols import protocol,AnswerTrie
from fire_r2.teacher_runner import build_batched_cache
from probe_teacher_cpu import training_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True)
    p.add_argument('--threads',type=int,default=6);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--per-state',type=int,default=2);p.add_argument('--max-seconds',type=int,default=600)
    p.add_argument('--batch-seconds',type=int,default=120)
    p.add_argument('--output',default='artifacts/teacher-cpu-batch4-20261106');a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('CPU comparison must disable CUDA before imports')
    if not 1<=a.threads<=8 or not 1<=a.batch<=4 or not 1<=a.per_state<=16:raise ProjectError('Bound CPU batch/resources')
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Keep previous CPU comparison intact')
    out.mkdir(parents=True);spec=protocol('v2')
    meta=read_json(ROOT/'artifacts/teacher-audit-20261106/selected-fit-metadata.json')
    rows=[]
    for state in ('fire_only','smoke_only','both','empty'):
        rows.extend([r for r in meta['rows'] if r['state']==state][:a.per_state])
    model_dir=ROOT/'weights/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17'
    receipt=read_json(model_dir/'download-receipt.json')
    if not receipt.get('runtime_content_verified_against_hf'):raise ProjectError('Staged teacher assets not verified')
    report={'status':'loading','device':'cpu','dtype':'float32','threads':a.threads,'batch_size':a.batch,
            'samples':[r['sample_id'] for r in rows],'gpu_lock_acquired':False,'batches':[],
            'training_before':training_snapshot(),'new_downloads':False}
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):raise TimeoutError('CPU batch comparison time limit')
    signal.signal(signal.SIGALRM,timeout);write_json(out/'report.json',report)
    try:
        import torch
        torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True);processor.tokenizer.padding_side='left'
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,
            dtype=torch.float32,attn_implementation='sdpa').to('cpu').eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        if any(p.device.type!='cpu' for p in model.parameters()) or torch.cuda.is_initialized():
            raise ProjectError('CPU-only comparison contract violated')
        report['load_seconds']=time.monotonic()-started
        eos=model.generation_config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
        tries={task:AnswerTrie(processor.tokenizer,task,eos) for task in ('A','B')}
        report['status']='running'
        def infer(views,prompt,task):
            remaining=a.max_seconds-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('CPU batch stage budget reached')
            signal.alarm(max(1,min(a.batch_seconds,int(remaining))));tick=time.monotonic()
            texts=[]
            for view in views:
                message=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':prompt}]}]
                texts.append(processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False))
            inputs=processor(text=texts,images=views,padding=True,return_tensors='pt');trie=tries[task]
            with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,
                max_new_tokens=trie.max_answer_tokens+1,eos_token_id=eos,
                prefix_allowed_tokens_fn=trie.callback(inputs['input_ids'].shape[1]))
            ids=generated[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(ids,skip_special_tokens=True)
            lengths=[v.index(eos)+1 if eos in v else len(v) for v in ids.tolist()]
            signal.alarm(0)
            batch={'task':task,'batch_size':len(views),'seconds':time.monotonic()-tick,
                'view_sha256_in_input_order':[hashlib.sha256(v.tobytes()).hexdigest() for v in views],
                'input_shape':list(inputs['input_ids'].shape),'image_grid_thw':inputs['image_grid_thw'].tolist(),
                'raw_responses_in_output_order':raw,'output_tokens':lengths,
                'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            report['batches'].append(batch);write_json(out/'report.json',report)
            print(json.dumps({k:batch[k] for k in ('task','batch_size','seconds','output_tokens')}),flush=True)
            return list(zip(raw,lengths))
        with (out/'samples.jsonl').open('x') as journal:
            def record(ident,item):journal.write(json.dumps({'sample_id':ident,'item':item},ensure_ascii=False)+'\n');journal.flush()
            cache=build_batched_cache(rows,meta['source_manifest_sha256'],'/',infer,receipt['hf_revision'],a.batch,
                                      checkpoint=record,protocol_version='v2')
        calls=[c for item in cache['samples'].values() for c in item['calls']]
        cache.update(scope='pilot_fit_only',batch_size=a.batch,generation={'do_sample':False,'enable_thinking':False,
            'constrained_answer_grammar':True,'dtype':'float32'},runtime={'execution_device':'cpu','dtype':'float32',
            'threads':a.threads,'total_query_seconds':sum(c['seconds'] for c in calls),
            'current_segment_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        write_json(out/'cache.json',cache)
        report.update(status='completed' if all(c['parse_status']=='ok' for c in calls) else 'completed_with_failures',
                      calls=len(calls),parse_ok=sum(c['parse_status']=='ok' for c in calls),cuda_initialized=torch.cuda.is_initialized())
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        report.update(total_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      training_after=training_snapshot());write_json(out/'report.json',report)


if __name__=='__main__':main()
