"""Matched GPU base/adapter evaluation: complete heldout short tasks plus frozen v2 regression."""
from pathlib import Path
import argparse,hashlib,itertools,json,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,output_path,sha256
from fire_r2.runtime import cloud_gpu
from fire_r2.teacher_models import checked_processor
from fire_r2.teacher_runner import teacher_view
from fire_r2.teacher_protocols import AnswerTrie,protocol,valid_answers
from fire_r2.supervision import parse_advice
from fire_r2.stopping import TERMINAL_TRAINING


def main():
    signal.signal(signal.SIGINT,signal.default_int_handler)
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--adapter');p.add_argument('--baseline-report');a=p.parse_args()
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Keep earlier GPU evaluation')
    out.mkdir(parents=True);spec=read_json(ROOT/'configs/qwen3vl2b_lora_pilot.json');cloud=read_json(ROOT/a.cloud_config)
    started=time.monotonic();report={'status':'starting','variant':'adapter' if a.adapter else 'base',
        'device':'cuda','dtype':'float16_base','batch':4,'responses':[],'max_seconds':1200}
    def timeout(*args):raise TimeoutError('Teacher GPU evaluation time budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(1200)
    try:
        with cloud_gpu(ROOT,cloud):
            import torch
            from transformers import Qwen3VLForConditionalGeneration
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            processor,identity=checked_processor(ROOT,spec,cloud);processor.tokenizer.padding_side='left';report['identity']=identity
            model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/spec['model_dir'],local_files_only=True,trust_remote_code=False,dtype=torch.float16,attn_implementation='sdpa').to('cuda')
            if a.adapter:
                from peft import PeftModel
                adapter=output_path(ROOT,a.adapter);training=read_json(adapter.parent/'report.json')
                if training['status'] not in TERMINAL_TRAINING or training['mode'] not in ('fixed_CPU_task_LoRA','fixed_GPU_task_LoRA') or training['completed_updates']<1:raise ProjectError('A finalized task adapter with a saved checkpoint is required')
                if training['identity']!=identity:raise ProjectError('Adapter base identity differs')
                for item in training['adapter_files']:
                    if sha256(adapter.parent/item['name'])!=item['sha256']:raise ProjectError('Adapter bytes changed')
                model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False)
                report['adapter_training_report_sha256']=sha256(adapter.parent/'report.json')
            model.eval();model.requires_grad_(False);model.config.use_cache=True
            eos=model.generation_config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
            data=ROOT/spec['data_root'];receipt=read_json(data/'receipt.json');jobs=[]
            for cohort,part in [('heldout','teacher_holdout'),('v2_regression','development_regression')]:
                path=data/(part+'.instructions.jsonl')
                if sha256(path)!=receipt['partitions'][part]['instruction_sha256']:raise ProjectError('Frozen evaluation instructions changed')
                report[part+'_sha256']=sha256(path)
                for line in path.open():
                    item=json.loads(line)
                    if item['teacher_partition']!=part:raise ProjectError('Evaluation partition changed')
                    item.update(cohort=cohort);jobs.append(item)
            if len(jobs)!=640:raise ProjectError('Expected complete 512 heldout plus 128 regression queries')
            short={'scene_presence':[json.dumps({'fire_present':x,'smoke_present':y},separators=(',',':')) for x,y in itertools.product(('yes','no'),repeat=2)],
                   'region_class':[json.dumps({'target_class':x},separators=(',',':')) for x in ('fire','smoke')]}
            v2=protocol('v2');report.update(status='running',load_seconds=time.monotonic()-started);torch.cuda.reset_peak_memory_stats()
            with (out/'responses.jsonl').open('x') as journal:
                for cohort in ('heldout','v2_regression'):
                    for task in ('scene_presence','region_class'):
                        rows=[j for j in jobs if j['cohort']==cohort and j['task']==task];legacy='A' if task=='scene_presence' else 'B'
                        candidates=short[task] if cohort=='heldout' else valid_answers(legacy)
                        trie=AnswerTrie(processor.tokenizer,task,eos,candidates)
                        for start in range(0,len(rows),4):
                            batch=rows[start:start+4];views=[];texts=[];prompts=[];tick=time.monotonic()
                            for item in batch:
                                if sha256(Path(item['image_path']))!=item['image_sha256']:raise ProjectError('Evaluation image changed')
                                box={'bbox':item['bbox'],'format':item['bbox_format']} if task=='region_class' else None
                                view=teacher_view(item['image_path'],box,region_marker='thin_outline_after_resize');views.append(view)
                                prompt=item['instruction'] if cohort=='heldout' else v2[legacy];prompts.append(prompt)
                                msg=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':prompt}]}]
                                texts.append(processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False))
                            inputs=processor(text=texts,images=views,padding=True,return_tensors='pt').to('cuda')
                            with torch.inference_mode():pred=model.generate(**inputs,do_sample=False,eos_token_id=eos,max_new_tokens=trie.max_answer_tokens+1,prefix_allowed_tokens_fn=trie.callback(inputs['input_ids'].shape[1]))
                            raw=processor.batch_decode(pred[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True);torch.cuda.synchronize();elapsed=time.monotonic()-tick
                            for item,view,prompt,text in zip(batch,views,prompts,raw):
                                try:answer=json.loads(text)
                                except ValueError:answer=None
                                valid=answer in [json.loads(c) for c in candidates] if cohort=='heldout' else parse_advice(text,legacy)[1]=='ok'
                                r={'cohort':cohort,'sample_id':item['sample_id'],'task':task,'annotation_id':item['original_annotation_id'],
                                   'raw_response':text,'answer':answer,'valid':valid,'view_sha256':hashlib.sha256(view.tobytes()).hexdigest(),
                                   'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'seconds_amortized':elapsed/len(batch),
                                   'agrees_original_annotation':answer==item['target'] if cohort=='heldout' else None}
                                report['responses'].append(r);journal.write(json.dumps(r)+'\n');journal.flush()
                            write_json(out/'report.json',report);print(json.dumps({'queries':len(report['responses']),'batch_seconds':elapsed}),flush=True)
            report['heldout_tasks']={task:{'queries':256,'agree':sum(r['cohort']=='heldout' and r['task']==task and r['agrees_original_annotation'] for r in report['responses'])} for task in ('scene_presence','region_class')}
            if a.baseline_report:
                base=read_json(output_path(ROOT,a.baseline_report))
                if base['status']!='completed' or base['variant']!='base' or base['identity']!=identity:raise ProjectError('Matched GPU base report required')
                key=lambda r:(r['cohort'],r['sample_id'],r['task'],r['annotation_id'])
                left={key(r):r for r in base['responses']};right={key(r):r for r in report['responses']}
                if set(left)!=set(right):raise ProjectError('Paired GPU query sets differ')
                for k in left:
                    if any(left[k][f]!=right[k][f] for f in ('view_sha256','prompt_sha256')):raise ProjectError('Paired GPU inputs differ')
                report['matched_base_sha256']=sha256(output_path(ROOT,a.baseline_report))
                report['v2_changed_answers']=sum(left[k]['answer']!=right[k]['answer'] for k in left if k[0]=='v2_regression')
            report.update(status='completed',peak_gpu_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-started;write_json(out/'report.json',report)


if __name__=='__main__':main()
