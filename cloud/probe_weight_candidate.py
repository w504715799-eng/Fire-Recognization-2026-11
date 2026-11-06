"""One bounded CPU probe of conditional semantic support; v2 prompts/rendering stay frozen."""
from pathlib import Path
import argparse,hashlib,json,os,resource,signal,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.runtime import check_remote_environment
from fire_r2.teacher_protocols import protocol,AnswerTrie,valid_answers
from fire_r2.teacher_runner import teacher_view
from fire_r2.supervision import SEMANTIC_FIELDS,selected_regions,normalize_weights,group_key,parse_advice
from fire_r2.weight_candidate import candidate_raw_score
from probe_teacher_cpu import training_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True)
    p.add_argument('--output',default='artifacts/w-probability-candidate-20261106');a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('Explicit CPU-only probe required')
    cfg_path=ROOT/'configs/w_probability_candidate.json';cfg=read_json(cfg_path);spec=protocol('v2')
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Keep the first bounded attempt; no coefficient search or overwrite')
    out.mkdir(parents=True)
    rows=read_json(ROOT/'artifacts/teacher-audit-20261106/selected-fit-metadata.json')['rows']
    reference=read_json(ROOT/'artifacts/teacher-cpu-expanded-20261106/report.json')
    frozen={(c['sample_id'],c['task'],c.get('original_annotation_id')):c for c in reference['calls']}
    report={'status':'loading','prerecorded_criteria_sha256':sha256(cfg_path),'criteria':cfg,
            'device':'cpu','dtype':'float32','gpu_lock_acquired':False,'training_before':training_snapshot(),
            'queries':[],'samples':{},'original_GT_modified':False,'full_fit_training_authorized_by_probe':False}
    started=time.monotonic();old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(*args):raise TimeoutError('W candidate CPU budget exceeded')
    signal.signal(signal.SIGALRM,timeout);write_json(out/'report.json',report)
    try:
        import torch,numpy as np
        torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        model_dir=ROOT/'weights/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17'
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,
            dtype=torch.float32,attn_implementation='sdpa').to('cpu').eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        if torch.cuda.is_initialized() or any(p.device.type!='cpu' for p in model.parameters()):raise ProjectError('CPU contract violated')
        eos=model.generation_config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
        tries={t:AnswerTrie(processor.tokenizer,t,eos) for t in ('A','B')}
        candidates={t:[(processor.tokenizer.encode(s,add_special_tokens=False),json.loads(s)) for s in valid_answers(t)] for t in ('A','B')}
        report.update(status='running',load_seconds=time.monotonic()-started,torch=torch.__version__)
        with (out/'queries.jsonl').open('x') as journal:
            for row in rows:
                item={'scene':{},'regions':[]};image=Path(row['image_relpath']);digest=sha256(image)
                tasks=[('B',b) for b in selected_regions(row)] if row['boxes'] else [('A',None)]
                for task,box in tasks:
                    remaining=cfg['max_seconds']-(time.monotonic()-started)
                    if remaining<=0:raise TimeoutError('W candidate total budget exceeded')
                    signal.alarm(max(1,min(90,int(remaining))));tick=time.monotonic()
                    view=teacher_view(image,box,region_marker=spec['region_marker'])
                    message=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':spec[task]}]}]
                    text=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    inputs=processor(text=[text],images=[view],return_tensors='pt');length=inputs['input_ids'].shape[1]
                    traces={};fields=SEMANTIC_FIELDS if task=='A' else ('content','visibility');trie=tries[task]
                    def allowed(batch_id,ids):
                        prefix=ids[length:].tolist();tokens=trie.allowed(prefix)
                        if len(tokens)>1:
                            eligible=[(ts,obj) for ts,obj in candidates[task] if ts[:len(prefix)]==prefix]
                            field=next((f for f in fields if len({obj[f] for _,obj in eligible})>1),None)
                            if field is None:raise ProjectError('Unidentified conditional field branch')
                            mapping={token:{obj[field] for ts,obj in eligible if ts[len(prefix)]==token} for token in tokens}
                            if any(len(v)!=1 for v in mapping.values()):raise ProjectError('Ambiguous class token; stop this candidate implementation')
                            traces[field]={'step':len(prefix),'tokens':tokens,'labels':[next(iter(mapping[t])) for t in tokens]}
                        return tokens
                    with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,
                        max_new_tokens=trie.max_answer_tokens+1,eos_token_id=eos,prefix_allowed_tokens_fn=allowed,
                        return_dict_in_generate=True,output_scores=True)
                    raw=processor.batch_decode(generated.sequences[:,length:],skip_special_tokens=True)[0]
                    parsed,status=parse_advice(raw,task)
                    if status!='ok':raise ProjectError('Frozen constrained answer unexpectedly invalid')
                    probabilities={}
                    for field,trace in traces.items():
                        values=torch.softmax(generated.scores[trace['step']][0,trace['tokens']].float(),dim=0).tolist()
                        probs={}
                        for label,value in zip(trace['labels'],values):probs[label]=probs.get(label,0.)+value
                        if not all(np.isfinite(v) for v in probs.values()):raise ProjectError('Nonfinite conditional scores')
                        probabilities[field]=probs
                    signal.alarm(0)
                    key=(row['sample_id'],task,box['original_annotation_id'] if box else None)
                    old=frozen[key]
                    if old['image_sha256']!=digest or old['prompt_sha256']!=hashlib.sha256(spec[task].encode()).hexdigest():raise ProjectError('Frozen v2 binding differs')
                    record={'sample_id':row['sample_id'],'task':task,'original_annotation_id':key[2],
                        'raw_response':raw,'parsed':parsed,'probabilities':probabilities,'image_sha256':digest,
                        'prompt_sha256':old['prompt_sha256'],'seconds':time.monotonic()-tick,
                        'answer_matches_frozen_cpu_v2':parsed==old['parsed'],
                        'score_scope':'conditional probabilities among legal next-token labels; not calibrated correctness'}
                    report['queries'].append(record);journal.write(json.dumps(record)+'\n');journal.flush()
                    if task=='A':item['scene']=probabilities
                    else:item['regions'].append({'original_annotation_id':key[2],'content':probabilities['content']})
                    write_json(out/'report.json',report)
                    print(json.dumps({'queries':len(report['queries']),'seconds':record['seconds'],'same_answer':record['answer_matches_frozen_cpu_v2']}),flush=True)
                    del generated
                report['samples'][row['sample_id']]=item
        raw={r['sample_id']:candidate_raw_score(r,report['samples'][r['sample_id']]) for r in rows}
        weights=normalize_weights(raw,rows);groups={}
        for row in rows:groups.setdefault(group_key(row),[]).append(weights[row['sample_id']])
        summary={g:{'images':len(v),'IQR':float(np.percentile(v,75)-np.percentile(v,25)),
                    'sum':sum(v),'min':min(v),'max':max(v)} for g,v in groups.items()}
        qualifying=[g for g,v in summary.items() if v['images']>=6 and v['IQR']>=cfg['continue_signal']['minimum_normalized_weight_IQR']]
        ranked=sorted(weights,key=lambda k:(weights[k],k));review=ranked[:4]+ranked[-4:]
        report.update(status='completed',raw_scores=raw,weights=weights,groups=summary,qualifying_groups=qualifying,
            nonconstant_gate_passed=len(qualifying)>=cfg['continue_signal']['minimum_groups_with_at_least_6_images'],
            bounded_visual_screening_ids=review,visual_screening_status='pending',cuda_initialized=torch.cuda.is_initialized())
    except BaseException as exc:
        report.update(status='stopped',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
        report.update(total_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      training_after=training_snapshot());write_json(out/'report.json',report)


if __name__=='__main__':main()
