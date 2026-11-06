"""CPU-only real instruction view/token preflight; no weights or pixel cache."""
from pathlib import Path
import argparse,hashlib,json,os,resource,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256
from fire_r2.teacher_models import checked_processor
from fire_r2.teacher_runner import teacher_view
from fire_r2.teacher_adaptation import supervised_value_spans,semantic_token_labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('CPU-only preflight')
    spec=read_json(ROOT/'configs/qwen3vl2b_lora_pilot.json');out=ROOT/'artifacts/qwen3vl-2b-preparation-20261106/input-preflight'
    if out.exists():raise ProjectError('Keep previous preflight evidence')
    out.mkdir();started=time.monotonic();report={'status':'running','weights_loaded':False,'checked':0,'errors':[],'max_length':0}
    try:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        processor,identity=checked_processor(ROOT,spec,read_json(ROOT/a.cloud_config));report['identity']=identity
        with (out/'examples.jsonl').open('x') as log:
            for part in ('teacher_train','teacher_dev','teacher_holdout'):
                source=ROOT/spec['data_root']/(part+'.instructions.jsonl')
                for index,line in enumerate(source.open()):
                    if time.monotonic()-started>600:raise TimeoutError('10 minute preflight limit')
                    r=json.loads(line)
                    try:
                        if r['detector_role']!='fit' or r['teacher_partition']!=part:raise ProjectError('Partition mismatch')
                        if r.get('image_sha256') and sha256(Path(r['image_path']))!=r['image_sha256']:raise ProjectError('Image bytes changed')
                        box={'bbox':r['bbox'],'format':r['bbox_format']} if r['task']=='region_class' else None
                        view=teacher_view(r['image_path'],box,region_marker='thin_outline_after_resize')
                        msg=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':r['instruction']}]}]
                        prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        target,spans=supervised_value_spans(r['target']);text=prompt+target+processor.tokenizer.eos_token
                        plain=processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
                        inputs=processor(text=[text],images=[view],return_tensors='pt',add_special_tokens=False)
                        ids=inputs['input_ids'][0].tolist();labels=semantic_token_labels(plain['input_ids'],plain['offset_mapping'],ids,[(len(prompt)+x,len(prompt)+y) for x,y in spans])
                        if len(ids)>spec['max_sequence_length']:raise ProjectError('Sequence exceeds budget; do not truncate supervision')
                        item={'partition':part,'index':index,'sample_id':r['sample_id'],'task':r['task'],
                              'length':len(ids),'label_positions':[i for i,x in enumerate(labels) if x!=-100],
                              'view_sha256':hashlib.sha256(view.tobytes()).hexdigest()}
                        report['max_length']=max(report['max_length'],len(ids));report['checked']+=1
                    except Exception as exc:
                        item={'partition':part,'index':index,'sample_id':r['sample_id'],'error':str(exc)};report['errors'].append(item)
                    log.write(json.dumps(item)+'\n')
                    if (index+1)%100==0:write_json(out/'report.json',report)
        report['status']='passed' if not report['errors'] and report['checked']==2768 else 'failed'
    except BaseException as exc:
        report.update(status='failed',exception=str(exc),traceback=traceback.format_exc());raise
    finally:
        report.update(seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
