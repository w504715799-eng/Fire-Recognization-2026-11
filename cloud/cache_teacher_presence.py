"""Bounded offline 2B joint-presence cache, using frozen inputs and all four candidate answers."""
from contextlib import nullcontext
from pathlib import Path
import argparse,hashlib,io,json,os,resource,signal,sys,time,traceback,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,output_path,sha256,file_lock
from fire_r2.runtime import check_remote_environment,cloud_gpu
from fire_r2.research_data import load_manifest
from fire_r2.teacher_models import checked_processor
from fire_r2.teacher_runner import teacher_view,pilot_selection
from fire_r2.teacher_transfer import PROTOCOL,score_presence_views
from fire_r2.teacher_adaptation import SCENE_TASK


def main():
    signal.signal(signal.SIGINT,signal.default_int_handler)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cloud-config',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--pilot-per-state',type=int,default=0);p.add_argument('--output');a=p.parse_args()
    cfg=read_json(ROOT/a.config);cloud=read_json(ROOT/a.cloud_config);check_remote_environment(ROOT,cloud)
    cpu=a.device=='cpu'
    if cpu and (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not 1<=a.pilot_per_state<=2):raise ProjectError('CPU is restricted to this bounded pilot, not whole-fit model scoring')
    out=output_path(ROOT,a.output or cfg['output']);out.mkdir(parents=True,exist_ok=True)
    with file_lock(out/'cache.lock'):
        return run_locked(a,cfg,cloud,out)


def run_locked(a,cfg,cloud,out):
    cpu=a.device=='cpu'
    manifest=ROOT/cfg['manifest'];rows=load_manifest(manifest,roles={'fit'},enrich=False);full_count=len(rows)
    if a.pilot_per_state:rows=pilot_selection(rows,a.pilot_per_state)
    binding={'manifest_sha256':sha256(manifest),'teacher':cfg['teacher'],'protocol':PROTOCOL,
             'prompt_sha256':hashlib.sha256(SCENE_TASK.encode()).hexdigest(),'device':a.device,'dtype':'float32' if cpu else 'float16_base',
             'scoring_source_sha256':sha256(ROOT/'src/fire_r2/teacher_transfer.py'),
             'batch_images':1 if cpu else cfg['batch_images'],'selection_sha256':hashlib.sha256(json.dumps([r['sample_id'] for r in rows]).encode()).hexdigest()}
    prior_path=out/'binding.json';samples={}
    if prior_path.exists():
        if read_json(prior_path)!=binding:raise ProjectError('Resume cache binding differs')
        if (out/'cache.json').exists():
            cache=read_json(out/'cache.json')
            if cache['status']!='completed' or set(cache['samples'])!={r['sample_id'] for r in rows}:raise ProjectError('Existing cache is incomplete')
            print(json.dumps({'status':'reused_completed_cache','path':str(out/'cache.json')}));return
        journal_path=out/'samples.jsonl'
        if journal_path.exists():
            for line in journal_path.open():
                item=json.loads(line);ident=item.pop('sample_id')
                if ident in samples:raise ProjectError('Duplicate cache journal ID')
                samples[ident]=item
        if (out/'report.json').exists():
            history=out/'history';history.mkdir(exist_ok=True)
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            (history/(stamp+'.json')).write_bytes((out/'report.json').read_bytes())
    else:write_json(prior_path,binding)
    if not set(samples)<={r['sample_id'] for r in rows}:raise ProjectError('Resume sample IDs differ')
    limit=900 if cpu else cfg['max_seconds'];started=time.monotonic()
    report={'status':'starting','binding':binding,'full_fit_images':full_count,'selected_images':len(rows),'reused_images':len(samples),'completed_images':len(samples),'batches':[]}
    def timeout(*args):raise TimeoutError('Teacher cache stage time budget reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(limit)
    try:
        with nullcontext() if cpu else cloud_gpu(ROOT,cloud):
            import torch
            from transformers import Qwen3VLForConditionalGeneration
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            spec=read_json(ROOT/cfg['model_spec']);processor,identity=checked_processor(ROOT,spec,cloud);processor.tokenizer.padding_side='left'
            expected=cfg['teacher']
            if expected['teacher_id']!=identity['model_id'] or expected['teacher_revision']!=identity['revision']:raise ProjectError('Teacher identity differs')
            model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/spec['model_dir'],local_files_only=True,dtype=torch.float32 if cpu else torch.float16,attn_implementation='sdpa').to(a.device)
            if expected['teacher_variant']=='adapted':
                from peft import PeftModel
                adapter=ROOT/cfg['adapter'];training=read_json(adapter.parent/'report.json')
                if training['status']!='completed' or training['completed_updates']!=250 or training['identity']!=identity:raise ProjectError('Verified completed task adapter required')
                if sha256(adapter/'adapter_model.safetensors')!=expected['adapter_sha256']:raise ProjectError('Adapter bytes differ')
                model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False)
                report['adapter_training_report_sha256']=sha256(adapter.parent/'report.json')
            elif expected['teacher_variant']!='base' or expected['adapter_sha256'] is not None:raise ProjectError('Unexpected base teacher definition')
            model.eval();model.requires_grad_(False);report.update(status='running',model_identity=identity,load_seconds=time.monotonic()-started)
            if not cpu:torch.cuda.reset_peak_memory_stats()
            write_json(out/'report.json',report)
            with (out/'samples.jsonl').open('a') as log:
                n=binding['batch_images']
                for start in range(0,len(rows),n):
                    batch=rows[start:start+n]
                    if all(r['sample_id'] in samples for r in batch):continue
                    views=[];hashes=[];tick=time.monotonic()
                    for r in batch:
                        raw=Path(r.get('image_path',r.get('image_relpath'))).read_bytes();hashes.append(hashlib.sha256(raw).hexdigest())
                        views.append(teacher_view(io.BytesIO(raw),region_marker='thin_outline_after_resize'))
                    outputs=score_presence_views(model,processor,views,a.device)
                    if not cpu:torch.cuda.synchronize()
                    elapsed=time.monotonic()-tick
                    for row,view,image_hash,item in zip(batch,views,hashes,outputs):
                        ident=row['sample_id'];item.update(image_sha256=image_hash,view_sha256=hashlib.sha256(view.tobytes()).hexdigest(),seconds_amortized=elapsed/len(batch))
                        if ident in samples:
                            old=samples[ident]
                            if old['image_sha256']!=image_hash or any(abs(x-y)>1e-6 for x,y in zip(old['presence_probabilities'],item['presence_probabilities'])):raise ProjectError('Repeated partial batch changed; preserve both attempts')
                            continue
                        samples[ident]=item;log.write(json.dumps({'sample_id':ident,**item})+'\n');log.flush()
                    report.update(completed_images=len(samples));report['batches'].append({'images':len(batch),'seconds':elapsed});write_json(out/'report.json',report)
                    print(json.dumps({'images':len(samples),'total':len(rows),'seconds':elapsed}),flush=True)
            cache={**binding,'scope':'pilot_fit_only' if a.pilot_per_state else 'full_source_fit','status':'completed','samples':samples,
                   'full_fit_images':full_count,'original_annotations_changed':False,'all_fit_images_retained':not a.pilot_per_state}
            write_json(out/'cache.json',cache);report.update(status='completed',peak_gpu_bytes=torch.cuda.max_memory_allocated() if not cpu else 0)
    except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);report.update(seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write_json(out/'report.json',report)


if __name__=='__main__':main()
