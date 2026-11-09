"""Bounded remote task LoRA: CPU one-update integration or exclusive-GPU fixed pilot."""
from contextlib import nullcontext
from pathlib import Path
import argparse,datetime,json,os,random,resource,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,output_path,sha256
from fire_r2.runtime import check_remote_environment,cloud_gpu
from fire_r2.teacher_models import checked_processor
from fire_r2.teacher_adaptation import instruction_inputs


def main():
    signal.signal(signal.SIGINT,signal.default_int_handler)
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--output');p.add_argument('--full-cpu',action='store_true')
    p.add_argument('--resume-from');p.add_argument('--fp32',action='store_true');p.add_argument('--resume-max-seconds',type=int,default=5400)
    p.add_argument('--config',default='configs/qwen3vl2b_lora_pilot.json');a=p.parse_args()
    spec=read_json(ROOT/a.config);cloud=read_json(ROOT/a.cloud_config);check_remote_environment(ROOT,cloud)
    cpu=a.device=='cpu'
    cpu_check=cpu and not a.full_cpu
    fp32=cpu or a.fp32
    if a.resume_from and (cpu_check or not a.output):raise ProjectError('Resume requires full training and a new output directory')
    if a.full_cpu and (not cpu or spec.get('training_device')!='cpu' or spec.get('training_dtype')!='float32'):
        raise ProjectError('Full CPU training requires its explicitly recorded FP32 CPU configuration')
    if a.full_cpu:
        check=read_json(output_path(ROOT,spec['cpu_step_report']))
        if check['status']!='completed' or check['completed_updates']!=1 or check['device']!='cpu' or check['first_update']['parameter_delta_L1']<=0:
            raise ProjectError('A successful actual CPU optimizer update is required')
    if cpu and os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('CPU check requires CUDA disabled before imports')
    if cpu_check and not a.output:raise ProjectError('CPU integration needs its own output, never the GPU adapter directory')
    out=output_path(ROOT,a.output or spec['output_dir'])
    if out.exists():raise ProjectError('Keep prior adapter/run evidence')
    out.mkdir(parents=True);started=time.monotonic();model=None;adapter_saved=False
    updates=1 if cpu_check else spec['optimizer_updates'];limit=a.resume_max_seconds if a.resume_from else 900 if cpu_check else spec['max_training_seconds']
    report={'status':'starting','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':a.device,
        'mode':'CPU_one_update_integration_not_research_adapter' if cpu_check else 'fixed_CPU_task_LoRA' if cpu else 'fixed_GPU_task_LoRA',
        'config_sha256':sha256(ROOT/a.config),'planned_updates':updates,'completed_updates':0,
        'CPU_adapter_saved':False,'base_weights_saved':False,'steps':[],'max_seconds':limit}
    write_json(out/'report.json',report)
    def timeout(*args):raise TimeoutError('Bounded teacher LoRA stage time reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(limit)
    try:
        with nullcontext() if cpu else cloud_gpu(ROOT,cloud):
            import torch,peft
            from transformers import Qwen3VLForConditionalGeneration
            torch.set_num_threads(8 if cpu else 4);torch.set_num_interop_threads(1);torch.manual_seed(spec['seed'])
            directory=ROOT/spec['model_dir'];processor,identity=checked_processor(ROOT,spec,cloud);report['identity']=identity
            data=ROOT/spec['data_root'];receipt=read_json(data/'receipt.json');source=data/'teacher_train.instructions.jsonl'
            if sha256(source)!=receipt['partitions']['teacher_train']['instruction_sha256']:raise ProjectError('Training instructions changed')
            records=[json.loads(line) for line in source.open()]
            if len(records)!=spec['train_instruction_examples'] or any(r['teacher_partition']!='teacher_train' for r in records):raise ProjectError('Wrong teacher training partition')
            order=list(range(len(records)));random.Random(spec['seed']).shuffle(order)
            report['instruction_sha256']=sha256(source);report['instruction_order_sha256']=__import__('hashlib').sha256(json.dumps(order).encode()).hexdigest()
            dtype=torch.float32 if fp32 else torch.float16
            model=Qwen3VLForConditionalGeneration.from_pretrained(directory,local_files_only=True,trust_remote_code=False,dtype=dtype,attn_implementation='sdpa').to(a.device)
            targets=[n for n,m in model.named_modules() if 'language_model.layers.' in n and n.endswith(('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj'))]
            if len(targets)!=112:raise ProjectError('Expected 112 fixed language attention targets')
            lc=spec['lora'];model=peft.get_peft_model(model,peft.LoraConfig(r=lc['rank'],lora_alpha=lc['alpha'],lora_dropout=lc['dropout'],bias='none',target_modules=targets,task_type='CAUSAL_LM'))
            trainable=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
            if any('lora_' not in n or 'language_model' not in n for n,_ in trainable):raise ProjectError('Unexpected trainable base/vision parameter')
            report['trainable_parameters']=sum(v.numel() for _,v in trainable);report['dtype']=str(dtype)
            model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
            optimizer=torch.optim.AdamW([v for _,v in trainable],lr=spec['learning_rate'],weight_decay=spec['weight_decay'],foreach=False)
            scaler=torch.amp.GradScaler('cuda',enabled=not fp32,init_scale=1024.)
            start_update=0
            if a.resume_from:
                from safetensors.torch import load_file
                previous=output_path(ROOT,a.resume_from);prior=read_json(previous/'report.json')
                checkpoint=previous/'last-training-state.pt';adapter=previous/'last-adapter/adapter_model.safetensors'
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if checkpoint.stat().st_mtime<adapter.stat().st_mtime:raise ProjectError('Optimizer snapshot predates adapter')
                for key in ('identity','instruction_sha256','instruction_order_sha256'):
                    if state[key]!=report[key]:raise ProjectError('Resume identity/data/order mismatch: '+key)
                start_update=int(state['completed_updates'])
                if not 0<start_update<updates or start_update>prior['completed_updates']:raise ProjectError('Invalid recovered update count')
                if any(float(v['step'])!=start_update for v in state['optimizer']['state'].values()):raise ProjectError('Optimizer update counts differ')
                loaded=peft.set_peft_model_state_dict(model,load_file(str(adapter)))
                if loaded.unexpected_keys or any('lora_' in n for n in loaded.missing_keys):raise ProjectError('Adapter parameter mapping mismatch')
                optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
                torch.set_rng_state(state['torch_rng_state'])
                report.update(completed_updates=start_update,steps=prior['steps'][:start_update],
                    resume={'source':str(previous),'restored_update':start_update,'previous_logged_updates':prior['completed_updates'],
                            'checkpoint_sha256':sha256(checkpoint),'adapter_sha256':sha256(adapter),
                            'source_report_sha256':sha256(previous/'report.json'),'source_device':prior['device'],
                            'target_device':a.device,'precision_preserved':fp32 and prior['dtype']=='torch.float32',
                            'CPU_rng_restored':True,'CUDA_rng_seed':spec['seed'] if not cpu else None,
                            'cross_device_bitwise_replay_claimed':False})
            reference={n:v.detach().float().cpu().clone() for n,v in trainable};accum=spec['gradient_accumulation_steps']
            if updates*accum>len(records):raise ProjectError('Do not silently cycle instruction data')
            report.update(status='training',load_seconds=time.monotonic()-started);write_json(out/'report.json',report)
            if not cpu:torch.cuda.reset_peak_memory_stats()
            for step in range(start_update,updates):
                tick=time.monotonic();optimizer.zero_grad(set_to_none=True);losses=[];examples=[]
                ratio=min((step+1)/spec['warmup_updates'],(spec['optimizer_updates']-step)/max(1,spec['optimizer_updates']-spec['warmup_updates']))
                for group in optimizer.param_groups:group['lr']=spec['learning_rate']*ratio
                for micro in range(accum):
                    item=records[order[step*accum+micro]];inputs,positions,labels=instruction_inputs(processor,item,spec['max_sequence_length'])
                    inputs={k:v.to(a.device) for k,v in inputs.items()};pos=torch.tensor(positions,device=a.device);target=torch.tensor(labels,device=a.device)
                    with torch.autocast(device_type=a.device,dtype=torch.float16,enabled=not fp32):
                        logits=model(**inputs,logits_to_keep=pos,use_cache=False).logits[0]
                        if logits.shape[0]!=len(labels):raise ProjectError('Selected next-token logits do not align')
                        loss=torch.nn.functional.cross_entropy(logits.float(),target)
                    if not torch.isfinite(loss):raise ProjectError('Nonfinite semantic loss')
                    scaler.scale(loss/accum).backward();losses.append(float(loss.detach()));examples.append({'sample_id':item['sample_id'],'task':item['task']})
                    if not cpu and torch.cuda.max_memory_allocated()>spec['peak_allocated_memory_budget_GiB']*1024**3:raise ProjectError('Pilot GPU memory budget exceeded')
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_([v for _,v in trainable],spec['max_grad_norm'],error_if_nonfinite=True)
                if not any(v.grad is not None and bool(torch.any(v.grad!=0)) for _,v in trainable):raise ProjectError('No nonzero adapter gradient')
                scaler.step(optimizer);scaler.update()
                if step==start_update:
                    delta=sum(float((v.detach().float().cpu()-reference[n]).abs().sum()) for n,v in trainable)
                    if delta<=0:raise ProjectError('Optimizer did not change adapter parameters')
                    report['first_update']={'parameter_delta_L1':delta,'frozen_base_and_vision':True,'GPU_step_passed':not cpu};reference.clear()
                entry={'update':step+1,'mean_semantic_loss':sum(losses)/len(losses),'gradient_norm':float(norm),'seconds':time.monotonic()-tick,'examples':examples,'lr':optimizer.param_groups[0]['lr']}
                report['steps'].append(entry);report['completed_updates']=step+1;write_json(out/'report.json',report);print(json.dumps(entry),flush=True)
                if not cpu_check and (step+1)%25==0:
                    model.save_pretrained(out/'last-adapter',safe_serialization=True,save_embedding_layers=False)
                    state_path=out/'last-training-state.pt';temporary=out/'last-training-state.pt.partial'
                    torch.save({'completed_updates':step+1,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                                'torch_rng_state':torch.get_rng_state(),'instruction_sha256':report['instruction_sha256'],
                                'instruction_order_sha256':report['instruction_order_sha256'],'identity':identity},temporary)
                    temporary.replace(state_path);report['last_recoverable_update']=step+1;write_json(out/'report.json',report)
            if not cpu_check:
                model.save_pretrained(out/'final-adapter',safe_serialization=True,save_embedding_layers=False);adapter_saved=True
                report['CPU_adapter_saved']=cpu
                report['adapter_files']=[{'name':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha256(p)} for p in (out/'final-adapter').iterdir() if p.is_file()]
            report.update(status='completed',cuda_initialized=torch.cuda.is_initialized(),peak_gpu_bytes=torch.cuda.max_memory_allocated() if not cpu else 0)
    except BaseException as exc:
        report.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        if not cpu_check and model is not None and hasattr(model,'peft_config') and not adapter_saved:
            try:model.save_pretrained(out/'interrupted-adapter',safe_serialization=True,save_embedding_layers=False)
            except Exception as save_error:report['adapter_save_error']=str(save_error)
        raise
    finally:
        signal.alarm(0);report.update(seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write_json(out/'report.json',report)


if __name__=='__main__':main()
