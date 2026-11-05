"""Authorized GPU integration check; synthetic inputs are not research results."""
from pathlib import Path
import argparse,json,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,sha256,write_json,output_path
from fire_r2.runtime import cloud_gpu


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True)
    p.add_argument('--output',default='artifacts/methods-20261105/gpu-preflight.json');a=p.parse_args()
    cloud=read_json(ROOT/a.cloud_config);base=read_json(ROOT/'configs/train_dfire_full.json')
    checkpoint=ROOT/base['weights'];out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preflight output exists; preserve the previous result')
    report={'status':'starting','synthetic_inputs':True,'publication_effectiveness_result':False,
            'weights_sha256':base['weights_sha256'],'variants':[],
            'source_sha256':{p.name:sha256(p) for p in (ROOT/'src/fire_r2').glob('*.py')}}
    started=time.monotonic()
    try:
        with cloud_gpu(ROOT,cloud):
            from fire_r2.backend_lock import verify_backend
            verify_backend(ROOT,base['ultralytics_commit'])
            if sha256(checkpoint)!=base['weights_sha256']:raise ProjectError('Staged detector checkpoint changed')
            import torch
            from ultralytics import YOLO
            from ultralytics.utils import DEFAULT_CFG_DICT,IterableSimpleNamespace
            from fire_r2.experiments import factor_config
            from fire_r2.ultralytics_backend import ResearchDetectionModel
            pretrained=YOLO(str(checkpoint)).model
            report.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,status='running')
            for variant in ('B','B+W','B+P','B+S','M-ordinary','M','M+W+S'):
                torch.manual_seed(17);torch.cuda.manual_seed_all(17)
                factors=factor_config(variant)
                model=ResearchDetectionModel(pretrained.yaml,nc=2,verbose=False)
                model.load(pretrained);model.configure_research(factors)
                model.args=IterableSimpleNamespace(**{**DEFAULT_CFG_DICT,**base['train']})
                model=model.cuda().train();optimizer=torch.optim.SGD(model.parameters(),lr=.001)
                scaler=torch.amp.GradScaler('cuda');torch.cuda.reset_peak_memory_stats()
                batch={'img':torch.rand(2,3,128,128,device='cuda'),
                       'batch_idx':torch.tensor([0.,1.],device='cuda'),'cls':torch.tensor([[0.],[1.]],device='cuda'),
                       'bboxes':torch.tensor([[.5,.5,.4,.4],[.3,.3,.2,.2]],device='cuda'),
                       'r2_clean':torch.tensor([True,True],device='cuda'),
                       'r2_weights':torch.tensor([.8,1.2] if factors['weighting']!='uniform' else [1.,1.],device='cuda'),
                       'r2_targets':torch.tensor([[1,0,0,1,0,0],[0,1,1,0,0,1]],device='cuda',dtype=torch.float32),
                       'r2_mask':torch.ones(2,6,device='cuda',dtype=torch.bool),'r2_warmup':1.}
                losses=[];tick=time.monotonic();overflows=0
                for attempt in range(12):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):loss,items=model(batch)
                    if not torch.isfinite(loss):raise ProjectError(f'Nonfinite {variant} loss')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    # Follow native GradScaler overflow recovery: skip the update
                    # and reduce the scale, rather than misclassifying warmup as
                    # a successful optimizer step or silently accepting NaNs.
                    scaler.step(optimizer);scaler.update()
                    if finite:losses.append(float(loss.detach()))
                    else:overflows+=1
                    if len(losses)==2:break
                if len(losses)!=2:raise ProjectError(f'AMP gradients did not stabilize for {variant}')
                model.eval()
                with torch.no_grad():before=model(batch['img'])[0].clone()
                model.strip_training_heads()
                with torch.no_grad():after=model(batch['img'])[0]
                if not torch.equal(before,after):raise ProjectError('Removing training-only heads changed deployment output')
                torch.cuda.synchronize()
                report['variants'].append({'variant':variant,'optimizer_steps':2,'losses':losses,
                    'loss_items':{k:float(v) for k,v in items.items()},'finite_gradients':True,'amp':True,
                    'amp_skipped_overflow_steps':overflows,'final_grad_scale':scaler.get_scale(),
                    'deployment_output_unchanged':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'seconds':time.monotonic()-tick})
                write_json(out,report);print(json.dumps(report['variants'][-1]),flush=True)
                del model,optimizer,scaler,batch,before,after,loss,items
                torch.cuda.empty_cache()
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;write_json(out,report)


if __name__=='__main__':main()
