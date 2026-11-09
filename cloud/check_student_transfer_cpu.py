"""Small real-student forward/backward integration, synthetic inputs, no research training result."""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,sha256,ProjectError
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('Cloud CPU integration requires CUDA invisibility')
    out=ROOT/'artifacts/teacher-transfer-20261109/student-cpu-integration.json'
    if out.exists():raise ProjectError('Preserve prior integration evidence')
    report={'status':'starting','synthetic_inputs':True,'publication_effectiveness_result':False,'variants':[]};started=time.monotonic()
    try:
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import DEFAULT_CFG_DICT,IterableSimpleNamespace
        from fire_r2.experiments import factor_config
        from fire_r2.ultralytics_backend import ResearchDetectionModel
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        cfg=read_json(ROOT/'configs/train_dfire_P_control_20261109.json');checkpoint=ROOT/cfg['weights']
        if sha256(checkpoint)!=cfg['weights_sha256']:raise ProjectError('Existing student initialization changed')
        pretrained=YOLO(str(checkpoint)).model;torch.manual_seed(101)
        common={'img':torch.rand(4,3,128,128),'batch_idx':torch.tensor([1.,2.,3.,3.]),'cls':torch.tensor([[1.],[0.],[0.],[1.]]),
                'bboxes':torch.tensor([[.5,.5,.4,.4],[.5,.5,.4,.4],[.3,.4,.2,.3],[.7,.5,.2,.3]]),
                'r2_clean':torch.ones(4,dtype=torch.bool),'r2_weights':torch.ones(4),'r2_mask':torch.ones(4,6,dtype=torch.bool),'r2_warmup':1.}
        presence=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]]);truth=torch.eye(4)
        distributions={'P_control':truth,'T_base':.75*truth+.25*torch.tensor([[.1,.2,.3,.4]]*4),'T_adapted':.75*truth+.25*(.8*truth+.05)}
        hashes=[];detection=[];grads=[]
        for name,joint in distributions.items():
            torch.manual_seed(17);model=ResearchDetectionModel(pretrained.yaml,nc=2,verbose=False);model.load(pretrained)
            model.configure_research(factor_config('B+JP' if name=='P_control' else 'B+TP'));model.args=IterableSimpleNamespace(**{**DEFAULT_CFG_DICT,**cfg['train']});model.train()
            digest=hashlib.sha256()
            for key,value in model.state_dict().items():digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
            hashes.append(digest.hexdigest());batch=copy.deepcopy(common);batch['r2_targets']=torch.cat([presence,joint],dim=1)
            loss,items=model(batch)
            if not torch.isfinite(loss):raise ProjectError('Nonfinite student integration loss')
            loss.backward();gradient=model.research_semantic.output.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or not torch.any(gradient[2:]!=0):raise ProjectError('Joint auxiliary head gradient is missing')
            grads.append(gradient.clone());detection.append([float(items[k]) for k in ['box_loss','cls_loss','l1_loss']])
            model.eval()
            with torch.no_grad():before=model(common['img'])[0].clone();model.strip_training_heads();after=model(common['img'])[0]
            if not torch.equal(before,after):raise ProjectError('Training-head removal changed inference')
            report['variants'].append({'name':name,'loss':float(loss.detach()),'loss_items':{k:float(v) for k,v in items.items()},'joint_gradient_norm':float(gradient[2:].norm()),'deployment_output_unchanged':True})
            del model,batch,loss,items,before,after
        if len(set(hashes))!=1 or any(max(abs(a-b) for a,b in zip(detection[0],x))>1e-6 for x in detection[1:]):raise ProjectError('Control initialization or original detection loss differs')
        if any(torch.allclose(grads[0][2:],x[2:]) for x in grads[1:]):raise ProjectError('Teacher targets did not affect joint-head gradients')
        report.update(status='passed',same_initial_state_sha256=hashes[0],original_detection_loss_equal=True,teacher_changes_joint_gradients=True,cuda_initialized=torch.cuda.is_initialized())
    except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:report['seconds']=time.monotonic()-started;write_json(out,report);print(json.dumps(report))


if __name__=='__main__':main()
