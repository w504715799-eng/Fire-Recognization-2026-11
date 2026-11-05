"""Explicit cloud-only resume of the existing Home-fire baseline, not a new run."""
from pathlib import Path
import datetime
import os
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,sha256,file_lock,ProjectError
from fire_r2.cloud import require_remote
from fire_r2.train import verify_backend_source,offline_connector
import importlib.metadata


def main():
    name='homefire_B_native_seed17_full_20261105'
    run=ROOT/'runs'/name
    cloud=read_json(ROOT/'configs/cloud.homefire_full_20261105.local.json')
    require_remote(ROOT,cloud,'training')
    if Path(sys.prefix).resolve()!=Path(cloud['environment_prefix']).resolve():
        raise ProjectError('Use the existing private cloud sci-env')
    launched=datetime.datetime.fromisoformat(read_json(ROOT/'artifacts/run-records'/f'{name}.launch.json')['launched_at'])
    deadline=launched+datetime.timedelta(hours=4)
    if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise ProjectError('Original stage budget elapsed; revise the concrete run budget first')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt=ROOT/'artifacts/run-records'/f'{name}.resume-{stamp}.json'
    view=ROOT/'artifacts/data/homefire-v1';meta=read_json(view/'view.json')
    if not meta['training_ready'] or meta['training_yaml_sha256']!=sha256(view/'train.yaml') or meta['view_manifest_sha256']!=sha256(view/'manifest.jsonl'):
        raise ProjectError('Prepared data changed or not ready')
    for role in ('fit','dev'):
        if meta['role_list_sha256'][role]!=sha256(view/f'{role}.txt'):raise ProjectError('Role list changed')
        if any(not Path(p).is_file() for p in (view/f'{role}.txt').read_text().splitlines()):raise ProjectError('An image is missing')
    config=read_json(ROOT/'configs/train_homefire_full_20261105.json')
    verify_backend_source(importlib.metadata.distribution('ultralytics').read_text('direct_url.json'),config)
    with file_lock(Path(cloud['shared_gpu_lock_root'])/'gpu-0.lock'):
        if subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            raise ProjectError('GPU already occupied; no competing resume')
        os.environ.update(CUDA_VISIBLE_DEVICES='0',YOLO_OFFLINE='true',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        original_connect=socket.socket.connect;socket.socket.connect=offline_connector(original_connect)
        started=time.monotonic()
        record={'status':'initializing_resume','original_run':name,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'deadline':deadline.isoformat(),'source':'User explicitly requested resume after server restart','server_powered_off':False}
        write_json(receipt,record)
        try:
            import torch
            from ultralytics import YOLO
            if not torch.cuda.is_available():raise ProjectError('CUDA unavailable')
            last=run/'weights/last.pt'
            record['resume_checkpoint_sha256']=sha256(last)
            model=YOLO(str(last),task='detect')
            ckpt=model.ckpt
            epoch=int(ckpt['epoch']);args=ckpt['train_args']
            if epoch<0 or ckpt.get('optimizer') is None:raise ProjectError('Checkpoint lacks resumable optimizer/epoch state')
            if int(args['epochs'])!=100 or str(args['data'])!=str(view/'train.yaml'):
                raise ProjectError('Checkpoint training protocol differs from the requested run')
            record.update(completed_epochs=epoch+1,resume_human_epoch=epoch+2,optimizer_state_present=True)
            shutil.copy2(last,run/'weights'/f'last.before-resume-{stamp}.pt')
            prior=run/'weights/best_fire.pt'
            shutil.copy2(prior,run/'weights'/f'best_fire.before-resume-{stamp}.pt')
            # Restore the comparison score on the same dev set; never consult cal/test.
            prior_model=YOLO(str(prior),task='detect')
            metrics=prior_model.val(data=str(view/'train.yaml'),split='val',device=0,imgsz=640,batch=16,
                workers=2,half=False,plots=False,verbose=False,project=str(ROOT/'artifacts/resume-validation'),name=stamp)
            best_fire=float(metrics.box.maps[0]);record['prior_best_fire_dev_ap_rechecked']=best_fire
            del prior_model,metrics;torch.cuda.empty_cache()
            def saved(trainer):
                nonlocal best_fire
                score=float(trainer.validator.metrics.box.maps[0])
                if score>best_fire:
                    best_fire=score;shutil.copy2(trainer.last,run/'weights/best_fire.pt')
                record.update(status='running',last_completed_epoch=int(trainer.epoch)+1,best_fire_dev_ap=best_fire)
                write_json(receipt,record)
            def epoch_end(trainer):
                if datetime.datetime.now(datetime.timezone.utc)>=deadline:trainer.stop=True
            model.add_callback('on_model_save',saved);model.add_callback('on_train_epoch_end',epoch_end)
            record['status']='resuming';write_json(receipt,record)
            print(f'RESUME_CONFIRMED: completed {epoch+1} epochs; continue at epoch {epoch+2}/100 with optimizer state',flush=True)
            model.train(resume=True,device=0)
            record['status']='completed_or_budget_stopped'
        except BaseException as exc:
            record.update(status='failed_or_interrupted',error=repr(exc));raise
        finally:
            record.update(elapsed_seconds=time.monotonic()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(receipt,record);socket.socket.connect=original_connect


if __name__=='__main__':main()
