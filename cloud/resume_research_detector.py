"""Resume the existing research detector, restoring fusion weights before optimizer/EMA state."""
from pathlib import Path
import argparse,csv,datetime,importlib.metadata,json,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.runtime import cloud_gpu
from fire_r2.train import training_plan,verify_backend_source,staged_checkpoint_amp_check
from fire_r2.stopping import ValidationProgress


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--cloud-config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=read_json(ROOT/a.config);cloud=read_json(ROOT/a.cloud_config);plan=training_plan(ROOT,config,cloud)
    run=ROOT/'runs'/config['run_name'];view=Path(plan['view']);out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve previous recovery receipt')
    stop_file=ROOT/'artifacts/run-controls'/(config['run_name']+'.stop')
    if stop_file.exists():raise ProjectError('This run was explicitly stopped; evaluate its best checkpoint instead of automatically resuming')
    info=read_json(view/'view.json')
    if info['view_manifest_sha256']!=sha256(view/'manifest.jsonl') or info['training_yaml_sha256']!=sha256(view/'train.yaml'):raise ProjectError('Frozen data view differs')
    for role in ('fit','dev'):
        if info['role_list_sha256'][role]!=sha256(view/(role+'.txt')) or any(not Path(p).is_file() for p in (view/(role+'.txt')).read_text().splitlines()):raise ProjectError('Frozen image list differs/missing image')
    for line in (view/'manifest.jsonl').open():
        row=json.loads(line)
        if row['role'] in ('fit','dev') and sha256(view/row['view_label'])!=row['view_label_sha256']:raise ProjectError('Frozen derived label changed')
    verify_backend_source(importlib.metadata.distribution('ultralytics').read_text('direct_url.json'),config)
    record={'status':'preparing','original_run':config['run_name'],'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'CSV_time_resets_on_resume':True,'augmentation_rng_bitwise_replay_claimed':False};write_json(out,record)
    started=time.monotonic()
    try:
        with cloud_gpu(ROOT,cloud):
            import torch
            from ultralytics import YOLO
            from ultralytics.engine import trainer as engine
            from fire_r2.backend_lock import verify_backend
            from fire_r2.ultralytics_backend import ResearchTrainer,ResearchDetectionModel
            from fire_r2.unified_validator import UnifiedSelectionMixin
            from fire_r2.supervision import validate_supervision
            verify_backend(ROOT,config['ultralytics_commit']);validate_supervision(read_json(ROOT/config['supervision']),view/'manifest.jsonl',plan['factors'])
            if plan['factors'].get('teacher_joint_mix'):
                from fire_r2.teacher_transfer import validate_transfer
                from fire_r2.research_data import load_manifest
                validate_transfer(read_json(ROOT/config['supervision']),load_manifest(view/'manifest.jsonl',roles={'fit'},enrich=False),config['teacher_expected'])
            checkpoint=run/'weights/resume_last.pt';ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
            start=int(ckpt['epoch'])+1
            cap=int(config['train']['epochs'])
            if ckpt.get('optimizer') is None or not 0<start<cap:raise ProjectError('No resumable optimizer/epoch within the safety cap')
            if any(ckpt['train_args'][k]!=config['train'][k] for k in ('epochs','batch','imgsz')):raise ProjectError('Resume recipe differs')
            saved_model=ckpt['ema']
            if saved_model.research_config!=plan['factors']:raise ProjectError('Saved research factors differ')
            existing=list(csv.DictReader((run/'results.csv').open()))
            if len(existing)!=start:raise ProjectError('CSV/checkpoint epoch mismatch; preserve and diagnose')
            prior_reports=[read_json(p) for p in sorted((run/'original-dev').glob('epoch-*.json'))]
            winner=max(prior_reports,key=lambda r:(r['evaluation']['classes']['fire']['AP50_95'],-r['epoch']))
            best_fire=float(winner['evaluation']['classes']['fire']['AP50_95'])
            policy=config.get('validation_stopping',{'patience_epochs':10,'min_delta':.001});progress=ValidationProgress(policy['patience_epochs'],policy['min_delta'])
            for row in prior_reports:progress.observe(row['epoch']+1,row['evaluation']['classes']['fire']['AP50_95'])
            if progress.last_index-progress.last_improvement>=progress.patience:
                record.update(status='early_stopped',stop_reason='validation_plateau_already_present',completed_epochs=start,best_fire_ap=best_fire,validation_progress=progress.snapshot())
                return
            backup=ROOT/'artifacts/recovery-20261109/student-checkpoints'/out.stem;backup.mkdir(parents=True,exist_ok=True)
            for name in ('resume_last.pt','best_fire.pt'):
                if (backup/name).exists():raise ProjectError('Recovery checkpoint backup exists')
                shutil.copy2(run/'weights'/name,backup/name)
            record.update(restored_epoch=start,checkpoint_sha256=sha256(checkpoint),prior_best_fire_ap=best_fire,
                          prior_csv_sha256=sha256(run/'results.csv'),status='resuming')
            write_json(out,record)
            expected={k:v.float().clone() for k,v in saved_model.state_dict().items()}
            class ResumeTrainer(UnifiedSelectionMixin,ResearchTrainer):
                def get_model(self,cfg=None,weights=None,verbose=True):
                    model=self.set_model_names_for_load(ResearchDetectionModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose))
                    model.configure_research(self.research_config)
                    if weights:model.load(weights)
                    return model
                def resume_training(self,checkpoint):
                    super().resume_training(checkpoint)
                    self.train_loader.batch_sampler.epoch=self.start_epoch
            ResumeTrainer.research_config=plan['factors'];ResumeTrainer.supervision_file=str(ROOT/config['supervision']);ResumeTrainer.manifest_file=str(view/'manifest.jsonl');ResumeTrainer._r2_saved_best=round(best_fire,5)
            model=YOLO(str(checkpoint),task='detect')
            def ready(trainer):
                actual=trainer.model.state_dict()
                if set(actual)!=set(expected) or any(not torch.equal(actual[k].detach().cpu().float(),v) for k,v in expected.items()):raise ProjectError('Resume did not restore all base/fusion tensors')
                if trainer.start_epoch!=start or trainer.train_loader.batch_sampler.epoch!=start:raise ProjectError('Epoch/sampler resume offset differs')
                if bool(trainer.amp)!=bool(config.get('amp_enabled')):raise ProjectError('Resume AMP mode differs')
                for role,loader in [('fit',trainer.train_loader),('dev',trainer.test_loader)]:
                    if sorted(Path(p).name for p in (view/(role+'.txt')).read_text().splitlines())!=sorted(Path(p).name for p in loader.dataset.im_files):raise ProjectError('Resumed image coverage differs')
                record.update(status='running',all_base_and_fusion_tensors_restored=True,sampler_epoch=start,actual_amp=bool(trainer.amp));write_json(out,record);expected.clear()
            def saved(trainer):
                nonlocal best_fire
                shutil.copy2(trainer.last,run/'weights/resume_last.pt')
                score=float(trainer.validator.original_report['classes']['fire']['AP50_95'])
                if score>best_fire:best_fire=score;shutil.copy2(trainer.last,run/'weights/best_fire.pt')
                if progress.observe(trainer.epoch+1,score):trainer.stop=True;record['stop_reason']='validation_plateau'
                record['validation_progress']=progress.snapshot()
                record.update(last_completed_epoch=trainer.epoch+1,best_fire_ap=best_fire);write_json(out,record)
            def end_epoch(trainer):
                if stop_file.exists() or time.monotonic()-started>8*3600:
                    trainer.stop=True;record['stop_reason']='user_request' if stop_file.exists() else 'time_budget'
            model.add_callback('on_pretrain_routine_end',ready);model.add_callback('on_model_save',saved);model.add_callback('on_train_epoch_end',end_epoch)
            old_amp=engine.check_amp;engine.check_amp=lambda current:staged_checkpoint_amp_check(ROOT/config['weights'],current)
            try:model.train(trainer=ResumeTrainer,resume=True,device=0,patience=0)
            finally:engine.check_amp=old_amp
            count=len(list(csv.DictReader((run/'results.csv').open())));record.update(status='completed' if count>=cap else 'early_stopped',completed_epochs=count,best_fire_ap=best_fire)
            record.setdefault('stop_reason','epoch_cap' if count>=cap else 'native_validation_early_stop')
    except BaseException as exc:record.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:record['seconds']=time.monotonic()-started;write_json(out,record)


if __name__=='__main__':main()
