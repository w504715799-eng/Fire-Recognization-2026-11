"""Wait only for our already-running baseline, then run bounded authorized checks.

No rental, reboot, upgrade, remote notification or automatic full-factor sweep.
"""
from pathlib import Path
import argparse,csv,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,child,file_lock,read_json,write_json,output_path
from fire_r2.runtime import check_remote_environment
from fire_r2.cloud import now


def baseline_process_exists(config_name):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if config_name.encode() in args and b'train' in args and (proc/'cwd').resolve()==ROOT:return True
        except (OSError,PermissionError):pass
    return False


def own_command_exists(tokens):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if all(token.encode() in args for token in tokens) and (proc/'cwd').resolve()==ROOT:return True
        except (OSError,PermissionError):pass
    return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    cloud=read_json(ROOT/a.cloud_config);cfg=read_json(ROOT/a.config)
    check_remote_environment(ROOT,cloud)
    dest=output_path(ROOT,cfg.get('status_output','artifacts/methods-20261105/next-stage-status.json'))
    queue_id=cfg.get('queue_id','next-stage')
    if not queue_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in queue_id):
        raise ProjectError('Invalid queue ID')
    record={'status':'waiting_for_own_baseline','started_at':now(),'baseline':cfg['baseline_run'],'stages':[]}
    started=time.monotonic()
    with file_lock(ROOT/'artifacts/.locks'/(queue_id+'.lock')):
        if dest.exists():raise ProjectError('Queue status exists; inspect before restarting')
        write_json(dest,record)
        try:
            while True:
                check_remote_environment(ROOT,cloud)
                if time.monotonic()-started>float(cfg.get('max_wait_hours',6))*3600:
                    raise ProjectError('Own-baseline waiting bound reached')
                if cfg.get('prerequisite_queue'):
                    prerequisite=read_json(ROOT/cfg['prerequisite_queue'])
                    if prerequisite['status']!='completed':
                        if prerequisite['status'] not in ('waiting_for_own_baseline','validating'):
                            raise ProjectError('Prerequisite queue needs attention')
                        record.update(last_checked_at=now(),prerequisite_queue_status=prerequisite['status']);write_json(dest,record)
                        time.sleep(30)
                        continue
                waiting=[]
                for prerequisite in cfg.get('prerequisite_reports',[]):
                    path=ROOT/prerequisite['path'];previous=read_json(path) if path.exists() else {'status':'not_started'}
                    if previous.get('status')=='failed':raise ProjectError('Prerequisite experiment failed: '+prerequisite['path'])
                    if previous.get('status')!='completed':waiting.append(prerequisite['path']);continue
                    if previous.get('completed_updates')!=prerequisite.get('completed_updates'):
                        raise ProjectError('Prerequisite did not complete the fixed update count')
                    if prerequisite.get('process_tokens') and own_command_exists(prerequisite['process_tokens']):waiting.append(prerequisite['path'])
                if waiting:
                    record.update(last_checked_at=now(),waiting_for_reports=waiting);write_json(dest,record)
                    time.sleep(30)
                    continue
                run=ROOT/'runs'/cfg['baseline_run'];csv_file=run/'results.csv'
                epochs=len(list(csv.DictReader(csv_file.open()))) if csv_file.exists() else 0
                report_path=ROOT/'artifacts/run-records'/(cfg['baseline_run']+'.json')
                report=read_json(report_path) if report_path.exists() else {}
                record.update(last_checked_at=now(),completed_epochs=epochs);write_json(dest,record)
                if report.get('status')=='completed_or_budget_stopped' and epochs>=cfg['expected_epochs']:
                    if baseline_process_exists(cfg['baseline_config']):
                        time.sleep(5)
                        continue  # Let our completed process release CUDA before a successor takes the GPU.
                    # Nonblocking ownership check: if another service took the GPU, stop.
                    with file_lock(Path(cloud['shared_gpu_lock_root'])/f"gpu-{cloud['gpu_id']}.lock"):pass
                    break
                if report.get('status') in ('failed_or_interrupted','completed_or_budget_stopped') or not baseline_process_exists(cfg['baseline_config']):
                    raise ProjectError('Baseline needs attention; no GPU work was started')
                if time.monotonic()-started>float(cfg.get('max_wait_hours',6))*3600:
                    raise ProjectError('Own-baseline waiting bound reached')
                time.sleep(30)
            record.update(status='validating',gpu_stage_started_at=now());write_json(dest,record)
            for stage in cfg['stages']:
                log=child(ROOT,stage['log']);log.parent.mkdir(parents=True,exist_ok=True)
                args=[sys.executable,'-B',*stage['args']]
                stamp={'name':stage['name'],'started_at':now(),'log':stage['log']};record['stages'].append(stamp);write_json(dest,record)
                try:
                    with log.open('x') as stream:
                        timeout=stage.get('timeout_seconds')
                        result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                              timeout=float(timeout) if timeout is not None else None)
                    stamp.update(finished_at=now(),returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    stamp.update(finished_at=now(),returncode=None,error='stage_timeout')
                if stamp['returncode']==0 and stage.get('expected_training_epochs'):
                    csv_path=ROOT/'runs'/stage['run_name']/'results.csv'
                    completed=len(list(csv.DictReader(csv_path.open()))) if csv_path.exists() else 0
                    stamp['completed_epochs']=completed
                    stamp['expected_epochs']=stage['expected_training_epochs']
                    if completed<stage['expected_training_epochs']:
                        stamp['error']='training_stopped_before_expected_epochs'
                        write_json(dest,record)
                        raise ProjectError('Training checkpoint retained; expected epoch count not completed')
                write_json(dest,record)
                if stamp['returncode']!=0 and not stage.get('continue_on_failure',False):
                    raise ProjectError('Stage failed; preserve logs and diagnose: '+stage['name'])
            record.update(status='completed_with_stage_failures' if any(s['returncode']!=0 for s in record['stages']) else 'completed',finished_at=now())
        except BaseException as exc:
            record.update(status='needs_attention',error=f'{type(exc).__name__}: {exc}',finished_at=now())
            raise
        finally:
            record['server_powered_off']=False;write_json(dest,record)


if __name__=='__main__':main()
