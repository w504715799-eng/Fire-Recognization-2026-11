"""Wait only for our already-running baseline, then run bounded authorized checks.

No rental, reboot, upgrade, remote notification or automatic full-factor sweep.
"""
from pathlib import Path
import argparse,csv,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,child,file_lock,read_json,write_json
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);p.add_argument('--config',required=True);a=p.parse_args()
    cloud=read_json(ROOT/a.cloud_config);cfg=read_json(ROOT/a.config)
    check_remote_environment(ROOT,cloud)
    dest=ROOT/'artifacts/methods-20261105/next-stage-status.json'
    record={'status':'waiting_for_own_baseline','started_at':now(),'baseline':cfg['baseline_run'],'stages':[]}
    started=time.monotonic()
    with file_lock(ROOT/'artifacts/.locks/next-stage.lock'):
        if dest.exists():raise ProjectError('Queue status exists; inspect before restarting')
        write_json(dest,record)
        try:
            while True:
                check_remote_environment(ROOT,cloud)
                run=ROOT/'runs'/cfg['baseline_run'];csv_file=run/'results.csv'
                epochs=len(list(csv.DictReader(csv_file.open()))) if csv_file.exists() else 0
                report_path=ROOT/'artifacts/run-records'/(cfg['baseline_run']+'.json')
                report=read_json(report_path) if report_path.exists() else {}
                record.update(last_checked_at=now(),completed_epochs=epochs);write_json(dest,record)
                if report.get('status')=='completed_or_budget_stopped' and epochs>=cfg['expected_epochs']:
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
                with log.open('x') as stream:
                    result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                          timeout=float(stage['timeout_seconds']))
                stamp.update(finished_at=now(),returncode=result.returncode);write_json(dest,record)
                if result.returncode:raise ProjectError('Stage failed; preserve logs and diagnose: '+stage['name'])
            record.update(status='completed',finished_at=now())
        except BaseException as exc:
            record.update(status='needs_attention',error=f'{type(exc).__name__}: {exc}',finished_at=now())
            raise
        finally:
            record['server_powered_off']=False;write_json(dest,record)


if __name__=='__main__':main()
