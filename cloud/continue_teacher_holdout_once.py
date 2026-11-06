"""One finite continuation of the current CPU 2B holdout if its first time budget expires."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,file_lock
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    source=ROOT/'artifacts/teacher-heldout-20261106/base-2b/report.json'
    out=ROOT/'artifacts/teacher-heldout-20261106/continuation-status.json'
    with file_lock(ROOT/'artifacts/.locks/teacher-heldout-2b-continuation.lock'):
        if out.exists():raise ProjectError('Continuation already scheduled; inspect existing status')
        record={'status':'waiting_for_current_segment','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'max_extra_seconds':2700,'max_continuations':1,'changes_model_or_data':False}
        started=time.monotonic();write_json(out,record)
        try:
            while time.monotonic()-started<7200:
                prior=read_json(source)
                if prior['status']=='completed':record['status']='not_needed_completed';return
                if prior['status']=='failed':
                    error=prior.get('error','').lower()
                    if not ('timed out' in error or 'stage budget' in error):raise ProjectError('Non-timeout failure needs diagnosis: '+str(prior.get('error')))
                    break
                time.sleep(20)
            else:raise ProjectError('Current segment did not terminate in bounded observation')
            # The terminal report is written in finally just before interpreter cleanup.
            time.sleep(5)
            record.update(status='continuing_once',previous_queries=len(prior['responses']),previous_error=prior.get('error'));write_json(out,record)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            log=ROOT/'logs/teacher-heldout-base-2b-continuation-20261106.log'
            with log.open('x') as stream:
                r=subprocess.run([sys.executable,'-B','-u','cloud/evaluate_teacher_holdout_cpu.py',
                    '--cloud-config',a.cloud_config,'--model','2b','--threads','4','--max-seconds','2700',
                    '--resume-report',str(source.relative_to(ROOT)),
                    '--output','artifacts/teacher-heldout-20261106/base-2b-continuation'],cwd=ROOT,env=env,
                    stdout=stream,stderr=subprocess.STDOUT,timeout=2820)
            record.update(status='completed' if r.returncode==0 else 'needs_attention',returncode=r.returncode)
        except BaseException as exc:record.update(status='needs_attention',error=str(exc));raise
        finally:
            record['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();write_json(out,record)


if __name__=='__main__':main()
