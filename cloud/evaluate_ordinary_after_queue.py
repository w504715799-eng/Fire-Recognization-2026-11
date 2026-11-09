"""Run the already-planned ordinary-fusion CPU evaluation while the next GPU control trains."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,file_lock,ProjectError
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config));out=ROOT/'artifacts/student-transfer-20261109/ordinary-cpu-eval-status.json'
    with file_lock(ROOT/'artifacts/.locks/ordinary-cpu-eval-20261109.lock'):
        if out.exists():raise ProjectError('Inspect existing CPU evaluation job')
        report={'status':'waiting_for_ordinary','GPU_lock_acquired':False};started=time.monotonic();write_json(out,report)
        try:
            while time.monotonic()-started<8*3600:
                prior=read_json(ROOT/'artifacts/recovery-20261109/queue-status.json')
                if prior['status']=='completed':break
                if prior['status']=='needs_attention':raise ProjectError('Ordinary queue needs attention')
                time.sleep(30)
            else:raise TimeoutError('Bounded ordinary evaluation dependency wait')
            report['status']='evaluating_cpu';write_json(out,report)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            with (ROOT/'logs/ordinary-final-cpu-evaluation-20261109.log').open('x') as log:
                subprocess.run([sys.executable,'-B','cloud/evaluate_student_control.py','--train-config','configs/train_dfire_M_ordinary_20261106.json',
                    '--cloud-config',a.cloud_config,'--device','cpu','--output','artifacts/evaluation-20261109/ordinary-final'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3900)
            paths={'B':'artifacts/evaluation-cpu-20261106/paired-B-dev-metrics.json','M':'artifacts/evaluation-20261109/M-dev-metrics.json','ordinary':'artifacts/evaluation-20261109/ordinary-final/dev-metrics.json'}
            metrics={name:read_json(ROOT/path) for name,path in paths.items()}
            if len({r['selection_sha256'] for r in metrics.values()})!=1:raise ProjectError('Common CPU comparison image sets differ')
            write_json(ROOT/'artifacts/evaluation-20261109/B-M-ordinary-comparison.json',{'scope':'same source dev, CPU FP32; model-specific source-cal thresholds','reports':paths,
                'models':{name:{k:r[k] for k in ['AP','small640_AP','fire_at_source_cal_threshold','groups','checkpoint_sha256']} for name,r in metrics.items()}})
            report['status']='completed'
        except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
        finally:report['elapsed_including_wait']=time.monotonic()-started;write_json(out,report)


if __name__=='__main__':main()
