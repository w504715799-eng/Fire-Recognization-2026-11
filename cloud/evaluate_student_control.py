"""Complete source dev/cal evaluation for a completed fixed student run; no test-set tuning."""
from pathlib import Path
import argparse,csv,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,sha256,output_path,ProjectError
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-config',required=True);p.add_argument('--cloud-config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config));cfg=read_json(ROOT/a.train_config);out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve earlier complete predictions/evaluations')
    out.mkdir(parents=True);run=ROOT/'runs'/cfg['run_name'];rows=list(csv.DictReader((run/'results.csv').open()))
    if len(rows)!=100:raise ProjectError('Evaluate the planned complete 100-epoch run')
    weights=run/'weights/best_fire.pt';manifest=ROOT/cfg['prepared_view']/'manifest.jsonl';report={'status':'running','run':cfg['run_name'],'checkpoint_sha256':sha256(weights),'device':a.device,'test_accessed':False}
    started=time.monotonic();write_json(out/'status.json',report)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    if a.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    def execute(args,name):
        with (out/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable,'-B','code/fire_r2_cli.py',*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
    try:
        for role in ('dev','cal'):
            spec={'weights':str(weights.relative_to(ROOT)),'weights_sha256':report['checkpoint_sha256'],
                  'manifest':str(manifest.relative_to(ROOT)),'ultralytics_commit':cfg['ultralytics_commit'],
                  'training_source':cfg['dataset'],'target_source':cfg['dataset'],'seed':cfg['seed'],'device':a.device,
                  'cpu_threads':4,'batch':4,'half':False,'purpose':role,'output':str((out/(role+'-predictions.json')).relative_to(ROOT))}
            path=out/(role+'-config.json');write_json(path,spec)
            execute(['predict','--config',str(path.relative_to(ROOT)),'--cloud-config',a.cloud_config,'--execute'],role+'-predict')
        calibration=out/'calibration.json';metrics=out/'dev-metrics.json'
        execute(['calibrate','--manifest',str(manifest.relative_to(ROOT)),'--predictions',str((out/'cal-predictions.json').relative_to(ROOT)),'--output',str(calibration.relative_to(ROOT))],'calibrate')
        execute(['evaluate','--manifest',str(manifest.relative_to(ROOT)),'--predictions',str((out/'dev-predictions.json').relative_to(ROOT)),'--calibration',str(calibration.relative_to(ROOT)),'--output',str(metrics.relative_to(ROOT))],'evaluate')
        report.update(status='completed',metrics_sha256=sha256(metrics),calibration_sha256=sha256(calibration))
    except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:report['seconds']=time.monotonic()-started;write_json(out/'status.json',report)


if __name__=='__main__':main()
