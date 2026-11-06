"""Run two disjoint whole-batch CPU remainder workers and merge all 512 original queries."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,file_lock,sha256
from fire_r2.runtime import check_remote_environment
from fire_r2.teacher_holdout_resume import merge_remainders


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    out=ROOT/'artifacts/teacher-heldout-20261106/base-2b-parallel'
    with file_lock(ROOT/'artifacts/.locks/teacher-heldout-2b-parallel.lock'):
        if out.exists():raise ProjectError('Parallel remainder already exists')
        out.mkdir();source=ROOT/'artifacts/teacher-heldout-20261106/base-2b/report.json';prior=read_json(source)
        reschedule=read_json(ROOT/'artifacts/teacher-heldout-20261106/parallel-reschedule.json')
        if prior['status']!='failed' or reschedule.get('terminal_report_sha256')!=sha256(source):raise ProjectError('Expected hash-bound controlled interruption receipt')
        record={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'source_report_sha256':sha256(source),'reused_queries':len(prior['responses']),'workers':[]}
        write_json(out/'status.json',record);processes=[];streams=[];started=time.monotonic()
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            for index in (0,1):
                folder=out/('worker-'+str(index));log=ROOT/('logs/teacher-heldout-2b-parallel-'+str(index)+'-20261106.log')
                stream=log.open('x');streams.append(stream)
                cmd=[sys.executable,'-B','-u','cloud/evaluate_teacher_holdout_cpu.py','--cloud-config',a.cloud_config,
                     '--model','2b','--threads','4','--max-seconds','2700','--resume-report',str(source.relative_to(ROOT)),
                     '--allow-interrupted-resume','--shard-count','2','--shard-index',str(index),'--output',str(folder.relative_to(ROOT))]
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT);processes.append(process)
                record['workers'].append({'index':index,'pid':process.pid,'output':str(folder.relative_to(ROOT))})
            write_json(out/'status.json',record)
            while any(p.poll() is None for p in processes):
                if time.monotonic()-started>2820:raise TimeoutError('Parallel remainder controller time budget')
                time.sleep(10)
            if any(p.returncode for p in processes):raise ProjectError('A remainder worker failed; preserve both reports')
            reports=[read_json(out/('worker-'+str(i))/'report.json') for i in (0,1)]
            expected_rows=[json.loads(line) for line in (ROOT/'artifacts/qwen3vl-2b-preparation-20261106/instructions/teacher_holdout.instructions.jsonl').open()]
            merged=merge_remainders(prior,reports,expected_rows)
            if len(merged)!=512:raise ProjectError('Merged result does not cover complete 512-query holdout')
            final={**prior,'status':'completed','responses':merged,
                   'error':None,'traceback':None,'completed_queries':512,'resumed_as_two_disjoint_workers':True,
                   'source_report_sha256':sha256(source),'previous_segment_seconds':prior['seconds'],
                   'parallel_segment_seconds':time.monotonic()-started,'seconds':prior['seconds']+time.monotonic()-started,
                   'cuda_initialized':False,'peak_rss_kib':None,'worker_peak_rss_kib':[r['peak_rss_kib'] for r in reports],
                   'timing_note':'wall time sums original and parallel segments; per-query times retain actual worker timing; process RSS is not summed peak memory'}
            final['tasks']={task:{'queries':sum(x['task']==task for x in final['responses']),
                                  'agree':sum(x['task']==task and x['agrees_original_annotation'] for x in final['responses'])} for task in ('scene_presence','region_class')}
            write_json(out/'report.json',final);record['status']='completed'
        except BaseException as exc:
            record.update(status='needs_attention',error=str(exc))
            for process in processes:
                if process.poll() is None:process.send_signal(2)
            raise
        finally:
            for stream in streams:stream.close()
            record['seconds']=time.monotonic()-started;write_json(out/'status.json',record)


if __name__=='__main__':main()
