"""CPU-only aggregation of completed teacher experiments, explicitly retaining pending/failures."""
from pathlib import Path
import argparse,datetime,json,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,output_path,sha256
from fire_r2.teacher_audit import audit_cache,cpu_diagnostic_cache
from fire_r2.teacher_comparison import paired_answers,query_cost


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Keep earlier comparison snapshots; use a new output')
    meta=read_json(ROOT/'artifacts/teacher-audit-20261106/selected-fit-metadata.json');old=set(meta['original_8'])
    specs={
      'gpu_fp16_v1_8':('cache','artifacts/teacher/dfire-fit-pilot-v1.json','GPU','float16',1),
      'gpu_fp16_v2_serial8':('cache','artifacts/teacher/dfire-fit-pilot-v2_serial.json','GPU','float16',1),
      'gpu_fp16_v2_batch4_8':('cache','artifacts/teacher/dfire-fit-pilot-v2_batch4.json','GPU','float16',4),
      'gpu_fp16_v2_batch4_64':('cache','artifacts/teacher/dfire-fit-pilot-v2_expanded.json','GPU','float16',4),
      'cpu_fp32_v1_8':('cpu','artifacts/teacher-cpu-v1-control-20261106/report.json','CPU','float32',1),
      'cpu_fp32_v1_64':('cpu','artifacts/teacher-cpu-v1-expanded-20261106/report.json','CPU','float32',1),
      'cpu_fp32_v2_serial64':('cpu','artifacts/teacher-cpu-expanded-20261106/report.json','CPU','float32',1),
      'cpu_fp32_v2_batch4_8':('cache','artifacts/teacher-cpu-batch4-20261106/cache.json','CPU','float32',4),
      'cpu_fp32_v2_batch4_64':('cache','artifacts/teacher-cpu-batch4-expanded-20261106/cache.json','CPU','float32',4)}
    report={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':{},'pairs':{},
            'independent_review_completed':False,'student_gain_measured_here':False};caches={}
    for name,(kind,relative,device,dtype,batch) in specs.items():
        path=ROOT/relative;entry={'path':relative,'device':device,'dtype':dtype,'batch':batch};report['cases'][name]=entry
        if not path.exists():
            entry['status']='pending'
            failure_report=path.parent/'report.json'
            if kind=='cache' and failure_report.exists():
                f=read_json(failure_report)
                if f.get('status')=='stopped':entry.update(status='stopped',error=f.get('error'),report_path=str(failure_report))
            continue
        obj=read_json(path);entry['file_sha256']=sha256(path)
        if kind=='cpu' and obj.get('status')!='completed':
            entry.update(status=obj.get('status','unknown'),completed_queries=len(obj.get('calls',[])),error=obj.get('error'));continue
        cache=cpu_diagnostic_cache(obj) if kind=='cpu' else obj
        if cache['manifest_sha256']!=meta['source_manifest_sha256']:raise ProjectError('Comparison manifest differs')
        caches[name]=cache;entry.update(status='completed',cost=query_cost(cache),runtime=cache.get('runtime'),
            audit=audit_cache(cache,meta['rows'],old))
    pairs=[('cpu_fp32_v1_64','cpu_fp32_v2_serial64',False,True),
           ('cpu_fp32_v2_serial64','cpu_fp32_v2_batch4_8',True,False),
           ('cpu_fp32_v2_serial64','cpu_fp32_v2_batch4_64',False,False),
           ('gpu_fp16_v2_serial8','gpu_fp16_v2_batch4_8',True,False),
           ('cpu_fp32_v2_serial64','gpu_fp16_v2_serial8',True,False),
           ('cpu_fp32_v2_batch4_8','gpu_fp16_v2_batch4_8',True,False),
           ('cpu_fp32_v2_batch4_64','gpu_fp16_v2_batch4_64',False,False)]
    for left,right,original_only,changing in pairs:
        key=left+'__'+right
        if left not in caches or right not in caches:report['pairs'][key]={'status':'pending'};continue
        r=paired_answers(caches[left],caches[right],old if original_only else None,changing)
        r.update(status='completed',cohort='original_8' if original_only else 'all_64',
                 device_or_dtype_changes=specs[left][2:4]!=specs[right][2:4],batch_changes=specs[left][4]!=specs[right][4])
        if not original_only:
            all_ids=set(caches[left]['samples']);new=all_ids-old
            r['new_56']=paired_answers(caches[left],caches[right],new,changing)
            r['original_8']=paired_answers(caches[left],caches[right],old,changing)
        if changing:
            r['syntax_only_normalized_comparison']=paired_answers(caches[left],caches[right],old if original_only else None,True,True)
        report['pairs'][key]=r
    queue=ROOT/'artifacts/methods-20261106/after-B-status.json'
    if queue.exists():
        report['existing_gpu_queue']=read_json(queue)
        mapping={'teacher_v2_serial':'gpu_fp16_v2_serial8','teacher_v2_batch4':'gpu_fp16_v2_batch4_8',
                 'teacher_v2_expanded':'gpu_fp16_v2_batch4_64'}
        for stage in report['existing_gpu_queue'].get('stages',[]):
            case=mapping.get(stage['name'])
            if case and 'returncode' in stage and stage['returncode']!=0:
                report['cases'][case].update(status='failed',queue_stage=stage)
    write_json(out,report);print(json.dumps({'output':str(out),'completed_cases':list(caches),
                                          'pending_cases':[k for k,v in report['cases'].items() if v['status']!='completed']}))


if __name__=='__main__':main()
