"""CPU-only audit of existing complete teacher caches and completed student evaluations."""
from pathlib import Path
import argparse,csv,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.research_data import load_manifest
from fire_r2.teacher_transfer import make_transfer_supervision,validate_transfer,audit_joint_caches
from fire_r2.supervision import digest_object


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve previous audit')
    started=time.monotonic();caches={};result={'status':'running','models_loaded':False,'caches':{},'completed_students':{}}
    uniform=read_json(ROOT/'artifacts/supervision/dfire-B.json')
    for name in ('base','adapted'):
        cfg=read_json(ROOT/f'configs/train_dfire_T_{name}_20261109.json')
        rows=load_manifest(ROOT/cfg['prepared_view']/'manifest.jsonl',roles={'fit'},enrich=False)
        directory=ROOT/f'artifacts/teacher-transfer-20261109/{name}'
        cache=read_json(directory/'cache.json');report=read_json(directory/'report.json');caches[name]=cache
        advice=read_json(ROOT/cfg['supervision'])
        if set(advice['samples'])!={r['sample_id'] for r in rows}:raise ProjectError('Full fit coverage mismatch')
        regenerated=make_transfer_supervision(uniform,cache,cfg['teacher_expected'])
        if digest_object(regenerated)!=digest_object(advice):raise ProjectError('Stored supervision differs from full cache derivation')
        validate_transfer(advice,rows,cfg['teacher_expected'])
        if report['status']!='completed' or report['completed_images']!=len(rows):raise ProjectError('Incomplete cache report')
        result['caches'][name]={'images':len(rows),'cache_sha256':sha256(directory/'cache.json'),
            'supervision_sha256':sha256(ROOT/cfg['supervision']),'all_original_fit_retained':True,
            'original_GT_and_uniform_weights_preserved':True,'seconds':report['seconds'],
            'images_per_second_including_load':len(rows)/report['seconds'],'peak_gpu_bytes':report['peak_gpu_bytes']}
    for key in ('manifest_sha256','protocol','prompt_sha256','device','dtype','batch_images','selection_sha256','scoring_source_sha256'):
        if caches['base'][key]!=caches['adapted'][key]:raise ProjectError('Paired cache protocol differs: '+key)
    data=ROOT/'artifacts/qwen3vl-2b-preparation-20261106/instructions'
    held=data/'teacher_holdout.images.jsonl';receipt=read_json(data/'receipt.json')
    if sha256(held)!=receipt['partitions']['teacher_holdout']['image_manifest_sha256']:raise ProjectError('Holdout differs')
    audit=audit_joint_caches(caches,[json.loads(line) for line in held.open()])
    if audit!=read_json(ROOT/'artifacts/teacher-transfer-20261109/production-holdout-audit.json'):raise ProjectError('Production probability audit differs')
    result['probability_audit']=audit
    for name in ('B_reference','P_control','T_base','T_adapted'):
        directory=ROOT/'artifacts/student-transfer-20261109/evaluation'/name
        if not (directory/'status.json').exists():continue
        state=read_json(directory/'status.json')
        if state['status']!='completed':continue
        metrics=read_json(directory/'dev-metrics.json');run=ROOT/'runs'/state['run']
        if sha256(directory/'dev-metrics.json')!=state['metrics_sha256']:raise ProjectError('Completed metrics changed')
        result['completed_students'][name]={'actual_epochs_from_CSV':len(list(csv.DictReader((run/'results.csv').open()))),
            'training_stop_reason':state['training_stop_reason'],'checkpoint_sha256':state['checkpoint_sha256'],
            'selection_sha256':metrics['selection_sha256'],'images':metrics['images'],
            'AP':metrics['AP'],'small640_AP':metrics['small640_AP'],
            'fire_at_source_cal_threshold':metrics['fire_at_source_cal_threshold'],'groups':metrics['groups']}
    if len({v['selection_sha256'] for v in result['completed_students'].values()})!=1:raise ProjectError('Evaluation image sets differ')
    result.update(status='completed',seconds=time.monotonic()-started)
    write_json(out,result);print(json.dumps({'status':'completed','caches':result['caches'],'students':list(result['completed_students'])}))


if __name__=='__main__':main()
