"""Build complete uniform-weight student supervision from a verified production teacher cache."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,output_path,ProjectError,sha256
from fire_r2.teacher_transfer import make_transfer_supervision,make_joint_control,validate_transfer,audit_joint_caches
from fire_r2.research_data import load_manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-config',required=True);p.add_argument('--cache');p.add_argument('--control',action='store_true');a=p.parse_args()
    cfg=read_json(ROOT/a.train_config);target=output_path(ROOT,cfg['supervision'])
    if target.exists():raise ProjectError('Keep existing supervision')
    uniform=read_json(ROOT/'artifacts/supervision/dfire-B.json')
    if a.control:
        if a.cache or cfg.get('teacher_expected'):raise ProjectError('No teacher in the control')
        result=make_joint_control(uniform)
        rows=load_manifest(ROOT/cfg['prepared_view']/'manifest.jsonl',roles={'fit'},enrich=False);validate_transfer(result,rows,alpha_expected=0.)
        write_json(target,result);print(json.dumps({'images':len(result['samples']),'output':str(target),'teacher':None}));return
    if not a.cache:raise ProjectError('Teacher cache required')
    cache=read_json(ROOT/a.cache)
    if cfg.get('reference_teacher_cache'):
        reference=read_json(ROOT/cfg['reference_teacher_cache'])
        for key in ('manifest_sha256','protocol','prompt_sha256','device','dtype','batch_images','selection_sha256','scoring_source_sha256'):
            if reference.get(key)!=cache.get(key):raise ProjectError('Base/adapted cache protocol differs: '+key)
        data=ROOT/'artifacts/qwen3vl-2b-preparation-20261106/instructions';receipt=read_json(data/'receipt.json');held=data/'teacher_holdout.images.jsonl'
        if sha256(held)!=receipt['partitions']['teacher_holdout']['image_manifest_sha256']:raise ProjectError('Teacher holdout selection changed')
        heldout=[json.loads(line) for line in held.open()]
        audit=audit_joint_caches({'base':reference,'adapted':cache},heldout)
        audit_path=ROOT/'artifacts/teacher-transfer-20261109/production-holdout-audit.json'
        if not audit_path.exists():write_json(audit_path,audit)
    result=make_transfer_supervision(uniform,cache,cfg['teacher_expected'])
    rows=load_manifest(ROOT/cfg['prepared_view']/'manifest.jsonl',roles={'fit'},enrich=False);validate_transfer(result,rows,cfg['teacher_expected'])
    write_json(target,result);print(json.dumps({'images':len(result['samples']),'teacher':result['teacher'],'output':str(target),'all_original_images_retained':True,'original_annotations_changed':False}))


if __name__=='__main__':main()
