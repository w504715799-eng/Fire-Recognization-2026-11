"""CPU metadata-only preparation; images referenced, no pixel copies and no model execution."""
from pathlib import Path
import argparse,json,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.research_data import load_manifest,canonical_row
from fire_r2.teacher_adaptation import partition_fit,instruction_records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/qwen3vl-2b-preparation-20261106/instructions');a=p.parse_args()
    if ROOT!=Path('/root/rivermind-data/FIRE_R2'):raise ProjectError('Prepare source instruction views in the authorized cloud project')
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Frozen teacher split exists; do not overwrite or reselect from results')
    manifest=ROOT/'artifacts/data/dfire-kaggle-21527-sha65d81b60/manifest.jsonl';mh=sha256(manifest)
    old=read_json(ROOT/'artifacts/teacher-audit-20261106/selected-fit-metadata.json')
    if old['source_manifest_sha256']!=mh:raise ProjectError('Source manifest changed')
    raw=load_manifest(manifest,roles={'fit'},enrich=False)
    split=partition_fit(raw,{r['sample_id'] for r in old['rows']});out.mkdir(parents=True)
    report={'source':'dfire','detector_fit_images':len(raw),'detector_fit_unchanged':True,
            'source_manifest_sha256':mh,'original_annotations_modified':False,'image_copies_created':0,
            'supervised_tasks':['scene_presence','region_class'],'context_GT_created':False,
            'heldout_scope':'teacher holdout within detector fit, not detector test or cross-domain generalization',
            'partitions':{}}
    for part,selected in split.items():
        count=0;tasks=Counter();rows_file=out/(part+'.images.jsonl');instructions=out/(part+'.instructions.jsonl')
        with rows_file.open('x') as rf,instructions.open('x') as tf:
            for rawrow in selected:
                row=canonical_row(rawrow);row['image_sha256']=sha256(row['image_relpath'])
                rf.write(json.dumps(row,ensure_ascii=False)+'\n')
                for record in instruction_records(row,part,mh):
                    tf.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1;tasks[record['task']]+=1
        report['partitions'][part]={'images':len(selected),'states':dict(Counter(r['state'] for r in selected)),
            'instruction_examples':count,'tasks':dict(tasks),'image_manifest_sha256':sha256(rows_file),
            'instruction_sha256':sha256(instructions)}
    write_json(out/'receipt.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
