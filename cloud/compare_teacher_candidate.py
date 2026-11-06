"""Compare completed 0.8B/2B CPU development regressions without rewriting raw answers."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,ProjectError
from fire_r2.teacher_audit import cpu_diagnostic_cache,audit_cache
from fire_r2.teacher_comparison import compare_teacher_models


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    output=ROOT/a.output
    if output.exists():raise ProjectError('Keep earlier candidate reports')
    meta=read_json(ROOT/'artifacts/teacher-audit-20261106/selected-fit-metadata.json')
    paths={'existing_0_8b':'artifacts/teacher-cpu-expanded-20261106/report.json',
           'base_2b':'artifacts/teacher-qwen3vl2b/base/cpu-regression64-20261106/report.json'}
    caches={k:cpu_diagnostic_cache(read_json(ROOT/v)) for k,v in paths.items()}
    old=set(meta['original_8']);new=set(caches['base_2b']['samples'])-old
    report={'scope':'64 development images, not independent heldout generalization',
            'source_reports':paths,'adapted_model_executed':False,
            'models':{k:audit_cache(c,meta['rows'],old) for k,c in caches.items()},
            'comparison':{name:compare_teacher_models(caches['existing_0_8b'],caches['base_2b'],ids)
                          for name,ids in [('all_64',None),('original_8',old),('new_56',new)]}}
    write_json(output,report)
    print(json.dumps({'output':str(output),'models':{k:r['overall'] for k,r in report['models'].items()}}))


if __name__=='__main__':main()
