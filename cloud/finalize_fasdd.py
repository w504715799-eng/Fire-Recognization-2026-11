"""Finish one verified archive: restore originals, check all pixels, publish readiness."""
from pathlib import Path
import argparse,json,subprocess,sys,hashlib,datetime,platform

ROOT=Path('/root/rivermind-data/FIRE_R2')
DATA=Path('/root/rivermind-data/shared/datasets')

def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['cv','uav'],required=True);a=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=ROOT/'.envs/sci-env':raise RuntimeError('Use the existing project cloud sci-env')
    key='fasdd_'+a.part;base=DATA/('fasdd-'+a.part)/'v9';out=ROOT/'artifacts/fasdd-full-20261105'
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-B','cloud/restore_fasdd_archives.py','--part',a.part],cwd=ROOT,check=True)
    audit=out/(a.part+'-audit.json')
    with (out/(a.part+'-audit.log')).open('w') as log:
        subprocess.run([sys.executable,'-B','code/fire_r2_cli.py','--datasets-root',str(DATA),'audit',
                        '--dataset',key,'--check-images','--output',str(audit.relative_to(ROOT))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    view=ROOT/'artifacts/data'/('fasdd-'+a.part+'-v9')
    if not (view/'view.json').exists():
        with (out/(a.part+'-prepare.log')).open('w') as log:
            subprocess.run([sys.executable,'-B','code/fire_r2_cli.py','--datasets-root',str(DATA),'prepare',
                            '--dataset',key,'--output',str(view.relative_to(ROOT))],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    inspected=json.loads(audit.read_text());prepared=json.loads((view/'view.json').read_text())
    expected=95314 if a.part=='cv' else 25097
    assert inspected['images']==expected and prepared['all_images_decoded'] and prepared['training_ready']
    if a.part=='cv':assert inspected['frozen_manifest_check']['matched']
    else:assert prepared['roles']=={'external':25097} and not (view/'train.yaml').exists()
    source=json.loads((base/'EXTRACTED.json').read_text())
    ready={**source,'status':'ready','ready_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'images':expected,'all_images_decoded':True,'original_labels_modified':False,
           'use':'external_evaluation_only' if a.part=='uav' else 'source_training_and_evaluation',
           'audit':inspected,'project_view':str(view),'archive_retained':True}
    for name in ['READY.json','STATUS.json']:
        temp=base/(name+'.tmp');temp.write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n');temp.replace(base/name)
    (out/(a.part+'-READY.json')).write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'part':a.part,'status':'ready','images':expected,'roles':prepared['roles'],'archive_retained':True}),flush=True)

if __name__=='__main__':main()
