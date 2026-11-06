"""Bounded CPU audit alongside training. No model imports, GPU use, or queue mutation."""
from pathlib import Path
import argparse,csv,json,os,sys,time,hashlib,resource

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.research_data import canonical_row
from fire_r2.teacher_runner import pilot_selection,teacher_view
from fire_r2.supervision import selected_regions,digest_object
from fire_r2.teacher_audit import compare_caches,human_review_template


def progress():
    p=ROOT/'runs/dfire_B_seed17_full_20261106/results.csv'
    rows=list(csv.DictReader(p.open()))
    return {'epochs':len(rows),'recent_epoch_seconds':[round(float(b['time'])-float(a['time']),3) for a,b in zip(rows[-4:-1],rows[-3:])]}


def main():
    if ROOT!=Path('/root/rivermind-data/FIRE_R2'):raise ProjectError('Authorized cloud path required')
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/teacher-audit-20261106');a=p.parse_args()
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve prior audit; choose a fresh output directory')
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    started=time.monotonic();before=progress()
    names={'v1':'dfire-fit-pilot-v1.json','v2_serial':'dfire-fit-pilot-v2_serial.json',
           'v2_batch4':'dfire-fit-pilot-v2_batch4.json','v2_expanded':'dfire-fit-pilot-v2_expanded.json'}
    paths={name:ROOT/'artifacts/teacher'/file for name,file in names.items()}
    paths={name:path for name,path in paths.items() if path.exists()}
    if 'v1' not in paths:raise ProjectError('Original pilot cache is required')
    hashes={name:sha256(path) for name,path in paths.items()};caches={name:read_json(path) for name,path in paths.items()}
    manifest=ROOT/'artifacts/data/dfire-kaggle-21527-sha65d81b60/manifest.jsonl'
    raw=[]
    # One metadata pass; decode/header access only for 64 selected images.
    with manifest.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['role']=='fit':raw.append(row)
    selected=pilot_selection(raw,16);rows=[canonical_row(r) for r in selected]
    originals=set(caches['v1']['samples']);mh=sha256(manifest)
    for cache in caches.values():
        if cache['manifest_sha256']!=mh:raise ProjectError('Pilot has stale source manifest')
    report=compare_caches(caches,rows,originals)
    out.mkdir(parents=True)
    write_json(out/'selected-fit-metadata.json',{'source_manifest_sha256':mh,'rows':rows,'original_8':sorted(originals)})
    write_json(out/'independent-review-template.json',human_review_template(rows,{r['sample_id'] for r in rows}))
    input_audit=[]
    from PIL import Image
    for row in rows:
        if row['sample_id'] not in originals:continue
        path=Path(row['image_relpath']);image_hash=sha256(path)
        if image_hash!=caches['v1']['samples'][row['sample_id']]['image_sha256']:raise ProjectError('Pilot image bytes changed')
        with Image.open(path) as im:encoded=im.size;orientation=im.getexif().get(274,1)
        for box in selected_regions(row):
            b=teacher_view(path,box);v2=teacher_view(path,box,region_marker='thin_outline_after_resize')
            ident=hashlib.sha256((row['sample_id']+'|'+str(box['original_annotation_id'])).encode()).hexdigest()[:16]
            # These derived private audit views are never training/teacher source files.
            canvas=Image.new('RGB',(1280,640));canvas.paste(b,(0,0));canvas.paste(v2,(640,0))
            canvas.save(out/(ident+'.jpg'),quality=85)
            input_audit.append({'sample_id':row['sample_id'],'original_annotation_id':box['original_annotation_id'],
                'source_image_sha256':image_hash,'bbox':box['bbox'],'bbox_format':box['format'],
                'encoded_size':list(encoded),'display_size':[row['width'],row['height']],'exif_orientation':orientation,
                'v1_view_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'v2_view_sha256':hashlib.sha256(v2.tobytes()).hexdigest(),
                'output_size':[640,640],'comparison_image':ident+'.jpg'})
    write_json(out/'input-audit.json',input_audit)
    uniform=read_json(ROOT/'artifacts/supervision/dfire-B.json')
    if uniform['manifest_sha256']!=mh or len(uniform['samples'])!=len(raw):raise ProjectError('B supervision binding/coverage differs')
    report['full_fit_B_supervision']={'images':len(raw),'all_one':all(x['weight']==1 for x in uniform['samples'].values()),
                                    'fit_ids_match':set(uniform['samples'])=={r['sample_id'] for r in raw}}
    report['cache_file_sha256']=hashes
    report['input_audit_regions']=len(input_audit)
    report['independent_review_completed']=False
    report['resource_usage']={'threads_limit':1,'wall_seconds':time.monotonic()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'training_before':before,'training_after':progress(),'uses_gpu':False,'model_weights_loaded':False}
    assert hashes=={name:sha256(path) for name,path in paths.items()}
    write_json(out/'report.json',report)
    print(json.dumps({'output':str(out),'cache_variants':list(caches),'pending':report['pending'],
                      'resource_usage':report['resource_usage']}),flush=True)


if __name__=='__main__':main()
