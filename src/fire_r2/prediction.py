"""Explicit cloud-only export of predictions; all numerical scoring is separate."""
from pathlib import Path
import platform
import time

from .common import ProjectError,sha256,write_json,child,output_path
from .supervision import load_manifest,digest_object
from .evaluation import evaluation_rows


def prediction_plan(root,config):
    weights=child(root,config['weights']);manifest=child(root,config['manifest'])
    issues=[]
    if not weights.is_file():issues.append('checkpoint_missing_no_auto_download')
    if not manifest.is_file():issues.append('manifest_missing')
    return {'mode':'plan_only','issues':issues,'ready':not issues,'starts_model':False,
            'checkpoint':str(weights),'manifest':str(manifest),'output':str(output_path(root,config['output']))}


def export_predictions(root,config,cloud):
    from .runtime import check_remote_environment,cloud_gpu
    check_remote_environment(root,cloud)
    plan=prediction_plan(root,config)
    if not plan['ready']:raise ProjectError('Prediction prerequisites missing: '+','.join(plan['issues']))
    if output_path(root,config['output']).exists():raise ProjectError('Keep previous predictions; choose a new output')
    from .backend_lock import verify_backend
    verify_backend(Path(root),config['ultralytics_commit'])
    weights=Path(plan['checkpoint'])
    if not config.get('weights_sha256') or sha256(weights)!=config['weights_sha256']:
        raise ProjectError('Prediction requires a verified fixed checkpoint')
    rows=evaluation_rows(load_manifest(plan['manifest']),config['training_source'],config['target_source'],config.get('purpose','test'))
    files=[Path(r['image_relpath']) for r in rows]
    if not all(p.is_file() for p in files):raise ProjectError('Full selected images are required')
    with cloud_gpu(root,cloud):
        return _predict_locked(root,config,plan,rows,files)


def _predict_locked(root,config,plan,rows,files):
    from .ultralytics_backend import ResearchDetectionModel
    from ultralytics import YOLO
    import torch
    started=time.perf_counter();model=YOLO(plan['checkpoint'])
    if isinstance(model.model,ResearchDetectionModel):model.model.strip_training_heads()
    load_seconds=time.perf_counter()-started
    images=[];batch=int(config.get('batch',4));torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for start in range(0,len(rows),batch):
        results=model.predict(source=[str(p) for p in files[start:start+batch]],imgsz=640,conf=.001,
                              iou=.7,max_det=300,device=0,batch=batch,half=bool(config.get('half',False)),
                              rect=False,verbose=False,save=False,save_txt=False,stream=False)
        chunk=rows[start:start+batch]
        if len(results)!=len(chunk):raise ProjectError('Model returned incomplete image coverage')
        for row,result in zip(chunk,results):
            det=[]
            for xyxy,score,cls in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist()):
                x1,y1,x2,y2=xyxy;det.append({'bbox':[x1,y1,x2-x1,y2-y1],'score':score,'class_id':int(cls)})
            images.append({'sample_id':row['sample_id'],'detections':det})
    torch.cuda.synchronize()
    artifact={'schema_version':1,'training_source':config['training_source'],'target_source':config['target_source'],
              'purpose':config.get('purpose','test'),'seed':config.get('seed'),'checkpoint_sha256':sha256(Path(plan['checkpoint'])),
              'selection_sha256':digest_object(sorted(r['sample_id'] for r in rows)),
              'manifest_sha256':sha256(Path(plan['manifest'])),'inference':{'imgsz':640,'conf_floor':.001,'max_det':300,'iou':.7,'half':bool(config.get('half',False))},
              'runtime':{'model_load_seconds':load_seconds,'prediction_seconds':time.perf_counter()-started,
                         'prediction_includes_file_io_and_preprocessing':True,'warmup_excluded':False,
                         'max_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'deployment_parameters':sum(p.numel() for p in model.model.parameters())},'images':images}
    write_json(output_path(root,config['output']),artifact)
    return {'output':str(output_path(root,config['output'])),'images':len(images),'training_started':False}
