"""Explicit cloud-only export of predictions; all numerical scoring is separate."""
from pathlib import Path
import platform
import time
import os,resource,socket

from .common import ProjectError,sha256,write_json,child,output_path
from .supervision import load_manifest,digest_object
from .evaluation import evaluation_rows
from .research_data import canonical_row


def display_images(paths):
    from PIL import Image,ImageOps
    images=[]
    for path in paths:
        with Image.open(path) as im:images.append(ImageOps.exif_transpose(im).convert('RGB'))
    return images


def prediction_plan(root,config):
    weights=child(root,config['weights']);manifest=child(root,config['manifest'])
    issues=[]
    if not weights.is_file():issues.append('checkpoint_missing_no_auto_download')
    if not manifest.is_file():issues.append('manifest_missing')
    return {'mode':'plan_only','issues':issues,'ready':not issues,'starts_model':False,
            'checkpoint':str(weights),'manifest':str(manifest),'output':str(output_path(root,config['output'])),
            'device':config.get('device','cuda')}


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
    raw=load_manifest(plan['manifest'],enrich=False)
    rows=[canonical_row(r) for r in evaluation_rows(raw,config['training_source'],config['target_source'],config.get('purpose','test'))]
    files=[Path(r['image_relpath']) for r in rows]
    if not all(p.is_file() for p in files):raise ProjectError('Full selected images are required')
    if config.get('device')=='cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or config.get('half',False):
            raise ProjectError('CPU evaluation requires explicit CUDA invisibility and FP32')
        from .train import offline_connector
        original=socket.socket.connect;socket.socket.connect=offline_connector(original)
        try:return _predict_locked(root,config,plan,rows,files,device='cpu')
        finally:socket.socket.connect=original
    with cloud_gpu(root,cloud):return _predict_locked(root,config,plan,rows,files,device=0)


def _predict_locked(root,config,plan,rows,files,device=0):
    import torch
    cpu=device=='cpu'
    if cpu:
        threads=int(config.get('cpu_threads',4))
        if not 1<=threads<=8:raise ProjectError('Bound CPU evaluation threads')
        torch.set_num_threads(threads);torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized():raise ProjectError('CPU evaluation must not initialize CUDA')
    from .ultralytics_backend import ResearchDetectionModel
    from ultralytics import YOLO
    started=time.perf_counter();model=YOLO(plan['checkpoint'])
    if isinstance(model.model,ResearchDetectionModel):model.model.strip_training_heads()
    load_seconds=time.perf_counter()-started
    images=[];batch=int(config.get('batch',4))
    if not cpu:torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for start in range(0,len(rows),batch):
        results=model.predict(source=display_images(files[start:start+batch]),imgsz=640,conf=.001,
                              iou=.7,max_det=300,device=device,batch=batch,half=bool(config.get('half',False)),
                              rect=False,verbose=False,save=False,save_txt=False,stream=False)
        chunk=rows[start:start+batch]
        if len(results)!=len(chunk):raise ProjectError('Model returned incomplete image coverage')
        for row,result in zip(chunk,results):
            det=[]
            for xyxy,score,cls in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist()):
                x1,y1,x2,y2=xyxy;det.append({'bbox':[x1,y1,x2-x1,y2-y1],'score':score,'class_id':int(cls)})
            images.append({'sample_id':row['sample_id'],'detections':det})
        if start%max(batch,100//batch*batch)==0:
            write_json(output_path(root,config['output']).with_suffix('.progress.json'),
                       {'images':len(images),'total':len(rows),'device':str(device),'seconds':time.perf_counter()-started})
    if not cpu:torch.cuda.synchronize()
    artifact={'schema_version':1,'training_source':config['training_source'],'target_source':config['target_source'],
              'purpose':config.get('purpose','test'),'seed':config.get('seed'),'checkpoint_sha256':sha256(Path(plan['checkpoint'])),
              'selection_sha256':digest_object(sorted(r['sample_id'] for r in rows)),
              'manifest_sha256':sha256(Path(plan['manifest'])),'inference':{'imgsz':640,'conf_floor':.001,'max_det':300,'iou':.7,'half':bool(config.get('half',False))},
              'runtime':{'model_load_seconds':load_seconds,'prediction_seconds':time.perf_counter()-started,
                         'device':str(device),'precision':'float16' if config.get('half',False) else 'float32',
                         'cpu_threads':torch.get_num_threads(),'cuda_initialized':torch.cuda.is_initialized(),
                         'prediction_includes_file_io_and_preprocessing':True,'warmup_excluded':False,
                         'max_allocated_bytes':None if cpu else torch.cuda.max_memory_allocated(),
                         'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                         'deployment_parameters':sum(p.numel() for p in model.model.parameters())},'images':images}
    write_json(output_path(root,config['output']),artifact)
    return {'output':str(output_path(root,config['output'])),'images':len(images),'training_started':False}
