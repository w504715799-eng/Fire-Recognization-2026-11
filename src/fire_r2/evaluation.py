"""Frozen original-GT evaluation: COCO AP, cal-only thresholds, PR/FROC and false alarms."""
from __future__ import annotations

from collections import defaultdict
import contextlib
import io
import math
import numpy as np

from .common import ProjectError
from .research_data import coco_view
from .supervision import digest_object


def evaluation_rows(rows, training_source, target_source, purpose='test'):
    if any(not r['sample_id'].startswith(target_source+'/') for r in rows):
        raise ProjectError('Manifest source differs from the declared target dataset')
    if purpose in {'dev','cal'}:
        if training_source!=target_source:raise ProjectError('Target-domain dev/cal cannot select a source model or threshold')
        selected=[r for r in rows if r['role']==purpose]
    elif purpose=='test':selected=[r for r in rows if r['role']=='test'] if training_source==target_source else list(rows)
    elif purpose=='uav_official_test':
        if target_source!='fasdd_uav':raise ProjectError('Official UAV subset only applies to UAV')
        selected=[r for r in rows if r['official_split']=='test']
    else:raise ProjectError('Unknown evaluation purpose')
    if not selected:raise ProjectError('Evaluation selection is empty')
    return selected


def validate_predictions(rows, artifact):
    expected={r['sample_id'] for r in rows}
    images=artifact.get('images',[])
    ids=[x['sample_id'] for x in images]
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ProjectError('Predictions must explicitly include every selected image, including empty predictions')
    if artifact.get('selection_sha256')!=digest_object(sorted(expected)):
        raise ProjectError('Prediction selection hash differs from this evaluation selection')
    if not artifact.get('checkpoint_sha256'):raise ProjectError('Missing checkpoint identity')
    result={}
    for image in images:
        dets=image['detections']
        if len(dets)>300:raise ProjectError('Prediction protocol permits at most 300 detections per image')
        for det in dets:
            if det['class_id'] not in (0,1) or len(det['bbox'])!=4:raise ProjectError('Invalid prediction class/bbox')
            if not all(math.isfinite(float(v)) for v in [det['score'],*det['bbox']]):raise ProjectError('Nonfinite prediction')
            if not 0<=det['score']<=1 or min(det['bbox'][2:])<0:raise ProjectError('Invalid confidence or box size')
        result[image['sample_id']]=dets
    return result


def absolute_box(box,row):
    x,y,w,h=box['bbox']
    if box['format']=='cxcywh_normalized':
        W,H=row['width'],row['height']
        if W is None or H is None:raise ProjectError('Image dimensions are required')
        return [(x-w/2)*W,(y-h/2)*H,w*W,h*H]
    return [x,y,w,h]


def iou_xywh(a,b,crowd=False):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    area_a=max(aw,0)*max(ah,0);area_b=max(bw,0)*max(bh,0)
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    denom=area_a if crowd else area_a+area_b-inter
    return inter/denom if denom>0 else 0.


def matched_records(rows,predictions,cls=0,iou_threshold=.5):
    records=[];n_gt=0;negative_images=[]
    for row in rows:
        gt=[b for b in row['boxes'] if b['class_id']==cls]
        n_gt+=sum(not b.get('iscrowd',0) for b in gt)
        if not gt:negative_images.append(row['sample_id'])
        gt=sorted(gt,key=lambda b:bool(b.get('iscrowd',0)))
        matched=set()
        dets=sorted([d for d in predictions[row['sample_id']] if d['class_id']==cls],key=lambda d:-d['score'])[:300]
        for order,det in enumerate(dets):
            best=-1;overlap=min(iou_threshold,1-1e-10)
            for j,box in enumerate(gt):
                crowd=bool(box.get('iscrowd',0))
                if j in matched and not crowd:continue
                if best>=0 and not gt[best].get('iscrowd',0) and crowd:break
                v=iou_xywh(det['bbox'],absolute_box(box,row),crowd)
                if v>=overlap:best=j;overlap=v
            ignored=best>=0 and bool(gt[best].get('iscrowd',0))
            if best>=0:matched.add(best)
            records.append({'sample_id':row['sample_id'],'score':float(det['score']),
                            'tp':int(best>=0 and not ignored),'fp':int(best<0),'ignored':ignored,'order':order})
    records.sort(key=lambda d:-d['score'])
    return records,n_gt,negative_images


def operating_curve(rows,predictions,cls=0):
    records,n_gt,negative_images=matched_records(rows,predictions,cls)
    valid=[r for r in records if not r['ignored']]
    negatives=set(negative_images);alerted=set();tp=fp=0
    points=[{'threshold':math.nextafter(1.,math.inf),'tp':0,'fp':0,'precision':0.,'recall':0. if n_gt else None,'f1':0. if n_gt else None,'fp_per_image':0.,'negative_trigger_rate':0. if negatives else None}]
    for i,r in enumerate(valid):
        tp+=r['tp'];fp+=r['fp']
        if r['sample_id'] in negatives:alerted.add(r['sample_id'])
        if i+1<len(valid) and valid[i+1]['score']==r['score']:continue
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/n_gt if n_gt else None
        f1=2*tp/(n_gt+tp+fp) if n_gt else None
        points.append({'threshold':r['score'],'tp':tp,'fp':fp,'precision':precision,'recall':recall,'f1':f1,
                       'fp_per_image':fp/len(rows),'negative_trigger_rate':len(alerted)/len(negatives) if negatives else None})
    return {'iou':.5,'max_predictions_per_image':300,'images':len(rows),'original_gt_boxes':n_gt,
            'original_negative_images':len(negatives),'points':points}


def calibrate(rows,artifact,training_source):
    if any(r['role']!='cal' or not r['sample_id'].startswith(training_source+'/') for r in rows):
        raise ProjectError('Threshold fitting is restricted to the source cal partition')
    if artifact.get('training_source')!=training_source:raise ProjectError('Prediction model source differs from calibration source')
    predictions=validate_predictions(rows,artifact);curve=operating_curve(rows,predictions,0)
    if not curve['original_gt_boxes']:raise ProjectError('No fire GT on cal; cannot choose fire F1 threshold')
    # Highest threshold among exact ties: no target-test fitting.
    best=max(curve['points'],key=lambda p:(p['f1'],p['threshold']))
    return {'schema_version':1,'source':training_source,'checkpoint_sha256':artifact['checkpoint_sha256'],
            'fit_partition':'cal','rule':'fire_F1_IoU0.5_tie_highest_threshold','threshold':best['threshold'],
            'selection_sha256':artifact['selection_sha256'],'calibration_predictions_sha256':digest_object(artifact),
            'calibration_result':best,'calibration_images':len(rows)}


def operating_point(rows,predictions,threshold,cls=0):
    records,n_gt,negative_images=matched_records(rows,predictions,cls)
    chosen=[r for r in records if r['score']>=threshold and not r['ignored']]
    tp=sum(r['tp'] for r in chosen);fp=sum(r['fp'] for r in chosen)
    negatives=set(negative_images);alarms=len({r['sample_id'] for r in chosen if r['sample_id'] in negatives})
    return {'threshold':threshold,'tp':tp,'fp':fp,'fn':n_gt-tp,'images':len(rows),
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/n_gt if n_gt else None,
            'fp_per_image':fp/len(rows),'original_negative_images':len(negatives),'triggered_negative_images':alarms,
            'negative_trigger_rate':alarms/len(negatives) if negatives else None}


def coco_metrics(rows,predictions,small640=False):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:raise ProjectError('COCO AP requires pycocotools in the authorized evaluation environment; no automatic installation') from exc
    original=coco_view(rows)
    scale_by_id={i:(640/max(r['width'],r['height']))**2 for i,r in enumerate(rows,1)}
    if small640:
        # Only evaluation area units change. Raw files and bbox coordinates remain untouched.
        for ann in original['annotations']:
            ann['original_area']=ann['area']
            ann['area']=ann['bbox'][2]*ann['bbox'][3]*scale_by_id[ann['image_id']]
    dt=[]
    for ident,row in enumerate(rows,1):
        dt.extend({'image_id':ident,'category_id':d['class_id'],'bbox':d['bbox'],'score':d['score']} for d in predictions[row['sample_id']])
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=original;gt.createIndex()
        if dt:result=gt.loadRes(dt)
        else:
            result=COCO();result.dataset={**original,'annotations':[]};result.createIndex()
        if small640:
            for ann in result.dataset['annotations']:ann['area']*=scale_by_id[ann['image_id']]
        evaluator=COCOeval(gt,result,'bbox');evaluator.params.maxDets=[1,10,100]
        if small640:
            evaluator.params.areaRng=[[0.,math.nextafter(1024.,-math.inf)]]
            evaluator.params.areaRngLbl=['small_at_640']
        evaluator.evaluate();evaluator.accumulate()
    scores={}
    for index,name in enumerate(('fire','smoke')):
        precision=evaluator.eval['precision'][:,:,index,0,2]
        recall=evaluator.eval['recall'][:,index,0,2]
        valid=precision[precision>-1];p50=precision[0];p50=p50[p50>-1];r=recall[recall>-1]
        scores[name]={'AP50_95':float(valid.mean()) if len(valid) else None,'AP50':float(p50.mean()) if len(p50) else None,'AR100':float(r.mean()) if len(r) else None}
    values=[s['AP50_95'] for s in scores.values() if s['AP50_95'] is not None]
    return {'backend':'pycocotools.COCOeval bbox','maxDets':[1,10,100],'classes':scores,'mAP50_95':float(np.mean(values)) if values else None,'original_gt_preserved':True,'area_protocol':'bbox area at isotropic 640 canvas' if small640 else 'original author area'}


def evaluate(rows,artifact,calibration=None):
    predictions=validate_predictions(rows,artifact)
    if calibration:
        if calibration.get('fit_partition')!='cal' or calibration['source']!=artifact.get('training_source') or calibration['checkpoint_sha256']!=artifact['checkpoint_sha256']:
            raise ProjectError('Threshold must come from this checkpoint and its source cal partition')
    result={'schema_version':1,'images':len(rows),'selection_sha256':artifact['selection_sha256'],
            'checkpoint_sha256':artifact['checkpoint_sha256'],'training_source':artifact.get('training_source'),
            'target_source':artifact.get('target_source'),'purpose':artifact.get('purpose'),'seed':artifact.get('seed'),
            'AP':coco_metrics(rows,predictions),'small640_AP':coco_metrics(rows,predictions,True),'fire_curve':operating_curve(rows,predictions,0),'groups':{}}
    if calibration:result['fire_at_source_cal_threshold']=operating_point(rows,predictions,calibration['threshold'])
    groups={state:[r for r in rows if r['state']==state] for state in ('fire_only','smoke_only','both','empty')}
    groups['contains_small_fire']=[r for r in rows if r.get('has_small_fire') is True]
    for key,subset in groups.items():
        group={'images':len(subset)}
        if subset:
            group['AP']=coco_metrics(subset,predictions)
            if calibration:group['fire_at_source_cal_threshold']=operating_point(subset,predictions,calibration['threshold'])
        result['groups'][key]=group
    result['small_fire_definition']='original fire bbox resized isotropically to max-side 640: area < 32^2; group AP includes all GT in selected images'
    result['alarm_interpretation']='Original-label image trigger rate, not field alarms per hour'
    return result


def select_checkpoint(epoch_records):
    if not epoch_records:raise ProjectError('No dev evaluation records')
    if any(x.get('partition')!='dev' for x in epoch_records):raise ProjectError('Checkpoint selection only uses dev')
    valid=[x for x in epoch_records if x.get('fire_AP50_95') is not None and math.isfinite(x['fire_AP50_95'])]
    if not valid:raise ProjectError('No defined fire AP for checkpoint selection')
    return max(valid,key=lambda x:(x['fire_AP50_95'],-x['epoch']))


def summarize_repeats(reports):
    if len(reports)<2:raise ProjectError('At least two independent repeated runs are needed for variation')
    if len({r['checkpoint_sha256'] for r in reports})!=len(reports):raise ProjectError('Repeated copies of one checkpoint are not independent runs')
    keys={(r['training_source'],r.get('target_source'),r['selection_sha256']) for r in reports}
    if len(keys)!=1:raise ProjectError('Repeated-run reports must use the same source, target and image selection')
    values=[r['AP']['classes']['fire']['AP50_95'] for r in reports]
    if any(v is None for v in values):raise ProjectError('Fire AP is undefined in one repeat')
    return {'runs':len(reports),'fire_AP50_95_mean':float(np.mean(values)),
            'fire_AP50_95_sample_std':float(np.std(values,ddof=1)),
            'fire_AP50_95_values':values,'seeds':[r.get('seed') for r in reports],
            'note':'Across-run variation; not an independent-event confidence interval'}
