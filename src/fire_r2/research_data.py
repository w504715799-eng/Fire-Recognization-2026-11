"""In-memory research view of the frozen R2 manifest; never rewrite source GT."""
from copy import deepcopy
import json
from pathlib import Path

from .common import ProjectError


def canonical_row(original):
    row=deepcopy(original)
    row['image_relpath']=row.get('image_path',row.get('image_relpath'))
    width=row.get('width',row.get('decoded_width'))
    height=row.get('height',row.get('decoded_height'))
    if width is None or height is None:
        from PIL import Image
        with Image.open(row['image_relpath']) as image:
            width,height=image.size
            if image.getexif().get(274,1) in (5,6,7,8):width,height=height,width
    if width<=0 or height<=0:raise ProjectError('Positive original image dimensions required')
    row.update(width=width,height=height)
    for number,box in enumerate(row['boxes'],1):
        coordinates=box.get('coordinates')
        if coordinates:
            if coordinates not in ('normalized_xywh','pixel_xywh'):
                raise ProjectError('Unknown box coordinates; do not guess or rewrite GT')
            box['format']='cxcywh_normalized' if coordinates=='normalized_xywh' else 'xywh_absolute'
        if box.get('format') not in ('cxcywh_normalized','xywh_absolute'):
            raise ProjectError('Unknown research box format')
        box['original_annotation_id']=box.get('original_annotation_id',box.get('annotation_id',number))
    scale=640/max(width,height)
    def small(box):
        w,h=box['bbox'][2:]
        if box['format']=='cxcywh_normalized':w,h=w*width,h*height
        return w>0 and h>0 and w*h*scale*scale<1024
    row['has_small_fire']=any(b['class_id']==0 and small(b) for b in row['boxes'])
    return row


def load_manifest(path,roles=None,enrich=True):
    rows=[];seen=set()
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():continue
            original=json.loads(line);ident=original['sample_id']
            if ident in seen:raise ProjectError('Repeated manifest ID; no sample was discarded')
            seen.add(ident)
            if roles is None or original['role'] in roles:
                rows.append(canonical_row(original) if enrich else original)
    return rows


def coco_view(rows):
    images=[];annotations=[]
    for image_id,row in enumerate(rows,1):
        W,H=row['width'],row['height']
        images.append({'id':image_id,'file_name':row['image_relpath'],'width':W,'height':H,'sample_id':row['sample_id']})
        for box in row['boxes']:
            x,y,w,h=box['bbox']
            if box['format']=='cxcywh_normalized':x,y,w,h=(x-w/2)*W,(y-h/2)*H,w*W,h*H
            area=box.get('area')
            annotations.append({'id':len(annotations)+1,'image_id':image_id,'category_id':box['class_id'],
                'bbox':[x,y,w,h],'area':w*h if area is None else area,'iscrowd':box.get('iscrowd',0),
                'original_annotation_id':box['original_annotation_id']})
    return {'images':images,'annotations':annotations,'categories':[{'id':0,'name':'fire'},{'id':1,'name':'smoke'}]}
