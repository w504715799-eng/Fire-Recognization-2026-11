"""YOLO26 adapter for pinned Ultralytics 2e241d0 (AGPL-3.0 upstream interfaces).

The detection term decomposition follows ultralytics/utils/loss.py at that commit.
No weights are fetched. This module is imported only by authorized cloud code.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.augment import Compose, LetterBox, Format
from ultralytics.utils.loss import v8DetectionLoss, E2ELoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors, bbox2dist

from .common import ProjectError, read_json, write_json
from .supervision import load_manifest, consistent_batch, digest_object
from .torch_modules import (FireSmokeFusion,SemanticHead,positive_weighted_reduction,
                            no_extra_bn_updates,gain_loss,semantic_loss,joint_presence_loss,smoke_box_mask)


class DecomposedLoss(v8DetectionLoss):
    def terms(self,preds,batch,assignment=None):
        if self.reg_max!=1:raise ProjectError('This pinned YOLO26 adapter expects reg_max=1')
        distance=preds['boxes'].permute(0,2,1).contiguous()
        logits=preds['scores'].permute(0,2,1).contiguous()
        anchors,stride=make_anchors(preds['feats'],self.stride,.5)
        image_size=torch.tensor(preds['feats'][0].shape[2:],device=logits.device,dtype=logits.dtype)*self.stride[0]
        boxes=self.bbox_decode(anchors,distance)
        if assignment is None:
            targets=torch.cat([batch['batch_idx'].view(-1,1),batch['cls'].view(-1,1),batch['bboxes']],1)
            targets=self.preprocess(targets.to(self.device),logits.shape[0],image_size[[1,0,1,0]])
            labels,gt=targets.split((1,4),2)
            _,target_boxes,scores,fg,target_indices=self.assigner(logits.detach().sigmoid(),
                (boxes.detach()*stride).type(gt.dtype),anchors*stride,labels,gt,gt.sum(2,keepdim=True).gt_(0.))
            assignment={'boxes':target_boxes,'scores':scores,'fg':fg,'indices':target_indices}
        target_boxes=assignment['boxes']/stride;scores=assignment['scores'];fg=assignment['fg']
        denominator=scores.sum().clamp_min(1)
        cls=F.binary_cross_entropy_with_logits(logits,scores.to(logits.dtype),reduction='none')
        if self.class_weights is not None:cls=cls*self.class_weights
        cls=cls.sum(-1)
        bi,ai=fg.nonzero(as_tuple=True);positive_weight=scores[bi,ai].sum(-1)
        ciou=bbox_iou(boxes[bi,ai],target_boxes[bi,ai],xywh=False,CIoU=True).reshape(-1)
        iou_terms=boxes.sum(-1)*0
        iou_terms=iou_terms.index_put((bi,ai),(1-ciou)*positive_weight)
        target_distance=bbox2dist(anchors,target_boxes)*stride
        pred_distance=distance*stride
        scale=image_size[[1,0,1,0]]
        l1=(pred_distance/scale-target_distance/scale).abs().mean(-1)
        l1_terms=distance.sum(-1)*0
        l1_terms=l1_terms.index_put((bi,ai),l1[bi,ai]*positive_weight)
        terms=torch.stack([iou_terms*self.hyp.box,cls*self.hyp.cls,l1_terms*self.hyp.dfl],-1)
        return terms,denominator,assignment


class ResearchCriterion(E2ELoss):
    def __init__(self,model):
        super().__init__(model,DecomposedLoss)
        self.research=deepcopy(model.research_config)

    def __call__(self,preds,batch):
        preds=self.one2many.parse_output(preds)
        B=preds['one2many']['boxes'].shape[0]
        device=preds['one2many']['boxes'].device
        clean=batch.get('r2_clean',torch.zeros(B,device=device,dtype=torch.bool))
        weights=batch.get('r2_weights',torch.ones(B,device=device))
        weights=torch.where(clean,weights,torch.ones_like(weights))
        final=[];contexts=[]
        for loss,name in [(self.one2many,'one2many'),(self.one2one,'one2one')]:
            terms,denom,assignment=loss.terms(preds[name],batch)
            final.append(positive_weighted_reduction(terms,weights,denom));contexts.append((denom,assignment))
        detection=final[0]*self.o2m+final[1]*self.o2o
        aux=preds.get('_research',{})
        zero=detection.sum()*0;gain=zero;semantic=zero;smoke=zero
        if 'local_predictions' in aux:
            alternatives=[];unweighted=[]
            for branch in ('local_predictions','context_predictions'):
                parts=[]
                for idx,(loss,name) in enumerate([(self.one2many,'one2many'),(self.one2one,'one2one')]):
                    terms,denom,_=loss.terms(aux[branch][name],batch,contexts[idx][1])
                    parts.append(positive_weighted_reduction(terms,weights,denom))
                    if idx==0:unweighted.append(terms.sum(-1)/denom)
                alternatives.append(parts[0]*self.o2m+parts[1]*self.o2o)
            detection=(detection+.25*alternatives[0]+.25*alternatives[1])/1.5
            if self.research.get('m_mode')=='gain':
                count=aux['gate_logits'].shape[-2]*aux['gate_logits'].shape[-1]
                assignment=contexts[0][1]
                valid=assignment['fg'][:,:count] & (assignment['scores'][:,:count,0]>0)
                gain=gain_loss(aux['gate_logits'],unweighted[0][:,:count],unweighted[1][:,:count],valid)
        if 'semantic_logits' in aux:
            if self.research.get('semantic')=='joint_presence':
                semantic,_=joint_presence_loss(aux['semantic_logits'],batch['r2_targets'],batch['r2_mask'],clean,batch.get('r2_warmup',1.))
            else:
                semantic,_=semantic_loss(aux['semantic_logits'],batch['r2_targets'],batch['r2_mask'],clean,
                                         batch.get('r2_warmup',1.),.1,.05 if self.research.get('semantic')=='qwen' else 0.)
        if 'smoke_logits' in aux:
            logits=aux['smoke_logits'];target=smoke_box_mask(batch,*logits.shape[-2:])
            smoke=F.binary_cross_entropy_with_logits(logits,target)
        gain_scaled=float(self.research.get('gain_weight',.05))*gain
        smoke_scaled=float(self.research.get('smoke_weight',.2))*smoke
        total=detection.sum()+semantic+gain_scaled+smoke_scaled
        items=dict(zip(self.one2one.loss_names,detection.detach()))
        items.update(semantic_loss=semantic.detach(),gain_loss=gain_scaled.detach(),smoke_aux_loss=smoke_scaled.detach())
        return total*B,items


class ResearchDetectionModel(DetectionModel):
    def configure_research(self,config):
        self.research_config=deepcopy(config)
        head=self.model[-1]
        if list(head.f)!=[16,19,22] or head.reg_max!=1 or head.nc!=2:
            raise ProjectError('Expected the pinned two-class YOLO26 P3/P4/P5 graph')
        self.end2end=True
        channels=head.cv2[0][0].conv.in_channels
        c2_channels=self.model[2].cv2.conv.out_channels
        width=int(config.get('width',64))
        self.research_fusion=FireSmokeFusion(channels,c2_channels,width,config['m_mode'],config.get('detail',True),config.get('smoke_aux',True),not config.get('fixed_split',False)) if config.get('m_mode') else None
        self.research_semantic=SemanticHead([width,width,width] if self.research_fusion else [channels],width) if config.get('semantic') else None
        self.research_sup=nn.Conv2d(channels,1,1) if config.get('sup_only') else None
        self.criterion=None
        return self

    def _predict_once(self,x,profile=False,embed=None):
        if not hasattr(self,'research_config'):return super()._predict_once(x,profile,embed)
        if profile or embed:raise ProjectError('Use explicit end-to-end timing for this research model, without layer profiling/embedding mode')
        saved=[];c2=None
        for module in self.model:
            if module.f!=-1:x=saved[module.f] if isinstance(module.f,int) else [x if j==-1 else saved[j] for j in module.f]
            if module is self.model[-1]:
                auxiliary={};features=list(x)
                if self.research_fusion is not None:
                    fusion=self.research_fusion(features[0],c2)
                    features[0]=fusion['final']
                    if self.training:auxiliary.update(fusion)
                if self.training and self.research_semantic is not None:
                    semantic_features=auxiliary['semantic_features'] if self.research_fusion else [features[0]]
                    auxiliary['semantic_logits']=self.research_semantic(semantic_features)
                if self.training and self.research_sup is not None:auxiliary['smoke_logits']=self.research_sup(features[0])
                x=module(features)
                if self.training:
                    if self.research_fusion is not None:
                        with no_extra_bn_updates(module):
                            auxiliary['local_predictions']=module([auxiliary['local'],*features[1:]])
                            auxiliary['context_predictions']=module([auxiliary['context'],*features[1:]])
                    x['_research']=auxiliary
            else:x=module(x)
            if module.i==2:c2=x
            saved.append(x if module.i in self.save else None)
        return x

    def init_criterion(self):
        return ResearchCriterion(self) if hasattr(self,'research_config') else super().init_criterion()

    def strip_training_heads(self):
        self.research_semantic=None;self.research_sup=None;self.criterion=None
        if self.research_fusion is not None:self.research_fusion.strip_training_heads()
        return self.eval()


class ViewDataset(Dataset):
    def __init__(self,base,supervision,rows):
        self.base=base;self.supervision=supervision
        self.by_file={Path(r['view_image']).stem:r for r in rows if r['role']=='fit'}
        self.clean_transform=Compose([LetterBox(new_shape=(base.imgsz,base.imgsz),auto=False,scaleup=True),Format(bbox_format='xywh',normalize=True,batch_idx=True)])

    def __len__(self):return len(self.base)

    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)

    def __getitem__(self,index):
        index,clean=index
        row=self.by_file[Path(self.base.im_files[index]).stem]
        item=self.clean_transform(self.base.get_image_and_label(index)) if clean else self.base[index]
        advice=self.supervision['samples'][row['sample_id']]
        item.update(r2_clean=clean,r2_weight=advice['weight'] if clean else 1.,
                    r2_targets=advice['targets'],r2_mask=advice['mask'])
        return item

    def close_mosaic(self,hyp):return self.base.close_mosaic(hyp)

    def collate_fn(self,items):
        extra={k:[x.pop(k) for x in items] for k in ('r2_clean','r2_weight','r2_targets','r2_mask')}
        batch=self.base.collate_fn(items)
        batch.update(r2_clean=torch.tensor(extra['r2_clean'],dtype=torch.bool),r2_weights=torch.tensor(extra['r2_weight']),
                     r2_targets=torch.tensor(extra['r2_targets']),r2_mask=torch.tensor(extra['r2_mask'],dtype=torch.bool))
        return batch


class ViewBatchSampler(Sampler):
    def __init__(self,size,batch_size,seed):self.size=size;self.batch_size=batch_size;self.seed=seed;self.epoch=0
    def __len__(self):return (self.size+self.batch_size-1)//self.batch_size
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch)
        order=torch.randperm(self.size,generator=generator).tolist();self.epoch+=1
        for i,start in enumerate(range(0,self.size,self.batch_size)):
            yield [(idx,consistent_batch(i)) for idx in order[start:start+self.batch_size]]


class ResearchDataLoader(DataLoader):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # PyTorch clears batch_size for a custom batch sampler; the pinned trainer
        # uses this metadata to detect a short final batch. Iteration still uses
        # batch_sampler, so clean/augmented whole-batch scheduling is unchanged.
        object.__setattr__(self,'batch_size',self.batch_sampler.batch_size)

    def reset(self):pass  # No persistent prefetched iterator; next epoch creates a fresh iterator.


class ResearchTrainer(DetectionTrainer):
    research_config=None
    supervision_file=None
    manifest_file=None

    def get_model(self,cfg=None,weights=None,verbose=True):
        model=self.set_model_names_for_load(ResearchDetectionModel(cfg,nc=self.data['nc'],ch=self.data['channels'],verbose=verbose))
        if weights:model.load(weights)
        return model.configure_research(self.research_config)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        if self.world_size>1:raise ProjectError('Initial adapter supports a single GPU; DDP scheduling must be explicitly validated')
        base=self.build_dataset(dataset_path,mode,batch_size)
        rows=load_manifest(self.manifest_file,roles={'fit'},enrich=False);supervision=read_json(self.supervision_file)
        dataset=ViewDataset(base,supervision,rows)
        sampler=ViewBatchSampler(len(dataset),batch_size,self.args.seed)
        sampler.epoch=int(getattr(self,'start_epoch',0))
        return ResearchDataLoader(dataset,batch_sampler=sampler,num_workers=self.args.workers,
                                  collate_fn=dataset.collate_fn,pin_memory=True,persistent_workers=False)

    def preprocess_batch(self,batch):
        batch=super().preprocess_batch(batch)
        for key in ('r2_clean','r2_weights','r2_targets','r2_mask'):batch[key]=batch[key].to(self.device)
        batch['r2_warmup']=min((self.epoch+1)/max(float(self.research_config.get('semantic_warmup_epochs',5)),1),1.)
        return batch
