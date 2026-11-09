"""W/S/M tensor modules. Import only in an authorized PyTorch environment."""
from __future__ import annotations

from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F


def positive_weighted_reduction(terms,weights,denominator):
    """terms[B,A,K] are unnormalized native contributions; denominator is unweighted."""
    if weights.shape!=(terms.shape[0],) or not torch.isfinite(weights).all() or (weights<=0).any():
        raise ValueError('Expected one finite positive weight per image')
    return (terms*weights[:,None,None]).sum(dim=(0,1))/denominator.clamp_min(1)


def masked_bce(logits,targets,mask):
    if logits.shape!=targets.shape or mask.shape!=logits.shape:raise ValueError('Semantic tensor shape mismatch')
    mask=mask.to(dtype=logits.dtype)
    # Undefined target values must not enter BCE before masking.
    targets=torch.where(mask.bool(),targets,torch.zeros_like(targets))
    return (F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*mask).sum()/mask.sum().clamp_min(1)


class SemanticHead(nn.Module):
    def __init__(self,channels,width=64):
        super().__init__()
        self.projections=nn.ModuleList(nn.Sequential(nn.Conv2d(c,width,1),nn.SiLU()) for c in channels)
        self.output=nn.Linear(width*len(channels),6)

    def forward(self,features):
        if len(features)!=len(self.projections):raise ValueError('Semantic feature count mismatch')
        pools=[F.adaptive_avg_pool2d(project(x),1).flatten(1) for project,x in zip(self.projections,features)]
        return self.output(torch.cat(pools,dim=1))


def light_path(width):
    return nn.Sequential(nn.Conv2d(width,width,3,padding=1,groups=width,bias=False),
                         nn.Conv2d(width,width,1,bias=False),nn.GroupNorm(1,width),nn.SiLU())


class FireSmokeFusion(nn.Module):
    def __init__(self,p3_channels,c2_channels,width=64,mode='gain',detail=True,smoke_aux=True,learn_split=True):
        super().__init__()
        if mode not in {'ordinary','split','gate','gain'}:raise ValueError('Unknown M mode')
        self.mode=mode;self.use_detail=detail
        self.project=nn.Conv2d(p3_channels,width,1)
        self.detail=nn.Conv2d(c2_channels,width,1) if detail else None
        self.split=nn.Conv2d(width,1,1) if mode!='ordinary' and learn_split else None
        self.smoke=light_path(width);self.scene=light_path(width)
        self.mix=nn.Sequential(nn.Conv2d(width*2,width,1),nn.SiLU())
        self.gate=nn.Conv2d(width*2,1,1) if mode in {'gate','gain'} else None
        self.out=nn.Conv2d(width,p3_channels,1)
        self.smoke_head=nn.Conv2d(width,1,1) if smoke_aux else None
        for module in [self.split,self.gate,self.out]:
            if module is not None:nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)

    def forward(self,p3,c2,auxiliary=None):
        auxiliary=self.training if auxiliary is None else auxiliary
        feature=self.project(p3)
        detail=F.adaptive_avg_pool2d(self.detail(c2),feature.shape[-2:]) if self.detail else torch.zeros_like(feature)
        a=torch.sigmoid(self.split(feature)) if self.split else feature.new_full((feature.shape[0],1,*feature.shape[-2:]),.5)
        smoke=self.smoke(a*feature);scene=self.scene((1-a)*feature)
        context=self.mix(torch.cat([smoke,scene],1))
        gate_logits=self.gate(torch.cat([feature+detail,context],1)) if self.gate else None
        gate=gate_logits.sigmoid() if gate_logits is not None else 1.
        result={'final':p3+self.out(detail+gate*context)}
        if auxiliary:
            result.update(local=p3+self.out(detail),context=p3+self.out(detail+context),
                          semantic_features=[smoke,scene,context],gate_logits=gate_logits)
            if self.smoke_head is not None:result['smoke_logits']=self.smoke_head(smoke)
        return result

    def strip_training_heads(self):
        self.smoke_head=None


@contextmanager
def no_extra_bn_updates(module):
    """Auxiliary forwards use current-batch statistics without a second running-stat update."""
    changed=[(m,m.track_running_stats) for m in module.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    try:
        for m,_ in changed:m.track_running_stats=False
        yield
    finally:
        for m,value in changed:m.track_running_stats=value


def gain_loss(gate_logits,local_terms,context_terms,valid_fire_p3):
    delta=(local_terms-context_terms).detach()
    valid=valid_fire_p3.bool() & torch.isfinite(delta) & (delta!=0)
    logits=gate_logits.flatten(1)
    if logits.shape!=delta.shape:raise ValueError('Gain targets must align with the P3 grid')
    return masked_bce(logits,(delta>0).to(logits.dtype),valid)


def smoke_box_mask(batch,height,width):
    """Weak box-derived target on the current transformed view, not pixel GT."""
    device=batch['img'].device;size=batch['img'].shape[0]
    ys=(torch.arange(height,device=device)+.5)/height;xs=(torch.arange(width,device=device)+.5)/width
    yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    target=torch.zeros((size,1,height,width),device=device)
    for image,cls,box in zip(batch['batch_idx'].long(),batch['cls'].flatten().long(),batch['bboxes']):
        if int(cls)!=1:continue
        x,y,w,h=box
        target[image,0]=torch.maximum(target[image,0],((xx>=x-w/2)&(xx<=x+w/2)&(yy>=y-h/2)&(yy<=y+h/2)).float())
    return target


def semantic_loss(logits,targets,mask,clean,warmup,presence_weight=.1,semantic_weight=.05):
    effective=mask.bool() & clean.bool()[:,None]
    p=masked_bce(logits[:,:2],targets[:,:2],effective[:,:2])
    s=masked_bce(logits[:,2:],targets[:,2:],effective[:,2:])
    return warmup*(presence_weight*p+semantic_weight*s),{'presence':p.detach(),'semantic':s.detach()}


def joint_presence_loss(logits,targets,mask,clean,warmup,presence_weight=.1,joint_weight=.05):
    """Same hard-GT presence term and four-state CE capacity for all three student controls."""
    effective=mask.bool() & clean.bool()[:,None]
    presence=masked_bce(logits[:,:2],targets[:,:2],effective[:,:2])
    active=effective[:,2:].all(dim=1)
    per_image=-(targets[:,2:]*F.log_softmax(logits[:,2:].float(),dim=-1)).sum(dim=-1)
    joint=(per_image*active.to(per_image.dtype)).sum()/active.sum().clamp_min(1)
    return warmup*(presence_weight*presence+joint_weight*joint),{'presence':presence.detach(),'joint':joint.detach()}
