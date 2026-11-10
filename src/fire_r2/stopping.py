"""Validation progress, not a prescribed epoch count, controls training completion."""
import math
from .common import ProjectError

TERMINAL_TRAINING={'completed','completed_or_budget_stopped','early_stopped','stopped_by_user','budget_stopped'}


class ValidationProgress:
    def __init__(self,patience=10,min_delta=.001,mode='max'):
        if patience<1 or min_delta<0 or mode not in ('min','max'):raise ProjectError('Invalid progress stopping policy')
        self.patience=patience;self.min_delta=min_delta;self.mode=mode
        self.reference=None;self.last_improvement=0;self.last_index=0

    def observe(self,index,value):
        if not math.isfinite(value):raise ProjectError('Nonfinite validation metric')
        if index<=self.last_index:return False
        improvement=self.reference is None or (value-self.reference if self.mode=='max' else self.reference-value)>self.min_delta
        if improvement:self.reference=value;self.last_improvement=index
        self.last_index=index
        return index-self.last_improvement>=self.patience

    def snapshot(self):
        return {'patience':self.patience,'min_delta':self.min_delta,'mode':self.mode,
                'reference':self.reference,'last_improvement':self.last_improvement,'last_index':self.last_index}


def require_evaluable_training(record,completed_epochs,checkpoint_exists):
    if completed_epochs<1 or not checkpoint_exists:raise ProjectError('A saved evaluated checkpoint is required')
    if record.get('status') not in TERMINAL_TRAINING:raise ProjectError('Training must be finalized or explicitly stopped before this evaluation')
    return True


def training_validation_epoch(trainer):
    """Ultralytics final_eval fires fit-end again at epoch+1, without another training epoch."""
    if not getattr(trainer.validator,'training',True):return None
    return int(trainer.epoch)+1
