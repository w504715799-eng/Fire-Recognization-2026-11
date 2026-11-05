"""Cloud validator: preserve original manifest GT and select checkpoints on fire dev AP."""
from copy import copy
import hashlib
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionValidator, DetectionTrainer

from .common import ProjectError, write_json
from .supervision import load_manifest
from .evaluation import coco_metrics


class OriginalGTValidator(DetectionValidator):
    manifest_file=None

    def init_metrics(self,model):
        super().init_metrics(model)
        if not hasattr(self,'original_rows'):
            self.original_rows=load_manifest(self.manifest_file,roles={'dev'})
        self.by_stem={Path(r['view_image']).stem:r for r in self.original_rows}
        self.original_predictions={}

    def update_metrics(self,preds,batch):
        for i,pred in enumerate(preds):
            row=self.by_stem.get(Path(batch['im_file'][i]).stem)
            if row is None:raise ProjectError('Validation loader contains a non-dev image')
            pbatch=self._prepare_batch(i,batch)
            scaled=self.scale_preds(pred,pbatch)
            detections=[]
            for box,score,cls in zip(scaled['bboxes'].tolist(),scaled['conf'].tolist(),scaled['cls'].tolist()):
                x1,y1,x2,y2=box
                detections.append({'bbox':[x1,y1,x2-x1,y2-y1],'class_id':int(cls),'score':float(score)})
            self.original_predictions[row['sample_id']]=detections
        super().update_metrics(preds,batch)

    def get_stats(self):
        stats=super().get_stats()
        if set(self.original_predictions)!={r['sample_id'] for r in self.original_rows}:
            raise ProjectError('Original dev image coverage differs from validation loader')
        self.original_report=coco_metrics(self.original_rows,self.original_predictions)
        score=self.original_report['classes']['fire']['AP50_95']
        if score is None:raise ProjectError('Original dev has no defined fire AP')
        stats['metrics/fire_AP50_95_original']=score
        stats['fitness']=score
        return stats


class UnifiedSelectionMixin:
    def get_validator(self):
        validator=OriginalGTValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
        validator.manifest_file=self.manifest_file
        return validator

    def validate(self):
        metrics,fitness=super().validate()
        if metrics is not None:
            record={'epoch':int(self.epoch),'partition':'dev','fire_AP50_95':float(fitness),
                    'evaluation':self.validator.original_report}
            write_json(self.save_dir/'original-dev'/f'epoch-{self.epoch:04d}.json',record)
        return metrics,fitness

    def save_model(self):
        # Upstream overwrites best.pt on equal fitness; preserve the earlier epoch on ties.
        is_tie=getattr(self,'_r2_saved_best',None)==self.fitness and self.best.exists()
        old_best_path=self.best
        if is_tie:self.best=self.last
        try:result=super().save_model()
        finally:self.best=old_best_path
        if not is_tie and self.best.exists() and self.best_fitness==self.fitness:self._r2_saved_best=self.fitness
        return result


class NativeUnifiedTrainer(UnifiedSelectionMixin,DetectionTrainer):
    manifest_file=None
