"""Cloud tensor checks only: random tensors, no checkpoint or optimizer step."""
import importlib.util
from pathlib import Path
import sys,platform
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
HAS_TORCH=platform.system()=='Linux' and importlib.util.find_spec('torch') is not None
HAS_BACKEND=HAS_TORCH and importlib.util.find_spec('ultralytics') is not None


@unittest.skipUnless(HAS_TORCH,'Use the already installed cloud PyTorch; do not install locally')
class TensorTests(unittest.TestCase):
    def test_weight_one_native_reduction_and_gradient(self):
        import torch
        from fire_r2.torch_modules import positive_weighted_reduction
        terms=torch.arange(24,dtype=torch.float32).reshape(2,4,3).requires_grad_();denom=torch.tensor(7.)
        result=positive_weighted_reduction(terms,torch.ones(2),denom)
        self.assertTrue(torch.equal(result,terms.sum((0,1))/denom))
        result.sum().backward();self.assertTrue(torch.allclose(terms.grad,torch.full_like(terms,1/7)))

    def test_fusion_starts_as_identity_and_semantic_head_shapes(self):
        import torch
        from fire_r2.torch_modules import FireSmokeFusion,SemanticHead
        m=FireSmokeFusion(16,8,width=8);p=torch.randn(2,16,5,7,requires_grad=True);c=torch.randn(2,8,10,14)
        out=m(p,c)
        self.assertTrue(torch.equal(out['final'],p));self.assertTrue(torch.equal(out['local'],p))
        logits=SemanticHead([8,8,8],width=8)(out['semantic_features']);self.assertEqual(tuple(logits.shape),(2,6))
        (out['final'].sum()+logits.sum()).backward();self.assertIsNotNone(p.grad)
        m.eval();self.assertEqual(set(m(p,c)),{'final'})

    def test_unknown_semantics_and_gain_targets_detached(self):
        import torch
        from fire_r2.torch_modules import masked_bce,gain_loss
        z=torch.zeros(1,2,requires_grad=True)
        loss=masked_bce(z,torch.tensor([[float('nan'),0.]]),torch.zeros(1,2,dtype=torch.bool))
        self.assertEqual(loss.item(),0.)
        local=torch.tensor([[2.,1.]],requires_grad=True);context=torch.tensor([[1.,1.]],requires_grad=True)
        gate=torch.zeros(1,1,1,2,requires_grad=True)
        gain_loss(gate,local,context,torch.ones(1,2,dtype=torch.bool)).backward()
        self.assertIsNone(local.grad);self.assertIsNone(context.grad)
        self.assertLess(gate.grad.flatten()[0].item(),0.);self.assertEqual(gate.grad.flatten()[1].item(),0.)

    def test_extra_head_forward_does_not_update_bn_twice(self):
        import torch
        from fire_r2.torch_modules import no_extra_bn_updates
        m=torch.nn.BatchNorm2d(2);x=torch.randn(2,2,3,3)
        m(x);mean=m.running_mean.clone();count=m.num_batches_tracked.clone()
        with no_extra_bn_updates(m):m(x+10)
        self.assertTrue(torch.equal(m.running_mean,mean));self.assertTrue(torch.equal(m.num_batches_tracked,count))


@unittest.skipUnless(HAS_BACKEND,'Pinned backend checks run only in cloud')
class BackendTests(unittest.TestCase):
    def make_model(self):
        import torch
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import YAML,ROOT as ULTRA_ROOT,DEFAULT_CFG_DICT,IterableSimpleNamespace
        cfg=YAML.load(ULTRA_ROOT/'cfg/models/26/yolo26.yaml');cfg['scale']='n'
        model=DetectionModel(cfg,nc=2,verbose=False)
        model.args=IterableSimpleNamespace(**{**DEFAULT_CFG_DICT,'epochs':10})
        return model

    def test_decomposition_matches_native_empty_and_positive(self):
        import torch
        from fire_r2.ultralytics_backend import DecomposedLoss
        from ultralytics.utils.loss import v8DetectionLoss
        from fire_r2.torch_modules import positive_weighted_reduction
        torch.set_num_threads(2);model=self.make_model();model.train()
        preds=model(torch.randn(2,3,64,64))['one2many']
        for empty in (False,True):
            batch={'batch_idx':torch.tensor([] if empty else [0.,1.]),'cls':torch.empty(0,1) if empty else torch.tensor([[0.],[1.]]),
                   'bboxes':torch.empty(0,4) if empty else torch.tensor([[.5,.5,.4,.4],[.3,.3,.2,.2]])}
            native=v8DetectionLoss(model).loss(preds,batch)[0]
            terms,denom,_=DecomposedLoss(model).terms(preds,batch)
            adapted=positive_weighted_reduction(terms,torch.ones(2),denom)*2
            self.assertTrue(torch.allclose(native,adapted,atol=2e-5,rtol=2e-5),(native,adapted))

    def test_research_graph_and_shared_assignment_backward(self):
        import torch
        from fire_r2.ultralytics_backend import ResearchDetectionModel
        from fire_r2.experiments import factor_config
        baseline=self.make_model()
        model=ResearchDetectionModel(baseline.yaml,nc=2,verbose=False);model.args=baseline.args
        model.configure_research(factor_config('M+W+S'));model.train()
        batch={'img':torch.rand(2,3,64,64),'batch_idx':torch.tensor([0.,1.]),'cls':torch.tensor([[0.],[1.]]),
               'bboxes':torch.tensor([[.5,.5,.4,.4],[.3,.3,.2,.2]]),'r2_clean':torch.tensor([True,True]),
               'r2_weights':torch.tensor([.8,1.2]),'r2_targets':torch.zeros(2,6),'r2_mask':torch.ones(2,6,dtype=torch.bool),'r2_warmup':.2}
        loss,items=model(batch);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertIsNotNone(model.research_fusion.out.weight.grad)
        self.assertIsNotNone(model.research_semantic.output.weight.grad)
        self.assertIn('gain_loss',items)
        model.eval()
        with torch.no_grad():before=model(batch['img'])[0].clone()
        model.strip_training_heads()
        with torch.no_grad():out=model(batch['img'])
        self.assertTrue(torch.equal(before,out[0]))
        self.assertEqual(out[0].shape[-1],6)

    def test_view_sampler_keeps_every_image_and_whole_clean_batches(self):
        from fire_r2.ultralytics_backend import ViewBatchSampler
        sampler=ViewBatchSampler(9,2,17);batches=list(sampler)
        self.assertEqual(sorted(i for batch in batches for i,_ in batch),list(range(9)))
        self.assertTrue(all(len({v for _,v in batch})==1 for batch in batches))
        self.assertEqual([i for i,b in enumerate(batches) if b[0][1]],[0,4])

    def test_native_r2_manifest_dataset_wrapper(self):
        import copy,json,tempfile
        from test_data import DataContractTests
        from fire_r2.data import load_dataset,prepare_view
        from fire_r2.supervision import load_manifest,prepare_supervision
        from fire_r2.ultralytics_backend import ViewDataset
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils import DEFAULT_CFG
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);cfg,ann,images=DataContractTests().fixture(root)
            # Use valid synthetic GT to exercise the actual pinned loader, not filtering rules.
            for p in ann.glob('*.json'):
                obj=json.loads(p.read_text())
                for a in obj['annotations']:a.update(bbox=[1,1,4,4],area=16)
                p.write_text(json.dumps(obj))
            rows,summary=load_dataset(root,cfg,'demo');prepare_view(root,rows,summary,'artifacts/data/demo')
            view=root/'artifacts/data/demo';rows=load_manifest(view/'manifest.jsonl')
            base=YOLODataset(img_path=str(view/'fit.txt'),imgsz=64,batch_size=1,augment=True,
                            hyp=copy.deepcopy(DEFAULT_CFG),data={'names':{0:'fire',1:'smoke'},'nc':2,'channels':3})
            dataset=ViewDataset(base,prepare_supervision(rows,'m',variant='uniform'),rows)
            batch=dataset.collate_fn([dataset[(0,True)]])
            self.assertEqual(tuple(batch['img'].shape),(1,3,64,64))
            self.assertTrue(batch['r2_clean'].item());self.assertEqual(batch['r2_weights'].item(),1.)
            self.assertEqual(len(dataset.by_file),1)

    def test_cpu_trainer_checkpoint_and_original_gt_validator(self):
        import copy,json,tempfile,torch
        from test_data import DataContractTests
        from fire_r2.data import load_dataset,prepare_view
        from fire_r2.supervision import load_manifest,prepare_supervision
        from fire_r2.ultralytics_backend import ResearchTrainer
        from fire_r2.unified_validator import UnifiedSelectionMixin
        from fire_r2.experiments import factor_config
        from ultralytics.utils import YAML,ROOT as ULTRA_ROOT
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);cfg,ann,images=DataContractTests().fixture(root)
            for p in ann.glob('*.json'):
                obj=json.loads(p.read_text())
                for a in obj['annotations']:a.update(bbox=[1,1,4,4],area=16)
                p.write_text(json.dumps(obj))
            rows,summary=load_dataset(root,cfg,'demo');prepare_view(root,rows,summary,'artifacts/data/demo')
            view=root/'artifacts/data/demo';manifest=view/'manifest.jsonl';rows=load_manifest(manifest)
            supervision=root/'uniform.json';supervision.write_text(json.dumps(prepare_supervision(rows,'m',variant='uniform')))
            model_cfg=YAML.load(ULTRA_ROOT/'cfg/models/26/yolo26.yaml');model_cfg['scale']='n'
            model_yaml=root/'model.yaml';YAML.save(model_yaml,model_cfg)
            class TinyTrainer(UnifiedSelectionMixin,ResearchTrainer):pass
            TinyTrainer.manifest_file=str(manifest);TinyTrainer.supervision_file=str(supervision)
            TinyTrainer.research_config=factor_config('M+P')
            trainer=TinyTrainer(overrides={'model':str(model_yaml),'data':str(view/'train.yaml'),
                'device':'cpu','imgsz':64,'batch':1,'epochs':1,'workers':0,'amp':False,'plots':False,
                'pretrained':False,'project':str(root/'run'),'name':'tiny','save':True,'val':True,
                'mosaic':0.,'mixup':0.,'close_mosaic':0,'verbose':False})
            trainer.train()
            self.assertTrue(trainer.best.is_file())
            report=json.loads((trainer.save_dir/'original-dev/epoch-0000.json').read_text())
            self.assertEqual(report['partition'],'dev')
            self.assertIsNotNone(report['fire_AP50_95'])
