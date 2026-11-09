import copy,sys,unittest,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fire_r2.common import ProjectError
from fire_r2.teacher_transfer import presence_marginals,joint_target,make_joint_control,make_transfer_supervision,validate_transfer,audit_joint_caches,PROTOCOL
from fire_r2.supervision import prepare_supervision
from test_research import row


class TransferTests(unittest.TestCase):
    def test_probability_audit_uses_full_joint_and_original_targets(self):
        r=row();perfect={'teacher':{'teacher_variant':'test'},'samples':{r['sample_id']:{'joint_probabilities':[0.,0.,1.,0.]}}}
        result=audit_joint_caches({'model':perfect},[r])['models']['model']
        self.assertEqual(result['joint_MAP_correct'],1);self.assertEqual(result['mean_Brier_sum_over_four_classes'],0.);self.assertEqual(result['mean_NLL_nats_floor_1e_12'],0.)
        perfect['samples'][r['sample_id']]['joint_probabilities']=[1.,0.,0.,0.]
        result=audit_joint_caches({'model':perfect},[r])['models']['model'];self.assertEqual(result['mean_Brier_sum_over_four_classes'],2.)

    def test_joint_marginals_and_gt_sign_are_preserved(self):
        self.assertEqual(presence_marginals([.1,.2,.3,.4]),[.7,.6000000000000001])
        self.assertEqual(joint_target([1.,0.]),[0.,0.,1.,0.])
        with self.assertRaises(ProjectError):presence_marginals([.1,.2,.3,.5])

    def test_fp32_roundoff_does_not_create_probability_above_one(self):
        p=presence_marginals([0.,0.,1.,3e-8])
        self.assertEqual(p[0],1.);self.assertTrue(all(0<=x<=1 for x in p))

    def test_all_images_preserved_and_teacher_identity_bound(self):
        rows=[row(str(i),state=s) for i,s in enumerate(('fire_only','smoke_only','both','empty'))]
        gt=prepare_supervision(rows,'m',variant='uniform');before=copy.deepcopy(gt);teacher={'teacher_variant':'adapted','adapter_sha256':'fixed'}
        cache={'scope':'full_source_fit','status':'completed','protocol':PROTOCOL,'manifest_sha256':'m','teacher':teacher,
               'samples':{r['sample_id']:{'joint_probabilities':[.1,.2,.3,.4],'presence_probabilities':[.7,.6],'image_sha256':'a'*64} for r in rows}}
        result=make_transfer_supervision(gt,cache,teacher);validate_transfer(result,rows,teacher)
        self.assertEqual(gt,before);self.assertEqual(set(result['samples']),set(gt['samples']));self.assertTrue(all(v['weight']==1 for v in result['samples'].values()))
        control=make_joint_control(gt);validate_transfer(control,rows,alpha_expected=0.)
        for ident,item in result['samples'].items():
            self.assertEqual(item['targets'][:2],gt['samples'][ident]['targets'][:2])
            index=item['original_joint'].index(1.);self.assertGreaterEqual(item['targets'][index+2],.75)
        with self.assertRaises(ProjectError):validate_transfer(result,rows,alpha_expected=0.)
        with self.assertRaises(ProjectError):make_transfer_supervision(gt,cache,{'teacher_variant':'base'})
        cache['scope']='pilot_fit_only'
        with self.assertRaises(ProjectError):make_transfer_supervision(gt,cache,teacher)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Run tensor check in the existing cloud sci-env; do not install Torch locally')
    def test_same_loss_budget_and_nonzero_teacher_gradient(self):
        import torch
        from fire_r2.torch_modules import joint_presence_loss
        logits=torch.tensor([[.2,-.3,.1,-.2,.3,0.]],requires_grad=True);gt=torch.tensor([[1.,0.,0.,0.,1.,0.]]);teacher=torch.tensor([[1.,0.,.1,.2,.3,.4]])
        mask=torch.ones((1,6),dtype=torch.bool);clean=torch.tensor([True])
        mixed=.75*gt+.25*teacher
        a=joint_presence_loss(logits,mixed,mask,clean,1.)[0]
        b=.75*joint_presence_loss(logits,gt,mask,clean,1.)[0]+.25*joint_presence_loss(logits,teacher,mask,clean,1.)[0]
        self.assertTrue(torch.allclose(a,b))
        ga=torch.autograd.grad(a,logits,retain_graph=True)[0];gb=torch.autograd.grad(b,logits,retain_graph=True)[0]
        self.assertTrue(torch.allclose(ga,gb));gg=torch.autograd.grad(joint_presence_loss(logits,gt,mask,clean,1.)[0],logits)[0]
        self.assertFalse(torch.allclose(ga,gg));self.assertTrue(torch.allclose(ga[:,:2],gg[:,:2]))
