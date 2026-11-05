"""Tiny original-GT fixtures; no real images, model weights, optimizer or training."""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest
import tempfile
import json

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'tmp').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError
from fire_r2.supervision import (SEMANTIC_FIELDS,SCENE_PROMPT,REGION_PROMPT,digest_object,
    prepare_supervision,selected_regions,semantic_targets,quality_score,normalize_weights,teacher_jobs,parse_advice,consistent_batch)
from fire_r2.evaluation import (iou_xywh,calibrate,evaluate,coco_metrics,validate_predictions,
    evaluation_rows,operating_curve,operating_point,select_checkpoint)
from fire_r2.experiments import factor_config,experiment_matrix


def row(ident='x',role='fit',state='fire_only'):
    classes={'fire_only':[0],'smoke_only':[1],'both':[0,1],'empty':[]}[state]
    return {'sample_id':f'dfire/{role}/{ident}','file':ident+'.png','image_relpath':ident+'.png',
            'official_split':'train' if role=='fit' else role,'role':role,'width':100,'height':100,
            'state':state,'has_small_fire':False,'boxes':[{'class_id':c,'original_class_id':c,
            'original_annotation_id':i,'bbox':[10.,10.,40.,40.],'format':'xywh_absolute','area':1600.,'iscrowd':0} for i,c in enumerate(classes)]}


def cache_for(rows):
    semantic={f:'no' for f in SEMANTIC_FIELDS}
    return {'manifest_sha256':'manifest','teacher_id':'Qwen/Qwen3.5-0.8B','teacher_revision':'fixed-revision',
        'prompt_hashes':{'A':hashlib.sha256(SCENE_PROMPT.encode()).hexdigest(),'B':hashlib.sha256(REGION_PROMPT.encode()).hexdigest()},
        'samples':{r['sample_id']:{'image_sha256':'a'*64,'semantic':semantic.copy(),
            'regions':[{'original_annotation_id':b['original_annotation_id'],'response':{'content':'fire' if b['class_id']==0 else 'smoke','visibility':'clear'}} for b in selected_regions(r)]} for r in rows if r['role']=='fit'}}


def predictions(rows,detections):
    return {'checkpoint_sha256':'checkpoint','training_source':'dfire',
            'selection_sha256':digest_object(sorted(r['sample_id'] for r in rows)),
            'images':[{'sample_id':r['sample_id'],'detections':d} for r,d in zip(rows,detections)]}


class SupervisionTests(unittest.TestCase):
    def test_mock_teacher_pipeline_uses_images_only_and_keeps_failures(self):
        from PIL import Image
        from fire_r2.teacher_runner import build_cache
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            p=Path(directory)/'x.png';Image.new('RGB',(100,100),'gray').save(p)
            r=row();calls=[]
            def infer(view,prompt):
                calls.append(prompt);self.assertEqual(view.size,(640,640));self.assertNotIn(r['sample_id'],prompt)
                if prompt==SCENE_PROMPT:return json.dumps({f:'uncertain' for f in SEMANTIC_FIELDS}),12
                return 'unparseable',3
            cache=build_cache([r,row('holdout','test')],'manifest',directory,infer,'fixed')
            self.assertEqual(len(calls),2);self.assertEqual(len(cache['samples']),1)
            result=prepare_supervision([r],'manifest',cache)
            self.assertEqual(result['samples'][r['sample_id']]['weight'],1.)

    def test_teacher_cache_image_digest_is_bound_to_verified_image(self):
        rows=[row()];rows[0]['image_sha256']='b'*64
        with self.assertRaises(ProjectError):prepare_supervision(rows,'manifest',cache_for(rows))
    def test_weights_positive_group_mass_and_no_gt_change(self):
        rows=[row('a'),row('b'),row('c',state='empty')];original=copy.deepcopy(rows);cache=cache_for(rows)
        cache['samples'][rows[1]['sample_id']]['regions'][0]['response']['content']='other'
        result=prepare_supervision(rows,'manifest',cache)
        weights=[result['samples'][r['sample_id']]['weight'] for r in rows]
        self.assertEqual(weights,[1.25,.75,1.]);self.assertEqual(rows,original)

    def test_failure_is_neutral_and_uncertain_semantics_are_masked(self):
        r=row();item={'semantic':{'fire_visible':'yes','smoke_visible':'yes',**{f:'uncertain' for f in SEMANTIC_FIELDS[2:]}},'regions':[]}
        self.assertEqual(quality_score(r,item),1.)
        target=semantic_targets(r,item)
        self.assertEqual(target['targets'][:2],[1.,0.])
        self.assertEqual(target['mask'],[True,True,False,False,False,False])

    def test_cache_rejects_test_leakage_and_stale_manifest(self):
        rows=[row()];cache=cache_for(rows);cache['samples']['dfire/test/leak']={}
        with self.assertRaises(ProjectError):prepare_supervision(rows,'manifest',cache)
        with self.assertRaises(ProjectError):prepare_supervision(rows,'changed',cache_for(rows))

    def test_zero_weight_and_unknown_dimensions_refused(self):
        r=row()
        with self.assertRaises(ProjectError):normalize_weights({r['sample_id']:0},[r])
        r['has_small_fire']=None
        with self.assertRaises(ProjectError):prepare_supervision([r],'x',variant='uniform')

    def test_shuffle_preserves_weight_multiset_and_every_sample(self):
        rows=[row(str(i)) for i in range(8)];cache=cache_for(rows)
        for i,r in enumerate(rows):
            if i%2:cache['samples'][r['sample_id']]['regions'][0]['response']['content']='other'
        a=prepare_supervision(rows,'manifest',cache);b=prepare_supervision(rows,'manifest',cache,'shuffle')
        self.assertEqual(sorted(x['weight'] for x in a['samples'].values()),sorted(x['weight'] for x in b['samples'].values()))
        self.assertEqual(set(a['samples']),set(b['samples']))

    def test_teacher_jobs_fit_only_and_degenerate_roi_not_queried(self):
        r=row();r['boxes'][0]['bbox'][2]=0
        jobs=teacher_jobs([r,row('heldout','test')],'m')['jobs']
        self.assertEqual(len(jobs),1);self.assertEqual(jobs[0]['task'],'A')
        self.assertNotIn(r['sample_id'],jobs[0]['prompt'])

    def test_schema_does_not_accept_extra_fields(self):
        raw={f:'yes' for f in SEMANTIC_FIELDS};raw['class']='fire'
        self.assertIsNone(parse_advice(raw,'A')[0])

    def test_single_json_fence_is_transport_syntax_not_new_semantics(self):
        raw={f:'uncertain' for f in SEMANTIC_FIELDS}
        answer,status=parse_advice('```json\n'+json.dumps(raw)+'\n```','A')
        self.assertEqual((answer,status),(raw,'ok'))
        self.assertIsNone(parse_advice('Explanation before JSON '+json.dumps(raw),'A')[0])

    def test_schedule_one_quarter_batches(self):
        self.assertEqual([i for i in range(12) if consistent_batch(i)],[0,4,8])

    def test_semantic_shuffle_keeps_presence_and_unknown_masks(self):
        rows=[row(str(i)) for i in range(4)];cache=cache_for(rows)
        for i,r in enumerate(rows):cache['samples'][r['sample_id']]['semantic']['cloud_fog_like']='yes' if i%2 else 'no'
        a=prepare_supervision(rows,'manifest',cache,'uniform');b=prepare_supervision(rows,'manifest',cache,'uniform',shuffle_semantic=True)
        self.assertEqual(sorted(x['targets'][2] for x in a['samples'].values()),sorted(x['targets'][2] for x in b['samples'].values()))
        self.assertTrue(all(x['targets'][:2]==[1.,0.] for x in b['samples'].values()))


class EvaluationTests(unittest.TestCase):
    def test_iou_zero_area_and_unclipped_border(self):
        self.assertEqual(iou_xywh([0,0,0,1],[0,0,1,1]),0.)
        self.assertAlmostEqual(iou_xywh([-1,0,2,2],[0,0,2,2]),1/3)

    def test_calibration_ties_threshold_and_test_leakage(self):
        rows=[row('a','cal'),row('b','cal','empty')]
        artifact=predictions(rows,[[{'class_id':0,'bbox':[10,10,40,40],'score':.8}],[{'class_id':0,'bbox':[0,0,10,10],'score':.4}]])
        cal=calibrate(rows,artifact,'dfire');self.assertEqual(cal['threshold'],.8)
        rows[0]['role']='test'
        with self.assertRaises(ProjectError):calibrate(rows,artifact,'dfire')

    def test_false_alarm_denominator_includes_smoke_only(self):
        rows=[row('a','test'),row('b','test','smoke_only'),row('c','test','empty')]
        p={r['sample_id']:[] for r in rows};p[rows[1]['sample_id']]=[{'class_id':0,'bbox':[1,1,5,5],'score':.8}]
        result=operating_point(rows,p,.5)
        self.assertEqual(result['original_negative_images'],2)
        self.assertEqual(result['negative_trigger_rate'],.5)
        self.assertEqual(result['fp_per_image'],1/3)

    def test_missing_prediction_image_is_error_not_silent_empty(self):
        rows=[row('a','test'),row('b','test')]
        with self.assertRaises(ProjectError):validate_predictions(rows,predictions(rows,[[]]))

    def test_same_source_test_and_cross_source_all(self):
        rows=[row('a','fit'),row('b','dev'),row('c','cal'),row('d','test')]
        self.assertEqual(len(evaluation_rows(rows,'dfire','dfire')),1)
        self.assertEqual(len(evaluation_rows(rows,'homefire','dfire')),4)
        with self.assertRaises(ProjectError):evaluation_rows(rows,'homefire','dfire','cal')

    def test_checkpoint_earliest_equal_dev_score(self):
        rows=[{'partition':'dev','epoch':2,'fire_AP50_95':.4},{'partition':'dev','epoch':1,'fire_AP50_95':.4}]
        self.assertEqual(select_checkpoint(rows)['epoch'],1)

    def test_duplicate_detection_is_false_positive(self):
        r=row('a','test');det={'class_id':0,'bbox':[10,10,40,40],'score':.8}
        curve=operating_curve([r],{r['sample_id']:[det,dict(det,score=.7)]})
        self.assertEqual(curve['points'][-1]['tp'],1);self.assertEqual(curve['points'][-1]['fp'],1)

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'COCO backend tested in the existing cloud environment')
    def test_official_coco_perfect_and_empty_predictions(self):
        r=row('a','test');p={r['sample_id']:[{'class_id':0,'bbox':[10,10,40,40],'score':.8}]}
        self.assertAlmostEqual(coco_metrics([r],p)['classes']['fire']['AP50_95'],1.)
        self.assertEqual(coco_metrics([r],{r['sample_id']:[]})['classes']['fire']['AP50_95'],0.)

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'),'COCO backend tested in cloud')
    def test_reference_size_small_metric_does_not_change_geometry(self):
        r=row('a','test');r['width']=r['height']=1000;original=copy.deepcopy(r)
        p={r['sample_id']:[{'class_id':0,'bbox':[10,10,40,40],'score':.8}]}
        self.assertAlmostEqual(coco_metrics([r],p,True)['classes']['fire']['AP50_95'],1.)
        self.assertEqual(r,original)
        r['width']=r['height']=100
        self.assertIsNone(coco_metrics([r],p,True)['classes']['fire']['AP50_95'])

    def test_factor_menu_preserves_pairwise_and_ordinary_controls(self):
        matrix=experiment_matrix({'dataset':'dfire','seed':17})
        keys={x['experiment'] for x in matrix['experiments']}
        self.assertTrue({'B','B+W','B+S','B+W+S','M-ordinary','M','M-noD','M-noGain'}<=keys)
        self.assertEqual(factor_config('B+P')['semantic'],'presence')
