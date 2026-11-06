import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));(ROOT/'tmp').mkdir(exist_ok=True)
from fire_r2.common import ProjectError
from fire_r2.teacher_audit import normalize_syntax,audit_cache,human_review_template,cpu_diagnostic_cache
from fire_r2.supervision import SEMANTIC_FIELDS,SCENE_PROMPT,REGION_PROMPT
from fire_r2.teacher_runner import build_cache,build_batched_cache
from test_research import row


class AuditTests(unittest.TestCase):
    def test_incomplete_or_gpu_reports_cannot_be_recast_as_complete_cpu_audit(self):
        with self.assertRaises(ProjectError):cpu_diagnostic_cache({'status':'stopped'})
        with self.assertRaises(ProjectError):cpu_diagnostic_cache({'status':'completed','execution_device':'cuda'})

    def test_syntax_only_recovers_singleton_and_bare_enums(self):
        text='```json\n['+'{'+','.join(json.dumps(k)+':yes' for k in SEMANTIC_FIELDS)+'}'+']\n```'
        answer,changes=normalize_syntax(text,'A')
        self.assertEqual(answer,{k:'yes' for k in SEMANTIC_FIELDS})
        self.assertIn('quote_exact_bare_enums',changes);self.assertIn('unwrap_singleton_object_array',changes)

    def test_invalid_semantics_missing_keys_and_duplicates_are_not_repaired(self):
        self.assertIsNone(normalize_syntax('{"content":"1","visibility":"clear"}','B')[0])
        self.assertIsNone(normalize_syntax('{"content":"smoke"}','B')[0])
        self.assertIsNone(normalize_syntax('{"content":"fire","content":"smoke","visibility":"clear"}','B')[0])
        self.assertIsNone(normalize_syntax('[{},{}]','A')[0])

    def test_cache_not_mutated_and_label_agreement_not_human_truth(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            r=row();Image.new('RGB',(100,100)).save(Path(tmp)/r['image_relpath'])
            def infer(view,prompt):
                return (json.dumps([{k:'no' for k in SEMANTIC_FIELDS}]),10) if prompt==SCENE_PROMPT else ('{"content":"fire","visibility":"limited"}',8)
            cache=build_cache([r],'m',tmp,infer,'fixed');before=copy.deepcopy(cache)
            report=audit_cache(cache,[r],{r['sample_id']})
            self.assertEqual(cache,before)
            self.assertEqual(report['format']['A']['recorded_schema_pass'],0)
            self.assertEqual(report['format']['A']['syntax_normalized_pass'],1)
            self.assertEqual(report['overall']['presence']['fire_visible']['agree'],0)
            self.assertEqual(report['overall']['known_context_fields_from_original_cache'],0)
            self.assertEqual(report['semantic_accuracy'],'unmeasured_until_independent_review')
            self.assertEqual(report['per_image'][0]['raw_score'],1.)
            cache['samples'][r['sample_id']]['calls'][-1]['original_annotation_id']=99
            with self.assertRaises(ProjectError):audit_cache(cache,[r],set())

    def test_manual_review_is_pending_and_blind_to_teacher(self):
        r=row();template=human_review_template([r],{r['sample_id']})
        self.assertFalse(template['independent_review_completed'])
        self.assertTrue(all(x['reviewer_id'] is None and x['content'] is None for x in template['records']))
        self.assertTrue(all('teacher_answer' not in x for x in template['records']))

    def test_distinct_pixel_responses_stay_attached_to_batch_image_ids(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            rows=[row('red'),row('blue',state='smoke_only')]
            for r,color in zip(rows,('red','blue')):Image.new('RGB',(100,100),color).save(Path(tmp)/r['image_relpath'])
            def infer(views,prompt,task):
                results=[]
                for view in views:
                    red=view.getpixel((320,320))[0]>view.getpixel((320,320))[2]
                    answer={k:'no' for k in SEMANTIC_FIELDS} if task=='A' else {'content':'fire' if red else 'smoke','visibility':'clear'}
                    if task=='A':answer['fire_visible']='yes' if red else 'no';answer['smoke_visible']='no' if red else 'yes'
                    results.append((json.dumps(answer),10))
                return results
            cache=build_batched_cache(rows,'m',tmp,infer,'fixed',2)
            self.assertEqual(cache['samples'][rows[0]['sample_id']]['regions'][0]['response']['content'],'fire')
            self.assertEqual(cache['samples'][rows[1]['sample_id']]['regions'][0]['response']['content'],'smoke')
            result=audit_cache(cache,rows,{rows[0]['sample_id']})
            self.assertEqual(result['cohorts']['original_8']['images'],1)
            self.assertEqual(result['cohorts']['new_fit']['images'],1)
