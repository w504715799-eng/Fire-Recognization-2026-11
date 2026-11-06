import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));(ROOT/'tmp').mkdir(exist_ok=True)
from fire_r2.common import ProjectError
from fire_r2.teacher_protocols import AnswerTrie,valid_answers,protocol
from fire_r2.teacher_runner import build_batched_cache,teacher_view
from fire_r2.supervision import SEMANTIC_FIELDS,prepare_supervision
from test_research import row


class CharacterTokenizer:
    pad_token_id=0
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]


class TeacherProtocolTests(unittest.TestCase):
    def test_short_task_grammar_contains_all_candidates_independent_of_gt(self):
        values=['{"target_class":"fire"}','{"target_class":"smoke"}']
        trie=AnswerTrie(CharacterTokenizer(),'region_class',1,answers=values)
        self.assertEqual(trie.answer_count,2)
        for value in values:self.assertEqual(trie.allowed(CharacterTokenizer().encode(value)),[1])
        with self.assertRaises(ProjectError):AnswerTrie(CharacterTokenizer(),'region_class',1,answers=[])

    def test_every_legal_answer_is_reachable_no_invalid_enum_or_marker(self):
        for task,count in [('A',729),('B',15)]:
            trie=AnswerTrie(CharacterTokenizer(),task,1)
            self.assertEqual(trie.answer_count,count)
            for answer in valid_answers(task):
                prefix=[]
                for token in CharacterTokenizer().encode(answer):
                    self.assertIn(token,trie.allowed(prefix));prefix.append(token)
                self.assertEqual(trie.allowed(prefix),[1])
                self.assertEqual(trie.allowed(prefix+[1,0,0]),[1])
            with self.assertRaises(ProjectError):trie.allowed([ord('[')])
        self.assertNotIn('"1"',str(valid_answers('B')))

    def test_small_roi_has_no_magnified_number_after_resize(self):
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            p=Path(tmp)/'x.png';Image.new('RGB',(40,40),'black').save(p);before=p.read_bytes()
            box={'bbox':[17,17,4,4],'format':'xywh_absolute'}
            old=np.asarray(teacher_view(p,box))
            new=np.asarray(teacher_view(p,box,region_marker='thin_outline_after_resize'))
            self.assertLess((new[:,:,0]>200).sum(),(old[:,:,0]>200).sum()/5)
            self.assertEqual(p.read_bytes(),before)

    def test_batched_queries_preserve_fit_roi_coverage_and_unknown_fields(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            rows=[row(str(i),state=s) for i,s in enumerate(('fire_only','smoke_only','both','empty'))]
            for r in rows:Image.new('RGB',(100,100),'gray').save(Path(tmp)/r['image_relpath'])
            before=copy.deepcopy(rows);batches=[]
            def infer(views,prompt,task):
                batches.append((task,len(views)))
                answer={f:'uncertain' for f in SEMANTIC_FIELDS} if task=='A' else {'content':'uncertain','visibility':'unjudgeable'}
                return [(json.dumps(answer),12) for _ in views]
            cache=build_batched_cache(rows+[row('heldout','test')],'m',tmp,infer,'fixed',2,protocol_version='v2')
            self.assertEqual(len(cache['samples']),4);self.assertEqual(rows,before)
            self.assertEqual(sum(len(x['calls']) for x in cache['samples'].values()),8)
            self.assertTrue(any(n==2 for _,n in batches))
            result=prepare_supervision(rows,'m',cache)
            self.assertTrue(all(v['weight']==1 for v in result['samples'].values()))
            self.assertTrue(all(v['mask']==[True,True,False,False,False,False] for v in result['samples'].values()))

    def test_batch_failure_is_explicit_and_not_sample_deletion(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            r=row('x');Image.new('RGB',(100,100),'gray').save(Path(tmp)/r['image_relpath'])
            def infer(*args):raise RuntimeError('mock transport failure')
            cache=build_batched_cache([r],'m',tmp,infer,'fixed',4)
            calls=cache['samples'][r['sample_id']]['calls']
            self.assertEqual(len(calls),2)
            self.assertTrue(all(c['parse_status']=='technical_failure' and c['attempts']==2 for c in calls))
