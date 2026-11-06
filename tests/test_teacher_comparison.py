import copy,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError
from fire_r2.teacher_comparison import paired_answers,query_cost,compare_teacher_models


def cache():
    return {'teacher_revision':'fixed','prompt_hashes':{'B':'fixed'},'samples':{
        str(i):{'image_sha256':str(i),'calls':[{'task':'B','original_annotation_id':1,
            'raw_response':'{"content":"smoke","visibility":"clear"}','seconds':.5,'batch_seconds':2.,
            'batch_size':4,'output_tokens':13}]} for i in range(4)}}


class ComparisonTests(unittest.TestCase):
    def test_batch_wall_time_not_counted_four_times(self):
        self.assertEqual(query_cost(cache())['query_wall_seconds'],2.)

    def test_invalid_pairs_do_not_count_as_correct_semantics(self):
        a=cache();b=copy.deepcopy(a)
        for c in (a,b):c['samples']['0']['calls'][0]['raw_response']='bad'
        r=paired_answers(a,b);self.assertEqual(r['exact_valid_answers'],3);self.assertEqual(r['both_schema_valid'],3)

    def test_same_query_binding_and_explicit_protocol_change(self):
        a=cache();b=copy.deepcopy(a);b['prompt_hashes']={'B':'changed'}
        with self.assertRaises(ProjectError):paired_answers(a,b)
        self.assertEqual(paired_answers(a,b,allow_protocol_change=True)['queries'],4)
        b['samples']['0']['image_sha256']='changed'
        with self.assertRaises(ProjectError):paired_answers(a,b,allow_protocol_change=True)

    def test_cross_model_path_preserves_view_and_identity_bindings(self):
        a=cache();b=cache()
        for c,name in [(a,'model_a'),(b,'model_b')]:
            c.update(teacher_id=name,manifest_sha256='same')
            for r in c['samples'].values():r['calls'][0]['view_sha256']='view'
        self.assertEqual(compare_teacher_models(a,b)['identical_answers'],4)
        b['samples']['0']['calls'][0]['view_sha256']='changed'
        with self.assertRaises(ProjectError):compare_teacher_models(a,b)
