import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fire_r2.common import ProjectError
from fire_r2.teacher_holdout_resume import resume_answers,owns_batch,merge_remainders


class ResumeTests(unittest.TestCase):
    def test_merge_requires_complete_union_and_exact_agreement_on_reused_answers(self):
        now,rows,prior=self.fixture()
        a={**prior,'status':'completed_shard','responses':prior['responses'][:3]}
        b={**prior,'status':'completed_shard','responses':prior['responses'][2:]}
        self.assertEqual(merge_remainders(prior,[a,b],rows),prior['responses'])
        with self.assertRaises(ProjectError):merge_remainders(prior,[a],rows)
        b=copy.deepcopy(b);b['responses'][0]['raw_response']='changed'
        with self.assertRaises(ProjectError):merge_remainders(prior,[a,b],rows)

    def test_disjoint_whole_batches_and_explicit_interruption_only(self):
        shards=[{i for i in range(0,256,4) if owns_batch(i,4,s,2)} for s in (0,1)]
        self.assertFalse(shards[0]&shards[1]);self.assertEqual(shards[0]|shards[1],set(range(0,256,4)))
        now,rows,prior=self.fixture();prior.update(error='',traceback='traceback\nKeyboardInterrupt')
        with self.assertRaises(ProjectError):resume_answers(prior,now,{'model':'2b'},rows)
        self.assertEqual(len(resume_answers(prior,now,{'model':'2b'},rows,True)),4)

    def fixture(self):
        now={'device':'cpu','dtype':'float32','threads':4,'batch':4,'instruction_sha256':'m'}
        rows=[{'sample_id':str(i),'task':'region_class','original_annotation_id':i,'instruction_sha256':'p','target':{'target_class':'fire'}} for i in range(4)]
        calls=[{'sample_id':r['sample_id'],'task':r['task'],'annotation_id':r['original_annotation_id'],'prompt_sha256':'p',
                'target_from_original_annotation':r['target'],'answer':r['target'],'raw_response':json.dumps(r['target'])} for r in rows]
        return now,rows,{**now,'identity':{'model':'2b'},'status':'failed','error':'timed out','responses':calls}

    def test_same_complete_batch_retains_answers_and_rejects_model_or_prompt_changes(self):
        now,rows,prior=self.fixture();before=copy.deepcopy(prior)
        self.assertEqual(len(resume_answers(prior,now,{'model':'2b'},rows)),4);self.assertEqual(prior,before)
        with self.assertRaises(ProjectError):resume_answers(prior,now,{'model':'other'},rows)
        rows[0]['instruction_sha256']='changed'
        with self.assertRaises(ProjectError):resume_answers(prior,now,{'model':'2b'},rows)

    def test_running_partial_batch_and_non_timeout_failures_are_rejected(self):
        now,rows,prior=self.fixture();prior['status']='running'
        with self.assertRaises(ProjectError):resume_answers(prior,now,{'model':'2b'},rows)
        prior['status']='failed';prior['responses'].pop()
        with self.assertRaises(ProjectError):resume_answers(prior,now,{'model':'2b'},rows)
        prior['error']='out of memory'
        with self.assertRaises(ProjectError):resume_answers(prior,now,{'model':'2b'},rows)
