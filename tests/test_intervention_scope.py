import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,sha256
from fire_r2.supervision import prepare_supervision,validate_supervision
from fire_r2.experiments import factor_config
from fire_r2.teacher_runner import build_batched_cache
from test_research import row,cache_for


class InterventionScopeTests(unittest.TestCase):
    def test_scene_only_supports_S_but_not_W(self):
        rows=[row()];cache=cache_for(rows);cache['tasks']=['A'];cache['prompt_hashes'].pop('B')
        for x in cache['samples'].values():x['regions']=[]
        result=prepare_supervision(rows,'manifest',cache,variant='uniform')
        self.assertTrue(all(result['samples'][rows[0]['sample_id']]['mask']))
        with self.assertRaises(ProjectError):prepare_supervision(rows,'manifest',cache,variant='qwen')

    def test_neutral_W_reuses_unweighted_result(self):
        rows=[row()]
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            p=Path(tmp)/'manifest.jsonl';p.write_text(json.dumps(rows[0])+'\n');cache=cache_for(rows);cache['manifest_sha256']=sha256(p)
            result=prepare_supervision(rows,sha256(p),cache)
            with self.assertRaises(ProjectError):validate_supervision(result,p,factor_config('B+W'))
            neutral=prepare_supervision(rows,sha256(p),variant='uniform')
            validate_supervision(neutral,p,factor_config('B'))

    def test_scene_only_builder_never_queries_ROIs(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            r=row();Image.new('RGB',(100,100)).save(Path(tmp)/r['image_relpath']);tasks=[]
            def infer(views,prompt,task):
                from fire_r2.supervision import SEMANTIC_FIELDS
                tasks.append(task);return [(json.dumps({k:'no' for k in SEMANTIC_FIELDS}),10) for _ in views]
            cache=build_batched_cache([r],'m',tmp,infer,'fixed',1,tasks=('A',))
            self.assertEqual(tasks,['A']);self.assertEqual(cache['tasks'],['A']);self.assertEqual(cache['samples'][r['sample_id']]['regions'],[])
