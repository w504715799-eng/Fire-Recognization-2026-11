from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from fire_r2.common import ProjectError, output_path
from fire_r2.data import assign_roles, load_dataset, parse_yolo, prepare_view, verify_existing_manifest


class DataContractTests(unittest.TestCase):
    def test_dfire_mapping_and_unknown_classes(self):
        boxes=parse_yolo("0 .5 .5 .2 .4\n1 .4 .3 .1 .2\n",{"0":1,"1":0})
        self.assertEqual([b["class_id"] for b in boxes],[1,0])
        self.assertEqual(parse_yolo("",{"0":0}),[])
        with self.assertRaises(ProjectError):parse_yolo("2 .5 .5 .2 .2",{"0":0,"1":1})
        with self.assertRaises(ProjectError):parse_yolo("0 nan .5 .2 .2",{"0":0})

    def test_author_train_test_kept_and_partition_reproducible(self):
        rows=[{"file":f"{sp}/{i}.jpg","official_split":sp} for sp,n in [("train",5),("val",7),("test",3)] for i in range(n)]
        other=[dict(r) for r in reversed(rows)]
        spec={"split_policy":"val_half","split_key":"homefire"}
        assign_roles(rows,spec,"fire-master-v1-20261022")
        assign_roles(other,spec,"fire-master-v1-20261022")
        self.assertEqual({r['file']:r['role'] for r in rows},{r['file']:r['role'] for r in other})
        self.assertEqual(sum(r['role']=='dev' for r in rows),3)
        self.assertEqual(sum(r['role']=='cal' for r in rows),4)
        self.assertTrue(all(r['role']=='test' for r in rows if r['official_split']=='test'))
        self.assertTrue(all(r['role']=='fit' for r in rows if r['official_split']=='train'))

    def test_dfire_rounding_and_external_all(self):
        rows=[{"file":f"train/images/{i}.jpg","official_split":"train"} for i in range(21)]
        rows.append({"file":"test/images/test.jpg","official_split":"test"})
        assign_roles(rows,{"split_policy":"train_80_10_10","split_key":"dfire"},"s")
        self.assertEqual([sum(r['role']==x for r in rows) for x in ['fit','dev','cal','test']],[16,2,3,1])
        assign_roles(rows,{"split_policy":"external_only"},"s")
        self.assertTrue(all(r['role']=='external' for r in rows))

    def fixture(self,root):
        from PIL import Image
        ann=root/'datasets/demo/annotations';images=root/'datasets/demo/images'
        ann.mkdir(parents=True);images.mkdir(parents=True)
        counts={"train":1,"val":2,"test":1}
        for sp,n in counts.items():
            obj={'categories':[{'id':0,'name':'fire'},{'id':1,'name':'smoke'}],'images':[],'annotations':[]}
            for i in range(n):
                name=f'{sp}-{i}.jpg'
                Image.new('RGB',(10,10)).save(images/name)
                obj['images'].append({'id':i,'file_name':name,'width':10,'height':10})
                obj['annotations'].append({'id':i,'image_id':i,'category_id':0,'bbox':[0,0,11,0],'area':0,'iscrowd':0})
            (ann/f'{sp}.json').write_text(json.dumps(obj))
        cfg=root/'config.json'
        cfg.write_text(json.dumps({'names':['fire','smoke'],'split_salt':'s','datasets':{'demo':{
            'version':'v1','format':'coco','split_key':'demo','annotation_root':'demo/annotations',
            'image_root':'demo/images','official_splits':list(counts),'class_map':{'0':0,'1':1},
            'split_policy':'val_half','expected_counts':counts}}}))
        return cfg,ann,images

    def test_coco_geometry_preserved_and_training_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,ann,images=self.fixture(root)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ann.iterdir()}
            rows,summary=load_dataset(root,cfg,'demo')
            self.assertEqual(summary['nonpositive_boxes'],4)
            self.assertEqual(rows[0]['boxes'][0]['annotation_id'],0)
            result=prepare_view(root,rows,summary,'artifacts/data/demo')
            self.assertTrue(result['all_images_decoded'])
            view=root/'artifacts/data/demo'
            yaml=(view/'train.yaml').read_text()
            self.assertIn('train: fit.txt',yaml);self.assertIn('val: dev.txt',yaml)
            self.assertNotIn('test:',yaml);self.assertNotIn('cal:',yaml)
            self.assertTrue(all(p.is_symlink() for p in (view/'images').rglob('*.jpg')))
            self.assertIn('1.1000000000000001 0',next((view/'labels').rglob('*.txt')).read_text())
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ann.iterdir()})
            with self.assertRaises(ProjectError):prepare_view(root,rows,summary,'artifacts/data/demo')

    def test_missing_and_corrupt_images_cannot_produce_ready_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,ann,images=self.fixture(root)
            (images/'test-0.jpg').unlink()
            rows,summary=load_dataset(root,cfg,'demo')
            with self.assertRaises(ProjectError):prepare_view(root,rows,summary,'artifacts/data/demo')
            self.assertFalse((root/'artifacts/data/demo').exists())
            (images/'test-0.jpg').write_bytes(b'invalid image')
            with self.assertRaises(OSError):prepare_view(root,rows,summary,'artifacts/data/demo')
            self.assertFalse((root/'artifacts/data/demo').exists())
            self.assertFalse(list((root/'artifacts/data').glob('.prepare-*')))

    def test_output_cannot_point_to_shared_source_and_manifest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ProjectError):output_path(root,'datasets/source')
            with self.assertRaises(ProjectError):output_path(root,'artifacts/../../escape')
            p=root/'manifest.jsonl';p.write_text(json.dumps({'file':'x','official_split':'test','role':'fit'})+'\n')
            with self.assertRaises(ProjectError):verify_existing_manifest([{'file':'x','official_split':'test','role':'test'}],p)


if __name__=='__main__':unittest.main()
