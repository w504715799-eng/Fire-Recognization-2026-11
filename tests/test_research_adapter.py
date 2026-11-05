import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError
from fire_r2.research_data import canonical_row,coco_view
from fire_r2.supervision import prepare_supervision,parse_advice
from fire_r2.teacher_runner import pilot_selection,teacher_view
from test_research import row,cache_for


class AdapterTests(unittest.TestCase):
    def test_r2_box_conversion_keeps_raw_geometry_and_zero_id(self):
        source={'sample_id':'fasdd_cv/train/x','image_path':'x.jpg','view_image':'images/fit/short.jpg',
                'role':'fit','state':'fire_only','width':100,'height':80,
                'boxes':[{'class_id':0,'coordinates':'pixel_xywh','bbox':[-1,3,8,0],'annotation_id':0,'area':0}]}
        original=copy.deepcopy(source);r=canonical_row(source);coco=coco_view([r])
        self.assertEqual(source,original)
        self.assertEqual(coco['annotations'][0]['bbox'],[-1,3,8,0])
        self.assertEqual(coco['annotations'][0]['original_annotation_id'],0)
        self.assertFalse(r['has_small_fire'])

    def test_old_yolo_manifest_uses_header_without_rewriting_original(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            p=Path(tmp)/'x.jpg';im=Image.new('RGB',(80,100));exif=im.getexif();exif[274]=6;im.save(p,exif=exif)
            original=p.read_bytes()
            r=canonical_row({'sample_id':'dfire/train/x','image_path':str(p),'role':'fit','state':'fire_only',
                             'boxes':[{'class_id':0,'coordinates':'normalized_xywh','bbox':[.5,.5,.1,.1]}]})
            self.assertEqual((r['width'],r['height']),(100,80))
            self.assertEqual(r['boxes'][0]['original_annotation_id'],1)
            self.assertEqual(coco_view([r])['annotations'][0]['bbox'],[45.,36.,10.,8.])
            self.assertEqual(p.read_bytes(),original)

    def test_teacher_crop_obeys_exif(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            p=Path(tmp)/'x.jpg';im=Image.new('RGB',(80,100));exif=im.getexif();exif[274]=6;im.save(p,exif=exif)
            self.assertEqual(teacher_view(p).size,(640,640))
            self.assertEqual(teacher_view(p,{'bbox':[.5,.5,.2,.2],'format':'cxcywh_normalized'}).size,(640,640))

    def test_pilot_cache_cannot_train_full_benchmark(self):
        rows=[row('a')];cache=cache_for(rows);cache['scope']='pilot_fit_only'
        with self.assertRaises(ProjectError):prepare_supervision(rows,'manifest',cache)

    def test_region_coverage_and_teacher_identity_are_enforced(self):
        rows=[row('a')];cache=cache_for(rows);cache['samples'][rows[0]['sample_id']]['regions']=[]
        with self.assertRaises(ProjectError):prepare_supervision(rows,'manifest',cache)
        cache=cache_for(rows);cache['teacher_id']='another-model'
        with self.assertRaises(ProjectError):prepare_supervision(rows,'manifest',cache)

    def test_pilot_selection_is_before_predictions_fit_only_and_stratified(self):
        rows=[row(str(i)+state,state=state) for state in ('fire_only','smoke_only','both','empty') for i in range(4)]
        rows.append(row('heldout','test'))
        a=pilot_selection(rows,2);b=pilot_selection(list(reversed(rows)),2)
        self.assertEqual([r['sample_id'] for r in a],[r['sample_id'] for r in b])
        self.assertEqual(len(a),8);self.assertTrue(all(r['role']=='fit' for r in a))
