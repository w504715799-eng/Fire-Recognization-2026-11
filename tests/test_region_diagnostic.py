import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloud.region_diagnostic import auc,metrics,classify_detection


class RegionDiagnosticTests(unittest.TestCase):
    def test_auc_and_ties(self):
        self.assertEqual(auc([0,1],[0,1]),1.)
        self.assertEqual(auc([0,1],[1,1]),.5)
        self.assertEqual(auc([0,1],[1,0]),0.)
        self.assertIsNone(auc([1,1],[.2,.5]))

    def test_equal_retention_includes_score_ties(self):
        rows=[{'positive':1,'score':.8},{'positive':1,'score':.8},{'positive':0,'score':.8},{'positive':0,'score':.1}]
        m=metrics(rows,'score')['retention']['0.8'];self.assertEqual(m['TP_retained'],2);self.assertEqual(m['FP_retained'],1)

    def test_error_types_not_all_background(self):
        gt=[{'class_id':0}];boxes=[[0,0,10,10]];used=set()
        self.assertEqual(classify_detection({'class_id':0,'bbox':boxes[0]},gt,boxes,used)[0],'tp')
        self.assertEqual(classify_detection({'class_id':0,'bbox':boxes[0]},gt,boxes,used)[0],'duplicate')
        self.assertEqual(classify_detection({'class_id':1,'bbox':boxes[0]},gt,boxes,used)[0],'cross_class')
        self.assertEqual(classify_detection({'class_id':0,'bbox':[20,20,10,10]},gt,boxes,used)[0],'background')
        self.assertEqual(classify_detection({'class_id':0,'bbox':[2,2,1,1]},gt,boxes,used)[0],'localization')


if __name__=='__main__':unittest.main()
