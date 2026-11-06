import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.weight_candidate import support_margin,candidate_raw_score


class CandidateTests(unittest.TestCase):
    def test_uncertainty_and_uniform_distribution_are_neutral(self):
        self.assertEqual(support_margin({k:.2 for k in ('fire','smoke','both','other','uncertain')},0),0.)
        self.assertEqual(support_margin({'fire':0,'smoke':0,'both':0,'other':0,'uncertain':1},1),0.)

    def test_signed_support_retains_fixed_positive_range(self):
        r={'boxes':[{'class_id':0,'original_annotation_id':1}]}
        def score(cls):return candidate_raw_score(r,{'regions':[{'original_annotation_id':1,'content':{k:float(k==cls) for k in ('fire','smoke','both','other','uncertain')}}]})
        self.assertEqual(score('fire'),1.25);self.assertEqual(score('smoke'),.75);self.assertEqual(score('uncertain'),1.)
