from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fire_r2.repeats import summarize_pairs,repeat_evidence_status
from fire_r2.common import ProjectError


class PairedRepeatTests(unittest.TestCase):
    def test_cost_stop_is_not_a_completed_replication_or_a_fixed_epoch_gate(self):
        rows=[{'seed':29,'role':'P_control','stop_reason':'validation_plateau','actual_epochs':7},
              {'seed':29,'role':'T_adapted','stop_reason':'user_request','actual_epochs':23}]
        result=repeat_evidence_status(rows)
        self.assertFalse(result['method_effect_conclusion_allowed']);self.assertEqual(result['replication_status'],'cost_stopped_incomplete')
        rows[1]['stop_reason']='validation_plateau';rows[1]['actual_epochs']=9
        self.assertTrue(repeat_evidence_status(rows)['method_effect_conclusion_allowed'])

    def records(self):
        rows=[]
        for seed,base,gain in [(17,.4,.01),(29,.45,-.03)]:
            for role,value in [('P_control',base),('T_adapted',base+gain)]:
                rows.append({'seed':seed,'role':role,'checkpoint_sha256':str(seed)+role,
                    'selection_sha256':'dev','cal_selection_sha256':'cal','manifest_sha256':'data',
                    'initial_weights_sha256':'initial','inference':{'half':False},
                    'metrics':{'fire_AP50_95':value,'false_positive_boxes':100 if role=='P_control' else 90,
                               'smoke_only_trigger_rate':.05 if role=='P_control' else .04,
                               'background_trigger_rate':.01 if role=='P_control' else .02}})
        return rows

    def test_keeps_unfavorable_seed_and_pairs_methods(self):
        report=summarize_pairs(self.records());fire=report['paired_deltas']['fire_AP50_95']
        self.assertAlmostEqual(fire['mean'],-.01)
        self.assertEqual(fire['favorable_seed_count'],1);self.assertFalse(fire['favorable_in_every_seed'])
        self.assertTrue(report['paired_deltas']['false_positive_boxes']['favorable_in_every_seed'])
        self.assertTrue(report['paired_deltas']['smoke_only_trigger_rate']['favorable_in_every_seed'])
        self.assertEqual(report['paired_deltas']['background_trigger_rate']['favorable_seed_count'],0)

    def test_rejects_partial_or_copied_runs(self):
        with self.assertRaises(ProjectError):summarize_pairs(self.records()[:-1])
        rows=self.records();rows[3]['checkpoint_sha256']=rows[1]['checkpoint_sha256']
        with self.assertRaises(ProjectError):summarize_pairs(rows)

    def test_rejects_mismatched_protocol_or_nonfinite_scores(self):
        for field,value in [('selection_sha256','other'),('inference',{'half':True}),('seed',43)]:
            rows=self.records();rows[0][field]=value
            with self.assertRaises(ProjectError):summarize_pairs(rows)
        rows=self.records();rows[0]['metrics']['fire_AP50_95']=float('nan')
        with self.assertRaises(ProjectError):summarize_pairs(rows)
