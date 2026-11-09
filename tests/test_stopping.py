from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fire_r2.stopping import ValidationProgress,require_evaluable_training
from fire_r2.common import ProjectError


class StoppingTests(unittest.TestCase):
    def test_noise_does_not_reset_progress_but_real_gain_does(self):
        m=ValidationProgress(patience=3,min_delta=.001)
        self.assertFalse(m.observe(1,.4));self.assertFalse(m.observe(2,.4002));self.assertFalse(m.observe(3,.4004));self.assertTrue(m.observe(4,.4005))
        m=ValidationProgress(patience=3,min_delta=.001)
        m.observe(1,.4);self.assertFalse(m.observe(3,.402));self.assertFalse(m.observe(5,.4021));self.assertTrue(m.observe(6,.4022))

    def test_min_mode_and_repeated_callbacks(self):
        m=ValidationProgress(patience=2,min_delta=.01,mode='min');m.observe(1,1.)
        self.assertFalse(m.observe(1,1.));self.assertFalse(m.observe(2,.995));self.assertTrue(m.observe(3,.994))

    def test_early_stopped_run_is_evaluable_without_100_epochs(self):
        self.assertTrue(require_evaluable_training({'status':'completed_or_budget_stopped'},37,True))
        with self.assertRaises(ProjectError):require_evaluable_training({'status':'running'},100,True)
        with self.assertRaises(ProjectError):require_evaluable_training({'status':'completed'},0,False)
