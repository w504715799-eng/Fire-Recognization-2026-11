import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError
from fire_r2.teacher_adaptation import partition_fit,instruction_records,supervised_value_spans,semantic_token_labels
from test_research import row


class AdaptationTests(unittest.TestCase):
    def test_fixed_disjoint_fit_partitions_exclude_development_and_preserve_detector(self):
        rows=[row(str(i)+s,state=s) for s in ('fire_only','smoke_only','both','empty') for i in range(8)];before=copy.deepcopy(rows)
        excluded={r['sample_id'] for r in rows if r['file'].startswith('0')}
        counts={'teacher_holdout':1,'teacher_dev':1,'teacher_train':2}
        a=partition_fit(rows,excluded,counts);b=partition_fit(list(reversed(rows)),excluded,counts)
        for key in ('teacher_train','teacher_dev','teacher_holdout'):
            self.assertEqual([r['sample_id'] for r in a[key]],[r['sample_id'] for r in b[key]])
            self.assertFalse(excluded&{r['sample_id'] for r in a[key]})
        self.assertEqual(rows,before);self.assertEqual(sum(len(x) for x in a.values()),20)

    def test_targets_only_use_available_original_supervision(self):
        r=row(state='both');before=copy.deepcopy(r);records=instruction_records(r,'teacher_train','m')
        self.assertEqual(len(records),3)
        self.assertEqual(records[0]['target'],{'fire_present':'yes','smoke_present':'yes'})
        self.assertEqual({x['target'].get('target_class') for x in records[1:]},{'fire','smoke'})
        self.assertTrue(all(not x['context_fields_supervised'] for x in records));self.assertEqual(r,before)
        with self.assertRaises(ProjectError):instruction_records(row(role='test'),'teacher_train','m')

    def test_value_token_spans_exclude_format_keys(self):
        target={'fire_present':'yes','smoke_present':'no'};text,spans=supervised_value_spans(target)
        self.assertEqual([text[a:b] for a,b in spans],['yes','no'])
        self.assertEqual(json.loads(text),target)

    def test_image_expansion_does_not_shift_loss_onto_prompt(self):
        plain=[10,20,30,40];offsets=[(0,1),(1,2),(2,3),(3,4)]
        labels=semantic_token_labels(plain,offsets,[10,11,12,20,30,40],[(2,3)])
        self.assertEqual(labels,[-100,-100,-100,-100,30,-100])
        with self.assertRaises(ProjectError):semantic_token_labels(plain,offsets,[10,11,12,20,99,40],[(2,3)])
