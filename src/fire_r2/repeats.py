"""Paired training-seed summaries; never pool repeated images or choose the best seed."""
import math
import statistics
from .common import ProjectError


def repeat_evidence_status(records,replication_seed=29):
    stopped=[{'seed':r['seed'],'role':r['role'],'stop_reason':r['stop_reason'],'actual_epochs':r['actual_epochs']}
             for r in records if r['seed']==replication_seed and r.get('stop_reason') in ('user_request','time_budget')]
    return {'replication_status':'cost_stopped_incomplete' if stopped else 'completed_as_planned',
            'method_effect_conclusion_allowed':not stopped,'cost_stopped_runs':stopped}


def summarize_pairs(records,expected_seeds=(17,29)):
    roles=('P_control','T_adapted');indexed={}
    for record in records:
        key=(record['seed'],record['role'])
        if key in indexed:raise ProjectError('Duplicate seed/method pair')
        indexed[key]=record
    if set(indexed)!={(s,r) for s in expected_seeds for r in roles}:raise ProjectError('Complete prespecified seed/method pairs required')
    if len({r['checkpoint_sha256'] for r in records})!=len(records):raise ProjectError('Copied checkpoints are not independent training runs')
    for field in ('selection_sha256','cal_selection_sha256','manifest_sha256','initial_weights_sha256'):
        if len({r[field] for r in records})!=1:raise ProjectError('Paired protocol differs: '+field)
    if any(r['inference']!=records[0]['inference'] for r in records):raise ProjectError('Paired inference protocols differ')
    keys=set(records[0]['metrics'])
    if not keys or any(set(r['metrics'])!=keys for r in records):raise ProjectError('Metric vectors differ')
    if any(not math.isfinite(float(v)) for r in records for v in r['metrics'].values()):raise ProjectError('Undefined/nonfinite paired metric')
    result={'seeds':list(expected_seeds),'methods':{},'paired_deltas':{},
            'difference_convention':'T_adapted minus P_control within the same seed',
            'interpretation':'Means/std describe training-seed variability on the same dev images; no pooling of predictions, no extra independent images, no two-seed significance claim.'}
    for role in roles:
        result['methods'][role]={}
        for key in sorted(keys):
            values=[float(indexed[(s,role)]['metrics'][key]) for s in expected_seeds]
            result['methods'][role][key]={'values':values,'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else None}
    for key in sorted(keys):
        values=[float(indexed[(s,'T_adapted')]['metrics'][key])-float(indexed[(s,'P_control')]['metrics'][key]) for s in expected_seeds]
        lower=key in ('negative_trigger_rate','smoke_only_trigger_rate','background_trigger_rate','fp_per_image','false_positive_boxes') or key.startswith(('FP_at_dev_recall_','alerts_at_dev_recall_'))
        favorable=[(-v if lower else v)>0 for v in values]
        result['paired_deltas'][key]={'values':values,'mean':statistics.mean(values),
            'sample_std':statistics.stdev(values) if len(values)>1 else None,
            'lower_is_better':lower,'favorable_seed_count':sum(favorable),'favorable_in_every_seed':all(favorable)}
    return result
