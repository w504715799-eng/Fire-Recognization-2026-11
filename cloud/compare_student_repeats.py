"""Validate and compare complete P0/adapted seed17 and seed29 dev/cal experiments."""
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.repeats import summarize_pairs


def read_run(seed,role):
    folder='student-transfer-20261109' if seed==17 else 'student-transfer-seed29-20261110'
    directory=ROOT/'artifacts'/folder/'evaluation'/role
    status=read_json(directory/'status.json')
    if status['status']!='completed':raise ProjectError('A paired evaluation is not completed')
    metrics=read_json(directory/'dev-metrics.json');predictions=read_json(directory/'dev-predictions.json')
    calibration=read_json(directory/'calibration.json');cal_predictions=read_json(directory/'cal-predictions.json')
    suffix='20261109' if seed==17 else 'seed29_20261110'
    config_path=ROOT/f'configs/train_dfire_{role}_{suffix}.json';config=read_json(config_path)
    if status['run']!=config['run_name']:raise ProjectError('Evaluation belongs to another run')
    if any(r.get('seed')!=seed for r in (metrics,predictions,cal_predictions,config)):raise ProjectError('Mislabeled training seed')
    if any(r.get('training_source')!='dfire' or r.get('target_source')!='dfire' for r in (metrics,predictions,cal_predictions)):raise ProjectError('Mislabeled source dataset')
    if metrics['purpose']!='dev' or predictions['purpose']!='dev' or cal_predictions['purpose']!='cal':raise ProjectError('Mislabeled evaluation partition')
    if sha256(directory/'dev-metrics.json')!=status['metrics_sha256'] or sha256(directory/'calibration.json')!=status['calibration_sha256']:raise ProjectError('Completed artifact changed')
    identities={r['checkpoint_sha256'] for r in (metrics,predictions,cal_predictions,calibration,status)}
    if len(identities)!=1:raise ProjectError('Mismatched model/calibration identity')
    if metrics['images']!=1549 or calibration['calibration_images']!=1550 or status['test_accessed']:raise ProjectError('Frozen dev/cal scope differs')
    if calibration['fit_partition']!='cal' or calibration['rule']!='fire_F1_IoU0.5_tie_highest_threshold':raise ProjectError('Calibration protocol differs')
    if predictions['manifest_sha256']!=cal_predictions['manifest_sha256'] or calibration['selection_sha256']!=cal_predictions['selection_sha256']:raise ProjectError('Manifest/cal selection differs')
    if predictions['inference']!=cal_predictions['inference']:raise ProjectError('Dev/cal inference differs')
    if predictions['runtime']['precision']!='float32' or predictions['runtime']['device']!='0':raise ProjectError('Matched GPU FP32 evaluations required')
    run=ROOT/'runs'/status['run'];rows=list(csv.DictReader((run/'results.csv').open()))
    if len(rows)!=status['completed_training_epochs']:raise ProjectError('Actual training length differs from evaluation receipt')
    selection=[read_json(p) for p in sorted((run/'original-dev').glob('epoch-*.json'))]
    best=max(selection,key=lambda r:(r['evaluation']['classes']['fire']['AP50_95'],-r['epoch']))
    prior=0.;segments=[]
    for row in rows:
        value=float(row['time'])
        if value<prior:segments.append(prior)
        prior=value
    segments.append(prior)
    point=metrics['fire_at_source_cal_threshold'];pure=metrics['groups']['fire_only']['fire_at_source_cal_threshold']
    vector={'fire_AP50_95':metrics['AP']['classes']['fire']['AP50_95'],'fire_AP50':metrics['AP']['classes']['fire']['AP50'],
            'smoke_AP50_95':metrics['AP']['classes']['smoke']['AP50_95'],'dual_AP50_95':metrics['AP']['mAP50_95'],
            'small_fire_AP50_95':metrics['small640_AP']['classes']['fire']['AP50_95'],
            'fire_precision':point['precision'],'fire_recall':point['recall'],
            'fire_F1':2*point['tp']/(2*point['tp']+point['fp']+point['fn']),
            'negative_trigger_rate':point['negative_trigger_rate'],'fp_per_image':point['fp_per_image'],
            'false_positive_boxes':point['fp'],'pure_fire_recall':pure['recall'],
            'pure_fire_AP50_95':metrics['groups']['fire_only']['AP']['classes']['fire']['AP50_95'],
            'coexistence_fire_AP50_95':metrics['groups']['both']['AP']['classes']['fire']['AP50_95']}
    grid={}
    for target in (.5,.6,.7,.75,.8):
        options=[p for p in metrics['fire_curve']['points'] if p['recall'] is not None and p['recall']>=target]
        p=max(options,key=lambda p:p['threshold']) if options else None;grid[str(target)]=p
        if p:
            vector[f'FP_at_dev_recall_{target}']=p['fp']
            vector[f'alerts_at_dev_recall_{target}']=round(p['negative_trigger_rate']*metrics['fire_curve']['original_negative_images'])
    return {'seed':seed,'role':role,'run':status['run'],'checkpoint_sha256':status['checkpoint_sha256'],
            'selection_sha256':metrics['selection_sha256'],'cal_selection_sha256':calibration['selection_sha256'],
            'manifest_sha256':predictions['manifest_sha256'],'initial_weights_sha256':config['weights_sha256'],
            'inference':predictions['inference'],'metrics':vector,'recall_grid':grid,
            'full_metrics_without_curve':{k:v for k,v in metrics.items() if k!='fire_curve'},
            'actual_epochs':len(rows),'best_epoch':best['epoch']+1,'stop_reason':status['training_stop_reason'],
            'validation_stopping':config.get('validation_stopping','historical user-stop run'),
            'CSV_elapsed_segments_seconds':segments,'CSV_elapsed_sum_seconds':sum(segments),
            'runtime':predictions['runtime'],'config_sha256':sha256(config_path),
            'files':{name:{'path':str((directory/name).relative_to(ROOT)),'sha256':sha256(directory/name)} for name in ('status.json','dev-metrics.json','dev-predictions.json','cal-predictions.json','calibration.json')}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve earlier repeat comparison')
    runs=[read_run(seed,role) for seed in (17,29) for role in ('P_control','T_adapted')]
    # Unattained recall is a reported outcome, not a reason to hide the entire comparison.
    common=set.intersection(*(set(r['metrics']) for r in runs))
    unattained=[{'seed':r['seed'],'role':r['role'],'target':target} for r in runs for target,point in r['recall_grid'].items() if point is None]
    summary=summarize_pairs([{**r,'metrics':{k:v for k,v in r['metrics'].items() if k in common}} for r in runs])
    historical=read_json(ROOT/'artifacts/student-transfer-complete-20261110/full-analysis.json')['student_analysis']['models']
    result={'status':'completed','scope':'Two prespecified paired training seeds; original D-Fire dev/cal only; final test untouched',
            'runs':runs,'summary':summary,'unattained_recall_grid':unattained,'seed17_nonreplicated_reference_only':{name:historical[name] for name in ('B_reference','T_base')},
            'notes':['B and T_base have no seed29 run and are not included in two-seed means.',
                     'Same dev images recur across seeds; do not claim twice as many independent evaluation images.',
                     'Seed17 P0 was user-stopped and adapted training resumed after interruption; seed29 uses a common stopping policy.',
                     'Recall grids describe dev curves; source-cal F1 thresholds remain the deployed evaluation operating points.',
                     'Unattained recall points remain explicit; their undefined FP counts are not averaged. Direction counts are not practical/significance gates.',
                     'Training length/cost are recorded, not quality criteria; no automatic expansion or parameter sweep follows this report.']}
    write_json(out,result);print(json.dumps({'status':'completed','fire_AP50_95':summary['paired_deltas']['fire_AP50_95'],
                                           'fire_F1':summary['paired_deltas']['fire_F1'],'smoke_AP50_95':summary['paired_deltas']['smoke_AP50_95']}))


if __name__=='__main__':main()
