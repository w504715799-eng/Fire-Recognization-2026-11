"""Descriptive CPU-only analysis of frozen caches and completed dev/cal controls."""
from pathlib import Path
import argparse,csv,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,output_path,sha256
from fire_r2.teacher_transfer import normalized_joint,joint_target


def paired_image_intervals():
    """Exploratory image-level uncertainty for the existing source-cal operating points."""
    import numpy as np
    from fire_r2.research_data import load_manifest
    from fire_r2.evaluation import validate_predictions,matched_records
    rows=load_manifest(ROOT/'artifacts/data/dfire-kaggle-21527-sha65d81b60/manifest.jsonl',roles={'dev'})
    indices={r['sample_id']:i for i,r in enumerate(rows)};counts={}
    root=ROOT/'artifacts/student-transfer-20261109/evaluation'
    for name in ('B_reference','P_control','T_base','T_adapted'):
        metrics=read_json(root/name/'dev-metrics.json');point=metrics['fire_at_source_cal_threshold']
        predictions=validate_predictions(rows,read_json(root/name/'dev-predictions.json'))
        records,_,_=matched_records(rows,predictions);v=np.zeros((len(rows),7),dtype=np.int64)
        for i,row in enumerate(rows):
            fire=[b for b in row['boxes'] if b['class_id']==0]
            v[i,2]=sum(not b.get('iscrowd',0) for b in fire);v[i,3]=int(not fire)
            if row['state']=='fire_only':v[i,6]=v[i,2]
        for record in records:
            if record['ignored'] or record['score']<point['threshold']:continue
            i=indices[record['sample_id']];v[i,0]+=record['tp'];v[i,1]+=record['fp']
            if v[i,3]:v[i,4]=1
            if rows[i]['state']=='fire_only':v[i,5]+=record['tp']
        total=v.sum(axis=0)
        if tuple(total[:3])!=(point['tp'],point['fp'],point['tp']+point['fn']):raise ProjectError('Paired count audit differs')
        if total[4]!=point['triggered_negative_images']:raise ProjectError('Paired negative alerts differ')
        counts[name]=v
    rng=np.random.default_rng(20261110);sample_indices=rng.integers(0,len(rows),size=(2000,len(rows)))
    def metrics_from_counts(v):
        tp,fp,gt,neg,alerts,puretp,puregt=np.moveaxis(v,-1,0)
        return {'F1':2*tp/(gt+tp+fp),'precision':tp/(tp+fp),'recall':tp/gt,
                'negative_trigger_rate':alerts/neg,'pure_fire_recall':puretp/puregt}
    samples={name:metrics_from_counts(v[sample_indices].sum(axis=1)) for name,v in counts.items()}
    point={name:metrics_from_counts(v.sum(axis=0)) for name,v in counts.items()};deltas={}
    for label,left,right in [('GT_auxiliary','B_reference','P_control'),('base_teacher','P_control','T_base'),('adapted_teacher','P_control','T_adapted'),('teacher_adaptation','T_base','T_adapted')]:
        deltas[label]={k:{'point_delta':float(point[right][k]-point[left][k]),
                         'percentile_95_interval':np.quantile(samples[right][k]-samples[left][k],[.025,.975]).tolist()} for k in point[left]}
    return {'resamples':2000,'seed':20261110,'unit':'paired dev images',
            'note':'Exploratory intervals at model-specific source-cal F1 thresholds; correlated frames may narrow intervals. Not seed variance, event-independent confidence or a final-test significance claim.',
            'contrasts':deltas}


def cache_diagnostics():
    uniform=read_json(ROOT/'artifacts/supervision/dfire-B.json');result={}
    for name in ('base','adapted'):
        path=ROOT/f'artifacts/teacher-transfer-20261109/{name}/cache.json';cache=read_json(path)
        if cache['status']!='completed' or set(cache['samples'])!=set(uniform['samples']):raise ProjectError('Complete fit caches required')
        groups={}
        for ident,item in uniform['samples'].items():
            truth=joint_target(item['targets'][:2]);actual=truth.index(1.)
            prob=normalized_joint(cache['samples'][ident]['joint_probabilities'])
            mixed=[.75*g+.25*t for g,t in zip(truth,prob)]
            for group in ('all',('empty','smoke_only','fire_only','both')[actual]):
                g=groups.setdefault(group,{'images':0,'MAP_correct':0,'teacher_GT_mass':0.,'teacher_entropy_nats':0.,'mixed_GT_mass':0.,'mixed_entropy_nats':0.})
                g['images']+=1;g['MAP_correct']+=int(max(range(4),key=lambda i:prob[i])==actual)
                g['teacher_GT_mass']+=prob[actual];g['mixed_GT_mass']+=mixed[actual]
                g['teacher_entropy_nats']-=sum(p*math.log(p) for p in prob if p>0)
                g['mixed_entropy_nats']-=sum(p*math.log(p) for p in mixed if p>0)
        for g in groups.values():
            for k in ('teacher_GT_mass','teacher_entropy_nats','mixed_GT_mass','mixed_entropy_nats'):g['mean_'+k]=g.pop(k)/g['images']
        result[name]={'cache_sha256':sha256(path),'groups':groups}
    return {'scope':'all detector fit, includes teacher adaptation images; descriptive, not an independent accuracy estimate',
            'original_labels_changed':False,'causal_explanation_proven':False,'models':result}


def completed_analysis():
    result={'scope':'single-seed source dev/cal; final test untouched','models':{},
            'curve_grid_note':'Descriptive dev-curve comparison at requested recall; these thresholds are not deployment/calibration choices. No interpolation; ties retain highest threshold.',
            'runtime_note':'Prediction includes loading pipeline/file IO and no excluded warmup; not a dedicated latency benchmark.'}
    selections=set()
    for name in ('B_reference','P_control','T_base','T_adapted'):
        root=ROOT/'artifacts/student-transfer-20261109/evaluation'/name;status=read_json(root/'status.json')
        if status['status']!='completed':raise ProjectError('Full four-model evaluation required')
        metrics=read_json(root/'dev-metrics.json')
        if sha256(root/'dev-metrics.json')!=status['metrics_sha256']:raise ProjectError('Metric digest differs')
        selections.add(metrics['selection_sha256']);run=ROOT/'runs'/status['run']
        rows=list(csv.DictReader((run/'results.csv').open()));segments=[];previous=0.
        for row in rows:
            seconds=float(row['time'])
            if seconds<previous:segments.append(previous)
            previous=seconds
        segments.append(previous)
        selection=[read_json(p) for p in sorted((run/'original-dev').glob('epoch-*.json'))]
        best=max(selection,key=lambda r:(r['evaluation']['classes']['fire']['AP50_95'],-r['epoch']))
        points=metrics['fire_curve']['points'];grid={}
        for target in (.5,.6,.7,.75,.8):
            eligible=[p for p in points if p['recall'] is not None and p['recall']>=target]
            grid[str(target)]=max(eligible,key=lambda p:p['threshold']) if eligible else None
        result['models'][name]={k:metrics[k] for k in ('AP','small640_AP','groups','fire_at_source_cal_threshold','checkpoint_sha256','images')}
        result['models'][name].update(actual_epochs=len(rows),best_epoch=int(best['epoch'])+1,
            best_training_dev_fire_AP=best['evaluation']['classes']['fire']['AP50_95'],
            training_stop_reason=status['training_stop_reason'],CSV_elapsed_segments_seconds=segments,
            CSV_elapsed_sum_seconds=sum(segments),
            cost_note='Sum of completed-epoch elapsed segments; excludes unsaved interrupted work and some setup/final evaluation. Not full billed time.',
            recall_grid=grid,prediction_runtime=read_json(root/'dev-predictions.json')['runtime'])
    if len(selections)!=1:raise ProjectError('Different dev image selections')
    result['selection_sha256']=next(iter(selections))
    result['contrasts']=read_json(ROOT/'artifacts/student-transfer-20261109/evaluation/comparison.json')['contrasts']
    result['paired_operating_point_intervals']=paired_image_intervals()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cache-only',action='store_true');a=p.parse_args()
    out=output_path(ROOT,a.output)
    if out.exists():raise ProjectError('Preserve previous analysis')
    result={'status':'completed','cache_diagnostics':cache_diagnostics()}
    if not a.cache_only:result['student_analysis']=completed_analysis()
    write_json(out,result);print(json.dumps({'status':'completed','output':str(out),'cache_only':a.cache_only}))


if __name__=='__main__':main()
