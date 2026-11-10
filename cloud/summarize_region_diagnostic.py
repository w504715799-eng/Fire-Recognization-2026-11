"""Complete fixed-view region summaries; no new inference, fitted score or training."""
from pathlib import Path
import collections,json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from cloud.region_diagnostic import metrics
from fire_r2.common import read_json,write_json,sha256,ProjectError


def main():
    out=ROOT/'artifacts/region-diagnostic-20261110';cfg=read_json(out/'protocol.json');inf=read_json(out/'inference.json')
    if inf['status']!='completed' or sha256(out/'scores.jsonl')!=inf['scores_sha256']:raise ProjectError('Complete unchanged score journal required')
    dest=out/'view-and-error-summary.json'
    if dest.exists():raise ProjectError('Keep previous complete summary')
    regions=[json.loads(x) for x in (out/'regions.jsonl').read_text().splitlines()]
    scores=[json.loads(x) for x in (out/'scores.jsonl').read_text().splitlines()];lookup={(r['id'],r['view']):r['probabilities'] for r in scores}
    result={'scope':'fixed-region diagnostic, not whole-dataset detector improvement','view_metrics':{},'all_region_groups':{},'high_teacher_support_errors':{},'no_model_loaded':True}
    for view in cfg['views']:
        for cls in (0,1):
            name=('fire','smoke')[cls];rows=[]
            for r in regions:
                if r['predicted_class']!=cls or r['kind'] not in ('prediction_tp','prediction_cross_class','prediction_background'):continue
                p=lookup[(r['id'],view)];s=p[name]+p['both']
                rows.append({'positive':int(r['kind']=='prediction_tp'),'P0':r['P0_score'],'teacher':s,'combined':math.sqrt(s*r['P0_score'])})
            result['view_metrics'][view+'|'+name]={key:metrics(rows,key) for key in ('P0','teacher','combined')}
            high=[r for r in rows if r['teacher']>=.8]
            result['high_teacher_support_errors'][view+'|'+name]={'threshold':.8,'regions_above_threshold':len(high),'false_regions_above_threshold':sum(not r['positive'] for r in high),'diagnostic_prevalence_not_population':True}
    buckets=collections.defaultdict(list)
    for r in regions:
        p=lookup[(r['id'],'context1.5')];cls=r['original_GT_class'] if r['kind'].startswith(('GT','external')) else r['predicted_class']
        support=p[('fire','smoke')[cls]]+p['both'] if cls in (0,1) else p['background']+p['light']
        buckets[(r['source'],r['kind'],cls)].append((support,max(p,key=p.get),r['small640']))
    for key,values in buckets.items():
        result['all_region_groups'][str(key)]={'regions':len(values),'small640':sum(v[2] for v in values),'mean_support':sum(v[0] for v in values)/len(values),'support_ge_0.5':sum(v[0]>=.5 for v in values),'top_choice_counts':dict(collections.Counter(v[1] for v in values))}
    write_json(dest,result)


if __name__=='__main__':main()
