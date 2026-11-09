"""Retain all source-dev control results and contrasts; no favorable-metric selection."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,ProjectError,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/student-transfer-20261109/evaluation');a=p.parse_args()
    root=ROOT/a.root;output=root/'comparison.json'
    if output.exists():raise ProjectError('Preserve previous comparison')
    reports={}
    for name in ('B_reference','P_control','T_base','T_adapted'):
        path=root/name
        if read_json(path/'status.json')['status']!='completed':raise ProjectError('All planned controls must complete evaluation')
        reports[name]=read_json(path/'dev-metrics.json')
    if len({r['selection_sha256'] for r in reports.values()})!=1:raise ProjectError('Control image sets differ')
    result={'scope':'single seed, complete same-source dev/cal controls, no test-based selection','models':{},'contrasts':{}}
    for name,r in reports.items():
        result['models'][name]={k:r[k] for k in ('AP','small640_AP','fire_at_source_cal_threshold','groups','checkpoint_sha256','images')}
    for label,left,right in [('GT_auxiliary','B_reference','P_control'),('base_teacher','P_control','T_base'),('adapted_teacher','P_control','T_adapted'),('teacher_adaptation','T_base','T_adapted')]:
        result['contrasts'][label]={'left':left,'right':right,'fire_AP50_95_delta':reports[right]['AP']['classes']['fire']['AP50_95']-reports[left]['AP']['classes']['fire']['AP50_95'],
            'smoke_AP50_95_delta':reports[right]['AP']['classes']['smoke']['AP50_95']-reports[left]['AP']['classes']['smoke']['AP50_95'],
            'source_cal_operating_points_are_model_specific':True}
    write_json(output,result);print(json.dumps(result['contrasts']))


if __name__=='__main__':main()
