"""Plot all four frozen dev curves and their source-cal operating points."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'artifacts/student-transfer-20261109/evaluation'
    out=ROOT/'artifacts/student-transfer-complete-20261110'
    targets=[out/'student-PR-FROC.png',out/'student-PR-FROC.svg']
    if any(p.exists() for p in targets):raise RuntimeError('Preserve prior figure')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    selection=set()
    for name,label,color in [('B_reference','B','#64748b'),('P_control','P0: GT auxiliary','#2563eb'),('T_base','Base teacher','#d97706'),('T_adapted','Adapted teacher','#059669')]:
        status=json.loads((root/name/'status.json').read_text());assert status['status']=='completed'
        data=json.loads((root/name/'dev-metrics.json').read_text());selection.add(data['selection_sha256'])
        curve=data['fire_curve']['points'];point=data['fire_at_source_cal_threshold']
        axes[0].plot([p['recall'] for p in curve],[p['precision'] for p in curve],color=color,lw=1.2,label=label)
        axes[1].plot([p['fp_per_image'] for p in curve],[p['recall'] for p in curve],color=color,lw=1.2,label=label)
        axes[0].scatter(point['recall'],point['precision'],color=color,s=30,zorder=5,edgecolors='white',linewidths=.5)
        axes[1].scatter(point['fp_per_image'],point['recall'],color=color,s=30,zorder=5,edgecolors='white',linewidths=.5)
    assert len(selection)==1
    axes[0].set(xlabel='Fire box recall',ylabel='Fire box precision',title='Precision-recall at IoU 0.5',xlim=(0,1),ylim=(0,1.02))
    axes[1].set(xlabel='False-positive fire boxes / image (symlog)',ylabel='Fire box recall',title='FROC at IoU 0.5',ylim=(0,1.02))
    axes[1].set_xscale('symlog',linthresh=.001)
    axes[1].set_xlim(left=0)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('D-Fire dev: 1,549 images | GPU FP32 | dots: source-cal F1 thresholds',fontsize=11)
    out.mkdir(exist_ok=True)
    for p in targets:fig.savefig(p,dpi=180)
    plt.close(fig)
    print(json.dumps({p.name:p.stat().st_size for p in targets}))


if __name__=='__main__':main()
