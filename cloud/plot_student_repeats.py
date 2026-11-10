"""Visualize both paired seeds without pooling images or hiding unfavorable metrics."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,output_path,ProjectError


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',required=True);a=p.parse_args()
    source=output_path(ROOT,a.comparison);data=read_json(source)
    if data['status']!='completed':raise ProjectError('Completed paired comparison required')
    if not data.get('method_effect_conclusion_allowed',False):raise ProjectError('Cost-stopped intermediate runs must not be plotted as a completed replication-effect comparison')
    out=source.parent;targets=[out/(stem+suffix) for stem in ('paired-deltas','paired-PR-FROC') for suffix in ('.png','.svg')]
    if any(p.exists() for p in targets):raise ProjectError('Preserve prior figures')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,stem):
        fig.savefig(out/(stem+'.png'),dpi=180)
        svg=out/(stem+'.svg');fig.savefig(svg)
        # Whitespace normalization is part of generation; geometry and recorded data stay unchanged.
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    metrics=[('fire_AP50_95','Fire AP50:95'),('smoke_AP50_95','Smoke AP50:95'),('fire_F1','Fire F1 at source-cal point'),
             ('fire_recall','Fire recall at source-cal point'),('pure_fire_recall','Pure-fire recall'),('negative_trigger_rate','Negative image trigger rate (lower is better)')]
    for ax,(key,label) in zip(axes.flat,metrics):
        values=[100*x for x in data['summary']['paired_deltas'][key]['values']]
        ax.bar([0,1],values,color=['#64748b','#2563eb'],width=.55);ax.axhline(0,color='black',lw=.7)
        extent=max(.1,max(abs(x) for x in values))*1.35;ax.set_ylim(-extent,extent)
        ax.set_xticks([0,1],['seed17','seed29']);ax.set_title(label,fontsize=10);ax.set_ylabel('Adapted minus P0 (percentage points)');ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(values):ax.annotate(f'{v:+.3f}',(i,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
    fig.suptitle('D-Fire dev paired-seed replication | two distinct training runs per method',fontsize=12)
    save(fig,'paired-deltas')
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8),layout='constrained')
    for run in data['runs']:
        metrics=read_json(ROOT/run['files']['dev-metrics.json']['path']);curve=metrics['fire_curve']['points'];point=metrics['fire_at_source_cal_threshold']
        color='#2563eb' if run['role']=='P_control' else '#059669';style='-' if run['seed']==17 else '--'
        label=f"{'P0' if run['role']=='P_control' else 'Adapted'} seed{run['seed']}"
        axes[0].plot([p['recall'] for p in curve],[p['precision'] for p in curve],color=color,ls=style,lw=1.2,label=label)
        axes[1].plot([p['fp_per_image'] for p in curve],[p['recall'] for p in curve],color=color,ls=style,lw=1.2,label=label)
        marker='o' if run['seed']==17 else 's'
        axes[0].scatter(point['recall'],point['precision'],color=color,marker=marker,s=25,zorder=5)
        axes[1].scatter(point['fp_per_image'],point['recall'],color=color,marker=marker,s=25,zorder=5)
    axes[0].set(xlabel='Fire box recall',ylabel='Fire box precision',title='Precision-recall at IoU 0.5',xlim=(0,1),ylim=(0,1.02))
    axes[1].set(xlabel='False-positive fire boxes / image (symlog)',ylabel='Fire box recall',title='FROC at IoU 0.5',ylim=(0,1.02))
    axes[1].set_xscale('symlog',linthresh=.001);axes[1].set_xlim(left=0)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Same 1,549 dev images | GPU FP32 | markers: model-specific source-cal F1 points',fontsize=11)
    save(fig,'paired-PR-FROC')
    print(json.dumps({p.name:p.stat().st_size for p in targets}))


if __name__=='__main__':main()
