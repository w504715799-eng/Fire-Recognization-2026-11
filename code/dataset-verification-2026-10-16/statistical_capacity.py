"""Conditional sample-size planning; no predictions and no model experiments."""
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binom, norm

if Path(sys.prefix).name!='sci-env':raise RuntimeError('Use sci-env')
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'references/dataset-verification-2026-10-16'

def exact_paired_power(n,q,delta=.003,alpha=.05):
    # Sum over the discordant count; test conditional on it via one-sided exact McNemar.
    lo=max(0,int(binom.ppf(1e-12,n,q)));hi=min(n,int(binom.ppf(1-1e-12,n,q)))
    d=np.arange(lo,hi+1)
    critical=binom.isf(alpha,d,.5)+1
    return float(np.sum(binom.pmf(d,n,q)*binom.sf(critical-1,d,(q+delta)/(2*q))))

def required_n(q):
    low,high=1,50000
    while low<high:
        mid=(low+high)//2
        if exact_paired_power(mid,q)>=.8:high=mid
        else:low=mid+1
    return low

z=norm.ppf(.95)+norm.ppf(.8)
denominators={'cqu_nominal_pure_fire':3999,'cqu_test_nominal_pure_fire':395,
              'indoor_label0_only':2165,'indoor_test_label0_only':326}
negative={'cqu_all_nominal':4192,'cqu_test_nominal':415,
          'indoor_all_label1_only_provisional':2376,'indoor_test_label1_only_provisional':358}
obj={'date':'2026-10-16','not_observed_model_results':True,
 'assumptions':'Correct complete truth; independent paired binary image outcomes; one-sided alpha .05 and 80% power; no multiplicity adjustment. Raw label groups do not satisfy these assumptions yet.',
 'pure_fire_noninferiority':{'margin':.01,'true_difference':0,
   'approximate_required_n_by_discordance':{str(q):math.ceil(z*z*q/.01**2) for q in [.01,.02,.05,.1]},
   'zero_new_miss_one_sided95_upper':{name:{'n':n,'upper':1-.05**(1/n)} for name,n in denominators.items()},
   'note':'Image-level examples, not box-level recall or clustered-event proof; zero new misses differs from net noninferiority.'},
 'far_improvement':{'baseline':.01,'proposed':.007,'absolute_difference':.003,
   'exact_one_sided_mcnemar_required_n':{str(q):required_n(q) for q in [.003,.007,.017]},
   'power_by_nominal_group':{name:{'n':n,'expected_baseline_alarms':n*.01,
       'power_by_discordance':{str(q):exact_paired_power(n,q) for q in [.003,.007,.017]}}
       for name,n in negative.items()}},
 'decision':'Neither dataset is certified to fill the pure-fire or low-FAR validation gap; label semantics/completeness and source independence are unresolved.'}
(OUT/'statistical-capacity.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(obj,ensure_ascii=False,indent=2))
