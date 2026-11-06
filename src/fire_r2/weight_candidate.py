"""One preregistered diagnostic candidate, not an enabled training method."""
from .common import ProjectError
import math


def support_margin(probabilities,class_id):
    expected={'fire','smoke','both','other','uncertain'}
    if class_id not in (0,1) or set(probabilities)!=expected or any(not math.isfinite(v) or not 0<=v<=1 for v in probabilities.values()) or abs(sum(probabilities.values())-1)>1e-5:
        raise ProjectError('Expected a normalized five-content distribution')
    positive={'fire','both'} if class_id==0 else {'smoke','both'}
    negative={'smoke','other'} if class_id==0 else {'fire','other'}
    return sum(probabilities[k] for k in positive)-sum(probabilities[k] for k in negative)


def candidate_raw_score(row,probabilities):
    """Expected signed class agreement, fixed legacy amplitude; uncertainty contributes zero.

    These are conditional model scores, not calibrated correctness probabilities.
    Original class labels enter only here, after blind model inference.
    """
    if row['boxes']:
        by_id={b['original_annotation_id']:b for b in row['boxes']}
        margins=[support_margin(r['content'],by_id[r['original_annotation_id']]['class_id']) for r in probabilities['regions']]
    else:
        margins=[probabilities['scene'][key]['no']-probabilities['scene'][key]['yes'] for key in ('fire_visible','smoke_visible')]
    return 1.+.25*(sum(margins)/len(margins) if margins else 0.)
