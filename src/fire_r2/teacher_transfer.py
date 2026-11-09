"""GT-dominant presence transfer: identical student head/loss budget across all controls."""
from copy import deepcopy
import hashlib,itertools,json,math
from .common import ProjectError
from .teacher_adaptation import SCENE_TASK,supervised_value_spans,semantic_token_labels
from .supervision import digest_object

PROTOCOL='joint_presence_v1'
STATES=list(itertools.product((0,1),repeat=2))


def presence_marginals(joint):
    mass=math.fsum(joint)
    if len(joint)!=4 or any(not math.isfinite(x) or x<0 for x in joint) or abs(mass-1)>1e-5:
        raise ProjectError('Four normalized finite joint probabilities required')
    # FP32 softmax serialization can sum to 1+3e-8; correct only this numerical residue.
    return [min(1.,max(0.,math.fsum(p*s[i] for p,s in zip(joint,STATES))/mass)) for i in (0,1)]


def joint_target(presence):
    state=tuple(int(v) for v in presence)
    if len(presence)!=2 or any(v not in (0.,1.) for v in presence):raise ProjectError('Original binary presence required')
    return [float(s==state) for s in STATES]


def normalized_joint(values):
    presence_marginals(values)  # same finite/nonnegative/unit-mass checks
    mass=math.fsum(values)
    return [x/mass for x in values]


def make_joint_control(uniform):
    out=deepcopy(uniform)
    if out['variant']!='uniform' or any(x['weight']!=1. for x in out['samples'].values()):raise ProjectError('Uniform original-data control required')
    out.update(student_protocol='joint_presence_transfer_v1',teacher_joint_mix=0.,teacher_cache_sha256=None)
    for item in out['samples'].values():
        item['original_presence']=item['targets'][:2];item['original_joint']=joint_target(item['original_presence'])
        item['targets']=item['original_presence']+item['original_joint'];item['mask']=[True]*6
    return out


def score_presence_views(model,processor,views,device):
    """Enumerate all four answers, using only pixels and the frozen prompt, never GT labels."""
    import torch
    texts=[];plains=[];spans_all=[];images=[]
    for view in views:
        msg=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':SCENE_TASK}]}]
        prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        for state in STATES:
            target={k:'yes' if value else 'no' for k,value in zip(('fire_present','smoke_present'),state)}
            target_text,spans=supervised_value_spans(target);text=prompt+target_text+processor.tokenizer.eos_token
            texts.append(text);images.append(view);plains.append(processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True))
            spans_all.append([(len(prompt)+a,len(prompt)+b) for a,b in spans])
    inputs=processor(text=texts,images=images,padding=True,return_tensors='pt',add_special_tokens=False)
    positions=[];labels=[];options={}
    for i,plain in enumerate(plains):
        values=semantic_token_labels(plain['input_ids'],plain['offset_mapping'],inputs['input_ids'][i].tolist(),spans_all[i])
        pairs=[(p-1,x) for p,x in enumerate(values) if x!=-100]
        if len(pairs)!=2 or min(p for p,x in pairs)<0:raise ProjectError('This fixed protocol requires one token per yes/no value')
        positions.append([p for p,x in pairs]);labels.append([x for p,x in pairs])
        for field,bit in enumerate(STATES[i%4]):
            key=(field,bit);token=labels[-1][field]
            if key in options and options[key]!=token:raise ProjectError('Context-dependent answer tokens need explicit support')
            options[key]=token
    selected=sorted(set(p for row in positions for p in row));position_index={p:i for i,p in enumerate(selected)}
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.inference_mode():logits=model(**inputs,logits_to_keep=torch.tensor(selected,device=device),use_cache=False).logits
    scores=[]
    for i,row in enumerate(positions):
        score=0.
        for field,p in enumerate(row):
            allowed=[options[(field,0)],options[(field,1)]]
            logp=torch.log_softmax(logits[i,position_index[p],allowed].float(),dim=-1)
            score=score+logp[STATES[i%4][field]]
        scores.append(score)
    joint=torch.softmax(torch.stack(scores).reshape(len(views),4),dim=-1).cpu().tolist()
    return [{'joint_probabilities':p,'presence_probabilities':presence_marginals(p)} for p in joint]


def make_transfer_supervision(uniform,cache,expected,alpha=.25):
    if cache.get('scope')!='full_source_fit' or cache.get('status')!='completed' or cache.get('protocol')!=PROTOCOL:
        raise ProjectError('Complete production source-fit soft cache required')
    if cache.get('manifest_sha256')!=uniform['manifest_sha256'] or set(cache['samples'])!=set(uniform['samples']):
        raise ProjectError('Teacher cache must cover exactly the complete detector fit')
    if cache.get('teacher')!=expected:raise ProjectError('Wrong teacher identity/adapter for this control')
    out=make_joint_control(uniform)
    out.update(teacher_cache_sha256=digest_object(cache),teacher=expected,teacher_joint_mix=alpha,
               teacher_cache_protocol=PROTOCOL,teacher_cache_input_manifest_sha256=cache['manifest_sha256'])
    for ident,item in out['samples'].items():
        source=cache['samples'][ident];prob=presence_marginals(source['joint_probabilities'])
        if any(abs(a-b)>1e-6 for a,b in zip(prob,source['presence_probabilities'])):raise ProjectError('Marginal probabilities do not match joint cache')
        item['teacher_presence']=prob;item['teacher_joint']=normalized_joint(source['joint_probabilities'])
        joint=[(1-alpha)*g+alpha*t for g,t in zip(item['original_joint'],item['teacher_joint'])]
        item['targets']=item['original_presence']+joint
        item['teacher_image_sha256']=source['image_sha256']
    return out


def validate_transfer(advice,rows,expected=None,alpha_expected=.25):
    if expected is not None and advice.get('teacher')!=expected:raise ProjectError('Student teacher binding differs from configuration')
    alpha=advice.get('teacher_joint_mix')
    if advice.get('student_protocol')!='joint_presence_transfer_v1' or alpha!=alpha_expected or alpha not in (0.,.25):
        raise ProjectError('Frozen teacher transfer protocol/mixing coefficient differs')
    if alpha and (advice.get('teacher_cache_protocol')!=PROTOCOL or not advice.get('teacher_cache_sha256')):raise ProjectError('Teacher cache provenance missing')
    if not alpha and (advice.get('teacher') or advice.get('teacher_cache_sha256')):raise ProjectError('No-teacher control must not use a teacher')
    for row in rows:
        item=advice['samples'][row['sample_id']];classes={b['class_id'] for b in row['boxes']};gt=[float(0 in classes),float(1 in classes)]
        if len(item['targets'])!=6 or any(not math.isfinite(x) or not 0<=x<=1 for x in item['targets']):raise ProjectError('Six valid auxiliary target values required')
        truth=joint_target(gt);target=truth if not alpha else [(1-alpha)*g+alpha*t for g,t in zip(truth,normalized_joint(item['teacher_joint']))]
        if item.get('original_presence')!=gt or item.get('original_joint')!=truth or any(abs(x-y)>1e-7 for x,y in zip(item['targets'],gt+target)):
            raise ProjectError('Auxiliary mixture is not bound to original GT')
        if item['mask']!=[True]*6 or item['weight']!=1.:raise ProjectError('Transfer control changed supervision scope/weights')
    return advice


def audit_joint_caches(caches,heldout_rows):
    """Reuse production scores for the fixed teacher holdout; no extra inference or tuning."""
    result={'scope':'fixed teacher-internal holdout within detector fit','scoring_protocol':PROTOCOL,
            'not_a_training_gate':True,'probability_metrics_do_not_establish_perfect_calibration':True,'models':{}}
    for name,cache in caches.items():
        nll=brier=correct=0;states={}
        for row in heldout_rows:
            classes={b['class_id'] for b in row['boxes']};truth=joint_target([float(0 in classes),float(1 in classes)])
            probabilities=normalized_joint(cache['samples'][row['sample_id']]['joint_probabilities']);actual=truth.index(1.)
            predicted=max(range(4),key=lambda i:probabilities[i]);agree=predicted==actual;correct+=agree
            nll-=math.log(max(probabilities[actual],1e-12));brier+=sum((p-y)**2 for p,y in zip(probabilities,truth))
            state=states.setdefault(row['state'],{'images':0,'correct':0});state['images']+=1;state['correct']+=agree
        n=len(heldout_rows)
        result['models'][name]={'images':n,'joint_MAP_correct':correct,'mean_NLL_nats_floor_1e_12':nll/n,
                               'mean_Brier_sum_over_four_classes':brier/n,'states':states,'teacher':cache['teacher']}
    return result
