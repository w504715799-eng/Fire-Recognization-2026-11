"""Versioned prompts and exact-output tries; no labels or model imports."""
import itertools,json
from .common import ProjectError
from .supervision import SEMANTIC_FIELDS,SCENE_PROMPT,REGION_PROMPT


def protocol(version='v1'):
    if version=='v1':
        return {'version':'v1','A':SCENE_PROMPT,'B':REGION_PROMPT,'constrained':False,
                'region_marker':'number_before_resize','max_new_tokens':192}
    if version!='v2':raise ProjectError('Unknown teacher protocol')
    return {'version':'v2','A':
        'Inspect the image. Report visible evidence only, not the cause of smoke. '
        'fire_visible means actual visible flames; smoke_visible means visible smoke. '
        'cloud_fog_like means cloud or fog appearance; bright_light_like means a bright light source; '
        'reflection_like means a visible reflection; low_visibility means detail is difficult to see. '
        'Return one JSON object, not an array, with exactly these six keys: '+', '.join(SEMANTIC_FIELDS)+'. '
        'Every value must be the quoted string "yes", "no", or "uncertain". '
        'Smoke alone is not visible flame. Use uncertain when visual evidence is insufficient. No markdown.',
        'B':'Inspect only the visible content inside the thin rectangular outline; ignore the outline itself. '
        'Use smoke for smoke without visible flames, fire for visible flames without smoke, both if both are visible, '
        'other if neither is visible, and uncertain if the content cannot be identified. '
        'Return one JSON object with exactly content and visibility. content must be one of '
        '"fire", "smoke", "both", "other", "uncertain". visibility must be one of '
        '"clear", "limited", "unjudgeable". No markdown or numeric labels.',
        'constrained':True,'region_marker':'thin_outline_after_resize','max_new_tokens':192}


def valid_answers(task):
    if task=='A':
        return [json.dumps(dict(zip(SEMANTIC_FIELDS,values)),separators=(',',':'))
                for values in itertools.product(('yes','no','uncertain'),repeat=6)]
    if task=='B':
        return [json.dumps({'content':content,'visibility':visibility},separators=(',',':'))
                for content,visibility in itertools.product(('fire','smoke','both','other','uncertain'),
                                                            ('clear','limited','unjudgeable'))]
    raise ProjectError('Unknown teacher task')


class AnswerTrie:
    """Constrain syntax/enums without consulting GT; preserve model choice among all valid answers."""
    def __init__(self,tokenizer,task,eos_token_id):
        self.root={};self.eos=int(eos_token_id);self.pad=getattr(tokenizer,'pad_token_id',None)
        self.answer_count=0;self.max_answer_tokens=0
        for text in valid_answers(task):
            tokens=tokenizer.encode(text,add_special_tokens=False)
            if self.eos in tokens:raise ProjectError('EOS occurs inside an answer')
            node=self.root
            for token in tokens:node=node.setdefault(int(token),{})
            node[self.eos]={};self.answer_count+=1;self.max_answer_tokens=max(self.max_answer_tokens,len(tokens))

    def allowed(self,prefix):
        node=self.root
        for index,token in enumerate(prefix):
            if int(token) not in node:raise ProjectError('Generation left the fixed answer grammar')
            node=node[int(token)]
            if int(token)==self.eos:
                if not all(int(x) in (self.eos,self.pad) for x in prefix[index+1:]):
                    raise ProjectError('Non-padding tokens after completed answer')
                return [self.eos]
        return list(node) or [self.eos]

    def callback(self,prompt_length):
        def next_tokens(batch_id,input_ids):
            return self.allowed(input_ids[prompt_length:].tolist())
        return next_tokens
