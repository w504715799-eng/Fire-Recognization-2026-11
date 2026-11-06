"""Check the new model's own processor/trie and instruction token masks on CPU, without weights."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json
from fire_r2.teacher_models import checked_processor,candidate_plan
from fire_r2.teacher_protocols import AnswerTrie,valid_answers,protocol
from fire_r2.teacher_adaptation import supervised_value_spans,semantic_token_labels


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/teacher_qwen3vl2b_base.json')
    p.add_argument('--cloud-config',required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('Processor-only CPU check must disable CUDA')
    spec=read_json(ROOT/a.config);processor,identity=checked_processor(ROOT,spec,read_json(ROOT/a.cloud_config))
    from PIL import Image
    report={'identity':identity,'model_weights_loaded':False,'uses_gpu':False,'fresh_trie':{},'token_masks':[]}
    for task in ('A','B'):
        trie=AnswerTrie(processor.tokenizer,task,processor.tokenizer.eos_token_id)
        for text in valid_answers(task):
            tokens=processor.tokenizer.encode(text,add_special_tokens=False);prefix=[]
            for token in tokens:
                if token not in trie.allowed(prefix):raise ProjectError('New tokenizer trie mismatch')
                prefix.append(token)
            if trie.eos not in trie.allowed(prefix):raise ProjectError('New model EOS mismatch')
        report['fresh_trie'][task]={'answers':trie.answer_count,'max_new_tokens':trie.max_answer_tokens+1,
                                  'eos':trie.eos,'tokenizer_sha256':identity['tokenizer_sha256']}
    path=ROOT/'artifacts/qwen3vl-2b-preparation-20261106/instructions/teacher_train.instructions.jsonl'
    examples={}
    for line in path.open():
        r=json.loads(line);examples.setdefault(r['task'],r)
        if len(examples)==2:break
    for task,example in examples.items():
        image=Image.new('RGB',(640,640),'gray')
        message=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':example['instruction']}]}]
        prompt=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        target,spans=supervised_value_spans(example['target']);text=prompt+target+processor.tokenizer.eos_token
        plain=processor.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        inputs=processor(text=[text],images=[image],return_tensors='pt',add_special_tokens=False)
        ids=inputs['input_ids'][0].tolist()
        labels=semantic_token_labels(plain['input_ids'],plain['offset_mapping'],ids,
                                   [(len(prompt)+start,len(prompt)+end) for start,end in spans])
        selected=[i for i,x in enumerate(labels) if x!=-100]
        count=sum(x==identity['image_token_id'] for x in ids)
        if count<=0 or len(ids)>768:raise ProjectError('Image tokens or sequence budget invalid')
        report['token_masks'].append({'task':task,'input_length':len(ids),'image_tokens':count,
            'supervised_positions':selected,'supervised_token_text':processor.tokenizer.decode([labels[i] for i in selected]),
            'format_and_prompt_tokens_masked':True,'synthetic_processor_fixture_not_training':True})
    report['status']='passed';write_json(ROOT/'artifacts/qwen3vl-2b-preparation-20261106/processor-check.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
