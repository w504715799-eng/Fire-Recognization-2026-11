"""CPU-only check of real tokenizer grammar and batched image processor; no weights."""
from pathlib import Path
import json,os,platform,sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,write_json


def main():
    if platform.system()!='Linux' or ROOT!=Path('/root/rivermind-data/FIRE_R2'):
        raise ProjectError('This processor check belongs to the authorized cloud project')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('Disable CUDA for this CPU-only check')
    from transformers import AutoProcessor
    from PIL import Image
    from fire_r2.teacher_protocols import protocol,AnswerTrie,valid_answers
    from fire_r2.supervision import parse_advice
    model_dir=ROOT/'weights/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17'
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True);processor.tokenizer.padding_side='left'
    spec=protocol('v2');result={'model_weights_loaded':False,'uses_gpu':False,'protocol':'v2','tasks':{}}
    for task in ('A','B'):
        tokenizer=processor.tokenizer;trie=AnswerTrie(tokenizer,task,tokenizer.eos_token_id)
        for answer in valid_answers(task):
            ids=tokenizer.encode(answer,add_special_tokens=False);prefix=[]
            for token in ids:
                assert token in trie.allowed(prefix);prefix.append(token)
            assert trie.eos in trie.allowed(prefix)
            assert parse_advice(tokenizer.decode(ids),task)[1]=='ok'
        assert trie.max_answer_tokens+1<=192
        views=[Image.new('RGB',(640,640),c) for c in ('red','blue')];texts=[]
        for view in views:
            messages=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':spec[task]}]}]
            texts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
        inputs=processor(text=texts,images=views,padding=True,return_tensors='pt')
        grid=inputs['image_grid_thw'];pixels=inputs['pixel_values'];n=int(grid[0].prod())
        assert inputs['input_ids'].shape[0]==2 and grid.shape[0]==2
        assert not (pixels[:n]==pixels[n:2*n]).all()
        result['tasks'][task]={'answer_count':trie.answer_count,'max_new_tokens':trie.max_answer_tokens+1,
            'input_shape':list(inputs['input_ids'].shape),'pixel_values_shape':list(pixels.shape),
            'image_grid_thw':grid.tolist(),'different_images_have_different_pixels':True}
    result['status']='passed';write_json(ROOT/'artifacts/methods-20261106/teacher-protocol-cpu.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
