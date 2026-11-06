"""Reproduce the cloud meta-device LoRA target check without loading weights."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('CPU/meta check must disable CUDA')
    out=ROOT/a.output
    if out.exists():raise ProjectError('Keep prior check report')
    import torch,peft,accelerate
    from transformers import AutoConfig,Qwen3VLForConditionalGeneration
    torch.set_num_threads(1);spec=read_json(ROOT/'configs/qwen3vl2b_lora_pilot.json')
    config=AutoConfig.from_pretrained(ROOT/spec['model_dir'],local_files_only=True)
    with torch.device('meta'):
        model=Qwen3VLForConditionalGeneration(config)
        targets=[n for n,m in model.named_modules() if 'language_model.layers.' in n and n.endswith(('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj'))]
        model=peft.get_peft_model(model,peft.LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
        trainable=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    if len(targets)!=112 or not all('lora_' in n and 'language_model' in n for n,_ in trainable):raise ProjectError('LoRA targets differ from frozen design')
    write_json(out,{'status':'passed_meta_structure_only','peft':peft.__version__,'accelerate':accelerate.__version__,
        'weights_loaded':False,'GPU_used':False,'target_modules':len(targets),'trainable_parameters':sum(n for _,n in trainable),
        'all_trainable_are_language_lora':True,'real_training_step_passed':False})


if __name__=='__main__':main()
