"""Small synthetic Qwen3-VL adapter persistence check, no pretrained weights loaded."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import read_json,write_json,ProjectError
from fire_r2.runtime import check_remote_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--cloud-config',required=True);a=p.parse_args()
    check_remote_environment(ROOT,read_json(ROOT/a.cloud_config))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ProjectError('CPU-only serialization check')
    out=ROOT/'artifacts/lora-persistence-check-20261106'
    if out.exists():raise ProjectError('Keep existing serialization evidence')
    out.mkdir()
    import torch,peft
    from transformers import AutoConfig,Qwen3VLForConditionalGeneration
    torch.set_num_threads(1)
    spec=read_json(ROOT/'configs/qwen3vl2b_lora_cpu_full.json')
    config=AutoConfig.from_pretrained(ROOT/spec['model_dir'],local_files_only=True)
    t=config.text_config;t.num_hidden_layers=1;t.hidden_size=32;t.intermediate_size=64;t.num_attention_heads=4;t.num_key_value_heads=2;t.head_dim=8;t.vocab_size=64
    v=config.vision_config;v.depth=1;v.hidden_size=32;v.intermediate_size=64;v.num_heads=4;v.out_hidden_size=32
    model=Qwen3VLForConditionalGeneration(config)
    targets=[n for n,m in model.named_modules() if 'language_model.layers.' in n and n.endswith(('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj'))]
    model=peft.get_peft_model(model,peft.LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-4)
    loss=sum(p.square().sum()+p.sum() for p in params);loss.backward();optimizer.step()
    original={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.save_pretrained(out/'adapter',safe_serialization=True,save_embedding_layers=False)
    torch.save({'optimizer':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state()},out/'training-state.pt')
    loaded=peft.PeftModel.from_pretrained(Qwen3VLForConditionalGeneration(config),out/'adapter',local_files_only=True,is_trainable=True)
    restored={n:p for n,p in loaded.named_parameters() if p.requires_grad}
    if set(original)!=set(restored) or not all(torch.equal(p,restored[n]) for n,p in original.items()):raise ProjectError('Adapter save/reload mismatch')
    state=torch.load(out/'training-state.pt',weights_only=True);other=torch.optim.AdamW([p for p in loaded.parameters() if p.requires_grad],lr=1e-4);other.load_state_dict(state['optimizer'])
    report={'status':'passed','pretrained_weights_loaded':False,'GPU_used':False,'synthetic_small_config':True,
            'adapter_tensors_verified':len(original),'optimizer_tensors_restored':len(other.state),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    write_json(out/'report.json',report);print(json.dumps(report))


if __name__=='__main__':main()
