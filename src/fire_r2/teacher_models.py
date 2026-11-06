"""Explicit candidate identity checks; no model imports in planning or identity validation."""
from pathlib import Path
from .common import ProjectError,read_json,child,sha256

QWEN3VL_ID='Qwen/Qwen3-VL-2B-Instruct'
QWEN3VL_REVISION='89644892e4d85e24eaac8bacfd4f463576704203'


def validate_identity(spec,config,receipt):
    if spec['model_id']!=QWEN3VL_ID or spec['revision']!=QWEN3VL_REVISION:raise ProjectError('Unexpected candidate model or revision')
    if receipt.get('model_id')!=spec['model_id'] or receipt.get('hf_revision')!=spec['revision']:
        raise ProjectError('Candidate receipt identity differs')
    if not receipt.get('runtime_content_verified_against_hf'):raise ProjectError('Candidate files have not passed content verification')
    if config.get('model_type')!='qwen3_vl' or 'Qwen3VLForConditionalGeneration' not in config.get('architectures',[]):
        raise ProjectError('Wrong architecture; do not reinterpret a Qwen3.5 checkpoint as Qwen3-VL')
    return {'model_id':spec['model_id'],'revision':spec['revision'],'architecture':'Qwen3VLForConditionalGeneration',
            'vocab_size':config['text_config']['vocab_size'],'image_token_id':config['image_token_id']}


def candidate_plan(root,spec):
    if spec.get('model_id')!=QWEN3VL_ID:raise ProjectError('Use the exact user-selected candidate ID')
    directory=child(root,spec['model_dir']);issues=[];identity=None
    for file in ('config.json','tokenizer.json','download-receipt.json','model.safetensors'):
        if not (directory/file).is_file():issues.append('missing:'+file)
    if not issues:identity=validate_identity(spec,read_json(directory/'config.json'),read_json(directory/'download-receipt.json'))
    return {'mode':'plan_only','model_id':spec['model_id'],'revision':spec['revision'],'directory':str(directory),
            'physical_directory':str(directory.resolve()),'issues':issues,'identity':identity,
            'model_loaded':False,'downloads_started':False,'fresh_processor_and_trie_required':True}


def checked_processor(root,spec,cloud):
    from .runtime import check_remote_environment
    check_remote_environment(root,cloud)
    plan=candidate_plan(root,spec)
    if plan['issues']:raise ProjectError('Candidate assets are missing: '+','.join(plan['issues']))
    directory=Path(plan['directory'])
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    return processor,{**plan['identity'],'tokenizer_sha256':sha256(directory/'tokenizer.json')}
