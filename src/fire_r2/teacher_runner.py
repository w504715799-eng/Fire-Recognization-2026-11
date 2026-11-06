"""Frozen offline teacher bridge. Only explicit cloud execution can load local weights."""
import hashlib
import platform
from pathlib import Path
import time
import json

from .common import ProjectError,read_json,sha256,write_json,child,output_path,file_lock
from .supervision import (fit_rows,load_manifest,selected_regions,parse_advice,SCENE_PROMPT,REGION_PROMPT)


def teacher_view(image_path,box=None,size=640,region_marker='number_before_resize'):
    from PIL import Image,ImageDraw,ImageOps
    with Image.open(image_path) as source:image=ImageOps.exif_transpose(source).convert('RGB')
    outline=None
    if box is not None:
        x,y,w,h=box['bbox']
        if box['format']=='cxcywh_normalized':
            x,y,w,h=(x-w/2)*image.width,(y-h/2)*image.height,w*image.width,h*image.height
        if w<=0 or h<=0:raise ProjectError('Degenerate ROI is not a teacher query')
        left=max(0,int(x-.25*w));top=max(0,int(y-.25*h))
        right=min(image.width,int(x+1.25*w+1));bottom=min(image.height,int(y+1.25*h+1))
        if right<=left or bottom<=top:raise ProjectError('ROI context is outside the image')
        image=image.crop((left,top,right,bottom))
        outline=[x-left,y-top,x+w-left,y+h-top]
        if region_marker=='number_before_resize':
            draw=ImageDraw.Draw(image);draw.rectangle(outline,outline='white',width=1)
            draw.text((max(x-left,0),max(y-top,0)),'1',fill='white',stroke_width=1,stroke_fill='black')
        elif region_marker!='thin_outline_after_resize':raise ProjectError('Unknown region marker')
    ratio=min(size/image.width,size/image.height)
    image=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),(114,114,114));dx,dy=(size-image.width)//2,(size-image.height)//2
    canvas.paste(image,(dx,dy))
    if outline is not None and region_marker=='thin_outline_after_resize':
        # Drawing after resizing avoids enlarging a 10-pixel glyph over a tiny ROI.
        transformed=[outline[0]*ratio+dx,outline[1]*ratio+dy,outline[2]*ratio+dx,outline[3]*ratio+dy]
        ImageDraw.Draw(canvas).rectangle(transformed,outline='white',width=1)
    return canvas


def build_cache(rows,manifest_hash,root,infer,revision,checkpoint=None,initial_samples=None,protocol_version='v1'):
    from .teacher_protocols import protocol
    spec=protocol(protocol_version);scene_prompt=spec['A'];region_prompt=spec['B']
    cache={'schema_version':1,'teacher_id':'Qwen/Qwen3.5-0.8B','teacher_revision':revision,
           'manifest_sha256':manifest_hash,'protocol':protocol_version,
           'prompt_hashes':{'A':hashlib.sha256(scene_prompt.encode()).hexdigest(),
           'B':hashlib.sha256(region_prompt.encode()).hexdigest()},'samples':dict(initial_samples or {})}
    if not set(cache['samples'])<={r['sample_id'] for r in fit_rows(rows)}:raise ProjectError('Resume cache includes non-fit images')
    for row in fit_rows(rows):
        image=Path(root)/row['image_relpath'];image_hash=sha256(image)
        if row.get('image_sha256') and image_hash!=row['image_sha256']:raise ProjectError('Source image changed since preparation')
        if row['sample_id'] in cache['samples']:
            if cache['samples'][row['sample_id']]['image_sha256']!=image_hash:raise ProjectError('Resume cache input image changed')
            continue
        item={'image_sha256':image_hash,'semantic':None,'regions':[],'calls':[]}
        tasks=[('A',None,scene_prompt)]+[('B',b,region_prompt) for b in selected_regions(row)]
        for task,box,prompt in tasks:
            start=time.monotonic();raw='';tokens=None;error=None;answer=None;status='technical_failure'
            # One retry for a technical exception only, never to obtain agreement with GT.
            for attempt in range(2):
                try:
                    raw,tokens=infer(teacher_view(image,box,region_marker=spec['region_marker']),prompt)
                    answer,status=parse_advice(raw,task);error=None;break
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
            call={'task':task,'raw_response':raw,'parse_status':status,'seconds':time.monotonic()-start,
                  'output_tokens':tokens,'attempts':attempt+1,'technical_error':error,
                  'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                  'view':'letterbox_640' if task=='A' else 'context_crop_1.5_'+spec['region_marker'],
                  'original_annotation_id':box['original_annotation_id'] if box else None}
            item['calls'].append(call)
            if task=='A':item['semantic']=answer
            else:item['regions'].append({'original_annotation_id':box['original_annotation_id'],'response':answer})
        cache['samples'][row['sample_id']]=item
        if checkpoint is not None:checkpoint(row['sample_id'],item)
    return cache


def build_batched_cache(rows,manifest_hash,root,infer_batch,revision,batch_size,checkpoint=None,
                        initial_samples=None,protocol_version='v2'):
    """Batch independent queries; split/hash/ROI choices remain identical to serial mode."""
    from .teacher_protocols import protocol
    spec=protocol(protocol_version);samples=dict(initial_samples or {});fit=fit_rows(rows)
    if not isinstance(batch_size,int) or batch_size<1:raise ProjectError('Positive teacher batch size required')
    if not set(samples)<={r['sample_id'] for r in fit}:raise ProjectError('Resume includes non-fit samples')
    pending=[]
    for row in fit:
        digest=sha256(Path(root)/row['image_relpath'])
        if row.get('image_sha256') and row['image_sha256']!=digest:raise ProjectError('Original image changed')
        if row['sample_id'] in samples:
            if samples[row['sample_id']]['image_sha256']!=digest:raise ProjectError('Resume image changed')
        else:pending.append((row,digest))
    for start in range(0,len(pending),batch_size):
        chunk=pending[start:start+batch_size]
        items={r['sample_id']:{'image_sha256':digest,'semantic':None,'regions':[],'calls':[]} for r,digest in chunk}
        for task in ('A','B'):
            jobs=[(r,b) for r,_ in chunk for b in ([None] if task=='A' else selected_regions(r))]
            for offset in range(0,len(jobs),batch_size):
                group=jobs[offset:offset+batch_size];tick=time.monotonic();error=None;responses=None
                views=[teacher_view(Path(root)/r['image_relpath'],b,region_marker=spec['region_marker']) for r,b in group]
                for attempt in range(2):
                    try:
                        responses=infer_batch(views,spec[task],task)
                        if len(responses)!=len(group):raise ProjectError('Teacher batch returned incomplete coverage')
                        error=None;break
                    except Exception as exc:error=f'{type(exc).__name__}: {exc}';responses=None
                elapsed=time.monotonic()-tick
                for index,(row,box) in enumerate(group):
                    raw,tokens=responses[index] if responses is not None else ('',None)
                    answer,status=parse_advice(raw,task) if responses is not None else (None,'technical_failure')
                    item=items[row['sample_id']]
                    item['calls'].append({'task':task,'raw_response':raw,'parse_status':status,
                        'seconds':elapsed/len(group),'batch_seconds':elapsed,'batch_size':len(group),
                        'timing_scope':'amortized_batch_wall_time','output_tokens':tokens,'attempts':attempt+1,
                        'technical_error':error,'prompt_sha256':hashlib.sha256(spec[task].encode()).hexdigest(),
                        'view':'letterbox_640' if task=='A' else 'context_crop_1.5_'+spec['region_marker'],
                        'original_annotation_id':box['original_annotation_id'] if box else None})
                    if task=='A':item['semantic']=answer
                    else:item['regions'].append({'original_annotation_id':box['original_annotation_id'],'response':answer})
        for row,_ in chunk:
            ident=row['sample_id'];samples[ident]=items[ident]
            if checkpoint is not None:checkpoint(ident,items[ident])
    return {'schema_version':1,'teacher_id':'Qwen/Qwen3.5-0.8B','teacher_revision':revision,
            'manifest_sha256':manifest_hash,'protocol':protocol_version,
            'prompt_hashes':{task:hashlib.sha256(spec[task].encode()).hexdigest() for task in ('A','B')},'samples':samples}


def pilot_selection(rows,per_state):
    """Engineering timing sample, selected before inference, never a reduced benchmark."""
    if not isinstance(per_state,int) or per_state<1:raise ProjectError('Positive pilot count required')
    selected=[]
    for state in ('fire_only','smoke_only','both','empty'):
        pool=[r for r in fit_rows(rows) if r['state']==state]
        selected.extend(sorted(pool,key=lambda r:hashlib.sha256(('teacher-pilot-v1|'+r['sample_id']).encode()).hexdigest())[:per_state])
    return selected


def teacher_plan(root,config):
    manifest=child(root,config['manifest']);model_dir=child(root,config['model_dir'])
    output=output_path(root,config['output'])
    return {'mode':'plan_only','manifest':str(manifest),'model_dir':str(model_dir),'output':str(output),
            'scope':'pilot_fit_only' if config.get('pilot_per_state') else 'all_source_fit',
            'teacher_revision':config['teacher_revision'],'model_loaded':False,'downloads':False}


def run_teacher(root,config,cloud,execute=False):
    plan=teacher_plan(root,config)
    if not execute:return plan
    from .runtime import check_remote_environment,cloud_gpu
    check_remote_environment(root,cloud)
    from .teacher_protocols import protocol,AnswerTrie
    spec=protocol(config.get('protocol','v1'));batch_size=int(config.get('batch_size',1))
    manifest=Path(plan['manifest']);model_dir=Path(plan['model_dir']);output=Path(plan['output'])
    if output.exists():raise ProjectError('Completed teacher cache is immutable; choose a new output')
    receipt=read_json(model_dir/'download-receipt.json')
    if receipt.get('hf_revision')!=config['teacher_revision'] or not receipt.get('runtime_content_verified_against_hf'):
        raise ProjectError('Verified fixed teacher revision is required')
    # Verify assets by SHA before loading; no implicit download or trust_remote_code.
    for f in receipt['files']:
        if not f.get('documentation') and sha256(child(model_dir,f['path']))!=f['sha256']:
            raise ProjectError('Teacher runtime file changed: '+f['path'])
    rows=load_manifest(manifest,roles={'fit'});full_fit_count=len(rows)
    if config.get('pilot_per_state'):rows=pilot_selection(rows,config['pilot_per_state'])
    mh=sha256(manifest)
    from .supervision import digest_object
    binding={'manifest_sha256':mh,'teacher_revision':config['teacher_revision'],
             'selection_sha256':digest_object(sorted(r['sample_id'] for r in rows)),
             'scope':plan['scope'],'protocol':spec['version'],'batch_size':batch_size,
             'prompt_hashes':[hashlib.sha256(spec[task].encode()).hexdigest() for task in ('A','B')],
             'parser':'exact_json_or_single_json_fence_v1',
             'generation':{'do_sample':False,'max_new_tokens':192,'enable_thinking':False,'dtype':'float16',
                           'constrained_answer_grammar':spec['constrained']}}
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix('.partial.jsonl');meta=partial.with_suffix('.meta.json')
    with file_lock(output.with_suffix('.lock')),cloud_gpu(root,cloud):
        if output.exists():raise ProjectError('Output appeared while acquiring lock')
        initial={}
        if partial.exists():
            if not config.get('resume') or read_json(meta)!=binding:raise ProjectError('Explicit resume with identical cache binding required')
            for line in partial.read_text().splitlines():
                record=json.loads(line)
                if record['sample_id'] in initial:raise ProjectError('Duplicate teacher journal record')
                initial[record['sample_id']]=record['item']
        else:write_json(meta,binding)
        import torch
        import importlib.metadata
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
        processor.tokenizer.padding_side='left'
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,dtype=torch.float16,
                                                            attn_implementation='sdpa').to('cuda').eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        eos=model.generation_config.eos_token_id
        eos=eos[0] if isinstance(eos,list) else eos
        tries={task:AnswerTrie(processor.tokenizer,task,eos) for task in ('A','B')} if spec['constrained'] else {}
        def infer_batch(views,prompt,task):
            texts=[]
            for view in views:
                messages=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':prompt}]}]
                texts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
            inputs=processor(text=texts,images=views,padding=True,return_tensors='pt').to('cuda')
            extra={}
            if tries:
                extra={'prefix_allowed_tokens_fn':tries[task].callback(inputs['input_ids'].shape[1]),
                       'eos_token_id':eos,'max_new_tokens':tries[task].max_answer_tokens+1}
            else:extra={'max_new_tokens':192}
            with torch.inference_mode():generated=model.generate(**inputs,do_sample=False,**extra)
            output_ids=generated[:,inputs['input_ids'].shape[1]:]
            answers=processor.batch_decode(output_ids,skip_special_tokens=True)
            lengths=[]
            for ids in output_ids.tolist():lengths.append(ids.index(eos)+1 if eos in ids else len(ids))
            return list(zip(answers,lengths))
        limit=min(float(config.get('max_seconds',3600)),float(cloud['approval']['stage_budget_hours'])*3600)
        with partial.open('a' if partial.exists() else 'x') as journal:
            def record(ident,item):
                journal.write(json.dumps({'sample_id':ident,'item':item},ensure_ascii=False)+'\n');journal.flush()
                if time.monotonic()-started>limit:raise ProjectError('Teacher stage budget reached; fit journal retained for exact resume')
            cache=build_batched_cache(rows,mh,'/',infer_batch,config['teacher_revision'],batch_size,
                                      checkpoint=record,initial_samples=initial,protocol_version=spec['version'])
        calls=[c for item in cache['samples'].values() for c in item['calls']]
        cache.update(scope=plan['scope'],full_fit_count=full_fit_count,selection_sha256=binding['selection_sha256'],
                     generation=binding['generation'],parser=binding['parser'],batch_size=batch_size,
                     runtime={'elapsed_seconds':time.monotonic()-started,
                         'runner_sha256':sha256(__file__),
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'calls':len(calls),
                         'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
                         'gpu':torch.cuda.get_device_name(0),
                         'parse_ok':sum(c['parse_status']=='ok' for c in calls),
                         'technical_failures':sum(c['parse_status']=='technical_failure' for c in calls)})
        write_json(output,cache)
        # Preserve the successful journal and provenance as reproducibility evidence.
        return {'output':str(output),'fit_images':len(rows),'scope':cache['scope'],'runtime':cache['runtime'],'training_started':False}
