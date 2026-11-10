"""Bounded, isolated region diagnostic. No training, no source mutation, no new model assets."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,math,os,random,signal,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.common import ProjectError,read_json,write_json,sha256,output_path
from fire_r2.research_data import load_manifest
from fire_r2.evaluation import absolute_box,iou_xywh,validate_predictions


def digest(s):return hashlib.sha256(s.encode()).hexdigest()


def pick(pool,limit,salt):
    chosen=[];images=set()
    for item in sorted(pool,key=lambda x:digest(salt+'|'+x['id'])):
        if item['sample_id'] in images:continue
        chosen.append(item);images.add(item['sample_id'])
        if len(chosen)>=limit:break
    return chosen


def classify_detection(det,gt,boxes,used):
    overlaps=[iou_xywh(det['bbox'],b) for b in boxes]
    same=[i for i,g in enumerate(gt) if g['class_id']==det['class_id'] and overlaps[i]>=.5]
    unused=[i for i in same if i not in used]
    if unused:
        i=max(unused,key=lambda i:overlaps[i]);used.add(i);return 'tp',i
    if same:return 'duplicate',max(same,key=lambda i:overlaps[i])
    other=[i for i,g in enumerate(gt) if g['class_id']!=det['class_id'] and overlaps[i]>=.5]
    if other:return 'cross_class',max(other,key=lambda i:overlaps[i])
    # A small region inside a large GT may have low IoU but still contains foreground.
    ax,ay,aw,ah=det['bbox'];coverage=[]
    for bx,by,bw,bh in boxes:
        inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
        coverage.append(max(inter/max(aw*ah,1e-12),inter/max(bw*bh,1e-12)))
    if overlaps and (max(overlaps)>=.1 or max(coverage)>=.2):return 'localization',max(range(len(gt)),key=lambda i:max(overlaps[i],coverage[i]))
    return 'background',None


def prepare(cfg,out):
    if (out/'preparation.json').exists():raise ProjectError('Keep earlier region selection')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    artifact=read_json(ROOT/cfg['P0_predictions'])
    if artifact['checkpoint_sha256']!=cfg['P0_checkpoint_sha256'] or artifact['purpose']!='dev':raise ProjectError('Wrong P0 predictions')
    rows=load_manifest(ROOT/cfg['dfire_manifest'],roles={'dev'});pred=validate_predictions(rows,artifact)
    if artifact['manifest_sha256']!=sha256(ROOT/cfg['dfire_manifest']):raise ProjectError('P0 manifest mismatch')
    pools=collections.defaultdict(list);gts=collections.defaultdict(list);missed=collections.defaultdict(list);invalid=[]
    def item(row,box,ident,kind,cls,score=None,truth=None):
        x,y,w,h=box
        return {'id':ident,'sample_id':row['sample_id'],'image_path':row['image_relpath'],
                'source':row['sample_id'].split('/')[0],'role':row['role'],'state':row['state'],
                'image_sha256':row.get('image_sha256'),'bbox':box,'width':row['width'],'height':row['height'],
                'small640':w*h*(640/max(row['width'],row['height']))**2<1024,
                'kind':kind,'predicted_class':cls,'P0_score':score,'original_GT_class':truth,
                'diagnostic_half':int(digest('half|'+row['sample_id'])[:8],16)%2,
                'occlusion_annotation_available':False}
    for row in rows:
        gt=[b for b in row['boxes'] if not b.get('iscrowd',0) and min(b['bbox'][2:])>0]
        invalid.extend({'sample_id':row['sample_id'],'reason':'nonpositive_or_crowd_GT_not_queryable'} for b in row['boxes'] if b not in gt)
        boxes=[absolute_box(b,row) for b in gt];used=set()
        dets=sorted(pred[row['sample_id']],key=lambda d:-d['score'])
        for j,det in enumerate(dets):
            if det['score']<cfg['prediction_confidence_floor']:continue
            if min(det['bbox'][2:])<=0:invalid.append({'sample_id':row['sample_id'],'reason':'degenerate_prediction'});continue
            kind,match=classify_detection(det,gt,boxes,used)
            r=item(row,det['bbox'],row['sample_id']+f'|pred{j}','prediction_'+kind,det['class_id'],det['score'],gt[match]['class_id'] if match is not None else None)
            pools[(kind,det['class_id'])].append(r)
        for j,(g,box) in enumerate(zip(gt,boxes)):
            overlaps=[d['score'] for d in dets if d['class_id']==g['class_id'] and iou_xywh(d['bbox'],box)>=.5]
            score=max(overlaps,default=0.)
            r=item(row,box,row['sample_id']+f'|GT{j}','GT',g['class_id'],score,g['class_id']);gts[g['class_id']].append(r)
            if score<cfg['prediction_confidence_floor']:
                missed[g['class_id']].append({**r,'id':r['id']+'|missed','kind':'GT_missed'})
    selected=[];counts={}
    bins=[.05,.2,.4,.6,1.00001]
    for (kind,cls),pool in sorted(pools.items()):
        chosen=[]
        for i in range(4):chosen+=pick([r for r in pool if bins[i]<=r['P0_score']<bins[i+1]],cfg['candidate_cap_per_error_class']//4,cfg['selection_salt'])
        selected+=chosen;counts[f'{kind}_{cls}']={'available':len(pool),'selected':len(chosen)}
    for cls in (0,1):
        chosen=pick(gts[cls],cfg['dfire_GT_cap_per_class'],cfg['selection_salt']);selected+=chosen
        already={r['id'] for r in chosen}
        selected+=pick([r for r in missed[cls] if r['id'].removesuffix('|missed') not in already],cfg['dfire_missed_GT_cap_per_class'],cfg['selection_salt'])
    extra_counts={}
    for manifest in cfg['extra_fit_manifests']:
        extra=load_manifest(ROOT/manifest,roles={'fit'});pools2=collections.defaultdict(list)
        for row in extra:
            for j,g in enumerate(row['boxes']):
                if min(g['bbox'][2:])<=0 or g.get('iscrowd',0):continue
                box=absolute_box(g,row);pools2[g['class_id']].append(item(row,box,row['sample_id']+f'|GT{j}','external_GT',g['class_id'],truth=g['class_id']))
            if not row['boxes']:pools2[2].append(item(row,[0,0,row['width'],row['height']],row['sample_id']+'|empty','external_empty',None))
        chosen=[]
        for cls in (0,1,2):chosen+=pick(pools2[cls],cfg['extra_fit_empty_image_cap'] if cls==2 else cfg['extra_fit_GT_cap_per_class'],cfg['selection_salt'])
        selected+=chosen;extra_counts[manifest]={'selected':len(chosen),'role':'fit','P0_predictions_available':False}
    if len({r['id'] for r in selected})!=len(selected):raise ProjectError('Repeated region ID')
    for r in selected:
        actual=sha256(Path(r['image_path']))
        if r['image_sha256'] and actual!=r['image_sha256']:raise ProjectError('Source image changed')
        r['image_sha256']=actual
    with (out/'regions.jsonl').open('x') as f:
        for r in selected:f.write(json.dumps(r)+'\n')
    write_json(out/'protocol.json',cfg)
    write_json(out/'preparation.json',{'status':'completed','regions':len(selected),'queries':len(selected)*len(cfg['views']),
        'regions_sha256':sha256(out/'regions.jsonl'),'P0_predictions_sha256':sha256(ROOT/cfg['P0_predictions']),
        'source_manifests_sha256':{m:sha256(ROOT/m) for m in [cfg['dfire_manifest'],*cfg['extra_fit_manifests']]},
        'selection_counts':counts,'extra_sources':extra_counts,'unqueryable_records':invalid,'seconds':time.monotonic()-started,
        'annotation_changes':False,'source_images_deleted':False,'sample_design':'fixed hash per error/class/confidence bin; diagnostic prevalence is not population prevalence',
        'occlusion_limitation':'No original occlusion labels; small640 is measured, occlusion is not invented',
        'no_training_data_output':True})


def region_view(r,factor):
    from PIL import Image,ImageOps,ImageDraw
    with Image.open(r['image_path']) as f:im=ImageOps.exif_transpose(f).convert('RGB')
    if im.size!=(r['width'],r['height']):raise ProjectError('Image geometry differs')
    x,y,w,h=r['bbox'];pad=(factor-1)/2
    l=max(0,math.floor(x-pad*w));t=max(0,math.floor(y-pad*h));right=min(im.width,math.ceil(x+(1+pad)*w));bottom=min(im.height,math.ceil(y+(1+pad)*h))
    if right<=l or bottom<=t:raise ProjectError('Region outside image')
    im=im.crop((l,t,right,bottom));scale=640/max(im.size);im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(640,640),(114,114,114));dx,dy=(640-im.width)//2,(640-im.height)//2;canvas.paste(im,(dx,dy))
    outline=[max(dx,(x-l)*scale+dx),max(dy,(y-t)*scale+dy),min(dx+im.width-1,(x+w-l)*scale+dx),min(dy+im.height-1,(y+h-t)*scale+dy)]
    ImageDraw.Draw(canvas).rectangle(outline,outline='white',width=1);return canvas


def infer(cfg,out,cloud_path):
    from fire_r2.runtime import cloud_gpu
    from fire_r2.teacher_models import checked_processor
    prep=read_json(out/'preparation.json')
    if sha256(out/'regions.jsonl')!=prep['regions_sha256'] or read_json(out/'protocol.json')!=cfg:raise ProjectError('Frozen diagnostic input changed')
    if (out/'inference.json').exists():raise ProjectError('Keep earlier inference/failed attempt')
    queue=read_json(ROOT/cfg['wait_for_queue'])
    if queue['status']!='completed':raise ProjectError('seed29 queue has not completed; do not take GPU')
    cloud=read_json(ROOT/cloud_path);teacher=read_json(ROOT/cfg['teacher_config']);spec=read_json(ROOT/teacher['model_spec'])
    report={'status':'starting','teacher':teacher['teacher'],'model_training':False,'regions_sha256':prep['regions_sha256'],
            'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_queries':0,'max_seconds':cfg['max_gpu_seconds'],'batches':[]}
    started=time.monotonic();write_json(out/'inference.json',report)
    def timeout(*args):raise TimeoutError('Diagnostic GPU budget reached; no automatic extension')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(cfg['max_gpu_seconds'])
    try:
        with cloud_gpu(ROOT,cloud):
            import torch
            from transformers import Qwen3VLForConditionalGeneration
            from peft import PeftModel
            torch.set_num_threads(4);torch.set_num_interop_threads(1)
            processor,identity=checked_processor(ROOT,spec,cloud);processor.tokenizer.padding_side='left'
            adapter=ROOT/teacher['adapter']
            if sha256(adapter/'adapter_model.safetensors')!=teacher['teacher']['adapter_sha256']:raise ProjectError('Adapter differs')
            training=read_json(adapter.parent/'report.json')
            if training['identity']!=identity or training['status']!='completed':raise ProjectError('Adapter provenance differs')
            model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/spec['model_dir'],local_files_only=True,trust_remote_code=False,dtype=torch.float16,attn_implementation='sdpa').to('cuda')
            model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=False);model.eval();model.requires_grad_(False)
            report.update(identity=identity,status='running',load_seconds=time.monotonic()-started);torch.cuda.reset_peak_memory_stats()
            regions=[json.loads(s) for s in (out/'regions.jsonl').read_text().splitlines()]
            jobs=[(r,view) for view in cfg['views'] for r in regions]
            with (out/'scores.jsonl').open('x') as journal:
                for offset in range(0,len(jobs),cfg['batch_images']):
                    batch=jobs[offset:offset+cfg['batch_images']];tick=time.monotonic();texts=[];images=[];metas=[];tokens_batch=[]
                    for r,variant in batch:
                        if sha256(Path(r['image_path']))!=r['image_sha256']:raise ProjectError('Image changed')
                        choices=list(cfg['choices']);factor=1.0 if variant=='context1.0' else 1.5
                        if variant.endswith('reversed_options'):choices.reverse()
                        prompt=cfg['instruction']+'; '.join(f'{chr(65+i)}: {cfg["description"][c]}' for i,c in enumerate(choices))+'.'
                        view=region_view(r,factor);images.append(view)
                        message=[{'role':'user','content':[{'type':'image','image':view},{'type':'text','text':prompt}]}]
                        text=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True,enable_thinking=False)+'{"choice":"'
                        before=processor.tokenizer.encode(text,add_special_tokens=False);tokens=[]
                        for i in range(len(choices)):
                            after=processor.tokenizer.encode(text+chr(65+i),add_special_tokens=False)
                            if len(after)!=len(before)+1 or after[:-1]!=before:raise ProjectError('Choice token boundary changed')
                            tokens.append(after[-1])
                        if len(set(tokens))!=len(tokens):raise ProjectError('Choice tokens not distinct')
                        texts.append(text);tokens_batch.append(tokens)
                        metas.append({'id':r['id'],'view':variant,'choices':choices,'prompt_sha256':digest(prompt),'view_sha256':hashlib.sha256(view.tobytes()).hexdigest()})
                    inputs=processor(text=texts,images=images,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
                    with torch.inference_mode():logits=model(**inputs,logits_to_keep=1,use_cache=False).logits[:,-1,:]
                    for i,meta in enumerate(metas):
                        raw=logits[i,tokens_batch[i]].float();prob=torch.softmax(raw,dim=-1).cpu().tolist()
                        if not all(math.isfinite(x) for x in prob):raise ProjectError('Nonfinite teacher score')
                        meta.update(probabilities=dict(zip(meta['choices'],prob)),option_logits=raw.cpu().tolist(),option_token_ids=tokens_batch[i]);journal.write(json.dumps(meta)+'\n')
                    journal.flush();torch.cuda.synchronize();elapsed=time.monotonic()-tick
                    report['completed_queries']+=len(batch);report['batches'].append({'queries':len(batch),'seconds':elapsed});write_json(out/'inference.json',report)
                    print(json.dumps({'queries':report['completed_queries'],'total':len(jobs),'seconds':time.monotonic()-started}),flush=True)
            report.update(status='completed',peak_gpu_bytes=torch.cuda.max_memory_allocated(),scores_sha256=sha256(out/'scores.jsonl'))
    except BaseException as exc:report.update(status='failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:signal.alarm(0);report['seconds']=time.monotonic()-started;write_json(out/'inference.json',report)


def auc(labels,scores):
    pos=sum(labels);neg=len(labels)-pos
    if not pos or not neg:return None
    order=sorted(zip(scores,labels));rank_sum=0.;i=0
    while i<len(order):
        j=i+1
        while j<len(order) and order[j][0]==order[i][0]:j+=1
        rank_sum+=sum(y for _,y in order[i:j])*(i+1+j)/2;i=j
    return (rank_sum-pos*(pos+1)/2)/(pos*neg)


def metrics(rows,key):
    labels=[r['positive'] for r in rows];scores=[r[key] for r in rows];pos=sum(labels);neg=len(rows)-pos
    result={'regions':len(rows),'positive':pos,'negative':neg,'AUROC':auc(labels,scores),'retention':{}}
    if not pos or not neg:return result
    for target in (.8,.9,.95):
        threshold=sorted([r[key] for r in rows if r['positive']],reverse=True)[math.ceil(pos*target)-1]
        tp=sum(r['positive'] and r[key]>=threshold for r in rows);fp=sum(not r['positive'] and r[key]>=threshold for r in rows)
        result['retention'][str(target)]={'threshold':threshold,'TP_retained':tp,'positive_retention':tp/pos,'FP_retained':fp,'negative_retention':fp/neg}
    return result


def analyze(cfg,out):
    report=read_json(out/'inference.json');prep=read_json(out/'preparation.json')
    if report['status']!='completed' or sha256(out/'scores.jsonl')!=report['scores_sha256']:raise ProjectError('Complete unchanged inference required')
    if (out/'analysis.json').exists():raise ProjectError('Keep earlier analysis')
    regions=[json.loads(s) for s in (out/'regions.jsonl').read_text().splitlines()]
    scores=[json.loads(s) for s in (out/'scores.jsonl').read_text().splitlines()];lookup={(r['id'],r['view']):r for r in scores}
    if len(lookup)!=prep['queries']:raise ProjectError('All queries must be retained')
    keys=('P0','teacher','combined');result={'status':'completed','scope':'stratified diagnostic; ROI retention is not full-image detection recall or FAR',
        'scoring_is_not_calibrated_probability':True,'training_started':False,'label_changes':False,'candidate_ranking':{},'GT_support':{},'stability':{},'confidence_strata':{},'half_checks':{},'cost':{k:report[k] for k in ('seconds','load_seconds','peak_gpu_bytes','completed_queries')}}
    allpairs={}
    for cls in (0,1):
        subset=[r for r in regions if r['predicted_class']==cls and r['kind'] in ('prediction_tp','prediction_background','prediction_cross_class')]
        pairs=[];clsname=('fire','smoke')[cls]
        for r in subset:
            p=lookup[(r['id'],'context1.5')]['probabilities'];teacher=p[clsname]+p['both']
            pairs.append({**r,'positive':int(r['kind']=='prediction_tp'),'P0':r['P0_score'],'teacher':teacher,'combined':math.sqrt(r['P0_score']*teacher)})
        allpairs[clsname]=pairs;result['candidate_ranking'][clsname]={k:metrics(pairs,k) for k in keys}
        for half in (0,1):result['half_checks'][f'{clsname}_{half}']={k:metrics([r for r in pairs if r['diagnostic_half']==half],k) for k in keys}
        for name,selector in [('small640',lambda r:r['small640']),('not_small640',lambda r:not r['small640']),('both',lambda r:r['state']=='both')]:
            result['candidate_ranking'][clsname+'_'+name]={k:metrics([r for r in pairs if selector(r)],k) for k in keys}
        for lo,hi in zip([.05,.2,.4,.6],[.2,.4,.6,1.00001]):result['confidence_strata'][f'{clsname}_{lo}_{hi}']={k:metrics([r for r in pairs if lo<=r['P0']<hi],k) for k in keys}
        ids=sorted({r['sample_id'] for r in pairs});groups={ident:[r for r in pairs if r['sample_id']==ident] for ident in ids};rng=random.Random(17);deltas=[]
        for _ in range(cfg['bootstrap_replicates']):
            sample=[r for _ in ids for r in groups[rng.choice(ids)]];a=auc([r['positive'] for r in sample],[r['P0'] for r in sample]);b=auc([r['positive'] for r in sample],[r['combined'] for r in sample])
            if a is not None and b is not None:deltas.append(b-a)
        deltas.sort();result['candidate_ranking'][clsname]['paired_image_bootstrap']={'replicates':len(deltas),'combined_minus_P0_AUROC_percentile95':[deltas[int(.025*(len(deltas)-1))],deltas[int(.975*(len(deltas)-1))]] if deltas else None,'images_not_events':True}
    buckets=collections.defaultdict(list)
    for r in regions:
        p=lookup[(r['id'],'context1.5')]['probabilities'];top=max(p,key=p.get);cls=r['original_GT_class']
        if r['kind'].startswith(('GT','external')):
            support=p[('fire','smoke')[cls]]+p['both'] if cls in (0,1) else p['background']+p['light']
            buckets[(r['source'],r['kind'],cls,r['small640'])].append({'support':support,'top_agrees':top in ((('fire','smoke')[cls],'both') if cls in (0,1) else ('background','light')),'uncertain':p['uncertain']})
    for key,rs in buckets.items():result['GT_support'][str(key)]={'regions':len(rs),'mean_support':sum(r['support'] for r in rs)/len(rs),'top_agreement':sum(r['top_agrees'] for r in rs)/len(rs),'mean_uncertain':sum(r['uncertain'] for r in rs)/len(rs)}
    for variant in cfg['views'][1:]:
        agreements=[];distances=[]
        for r in regions:
            a=lookup[(r['id'],'context1.5')]['probabilities'];b=lookup[(r['id'],variant)]['probabilities']
            agreements.append(max(a,key=a.get)==max(b,key=b.get));distances.append(sum(abs(a[k]-b[k]) for k in a)/2)
        result['stability'][variant]={'regions':len(regions),'top_agreement':sum(agreements)/len(agreements),'mean_probability_total_variation':sum(distances)/len(distances)}
    result['limits']=['Original GT matching, not independent physical truth; light/reflection have no separate GT.','No original occlusion labels.','External sources have GT-region checks only, no P0 cross-source candidate claims.','Existing adapted teacher was trained on original labels; this is not proof of annotation-error detection.','No fine tuning of prompts/weights/thresholds based on this diagnostic; diagnostic halves are not final test sets.']
    write_json(out/'analysis.json',result)


def wait_and_run(cfg,out,cloud_path):
    started=time.monotonic();status=out/'wait-status.json'
    while True:
        queue=read_json(ROOT/cfg['wait_for_queue']);write_json(status,{'status':'waiting_for_seed29','queue_status':queue['status'],'wait_seconds':time.monotonic()-started})
        if queue['status']=='completed':break
        if queue['status'] not in ('waiting_for_own_baseline','validating'):raise ProjectError('seed29 needs attention; do not take GPU')
        if time.monotonic()-started>cfg['max_wait_hours']*3600:raise ProjectError('Bounded waiting expired; no GPU action')
        time.sleep(30)
    infer(cfg,out,cloud_path);analyze(cfg,out);write_json(status,{'status':'completed','waiting_seconds':time.monotonic()-started-read_json(out/'inference.json')['seconds']})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','infer','analyze','wait']);p.add_argument('--config',default='configs/region_diagnostic_20261110.json');p.add_argument('--cloud-config');a=p.parse_args()
    if sys.platform!='linux' or ROOT!=Path('/root/rivermind-data/FIRE_R2'):raise ProjectError('Cloud only; local imports/tests may not execute main')
    cfg=read_json(ROOT/a.config);out=output_path(ROOT,cfg['output'])
    if a.action=='prepare':prepare(cfg,out)
    elif a.action=='analyze':analyze(cfg,out)
    elif a.action=='infer':infer(cfg,out,a.cloud_config)
    else:wait_and_run(cfg,out,a.cloud_config)


if __name__=='__main__':main()
