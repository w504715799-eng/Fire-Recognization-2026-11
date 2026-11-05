"""Inspect staged blobs without printing secrets or changing the index."""
from pathlib import PurePosixPath
import json
import gzip
import hashlib
import re
import subprocess

PRIVATE=re.compile(rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----')
TOKEN=re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,})\b')
PASSWORD=re.compile(rb'''(?i)["']?(?:password|passwd|access_token|api_key)["']?\s*[:=]\s*["']([^"'\n]{6,})["']''')
APPROVED_FINAL_MODEL={
    'results/homefire_B_native_seed17_full_20261105/model/best_fire_deploy.pt':
        ('9923b770c51f791482e5ae8bcd22a31fa888d368896d0c7c02edbccbf6496844',20338477)
}


def main():
    listing=subprocess.check_output(['git','ls-files','--stage','-z']).split(b'\0')
    blocked=[];total=0;files=0
    for entry in listing:
        if not entry:continue
        metadata,raw=entry.split(b'\t',1);mode,oid,stage=metadata.decode().split();name=raw.decode()
        p=PurePosixPath(name)
        if name in APPROVED_FINAL_MODEL:
            content=subprocess.check_output(['git','cat-file','blob',oid]);expected,size=APPROVED_FINAL_MODEL[name]
            files+=1;total+=len(content)
            if len(content)!=size or hashlib.sha256(content).hexdigest()!=expected:
                blocked.append({'file':name,'reason':'approved final model SHA/size mismatch'})
            continue
        if mode not in ('100644','100755'):
            blocked.append({'file':name,'reason':'symlink/submodule requires explicit publication review'});continue
        if p.parts[0] in {'.ssh','.local','.cache','.envs','data','datasets','tmp','weights','runs'} or p.suffix.lower() in {'.pt','.pth','.ckpt','.safetensors','.pem','.key','.onnx','.gguf','.zip','.part'} or name.endswith('.local.json'):
            blocked.append({'file':name,'reason':'private/runtime/model/dataset path'});continue
        content=subprocess.check_output(['git','cat-file','blob',oid]);files+=1;total+=len(content)
        if len(content)>=100_000_000:blocked.append({'file':name,'reason':'file >=100MB'})
        inspected=gzip.decompress(content) if name.endswith('.gz') else content
        if PRIVATE.search(inspected) or TOKEN.search(inspected):blocked.append({'file':name,'reason':'private key or access-token pattern'})
        for value in PASSWORD.findall(inspected):
            if value.lower() not in {b'example',b'redacted',b'placeholder',b'password'} and not value.startswith((b'${',b'<',b'os.environ')):
                blocked.append({'file':name,'reason':'possible credential value; inspect locally before publishing'});break
    print(json.dumps({'staged_files':files,'bytes':total,'blocked':blocked},ensure_ascii=False,indent=2))
    return bool(blocked)


if __name__=='__main__':raise SystemExit(main())
