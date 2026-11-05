import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

from .common import ProjectError, output_path, read_json, write_json
from .data import load_dataset, inspect_images, prepare_view, verify_existing_manifest
from .cloud import cloud_plan, init_cloud, bundle_code, upload_file, upload_plan
from .train import training_plan, execute_training


def main(argv=None):
    parser=argparse.ArgumentParser(description="Fire R2 data preparation and explicit cloud entrypoints; no automatic downloads")
    parser.add_argument("--project-root",default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--registry",default="configs/datasets.json")
    parser.add_argument("--datasets-root",default=os.environ.get("FIRE_DATASETS_ROOT"))
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("audit","prepare"):
        p=sub.add_parser(command)
        p.add_argument("--dataset",required=True)
        p.add_argument("--output",required=command=="prepare")
        if command=="audit":p.add_argument("--check-images",action="store_true")
    for command in ("cloud-plan","cloud-init","train"):
        p=sub.add_parser(command)
        p.add_argument("--cloud-config",default="configs/cloud.example.json")
        if command!="cloud-plan":p.add_argument("--execute",action="store_true")
        if command=="train":p.add_argument("--config",default="configs/train_baseline.json")
    p=sub.add_parser("bundle");p.add_argument("--output",default="artifacts/staging/fire-r2-code.zip")
    p=sub.add_parser("upload")
    p.add_argument("--cloud-config",default="configs/cloud.example.json")
    p.add_argument("--source",required=True);p.add_argument("--kind",choices=["code","data"],required=True)
    p.add_argument("--remote-relative",required=True);p.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv);root=Path(args.project_root).resolve()
    try:
        if args.command in ("audit","prepare"):
            registry=Path(args.registry)
            if not registry.is_absolute():registry=root/registry
            rows,result=load_dataset(root,registry,args.dataset,args.datasets_root)
            frozen=read_json(registry)["datasets"][args.dataset].get("frozen_manifest")
            if frozen:
                if not (root/frozen).is_file():
                    raise ProjectError(f"Missing required frozen split manifest: {frozen}")
                result["frozen_manifest_check"]=verify_existing_manifest(rows,root/frozen)
            if args.command=="prepare":result=prepare_view(root,rows,result,args.output)
            else:
                if args.check_images:inspect_images(rows,result)
                if args.output:write_json(output_path(root,args.output),result)
        elif args.command=="bundle":result=bundle_code(root,args.output)
        else:
            cloud=read_json(root/args.cloud_config)
            if args.command=="cloud-plan":result=cloud_plan(cloud)
            elif args.command=="cloud-init":result=init_cloud(root,cloud) if args.execute else cloud_plan(cloud)
            elif args.command=="train":
                cfg=read_json(root/args.config)
                result=execute_training(root,cfg,cloud) if args.execute else training_plan(root,cfg,cloud)
            else:
                fn=upload_file if args.execute else upload_plan
                result=fn(root,cloud,args.source,args.kind,args.remote_relative)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    except (ProjectError, OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status":"blocked","reason":str(exc),"downloads_attempted":False},ensure_ascii=False),file=sys.stderr)
        return 2
