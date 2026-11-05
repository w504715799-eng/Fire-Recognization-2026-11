from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import socket
import sys
import time
import subprocess
import json

from .cloud import now, require_remote
from .common import ProjectError, child, file_lock, read_json, sha256, write_json


def verify_backend_source(direct, config):
    """Accept a pinned VCS checkout or the recorded SHA of that commit's source ZIP."""
    provenance=json.loads(direct) if direct else {}
    if provenance.get("vcs_info",{}).get("commit_id")==config["ultralytics_commit"]:
        return
    expected=config.get("ultralytics_archive_sha256")
    archive=provenance.get("archive_info",{})
    actual=archive.get("hashes",{}).get("sha256")
    if not actual and archive.get("hash","").startswith("sha256="):
        actual=archive["hash"].split("=",1)[1]
    if expected and actual==expected:
        return
    raise ProjectError("Ultralytics source differs from the configured commit/archive; record the verified source ZIP SHA when using archive installation")


def offline_connector(original):
    def connect(sock, address):
        # DataLoader workers need local IPC; block public network fetches, not IPC.
        if sock.family == socket.AF_UNIX:
            return original(sock, address)
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost"):
            return original(sock, address)
        raise ProjectError("External network access is disabled in training; stage assets first")
    return connect


def training_plan(root, config, cloud):
    if config.get("experiment") != "B_native":
        raise ProjectError("Only B_native is implemented. W/S/M must not silently run as an unmodified baseline.")
    if config["dataset"] not in ("dfire","homefire","fasdd_cv"):
        raise ProjectError("External-only data cannot be selected for training")
    if config.get("seed") not in (17,29,43):
        raise ProjectError("Expected a registered seed: 17, 29, or 43")
    if not config.get("run_name") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in config["run_name"]):
        raise ProjectError("Invalid run name")
    return {"execution":False,"experiment":config["experiment"],"dataset":config["dataset"],
            "view":str(child(root,config["prepared_view"])),"weights":str(child(root,config["weights"])),
            "remote_project":cloud["project_root"],"gpu":cloud["gpu_id"],
            "notice":"No model import, weight load, network access or training occurs in plan mode",
            "future_requirements":["user notified before paid startup/upgrade", "training phase approval",
                "complete verified data view", "existing locally staged cloud checkpoint with SHA-256",
                "project-specific sci-env", "shared cooperative GPU lock"],
            "train_parameters":config["train"],
            "checkpoint_policy":"native Ultralytics best/last plus best_fire selected by validation fire AP"}


def execute_training(root, config, cloud):
    plan=training_plan(root,config,cloud)
    require_remote(root,cloud,"training")  # Before any model library import or weight access.
    marker=Path(root)/".cloud-project.json"
    if not marker.is_file() or read_json(marker).get("hostname")!=socket.gethostname() or read_json(marker).get("server_id")!=cloud["approval"]["server_id"]:
        raise ProjectError("Initialize the approved staging environment first")
    if Path(sys.prefix).resolve()!=Path(cloud["environment_prefix"]).resolve():
        raise ProjectError("Use the cloud project's dedicated sci-env, not the other service's environment")
    view=Path(plan["view"]); info=read_json(view/"view.json")
    if info["dataset"]!=config["dataset"] or not info.get("training_ready"):
        raise ProjectError("Prepared view is not ready or belongs to another dataset")
    if info.get("training_yaml_sha256")!=sha256(view/"train.yaml"):
        raise ProjectError("Prepared training configuration changed")
    if info.get("view_manifest_sha256")!=sha256(view/"manifest.jsonl"):
        raise ProjectError("Prepared manifest changed")
    for role in ("fit","dev"):
        if info.get("role_list_sha256",{}).get(role)!=sha256(view/f"{role}.txt"):
            raise ProjectError(f"Prepared {role} list changed")
        paths=(view/f"{role}.txt").read_text().splitlines()
        if not paths or any(not Path(p).is_file() for p in paths):
            raise ProjectError(f"Missing cloud images in {role}")
    with (view/"manifest.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line)
            if row["role"] in ("fit","dev"):
                if sha256(child(view,row["view_label"]))!=row["view_label_sha256"]:
                    raise ProjectError(f"Derived training label changed: {row['sample_id']}")
    checkpoint=Path(plan["weights"])
    if checkpoint.suffix!=".pt" or not checkpoint.is_file() or not config.get("weights_sha256") or sha256(checkpoint)!=config["weights_sha256"]:
        raise ProjectError("Provide an existing verified cloud checkpoint; automatic weight download is disabled")
    direct=importlib.metadata.distribution("ultralytics").read_text("direct_url.json")
    verify_backend_source(direct, config)
    run=Path(root)/"runs"/config["run_name"]
    if run.exists():raise ProjectError("Run already exists; do not overwrite another experiment")
    gpu=str(cloud["gpu_id"])
    if not gpu.isdecimal():raise ProjectError("Specify one physical GPU index; no automatic GPU selection")
    lock=Path(cloud["shared_gpu_lock_root"])/f"gpu-{gpu}.lock"
    with file_lock(lock):
        active=subprocess.check_output(["nvidia-smi","-i",gpu,"--query-compute-apps=pid",
                                        "--format=csv,noheader,nounits"],text=True).strip()
        if active:
            raise ProjectError(f"GPU {gpu} already has compute processes ({active}); coordinate with the other service")
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            YOLO_OFFLINE="true",YOLO_CONFIG_DIR=str(Path(root)/".cache/ultralytics"),
            TORCH_HOME=str(Path(root)/".cache/torch"))
        # Prevent implicit network fetches in this single-GPU process.
        original_connect=socket.socket.connect
        socket.socket.connect=offline_connector(original_connect)
        try:
            import torch
            if not torch.cuda.is_available():raise ProjectError("No CUDA GPU; do not fall back to billable CPU training")
            from ultralytics import YOLO
            model=YOLO(str(checkpoint),task="detect")
            started=time.monotonic(); best_fire=-float("inf")
            budget=float(cloud["approval"]["stage_budget_hours"])*3600
            record={"plan":plan,"config":config,"cloud_server_id":cloud["approval"]["server_id"],
                "hourly_price":cloud["approval"]["hourly_price"],"currency":cloud["approval"]["currency"],
                "started_at":now(),"gpu_name":torch.cuda.get_device_name(0),"status":"running",
                "torch_version":torch.__version__,"ultralytics_version":importlib.metadata.version("ultralytics")}
            record_path=Path(root)/"artifacts/run-records"/(config["run_name"]+".json")
            write_json(record_path,record)
            def on_epoch_end(trainer):
                if time.monotonic()-started>=budget:
                    trainer.stop=True
                    print("USER_NOTICE: Training stage budget reached. Stopping at epoch boundary; server remains billable until you stop it.",flush=True)
            def on_model_save(trainer):
                nonlocal best_fire
                import shutil
                metric=trainer.validator.metrics.box
                ids=list(metric.ap_class_index)
                if 0 not in ids:
                    raise ProjectError("No fire validation AP; cannot select the required checkpoint")
                score=float(metric.maps[0])
                if score>best_fire:
                    best_fire=score
                    shutil.copy2(trainer.last,Path(trainer.save_dir)/"weights/best_fire.pt")
            model.add_callback("on_train_epoch_end",on_epoch_end)
            model.add_callback("on_model_save",on_model_save)
            args=dict(config["train"])
            prohibited={"data","device","project","name","resume","pretrained","val","save","exist_ok","amp","plots","seed"}
            if prohibited.intersection(args):raise ProjectError("Training config contains reserved execution fields")
            try:
                model.train(data=str(view/"train.yaml"),device=0,project=str(Path(root)/"runs"),
                    name=config["run_name"],seed=config["seed"],resume=False,exist_ok=False,
                    val=True,save=True,amp=False,plots=False,**args)
                record["status"]="completed_or_budget_stopped"
            except BaseException as exc:
                record.update(status="failed_or_interrupted",error=str(exc));raise
            finally:
                record.update(finished_at=now(),elapsed_seconds=time.monotonic()-started,best_fire_ap=best_fire if best_fire>-float('inf') else None,
                              server_powered_off=False)
                write_json(record_path,record)
                print("USER_NOTICE: Training process ended. Check the billable server and the other service before changing or stopping it.",flush=True)
        finally:
            socket.socket.connect=original_connect
    return record
