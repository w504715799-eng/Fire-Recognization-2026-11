from __future__ import annotations

from datetime import datetime, timezone
import os
import math
from pathlib import Path
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
import zipfile

from .common import ProjectError, child, file_lock, output_path, read_json, sha256, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_cloud(config):
    if config.get("project_id") != "fire-r2":
        raise ProjectError("Expected the dedicated fire-r2 project")
    paths = [config[k] for k in ("project_root", "shared_dataset_root", "shared_gpu_lock_root")]
    for value in paths:
        if not value or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in Path(value).parts:
            raise ProjectError(f"Invalid remote path: {value}")
    project, datasets, locks = map(Path, paths)
    if project == Path("/") or any(project == p or project in p.parents or p in project.parents for p in (datasets, locks)):
        raise ProjectError("Project, shared data and GPU lock roots must be separate")
    if config["environment_prefix"] != str(project/".envs/sci-env"):
        raise ProjectError("Cloud environment must be this project's .envs/sci-env")


def cloud_plan(config):
    validate_cloud(config)
    return {"project_id":"fire-r2", "project_root":config["project_root"],
        "environment_prefix":config["environment_prefix"], "shared_datasets":config["shared_dataset_root"],
        "stages":[
            {"stage":"staging", "gpu_required":False, "purpose":"Upload, verify and prepare data/code on CPU or an inexpensive GPU",
             "before_start":"Notify user of provider, instance, price/hour, storage and expected preparation time; user selects and starts instance"},
            {"stage":"training", "gpu_required":True, "purpose":"Train only after user approves the selected GPU and stage budget",
             "before_upgrade":"Report measured staging readiness and expected GPU memory/time; user decides upgrade"}],
        "automation":{"rent":False,"power_on":False,"upgrade":False,"power_off":False},
        "shared_server_rule":"Dedicated project/env/results; common read-only data and cooperative per-GPU locks",
        "current_execution":"No connection or paid operation is performed by cloud-plan"}


def require_approval(config, phase):
    validate_cloud(config)
    approval = config.get("approval", {})
    needed = ["server_id", "user_notified_at", "user_approved_at"]
    if not all(approval.get(k) for k in needed) or approval.get("phase") != phase:
        raise ProjectError(f"{phase} requires a recorded user notification and explicit stage approval")
    price = approval.get("hourly_price")
    priced = isinstance(price, (float,int)) and not isinstance(price,bool) and math.isfinite(price) and price >= 0 and approval.get("currency")
    # An explicit bounded window on a server already started by the user is
    # sufficient for preparation, even when the provider price is not supplied.
    existing_window = approval.get("existing_server_authorized") is True and (
        phase == "staging" or (phase == "training" and approval.get("training_explicitly_authorized") is True)
    )
    if existing_window:
        try:
            expires = datetime.fromisoformat(approval["authorization_expires_at"])
            existing_window = expires.tzinfo is not None and datetime.now(timezone.utc) < expires
        except (KeyError, TypeError, ValueError):
            existing_window = False
    if not priced and not existing_window:
        raise ProjectError("Record price/currency or an explicit unexpired approved phase on the user-started server")
    if approval.get("authorization_expires_at"):
        try:
            expires = datetime.fromisoformat(approval["authorization_expires_at"])
            valid = expires.tzinfo is not None and datetime.now(timezone.utc) < expires
        except (TypeError, ValueError):
            valid = False
        if not valid:raise ProjectError("The recorded cloud authorization window has expired or is invalid")
    budget = approval.get("stage_budget_hours")
    if not isinstance(budget,(int,float)) or isinstance(budget,bool) or not math.isfinite(budget) or budget <= 0:
        raise ProjectError("A positive stage time budget is required")


def require_remote(root, config, phase):
    require_approval(config, phase)
    if platform.system() != "Linux" or Path(root).resolve() != Path(config["project_root"]).resolve():
        raise ProjectError("This operation is restricted to the configured Linux cloud project, never the local Mac")
    expected = config.get("expected_hostname")
    if not expected or socket.gethostname() != expected:
        raise ProjectError("Cloud hostname must match the explicitly configured server")


def init_cloud(root, config):
    require_remote(root, config, "staging")
    for name in ("artifacts", "runs", "incoming", "weights", "logs", ".cache/pip", ".cache/conda", "tmp"):
        child(root,name).mkdir(parents=True,exist_ok=True)
    write_json(Path(root)/".cloud-project.json", {"project_id":"fire-r2", "hostname":socket.gethostname(),
        "server_id":config["approval"]["server_id"], "created_at":now()})
    return {"cloud_project_initialized":True, "environment_created":False, "training_started":False,
            "note":"This records an already user-started server; it does not start billing or provision a server"}


def bundle_code(root, destination):
    destination = output_path(root, destination)
    if destination.exists():
        raise ProjectError("Bundle exists; choose a new output name")
    destination.parent.mkdir(parents=True,exist_ok=True)
    selected = []
    for prefix in ("src/fire_r2", "configs", "docs", "tests", "cloud"):
        folder = Path(root)/prefix
        if folder.exists():
            for p in folder.rglob("*"):
                if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".md", ".sh", ".txt"):
                    if ".local." not in p.name:
                        selected.append(p)
    for name in ("code/fire_r2_cli.py", "火焰识别研究方案.md", "AGENTS.md", "RESEARCH_REQUIREMENTS.md"):
        p = Path(root)/name
        if p.is_file():selected.append(p)
    registry = Path(root)/"configs/datasets.json"
    if registry.is_file():
        for spec in read_json(registry).get("datasets", {}).values():
            if spec.get("frozen_manifest"):
                p = child(root, spec["frozen_manifest"])
                if not p.is_file():raise ProjectError(f"Missing frozen split manifest: {p}")
                selected.append(p)
    with zipfile.ZipFile(destination,"x",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(selected):archive.write(p,p.relative_to(root))
    receipt={"bundle":str(destination),"bytes":destination.stat().st_size,"sha256":sha256(destination),
             "files":[str(p.relative_to(root)) for p in sorted(selected)],"excludes":"datasets, weights, environments, caches, secrets and *.local.* configs"}
    write_json(destination.with_suffix(".receipt.json"), receipt)
    return receipt


def upload_plan(root, config, source, kind, remote_relative):
    validate_cloud(config)
    host = config.get("ssh_host")
    if not host or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*",host):
        raise ProjectError("Set ssh_host to a configured SSH host alias or user@hostname")
    source = child(root, source)
    if not source.is_file() or not source.resolve().is_relative_to(Path(root).resolve()):
        raise ProjectError("Upload source must be an existing file belonging to this project")
    allowed = Path(root)/("artifacts/staging" if kind=="code" else "datasets")
    if not source.absolute().is_relative_to(allowed.absolute()):
        raise ProjectError("Upload code bundles from artifacts/staging or data from shared datasets only")
    remote_root = Path(config["project_root"])/"incoming" if kind=="code" else Path(config["shared_dataset_root"])
    target = child(remote_root,remote_relative)
    port=config.get("ssh_port",22)
    if isinstance(port,bool) or not isinstance(port,int) or not 1 <= port <= 65535:
        raise ProjectError("Invalid SSH port")
    ssh = ["ssh","-p",str(port),"-o","BatchMode=yes","-o","StrictHostKeyChecking=yes"]
    for key, flag in (("ssh_known_hosts","-o"),("ssh_control_path","-S")):
        value=config.get(key)
        if value:
            ssh.extend([flag,"UserKnownHostsFile="+str(value) if key=="ssh_known_hosts" else str(value)])
    return {"source":str(source),"target":str(target),"ssh":ssh+[host],
            "rsync":["rsync","-aL","--partial","--progress","-e",
                shlex.join(ssh),str(source),host+":"+shlex.quote(str(target))],
            "source_bytes":source.stat().st_size,"executed":False}


def upload_file(root, config, source, kind, remote_relative):
    """Single-file upload with a remote atomic lock, checksum and no deletion."""
    require_approval(config,"staging")
    plan = upload_plan(root,config,source,kind,remote_relative)
    target = Path(plan["target"])
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+",str(target)):
        raise ProjectError("Remote upload path requires shell-safe ASCII components")
    if kind=="data":
        parts=Path(remote_relative).parts
        if len(parts)<4 or parts[2]!="raw":
            raise ProjectError("Shared data upload requires <dataset>/<version>/raw/<file>")
        lock=str(Path(config["shared_dataset_root"])/".locks"/(parts[0]+"__"+parts[1]+".lock"))
    else:
        lock = str(target)+".fire-upload-lock"
    if not config.get("expected_hostname"):
        raise ProjectError("Set the actual cloud hostname before executing upload")
    local_hash=sha256(plan["source"])
    code = "from pathlib import Path; import sys,socket; assert socket.gethostname()==sys.argv[3], 'Wrong cloud host'; p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); lock=Path(sys.argv[2]); lock.parent.mkdir(parents=True,exist_ok=True); lock.mkdir()"
    command = "python3 -c "+shlex.quote(code)+" "+shlex.quote(str(target))+" "+shlex.quote(lock)+" "+shlex.quote(config["expected_hostname"])
    local_lock = Path(root)/"artifacts/.locks"/(hashlib_name(str(target))+"-upload.lock")
    receipt_path=Path(root)/"artifacts/transfers"/(hashlib_name(str(target))+".json")
    with file_lock(local_lock):
        # Failure to acquire does not remove another service's lock.
        subprocess.run(plan["ssh"]+[command],check=True)
        receipt={**plan,"started_at":now(),"status":"uploading","local_file_deleted":False}
        write_json(receipt_path,receipt)
        try:
            existing = "from pathlib import Path; import sys,hashlib; p=Path(sys.argv[1]); h=hashlib.sha256(); f=p.open('rb') if p.exists() else None; " \
                "[h.update(c) for c in iter(lambda:f.read(1048576),b'')] if f else None; print(h.hexdigest() if f else 'MISSING')"
            remote_hash=subprocess.check_output(plan["ssh"]+["python3 -c "+shlex.quote(existing)+" "+shlex.quote(str(target))],text=True).strip()
            if remote_hash not in ("MISSING",local_hash):
                raise ProjectError("Remote target has different content; retain it and select a versioned destination")
            if remote_hash==local_hash:
                receipt.update(status="reused_verified",sha256=local_hash,finished_at=now(),executed=True)
                return receipt
            subprocess.run(plan["rsync"],check=True)
            check = subprocess.check_output(plan["ssh"]+["sha256sum -- "+shlex.quote(str(target))],text=True).split()[0]
            if check!=local_hash:raise ProjectError("Remote SHA-256 mismatch; local file retained")
            receipt.update(status="verified",sha256=local_hash,finished_at=now(),executed=True)
        except BaseException as exc:
            receipt.update(status="failed_or_interrupted",error=str(exc))
            raise
        finally:
            write_json(receipt_path,receipt)
            subprocess.run(plan["ssh"]+["rmdir -- "+shlex.quote(lock)],check=False)
    return receipt


def hashlib_name(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:20]
