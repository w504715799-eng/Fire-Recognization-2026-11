"""Shared remote GPU ownership for teachers, prediction and integration checks."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys

from .cloud import require_remote
from .common import ProjectError,file_lock,read_json


def check_remote_environment(root,cloud):
    require_remote(root,cloud,'training')
    if Path(sys.prefix).resolve()!=Path(cloud['environment_prefix']).resolve():
        raise ProjectError('Use the existing project cloud sci-env')
    marker=read_json(Path(root)/'.cloud-project.json')
    if marker.get('hostname')!=socket.gethostname() or marker.get('server_id')!=cloud['approval']['server_id']:
        raise ProjectError('Cloud project identity mismatch')


@contextmanager
def cloud_gpu(root,cloud):
    check_remote_environment(root,cloud)
    gpu=str(cloud['gpu_id'])
    if not gpu.isdecimal():raise ProjectError('One physical GPU index is required')
    with file_lock(Path(cloud['shared_gpu_lock_root'])/f'gpu-{gpu}.lock'):
        active=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if active:raise ProjectError('GPU is occupied; no model was loaded and no other process was stopped')
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                          YOLO_OFFLINE='true',YOLO_CONFIG_DIR=str(Path(root)/'.cache/ultralytics'),
                          TORCH_HOME=str(Path(root)/'.cache/torch'))
        from .train import offline_connector
        previous=socket.socket.connect;socket.socket.connect=offline_connector(previous)
        try:
            import torch
            if not torch.cuda.is_available():raise ProjectError('CUDA unavailable; no CPU fallback')
            yield
        finally:socket.socket.connect=previous
