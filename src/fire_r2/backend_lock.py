"""Accept pinned VCS installs or the same verified source installed from an archive."""
import hashlib
import importlib.metadata
import json
from .common import ProjectError,read_json


def verify_backend(workspace,expected_revision):
    distribution=importlib.metadata.distribution('ultralytics')
    origin=json.loads(distribution.read_text('direct_url.json') or '{}')
    actual=origin.get('vcs_info',{}).get('commit_id')
    lock=read_json(workspace/'configs/ultralytics-lock.json')
    if lock['revision']!=expected_revision:raise ProjectError('Experiment and source-lock revisions differ')
    for relative,expected in lock['normalized_source_sha256'].items():
        path=distribution.locate_file(relative)
        text=path.read_text().replace('\r\n','\n').rstrip()+'\n'
        if hashlib.sha256(text.encode()).hexdigest()!=expected:raise ProjectError(f'Installed backend source differs: {relative}')
    if actual is not None and actual!=expected_revision:raise ProjectError('Installed VCS revision differs')
    return {'revision':expected_revision,'installation':origin,'source_files_verified':len(lock['normalized_source_sha256'])}
