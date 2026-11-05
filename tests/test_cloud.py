from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import socket
import hashlib
import subprocess
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fire_r2.cli import main
from fire_r2.common import ProjectError, file_lock
from fire_r2.cloud import bundle_code, require_approval, require_remote, upload_plan, upload_file, validate_cloud
from fire_r2.train import execute_training, training_plan, offline_connector, verify_backend_source


class CloudContractTests(unittest.TestCase):
    def setUp(self):
        self.cloud=json.loads((ROOT/'configs/cloud.example.json').read_text())
        self.train=json.loads((ROOT/'configs/train_baseline.json').read_text())

    def test_backend_archive_and_vcs_provenance(self):
        verify_backend_source(json.dumps({'vcs_info':{'commit_id':self.train['ultralytics_commit']}}),self.train)
        with self.assertRaises(ProjectError):verify_backend_source('{}',self.train)
        self.train['ultralytics_archive_sha256']='a'*64
        verify_backend_source(json.dumps({'archive_info':{'hashes':{'sha256':'a'*64}}}),self.train)
        with self.assertRaises(ProjectError):verify_backend_source(json.dumps({'archive_info':{'hashes':{'sha256':'b'*64}}}),self.train)

    def test_default_plan_has_no_model_import_or_network(self):
        before=set(sys.modules)
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--project-root',str(ROOT),'train']),0)
                self.assertEqual(main(['--project-root',str(ROOT),'cloud-plan']),0)
        self.assertFalse({'torch','transformers','ultralytics'} & (set(sys.modules)-before))

    def test_unapproved_execution_fails_before_models(self):
        with self.assertRaises(ProjectError):execute_training(ROOT,self.train,self.cloud)
        self.assertNotIn('ultralytics',sys.modules)

    def test_paid_phase_must_be_notified_and_local_mac_cannot_train(self):
        with self.assertRaises(ProjectError):require_approval(self.cloud,'training')
        self.cloud['approval']={'phase':'training','server_id':'test','user_notified_at':'test',
            'user_approved_at':'test','hourly_price':1,'currency':'CNY','stage_budget_hours':1}
        require_approval(self.cloud,'training')
        with self.assertRaises(ProjectError):require_approval(self.cloud,'staging')
        with patch('platform.system',return_value='Darwin'):
            with self.assertRaises(ProjectError):require_remote(ROOT,self.cloud,'training')

    def test_explicit_training_on_existing_server_with_unknown_price(self):
        self.cloud['approval']={'phase':'training','server_id':'test','user_notified_at':'recorded',
            'user_approved_at':'recorded','hourly_price':None,'currency':None,'stage_budget_hours':.5,
            'existing_server_authorized':True,'training_explicitly_authorized':True,
            'authorization_expires_at':'2099-01-01T00:00:00+00:00'}
        require_approval(self.cloud,'training')
        self.cloud['approval']['training_explicitly_authorized']=False
        with self.assertRaises(ProjectError):require_approval(self.cloud,'training')

    def test_external_dataset_and_unimplemented_methods_rejected(self):
        self.train['dataset']='fasdd_uav'
        with self.assertRaises(ProjectError):training_plan(ROOT,self.train,self.cloud)
        self.train['dataset']='dfire';self.train['experiment']='B+W'
        with self.assertRaises(ProjectError):training_plan(ROOT,self.train,self.cloud)

    def test_cloud_project_cannot_own_shared_dataset_root(self):
        self.cloud['shared_dataset_root']='/workspace/fire-r2/datasets'
        with self.assertRaises(ProjectError):validate_cloud(self.cloud)

    def test_existing_server_window_allows_preparation_without_unknown_price(self):
        self.cloud['approval']={'phase':'staging','server_id':'test','user_notified_at':'recorded',
            'user_approved_at':'recorded','hourly_price':None,'currency':None,'stage_budget_hours':1,
            'existing_server_authorized':True,'authorization_expires_at':'2099-01-01T00:00:00+00:00'}
        require_approval(self.cloud,'staging')
        self.cloud['approval']['authorization_expires_at']='2000-01-01T00:00:00+00:00'
        with self.assertRaises(ProjectError):require_approval(self.cloud,'staging')
        self.cloud['approval'].update(phase='training',authorization_expires_at='2099-01-01T00:00:00+00:00')
        with self.assertRaises(ProjectError):require_approval(self.cloud,'training')

    def test_bundle_carries_configured_frozen_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir();(root/'references').mkdir()
            (root/'references/frozen.jsonl').write_text('{}\n')
            (root/'configs/datasets.json').write_text(json.dumps({'datasets':{'a':{'frozen_manifest':'references/frozen.jsonl'}}}))
            result=bundle_code(root,'artifacts/staging/code.zip')
            with zipfile.ZipFile(result['bundle']) as z:self.assertIn('references/frozen.jsonl',z.namelist())
            (root/'references/frozen.jsonl').unlink()
            with self.assertRaises(ProjectError):bundle_code(root,'artifacts/staging/another.zip')

    def test_gpu_lock_mutual_exclusion_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gpu-0.lock'
            with file_lock(path):
                with self.assertRaises(ProjectError):
                    with file_lock(path):pass
            with file_lock(path):pass

    def test_offline_training_preserves_dataloader_ipc(self):
        original=Mock(return_value=None)
        connect=offline_connector(original)
        connect(Mock(family=socket.AF_UNIX),'/tmp/worker-socket')
        connect(Mock(family=socket.AF_INET),('127.0.0.1',12345))
        self.assertEqual(original.call_count,2)
        with self.assertRaises(ProjectError):connect(Mock(family=socket.AF_INET),('huggingface.co',443))
        self.assertEqual(original.call_count,2)

    def test_bundle_whitelist_and_upload_path_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'src/fire_r2').mkdir(parents=True);(root/'src/fire_r2/a.py').write_text('pass')
            (root/'configs').mkdir();(root/'configs/cloud.local.json').write_text('secret')
            (root/'configs/cloud.example.json').write_text('{}')
            (root/'.env').write_text('secret')
            (root/'weights').mkdir();(root/'weights/model.pt').write_text('not real weights')
            result=bundle_code(root,'artifacts/staging/code.zip')
            with zipfile.ZipFile(result['bundle']) as z:
                self.assertEqual(set(z.namelist()),{'src/fire_r2/a.py','configs/cloud.example.json'})
            self.cloud['ssh_host']='user@host'
            plan=upload_plan(root,self.cloud,'artifacts/staging/code.zip','code','code.zip')
            self.assertFalse(plan['executed'])
            self.cloud.update(ssh_port=30188,ssh_known_hosts='/project/.ssh/known_hosts')
            configured=upload_plan(root,self.cloud,'artifacts/staging/code.zip','code','code.zip')
            self.assertIn('30188',configured['ssh'])
            self.assertIn('30188',configured['rsync'][5])
            with self.assertRaises(ProjectError):upload_plan(root,self.cloud,'weights/model.pt','code','code.zip')
            with self.assertRaises(ProjectError):upload_plan(root,self.cloud,'artifacts/staging/code.zip','code','../escape')

    def upload_fixture(self, root):
        p=root/'artifacts/staging/code.zip';p.parent.mkdir(parents=True);p.write_bytes(b'code-package')
        self.cloud.update(ssh_host='user@host',expected_hostname='cloud')
        self.cloud['approval']={'phase':'staging','server_id':'instance','user_notified_at':'recorded',
            'user_approved_at':'recorded','hourly_price':1,'currency':'CNY','stage_budget_hours':1}
        return p

    def test_upload_reuses_existing_verified_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=self.upload_fixture(root)
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            with patch('subprocess.run') as run, patch('subprocess.check_output',return_value=digest+'\n'):
                result=upload_file(root,self.cloud,'artifacts/staging/code.zip','code','code.zip')
            self.assertEqual(result['status'],'reused_verified')
            self.assertFalse(any(call.args[0][0]=='rsync' for call in run.call_args_list))
            self.assertTrue(p.exists())

    def test_upload_never_overwrites_conflicting_remote_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.upload_fixture(root)
            with patch('subprocess.run') as run, patch('subprocess.check_output',return_value='different\n'):
                with self.assertRaises(ProjectError):upload_file(root,self.cloud,'artifacts/staging/code.zip','code','code.zip')
            self.assertFalse(any(call.args[0][0]=='rsync' for call in run.call_args_list))
            receipt=json.loads(next((root/'artifacts/transfers').glob('*.json')).read_text())
            self.assertEqual(receipt['status'],'failed_or_interrupted')

    def test_failed_remote_lock_acquisition_does_not_release_other_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.upload_fixture(root)
            with patch('subprocess.run',side_effect=subprocess.CalledProcessError(1,'ssh')) as run:
                with self.assertRaises(subprocess.CalledProcessError):upload_file(root,self.cloud,'artifacts/staging/code.zip','code','code.zip')
            self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
