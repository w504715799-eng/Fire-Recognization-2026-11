from pathlib import Path
from datetime import datetime, timezone
import json, shlex, subprocess, time
root=Path(__file__).resolve().parents[2]
session=root/'artifacts/cloud-session-2026-11-04'
host='root@hz01-ssh.gpuhome.cc'
ssh=['ssh','-S',str(root/'.ssh/gpuhome-data3.sock'),'-p','30188','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(root/'.ssh/known_hosts')]
receipts=[session/('home-'+x+'-transfer.json') for x in ('train','val','test')]
deadline=datetime(2026,11,4,10,25,tzinfo=timezone.utc).timestamp()
while time.time()<deadline:
    if all(p.exists() and json.loads(p.read_text()).get('status')=='verified_and_local_removed' for p in receipts):break
    time.sleep(3)
else:raise SystemExit('Preparation not started: uploads incomplete at conservative deadline')
print('All remote SHA checks passed; uploading receipts and preparing full Home-fire.',flush=True)
subprocess.run(['rsync','-a','-e',shlex.join(ssh),*[str(p) for p in receipts],host+':/root/rivermind-data/FIRE_R2/incoming/'],check=True,timeout=90)
command='cd /root/rivermind-data/FIRE_R2 && source cloud/activate.sh && python -B incoming/extract_home.py && python -B code/fire_r2_cli.py --datasets-root /root/rivermind-data/shared/datasets prepare --dataset homefire --output artifacts/data/homefire-v1'
subprocess.run(ssh+[host,'timeout 240 bash -c '+shlex.quote(command)+' > /root/rivermind-data/FIRE_R2/logs/homefire-prepare.log 2>&1'],check=True,timeout=270)
print('Full shared extraction and R2 frozen data view prepared.',flush=True)
