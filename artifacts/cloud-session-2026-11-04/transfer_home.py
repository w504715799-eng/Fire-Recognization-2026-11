"""One-session, resumable transit of the three explicitly authorized Home-fire assets."""
from pathlib import Path
import hashlib, json, shlex, subprocess, time, zipfile, os, sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/cloud-session-2026-11-04'
DEST = '/root/rivermind-data/shared/datasets/home-fire/v1/raw'
HOST = 'root@hz01-ssh.gpuhome.cc'
SSH = ['ssh', '-S', str(ROOT/'.ssh/gpuhome-direct.sock'), '-p', '30188', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(ROOT/'.ssh/known_hosts')]
SSH[2]=str(ROOT/'.ssh'/os.environ.get('FIRE_TRANSFER_SOCKET','gpuhome-direct.sock'))
EXPECTED = {'test':407778390, 'val':352482306, 'train':1060672433}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def remote(code, *args):
    return subprocess.check_output(SSH+[HOST, '/opt/conda/bin/python -c '+shlex.quote(code)+' '+shlex.join(args)], text=True)

def save(name, data):
    (OUT/('home-'+name+'-transfer.json')).write_text(json.dumps(data,indent=2)+'\n')

pending = set(sys.argv[1:] or EXPECTED)
assert pending <= set(EXPECTED)
deadline = datetime(2026,11,4,10,25,tzinfo=timezone.utc).timestamp()
while pending and time.time()<deadline:
    for name in EXPECTED:
        if name not in pending:continue
        receipt_path=OUT/('home-'+name+'-transfer.json')
        if receipt_path.exists():
            old=json.loads(receipt_path.read_text())
            if old.get('status')=='verified_and_local_removed':
                pending.remove(name);continue
        partial=ROOT/'datasets/home-fire/v1/raw'/(name+'.zip.part')
        archive=partial.with_suffix('')
        current=archive if archive.exists() else partial
        if not current.exists() or current.stat().st_size!=EXPECTED[name]:continue
        with zipfile.ZipFile(current) as z:
            if z.testzip() is not None:raise RuntimeError('Bad ZIP CRC: '+name)
            info={'members':len(z.infolist()),'uncompressed_bytes':sum(i.file_size for i in z.infolist()),'first_members':z.namelist()[:8]}
        if info['uncompressed_bytes']>30_000_000_000:raise RuntimeError('Expanded size exceeds budget')
        sha=digest(current)
        if current==partial:partial.rename(archive)
        target=DEST+'/'+archive.name
        data={'source':'https://github.com/PengBo0/Home-fire-dataset/releases/download/v1.0.0/'+archive.name,'local':str(archive),'remote':target,'bytes':EXPECTED[name],'sha256':sha,'zip':info,'status':'uploading','local_file_deleted':False,'started_at':datetime.now(timezone.utc).isoformat()}
        save(name,data)
        print('CRC verified; uploading',name,EXPECTED[name],sha,flush=True)
        probe="from pathlib import Path; import sys,hashlib; p=Path(sys.argv[1]); print(hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if p.exists() else 'MISSING')"
        existing=remote(probe,target).strip()
        if existing not in ('MISSING',sha):raise RuntimeError('Remote target conflict: '+target)
        if existing=='MISSING':
            # Separate partial filename is never advertised as the verified source archive.
            subprocess.run(['rsync','-a','--partial','--progress','--timeout=120','-e',shlex.join(SSH),str(archive),HOST+':'+target+'.uploading'],check=True,timeout=max(1,deadline-time.time()))
            uploaded=remote(probe,target+'.uploading').strip()
            if uploaded!=sha:raise RuntimeError('Remote SHA256 mismatch: '+target)
            remote("from pathlib import Path; import sys; p=Path(sys.argv[1]); q=Path(sys.argv[2]); assert not q.exists(); p.rename(q)",target+'.uploading',target)
        data.update(status='remote_verified',remote_sha256=sha,remote_verified_at=datetime.now(timezone.utc).isoformat())
        save(name,data)
        archive.unlink()
        data.update(status='verified_and_local_removed',local_file_deleted=True,finished_at=datetime.now(timezone.utc).isoformat())
        save(name,data)
        print('REMOTE VERIFIED; local archive removed:',name,flush=True)
        pending.remove(name)
    if pending:time.sleep(3)
if pending:raise SystemExit('Window deadline; pending archives retained: '+','.join(sorted(pending)))
print('Requested archive(s) verified remotely; local transit file(s) removed: '+','.join(sys.argv[1:] or EXPECTED),flush=True)
