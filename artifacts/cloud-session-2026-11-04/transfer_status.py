from pathlib import Path
import json, re, time
root=Path(__file__).resolve().parent
state={'time':time.time(),'progress_note':'rsync logical reconstruction bytes may reset on reconnection; only per-archive SHA verification means ready','files':{}}
for name,log in [('train','home-train-upload.log'),('val','home-val-upload.log'),('test','home-transfer.log')]:
    receipt=json.loads((root/('home-'+name+'-transfer.json')).read_text())
    found=re.findall(r'(\d+)\s+(\d+)%', (root/log).read_text(errors='replace')[-8000:])
    done=receipt['status']=='verified_and_local_removed'
    n=receipt['bytes'] if done else int(found[-1][0]) if found else 0
    state['files'][name]={'progress_bytes':n,'expected_bytes':receipt['bytes'],'percent':round(100*n/receipt['bytes'],1),'remote_sha256_verified':done,'local_file_deleted':receipt['local_file_deleted']}
state['total_progress_bytes']=sum(d['progress_bytes'] for d in state['files'].values())
history=root/'transfer-progress.jsonl'
previous=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
if previous:
    elapsed=state['time']-previous[-1]['time']
    delta=state['total_progress_bytes']-previous[-1]['total_progress_bytes']
    state['logical_progress_reset']=delta<0
    state['aggregate_mib_per_s']=delta/elapsed/1024**2 if delta>=0 else None
with history.open('a') as f:f.write(json.dumps(state)+'\n')
print(json.dumps(state,indent=2))
