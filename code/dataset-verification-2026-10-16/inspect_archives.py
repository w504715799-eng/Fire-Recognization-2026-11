"""Bounded public ZIP inspection and verified downloads; no model libraries."""
import argparse
import collections
import hashlib
import io
import json
import os
import socket
from pathlib import Path, PurePosixPath
import time
import zipfile
import requests

# The host's IPv6 TLS route to Zenodo times out; IPv4 succeeds with normal TLS verification.
requests.packages.urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'references/dataset-verification-2026-10-16'
DATA = ROOT / 'data/supplementary-verification-2026-10-16'
CAP = 30_000_000_000
if Path(os.sys.prefix).name != 'sci-env':
    raise RuntimeError('Use sci-env')
OUT.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def metadata(key):
    url = ('https://api.figshare.com/v2/articles/28747046' if key == 'cqu'
           else 'https://zenodo.org/api/records/15826133')
    r = requests.get(url, timeout=(20, 45)); r.raise_for_status()
    obj = r.json(); save(key + '-metadata.json', obj)
    f = obj['files'][0]
    return {'key': key, 'metadata_url': url, 'name': f.get('name', f.get('key')),
            'bytes': f['size'], 'url': f.get('download_url') or f['links']['self'],
            'md5': f.get('computed_md5') or f['checksum'].split(':')[-1]}


class RangeFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.session = requests.Session()
        self.cache = {}
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0: raise ValueError('negative seek')
        return self.pos

    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b''
        if n > 15_000_000: raise ValueError('unexpected large metadata read')
        start, end = self.pos, self.pos + n - 1
        cache_key = (start, end)
        if cache_key not in self.cache:
            with self.session.get(self.url, headers={'Range': f'bytes={start}-{end}',
                                  'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 50)) as r:
                if r.status_code != 206:
                    raise RuntimeError(f'Range unsupported: status={r.status_code}; body not downloaded')
                expected = f'bytes {start}-{end}/{self.size}'
                if r.headers.get('Content-Range') != expected:
                    raise RuntimeError(f'Range mismatch: {r.headers.get("Content-Range")} != {expected}')
                raw = r.raw.read(n + 1)
                if len(raw) != n: raise RuntimeError('Range length mismatch')
            self.cache[cache_key] = raw; self.fetched += n
        self.pos += n
        return self.cache[cache_key]


def preflight(key):
    m = metadata(key)
    if not 0 < m['bytes'] <= CAP: raise ValueError('compressed size cap')
    remote = RangeFile(m['url'], m['bytes'])
    with zipfile.ZipFile(remote) as z:
        entries = [{'path': x.filename, 'bytes': x.file_size, 'compressed_bytes': x.compress_size,
                    'crc32': f'{x.CRC:08x}', 'encrypted': bool(x.flag_bits & 1), 'directory': x.is_dir()}
                   for x in z.infolist()]
        total = sum(x['bytes'] for x in entries)
        if total > CAP: raise ValueError('unpacked size cap')
        unsafe = [x['path'] for x in entries if PurePosixPath(x['path']).is_absolute() or '..' in PurePosixPath(x['path']).parts]
        ext = collections.Counter(PurePosixPath(x['path']).suffix.lower() for x in entries if not x['directory'])
        names = collections.Counter(x['path'] for x in entries)
        summary = {**m, 'uncompressed_bytes': total, 'entries': len(entries),
                   'files': sum(not x['directory'] for x in entries), 'extensions': dict(ext),
                   'range_bytes_read': remote.fetched, 'unsafe_paths': unsafe,
                   'duplicate_archive_names': [k for k,v in names.items() if v > 1],
                   'full_download_allowed': total <= CAP and not unsafe and not any(x['encrypted'] for x in entries)}
        save(key + '-zip-entries.json', entries); save(key + '-preflight.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        small = {}
        for x in z.infolist():
            if x.file_size <= 100_000 and (PurePosixPath(x.filename).suffix.lower() in ['.yaml', '.yml', '.names', '.json'] or 'readme' in x.filename.lower() or PurePosixPath(x.filename).name.lower() == 'classes.txt'):
                small[x.filename] = z.read(x).decode('utf-8', errors='replace')
                if len(small) >= 30: break
        save(key + '-package-metadata.json', small)
        print(json.dumps({'key':key,'package_metadata':small},ensure_ascii=False),flush=True)


def download(key):
    m = json.loads((OUT / (key + '-preflight.json')).read_text())
    if not m['full_download_allowed']: raise RuntimeError('preflight not passed')
    dest = DATA / key / 'original.zip'; dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f'{key}: existing archive retained; not downloading', flush=True); return
    partial = dest.with_suffix('.zip.partial')
    if partial.exists(): raise RuntimeError('Partial already exists; inspect before retry')
    md5, sha, total = hashlib.md5(), hashlib.sha256(), 0
    start = last = time.monotonic()
    with requests.get(m['url'], stream=True, headers={'Accept-Encoding':'identity'}, timeout=(20, 60)) as r:
        r.raise_for_status()
        length = r.headers.get('Content-Length')
        if length is not None and int(length) != m['bytes']: raise RuntimeError('length changed')
        with partial.open('xb') as f:
            for chunk in r.iter_content(1 << 20):
                total += len(chunk)
                if total > m['bytes']: raise RuntimeError('excess bytes')
                f.write(chunk); md5.update(chunk); sha.update(chunk)
                if time.monotonic()-last > 20:
                    print(f'{key}: {total}/{m["bytes"]} bytes',flush=True); last=time.monotonic()
    if total != m['bytes'] or md5.hexdigest() != m['md5']: raise RuntimeError('download checksum/size failed')
    partial.rename(dest)
    receipt = {'key':key,'path':str(dest.relative_to(ROOT)),'bytes':total,'md5':md5.hexdigest(),
               'expected_md5':m['md5'],'sha256':sha.hexdigest(),'seconds':time.monotonic()-start,
               'complete':True,'remote_uploaded':False,'local_deletion_allowed':False}
    save(key+'-download-receipt.json',receipt); print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','download']);p.add_argument('key',choices=['cqu','indoor'])
    args=p.parse_args()
    (preflight if args.action=='preflight' else download)(args.key)
