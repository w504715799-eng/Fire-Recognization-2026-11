"""Restore exact frozen metadata; no network, dependencies or model loading."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    receipt=json.loads((ROOT/'references/repository-metadata.json').read_text())
    for item in receipt['files']:
        target=ROOT/item['original'];compressed=ROOT/item['compressed']
        assert target.resolve().is_relative_to(ROOT)
        data=gzip.decompress(compressed.read_bytes())
        assert hashlib.sha256(data).hexdigest()==item['sha256']
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],f'Existing manifest differs: {target}'
        else:
            with target.open('xb') as out:out.write(data)
        print(f'Verified {item["original"]}')


if __name__=='__main__':main()
