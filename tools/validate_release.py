#!/usr/bin/env python3
"""Check package completeness, anonymous paths, example formats and optional hashes."""
import argparse,ast,hashlib,json,re,sys
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hashes',action='store_true',help='Verify distributed files against SHA256SUMS')
    p.add_argument('--weights',action='store_true',help='Verify the downloaded model weights')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    import numpy as np
    from any2building.io import load_points,load_image
    errors=[]
    files=[f for f in root.rglob('*') if f.is_file() and not {'.git','__pycache__'}.intersection(f.relative_to(root).parts)]
    for f in files:
        if f.is_symlink():errors.append(f'External link: {f.relative_to(root)}')
        if f.suffix=='.py':
            try:ast.parse(f.read_text())
            except SyntaxError as e:errors.append(f'Syntax error: {f.relative_to(root)}: {e}')
        if f.suffix in {'.py','.json','.jsonl','.md','.txt','.sh','.toml'}:
            text=f.read_text(errors='replace')
            if re.search(r'/hpc[0-9a-z]+/home/|/home/[A-Za-z0-9_.-]+/|gpu[0-9]+-[0-9]+',text):
                errors.append(f'Private filesystem/node reference: {f.relative_to(root)}')
    manifest=json.loads((root/'examples/manifest.json').read_text())
    for row in manifest:
        if row['mode'] in ('point','both'):
            points=load_points(root/row['points'])
            if np.abs(points).max()>.501:errors.append(f'Point frame: {row["id"]}')
        if row['mode']!='point':
            im=load_image(root/row['image'])
            if im.shape!=(3,512,512):errors.append(f'Image shape: {row["id"]}')
        for key,value in row['input_sha256'].items():
            if sha(root/row[key])!=value:errors.append(f'Input hash: {row["id"]}/{key}')
    weights=json.loads((root/'weights/manifest.json').read_text())
    if a.weights:
        for w in weights:
            f=root/w['file']
            if not f.is_file() or f.stat().st_size!=w['bytes']:errors.append(f'Weight missing/size: {w["file"]}')
            elif a.hashes and sha(f)!=w['sha256']:errors.append(f'Weight checksum: {w["file"]}')
    if a.hashes:
        for line in (root/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split('  ',1)
            f=root/name
            if not f.is_file() or sha(f)!=expected:errors.append(f'Package checksum: {name}')
    weight_binaries=[f for f in files if f.suffix.lower() in {'.pt','.pth','.safetensors','.ckpt','.bin'}]
    report=dict(success=not errors,files=len(files),example_cases=len(manifest),weight_binaries_present=len(weight_binaries),external_weights_checked=a.weights,package_hashes_checked=a.hashes,errors=errors)
    print(json.dumps(report,indent=2))
    if errors:sys.exit(1)


if __name__=='__main__':main()
