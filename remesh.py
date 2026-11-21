#!/usr/bin/env python3
"""Optional RoLoPM adapter for compact LoD2 geometry."""
import argparse
from pathlib import Path
import json
import subprocess
import shlex


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True,help='Official SurfaceRemeshingCli_bin.exe')
    p.add_argument('--runner',default='',help='Optional command prefix, e.g. wine (Linux)')
    p.add_argument('--screen-size',type=int,default=100)
    a=p.parse_args()
    if a.outdir.exists():raise FileExistsError('Choose a fresh output directory')
    a.outdir.mkdir(parents=True)
    import trimesh
    src=a.outdir/'input.obj';trimesh.load(a.mesh,force='mesh',process=False).export(src)
    runner=shlex.split(a.runner)
    uses_wine=any(Path(part).name in ('wine','wine64') for part in runner)
    def cli_path(path):
        value=str(path.resolve())
        return 'Z:'+value.replace('/','\\') if uses_wine else value
    cmd=runner+[str(a.executable.resolve()),'-i',cli_path(src),'-n',str(a.screen_size),'-o',cli_path(a.outdir)]
    subprocess.run(cmd,check=True)
    outputs=sorted(p.name for p in a.outdir.glob('*.obj') if p!=src)
    if not outputs:raise RuntimeError('RoLoPM did not produce an output OBJ')
    (a.outdir/'parameters.json').write_text(json.dumps(dict(method='RoLoPM',screen_size=a.screen_size,outputs=outputs),indent=2)+'\n')


if __name__=='__main__':main()
