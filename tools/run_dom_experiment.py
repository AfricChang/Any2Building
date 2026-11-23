#!/usr/bin/env python3
"""Compare DOM-only (mode=image) vs DOM+DSM LoD1 prior (mode=image_prior) through Stages 1-3."""
import argparse
import concurrent.futures
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import tifffile
from matplotlib.path import Path as MplPath
from PIL import Image


def build_dsm_lod1_prior(anhui_dir: Path, cid: str, out_dir: Path, dom_meta: dict, dsm: np.ndarray):
    """Build 64^3 LoD1 prior purely from footprint.json + ResultDSM.tif (no oblique mesh)."""
    fp = json.loads((anhui_dir / f"obj/original/{cid}.footprint.json").read_text(encoding="utf-8"))
    gj = json.loads((anhui_dir / f"obj/original/{cid}.geojson").read_text(encoding="utf-8"))

    # 1. Extract outer ring in ENU meters (from footprint.json) and in WGS84 lon/lat (from geojson)
    geom_enu = fp["geometry"]
    rings_enu = geom_enu["coordinates"] if geom_enu["type"] == "Polygon" else [r for p in geom_enu["coordinates"] for r in p]
    outer_enu = np.asarray(rings_enu[0], dtype=np.float64)[:, :2]
    holes_enu = [np.asarray(r, dtype=np.float64)[:, :2] for r in rings_enu[1:]]

    def first_ring(obj):
        if isinstance(obj, dict):
            if "coordinates" in obj:
                return first_ring(obj["coordinates"])
            for k in ("geometry", "features"):
                if k in obj:
                    res = first_ring(obj[k])
                    if res is not None:
                        return res
        elif isinstance(obj, list) and obj:
            if isinstance(obj[0], list) and obj[0] and isinstance(obj[0][0], (int, float)):
                return np.asarray(obj, dtype=np.float64)[:, :2]
            return first_ring(obj[0])
        return None

    outer_ll = first_ring(gj)
    lon0, lat0, sx, sy = dom_meta["lon0"], dom_meta["lat0"], dom_meta["sx"], dom_meta["sy"]
    H, W = dsm.shape

    # 2. Sample DSM inside footprint and in a 6m surrounding ring for ground elevation
    cols = (outer_ll[:, 0] - lon0) / sx
    rows = (lat0 - outer_ll[:, 1]) / sy
    c0, c1 = max(0, int(np.floor(cols.min())) - 60), min(W, int(np.ceil(cols.max())) + 60)
    r0, r1 = max(0, int(np.floor(rows.min())) - 60), min(H, int(np.ceil(rows.max())) + 60)
    sub_dsm = dsm[r0:r1, c0:c1]
    yy, xx = np.mgrid[r0:r1, c0:c1]
    pts_px = np.column_stack([xx.ravel(), yy.ravel()])
    poly_px = MplPath(np.column_stack([cols, rows]))
    inside = poly_px.contains_points(pts_px).reshape(sub_dsm.shape)
    valid = np.isfinite(sub_dsm) & (sub_dsm > -100) & (sub_dsm < 1000)

    in_vals = sub_dsm[inside & valid]
    out_vals = sub_dsm[(~inside) & valid]
    z_ground = float(np.percentile(out_vals, 15.0)) if len(out_vals) > 50 else float(np.percentile(in_vals, 5.0))
    z_roof = float(np.percentile(in_vals, 85.0))
    height_m = max(z_roof - z_ground, 3.0)

    # 3. Rasterize filled LoD1 solid into 64^3 XYZ grid in uniform [-0.45, +0.45] frame
    xy_min = outer_enu.min(axis=0)
    xy_max = outer_enu.max(axis=0)
    xy_span = xy_max - xy_min
    center = np.array([(xy_min[0] + xy_max[0]) / 2.0, (xy_min[1] + xy_max[1]) / 2.0, z_ground + height_m / 2.0])
    scale = float(max(xy_span[0], xy_span[1], height_m) / 0.9)

    # Cell centers in [-0.5, 0.5]
    lin = (np.arange(64, dtype=np.float64) + 0.5) / 64.0 - 0.5
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    world_xy = np.column_stack([gx.ravel() * scale + center[0], gy.ravel() * scale + center[1]])
    mask_2d = MplPath(outer_enu).contains_points(world_xy)
    for h_ring in holes_enu:
        mask_2d &= ~MplPath(h_ring).contains_points(world_xy)
    mask_2d = mask_2d.reshape(64, 64)

    world_z = lin * scale + center[2]
    z_mask = (world_z >= z_ground) & (world_z <= z_roof)
    if not z_mask.any():
        z_mask[32] = True
    occ = np.zeros((64, 64, 64), dtype=bool)
    occ[:, :, z_mask] = mask_2d[:, :, None]

    out_dir.mkdir(parents=True, exist_ok=True)
    prior_path = out_dir / "prior_lod1.npz"
    np.savez_compressed(prior_path, occupancy=occ)
    meta = {
        "id": cid,
        "z_ground_dsm": z_ground,
        "z_roof_dsm": z_roof,
        "height_m_dsm": height_m,
        "span_m_dsm": [float(xy_span[0]), float(xy_span[1]), float(height_m)],
        "center_enu": center.tolist(),
        "scale": scale,
        "prior_voxels": int(occ.sum()),
    }
    (out_dir / "dsm_prior_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[PRIOR] {cid}: DSM z_ground={z_ground:.1f}m, z_roof={z_roof:.1f}m, H={height_m:.1f}m, "
          f"span=({xy_span[0]:.1f},{xy_span[1]:.1f},{height_m:.1f})m, prior_voxels={occ.sum()}", flush=True)
    return prior_path, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing ResultDOM.tiff, ResultDSM.tif, and obj/original/")
    p.add_argument("--cases", nargs="+", default=["smid_001", "smid_010", "smid_040"])
    p.add_argument("--outdir", type=Path, default=Path("outputs/anhui_dom_exp"))
    p.add_argument("--rolopm", type=Path, required=True, help="Path to SurfaceRemeshingCli_bin.exe")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--screen-size", type=int, default=100)
    p.add_argument("--offload", choices=["auto", "none", "model", "block"], default="block")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    default_attn = "flash_attn" if importlib.util.find_spec("flash_attn") else "sdpa"
    os.environ.setdefault("ATTN_BACKEND", default_attn)
    os.environ.setdefault("SPARSE_ATTN_BACKEND", default_attn)
    if "SPARSE_CONV_BACKEND" not in os.environ and importlib.util.find_spec("flex_gemm"):
        os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"

    anhui_dir = args.data_dir
    out_root = args.outdir
    rolopm_exe = args.rolopm
    cases = args.cases
    seed = args.seed
    steps = args.steps
    screen_size = args.screen_size

    with tifffile.TiffFile(anhui_dir / "ResultDOM.tiff") as tf:
        p0 = tf.pages[0]
        sx, sy, _ = p0.tags["ModelPixelScaleTag"].value
        tie = p0.tags["ModelTiepointTag"].value
        dom_meta = dict(lon0=tie[3], lat0=tie[4], sx=sx, sy=sy)
    with tifffile.TiffFile(anhui_dir / "ResultDSM.tif") as tf:
        dsm = tf.pages[0].asarray()

    t_all = time.time()
    jobs = []  # (job_id, cid, mode, img_path, prior_path, job_dir)
    print("=== PREPARE DSM LoD1 PRIORS ===", flush=True)
    for cid in cases:
        cdir = out_root / cid
        prior_path, _ = build_dsm_lod1_prior(anhui_dir, cid, cdir, dom_meta, dsm)
        img_path = cdir / "dom_crop.png"
        jobs.append((f"{cid}_image", cid, "image", img_path, None, cdir / "mode_image"))
        jobs.append((f"{cid}_image_prior", cid, "image_prior", img_path, prior_path, cdir / "mode_image_prior"))
    del dsm
    gc.collect()

    # --- STAGE 1 (image_enabled=True, ONE model load) ---
    import torch
    from any2building.pipeline import Any2BuildingPipeline
    from any2building.io import occupancy_mesh, sha256

    todo_s1 = [j for j in jobs if not (j[5] / "occupancy.npz").is_file()]
    if todo_s1:
        print(f"\n=== STAGE 1: DOM Inference ({len(todo_s1)} jobs) ===", flush=True)
        pipeline = Any2BuildingPipeline(root, variant="ema", image_enabled=True, dtype="auto", offload=args.offload)
        for job_id, cid, mode, img_path, prior_path, jdir in todo_s1:
            t0 = time.time()
            jdir.mkdir(parents=True, exist_ok=True)
            print(f"[STAGE 1] Running {job_id} (mode={mode})...", flush=True)
            res = pipeline(mode=mode, image=img_path, prior=prior_path, seed=seed, steps=steps, guidance=3.0, verbose=False)
            np.savez_compressed(jdir / "occupancy.npz", occupancy=res["occupancy"])
            np.savez_compressed(jdir / "structure_latent.npz", z=res["latent"])
            np.savez_compressed(jdir / "query_logits.npz", logits=res["query_logits"])
            occupancy_mesh(res["occupancy"], jdir / "structure_preview.ply")
            rec = dict(id=job_id, building=cid, mode=mode, occupied=int(res["occupancy"].sum()),
                       image_sha256=sha256(img_path))
            (jdir / "result.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            print(f"[STAGE 1] {job_id}: occupied={rec['occupied']} ({time.time()-t0:.1f}s)", flush=True)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    # --- STAGE 2 (Surface generation, ONE flow + ONE decoder load) ---
    todo_s2 = [j for j in jobs if not (j[5] / "surface.ply").is_file()]
    if todo_s2:
        from tools.run_oblique_batch import run_stage2_batch
        run_stage2_batch(root, [j[5] for j in todo_s2], seed=seed, steps=steps, guidance=1.0, offload=args.offload)

    # --- STAGE 3 (RoLoPM n=100, 2 workers) ---
    import trimesh

    def run_s3(job):
        job_id, cid, mode, img_path, prior_path, jdir = job
        lod2_dir = jdir / "lod2"
        final_obj = lod2_dir / "input_ours_final.obj"
        if not final_obj.is_file():
            t0 = time.time()
            cmd = [
                sys.executable, str(root / "remesh.py"),
                "--mesh", str(jdir / "surface.ply"),
                "--outdir", str(lod2_dir),
                "--executable", str(rolopm_exe),
                "--screen-size", str(screen_size),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            dt = time.time() - t0
        else:
            dt = 0.0
        m = trimesh.load(final_obj, force="mesh", process=False)
        # Export ENU meters for image_prior using DSM frame
        if mode == "image_prior":
            meta = json.loads((out_root / cid / "dsm_prior_meta.json").read_text())
            c = np.asarray(meta["center_enu"], dtype=np.float64)
            sc = float(meta["scale"])
            v_enu = np.asarray(m.vertices, dtype=np.float64) * sc + c
            trimesh.Trimesh(v_enu, np.asarray(m.faces), process=False).export(lod2_dir / f"{cid}_image_prior_enu.obj")
        print(f"[STAGE 3] RoLoPM {job_id}: {len(m.vertices)}v / {len(m.faces)}f ({dt:.1f}s)", flush=True)
        return job_id, len(m.vertices), len(m.faces)

    print(f"\n=== STAGE 3: RoLoPM n={screen_size} ({len(jobs)} jobs) ===", flush=True)
    summary = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_s3, j) for j in jobs]
        for fut in futs:
            job_id, nv, nf = fut.result()
            summary.append(dict(job_id=job_id, lod2_verts=nv, lod2_faces=nf))

    (out_root / "dom_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[DOM EXPERIMENT COMPLETE] in {(time.time()-t_all)/60:.1f} min", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
