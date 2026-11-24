#!/usr/bin/env python3
"""Batch Stage 1 -> Stage 2 -> Stage 3 (RoLoPM LoD2) reconstruction for oblique OBJ buildings."""
import argparse
import concurrent.futures
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def load_obj_geometry_fast(obj_path: Path):
    """Fast geometry-only OBJ parser (skips materials/textures/UVs)."""
    verts = []
    faces = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                idx = []
                for p in parts:
                    vi = int(p.split("/")[0])
                    idx.append(vi - 1 if vi > 0 else len(verts) + vi)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return v, f


def sample_surface_points(v: np.ndarray, f: np.ndarray, n_points: int, seed: int = 42):
    """Area-weighted uniform surface sampling (pure NumPy, deterministic)."""
    rng = np.random.default_rng(seed)
    if len(f) == 0:
        idx = rng.choice(len(v), size=n_points, replace=(len(v) < n_points))
        return v[idx]
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    valid = areas > 1e-12
    if not np.any(valid):
        idx = rng.choice(len(v), size=n_points, replace=(len(v) < n_points))
        return v[idx]
    v0, v1, v2, areas = v0[valid], v1[valid], v2[valid], areas[valid]
    probs = areas / areas.sum()
    tri_idx = rng.choice(len(areas), size=n_points, p=probs)
    r1 = np.sqrt(rng.random(n_points))[:, None]
    r2 = rng.random(n_points)[:, None]
    pts = (1.0 - r1) * v0[tri_idx] + (r1 * (1.0 - r2)) * v1[tri_idx] + (r1 * r2) * v2[tri_idx]
    return pts


def subdivide_mesh(v: np.ndarray, f: np.ndarray, max_edge: float, max_iter: int = 6):
    """Subdivide triangles until all edges are <= max_edge (in mesh units, meters)."""
    import trimesh
    vs, fs = trimesh.remesh.subdivide_to_size(v, f, max_edge=max_edge, max_iter=max_iter)
    return np.asarray(vs, dtype=np.float64), np.asarray(fs, dtype=np.int64)


def sample_lidar_clean_points(
    v: np.ndarray,
    f: np.ndarray,
    footprint_path: Path,
    n_points: int,
    seed: int = 42,
    densified: bool = False,
):
    """Simulate clean airborne LiDAR from an oblique photogrammetry mesh + surveyed footprint.

    1) Top-down 2.5D roof extraction: keep upward-facing triangles (n_z > 0.35), bin onto a fine XY grid
       and take the 95th-percentile Z per cell to eliminate recessed balconies, undersides, and noise.
    2) Clean bottom clipping at Z 2nd percentile to remove hanging skirt outliers.
    3) Sparse clean facade points along the surveyed footprint perimeter from z_ground up to local roof Z.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cr, axis=1)
    areas = 0.5 * norms
    nz = cr[:, 2] / (norms + 1e-12)

    # 1. Clean ground floor elevation (clip bottom 2% hanging fragments)
    z_ground = float(np.percentile(v[:, 2], 2.0))
    z_top = float(np.percentile(v[:, 2], 99.8))

    # 2. Dense upward-facing roof candidates -> 2.5D top-visible grid filter
    up_mask = (areas > 1e-12) & (nz > 0.35)
    if up_mask.sum() < 10:
        up_mask = (areas > 1e-12) & (nz > 0.0)
    up_f = f[up_mask]
    if densified:
        # Include subdivided upward vertices directly + denser surface sampling
        up_verts = v[np.unique(up_f)]
        surf_cand = sample_surface_points(v, up_f, n_points=300000, seed=seed)
        cand_pts = np.vstack([up_verts, surf_cand])
    else:
        cand_pts = sample_surface_points(v, up_f, n_points=120000, seed=seed)
    # Keep points above ground + small margin (avoid ground strips inside roof mask)
    min_roof_z = z_ground + 0.08 * (z_top - z_ground)
    roof_cand = cand_pts[cand_pts[:, 2] >= min_roof_z]
    if len(roof_cand) < 500:
        roof_cand = cand_pts

    xy_min = roof_cand[:, :2].min(axis=0)
    xy_span = np.maximum(roof_cand[:, :2].max(axis=0) - xy_min, 1e-3)
    cell = max(float(xy_span.max()) / 256.0, 0.12) if densified else max(float(xy_span.max()) / 128.0, 0.25)
    gx = np.floor((roof_cand[:, 0] - xy_min[0]) / cell).astype(np.int64)
    gy = np.floor((roof_cand[:, 1] - xy_min[1]) / cell).astype(np.int64)
    keys = (gx << 20) | gy
    order = np.argsort(keys)
    sk = keys[order]
    sp = roof_cand[order]
    uniq, starts, counts = np.unique(sk, return_index=True, return_counts=True)

    roof_grid_pts = np.empty((len(uniq), 3), dtype=np.float64)
    for i, (st, cnt) in enumerate(zip(starts, counts)):
        sl = sp[st : st + cnt]
        # Top-surface elevation per XY cell (95th percentile rejects stray spikes & balcony floors below)
        z_val = np.percentile(sl[:, 2], 95.0)
        roof_grid_pts[i, 0] = sl[:, 0].mean()
        roof_grid_pts[i, 1] = sl[:, 1].mean()
        roof_grid_pts[i, 2] = z_val

    # Target ~62% roof points, ~38% clean outer facade points (matches airborne LiDAR examples)
    n_roof = int(round(n_points * 0.62))
    n_wall = n_points - n_roof
    roof_idx = rng.choice(len(roof_grid_pts), size=n_roof, replace=(len(roof_grid_pts) < n_roof))
    roof_pts = roof_grid_pts[roof_idx]
    # Add tiny sub-cell XY jitter when repeating grid centers so FPS distributes evenly
    if len(roof_grid_pts) < n_roof:
        roof_pts = roof_pts.copy()
        roof_pts[:, :2] += rng.uniform(-0.35 * cell, 0.35 * cell, size=(n_roof, 2))

    # 3. Clean outer walls from footprint polygon extruded from z_ground to local roof Z
    wall_pts = None
    if footprint_path.is_file():
        fp = json.loads(footprint_path.read_text(encoding="utf-8"))
        geom = fp.get("geometry", {})
        rings = []
        if geom.get("type") == "Polygon":
            rings = geom.get("coordinates", [])
        elif geom.get("type") == "MultiPolygon":
            for poly in geom.get("coordinates", []):
                rings.extend(poly)
        segs_a, segs_b = [], []
        for ring in rings:
            arr = np.asarray(ring, dtype=np.float64)[:, :2]
            if len(arr) >= 2:
                segs_a.append(arr[:-1])
                segs_b.append(arr[1:])
        if segs_a:
            a = np.vstack(segs_a)
            b = np.vstack(segs_b)
            lens = np.linalg.norm(b - a, axis=1)
            valid = lens > 1e-6
            a, b, lens = a[valid], b[valid], lens[valid]
            # Query local roof height along each perimeter sample
            tree = cKDTree(roof_grid_pts[:, :2])
            seg_probs = lens / lens.sum()
            # Oversample then weight by local wall height so tall facades get uniform density
            m_cand = n_wall * 3
            s_idx = rng.choice(len(lens), size=m_cand, p=seg_probs)
            t = rng.random(m_cand)[:, None]
            xy = a[s_idx] + t * (b[s_idx] - a[s_idx])
            dists, nn = tree.query(xy, k=min(5, len(roof_grid_pts)))
            if nn.ndim == 1:
                local_z_top = roof_grid_pts[nn, 2]
            else:
                local_z_top = np.median(roof_grid_pts[nn, 2], axis=1)
            heights = np.maximum(local_z_top - z_ground, 0.5)
            h_probs = heights / heights.sum()
            keep = rng.choice(m_cand, size=n_wall, p=h_probs)
            xy_w = xy[keep]
            z_w = z_ground + rng.random(n_wall) * np.maximum(local_z_top[keep] - z_ground, 0.2)
            wall_pts = np.column_stack([xy_w, z_w])

    if wall_pts is None:
        # Fallback if no footprint.json: sparse wall triangles above z_ground
        wall_mask = (areas > 1e-12) & (nz >= -0.1) & (nz <= 0.35)
        w_cand = sample_surface_points(v, f[wall_mask] if wall_mask.any() else f, n_points=n_wall * 2, seed=seed + 1)
        w_cand = w_cand[w_cand[:, 2] >= z_ground]
        idx = rng.choice(len(w_cand), size=n_wall, replace=(len(w_cand) < n_wall))
        wall_pts = w_cand[idx]

    pts = np.vstack([roof_pts, wall_pts])
    pts[:, 2] = np.clip(pts[:, 2], z_ground, z_top)
    return pts


def prepare_case(
    obj_path: Path,
    case_dir: Path,
    n_points: int,
    seed: int,
    coord_sys: dict,
    preprocess: str = "raw",
    subdivide_max_edge: float = 0.0,
):
    case_dir.mkdir(parents=True, exist_ok=True)
    npz_path = case_dir / "input_points.npz"
    tf_path = case_dir / "transform.json"
    if npz_path.is_file() and tf_path.is_file():
        return json.loads(tf_path.read_text())

    t0 = time.time()
    v_orig, f_orig = load_obj_geometry_fast(obj_path)
    v, f = v_orig, f_orig
    if subdivide_max_edge > 0:
        v, f = subdivide_mesh(v_orig, f_orig, max_edge=subdivide_max_edge, max_iter=6)
    if preprocess == "lidar_clean":
        fp_path = obj_path.with_suffix(".footprint.json")
        pts_enu = sample_lidar_clean_points(
            v, f, fp_path, n_points=n_points, seed=seed, densified=(subdivide_max_edge > 0)
        )
    else:
        pts_enu = sample_surface_points(v, f, n_points=n_points, seed=seed)
    mn = pts_enu.min(axis=0)
    mx = pts_enu.max(axis=0)
    span = mx - mn
    center = (mn + mx) / 2.0
    scale = float(span.max() / 0.9)  # Match Any2Building examples [-0.45, 0.45]
    pts_norm = ((pts_enu - center) / scale).astype(np.float32)

    np.savez_compressed(npz_path, points=pts_norm)
    from tools.export_viewer import export_colored_point_ply
    export_colored_point_ply(pts_norm, case_dir / "input_points.ply")

    tf = {
        "id": obj_path.stem,
        "preprocess": preprocess,
        "subdivide_max_edge": float(subdivide_max_edge),
        "source_obj": str(obj_path),
        "source_verts": int(len(v_orig)),
        "source_faces": int(len(f_orig)),
        "subdivided_verts": int(len(v)),
        "subdivided_faces": int(len(f)),
        "sampled_points": int(n_points),
        "center_enu": center.tolist(),
        "scale": scale,
        "bounds_enu_min": mn.tolist(),
        "bounds_enu_max": mx.tolist(),
        "span_m": span.tolist(),
        "coordinate_system": coord_sys,
    }
    tf_path.write_text(json.dumps(tf, indent=2) + "\n", encoding="utf-8")
    sub_msg = f" -> subd({subdivide_max_edge:.2f}m) {len(v):,}v/{len(f):,}f" if subdivide_max_edge > 0 else ""
    print(f"[PREP:{preprocess}] {obj_path.stem}: {len(v_orig):,}v/{len(f_orig):,}f{sub_msg} -> {n_points} pts, "
          f"span=({span[0]:.1f}, {span[1]:.1f}, {span[2]:.1f})m ({time.time()-t0:.1f}s)", flush=True)
    return tf


def run_stage1_batch(root: Path, case_dirs: list[Path], seed: int, steps: int, guidance: float, offload: str = "auto"):
    todo = [d for d in case_dirs if not (d / "occupancy.npz").is_file()]
    if not todo:
        print("[STAGE 1] All cases already have occupancy.npz, skipping.", flush=True)
        return
    import torch
    from any2building.pipeline import Any2BuildingPipeline
    from any2building.io import occupancy_mesh, sha256

    print(f"\n=== STAGE 1: Structure Generation ({len(todo)} cases) ===", flush=True)
    pipeline = Any2BuildingPipeline(root, variant="ema", image_enabled=False, dtype="auto", offload=offload)
    for case_dir in todo:
        t0 = time.time()
        pts_path = case_dir / "input_points.npz"
        print(f"[STAGE 1] Running {case_dir.name}...", flush=True)
        res = pipeline(mode="point", points=pts_path, seed=seed, steps=steps, guidance=guidance, verbose=False)
        np.savez_compressed(case_dir / "occupancy.npz", occupancy=res["occupancy"])
        np.savez_compressed(case_dir / "structure_latent.npz", z=res["latent"])
        np.savez_compressed(case_dir / "query_logits.npz", logits=res["query_logits"])
        occupancy_mesh(res["occupancy"], case_dir / "structure_preview.ply")
        rec = dict(
            id=case_dir.name, mode="point", model="Any2Building", variant="ema",
            seed=seed, steps=steps, guidance=guidance, occupied=int(res["occupancy"].sum()),
            input_sha256={"points": sha256(pts_path)},
        )
        (case_dir / "result.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        print(f"[STAGE 1] {case_dir.name} done: occupied={rec['occupied']} ({time.time()-t0:.1f}s)", flush=True)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


def run_stage2_batch(root: Path, case_dirs: list[Path], seed: int, steps: int, guidance: float, offload: str = "auto"):
    todo = [d for d in case_dirs if not (d / "surface.ply").is_file()]
    if not todo:
        print("[STAGE 2] All cases already have surface.ply, skipping.", flush=True)
        return
    import torch
    import trimesh
    from any2building.io import load_occupancy, sha256
    from any2building.pipeline import _resolve_memory_strategy, _install_block_offload, _move_denoiser_shell
    from any2building.backbone import models
    from any2building.models.surface_generator import VoxelConditionalSurfaceGenerator
    from any2building.backbone.modules.sparse import SparseTensor
    from any2building.backbone.pipelines.samplers import FlowEulerCfgSampler
    from surface import _install_decoder_stage_offload

    print(f"\n=== STAGE 2: Dense Surface Generation ({len(todo)} cases, guidance={guidance}) ===", flush=True)
    device = torch.device("cuda")
    weight_dtype, offload = _resolve_memory_strategy(device, "auto", offload)
    print(f"[STAGE 2] device={device} weight_dtype={weight_dtype} offload={offload}", flush=True)

    cfg = json.loads((root / "configs/surface_generator.json").read_text())
    norm = cfg["dataset"]["args"]["normalization"]
    mean = torch.tensor(norm["mean"], device=device)[None]
    std = torch.tensor(norm["std"], device=device)[None]

    # 2A: Run surface flow for all cases with ONE weight load
    flow = VoxelConditionalSurfaceGenerator(**cfg["models"]["denoiser"]["args"]).eval().requires_grad_(False)
    flow.load_state_dict(
        torch.load(root / "weights/surface_generator/surface_flow.pt", map_location="cpu", weights_only=True, mmap=True),
        strict=True,
    )
    for block in flow.blocks:
        block.use_checkpoint = False
    if weight_dtype == "bfloat16":
        flow.convert_to(torch.bfloat16)
    if offload == "block":
        _install_block_offload(flow, device)
        _move_denoiser_shell(flow, device)
    else:
        flow.to(device)

    sampler = FlowEulerCfgSampler(sigma_min=1e-5)
    null = torch.zeros(1, 1, 1024, device=device)
    latents_cpu = {}
    with torch.no_grad():
        for case_dir in todo:
            t0 = time.time()
            occ_path = case_dir / "occupancy.npz"
            coords = np.argwhere(load_occupancy(occ_path)).astype(np.int32)
            c4 = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32), torch.from_numpy(coords)], 1).to(device)
            noise = SparseTensor(
                coords=c4,
                feats=torch.randn(len(c4), 32, device=device, generator=torch.Generator(device=device).manual_seed(seed)),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = sampler.sample(
                    flow, noise=noise, cond=null, neg_cond=null,
                    steps=steps, guidance_strength=guidance, verbose=False,
                ).samples
            feats_denorm = (latent.feats.float() * std + mean).cpu()
            latents_cpu[case_dir] = (c4.cpu(), feats_denorm, len(coords))
            del c4, noise, latent
            torch.cuda.empty_cache()
            print(f"[STAGE 2A] Flow {case_dir.name}: {len(coords)} voxels ({time.time()-t0:.1f}s)", flush=True)

    del flow, null, mean, std
    gc.collect()
    torch.cuda.empty_cache()

    # 2B: Decode all dense surfaces with ONE decoder load
    decoder = models.from_pretrained(str(root / "weights/surface_decoder/model")).eval().requires_grad_(False)
    decoder.set_resolution(1024)
    if offload != "none":
        decoder.from_latent.to(device)
        decoder.output_layer.to(device)
        _install_decoder_stage_offload(decoder, device)
    else:
        decoder.to(device)

    with torch.no_grad():
        for case_dir in todo:
            t0 = time.time()
            c4_cpu, feats_cpu, n_occ = latents_cpu[case_dir]
            latent = SparseTensor(coords=c4_cpu.to(device), feats=feats_cpu.to(device))
            mesh = decoder(latent)[0]
            v = mesh.vertices.detach().float().cpu().numpy()
            f = mesh.faces.detach().long().cpu().numpy()
            out_ply = case_dir / "surface.ply"
            trimesh.Trimesh(v, f, process=False).export(out_ply)
            rec = dict(
                model="Any2Building voxel-conditional surface generator", weights="ema",
                steps=steps, seed=seed, guidance=guidance,
                occupancy_sha256=sha256(case_dir / "occupancy.npz"),
                occupied=n_occ, vertices=int(len(v)), faces=int(len(f)),
            )
            out_ply.with_suffix(".json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            del latent, mesh
            torch.cuda.empty_cache()
            print(f"[STAGE 2B] Decoded {case_dir.name}: {len(v):,}v/{len(f):,}f ({time.time()-t0:.1f}s)", flush=True)

    del decoder, latents_cpu
    gc.collect()
    torch.cuda.empty_cache()


def run_stage3_case(root: Path, case_dir: Path, rolopm_exe: Path, screen_size: int):
    import trimesh

    lod2_dir = case_dir / "lod2"
    final_obj = lod2_dir / "input_ours_final.obj"
    enu_obj = lod2_dir / f"{case_dir.name}_lod2_enu.obj"
    if final_obj.is_file() and enu_obj.is_file():
        m = trimesh.load(final_obj, force="mesh", process=False)
        return case_dir.name, len(m.vertices), len(m.faces), 0.0

    t0 = time.time()
    if not final_obj.is_file():
        if lod2_dir.exists():
            import shutil
            shutil.rmtree(lod2_dir)
        cmd = [
            sys.executable, str(root / "remesh.py"),
            "--mesh", str(case_dir / "surface.ply"),
            "--outdir", str(lod2_dir),
            "--executable", str(rolopm_exe),
            "--screen-size", str(screen_size),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Also write world-coordinate ENU meters LoD2 OBJ aligned to the original oblique OBJ
    tf = json.loads((case_dir / "transform.json").read_text())
    center = np.asarray(tf["center_enu"], dtype=np.float64)
    scale = float(tf["scale"])
    m = trimesh.load(final_obj, force="mesh", process=False)
    v_enu = np.asarray(m.vertices, dtype=np.float64) * scale + center
    trimesh.Trimesh(v_enu, np.asarray(m.faces), process=False).export(enu_obj)
    dt = time.time() - t0
    print(f"[STAGE 3] RoLoPM {case_dir.name}: {len(m.vertices)} verts / {len(m.faces)} faces ({dt:.1f}s)", flush=True)
    return case_dir.name, len(m.vertices), len(m.faces), dt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--rolopm", type=Path, required=True)
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--screen-size", type=int, default=100)
    p.add_argument("--remesh-workers", type=int, default=2)
    p.add_argument("--preprocess", choices=["raw", "lidar_clean"], default="lidar_clean")
    p.add_argument("--subdivide-max-edge", type=float, default=0.0,
                   help="Subdivide original triangle mesh until all edges <= value (meters); 0 disables")
    p.add_argument("--offload", choices=["auto", "none", "model", "block"], default="block")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    default_attn = "flash_attn" if importlib.util.find_spec("flash_attn") else "sdpa"
    os.environ.setdefault("ATTN_BACKEND", default_attn)
    os.environ.setdefault("SPARSE_ATTN_BACKEND", default_attn)
    if "SPARSE_CONV_BACKEND" not in os.environ and importlib.util.find_spec("flex_gemm"):
        os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"

    coord_path = args.input_dir.parent / "coordinate_system.json"
    coord_sys = json.loads(coord_path.read_text(encoding="utf-8")) if coord_path.is_file() else {}

    t_all = time.time()
    case_dirs = []
    print(f"=== PREPARE: Sampling & normalizing {len(args.cases)} buildings ===", flush=True)
    for cid in args.cases:
        obj_path = args.input_dir / f"{cid}.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(obj_path)
        case_dir = args.outdir / cid
        prepare_case(
            obj_path, case_dir, args.num_points, args.seed, coord_sys,
            preprocess=args.preprocess, subdivide_max_edge=args.subdivide_max_edge,
        )
        case_dirs.append(case_dir)

    run_stage1_batch(root, case_dirs, seed=args.seed, steps=args.steps, guidance=3.0, offload=args.offload)
    run_stage2_batch(root, case_dirs, seed=args.seed, steps=args.steps, guidance=1.0, offload=args.offload)

    print(f"\n=== STAGE 3: RoLoPM LoD2 Remeshing ({len(case_dirs)} cases, {args.remesh_workers} workers) ===", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.remesh_workers) as ex:
        futs = [ex.submit(run_stage3_case, root, d, args.rolopm, args.screen_size) for d in case_dirs]
        for fut in futs:
            results.append(fut.result())

    print("\n=== EXPORTING INTERACTIVE VIEWER (Input Points + LoD2 Only) ===", flush=True)
    subprocess.run([
        sys.executable, str(root / "tools/export_viewer.py"),
        "--outdir", str(args.outdir), "--lod2-only",
    ], check=True)

    summary = []
    for cid, nv, nf, _ in results:
        tf = json.loads((args.outdir / cid / "transform.json").read_text())
        s1 = json.loads((args.outdir / cid / "result.json").read_text())
        s2 = json.loads((args.outdir / cid / "surface.json").read_text())
        summary.append({
            "id": cid,
            "span_m": [round(x, 2) for x in tf["span_m"]],
            "source_verts": tf["source_verts"],
            "source_faces": tf["source_faces"],
            "subdivided_verts": tf.get("subdivided_verts", tf["source_verts"]),
            "subdivided_faces": tf.get("subdivided_faces", tf["source_faces"]),
            "stage1_occupied": s1["occupied"],
            "stage2_verts": s2["vertices"],
            "stage2_faces": s2["faces"],
            "lod2_verts": nv,
            "lod2_faces": nf,
        })
    (args.outdir / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[BATCH COMPLETE] {len(summary)} buildings in {(time.time()-t_all)/60:.1f} min", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
