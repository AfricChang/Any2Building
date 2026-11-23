#!/usr/bin/env python3
"""Export input_points.ply for MeshLab and build a standalone interactive 3D HTML viewer."""
import argparse
import base64
import json
from pathlib import Path
import numpy as np
import trimesh


def export_colored_point_ply(points: np.ndarray, out_path: Path):
    z = points[:, 2]
    t = np.clip((z - z.min()) / (z.ptp() + 1e-8), 0.0, 1.0)
    # Vivid cyan -> yellow -> orange height ramp
    r = np.clip(255 * np.clip(1.8 * t - 0.1, 0, 1), 0, 255).astype(np.uint8)
    g = np.clip(255 * np.clip(1.2 - np.abs(t - 0.5) * 1.4, 0, 1), 0, 255).astype(np.uint8)
    b = np.clip(255 * np.clip(1.0 - 1.3 * t, 0, 1), 0, 255).astype(np.uint8)
    a = np.full_like(r, 255)
    colors = np.stack([r, g, b, a], axis=1)
    pc = trimesh.points.PointCloud(points.astype(np.float32), colors=colors)
    pc.export(out_path)


def _vertex_cluster_decimate(v: np.ndarray, f: np.ndarray, grid_res: int = 128):
    """Fast vectorized vertex clustering to keep browser previews watertight & solid."""
    vmin = v.min(axis=0)
    span = np.maximum(v.max(axis=0) - vmin, 1e-6)
    q = np.floor((v - vmin) / span * grid_res).astype(np.int64)
    keys = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    nv = len(counts)
    new_v = np.zeros((nv, 3), dtype=np.float64)
    np.add.at(new_v, inv, v)
    new_v = (new_v / counts[:, None]).astype(np.float32)
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[keep]
    # Deduplicate identical triangles
    s = np.sort(nf, axis=1)
    tkeys = (s[:, 0].astype(np.int64) << 42) | (s[:, 1].astype(np.int64) << 21) | s[:, 2].astype(np.int64)
    _, uidx = np.unique(tkeys, return_index=True)
    return new_v, nf[uidx].astype(np.uint32)


def encode_mesh_payload(mesh_path: Path, max_faces: int = 80000, include_edges: bool = True):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.uint32)
    orig_verts = int(len(v))
    orig_faces = int(len(f))
    if len(f) > max_faces:
        for res in (160, 128, 96, 72):
            dv, df = _vertex_cluster_decimate(v, f, grid_res=res)
            if len(df) <= max_faces or res == 72:
                v, f = dv, df
                break
    # Expand non-indexed triangles for crisp flat-shaded architectural facets
    tris = v[f]  # [F, 3, 3]
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    nlen = np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
    n = (n / nlen).astype(np.float32)
    normals = np.repeat(n[:, None, :], 3, axis=1).reshape(-1, 3).astype(np.float32)
    positions = tris.reshape(-1, 3).astype(np.float32)
    payload = {
        "verts": orig_verts,
        "faces": orig_faces,
        "shown_faces": int(len(f)),
        "pos_b64": base64.b64encode(positions.tobytes()).decode("ascii"),
        "nrm_b64": base64.b64encode(normals.tobytes()).decode("ascii"),
    }
    if include_edges and len(f) <= 20000:
        edges = np.stack([tris[:, 0], tris[:, 1], tris[:, 1], tris[:, 2], tris[:, 2], tris[:, 0]], axis=1).reshape(-1, 3).astype(np.float32)
        payload["edge_b64"] = base64.b64encode(edges.tobytes()).decode("ascii")
    return payload


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Any2Building 本地 3D 交互对比器</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #0f1218; color: #e6edf3; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 16px; padding: 10px 16px; background: #161b24; border-bottom: 1px solid #262f3f; font-size: 13px; }
  header .title { font-weight: 700; font-size: 15px; color: #79c0ff; margin-right: 4px; }
  .group { display: flex; align-items: center; gap: 6px; background: #1d2431; padding: 4px 10px; border-radius: 8px; border: 1px solid #2b3547; }
  button { background: #263142; color: #e6edf3; border: 1px solid #394860; border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 12px; }
  button:hover { background: #334259; }
  button.active { background: #2f81f7; border-color: #58a6ff; color: #fff; font-weight: 600; }
  label { display: inline-flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; }
  input[type=range] { width: 88px; accent-color: #58a6ff; vertical-align: middle; }
  #viewport { position: relative; flex: 1; display: flex; overflow: hidden; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .split-labels { position: absolute; top: 10px; left: 0; right: 0; pointer-events: none; display: none; justify-content: space-around; font-size: 13px; font-weight: 600; }
  .split-labels span { background: rgba(15,18,24,0.75); padding: 4px 12px; border-radius: 999px; border: 1px solid #303a4b; }
  .divider { position: absolute; top: 0; bottom: 0; left: 50%; width: 1px; background: #303a4b; pointer-events: none; display: none; }
  footer { display: flex; justify-content: space-between; align-items: center; padding: 6px 16px; background: #161b24; border-top: 1px solid #262f3f; font-size: 12px; color: #9aa7b8; }
</style>
</head>
<body>
<header>
  <span class="title">Any2Building 3D 对比</span>
  <div class="group" id="case-buttons"></div>
  <div class="group">
    <button id="mode-overlay" class="active">重叠相交 (Overlay)</button>
    <button id="mode-split">并排联动 (Split)</button>
  </div>
  <div class="group" id="mesh-buttons"></div>
  <div class="group">
    <label><input type="checkbox" id="show-pts" checked> 输入点云</label>
    <label><input type="checkbox" id="show-mesh" checked> LoD2 成果</label>
    <label><input type="checkbox" id="show-wire" checked> 线框</label>
  </div>
  <div class="group">
    <span>网格透明</span><input type="range" id="alpha" min="0.1" max="1" step="0.05" value="0.72">
    <span>点大小</span><input type="range" id="ptsize" min="1" max="6" step="0.5" value="2.5">
    <span>Z剖切</span><input type="range" id="clipz" min="-0.55" max="0.55" step="0.01" value="0.55">
  </div>
  <button id="reset-cam">重置视角</button>
</header>

<div id="viewport">
  <canvas id="gl"></canvas>
  <div class="divider" id="divider"></div>
  <div class="split-labels" id="split-labels">
    <span>原始输入点云 (Input Points)</span>
    <span id="right-label">重建三维网格</span>
  </div>
</div>

<footer>
  <span id="stats">Loading...</span>
  <span>左键拖拽旋转 · 右键平移 · 滚轮缩放 · 拖动 Z剖切 滑块可切开建筑看内部点云相交贴合</span>
</footer>

<script>
const DATA = __DATA_JSON__;

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl', { antialias: true, alpha: false });
if (!gl) { alert('WebGL 不可用'); }

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function link(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

const meshProg = link(`
  attribute vec3 aPos;
  attribute vec3 aNrm;
  uniform mat4 uMVP;
  varying vec3 vPos;
  varying vec3 vNrm;
  void main() {
    vPos = aPos;
    vNrm = aNrm;
    gl_Position = uMVP * vec4(aPos, 1.0);
  }
`, `
  precision mediump float;
  varying vec3 vPos;
  varying vec3 vNrm;
  uniform float uAlpha;
  uniform float uClipZ;
  uniform vec3 uColor;
  uniform int uWire;
  void main() {
    if (vPos.z > uClipZ) discard;
    if (uWire == 1) {
      gl_FragColor = vec4(0.95, 0.98, 1.0, 0.35);
      return;
    }
    vec3 n = normalize(vNrm);
    if (!gl_FrontFacing) n = -n;
    vec3 l1 = normalize(vec3(0.6, 0.5, 0.9));
    vec3 l2 = normalize(vec3(-0.7, -0.4, 0.4));
    float d = 0.28 + 0.55 * max(dot(n, l1), 0.0) + 0.25 * max(dot(n, l2), 0.0);
    vec3 col = uColor * d;
    gl_FragColor = vec4(col, uAlpha);
  }
`);

const ptProg = link(`
  attribute vec3 aPos;
  uniform mat4 uMVP;
  uniform float uPtSize;
  varying vec3 vPos;
  void main() {
    vPos = aPos;
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = uPtSize;
  }
`, `
  precision mediump float;
  varying vec3 vPos;
  uniform float uClipZ;
  uniform float uZMin;
  uniform float uZMax;
  void main() {
    if (vPos.z > uClipZ) discard;
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    float t = clamp((vPos.z - uZMin) / max(uZMax - uZMin, 1e-4), 0.0, 1.0);
    vec3 col = mix(vec3(0.08, 0.78, 0.98), vec3(1.0, 0.88, 0.18), t);
    col = mix(col, vec3(1.0, 0.36, 0.12), smoothstep(0.65, 1.0, t));
    gl_FragColor = vec4(col, 1.0);
  }
`);

const lineProg = link(`
  attribute vec3 aPos;
  attribute vec3 aCol;
  uniform mat4 uMVP;
  varying vec3 vCol;
  void main() {
    vCol = aCol;
    gl_Position = uMVP * vec4(aPos, 1.0);
  }
`, `
  precision mediump float;
  varying vec3 vCol;
  void main() { gl_FragColor = vec4(vCol, 0.6); }
`);

// Grid + Axes geometry
const gridVerts = [], gridCols = [];
for (let i = -5; i <= 5; i++) {
  const v = i * 0.1;
  gridVerts.push(v, -0.5, -0.5, v, 0.5, -0.5, -0.5, v, -0.5, 0.5, v, -0.5);
  for (let k = 0; k < 4; k++) gridCols.push(0.22, 0.27, 0.36);
}
// RGB XYZ axes at (-0.5, -0.5, -0.5)
const ax = [
  [-0.5,-0.5,-0.5, -0.25,-0.5,-0.5, 0.9,0.25,0.25],
  [-0.5,-0.5,-0.5, -0.5,-0.25,-0.5, 0.25,0.85,0.35],
  [-0.5,-0.5,-0.5, -0.5,-0.5,-0.25, 0.3,0.55,1.0],
];
for (const a of ax) {
  gridVerts.push(a[0],a[1],a[2], a[3],a[4],a[5]);
  gridCols.push(a[6],a[7],a[8], a[6],a[7],a[8]);
}
const gridPosBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, gridPosBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gridVerts), gl.STATIC_DRAW);
const gridColBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, gridColBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gridCols), gl.STATIC_DRAW);
const gridCount = gridVerts.length / 3;

// Decode buffers for all cases
const gpuCases = {};
for (const c of DATA.cases) {
  const pts = b64ToF32(c.points_b64);
  const ptBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, ptBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pts, gl.STATIC_DRAW);
  let zMin = 1e9, zMax = -1e9;
  for (let i = 2; i < pts.length; i += 3) {
    if (pts[i] < zMin) zMin = pts[i];
    if (pts[i] > zMax) zMax = pts[i];
  }
  const meshes = {};
  for (const [mKey, m] of Object.entries(c.meshes)) {
    const pos = b64ToF32(m.pos_b64);
    const nrm = b64ToF32(m.nrm_b64);
    const posBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    const nrmBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, nrmBuf);
    gl.bufferData(gl.ARRAY_BUFFER, nrm, gl.STATIC_DRAW);
    let edgeBuf = null, edgeCount = 0;
    if (m.edge_b64) {
      const edges = b64ToF32(m.edge_b64);
      edgeBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, edgeBuf);
      gl.bufferData(gl.ARRAY_BUFFER, edges, gl.STATIC_DRAW);
      edgeCount = edges.length / 3;
    }
    meshes[mKey] = { posBuf, nrmBuf, edgeBuf, count: pos.length / 3, edgeCount, meta: m };
  }
  gpuCases[c.id] = { meta: c, ptBuf, ptCount: pts.length / 3, zMin, zMax, meshes };
}

let activeCase = DATA.cases[0].id;
let activeMesh = Object.keys(DATA.cases[0].meshes)[0];
let splitMode = false;
let cam = { yaw: 0.78, pitch: 0.52, dist: 1.65, tx: 0, ty: 0, tz: 0 };

function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];
  return o;
}
function perspective(fov, aspect, near, far) {
  const f = 1 / Math.tan(fov / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function lookAt(eye, tgt, up) {
  let zx = eye[0]-tgt[0], zy = eye[1]-tgt[1], zz = eye[2]-tgt[2];
  let zl = Math.hypot(zx, zy, zz) || 1; zx/=zl; zy/=zl; zz/=zl;
  let xx = up[1]*zz - up[2]*zy, xy = up[2]*zx - up[0]*zz, xz = up[0]*zy - up[1]*zx;
  let xl = Math.hypot(xx, xy, xz) || 1; xx/=xl; xy/=xl; xz/=xl;
  let yx = zy*xz - zz*xy, yy = zz*xx - zx*xz, yz = zx*xy - zy*xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx*eye[0]+xy*eye[1]+xz*eye[2]),
    -(yx*eye[0]+yy*eye[1]+yz*eye[2]),
    -(zx*eye[0]+zy*eye[1]+zz*eye[2]),
    1
  ]);
}

function getMVP(aspect) {
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const eye = [cam.tx + cam.dist*cp*cy, cam.ty + cam.dist*cp*sy, cam.tz + cam.dist*sp];
  return mat4Mul(perspective(Math.PI/4, aspect, 0.02, 20.0), lookAt(eye, [cam.tx, cam.ty, cam.tz], [0, 0, 1]));
}

function drawScene(x, y, w, h, drawPts, drawMesh) {
  gl.viewport(x, y, w, h);
  gl.scissor(x, y, w, h);
  gl.clearColor(0.07, 0.09, 0.13, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const mvp = getMVP(w / Math.max(h, 1));
  const c = gpuCases[activeCase];
  const clipZ = parseFloat(document.getElementById('clipz').value);
  const alpha = parseFloat(document.getElementById('alpha').value);
  const ptSize = parseFloat(document.getElementById('ptsize').value) * (window.devicePixelRatio || 1);
  const wire = document.getElementById('show-wire').checked;

  // Grid
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, 'uMVP'), false, mvp);
  let loc = gl.getAttribLocation(lineProg, 'aPos');
  gl.bindBuffer(gl.ARRAY_BUFFER, gridPosBuf); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
  let cloc = gl.getAttribLocation(lineProg, 'aCol');
  gl.bindBuffer(gl.ARRAY_BUFFER, gridColBuf); gl.enableVertexAttribArray(cloc); gl.vertexAttribPointer(cloc, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.LINES, 0, gridCount);

  // Points
  if (drawPts) {
    gl.useProgram(ptProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(ptProg, 'uMVP'), false, mvp);
    gl.uniform1f(gl.getUniformLocation(ptProg, 'uPtSize'), ptSize);
    gl.uniform1f(gl.getUniformLocation(ptProg, 'uClipZ'), clipZ);
    gl.uniform1f(gl.getUniformLocation(ptProg, 'uZMin'), c.zMin);
    gl.uniform1f(gl.getUniformLocation(ptProg, 'uZMax'), c.zMax);
    const ploc = gl.getAttribLocation(ptProg, 'aPos');
    gl.bindBuffer(gl.ARRAY_BUFFER, c.ptBuf); gl.enableVertexAttribArray(ploc); gl.vertexAttribPointer(ploc, 3, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, c.ptCount);
  }

  // Mesh
  const m = c.meshes[activeMesh];
  if (drawMesh && m) {
    gl.useProgram(meshProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, 'uMVP'), false, mvp);
    gl.uniform1f(gl.getUniformLocation(meshProg, 'uAlpha'), alpha);
    gl.uniform1f(gl.getUniformLocation(meshProg, 'uClipZ'), clipZ);
    const col = activeMesh.startsWith('lod2') ? [0.46, 0.80, 0.98] : (activeMesh === 'surface' ? [0.64, 0.84, 0.72] : [0.76, 0.82, 0.92]);
    gl.uniform3fv(gl.getUniformLocation(meshProg, 'uColor'), col);
    gl.uniform1i(gl.getUniformLocation(meshProg, 'uWire'), 0);
    const pLoc = gl.getAttribLocation(meshProg, 'aPos');
    const nLoc = gl.getAttribLocation(meshProg, 'aNrm');
    gl.bindBuffer(gl.ARRAY_BUFFER, m.posBuf); gl.enableVertexAttribArray(pLoc); gl.vertexAttribPointer(pLoc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, m.nrmBuf); gl.enableVertexAttribArray(nLoc); gl.vertexAttribPointer(nLoc, 3, gl.FLOAT, false, 0, 0);
    if (alpha < 0.99) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.drawArrays(gl.TRIANGLES, 0, m.count);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    } else {
      gl.drawArrays(gl.TRIANGLES, 0, m.count);
    }
    if (wire && m.edgeBuf) {
      gl.uniform1i(gl.getUniformLocation(meshProg, 'uWire'), 1);
      gl.bindBuffer(gl.ARRAY_BUFFER, m.edgeBuf); gl.enableVertexAttribArray(pLoc); gl.vertexAttribPointer(pLoc, 3, gl.FLOAT, false, 0, 0);
      gl.disableVertexAttribArray(nLoc);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.LINES, 0, m.edgeCount);
      gl.disable(gl.BLEND);
    }
  }
}

function render() {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.SCISSOR_TEST);
  const showPts = document.getElementById('show-pts').checked;
  const showMesh = document.getElementById('show-mesh').checked;
  if (splitMode) {
    const hw = Math.floor(w / 2);
    drawScene(0, 0, hw, h, true, false);
    drawScene(hw, 0, w - hw, h, false, true);
  } else {
    drawScene(0, 0, w, h, showPts, showMesh);
  }
}

function updateUI() {
  const c = gpuCases[activeCase];
  const meshKeys = Object.keys(c.meshes);
  if (!meshKeys.includes(activeMesh)) activeMesh = meshKeys[0];

  const cb = document.getElementById('case-buttons');
  cb.innerHTML = '';
  for (const item of DATA.cases) {
    const b = document.createElement('button');
    b.textContent = `${item.id} (${item.label})`;
    if (item.id === activeCase) b.className = 'active';
    b.onclick = () => { activeCase = item.id; updateUI(); render(); };
    cb.appendChild(b);
  }

  const mb = document.getElementById('mesh-buttons');
  mb.innerHTML = '';
  const labels = {
    surface: '简化前 (阶段2高模)',
    lod2_n150: 'LoD2 (n=150)',
    lod2_n100: 'LoD2 (n=100)',
    lod2: 'LoD2 (n=50)',
    structure: '阶段1 体素网格',
  };
  mb.style.display = meshKeys.length > 1 ? 'flex' : 'none';
  for (const k of meshKeys) {
    const b = document.createElement('button');
    b.textContent = labels[k] || k;
    if (k === activeMesh) b.className = 'active';
    b.onclick = () => { activeMesh = k; updateUI(); render(); };
    mb.appendChild(b);
  }

  const m = c.meshes[activeMesh].meta;
  document.getElementById('right-label').textContent = labels[activeMesh] || activeMesh;
  document.getElementById('stats').textContent =
    `${c.meta.id} (${c.meta.label}) · 输入点云: ${c.ptCount.toLocaleString()} 点 · ` +
    `${labels[activeMesh] || activeMesh}: ${m.verts.toLocaleString()} 顶点 / ${m.faces.toLocaleString()} 面` +
    (m.shown_faces < m.faces ? ` (预览抽稀 ${m.shown_faces.toLocaleString()} 面)` : '');
  document.getElementById('divider').style.display = splitMode ? 'block' : 'none';
  document.getElementById('split-labels').style.display = splitMode ? 'flex' : 'none';
  document.getElementById('mode-overlay').className = splitMode ? '' : 'active';
  document.getElementById('mode-split').className = splitMode ? 'active' : '';
}

document.getElementById('mode-overlay').onclick = () => { splitMode = false; updateUI(); render(); };
document.getElementById('mode-split').onclick = () => { splitMode = true; updateUI(); render(); };
document.getElementById('reset-cam').onclick = () => { cam = { yaw: 0.78, pitch: 0.52, dist: 1.65, tx: 0, ty: 0, tz: 0 }; render(); };
for (const id of ['show-pts','show-mesh','show-wire','alpha','ptsize','clipz']) {
  document.getElementById(id).oninput = render;
}

let drag = 0, lx = 0, ly = 0;
canvas.oncontextmenu = e => e.preventDefault();
canvas.onmousedown = e => { drag = e.button === 2 ? 2 : 1; lx = e.clientX; ly = e.clientY; };
window.onmouseup = () => { drag = 0; };
window.onmousemove = e => {
  if (!drag) return;
  const dx = e.clientX - lx, dy = e.clientY - ly;
  lx = e.clientX; ly = e.clientY;
  if (drag === 1) {
    cam.yaw -= dx * 0.008;
    cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.008));
  } else {
    const s = cam.dist * 0.0018;
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
    cam.tx += sy * dx * s;
    cam.ty -= cy * dx * s;
    cam.tz += dy * s;
  }
  render();
};
canvas.onwheel = e => {
  e.preventDefault();
  cam.dist = Math.max(0.4, Math.min(6.0, cam.dist * (e.deltaY > 0 ? 1.08 : 0.92)));
  render();
};
window.onresize = render;
updateUI();
render();
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=Path, default=Path("outputs/local_examples"))
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--lod2-only", action="store_true", help="Include only input points and final LoD2 mesh")
    args = p.parse_args()

    manifest = {}
    manifest_path = args.root / "examples/manifest.json"
    if manifest_path.is_file():
        manifest = {r["id"]: r for r in json.loads(manifest_path.read_text())}
    cases = []
    for case_dir in sorted(args.outdir.iterdir()):
        if not case_dir.is_dir():
            continue
        cid = case_dir.name
        local_pts_npz = case_dir / "input_points.npz"
        if local_pts_npz.is_file():
            pts = np.load(local_pts_npz)["points"].astype(np.float32)
            label = cid
            tf_path = case_dir / "transform.json"
            if tf_path.is_file():
                tf = json.loads(tf_path.read_text())
                sp = tf.get("span_m")
                if sp:
                    label = f"{sp[0]:.0f}x{sp[1]:.0f}x{sp[2]:.0f}m"
        elif cid in manifest and manifest[cid]["mode"] in ("point", "both"):
            pts = np.load(args.root / manifest[cid]["points"])["points"].astype(np.float32)
            label = manifest[cid].get("dataset_case", cid)
        else:
            continue
        if not (case_dir / "input_points.ply").is_file():
            export_colored_point_ply(pts, case_dir / "input_points.ply")

        meshes = {}
        if not args.lod2_only:
            surf_ply = case_dir / "surface.ply"
            if surf_ply.is_file():
                meshes["surface"] = encode_mesh_payload(surf_ply, include_edges=False)
        for key, sub in [("lod2_n150", "lod2_n150"), ("lod2_n100", "lod2_n100"), ("lod2", "lod2")]:
            objs = sorted((case_dir / sub).glob("*.obj")) if (case_dir / sub).is_dir() else []
            objs = [o for o in objs if o.name != "input.obj" and not o.name.endswith("_enu.obj")]
            if objs:
                meshes[key] = encode_mesh_payload(objs[-1], include_edges=True)
        if not args.lod2_only:
            struct_ply = case_dir / "structure_preview.ply"
            if struct_ply.is_file():
                meshes["structure"] = encode_mesh_payload(struct_ply, include_edges=False)
        if not meshes:
            continue

        cases.append({
            "id": cid,
            "label": label,
            "points_b64": base64.b64encode(pts.tobytes()).decode("ascii"),
            "meshes": meshes,
        })

    out_html = args.outdir / "viewer.html"
    out_html.write_text(HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps({"cases": cases})), encoding="utf-8")
    print(f"Exported {len(cases)} cases -> {out_html} ({out_html.stat().st_size / (1024**2):.2f} MiB)")


if __name__ == "__main__":
    main()
